### adapted from: https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py

from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from ocnn.nn import octree2voxel
from ocnn.utils import scatter_add

from einops import rearrange
import math

import ocnn


def normalization(channels):
    num_groups = min(32, channels)
    return ocnn.nn.OctreeGroupNorm(in_channels = channels, group=num_groups, nempty=False)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class our_Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x



class Upsample(nn.Module):
    def __init__(self, channels, use_conv=True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.up_pool = ocnn.nn.OctreeUpsample(method="nearest", nempty=False)
        if use_conv:
            self.conv = ocnn.nn.OctreeConv(
                channels, channels, kernel_size=[3], nempty=False, use_bias=True)

    def forward(self, x, octree, depth):
        x = self.up_pool(x, octree, depth)

        if self.use_conv:
            x = self.conv(x, octree, depth + 1)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv=True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.op = ocnn.nn.OctreeConv(
                channels, channels, kernel_size=[3], stride=2, nempty=False, use_bias=True)
        else:
            self.op = ocnn.nn.OctreeMaxPool(nempty=False)

    def forward(self, x, octree, depth):
        return self.op(x, octree, depth)


class ResnetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, dim_out)
        )
        self.block1_norm = normalization(dim_in)
        self.block2_norm = normalization(dim_out)
        self.block1 = ocnn.nn.OctreeConv(
            dim_in, dim_out, kernel_size=[3], nempty=False, use_bias=True)
        self.block2 = zero_module(ocnn.nn.OctreeConv(
            dim_out, dim_out, kernel_size=[3], nempty=False, use_bias=True))

        self.silu = nn.SiLU()
        self.res_conv = ocnn.modules.Conv1x1(
            dim_in, dim_out, use_bias=True) if dim_in != dim_out else our_Identity()

    def forward(self, x, time_emb, octree, depth):
        h = self.silu(self.block1_norm(x, octree, depth))

        h = self.block1(h, octree, depth)

        batch_size = time_emb.shape[0]
        t = self.time_mlp(time_emb)
        for i in range(batch_size):
            h[octree.batch_id(depth) == i] += t[i]
        h = self.silu(self.block2_norm(h, octree, depth))

        h = self.block2(h, octree, depth)

        return h + self.res_conv(x)


class LearnedSinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class UNet3DModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    # def __init__(self, config_dict):
    def __init__(
        self,
        image_size,
        input_depth,
        in_split_channels,
        in_feature_channels,
        model_channels,
        out_split_channels,
        out_feature_channels,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=3,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        verbose = False,
    ):
        super().__init__()
        # import pdb; pdb.set_trace()

        self.image_size = image_size
        self.input_depth = input_depth
        self.in_split_channels = in_split_channels
        self.in_feature_channels = in_feature_channels
        self.model_channels = model_channels
        self.out_split_channels = out_split_channels
        self.out_feature_channels = out_feature_channels
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        self.verbose = verbose
        n_edge_type, avg_degree = 7, 7

        channels = [model_channels, *
                    map(lambda m: model_channels * m, channel_mult)]
        in_out = list(zip(channels[:-1], channels[1:]))

        time_embed_dim = model_channels * 4

        self.time_pos_emb = LearnedSinusoidalPosEmb(model_channels)
        self.time_emb = nn.Sequential(
            nn.Linear(model_channels + 1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        self.input_emb = ocnn.nn.OctreeConv(
            self.in_split_channels, model_channels, kernel_size=[3], nempty=False, use_bias=True)

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out,
                            emb_dim=time_embed_dim),
                Downsample(dim_out) if not is_last else our_Identity()
            ]))

        mid_dim = channels[-1]
        self.mid_block1 = ResnetBlock(
            mid_dim, mid_dim, emb_dim=time_embed_dim)

        self.mid_block2 = ResnetBlock(
            mid_dim, mid_dim, emb_dim=time_embed_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in,
                            emb_dim=time_embed_dim),
                Upsample(
                    dim_in) if not is_last else our_Identity()
            ]))

        self.end_norm = normalization(model_channels)
        self.end = nn.SiLU()
        self.out = ocnn.nn.OctreeConv(
            model_channels, self.out_split_channels, kernel_size=[3], nempty=False, use_bias=True)


    def forward(self, x_large, octree, timesteps = None, context = None, y = None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"


        if self.num_classes is not None:
            assert y.shape == (octree.batch_size,)
            emb = emb + self.label_emb(y)

        d = self.input_depth
        x = self.input_emb(x_large, octree, d)
        t = self.time_emb(self.time_pos_emb(timesteps))
        h = []

        for resnet, downsample in self.downs:
            x = resnet(x, t, octree, d)
            if self.verbose:
                print(d)
                print(x.shape)
            h.append(x)
            x = downsample(x, octree, d)
            d -= 1
            if self.verbose:
                print(x.shape)

        d += 1
        x = self.mid_block1(x, t, octree, d)
        if self.verbose:
            print(x.shape)
        x = self.mid_block2(x, t, octree, d)
        if self.verbose:
            print(x.shape)

        for resnet, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            if self.verbose:
                print(x.shape)
            x = resnet(x, t, octree, d)
            if self.verbose:
                print(x.shape)
            x = upsample(x, octree, d)
            d += 1
            if self.verbose:
                print(x.shape)

        x = self.end(self.end_norm(x, octree, d))

        if self.verbose:
            print(x.shape)
        return self.out(x, octree, d)
