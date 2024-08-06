### adapted from: https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py

from abc import abstractmethod
from functools import partial
import math
from typing import Iterable

import numpy as np

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from ocnn.nn import octree2voxel
from einops import rearrange

# from ldm.modules.diffusionmodules.util import (
# from external.ldm.modules.diffusionmodules.util import (
from models.networks.diffusion_networks.ldm_diffusion_util import (
    conv_nd,
    default,
)
from models.networks.modules import (
    ConvDownsample,
    ConvUpsample,
    ResnetBlock,
    AttentionBlock,
    LearnedSinusoidalPosEmb,
    activation_function,
    our_Identity,
    convnormalization,
)

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
        full_depth, 
        in_split_channels,
        model_channels,
        out_split_channels,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=-1,
        use_text_condition=False,
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        **kwargs,
    ):
        super().__init__()
        # import pdb; pdb.set_trace()
        self.full_depth = full_depth
        self.in_channels = in_split_channels
        self.model_channels = model_channels
        self.out_channels = out_split_channels
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float32
        self.num_heads = num_heads
        self.predict_codebook_ids = n_embed is not None

        channels = [self.model_channels, *
                    map(lambda m: self.model_channels * m, self.channel_mult)]
        in_out = list(zip(channels[:-1], channels[1:]))


        time_embed_dim = self.model_channels * 4

        self.time_pos_emb = LearnedSinusoidalPosEmb(self.model_channels)

        self.time_emb = nn.Sequential(
            nn.Linear(self.model_channels + 1, time_embed_dim),
            activation_function(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_emb = conv_nd(dims, 2 * self.in_channels, self.model_channels, 3, padding=1)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        ds = 1

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ResnetBlock(dims, dim_in, dim_out,
                            emb_dim=time_embed_dim, dropout=dropout, use_text_condition=use_text_condition),
                nn.Sequential(
                    convnormalization(dim_out),
                    activation_function(),
                    AttentionBlock(
                        dim_out, num_heads=num_heads)) if ds in attention_resolutions else our_Identity(),
                ConvDownsample(
                    dim_out, dims=dims) if not is_last else our_Identity()
            ]))
            if not is_last:
                ds *= 2

        mid_dim = channels[-1]
        self.mid_block1 = ResnetBlock(
            dims, mid_dim, mid_dim, emb_dim=time_embed_dim, dropout=dropout, use_text_condition=use_text_condition)

        self.mid_self_attn = nn.Sequential(
            convnormalization(mid_dim),
            activation_function(),
            AttentionBlock(mid_dim, num_heads=num_heads)
        ) if ds in attention_resolutions else our_Identity()

        self.mid_block2 = ResnetBlock(
            dims, mid_dim, mid_dim, emb_dim=time_embed_dim, dropout=dropout, use_text_condition=use_text_condition)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                ResnetBlock(dims, dim_out * 2, dim_in,
                            emb_dim=time_embed_dim, dropout=dropout, use_text_condition=use_text_condition),
                nn.Sequential(
                    convnormalization(dim_in),
                    activation_function(),
                    AttentionBlock(
                        dim_in, num_heads=num_heads)) if ds in attention_resolutions else our_Identity(),
                ConvUpsample(
                    dim_in, dims=dims) if not is_last else our_Identity()
            ]))
            if not is_last:
                ds //= 2

        self.end = nn.Sequential(
            convnormalization(self.model_channels),
            activation_function()
        )

        self.out = conv_nd(dims, self.model_channels, self.out_channels, 3, padding=1)

    def forward_as_middle(self, h, doctree, timesteps, label, context):
        h_lr = octree2voxel(data=h, octree=doctree.octree, depth=self.full_depth)
        h_lr = h_lr.permute(0, 4, 1, 2, 3).contiguous()
        h_lr = self.forward(x=h_lr, timesteps=timesteps, label=label, context=context, as_middle=True)
        x, y, z, b = doctree.octree.xyzb(self.full_depth)
        h_lr = h_lr.permute(0, 2, 3, 4, 1).contiguous()
        h_lr = h_lr[b, x, y, z, :]
        return h_lr
    
    def forward(self, x=None, timesteps=None, x_self_cond=None, label = None, context=None, as_middle=False, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (label is not None) == (
            self.num_classes is not None
        ), "must specify label if and only if the model is class-conditional"

        if not as_middle:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x, x_self_cond), dim=1)
            x = self.input_emb(x)

        h = []

        emb = self.time_emb(self.time_pos_emb(timesteps))

        if self.num_classes is not None:
            assert label.shape == (x.shape[0],)
            emb = emb + self.label_emb(label)

        for resnet, self_attn, downsample in self.downs:
            x = resnet(x, emb)
            x = self_attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, emb)
        x = self.mid_self_attn(x)
        x = self.mid_block2(x, emb)

        for resnet, self_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, emb)
            x = self_attn(x)
            x = upsample(x)

        x = self.end(x)
        if as_middle:
            return x
        else:
            return self.out(x)
