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

from einops import rearrange

# from ldm.modules.diffusionmodules.util import (
# from external.ldm.modules.diffusionmodules.util import (
from models.networks.diffusion_networks.ldm_diffusion_util import (
    conv_nd,
    avg_pool_nd,
    zero_module,
    default,
)

from models.networks.diffusion_networks.modules import (
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
        image_size,
        in_channels,
        model_channels,
        out_channels,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_text_condition=False,
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
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        channels = [model_channels, *
                    map(lambda m: model_channels * m, channel_mult)]
        in_out = list(zip(channels[:-1], channels[1:]))

        self.verbose = verbose

        time_embed_dim = model_channels * 4

        self.time_pos_emb = LearnedSinusoidalPosEmb(model_channels)

        self.time_emb = nn.Sequential(
            nn.Linear(model_channels + 1, time_embed_dim),
            activation_function(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_emb = conv_nd(dims, 2 * in_channels, model_channels, 3, padding=1)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        ds = 1

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            res = image_size // ds
            self.downs.append(nn.ModuleList([
                ResnetBlock(dims, dim_in, dim_out,
                            emb_dim=time_embed_dim, dropout=dropout, use_text_condition=use_text_condition),
                nn.Sequential(
                    convnormalization(dim_out),
                    activation_function(),
                    AttentionBlock(
                        dim_out, num_heads=num_heads)) if ds in attention_resolutions else our_Identity(),
                Downsample(
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
                Upsample(
                    dim_in, dims=dims) if not is_last else our_Identity()
            ]))
            if not is_last:
                ds //= 2

        self.end = nn.Sequential(
            convnormalization(model_channels),
            activation_function()
        )

        self.out = conv_nd(dims, model_channels, out_channels, 3, padding=1)


    def forward(self, x, timesteps=None, x_self_cond=None, label = None, context=None, **kwargs):
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
            if self.verbose:
                print(x.shape)
            x = self_attn(x)
            if self.verbose:
                print(x.shape)
            h.append(x)
            x = downsample(x)
            if self.verbose:
                print(x.shape)

        if self.verbose:
            print('enter bottle neck')
        x = self.mid_block1(x, emb)
        if self.verbose:
            print(x.shape)

        x = self.mid_self_attn(x)
        if self.verbose:
            print(x.shape)
        x = self.mid_block2(x, emb)
        if self.verbose:
            print(x.shape)
            print('finish bottle neck')

        for resnet, self_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            if self.verbose:
                print(x.shape)
            x = resnet(x, emb)
            if self.verbose:
                print(x.shape)
            x = self_attn(x)
            if self.verbose:
                print(x.shape)
            x = upsample(x)
            if self.verbose:
                print(x.shape)

        x = self.end(x)
        if self.verbose:
            print(x.shape)

        return self.out(x)
