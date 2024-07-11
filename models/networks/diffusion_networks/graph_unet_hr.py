### adapted from: https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py

from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from ocnn.nn import octree2voxel
from ocnn.utils import scatter_add

from models.networks.dualoctree_networks import dual_octree
from models.networks.diffusion_networks.ldm_diffusion_util import create_full_octree

# from ldm.modules.diffusionmodules.util import (
# from external.ldm.modules.diffusionmodules.util import (
from models.networks.diffusion_networks.ldm_diffusion_util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    voxelnormalization,
    timestep_embedding,
)

from models.networks.modules import (
    GraphConv,
    Conv1x1,
    graphnormalization,
    TimestepBlock,
    GraphDownsample,
    GraphUpsample,
    GraphResBlockEmbed,

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
        input_depth,
        full_depth,
        in_split_channels,
        in_feature_channels,
        model_channels,
        lr_model_channels,
        out_split_channels,
        out_feature_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=[1, 2, 4],
        dims=3,
        num_classes=None,
        use_checkpoint=False,
        num_heads=-1,
        use_scale_shift_norm=False,
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        **kwargs,
    ):
        super().__init__()

        self.image_size = image_size
        self.input_depth = input_depth
        self.full_depth = full_depth
        self.in_split_channels = in_split_channels
        self.in_feature_channels = in_feature_channels
        self.model_channels = model_channels
        self.out_split_channels = out_split_channels
        self.out_feature_channels = out_feature_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float32
        self.num_heads = num_heads
        n_edge_type, avg_degree = 7, 7

        time_embed_dim = model_channels * 4

        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim)
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        d = self.input_depth

        self.input_blocks = nn.ModuleList(
           [
              GraphConv(self.in_feature_channels, model_channels, n_edge_type, avg_degree, self.input_depth - 1)
           ]
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        for level, mult in enumerate(channel_mult):
            for _ in range(self.num_res_blocks[level]):
                resblk = GraphResBlockEmbed(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        n_edge_type = n_edge_type,
                        avg_degree = avg_degree,
                        n_node_type = d - 1,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ch = mult * model_channels
                self.input_blocks.append(resblk)
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                d -= 1
                self.input_blocks.append(
                       GraphDownsample(ch, out_ch,n_edge_type, avg_degree, d-1)
                    )
                ch = out_ch
                input_block_chans.append(ch)
                self._feature_size += ch

        self.middle_block1 = GraphResBlockEmbed(
            ch,
            time_embed_dim,
            dropout,
            out_channels = lr_model_channels,
            n_edge_type = n_edge_type,
            avg_degree = avg_degree,
            n_node_type = d - 1,
            dims=dims,
            use_checkpoint=use_checkpoint,
            use_scale_shift_norm=use_scale_shift_norm,
        )

        self.middle_block2 = GraphResBlockEmbed(
            lr_model_channels * 2,
            time_embed_dim,
            dropout,
            out_channels = ch,
            n_edge_type = n_edge_type,
            avg_degree = avg_degree,
            n_node_type = d - 1,
            dims=dims,
            use_checkpoint=use_checkpoint,
            use_scale_shift_norm=use_scale_shift_norm,
        )

        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                resblk = GraphResBlockEmbed(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        n_edge_type = n_edge_type,
                        avg_degree = avg_degree,
                        n_node_type = d - 1,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                self.output_blocks.append(resblk)
                ch = model_channels * mult
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    d += 1
                    upsample = GraphUpsample(ch, out_ch, n_edge_type, avg_degree, d-1)
                    self.output_blocks.append(upsample)
                self._feature_size += ch

        self.end_norm = graphnormalization(ch)
        self.end = nn.SiLU()
        self.out = zero_module(GraphConv(ch, self.out_feature_channels, n_edge_type, avg_degree, self.input_depth - 1))

    def forward(self, x_hr, doctree, unet_lr = None, timesteps = None, label = None, context = None, **kwargs):
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
        ), "must specify y if and only if the model is class-conditional"


        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert label.shape == (doctree.batch_size,)
            emb = emb + self.label_emb(label)

        d = self.input_depth

        h = x_hr

        for module in self.input_blocks:
            if isinstance(module, GraphConv):
                h = module(h, doctree, d)
            elif isinstance(module, GraphResBlockEmbed):
                h = module(h, emb, doctree, d)
            elif isinstance(module, GraphDownsample):
                h = module(h, doctree, d)
                d -= 1

            hs.append(h)

        h = self.middle_block1(h, emb, doctree, d)

        if unet_lr is not None:
            h_lr = octree2voxel(data=h, octree=doctree.octree, depth = self.full_depth)
            h_lr = h_lr.permute(0, 4, 1, 2, 3).contiguous()
            h_lr = unet_lr(x_hr=h_lr, timesteps=timesteps, label=label, context=context)
            x, y, z, b = doctree.octree.xyzb(self.full_depth)
            h_lr = h_lr.permute(0, 2, 3, 4, 1).contiguous()
            h_lr = h_lr[b, x, y, z, :]
            # h = h + h_lr
            h = torch.cat([h, h_lr], dim=1)
    
        h = self.middle_block2(h, emb, doctree, d)

        for module in self.output_blocks:
            if isinstance(module, GraphResBlockEmbed):
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, emb, doctree, d)
            elif isinstance(module, GraphUpsample):
                h = module(h, doctree, d)
                d += 1

        h = self.end(self.end_norm(h, doctree, d))

        out = self.out(h, doctree, d)

        assert out.shape[0] == x_hr.shape[0]

        return out
