### adapted from: https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py

from abc import abstractmethod
from functools import partial
import math
from typing import Iterable

import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import ocnn
from ocnn.utils import scatter_add

# from ldm.modules.diffusionmodules.util import (
# from external.ldm.modules.diffusionmodules.util import (
from models.networks.diffusion_networks.ldm_diffusion_util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    timestep_embedding,
    create_full_octree,
)

from models.networks.diffusion_networks.modules import (
    GraphConv,
    GraphResBlock,
    GraphDownsample,
    GraphUpsample,
    graphnormalization,
    Conv1x1,

)
from models.networks.diffusion_networks.graph_unet_attention import GraphTransformerBlock
from models.networks.diffusion_networks.las_attention import (
    LocalAwareCrossAttention
)
from models.networks.dualoctree_networks import dual_octree
from models.networks.dualoctree_networks.modules_v1 import doctree_align
from models.networks.diffusion_networks.openai_model_3d import ResBlock

# dummy replace
def convert_module_to_f16(x):
    pass

def convert_module_to_f32(x):
    pass


class UNet3DResNet(nn.Module):
    def __init__(self, model_channels, channel_mult, num_res_blocks, time_embed_dim, dims=3, use_scale_shift_norm=False, use_checkpoint=False):
        super().__init__()
    
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        
        ch = model_channels
        input_block_chans = [model_channels]
        for level, mult in enumerate(channel_mult):
            for i in range(num_res_blocks):
                resblk = ResBlock(
                    channels=ch,
                    emb_channels=time_embed_dim,
                    out_channels=mult * model_channels,
                    use_scale_shift_norm=use_scale_shift_norm,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    down=(i == num_res_blocks-1),
                )
                ch = mult * model_channels
                self.downs.append(resblk)
                input_block_chans.append(ch)
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks):
                ich = input_block_chans.pop()
                resblk = ResBlock(
                    channels=ch + ich,
                    emb_channels=time_embed_dim,
                    out_channels=mult * model_channels,
                    use_scale_shift_norm=use_scale_shift_norm,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    up=(i == num_res_blocks-1),
                )
                ch = mult * model_channels
                self.ups.append(resblk)
    
    def forward(self, x, emb=None, context=None, y=None, projection_matrix=None, **kwargs):
        h = x
        hs = []
        for module in self.downs:
            h = module(h, emb)
            hs.append(h)

        for module in self.ups:
            skip = hs.pop()
            h = torch.cat([h, skip], dim=1)
            h = module(h, emb)
        
        return h



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
        depth,
        full_depth,
        in_channels,
        model_channels,
        out_channels,
        split_channels,
        num_res_blocks,
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
        num_times=4,
        use_scale_shift_norm=False,
        transformer_type="cross_attn",    # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        verbose = False,
    ):
        super().__init__()
        # import pdb; pdb.set_trace()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.depth = depth
        self.full_depth = full_depth
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.transformer_type = transformer_type
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        self.verbose = verbose
        self.num_times = num_times
        n_edge_type, avg_degree = 7, 7

        time_embed_dim = model_channels * 2
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim)
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
           [
              GraphConv(in_channels, model_channels, n_edge_type, avg_degree, self.depth - 1)
           ]
        )

        self.input_conv = nn.ModuleDict()
        self.input_blocks = nn.ModuleDict()
        

        input_block_chans = [model_channels]
        ch = model_channels
        self.input_conv["latent"] = GraphConv(in_channels, ch, n_edge_type, avg_degree, self.depth - 1)
        
        for d in range(self.depth, self.full_depth-1, -1):
            mult_d = channel_mult[d]
            num_res_d = num_res_blocks[d]
            input_blocks_d = nn.ModuleList()

            self.input_conv[str(d)] = GraphConv(split_channels, ch, n_edge_type, avg_degree, d - 1)

            for _ in range(num_res_d):
                resblk = GraphResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    out_channels=mult_d * model_channels,
                    n_edge_type = n_edge_type,
                    avg_degree = avg_degree,
                    n_node_type = d - 1,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
                ch = mult_d * model_channels
                input_blocks_d.append(resblk)
                input_block_chans.append(ch)

            if d > self.full_depth:
                out_ch = ch
                input_blocks_d.append(
                    GraphDownsample(ch, out_ch, n_edge_type, avg_degree, d-2)
                )
                ch = out_ch
                input_block_chans.append(ch)
            self.input_blocks[str(d)] = input_blocks_d

        self.middle_block1 = GraphResBlock(
            ch,
            time_embed_dim,
            dropout,
            out_channels = None,
            n_edge_type = n_edge_type,
            avg_degree = avg_degree,
            n_node_type = self.full_depth - 1,
            dims=dims,
            use_checkpoint=use_checkpoint,
            use_scale_shift_norm=use_scale_shift_norm,
        )

        self.middle_block2 = GraphResBlock(
            ch,
            time_embed_dim,
            dropout,
            out_channels = None,
            n_edge_type = n_edge_type,
            avg_degree = avg_degree,
            n_node_type = self.full_depth - 1,
            dims=dims,
            use_checkpoint=use_checkpoint,
            use_scale_shift_norm=use_scale_shift_norm,
        )


        if self.transformer_type == "cross_attn":
            self.cross_attn = GraphTransformerBlock(
                ch, num_heads, dim_head, context_dim=context_dim, checkpoint=use_checkpoint
        )
        elif self.transformer_type == "local_aware":
            self.cross_attn = LocalAwareCrossAttention(
                ch, context_dim, kernel_size=1.0, num_heads=num_heads, world_dims=3, image_size=d ** 2
        )
        else:
            self.cross_attn = None
            
        self.output_blocks = nn.ModuleDict()
        self.predict = nn.ModuleDict()
        self.tanh = nn.Tanh()
        for d in range(self.full_depth, self.depth+1):
            mult_d = channel_mult[d]
            num_res_d = num_res_blocks[d]
            output_blocks_d = nn.ModuleList()
            for i in range(num_res_d + 1):
                ich = input_block_chans.pop()
                resblk = GraphResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult_d,
                        n_edge_type=n_edge_type,
                        avg_degree=avg_degree,
                        n_node_type=d-1,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                output_blocks_d.append(resblk)
                ch = model_channels * mult_d
            if d < self.depth:
                out_ch = ch
                self.predict[str(d)] = self._make_predict_module(out_ch, split_channels)
                upsample = GraphUpsample(ch, out_ch, n_edge_type, avg_degree, d-1)
                output_blocks_d.append(upsample)
            self.output_blocks[str(d)] = output_blocks_d

        self.end_norm = graphnormalization(ch)
        self.end = nn.SiLU()
        self.out = GraphConv(ch, out_channels, n_edge_type, avg_degree, self.depth - 1)

    def _make_predict_module(self, channel_in, channel_out=1, num_hidden=32):
        return torch.nn.Sequential(
            Conv1x1(channel_in, num_hidden),
            Conv1x1(num_hidden, channel_out, use_bias=True))

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, input_dict, doctree_in, doctree_out, timesteps=None, context=None, y=None, projection_matrix=None, **kwargs):
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
        # update_octree = doctree_out is None
        if doctree_out == None:
            octree_out = create_full_octree(depth=self.depth, full_depth=self.full_depth, batch_size = doctree_in.batch_size, device = doctree_in.device)
            octree_out.depth = self.full_depth
            doctree_out = dual_octree.DualOctree(octree_out)
            doctree_out.post_processing_for_docnn()

        doctree_out_copy = copy.deepcopy(doctree_out)

        logits = dict()
        hs = []
        h = 0.
        out = None

        emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(emb)

        if self.num_classes is not None:
            assert y.shape == (doctree_in.batch_size,)
            emb = emb + self.label_emb(y)
        
        if "latent" in input_dict:
            h = self.input_conv["latent"](input_dict["latent"], doctree_in, self.depth)
            hs.append(h)

        input_depth = doctree_in.depth

        for d in range(self.depth, self.full_depth-1, -1):
            if d > input_depth:
                continue
            if d in input_dict:
                h = h + self.input_conv[str(d)](input_dict[d].unsqueeze(-1), doctree_in, d)  # 送入score
            hs.append(h)

            for module in self.input_blocks[str(d)]:
                if isinstance(module, GraphResBlock):
                    h = module(h, emb, doctree_in, d)
                    hs.append(h)
                elif isinstance(module, GraphDownsample):
                    h = module(h, doctree_in, d)
                    

        h = self.middle_block1(h, emb, doctree_in, d)
        h = self.middle_block2(h, emb, doctree_in, d)

        if context is not None:
            if self.transformer_type == "cross_attn":
                h = self.cross_attn(h, doctree_in, d, context)
            elif self.transformer_type == "local_aware":
                h = self.cross_attn(h, doctree_in, d, context, projection_matrix)
        
        for d in range(self.full_depth, self.depth+1):
            if d > input_depth:
                break
            for module in self.output_blocks[str(d)]:
                if isinstance(module, GraphResBlock):
                    skip = doctree_align(hs.pop(), doctree_in.graph[d]['keyd'], doctree_out_copy.graph[d]['keyd'])
                    h = torch.cat([h, skip], dim = 1)
                    h = module(h, emb, doctree_out_copy, d)
                elif isinstance(module, GraphUpsample):
                    logit = self.predict[str(d)](h)
                    nnum = doctree_out_copy.nnum[d]
                    logits[d] = logit[-nnum:]
                    # logits[d] = self.tanh(logits[d])
                    logits[d] = logits[d].squeeze()
                    if d >= doctree_out.depth:
                        label = (logits[d] > 0).to(torch.int32)
                        octree_out = doctree_out_copy.octree
                        octree_out.octree_split(label, d)
                        octree_out.octree_grow(d + 1)
                        octree_out.depth += 1
                        doctree_out_copy = dual_octree.DualOctree(octree_out)
                        doctree_out_copy.post_processing_for_docnn()
                    h = module(h, doctree_out_copy, d)
                else:
                    raise ValueError

        if "latent" in input_dict:
            h = self.end(self.end_norm(h, doctree_out_copy, d))
            out = self.out(h, doctree_out_copy, d)

        return out, logits, doctree_out_copy