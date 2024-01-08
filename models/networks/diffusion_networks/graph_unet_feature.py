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

from models.networks.diffusion_networks.modules import (
    GraphConv,
    Conv1x1,
    DualOctreeGroupNorm,
    voxel2fulloctree,
)

def normalization(channels):
    num_groups = min(32, channels)
    return DualOctreeGroupNorm(channels, num_groups)

# dummy replace
def convert_module_to_f16(x):
    pass

def convert_module_to_f32(x):
    pass

class our_Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):  # 这个类的作用其实是，改变原本nn.Sequential在前向时只允许一个输入x的限制，现在允许输入x，emd，context三个变量了。
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb = None, edge_index = None, edge_type = None, node_type = None, leaf_mask = None, numd = None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, GraphUpsample):
               x = layer(x, edge_index, edge_type, node_type, leaf_mask, numd)
            else:
                x = layer(x)
        return x

class Conv1x1GnGeluSequential(torch.nn.Module):

  def __init__(self, channel_in, channel_out):
    super().__init__()
    self.conv = Conv1x1(channel_in, channel_out, use_bias=False)
    self.gn = DualOctreeGroupNorm(channel_out)
    self.gelu = torch.nn.GELU()

  def forward(self, data):
    x, doctree, depth = data
    out = self.conv(x)
    out = self.gn(out, doctree, depth)
    out = self.gelu(out)
    return out

class ConvDownsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ConvUpsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(torch.nn.Module):

  def __init__(self, channels):
    super().__init__()
    self.channels = channels

    self.weights = torch.nn.Parameter(
        torch.Tensor(channels, channels, 8))
    torch.nn.init.xavier_uniform_(self.weights)

  def forward(self, x):
    weights = self.weights.flatten(1).t()
    out = x.view(-1, self.channels * 8) @ weights
    return out

  def extra_repr(self):
    return 'channels={}'.format(self.channels)


class GraphDownsample(torch.nn.Module):

  def __init__(self, channels_in, channels_out, n_edge_type, avg_degree, n_node_type):
    super().__init__()
    self.channels_in = channels_in
    self.channels_out = channels_out
    self.downsample = Downsample(channels_in)
    self.conv = GraphConv(channels_in, channels_out, n_edge_type, avg_degree, n_node_type)

  def forward(self, x, doctree, d):  # 这里的写法是，GraphDownsample是把深度为d的对偶图下采样到深度为d-1
    # downsample nodes at layer depth
    numd = doctree.nnum[d]
    lnumd = doctree.lnum[d-1]
    leaf_mask = doctree.node_child(d-1) < 0
    outd = x[-numd:]
    outd = self.downsample(outd)

    # get the nodes at layer (depth-1)
    out = torch.zeros(leaf_mask.shape[0], x.shape[1], device=x.device)
    out[leaf_mask] = x[-lnumd-numd:-numd]
    out[leaf_mask.logical_not()] = outd

    # construct the final output
    out = torch.cat([x[:-numd-lnumd], out], dim=0)

    # conv
    out = self.conv(out, doctree, d-1)

    return out

class Upsample(torch.nn.Module):

  def __init__(self, channels):
    super().__init__()
    self.channels = channels

    self.weights = torch.nn.Parameter(
        torch.Tensor(channels, channels, 8))
    torch.nn.init.xavier_uniform_(self.weights)

  def forward(self, x):
    out = x @ self.weights.flatten(1)
    out = out.view(-1, self.channels)
    return out

  def extra_repr(self):
    return 'channels={}'.format(self.channels)


class GraphUpsample(torch.nn.Module):

  def __init__(self, channels_in, channels_out, n_edge_type, avg_degree, n_node_type):
    super().__init__()
    self.channels_in = channels_in
    self.channels_out = channels_out
    self.upsample = Upsample(channels_in)
    self.conv = GraphConv(channels_in, channels_out, n_edge_type, avg_degree, n_node_type)

  def forward(self, x, doctree, d):   # 这里的写法是，GraphUpsample是把深度为d的对偶图上采样到深度为d+1
    numd = doctree.nnum[d]
    leaf_mask = doctree.node_child(d) < 0
    # upsample nodes at layer (depth-1)
    outd = x[-numd:]
    out1 = outd[leaf_mask.logical_not()]
    out1 = self.upsample(out1)

    # construct the final output
    out = torch.cat([x[:-numd], outd[leaf_mask], out1], dim=0)

    # conv
    out = self.conv(out, doctree, d+1)

    return out


class GraphResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels,
        n_edge_type,
        avg_degree,
        n_node_type,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        if out_channels==None : self.out_channels = channels
        else: self.out_channels = out_channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.block1_norm = normalization(self.channels)
        self.silu = nn.SiLU()
        self.conv1 = GraphConv(self.channels, self.out_channels, n_edge_type, avg_degree, n_node_type)

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        self.block2_norm = normalization(self.out_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = zero_module(GraphConv(self.out_channels, self.out_channels, n_edge_type, avg_degree, n_node_type))

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = Conv1x1(self.channels, self.out_channels)

    def forward(self, x, emb, doctree, depth):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb, doctree, depth), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb, doctree, depth):
        h = self.block1_norm(data = x, doctree = doctree, depth = depth)
        h = self.silu(h)
        h = self.conv1(h, doctree, depth)
        emb_out = self.emb_layers(emb).type(h.dtype)

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)

        else:
            batch_size = doctree.batch_size
            batch_id = doctree.batch_id(depth)
            assert batch_size == emb_out.shape[0]
            for i in range(batch_size):
                h[batch_id == i] += emb_out[i]
            h = self.block2_norm(data = h, doctree = doctree, depth = depth)
            h = self.silu(h)
            h = self.dropout(h)
            h = self.conv2(h, doctree, depth)
        return self.skip_connection(x) + h

class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            voxelnormalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            voxelnormalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

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
        large_depth,
        in_split_channels,
        in_feature_channels,
        model_channels,
        out_split_channels,
        out_feature_channels,
        num_res_blocks,
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

        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.input_depth = input_depth
        self.large_depth = large_depth
        self.in_split_channels = in_split_channels
        self.in_feature_channels = in_feature_channels
        self.model_channels = model_channels
        self.out_split_channels = out_split_channels
        self.out_feature_channels = out_feature_channels
        self.num_res_blocks = num_res_blocks
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
                resblk = GraphResBlock(
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

        self.middle_block1 = GraphResBlock(
            ch,
            time_embed_dim,
            dropout,
            out_channels = None,
            n_edge_type = n_edge_type,
            avg_degree = avg_degree,
            n_node_type = d - 1,
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
                resblk = GraphResBlock(
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

        self.end_norm = normalization(ch)
        self.end = nn.SiLU()
        self.out = zero_module(GraphConv(model_channels, self.out_feature_channels, n_edge_type, avg_degree, self.input_depth - 1))

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

    def forward(self, x_feature, doctree, timesteps = None, context = None, y = None, **kwargs):
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

        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (doctree.batch_size,)
            emb = emb + self.label_emb(y)

        d = self.input_depth

        h = x_feature

        for module in self.input_blocks:
            if isinstance(module, GraphConv):
                h = module(h, doctree, d)
            elif isinstance(module, GraphResBlock):
                h = module(h, emb, doctree, d)
            elif isinstance(module, GraphDownsample):
                h = module(h, doctree, d)
                d -= 1

            hs.append(h)

        h = self.middle_block1(h, emb, doctree, d)
        h = self.middle_block2(h, emb, doctree, d)

        for module in self.output_blocks:
            if isinstance(module, GraphResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, emb, doctree, d)
            elif isinstance(module, GraphUpsample):
                h = module(h, doctree, d)
                d += 1

        h = self.end(self.end_norm(h, doctree, d))

        out = self.out(h, doctree, d)

        assert out.shape[0] == x_feature.shape[0]

        return out
