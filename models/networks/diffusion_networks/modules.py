# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------
from abc import abstractmethod
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .utils.scatter import scatter_mean

from einops import rearrange
from ocnn.utils import scatter_add
from models.networks.diffusion_networks.ldm_diffusion_util import (
	conv_nd,
	avg_pool_nd,
	zero_module,
    checkpoint,
    linear,
)

class GroupNorm32(nn.GroupNorm):
	def forward(self, x):
		return super().forward(x.float()).type(x.dtype)

def convnormalization(channels):
	_channels = min(channels, 32)
	return GroupNorm32(_channels, channels)

def graphnormalization(channels):
    num_groups = min(32, channels)
    return DualOctreeGroupNorm(channels, num_groups)

def activation_function():
	return nn.SiLU()

class our_Identity(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x, *args, **kwargs):
		return x
	
def ckpt_conv_wrapper(conv_op, x, edge_index, edge_type):
	def conv_wrapper(x, edge_index, edge_type, dummy_tensor):
		return conv_op(x, edge_index, edge_type)

	# The dummy tensor is a workaround when the checkpoint is used for the first conv layer:
	# https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/11
	dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)

	return torch.utils.checkpoint.checkpoint(
		conv_wrapper, x, edge_index, edge_type, dummy)

class ConvUpsample(nn.Module):
    def __init__(self, channels, use_conv=True, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class ConvDownsample(nn.Module):
    def __init__(self, channels, use_conv=True, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2
        if use_conv:
            self.op = conv_nd(dims, channels, channels,
                              3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class GraphConv(torch.nn.Module):

    def __init__(self, in_channels, out_channels, n_edge_type=7, avg_degree=7,
                             n_node_type=0, use_bias = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = use_bias
        self.n_edge_type = n_edge_type
        self.avg_degree = avg_degree
        self.n_node_type = n_node_type

        node_channel = n_node_type if n_node_type > 1 else 0
        self.weights = torch.nn.Parameter(
                torch.Tensor(n_edge_type * (in_channels + node_channel), out_channels))
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

        # if n_node_type > 0:
        #     self.node_weights = torch.nn.Parameter(
        #             torch.tensor([0.5 ** i for i in range(n_node_type)]))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan_in = self.avg_degree * self.in_channels
        fan_out = self.avg_degree * self.out_channels
        std = math.sqrt(2.0 / float(fan_in + fan_out))
        a = math.sqrt(3.0) * std
        torch.nn.init.uniform_(self.weights, -a, a)
        if self.use_bias:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x, doctree, d):
        edge_index = doctree.graph[d]['edge_idx']
        edge_type = doctree.graph[d]['edge_dir']
        node_type = doctree.graph[d]['node_type']
        has_node_type = node_type is not None
        if has_node_type and self.n_node_type > 1:
            # concatenate the one_hot vector
            one_hot = F.one_hot(node_type, num_classes=self.n_node_type)
            x = torch.cat([x, one_hot], dim=1)

        # x -> col_data
        row, col = edge_index[0], edge_index[1]
        # weights = torch.pow(0.5, node_type[col]) if has_node_type else None
        weights = None    # TODO: ablation the weights
        index = row * self.n_edge_type + edge_type
        col_data = scatter_mean(x[col], index, dim=0, weights=weights,
                                                        dim_size=x.shape[0] * self.n_edge_type)

        # matrix product
        output = col_data.view(x.shape[0], -1) @ self.weights

        if self.use_bias:
            output += self.bias

        return output

    def extra_repr(self) -> str:
        return ('channel_in={}, channel_out={}, n_edge_type={}, avg_degree={}, '
			'n_node_type={}'.format(self.in_channels, self.out_channels,
			self.n_edge_type, self.avg_degree, self.n_node_type))  # noqa

class DualOctreeGroupNorm(torch.nn.Module):
	r''' A group normalization layer for the dual octree.
	'''

	def __init__(self, in_channels: int, group: int = 32, nempty: bool = False):
		super().__init__()
		self.eps = 1e-5
		self.nempty = nempty

		self.in_channels = in_channels
		self.group = group

		assert in_channels % group == 0
		self.channels_per_group = in_channels // group

		self.weights = torch.nn.Parameter(torch.Tensor(1, in_channels))
		self.bias = torch.nn.Parameter(torch.Tensor(1, in_channels))
		self.reset_parameters()

	def reset_parameters(self):
		torch.nn.init.ones_(self.weights)
		torch.nn.init.zeros_(self.bias)


	def forward(self, data, doctree, depth):
		r''''''

		batch_size = doctree.batch_size
		batch_id = doctree.batch_id(depth)

		assert batch_id.shape[0]==data.shape[0]

		ones = data.new_ones([data.shape[0], 1])
		count = scatter_add(ones, batch_id, dim=0, dim_size=batch_size)
		count = count * self.channels_per_group  # element number in each group
		inv_count = 1.0 / (count + self.eps)  # there might be 0 element sometimes

		mean = scatter_add(data, batch_id, dim=0, dim_size=batch_size) * inv_count
		mean = self._adjust_for_group(mean)
		out = data - mean.index_select(0, batch_id)

		var = scatter_add(out**2, batch_id, dim=0, dim_size=batch_size) * inv_count
		var = self._adjust_for_group(var)
		inv_std = 1.0 / (var + self.eps).sqrt()
		out = out * inv_std.index_select(0, batch_id)

		out = out * self.weights + self.bias
		return out

	def _adjust_for_group(self, tensor: torch.Tensor):
		r''' Adjust the tensor for the group.
		'''

		if self.channels_per_group > 1:
			tensor = (tensor.reshape(-1, self.group, self.channels_per_group)
							.sum(-1, keepdim=True)
							.repeat(1, 1, self.channels_per_group)
							.reshape(-1, self.in_channels))
		return tensor

class Conv1x1(torch.nn.Module):

	def __init__(self, channel_in, channel_out, use_bias=False):
		super().__init__()
		self.linear = torch.nn.Linear(channel_in, channel_out, use_bias)

	def forward(self, x):
		return self.linear(x)

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

    def forward(self, x, doctree, d):    # 这里的写法是，GraphDownsample是把深度为d的对偶图下采样到深度为d-1
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

    def forward(self, x, doctree, d):     # 这里的写法是，GraphUpsample是把深度为d的对偶图上采样到深度为d+1
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
   
class ResnetBlock(nn.Module):
    def __init__(self, world_dims: int, dim_in: int, dim_out: int, emb_dim: int, dropout: float = 0.1,
                 use_text_condition: bool = True):
        super().__init__()
        self.world_dims = world_dims
        self.time_mlp = nn.Sequential(
            activation_function(),
            nn.Linear(emb_dim, dim_out)
        )
        self.use_text_condition = use_text_condition
        if self.use_text_condition:
            self.text_mlp = nn.Sequential(
                activation_function(),
                nn.Linear(emb_dim, dim_out),
            )

        self.block1 = nn.Sequential(
            convnormalization(dim_in),
            activation_function(),
            conv_nd(world_dims, dim_in, dim_out, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            convnormalization(dim_out),
            activation_function(),
            nn.Dropout(dropout),
            zero_module(conv_nd(world_dims, dim_out, dim_out, 3, padding=1)),
        )
        self.res_conv = conv_nd(
            world_dims, dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, time_emb, text_condition=None):
        h = self.block1(x)
        if self.use_text_condition:
            h = h * self.text_mlp(text_condition)[(...,) +
                                                  (None, )*self.world_dims] + self.time_mlp(time_emb)[(...,) + (None, )*self.world_dims]
        else:
            h += self.time_mlp(time_emb)[(...,) + (None, )*self.world_dims]

        h = self.block2(h)
        return h + self.res_conv(x)

class AttentionBlock(nn.Module):

    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = convnormalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    def forward(self, qkv):
        ch = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        return torch.einsum("bts,bcs->bct", weight, v)


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

        self.block1_norm = graphnormalization(self.channels)
        self.silu = nn.SiLU()
        self.conv1 = GraphConv(self.channels, self.out_channels, n_edge_type, avg_degree, n_node_type)

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        self.block2_norm = graphnormalization(self.out_channels)
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