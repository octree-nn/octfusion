# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import math
import torch
import torch.nn
import torch.nn.init
import torch.nn.functional as F
import torch.utils.checkpoint
# import torch_geometric.nn

from .utils.scatter import scatter_mean

from ocnn.octree import Octree
from ocnn.utils import scatter_add

bn_momentum, bn_eps = 0.01, 0.001    # the default value of Tensorflow 1.x
# bn_momentum, bn_eps = 0.1, 1e-05   # the default value of pytorch

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

class DualOctreeGroupNorm(torch.nn.Module):
  r''' A group normalization layer for the dual octree.
  '''

  def __init__(self, in_channels: int, group: int = 32, nempty: bool = False):
    super().__init__()
    self.eps = 1e-5
    self.nempty = nempty

    if in_channels <= 32:
      group = in_channels // 4
    elif in_channels % group != 0:
      group = 30

    self.in_channels = in_channels
    self.group = group

    assert self.in_channels % self.group == 0
    self.channels_per_group = in_channels // group

    self.weights = torch.nn.Parameter(torch.Tensor(1, in_channels))
    self.bias = torch.nn.Parameter(torch.Tensor(1, in_channels))
    self.reset_parameters()

  def reset_parameters(self):
    torch.nn.init.ones_(self.weights)
    torch.nn.init.zeros_(self.bias)


  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    batch_size = octree.batch_size
    batch_id = octree.batch_id(depth, self.nempty)
    batch_nnum = octree.batch_nnum
    nnum = octree.nnum
    batch_nnum_nempty = octree.batch_nnum_nempty
    nnum_nempty = octree.nnum_nempty
    lnum = nnum - nnum_nempty
    batch_lnum = batch_nnum - batch_nnum_nempty
    leaf_num = lnum[octree.full_depth: depth].sum()
    leaf_batch_id = batch_id.new_zeros(leaf_num)
    now = 0
    for d in range(octree.full_depth, depth):
      d_lnum = batch_lnum[d]
      for batch_idx,num in enumerate(d_lnum):
        leaf_batch_id[now:now+num] = batch_idx
        now = now + num
    batch_id = torch.cat([leaf_batch_id, batch_id],dim = 0)

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

  def extra_repr(self) -> str:
    return ('in_channels={}, group={}, nempty={}').format(
            self.in_channels, self.group, self.nempty)  # noqa


def ckpt_conv_wrapper(conv_op, x, edge_index, edge_type):
  def conv_wrapper(x, edge_index, edge_type, dummy_tensor):
    return conv_op(x, edge_index, edge_type)

  # The dummy tensor is a workaround when the checkpoint is used for the first conv layer:
  # https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/11
  dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)

  return torch.utils.checkpoint.checkpoint(
      conv_wrapper, x, edge_index, edge_type, dummy)


# class GraphConv_v0(torch_geometric.nn.MessagePassing):
class GraphConv_v0:
  ''' This implementation explicitly constructs the self.weights[edge_type],
  thus consuming a lot of computation and memory.
  '''

  def __init__(self, in_channels, out_channels, n_edge_type=7, avg_degree=7):
    super().__init__(aggr='add')
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.n_edge_type = n_edge_type
    self.avg_degree = avg_degree

    self.weights = torch.nn.Parameter(
        torch.Tensor(n_edge_type, out_channels, in_channels))
    self.reset_parameters()

  def reset_parameters(self) -> None:
    fan_in = self.avg_degree * self.weights.shape[2]
    fan_out = self.avg_degree * self.weights.shape[1]
    std = math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    torch.nn.init.uniform_(self.weights, -a, a)

  def forward(self, x, edge_index, edge_type):
    # x has shape [N, in_channels]
    # edge_index has shape [2, E]

    return self.propagate(edge_index, x=x, edge_type=edge_type)

  def message(self, x_j, edge_type):
    weights = self.weights[edge_type]    # (N, out_channels, in_channels)
    output = weights @ x_j.unsqueeze(-1)
    return output.squeeze(-1)


class GraphConv(torch.nn.Module):

  def __init__(self, in_channels, out_channels, n_edge_type=7, avg_degree=7,
               n_node_type=0):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.n_edge_type = n_edge_type
    self.avg_degree = avg_degree
    self.n_node_type = n_node_type

    node_channel = n_node_type if n_node_type > 1 else 0
    self.weights = torch.nn.Parameter(
        torch.Tensor(n_edge_type * (in_channels + node_channel), out_channels))
    # if n_node_type > 0:
    #   self.node_weights = torch.nn.Parameter(
    #       torch.tensor([0.5 ** i for i in range(n_node_type)]))
    self.reset_parameters()

  def reset_parameters(self) -> None:
    fan_in = self.avg_degree * self.in_channels
    fan_out = self.avg_degree * self.out_channels
    std = math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    torch.nn.init.uniform_(self.weights, -a, a)

  def forward(self, x, edge_index, edge_type, node_type=None):
    has_node_type = node_type is not None
    if has_node_type and self.n_node_type > 1:
      # concatenate the one_hot vector
      one_hot = F.one_hot(node_type, num_classes=self.n_node_type)
      x = torch.cat([x, one_hot], dim=1)

    # x -> col_data
    row, col = edge_index[0], edge_index[1]
    # weights = torch.pow(0.5, node_type[col]) if has_node_type else None
    weights = None  # TODO: ablation the weights
    index = row * self.n_edge_type + edge_type
    col_data = scatter_mean(x[col], index, dim=0, weights=weights,
                            dim_size=x.shape[0] * self.n_edge_type)

    # matrix product
    output = col_data.view(x.shape[0], -1) @ self.weights
    return output

  def extra_repr(self) -> str:
    return ('channel_in={}, channel_out={}, n_edge_type={}, avg_degree={}, '
            'n_node_type={}'.format(self.in_channels, self.out_channels,
             self.n_edge_type, self.avg_degree, self.n_node_type))  # noqa


class GraphConvBn(torch.nn.Module):

  def __init__(self, in_channels, out_channels, n_edge_type=7, avg_degree=7,
               n_node_type=0):
    super().__init__()
    self.conv = GraphConv(
        in_channels, out_channels, n_edge_type, avg_degree, n_node_type)
    self.bn = torch.nn.BatchNorm1d(out_channels, bn_eps, bn_momentum)

  def forward(self, x, edge_index, edge_type, node_type=None):
    out = self.conv(x, edge_index, edge_type, node_type)
    out = self.bn(out)
    return out

class GraphConvBnRelu(torch.nn.Module):

  def __init__(self, in_channels, out_channels, n_edge_type=7, avg_degree=7,
               n_node_type=0):
    super().__init__()
    self.conv = GraphConv(
        in_channels, out_channels, n_edge_type, avg_degree, n_node_type)
    self.bn = torch.nn.BatchNorm1d(out_channels, bn_eps, bn_momentum)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, x, edge_index, edge_type, node_type=None):
    out = self.conv(x, edge_index, edge_type, node_type)
    # out = ckpt_conv_wrapper(self.conv, x, edge_index, edge_type)
    out = self.bn(out)
    out = self.relu(out)
    return out

class Conv1x1(torch.nn.Module):

  def __init__(self, channel_in, channel_out, use_bias=False):
    super().__init__()
    self.linear = torch.nn.Linear(channel_in, channel_out, use_bias)

  def forward(self, x):
    return self.linear(x)

class Conv1x1Gn(torch.nn.Module):

  def __init__(self, channel_in, channel_out):
    super().__init__()
    self.conv = Conv1x1(channel_in, channel_out, use_bias=False)
    self.gn = DualOctreeGroupNorm(channel_out)

  def forward(self, x, octree, depth):
    out = self.conv(x)
    out = self.gn(out, octree, depth)
    return out

class Conv1x1GnGelu(torch.nn.Module):

  def __init__(self, channel_in, channel_out):
    super().__init__()
    self.conv = Conv1x1(channel_in, channel_out, use_bias=False)
    self.gn = DualOctreeGroupNorm(channel_out)
    self.gelu = torch.nn.GELU()

  def forward(self, x, octree, depth):
    out = self.conv(x)
    out = self.gn(out, octree, depth)
    out = self.gelu(out)
    return out

class Conv1x1GnGeluSequential(torch.nn.Module):

  def __init__(self, channel_in, channel_out):
    super().__init__()
    self.conv = Conv1x1(channel_in, channel_out, use_bias=False)
    self.gn = DualOctreeGroupNorm(channel_out)
    self.gelu = torch.nn.GELU()

  def forward(self, data):
    x, octree, depth = data
    out = self.conv(x)
    out = self.gn(out, octree, depth)
    out = self.gelu(out)
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

  def __init__(self, channels_in, channels_out=None):
    super().__init__()
    self.channels_in = channels_in
    self.channels_out = channels_out or channels_in
    self.downsample = Downsample(channels_in)
    if self.channels_in != self.channels_out:
      self.conv1x1 = Conv1x1GnGelu(self.channels_in, self.channels_out)

  def forward(self, x, octree, d, leaf_mask, numd, lnumd):
    # downsample nodes at layer depth
    outd = x[-numd:]
    outd = self.downsample(outd)

    # get the nodes at layer (depth-1)
    out = torch.zeros(leaf_mask.shape[0], x.shape[1], device=x.device)
    out[leaf_mask] = x[-lnumd-numd:-numd]
    out[leaf_mask.logical_not()] = outd

    # construct the final output
    out = torch.cat([x[:-numd-lnumd], out], dim=0)

    if self.channels_in != self.channels_out:
      out = self.conv1x1(out,octree, d)
    return out

  def extra_repr(self):
    return 'channels_in={}, channels_out={}'.format(
        self.channels_in, self.channels_out)


class GraphMaxpool(torch.nn.Module):

  def __init__(self):
    super().__init__()

  def forward(self, x, leaf_mask, numd, lnumd):
    # downsample nodes at layer depth
    channel = x.shape[1]
    outd = x[-numd:]
    outd, _ = outd.view(-1, 8, channel).max(dim=1)

    # get the nodes at layer (depth-1)
    out = torch.zeros(leaf_mask.shape[0], channel, device=x.device)
    out[leaf_mask] = x[-lnumd-numd:-numd]
    out[leaf_mask.logical_not()] = outd

    # construct the final output
    out = torch.cat([x[:-numd-lnumd], out], dim=0)
    return out


class GraphUpsample(torch.nn.Module):

  def __init__(self, channels_in, channels_out=None):
    super().__init__()
    self.channels_in = channels_in
    self.channels_out = channels_out or channels_in
    self.upsample = Upsample(channels_in)
    if self.channels_in != self.channels_out:
      self.conv1x1 = Conv1x1GnGelu(self.channels_in, self.channels_out)

  def forward(self, x, octree, d, leaf_mask, numd):
    # upsample nodes at layer (depth-1)
    outd = x[-numd:]
    out1 = outd[leaf_mask.logical_not()]
    out1 = self.upsample(out1)

    # construct the final output
    out = torch.cat([x[:-numd], outd[leaf_mask], out1], dim=0)
    if self.channels_in != self.channels_out:
      out = self.conv1x1(out, octree, d)
    return out

  def extra_repr(self):
    return 'channels_in={}, channels_out={}'.format(
        self.channels_in, self.channels_out)

class GraphResBlock(torch.nn.Module):

  def __init__(self, channel_in, channel_out,dropout, n_edge_type=7,
               avg_degree=7, n_node_type=0):
    super().__init__()
    self.channel_in = channel_in
    self.channel_out = channel_out

    self.norm1 = DualOctreeGroupNorm(channel_in)

    self.conv1 = GraphConv(
        channel_in, channel_out, n_edge_type, avg_degree, n_node_type)

    self.norm2 = DualOctreeGroupNorm(channel_out)
    self.dropout = torch.nn.Dropout(dropout)

    self.conv2 = GraphConv(
        channel_out, channel_out, n_edge_type, avg_degree, n_node_type)

    if self.channel_in != self.channel_out:
      self.conv1x1c = Conv1x1Gn(channel_in, channel_out)

  def forward(self, x, octree, depth, edge_index, edge_type, node_type):
    h = x
    h = self.norm1(data = h, octree = octree, depth = depth)
    h = nonlinearity(h)
    h = self.conv1(h,edge_index, edge_type, node_type)

    h = self.norm2(data = h, octree = octree, depth = depth)
    h = nonlinearity(h)
    h = self.dropout(h)
    h = self.conv2(h,edge_index, edge_type, node_type)

    if self.channel_in != self.channel_out:
      x = self.conv1x1c(x, octree, depth)

    out = h + x
    return out


class GraphResBlocks(torch.nn.Module):

  def __init__(self, channel_in, channel_out, dropout,resblk_num,
               n_edge_type=7, avg_degree=7, n_node_type=0):
    super().__init__()
    self.resblk_num = resblk_num
    channels = [channel_in] + [channel_out] * resblk_num
    self.resblks = torch.nn.ModuleList([
        GraphResBlock(channels[i], channels[i+1],dropout,
               n_edge_type, avg_degree, n_node_type)
        for i in range(self.resblk_num)])

  def forward(self, data, octree, depth, edge_index, edge_type, node_type):
    for i in range(self.resblk_num):
      data = self.resblks[i](data, octree, depth, edge_index, edge_type, node_type)
    return data


def doctree_align(value, key, query):
  # out-of-bound
  out_of_bound = query > key[-1]
  query[out_of_bound] = -1

  # search
  idx = torch.searchsorted(key, query)
  found = key[idx] == query

  # assign the found value to the output
  out = torch.zeros(query.shape[0], value.shape[1], device=value.device)
  out[found] = value[idx[found]]
  return out
