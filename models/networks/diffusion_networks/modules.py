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

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def create_full_octree(full_depth, batch_size, device):
    r''' Initialize a full octree for decoding.
    '''
    octree = Octree(full_depth, full_depth, batch_size, device)
    for d in range(full_depth+1):
      octree.octree_grow_full(depth=d)
    return octree

def voxel2fulloctree(voxel: torch.Tensor, depth ,batch_size, device, nempty: bool = False):
  r''' Converts the input feature to the full-voxel-based representation.

  Args:
    voxel (torch.Tensor): batch_size, channel, num, num, num
    depth (int): The depth of current octree.
    nempty (bool): If True, :attr:`data` only contains the features of non-empty
        octree nodes.
  '''
  channel = voxel.shape[1]
  octree = create_full_octree(depth, batch_size, device)
  x, y, z, b = octree.xyzb(depth, nempty)
  key = octree.key(depth, nempty)
  data = voxel.new_zeros(key.shape[0], channel)
  data = voxel[b,:, x,y,z]

  return data


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
    #   self.node_weights = torch.nn.Parameter(
    #       torch.tensor([0.5 ** i for i in range(n_node_type)]))

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
    weights = None  # TODO: ablation the weights
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
