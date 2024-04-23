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

bn_momentum, bn_eps = 0.01, 0.001    # the default value of Tensorflow 1.x
# bn_momentum, bn_eps = 0.1, 1e-05   # the default value of pytorch


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


class Conv1x1Bn(torch.nn.Module):

  def __init__(self, channel_in, channel_out):
    super().__init__()
    self.conv = Conv1x1(channel_in, channel_out, use_bias=False)
    self.bn = torch.nn.BatchNorm1d(channel_out, bn_eps, bn_momentum)

  def forward(self, x):
    out = self.conv(x)
    out = self.bn(out)
    return out


class Conv1x1BnRelu(torch.nn.Module):

  def __init__(self, channel_in, channel_out):
    super().__init__()
    self.conv = Conv1x1(channel_in, channel_out, use_bias=False)
    self.bn = torch.nn.BatchNorm1d(channel_out, bn_eps, bn_momentum)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, x):
    out = self.conv(x)
    out = self.bn(out)
    out = self.relu(out)
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
      self.conv1x1 = Conv1x1BnRelu(self.channels_in, self.channels_out)

  def forward(self, x, leaf_mask, numd, lnumd):
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
      out = self.conv1x1(out)
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
      self.conv1x1 = Conv1x1BnRelu(self.channels_in, self.channels_out)

  def forward(self, x, leaf_mask, numd):
    # upsample nodes at layer (depth-1)
    outd = x[-numd:]
    out1 = outd[leaf_mask.logical_not()]
    out1 = self.upsample(out1)

    # construct the final output
    out = torch.cat([x[:-numd], outd[leaf_mask], out1], dim=0)
    if self.channels_in != self.channels_out:
      out = self.conv1x1(out)
    return out

  def extra_repr(self):
    return 'channels_in={}, channels_out={}'.format(
        self.channels_in, self.channels_out)


class GraphResBlock2(torch.nn.Module):

  def __init__(self, channel_in, channel_out, bottleneck=1, n_edge_type=7,
               avg_degree=7, n_node_type=0):
    super().__init__()
    self.channel_in = channel_in
    self.channel_out = channel_out
    self.bottleneck = bottleneck
    channel_m = int(channel_out / bottleneck)

    self.conva = GraphConvBnRelu(
        channel_in, channel_m, n_edge_type, avg_degree, n_node_type)
    self.convb = GraphConvBn(
        channel_m, channel_out, n_edge_type, avg_degree, n_node_type)
    if self.channel_in != self.channel_out:
      self.conv1x1 = Conv1x1Bn(channel_in, channel_out)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, x, edge_index, edge_type, node_type):
    x1 = self.conva(x, edge_index, edge_type, node_type)
    x2 = self.convb(x1, edge_index, edge_type, node_type)

    if self.channel_in != self.channel_out:
      x = self.conv1x1(x)

    out = self.relu(x2 + x)
    return out


class GraphResBlock(torch.nn.Module):

  def __init__(self, channel_in, channel_out, bottleneck=4, n_edge_type=7,
               avg_degree=7, n_node_type=0):
    super().__init__()
    self.channel_in = channel_in
    self.channel_out = channel_out
    self.bottleneck = bottleneck
    channel_m = int(channel_out / bottleneck)

    self.conv1x1a = Conv1x1BnRelu(channel_in, channel_m)
    self.conv = GraphConvBnRelu(
        channel_m, channel_m, n_edge_type, avg_degree, n_node_type)
    self.conv1x1b = Conv1x1Bn(channel_m, channel_out)
    if self.channel_in != self.channel_out:
      self.conv1x1c = Conv1x1Bn(channel_in, channel_out)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, x, edge_index, edge_type, node_type):
    x1 = self.conv1x1a(x)
    x2 = self.conv(x1, edge_index, edge_type, node_type)
    x3 = self.conv1x1b(x2)

    if self.channel_in != self.channel_out:
      x = self.conv1x1c(x)

    out = self.relu(x3 + x)
    return out


class GraphResBlocks(torch.nn.Module):

  def __init__(self, channel_in, channel_out, resblk_num, bottleneck=4,
               n_edge_type=7, avg_degree=7, n_node_type=0,
               resblk_type='bottleneck'):
    super().__init__()
    self.resblk_num = resblk_num
    channels = [channel_in] + [channel_out] * resblk_num
    ResBlk = self._get_resblock(resblk_type)
    self.resblks = torch.nn.ModuleList([
        ResBlk(channels[i], channels[i+1], bottleneck,
               n_edge_type, avg_degree, n_node_type)
        for i in range(self.resblk_num)])

  def _get_resblock(self, resblk_type):
    if resblk_type == 'bottleneck':
      return GraphResBlock
    elif resblk_type == 'basic':
      return GraphResBlock2
    else:
      raise ValueError

  def forward(self, data, edge_index, edge_type, node_type):
    for i in range(self.resblk_num):
      data = self.resblks[i](data, edge_index, edge_type, node_type)
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
