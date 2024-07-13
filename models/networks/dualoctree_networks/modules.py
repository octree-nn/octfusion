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
from ocnn.octree import key2xyz, xyz2key

from ocnn.octree import Octree
from ocnn.utils import scatter_add
from models.networks.modules import (
    nonlinearity,
    ckpt_conv_wrapper,
    DualOctreeGroupNorm,
    Conv1x1,
    Conv1x1Gn,
    Conv1x1GnGelu,
    Conv1x1GnGeluSequential,
    Downsample,
    Upsample,
    GraphConv,
    GraphResBlock,
    GraphResBlocks,
    GraphDownsample,
    GraphUpsample,
)


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
            out = self.conv1x1(out, octree, d)
        return out

    def extra_repr(self):
        return 'channels_in={}, channels_out={}'.format(
            self.channels_in, self.channels_out)


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
