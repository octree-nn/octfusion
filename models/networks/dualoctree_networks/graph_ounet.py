# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import torch.nn

from ocnn.octree import Octree
from . import mpu
from . import modules
from . import dual_octree

class GraphOUNet(torch.nn.Module):

  def __init__(self, depth, channel_in, nout, full_depth=2, depth_stop = 6, depth_out=8, use_checkpoint = False,
               resblk_type='bottleneck', bottleneck=4, resblk_num = 3):
    super().__init__()
    self.depth = depth
    self.channel_in = channel_in
    self.nout = nout
    self.full_depth = full_depth
    self.depth_stop = depth_stop
    self.depth_out = depth_out
    self.use_checkpoint = use_checkpoint
    self.resblk_type = resblk_type
    self.bottleneck = bottleneck
    self.resblk_num = resblk_num
    self.neural_mpu = mpu.NeuralMPU(self.full_depth, self.depth_stop, self.depth_out)
    self._setup_channels_and_resblks()
    n_edge_type, avg_degree = 7, 7
    self.dropout = 0.0

    # encoder
    self.conv1 = modules.GraphConv(
        channel_in, self.channels[depth], n_edge_type, avg_degree, depth-1)
    self.encoder = torch.nn.ModuleList(
        [modules.GraphResBlocks(self.channels[d], self.channels[d],self.dropout,
         self.resblk_nums[d] - 1, n_edge_type, avg_degree, d-1, self.use_checkpoint)
         for d in range(depth, depth_stop-1, -1)])
    self.downsample = torch.nn.ModuleList(
        [modules.GraphDownsample(self.channels[d], self.channels[d-1])
         for d in range(depth, depth_stop, -1)])

    self.encoder_norm_out = modules.DualOctreeGroupNorm(self.channels[depth_stop])

    self.nonlinearity = torch.nn.GELU()

    # decoder
    self.upsample = torch.nn.ModuleList(
        [modules.GraphUpsample(self.channels[d-1], self.channels[d])
         for d in range(depth_stop + 1, depth + 1)])

    # self.decoder是在auto encoder里单独定义的
    self.decoder = torch.nn.ModuleList(
        [modules.GraphResBlocks(self.channels[d], self.channels[d],self.dropout,
         self.resblk_nums[d], n_edge_type, avg_degree, d-1, self.use_checkpoint)
         for d in range(full_depth+1, depth + 1)])

    # header
    self.predict = torch.nn.ModuleList(
        [self._make_predict_module(self.channels[d], 2)  # 这里的2就是当前节点是否要劈成八份的label
         for d in range(depth_stop, depth + 1)])
    self.regress = torch.nn.ModuleList(
        [self._make_predict_module(self.channels[d], 4)  # 这里的4就是王老师说的，MPU里一个node里的4个特征分别代表法向量和偏移量
         for d in range(depth_stop, depth + 1)])

  def _setup_channels_and_resblks(self):
    # self.resblk_num = [3] * 7 + [1] + [1] * 9
    # self.resblk_num = [3] * 16
    self.resblk_nums = [self.resblk_num] * 16      # resblk_num[d] 为深度d（分辨率）下resblock的数量。
    self.channels = [4, 512, 512, 256, 128, 64, 32, 32, 24, 8]  # depth i的channel为channels[i]

  def _make_predict_module(self, channel_in, channel_out=2, num_hidden=32):
    return torch.nn.Sequential(
      modules.Conv1x1GnGeluSequential(channel_in, num_hidden),
      modules.Conv1x1(num_hidden, channel_out, use_bias=True))

  def _get_input_feature(self, doctree):
    return doctree.get_input_feature()

  def octree_encoder(self, octree, doctree):
    depth, depth_stop = self.depth, self.depth_stop
    data = self._get_input_feature(doctree)

    convs = dict()
    convs[depth] = data   # channel为4
    for i, d in enumerate(range(depth, depth_stop-1, -1)):   # encoder的操作是从depth到depth_stop为止
      # perform graph conv
      convd = convs[d]  # get convd
      edge_idx = doctree.graph[d]['edge_idx']
      edge_type = doctree.graph[d]['edge_dir']
      node_type = doctree.graph[d]['node_type']
      if d == self.depth:  # the first conv
        convd = self.conv1(convd, edge_idx, edge_type, node_type)
      convd = self.encoder[i](convd, doctree, d, edge_idx, edge_type, node_type)
      convs[d] = convd  # update convd
      # print(convd.shape)

      # downsampleing
      if d > depth_stop:  # init convd
        nnum = doctree.nnum[d]
        lnum = doctree.lnum[d-1]
        leaf_mask = doctree.node_child(d-1) < 0
        convs[d-1] = self.downsample[i](convd, doctree, d-1, leaf_mask, nnum, lnum)

    convs[depth_stop] = self.encoder_norm_out(convs[depth_stop], doctree, depth_stop)
    convs[depth_stop] = self.nonlinearity(convs[depth_stop])

    return convs

  def octree_decoder(self, convs, doctree_out, doctree, update_octree=False):
    logits = dict()
    reg_voxs = dict()
    deconvs = dict()

    deconvs[self.full_depth] = convs[self.full_depth]
    for i, d in enumerate(range(self.full_depth, self.depth_out+1)):  # decoder的操作是从full_depth到depth_out为止，如果update_octree为true，则从full_depth开始逐渐增长八叉树，直至depth_out
      if d > self.full_depth:
        nnum = doctree_out.nnum[d-1]
        leaf_mask = doctree_out.node_child(d-1) < 0
        deconvd = self.upsample[i-1](deconvs[d-1], leaf_mask, nnum)
        skip = modules.doctree_align(
            convs[d], doctree.graph[d]['keyd'], doctree_out.graph[d]['keyd'])
        deconvd = deconvd + skip  # skip connections

        deconvs[d] = self.decoder[i-1](deconvd, doctree, d)

      # predict the splitting label
      logit = self.predict[i](deconvs[d])
      nnum = doctree_out.nnum[d]
      logits[d] = logit[-nnum:]

      # update the octree according to predicted labels
      if update_octree:
        label = logits[d].argmax(1).to(torch.int32)
        octree_out = doctree_out.octree
        octree_out.octree_split(label, d)
        if d < self.depth_out:   # 对初始化的满八叉树，根据预测的标签向上增长至depth_out
          octree_out.octree_grow(d + 1)
          octree_out.depth += 1
        doctree_out = dual_octree.DualOctree(octree_out)
        doctree_out.post_processing_for_docnn()

      # predict the signal
      reg_vox = self.regress[i](deconvs[d])

      # TODO: improve it
      # pad zeros to reg_vox to reuse the original code for ocnn
      node_mask = doctree_out.graph[d]['node_mask']
      shape = (node_mask.shape[0], reg_vox.shape[1])
      reg_vox_pad = torch.zeros(shape, device=reg_vox.device)
      reg_vox_pad[node_mask] = reg_vox
      reg_voxs[d] = reg_vox_pad

    return logits, reg_voxs, doctree_out.octree

  def create_full_octree(self, octree_in: Octree):
    r''' Initialize a full octree for decoding.
    '''

    device = octree_in.device
    batch_size = octree_in.batch_size
    octree = Octree(self.depth, self.full_depth, batch_size, device)
    for d in range(self.full_depth+1):
      octree.octree_grow_full(depth=d)
    return octree

  def create_child_octree(self, octree_in: Octree):
    octree_out = self.create_full_octree(octree_in)
    octree_out.depth = self.full_depth
    for d in range(self.full_depth, self.depth_stop):
      label = octree_in.nempty_mask(d).long()
      octree_out.octree_split(label, d)
      octree_out.octree_grow(d + 1)
      octree_out.depth += 1
    return octree_out

  def forward(self, octree_in, octree_out=None, pos=None): # 这里的pos的大小为[batch_size * 5000, 4]，意思是把所有batch的points都concate在一起，用4的最后一个维度来表示batch_idx
    # generate dual octrees
    doctree_in = dual_octree.DualOctree(octree_in)
    doctree_in.post_processing_for_docnn()

    update_octree = octree_out is None
    if update_octree:
      octree_out = self.create_full_octree(octree_in)
      octree_out.depth = self.full_depth
    doctree_out = dual_octree.DualOctree(octree_out)
    doctree_out.post_processing_for_docnn()

    # run encoder and decoder

    # for auto-encoder:
    latent_code = self.octree_encoder(octree_in, doctree_in)
    out = self.octree_decoder(latent_code, doctree_out, doctree_in, update_octree)
    output = {'logits': out[0], 'reg_voxs': out[1], 'octree_out': out[2]}

    # compute function value with mpu
    if pos is not None:
      output['mpus'] = self.neural_mpu(pos, out[1], out[2])

    # create the mpu wrapper
    def _neural_mpu(pos):
      pred = self.neural_mpu(pos, out[1], out[2])
      return pred[self.depth_out][0]
    output['neural_mpu'] = _neural_mpu  # 这个output['neural_mpu']主要用于测试阶段，相当于对于任意输入的pos，根据最后一层的reg_voxs返回pos对应的sdf值。

    return output
