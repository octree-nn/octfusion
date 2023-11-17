# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import torch.nn

from . import modules
from . import dual_octree
from . import graph_ounet

class GraphAE(graph_ounet.GraphOUNet):
  def __init__(self, depth, channel_in, nout, full_depth=2, depth_out=6,
              resblk_type='bottleneck', bottleneck=4, resblk_num=3, code_channel=3):
    super().__init__(depth, channel_in, nout, full_depth, depth_out,
                  resblk_type, bottleneck, resblk_num)
    # this is to make the encoder and decoder symmetric
    n_edge_type, avg_degree = 7, 7
    self.decoder = torch.nn.ModuleList(
      [modules.GraphResBlocks(self.channels[d], self.channels[d],
      self.resblk_num[d], bottleneck, n_edge_type, avg_degree, d-1, resblk_type)
      for d in range(full_depth, depth + 1)])

    self.code_channel = code_channel
    channel_in = self.channels[self.full_depth]
    self.project1 = modules.Conv1x1Bn(channel_in, self.code_channel)
    self.project2 = modules.Conv1x1BnRelu(self.code_channel, channel_in)

  def octree_encoder(self, octree, doctree): # encoder的操作是从depth到full-deth为止，在这里就是从6到2
    convs = super().octree_encoder(octree, doctree) # conv的channel随着八叉树深度从6到2的变化为[32, 64, 128, 256, 512]
    # reduce the dimension
    code = self.project1(convs[self.full_depth])
    # constrain the code in [-1, 1]
    code = torch.tanh(code)  # [batch_size * (2 ** full_depth) ** 3, code_channel]
    return code

  def octree_decoder(self, latent_code, doctree_out, doctree=None, update_octree=False):
    logits = dict()
    reg_voxs = dict()
    deconvs = dict()

    deconvs[self.full_depth] = self.project2(latent_code)
    for i, d in enumerate(range(self.full_depth, self.depth_out+1)): # decoder的操作是从full_depth到depth_out为止
      if d > self.full_depth:
        nnum = doctree_out.nnum[d-1]
        leaf_mask = doctree_out.node_child(d-1) < 0
        deconvs[d] = self.upsample[i-1](deconvs[d-1], leaf_mask, nnum)

      edge_idx = doctree_out.graph[d]['edge_idx']
      edge_type = doctree_out.graph[d]['edge_dir']
      node_type = doctree_out.graph[d]['node_type']
      deconvs[d] = self.decoder[i](deconvs[d], edge_idx, edge_type, node_type)

      # predict the splitting label
      logit = self.predict[i](deconvs[d])
      nnum = doctree_out.nnum[d]
      logits[d] = logit[-nnum:]

      # update the octree according to predicted labels
      if update_octree:   # 测试阶段：如果update_octree为true，则从full_depth开始逐渐增长八叉树，直至depth_out
        label = logits[d].argmax(1).to(torch.int32)
        octree_out = doctree_out.octree
        octree_out.octree_split(label, d)
        if d < self.depth_out:
          octree_out.octree_grow(d + 1)  # 对初始化的满八叉树，根据预测的标签向上增长至depth_out
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

  def extract_code(self, octree_in):
    doctree_in = dual_octree.DualOctree(octree_in)
    doctree_in.post_processing_for_docnn()

    code = self.octree_encoder(octree_in, doctree_in)
    shape = [octree_in.batch_size, 2 ** self.full_depth, 2 ** self.full_depth, 2 ** self.full_depth, self.code_channel]
    code = code.view(shape)
    code = code.permute(0,4,1,2,3).contiguous()
    return code

  def decode_code(self, code, octree_in):
    # generate dual octrees
    code = code.permute(0,2,3,4,1).contiguous()
    code = code.reshape([-1,3])
    octree_out = self.create_full_octree(octree_in)
    octree_out.depth = self.full_depth
    doctree_out = dual_octree.DualOctree(octree_out)
    doctree_out.post_processing_for_docnn()

    # run encoder and decoder
    out = self.octree_decoder(code, doctree_out, doctree=None, update_octree=True)
    output = {'logits': out[0], 'reg_voxs': out[1], 'octree_out': out[2]}

    # create the mpu wrapper
    def _neural_mpu(pos):
      pred = self.neural_mpu(pos, out[1], out[2])
      return pred[self.depth_out][0]
    output['neural_mpu'] = _neural_mpu

    return output
