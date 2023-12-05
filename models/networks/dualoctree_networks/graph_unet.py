# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import torch.nn

from . import dual_octree
from . import graph_ounet


class GraphUNet(graph_ounet.GraphOUNet):

  def _setup_channels_and_resblks(self):
    # self.resblk_num = [3] * 7 + [1] + [1] * 9
    self.resblk_num = [2] * 16
    self.channels = [4, 512, 512, 256, 128, 64, 32, 32, 32]

  def recons_decoder(self, convs, doctree_out):
    logits = dict()
    reg_voxs = dict()
    deconvs = dict()

    deconvs[self.full_depth] = convs[self.full_depth]
    for i, d in enumerate(range(self.full_depth, self.depth_out+1)):
      if d > self.full_depth:
        nnum = doctree_out.nnum[d-1]
        leaf_mask = doctree_out.node_child(d-1) < 0
        deconvd = self.upsample[i-1](deconvs[d-1], leaf_mask, nnum)
        deconvd = deconvd + convs[d]  # skip connections

        edge_idx = doctree_out.graph[d]['edge_idx']
        edge_type = doctree_out.graph[d]['edge_dir']
        node_type = doctree_out.graph[d]['node_type']
        deconvs[d] = self.decoder[i-1](deconvd, edge_idx, edge_type, node_type)

      # predict the splitting label
      logit = self.predict[i](deconvs[d])
      nnum = doctree_out.nnum[d]
      logits[d] = logit[-nnum:]

      # predict the signal
      reg_vox = self.regress[i](deconvs[d])

      # TODO: improve it
      # pad zeros to reg_vox to reuse the original code for ocnn
      node_mask = doctree_out.graph[d]['node_mask']
      shape = (node_mask.shape[0], reg_vox.shape[1])
      reg_vox_pad = torch.zeros(shape, device=reg_vox.device)
      reg_vox_pad[node_mask] = reg_vox
      reg_voxs[d] = reg_vox_pad

    return logits, reg_voxs

  def forward(self, octree_in, octree_out=None, pos=None):
    # octree_in and octree_out are the same for UNet
    doctree_in = dual_octree.DualOctree(octree_in)
    doctree_in.post_processing_for_docnn()

    # run encoder and decoder
    convs = self.octree_encoder(octree_in, doctree_in)
    out = self.recons_decoder(convs, doctree_in)
    output = {'reg_voxs': out[1], 'octree_out': octree_in}

    # compute function value with mpu
    if pos is not None:
      output['mpus'] = self.neural_mpu(pos, out[1], octree_in)

    # create the mpu wrapper
    def _neural_mpu(pos):
      pred = self.neural_mpu(pos, out[1], octree_in)
      return pred[self.depth_out][0]
    output['neural_mpu'] = _neural_mpu

    return output
