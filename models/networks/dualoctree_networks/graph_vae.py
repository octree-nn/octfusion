# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import torch.nn
from torch.nn import init

from .distributions import DiagonalGaussianDistribution
from . import modules_v1
from . import dual_octree
from . import graph_ounet_v1
from ocnn.nn import octree2voxel
from ocnn.octree import Octree
import copy

def init_weights(net, init_type='normal', gain=0.01):
    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'none':  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    net.apply(init_func)

    # propagate to children
    for m in net.children():
        m.apply(init_func)

class GraphVAE(graph_ounet_v1.GraphOUNet):

    def __init__(self, depth, channel_in, nout, full_depth=2, depth_stop = 6, depth_out=8, use_checkpoint = False,
                resblk_type='bottleneck', bottleneck=4,resblk_num=3, code_channel=3, embed_dim=3):
        super().__init__(depth, channel_in, nout, full_depth, depth_stop, depth_out, use_checkpoint,
                        resblk_type, bottleneck,resblk_num)
        # this is to make the encoder and decoder symmetric

        n_edge_type, avg_degree = 7, 7

        self.decoder_mid = torch.nn.Module()
        self.decoder_mid.block_1 = modules_v1.GraphResBlocks(self.channels[depth_stop], self.channels[depth_stop],self.dropout,
         self.resblk_nums[depth_stop], n_edge_type, avg_degree, depth_stop-1, self.use_checkpoint)
        self.decoder_mid.block_2 = modules_v1.GraphResBlocks(self.channels[depth_stop], self.channels[depth_stop],self.dropout,
         self.resblk_nums[depth_stop], n_edge_type, avg_degree, depth_stop-1, self.use_checkpoint)

        self.decoder = torch.nn.ModuleList(
            [modules_v1.GraphResBlocks(self.channels[d], self.channels[d],self.dropout,
         self.resblk_nums[d], n_edge_type, avg_degree, d-1, self.use_checkpoint)
            for d in range(depth_stop, depth + 1)])

        self.code_channel = code_channel
        ae_channel_in = self.channels[self.depth_stop]

        self.KL_conv = modules_v1.Conv1x1(ae_channel_in, 2 * embed_dim, use_bias = True)
        self.post_KL_conv = modules_v1.Conv1x1(embed_dim, ae_channel_in, use_bias = True)


    def octree_encoder(self, octree, doctree): # encoder的操作是从depth到full-deth为止，在这里就是从6到2
        convs = super().octree_encoder(octree, doctree) # conv的channel随着八叉树深度从6到2的变化为[32, 64, 128, 256, 512]
        # reduce the dimension
        code = self.KL_conv(convs[self.depth_stop])
        # print(code.max())
        # print(code.min())
        posterior = DiagonalGaussianDistribution(code, kl_std = 0.25)
        return posterior

    def octree_decoder(self, code, doctree_out, update_octree=False):
        #quant code [bs, 3, 16, 16, 16]
        code = self.post_KL_conv(code)   # [bs, code_channel, 16, 16, 16]
        octree_out = doctree_out.octree

        logits = dict()
        reg_voxs = dict()
        deconvs = dict()

        depth_stop = self.depth_stop

        deconvs[depth_stop] = code

        edge_idx = doctree_out.graph[depth_stop]['edge_idx']
        edge_type = doctree_out.graph[depth_stop]['edge_dir']
        node_type = doctree_out.graph[depth_stop]['node_type']

        deconvs[depth_stop] = self.decoder_mid.block_1(deconvs[depth_stop], doctree_out, depth_stop, edge_idx, edge_type, node_type)
        deconvs[depth_stop] = self.decoder_mid.block_2(deconvs[depth_stop], doctree_out, depth_stop, edge_idx, edge_type, node_type)

        for i, d in enumerate(range(self.depth_stop, self.depth_out+1)): # decoder的操作是从full_depth到depth_out为止
            if d > self.depth_stop:
                nnum = doctree_out.nnum[d-1]
                leaf_mask = doctree_out.node_child(d-1) < 0
                deconvs[d] = self.upsample[i-1](deconvs[d-1], doctree_out, d, leaf_mask, nnum)

            edge_idx = doctree_out.graph[d]['edge_idx']
            edge_type = doctree_out.graph[d]['edge_dir']
            node_type = doctree_out.graph[d]['node_type']
            octree_out = doctree_out.octree
            deconvs[d] = self.decoder[i](deconvs[d], doctree_out, d, edge_idx, edge_type, node_type)

            # predict the splitting label
            logit = self.predict[i]([deconvs[d], doctree_out, d])
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
            reg_vox = self.regress[i]([deconvs[d], doctree_out, d])

            # TODO: improve it
            # pad zeros to reg_vox to reuse the original code for ocnn
            node_mask = doctree_out.graph[d]['node_mask']
            shape = (node_mask.shape[0], reg_vox.shape[1])
            reg_vox_pad = torch.zeros(shape, device=reg_vox.device)
            reg_vox_pad[node_mask] = reg_vox
            reg_voxs[d] = reg_vox_pad

        return logits, reg_voxs, doctree_out.octree

    def forward(self, octree_in, octree_out=None, pos=None, evaluate=False): # 这里的pos的大小为[batch_size * 5000, 4]，意思是把所有batch的points都concate在一起，用4的最后一个维度来表示batch_idx
        # generate dual octrees
        doctree_in = dual_octree.DualOctree(octree_in)
        doctree_in.post_processing_for_docnn()

        update_octree = octree_out is None
        if update_octree:
            octree_out = self.create_full_octree(octree_in)
            octree_out.depth = self.full_depth
            for d in range(self.full_depth, self.depth_stop):
                label = octree_in.nempty_mask(d).long()
                octree_out.octree_split(label, d)
                octree_out.octree_grow(d + 1)
                octree_out.depth += 1

        doctree_out = dual_octree.DualOctree(octree_out)
        doctree_out.post_processing_for_docnn()

        # for auto-encoder:
        posterior = self.octree_encoder(octree_in, doctree_in)
        z = posterior.sample()

        if evaluate:
            z = posterior.sample()
            print(z.max(), z.min(), z.mean(), z.std())

        out = self.octree_decoder(z, doctree_out, update_octree)
        output = {'logits': out[0], 'reg_voxs': out[1], 'octree_out': out[2]}
        kl_loss = posterior.kl()
        output['kl_loss'] = kl_loss.mean()
        output['code_max'] = z.max()
        output['code_min'] = z.min()

        # compute function value with mpu
        if pos is not None:
            output['mpus'] = self.neural_mpu(pos, out[1], out[2])

        # create the mpu wrapper
        def _neural_mpu(pos):
            pred = self.neural_mpu(pos, out[1], out[2])
            return pred[self.depth_stop][0]
        output['neural_mpu'] = _neural_mpu  # 这个output['neural_mpu']主要用于测试阶段，相当于对于任意输入的pos，根据最后一层的reg_voxs返回pos对应的sdf值。

        return output

    def extract_code(self, octree_in):
        doctree_in = dual_octree.DualOctree(octree_in)
        doctree_in.post_processing_for_docnn()

        convs = super().octree_encoder(octree_in, doctree_in) # conv的channel随着八叉树深度从6到2的变化为[32, 64, 128, 256, 512]
        code = self.KL_conv(convs[self.depth_stop])
        posterior = DiagonalGaussianDistribution(code)
        return posterior.sample(), doctree_in

    def decode_code(self, code, doctree_in, update_octree = True, pos = None):

        octree_in = doctree_in.octree
        # generate dual octrees
        if update_octree:
            octree_out = self.create_child_octree(octree_in)
            doctree_out = dual_octree.DualOctree(octree_out)
            doctree_out.post_processing_for_docnn()
        else:
            doctree_out = doctree_in

        # run decoder
        out = self.octree_decoder(code, doctree_out, update_octree=update_octree)
        output = {'logits': out[0], 'reg_voxs': out[1], 'octree_out': out[2]}

        if pos is not None:
            output['mpus'] = self.neural_mpu(pos, out[1], out[2])

        # create the mpu wrapper
        def _neural_mpu(pos):
            pred = self.neural_mpu(pos, out[1], out[2])
            return pred[self.depth_out][0]
        output['neural_mpu'] = _neural_mpu

        return output
