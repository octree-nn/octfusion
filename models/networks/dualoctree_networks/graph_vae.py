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
from . import modules
from . import dual_octree
from . import mpu
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

class GraphVAE(torch.nn.Module):

    def __init__(self, depth, channel_in, nout, full_depth=2, depth_stop = 6, depth_out=8, use_checkpoint = False, resblk_type='bottleneck', bottleneck=4,resblk_num=3, code_channel=3, embed_dim=3):
        # super().__init__(depth, channel_in, nout, full_depth, depth_stop, depth_out, use_checkpoint, resblk_type, bottleneck,resblk_num)
        # this is to make the encoder and decoder symmetric

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
        n_edge_type, avg_degree = 7, 7

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
        self.decoder = torch.nn.ModuleList(
            [modules.GraphResBlocks(self.channels[d], self.channels[d],self.dropout,
         self.resblk_nums[d], n_edge_type, avg_degree, d-1, self.use_checkpoint)
            for d in range(depth_stop, depth + 1)])
        self.decoder_mid = torch.nn.Module()
        self.decoder_mid.block_1 = modules.GraphResBlocks(self.channels[depth_stop], self.channels[depth_stop],self.dropout,
         self.resblk_nums[depth_stop], n_edge_type, avg_degree, depth_stop-1, self.use_checkpoint)
        self.decoder_mid.block_2 = modules.GraphResBlocks(self.channels[depth_stop], self.channels[depth_stop],self.dropout,
         self.resblk_nums[depth_stop], n_edge_type, avg_degree, depth_stop-1, self.use_checkpoint)
        
        self.upsample = torch.nn.ModuleList(
            [modules.GraphUpsample(self.channels[d-1], self.channels[d])
            for d in range(depth_stop + 1, depth + 1)])

        # header
        self.predict = torch.nn.ModuleList(
            [self._make_predict_module(self.channels[d], 2)  # 这里的2就是当前节点是否要劈成八份的label
            for d in range(depth_stop, depth + 1)])
        self.regress = torch.nn.ModuleList(
            [self._make_predict_module(self.channels[d], 4)  # 这里的4就是王老师说的，MPU里一个node里的4个特征分别代表法向量和偏移量
            for d in range(depth_stop, depth + 1)])
        

        self.code_channel = code_channel
        ae_channel_in = self.channels[self.depth_stop]

        self.KL_conv = modules.Conv1x1(ae_channel_in, 2 * embed_dim, use_bias = True)
        self.post_KL_conv = modules.Conv1x1(embed_dim, ae_channel_in, use_bias = True)

    def _setup_channels_and_resblks(self):
        # self.resblk_num = [3] * 7 + [1] + [1] * 9
        # self.resblk_num = [3] * 16
        self.resblk_nums = [self.resblk_num] * 16      # resblk_num[d] 为深度d（分辨率）下resblock的数量。
        # self.channels = [4, 512, 512, 256, 128, 64, 32, 32, 24, 16, 8]  # depth i的channel为channels[i]
        self.channels = [4, 512, 512, 256, 128, 64, 32, 32, 24, 8]

    def _make_predict_module(self, channel_in, channel_out=2, num_hidden=32):
        return torch.nn.Sequential(
        modules.Conv1x1GnGeluSequential(channel_in, num_hidden),
        modules.Conv1x1(num_hidden, channel_out, use_bias=True))

    def _get_input_feature(self, doctree):
        return doctree.get_input_feature()

    def octree_encoder_step(self, octree, doctree):
        depth, depth_stop = self.depth, self.depth_stop
        data = self._get_input_feature(doctree)

        convs = dict()
        convs[depth] = data   # channel为4
        for i, d in enumerate(range(depth, depth_stop-1, -1)):   # encoder的操作是从depth到depth_stop为止
        # perform graph conv
            convd = convs[d]  # get convd
            if d == self.depth:  # the first conv
                convd = self.conv1(convd, doctree, d)
            convd = self.encoder[i](convd, doctree, d)
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
    
    def octree_encoder(self, octree, doctree): # encoder的操作是从depth到full-deth为止，在这里就是从6到2
        convs = self.octree_encoder_step(octree, doctree) # conv的channel随着八叉树深度从6到2的变化为[32, 64, 128, 256, 512]
        # reduce the dimension
        code = self.KL_conv(convs[self.depth_stop])
        # print(code.max())
        # print(code.min())
        posterior = DiagonalGaussianDistribution(code)
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

        deconvs[depth_stop] = self.decoder_mid.block_1(deconvs[depth_stop], doctree_out, depth_stop)
        deconvs[depth_stop] = self.decoder_mid.block_2(deconvs[depth_stop], doctree_out, depth_stop)

        for i, d in enumerate(range(self.depth_stop, self.depth_out+1)): # decoder的操作是从full_depth到depth_out为止
            if d > self.depth_stop:
                nnum = doctree_out.nnum[d-1]
                leaf_mask = doctree_out.node_child(d-1) < 0
                deconvs[d] = self.upsample[i-1](deconvs[d-1], doctree_out, d, leaf_mask, nnum)

            octree_out = doctree_out.octree
            deconvs[d] = self.decoder[i](deconvs[d], doctree_out, d)

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

        convs = self.octree_encoder_step(octree_in, doctree_in) # conv的channel随着八叉树深度从6到2的变化为[32, 64, 128, 256, 512]
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
