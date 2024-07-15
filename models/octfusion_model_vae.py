# Reference: diffusion is borrowed from the LDM repo: https://github.com/CompVis/latent-diffusion
# Specifically, functions from: https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/ddpm.py

import os
import sys
from collections import OrderedDict
from functools import partial
import copy

import numpy as np
from omegaconf import OmegaConf
from termcolor import colored, cprint
from einops import rearrange, repeat
from tqdm import tqdm
from random import random
import ocnn
from ocnn.nn import octree2voxel, octree_pad
from ocnn.octree import Octree, Points
from models.networks.dualoctree_networks import dual_octree

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.special import expm1

from models.base_model import BaseModel
from models.networks.diffusion_networks.graph_unet_union import UNet3DModel
from models.model_utils import load_dualoctree
from models.networks.diffusion_networks.ldm_diffusion_util import *

from models.networks.diffusion_networks.samplers.ddim_new import DDIMSampler
from models.networks.dualoctree_networks import loss
# distributed
from utils.distributed import reduce_loss_dict, get_rank

# rendering
from utils.util_dualoctree import calc_sdf
from utils.util import TorchRecoder, category_5_to_label

TRUNCATED_TIME = 0.7

class OctFusionModel(BaseModel):
    def name(self):
        return 'SDFusion-Model-Union-Two-Times'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.model_name = self.name()
        self.device = opt.device
        self.gradient_clip_val = 1.
        self.start_iter = opt.start_iter

        if self.isTrain:
            self.log_dir = os.path.join(opt.logs_dir, opt.name)
            self.train_dir = os.path.join(self.log_dir, 'train_temp')
            self.test_dir = os.path.join(self.log_dir, 'test_temp')


        ######## START: Define Networks ########
        assert opt.df_cfg is not None
        assert opt.vq_cfg is not None

        # init vae
        vq_conf = OmegaConf.load(opt.vq_cfg)

        self.vq_conf = vq_conf
        self.solver = self.vq_conf.solver

        self.input_depth = self.vq_conf.model.depth
        self.small_depth = self.vq_conf.model.depth_stop
        self.full_depth = self.vq_conf.model.full_depth

        self.load_octree = self.vq_conf.data.train.load_octree
        self.load_pointcloud = self.vq_conf.data.train.load_pointcloud
        self.load_split_small = self.vq_conf.data.train.load_split_small


        # init vqvae

        self.autoencoder = load_dualoctree(conf = vq_conf, ckpt = opt.vq_ckpt, opt = opt)
        self.autoencoder.train()
        set_requires_grad(self.autoencoder, True)

        ######## END: Define Networks ########

        if self.isTrain:

            # initialize optimizers
            self.optimizer = optim.AdamW([p for p in self.autoencoder.parameters() if p.requires_grad == True], lr=opt.lr)

            def poly(epoch, lr_power=0.9): 
                return (1 - epoch / opt.epochs) ** lr_power
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, poly)

            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]

            self.print_networks(verbose=False)

        if opt.ckpt is None and os.path.exists(os.path.join(opt.logs_dir, opt.name, "ckpt/df_steps-latest.pth")):
            opt.ckpt = os.path.join(opt.logs_dir, opt.name, "ckpt/df_steps-latest.pth")
        if opt.ckpt is not None:
            self.load_ckpt(opt.ckpt, self.autoencoder, load_opt=self.isTrain)
            if self.isTrain:
                self.optimizers = [self.optimizer]
        
        trainable_params_num = 0
        for m in [self.autoencoder]:
            trainable_params_num += sum([p.numel() for p in m.parameters() if p.requires_grad == True])
        print("Trainable_params: ", trainable_params_num)


        # for distributed training
        if self.opt.distributed:
            self.make_distributed(opt)
            self.autoencoder_module = self.autoencoder.module
        else:
            self.autoencoder_module = self.autoencoder

    def make_distributed(self, opt):
        if opt.sync_bn:
            self.autoencoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.autoencoder)
        self.autoencoder = nn.parallel.DistributedDataParallel(
            self.autoencoder,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )

    ############################ START: init diffusion params ############################

    def batch_to_cuda(self, batch):
        def points2octree(points):
            octree = ocnn.octree.Octree(depth = self.input_depth, full_depth = self.full_depth)
            octree.build_octree(points)
            return octree

        if self.load_pointcloud:
            points = [pts.cuda(non_blocking=True) for pts in batch['points']]
            octrees = [points2octree(pts) for pts in points]
            octree = ocnn.octree.merge_octrees(octrees)
            octree.construct_all_neigh()
            batch['octree_in'] = octree

        if self.load_octree:
            batch['octree_in'] = batch['octree_in'].cuda()

        for key in ['pos', 'sdf', 'grad']:
            batch[key] = batch[key].cuda()
        batch['pos'].requires_grad = True

    def set_input(self, input=None):
        self.batch_to_cuda(input)
        self.octree_in = input['octree_in']
        self.octree_gt = copy.deepcopy(self.octree_in)
        self.batch = input
        self.batch_size = self.octree_in.batch_size

    def switch_train(self):
        self.autoencoder.train()

    def switch_eval(self):
        self.autoencoder.eval()

    def get_loss_function(self):
        if self.vq_conf.loss.name.lower() == 'geometry':
            return loss.geometry_loss
        elif self.vq_conf.loss.name.lower() == 'color':
            return loss.geometry_color_loss
        else:
            raise ValueError


    def forward(self):

        model_out = self.autoencoder_module(self.octree_in, self.octree_gt, self.batch['pos'])
        loss_func = self.get_loss_function()
        output = loss_func(self.batch, model_out, self.vq_conf.loss.loss_type, kl_weight=self.vq_conf.loss.kl_weight)
        losses = [val for key, val in output.items() if 'loss' in key]
        output['loss'] = torch.sum(torch.stack(losses))
        output['code_max'] = model_out['code_max']
        output['code_min'] = model_out['code_min']
        self.loss = output['loss']
        self.output = output

    def inference(self):
        self.autoencoder.eval()
        output = self.autoencoder.forward(octree_in = self.batch['octree_in'], evaluate=True)
        filename = self.batch['filename'][0]
        pos = filename.rfind('.')
        if pos != -1: 
            filename = filename[:pos]  # remove the suffix
        save_dir = os.path.join(self.opt.logs_dir, self.opt.name, f"results_vae", filename)
        os.makedirs(save_dir, exist_ok=True)
        bbox = self.batch['bbox'][0].numpy() if 'bbox' in self.batch else None
        self.get_sdfs(output['neural_mpu'], self.batch_size, bbox)  # output['neural_mpu']是一个函数。
        self.export_mesh(save_dir, index = 0)

        # save the input point cloud
        pointcloud = trimesh.PointCloud(vertices=self.batch['points'][0].cpu().points.numpy())
        pointcloud.export(os.path.join(save_dir, 'input.ply'))


    def get_sdfs(self, neural_mpu, batch_size, bbox):
        # bbox used for marching cubes
        if bbox is not None:
            self.bbmin, self.bbmax = bbox[:3], bbox[3:]
        else:
            sdf_scale = self.solver.sdf_scale
            self.bbmin, self.bbmax = -sdf_scale, sdf_scale    # sdf_scale = 0.9

        self.sdfs = calc_sdf(neural_mpu, batch_size, size = self.solver.resolution, bbmin = self.bbmin, bbmax = self.bbmax)

    def export_mesh(self, save_dir, index = 0, level = 0, clean = False):
        try:
            os.makedirs(save_dir, exist_ok=True)
        except FileExistsError:
            pass
        ngen = self.sdfs.shape[0]
        size = self.solver.resolution
        mesh_scale=self.vq_conf.data.test.point_scale
        for i in range(ngen):
            filename = os.path.join(save_dir, f'{index + i}.obj')
            if ngen == 1:
                filename = os.path.join(save_dir, f'{index}.obj')
            sdf_value = self.sdfs[i].cpu().numpy()
            vtx, faces = np.zeros((0, 3)), np.zeros((0, 3))
            try:
                vtx, faces, _, _ = skimage.measure.marching_cubes(sdf_value, level)
            except:
                pass
            if vtx.size == 0 or faces.size == 0:
                print('Warning from marching cubes: Empty mesh!')
                return
            vtx = vtx * ((self.bbmax - self.bbmin) / size) + self.bbmin   # [0,sz]->[bbmin,bbmax]  把vertex放缩到[bbmin, bbmax]之间
            vtx = vtx * mesh_scale
            mesh = trimesh.Trimesh(vtx, faces)  # 利用Trimesh创建mesh并存储为obj文件。
            if clean:
                components = mesh.split(only_watertight=False)
                bbox = []
                for c in components:
                    bbmin = c.vertices.min(0)
                    bbmax = c.vertices.max(0)
                    bbox.append((bbmax - bbmin).max())
                max_component = np.argmax(bbox)
                mesh = components[max_component]
            mesh.export(filename)

    def backward(self):
        self.loss.backward()


    def optimize_parameters(self):

        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def get_current_errors(self):

        ret = OrderedDict([
            ('loss', self.loss.data),
        ])
        ret.update(self.output)

        if hasattr(self, 'loss_gamma'):
            ret['gamma'] = self.loss_gamma.data

        return ret

    def save(self, label, global_iter, save_opt=True):

        state_dict = {
            'autoencoder': self.autoencoder.state_dict(),
            'opt': self.optimizer.state_dict(),
            'global_step': global_iter,
        }

        # if save_opt:
        #     state_dict['opt'] = self.optimizer.state_dict()

        save_filename = 'vae_%s.pth' % (label)
        save_path = os.path.join(self.opt.ckpt_dir, save_filename)

        ckpts = os.listdir(self.opt.ckpt_dir)
        ckpts = [ck for ck in ckpts if ck!='df_steps-latest.pth']
        ckpts.sort(key=lambda x: int(x[9:-4]))
        if len(ckpts) > self.opt.ckpt_num:
            for ckpt in ckpts[:-self.opt.ckpt_num]:
                os.remove(os.path.join(self.opt.ckpt_dir, ckpt))

        torch.save(state_dict, save_path)

    def load_ckpt(self, ckpt, autoencoder, load_opt=False):
        map_fn = lambda storage, loc: storage
        if type(ckpt) == str:
            state_dict = torch.load(ckpt, map_location=map_fn)
        else:
            state_dict = ckpt
            
        autoencoder.load_state_dict(state_dict['autoencoder'])
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))

        if load_opt:
            self.start_iter = state_dict['global_step']
            print(colored('[*] training start from: %d' % self.start_iter, 'green'))
            self.optimizer.load_state_dict(state_dict['opt'])
            print(colored('[*] optimizer successfully restored from: %s' % ckpt, 'blue'))
