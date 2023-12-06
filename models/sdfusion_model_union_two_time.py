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
from models.networks.diffusion_networks.modules import octree_align

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.special import expm1

import torchvision.utils as vutils
import torchvision.transforms as transforms

from models.base_model import BaseModel
from models.networks.diffusion_networks.network_two_times import DiffusionUNet
from models.networks.diffusion_networks.ldm_diffusion_util import *

from models.networks.diffusion_networks.samplers.ddim_new import DDIMSampler

# distributed
from utils.distributed import reduce_loss_dict

# rendering
from utils.util_3d import init_mesh_renderer, render_sdf, render_sdf_dualoctree
from utils.util_dualoctree import calc_sdf

TRUNCATED_TIME = 0.7

class SDFusionModel(BaseModel):
    def name(self):
        return 'SDFusion-Model-Union-Two-Times'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.model_name = self.name()
        self.device = opt.device
        self.gradient_clip_val = 1.
        self.start_iter = opt.start_iter


        ######## START: Define Networks ########
        assert opt.df_cfg is not None
        assert opt.vq_cfg is not None

        # init df
        df_conf = OmegaConf.load(opt.df_cfg)
        vq_conf = OmegaConf.load(opt.vq_cfg)

        self.vq_conf = vq_conf
        self.solver = self.vq_conf.solver

        self.full_depth = self.vq_conf.model.full_depth
        self.small_depth = self.vq_conf.model.small_depth
        self.large_depth = self.vq_conf.model.depth_stop

        # init diffusion networks
        df_model_params = df_conf.model.params
        unet_params = df_conf.unet.params
        self.conditioning_key = df_model_params.conditioning_key
        self.num_timesteps = df_model_params.timesteps
        self.thres = 0.5
        if self.conditioning_key == 'adm':
            self.num_classes = unet_params.num_classes
        elif self.conditioning_key == 'None':
            self.num_classes = 1
        self.df = DiffusionUNet(unet_params, conditioning_key=self.conditioning_key)
        self.df.to(self.device)

        # record z_shape
        self.code_channel = 8
        z_sp_dim = 2 ** self.full_depth
        self.z_shape = (self.code_channel, z_sp_dim, z_sp_dim, z_sp_dim)

        self.ema_df = copy.deepcopy(self.df)
        self.ema_df.to(self.device)
        if opt.isTrain:
            self.ema_rate = opt.ema_rate
            self.ema_updater = EMA(self.ema_rate)
            self.reset_parameters()
            set_requires_grad(self.ema_df, False)

        self.noise_schedule = 'linear'
        if self.noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif self.noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {self.noise_schedule}')

        ######## END: Define Networks ########

        if self.isTrain:

            # initialize optimizers
            self.optimizer = optim.AdamW([p for p in self.df.parameters() if p.requires_grad == True], lr=opt.lr)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1000, 0.9)

            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]

            self.print_networks(verbose=False)

        if opt.ckpt is not None:
            self.load_ckpt(opt.ckpt, load_opt=self.isTrain)
            if self.isTrain:
                self.optimizers = [self.optimizer]
            # self.schedulers = [self.scheduler]


        # setup renderer
        if 'snet' in opt.dataset_mode:
            dist, elev, azim = 1.7, 20, 20
        elif 'pix3d' in opt.dataset_mode:
            dist, elev, azim = 1.7, 20, 20
        elif opt.dataset_mode == 'buildingnet':
            dist, elev, azim = 1.0, 20, 20

        self.renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim, device=self.device)

        # for distributed training
        if self.opt.distributed:
            self.make_distributed(opt)
            self.df_module = self.df.module

        else:
            self.df_module = self.df

    def reset_parameters(self):
        self.ema_df.load_state_dict(self.df.state_dict())

    def make_distributed(self, opt):
        self.df = nn.parallel.DistributedDataParallel(
            self.df,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    ############################ START: init diffusion params ############################

    def batch_to_cuda(self, batch):
        def points2octree(points):
            octree = ocnn.octree.Octree(depth = self.large_depth, full_depth = self.full_depth)
            octree.build_octree(points)
            return octree

        points = [pts.cuda(non_blocking=True) for pts in batch['points']]
        octrees = [points2octree(pts) for pts in points]
        octree = ocnn.octree.merge_octrees(octrees)
        octree.construct_all_neigh()
        batch['octree_in'] = octree

        batch['split_small'] = self.octree2split_small(batch['octree_in'])
        batch['split_large'] = self.octree2split_large(batch['octree_in'])

    def set_input(self, input=None):
        self.batch_to_cuda(input)
        self.split_small = input['split_small']
        self.split_large = input['split_large']
        self.octree_in = input['octree_in']
        self.batch_size = self.octree_in.batch_size
        self.label = input['label']

    def switch_train(self):
        self.df.train()

    def switch_eval(self):
        self.df.eval()


    # check: ddpm.py, line 871 forward
    # check: p_losses
    # check: q_sample, apply_mode

    def forward(self):

        self.df.train()

        c = None

        batch_size = self.batch_size

        self.stage_flag = ''

        if random() < 0.5:

            self.stage_flag = 'small'

            times1 = torch.zeros((batch_size,), device = self.device).float().uniform_(0,1)
            times2 = torch.ones((batch_size,), device = self.device).float()

            noise_level1 = self.log_snr(times1)
            padded_noise_level1 = right_pad_dims_to(self.split_small, noise_level1)
            alpha1, sigma1 = log_snr_to_alpha_sigma(padded_noise_level1)

            noise = torch.randn_like(self.split_small, device = self.device)
            noised_split_small = alpha1 * self.split_small + sigma1 * noise

            noised_octree_small = self.split2octree_small(noised_split_small)

            noise_level2 = self.log_snr(times2)

            noised_doctree = dual_octree.DualOctree(noised_octree_small)
            noised_doctree.post_processing_for_docnn()

        else:

            self.stage_flag = 'large'

            times1 = torch.zeros((batch_size,), device = self.device).float()
            times2 = torch.zeros((batch_size,), device = self.device).float().uniform_(0,1)

            noise_level1 = self.log_snr(times1)

            noise_level2 = self.log_snr(times2)
            alpha2, sigma2 = log_snr_to_alpha_sigma(noise_level2)

            noised_octree_small = self.split2octree_small(self.split_small)

            batch_id = noised_octree_small.batch_id(depth = self.small_depth)

            noise = torch.randn_like(self.split_large)
            noised_split_large = self.split_large.clone()

            for i in range(batch_size):
                noised_split_large[batch_id == i] *= alpha2[i]
                noise_i = noise[batch_id == i]
                sigma_i = sigma2[i] * noise_i
                noised_split_large[batch_id == i] += sigma_i

            noised_octree_large = self.split2octree_large(noised_octree_small, noised_split_large)

            noised_doctree = dual_octree.DualOctree(noised_octree_large)
            noised_doctree.post_processing_for_docnn()

        input_data = torch.zeros((noised_doctree.total_num,1), device = self.device)

        doctree_gt = dual_octree.DualOctree(self.octree_in)
        doctree_gt.post_processing_for_docnn()

        _, logits, _ = self.df(input_data, doctree_in = noised_doctree, doctree_out = doctree_gt, t1 = noise_level1, t2 = noise_level2)
        # self.df_feature_loss = F.mse_loss(out, output_data)
        self.df_feature_loss = torch.tensor(0.)

        self.df_split_loss = 0.

        if self.stage_flag == 'small':
            for d in range(self.full_depth, self.small_depth):
                logitd = logits[d]
                label_gt = self.octree_in.nempty_mask(d).float()
                label_gt = label_gt * 2 - 1
                self.df_split_loss += F.mse_loss(logitd, label_gt)

        elif self.stage_flag == 'large':
            for d in logits.keys():
                logitd = logits[d]
                label_gt = self.octree_in.nempty_mask(d).float()
                label_gt = label_gt * 2 - 1
                self.df_split_loss += F.mse_loss(logitd, label_gt)

        self.loss = self.df_feature_loss + self.df_split_loss

    def get_sampling_timesteps(self, batch, device, steps):
        times = torch.linspace(1., 0., steps + 1, device=device)
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    @torch.no_grad()
    def uncond(self, batch_size=16, steps=200, category = 0, ema = True, truncated_index: float = 0.0, save_dir = None, index = 0):

        if ema:
            self.ema_df.eval()
        else:
            self.df.eval()

        shape = (batch_size, *self.z_shape)

        small_time_pairs = self.get_sampling_timesteps(
            batch_size, device=self.device, steps=steps)

        noised_split_small = torch.randn(shape, device = self.device)
        noised_octree_small = self.split2octree_small(noised_split_small)

        noised_doctree = dual_octree.DualOctree(noised_octree_small)
        noised_doctree.post_processing_for_docnn()

        x_start_small = None

        small_iter = tqdm(small_time_pairs, desc='small sampling loop time step')

        for time1, time_next1 in small_iter:

            log_snr = self.log_snr(time1)
            log_snr_next = self.log_snr(time_next1)
            log_snr, log_snr_next = map(
                partial(right_pad_dims_to, noised_split_small), (log_snr, log_snr_next))

            alpha, _ = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            noise_cond1 = self.log_snr(time1)
            time2 = torch.ones(batch_size, device = self.device)
            noise_cond2 = self.log_snr(time2)

            input_data = torch.zeros((noised_doctree.total_num, 1), device = self.device)

            if ema:
                _,logits, doctree_out  = self.ema_df(input_data, doctree_in = noised_doctree, doctree_out = None, t1 = noise_cond1, t2 = noise_cond2)
            else:
                _,logits, doctree_out = self.df(input_data, doctree_in = noised_doctree, doctree_out = None, t1 = noise_cond1, t2 = noise_cond2)

            # self.export_octree(octree_out, depth = self.large_depth, save_dir = 'pred_hr_airplane', index = time.item())

            x_start_small = self.logits2voxel(logits, octree = noised_doctree.octree)

            if time1[0] < TRUNCATED_TIME:
                x_start_small.sign_()

            c = -expm1(log_snr - log_snr_next)
            mean = alpha_next * (noised_split_small * (1 - c) / alpha + c * x_start_small)
            variance = (sigma_next ** 2) * c
            noise = torch.where(
                rearrange(time_next1 > truncated_index, 'b -> b 1 1 1 1'),
                torch.randn_like(noised_split_small),
                torch.zeros_like(noised_split_small)
            )
            noised_split_small = mean + torch.sqrt(variance) * noise

            noised_octree_small = self.split2octree_small(noised_split_small)

            noised_doctree = dual_octree.DualOctree(noised_octree_small)
            noised_doctree.post_processing_for_docnn()

        self.export_octree(noised_octree_small, self.small_depth, save_dir = 'airplane_lr', index = index)

        noised_doctree_small = dual_octree.DualOctree(noised_octree_small)
        noised_doctree_small.post_processing_for_docnn()

        noised_octree_nnum = len(noised_octree_small.batch_id(depth = self.small_depth))
        noised_split_large = torch.randn((noised_octree_nnum, self.code_channel), device = self.device)
        noised_octree_large = self.split2octree_large(noised_octree_small, noised_split_large)

        noised_doctree = dual_octree.DualOctree(noised_octree_large)
        noised_doctree.post_processing_for_docnn()

        time_pairs = self.get_sampling_timesteps(
            batch_size, device=self.device, steps=steps)

        large_iter = tqdm(time_pairs, desc='large sampling loop time step')
        x_start_large = None

        for time2, time_next2 in large_iter:

            log_snr = self.log_snr(time2)
            log_snr_next = self.log_snr(time_next2)

            alpha, _ = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            time1 = torch.zeros(batch_size, device = self.device)
            noise_cond1 = self.log_snr(time1)
            noise_cond2 = self.log_snr(time2)

            input_data = torch.zeros((noised_doctree.total_num, 1), device = self.device)

            if ema:
                _,logits, doctree_out  = self.ema_df(input_data, doctree_in = noised_doctree, doctree_out = noised_doctree_small, t1 = noise_cond1, t2 = noise_cond2)
            else:
                _,logits, doctree_out = self.df(input_data, doctree_in = noised_doctree, doctree_out = noised_doctree_small, t1 = noise_cond1, t2 = noise_cond2)

            octree_out = doctree_out.octree
            # self.export_octree(octree_out, depth = self.large_depth, save_dir = 'pred_hr_airplane', index = time.item())

            x_start_large = self.octree2split_large(octree_out)

            c = -expm1(log_snr - log_snr_next)
            c, alpha, alpha_next, sigma_next = c[0].item(), alpha[0].item(), alpha_next[0].item(), sigma_next[0].item()
            mean = alpha_next * (noised_split_large * (1 - c) / alpha + c * x_start_large)
            variance = torch.tensor((sigma_next ** 2) * c).to(self.device)
            noised_split_large = mean + torch.sqrt(variance) * torch.randn_like(noised_split_large)

            noised_octree_large = self.split2octree_large(noised_octree_small, noised_split_large)

            noised_doctree = dual_octree.DualOctree(noised_octree_large)
            noised_doctree.post_processing_for_docnn()

        octree_out = noised_octree_large
        self.export_octree(octree_out, depth = self.large_depth, save_dir = 'airplane_hr', index = index)

    def logits2voxel(self, logits, octree):

        logit_full = logits[self.full_depth]
        logit_full_p1 = logits[self.full_depth + 1]
        logit_full_p1 = logit_full_p1.reshape(-1, 8)
        total_logits = torch.zeros((len(logit_full), 8), device = self.device) - 1
        total_logits[logit_full > 0] = logit_full_p1
        x_start = octree2voxel(total_logits, octree = octree, depth = self.full_depth)
        x_start = x_start.permute(0,4,1,2,3).contiguous()

        return x_start

    def octree2split_small(self, octree):

        child_full_p1 = octree.children[self.full_depth + 1]
        split_full_p1 = (child_full_p1 >= 0)
        split_full_p1 = split_full_p1.reshape(-1, 8)
        split_full = octree_pad(data = split_full_p1, octree = octree, depth = self.full_depth)
        split_full = octree2voxel(data=split_full, octree=octree, depth = self.full_depth)
        split_full = split_full.permute(0,4,1,2,3).contiguous()

        split_full = split_full.float()
        split_full = 2 * split_full - 1  # scale to [-1, 1]

        return split_full

    def octree2split_large(self, octree):

        child_small_p1 = octree.children[self.small_depth + 1]
        split_small_p1 = (child_small_p1 >= 0)
        split_small_p1 = split_small_p1.reshape(-1, 8)
        split_small = octree_pad(data = split_small_p1, octree = octree, depth = self.small_depth)

        split_small = split_small.float()
        split_small = 2 * split_small - 1    # scale to [-1, 1]

        return split_small

    def split2octree_small(self, split):

        discrete_split = copy.deepcopy(split)
        discrete_split[discrete_split > 0] = 1
        discrete_split[discrete_split < 0] = 0

        batch_size = discrete_split.shape[0]
        octree_out = create_full_octree(depth = self.large_depth, full_depth = self.full_depth, batch_size = batch_size, device = self.device)
        split_sum = torch.sum(discrete_split, dim = 1)
        nempty_mask_voxel = (split_sum > 0)
        x, y, z, b = octree_out.xyzb(self.full_depth)
        nempty_mask = nempty_mask_voxel[b,x,y,z]
        label = nempty_mask.long()
        octree_out.octree_split(label, self.full_depth)
        octree_out.octree_grow(self.full_depth + 1)
        octree_out.depth += 1

        x, y, z, b = octree_out.xyzb(depth = self.full_depth, nempty = True)
        nempty_mask_p1 = discrete_split[b,:,x,y,z]
        nempty_mask_p1 = nempty_mask_p1.reshape(-1)
        label_p1 = nempty_mask_p1.long()
        octree_out.octree_split(label_p1, self.full_depth + 1)
        octree_out.octree_grow(self.full_depth + 2)
        octree_out.depth += 1

        return octree_out

    def split2octree_large(self, octree, split):

        discrete_split = copy.deepcopy(split)
        discrete_split[discrete_split > 0] = 1
        discrete_split[discrete_split < 0] = 0

        octree_out = copy.deepcopy(octree)
        split_sum = torch.sum(discrete_split, dim = 1)
        nempty_mask_small = (split_sum > 0)
        label = nempty_mask_small.long()
        octree_out.octree_split(label, depth = self.small_depth)
        octree_out.octree_grow(self.small_depth + 1)
        octree_out.depth += 1

        nempty_mask_small_p1 = discrete_split[split_sum > 0]
        nempty_mask_small_p1 = nempty_mask_small_p1.reshape(-1)
        label_p1 = nempty_mask_small_p1.long()
        octree_out.octree_split(label_p1, depth = self.small_depth + 1)
        octree_out.octree_grow(self.small_depth + 2)
        octree_out.depth += 1

        return octree_out

    def export_octree(self, octree, depth, save_dir = None, index = 0):

        if not os.path.exists(save_dir): os.makedirs(save_dir)

        batch_id = octree.batch_id(depth = depth, nempty = False)
        data = torch.ones((len(batch_id), 1), device = self.device)
        data = octree2voxel(data = data, octree = octree, depth = depth, nempty = False)
        data = data.permute(0,4,1,2,3).contiguous()

        batch_size = octree.batch_size

        for i in tqdm(range(batch_size)):
            voxel = data[i].squeeze().cpu().numpy()
            mesh = voxel2mesh(voxel)
            if batch_size == 1:
                mesh.export(os.path.join(save_dir, f'{index}.obj'))
            else:
                mesh.export(os.path.join(save_dir, f'{i}.obj'))


    def get_sdfs(self, neural_mpu, batch_size, bbox):
        # bbox used for marching cubes
        if bbox is not None:
            self.bbmin, self.bbmax = bbox[:3], bbox[3:]
        else:
            sdf_scale = self.solver.sdf_scale
            self.bbmin, self.bbmax = -sdf_scale, sdf_scale    # sdf_scale = 0.9

        self.sdfs = calc_sdf(neural_mpu, batch_size, size = self.solver.resolution, bbmin = self.bbmin, bbmax = self.bbmax)

    def get_mesh(self,ngen, save_dir, level = 0):
        sdf_values = self.sdfs
        size = self.solver.resolution
        bbmin = self.bbmin
        bbmax = self.bbmax
        mesh_scale=self.vq_conf.data.test.point_scale
        for i in range(ngen):
            filename = os.path.join(save_dir, f'{i}.obj')
            sdf_value = sdf_values[i].cpu().numpy()
            vtx, faces = np.zeros((0, 3)), np.zeros((0, 3))
            try:
                vtx, faces, _, _ = skimage.measure.marching_cubes(sdf_value, level)
            except:
                pass
            if vtx.size == 0 or faces.size == 0:
                print('Warning from marching cubes: Empty mesh!')
                return
            vtx = vtx * ((bbmax - bbmin) / size) + bbmin   # [0,sz]->[bbmin,bbmax]  把vertex放缩到[bbmin, bbmax]之间
            vtx = vtx * mesh_scale
            mesh = trimesh.Trimesh(vtx, faces)  # 利用Trimesh创建mesh并存储为obj文件。
            mesh.export(filename)

    @torch.no_grad()
    def eval_metrics(self, dataloader, thres=0.0, global_step=0):
        self.eval()

        ret = OrderedDict([
            ('dummy_metrics', 0.0),
        ])
        self.train()
        return ret

    def backward(self):

        self.loss.backward()

    def update_EMA(self):
        update_moving_average(self.ema_df, self.df, self.ema_updater)

    def optimize_parameters(self):

        self.set_requires_grad([self.df], requires_grad=True)

        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
        self.update_EMA()

    def get_current_errors(self):

        ret = OrderedDict([
            ('diffusion feature', self.df_feature_loss.data),
            ('diffusion logit', self.df_split_loss.data),
            ('total', self.loss.data),
        ])

        if hasattr(self, 'loss_gamma'):
            ret['gamma'] = self.loss_gamma.data

        return ret, self.stage_flag

    def get_current_visuals(self):

        with torch.no_grad():
            self.img_gen_df = render_sdf_dualoctree(self.renderer, self.sdfs, level=0,
                                                bbmin = self.bbmin, bbmax = self.bbmax,
                                                mesh_scale = self.vq_conf.data.test.point_scale, render_all = True)
            # self.img_gen_df = render_sdf(self.renderer, self.gen_df)

        vis_tensor_names = [
            'img_gen_df',
        ]

        vis_ims = self.tnsrs2ims(vis_tensor_names)
        visuals = zip(vis_tensor_names, vis_ims)

        return OrderedDict(visuals)

    def save(self, label, global_iter, save_opt=True):

        state_dict = {
            'df': self.df_module.state_dict(),
            'ema_df': self.ema_df.state_dict(),
            'opt': self.optimizer.state_dict(),
            'global_step': global_iter,
        }

        # if save_opt:
        #     state_dict['opt'] = self.optimizer.state_dict()

        save_filename = 'df_%s.pth' % (label)
        save_path = os.path.join(self.opt.ckpt_dir, save_filename)

        ckpts = os.listdir(self.opt.ckpt_dir)
        ckpts = [ck for ck in ckpts if ck!='df_steps-latest.pth']
        ckpts.sort(key=lambda x: int(x[9:-4]))
        if len(ckpts) > self.opt.ckpt_num:
            for ckpt in ckpts[:-self.opt.ckpt_num]:
                os.remove(os.path.join(self.opt.ckpt_dir, ckpt))

        torch.save(state_dict, save_path)

    def load_ckpt(self, ckpt, load_opt=True):
        map_fn = lambda storage, loc: storage
        if type(ckpt) == str:
            state_dict = torch.load(ckpt, map_location=map_fn)
        else:
            state_dict = ckpt

        # self.vqvae.load_state_dict(state_dict['vqvae'])
        self.df.load_state_dict(state_dict['df'])
        self.ema_df.load_state_dict(state_dict['ema_df'])
        self.start_iter = state_dict['global_step']
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))

        if load_opt:
            self.optimizer.load_state_dict(state_dict['opt'])
            print(colored('[*] optimizer successfully restored from: %s' % ckpt, 'blue'))
