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

# distributed
from utils.distributed import reduce_loss_dict, get_rank

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

        if self.isTrain:
            self.log_dir = os.path.join(opt.logs_dir, opt.name)
            self.train_dir = os.path.join(self.log_dir, 'train_images')
            self.test_dir = os.path.join(self.log_dir, 'test_images')


        ######## START: Define Networks ########
        assert opt.df_cfg is not None
        assert opt.vq_cfg is not None

        # init df
        df_conf = OmegaConf.load(opt.df_cfg)
        vq_conf = OmegaConf.load(opt.vq_cfg)

        self.vq_conf = vq_conf
        self.solver = self.vq_conf.solver

        self.input_depth = self.vq_conf.model.depth
        self.small_depth = self.vq_conf.model.depth_stop
        self.full_depth = self.vq_conf.model.full_depth

        self.load_octree = self.vq_conf.data.train.load_octree
        self.load_pointcloud = self.vq_conf.data.train.load_pointcloud

        # init diffusion networks
        df_model_params = df_conf.model.params
        unet_params = df_conf.unet.params
        self.conditioning_key = df_model_params.conditioning_key
        self.num_timesteps = df_model_params.timesteps

        self.df = UNet3DModel(unet_params)
        self.df.to(self.device)

        # record z_shape
        self.split_channel = 8
        self.code_channel = self.vq_conf.model.embed_dim
        z_sp_dim = 2 ** self.full_depth
        self.z_shape = (self.split_channel, z_sp_dim, z_sp_dim, z_sp_dim)

        self.ema_df = copy.deepcopy(self.df)
        self.ema_df.to(self.device)
        if opt.isTrain:
            self.ema_rate = opt.ema_rate
            self.ema_updater = EMA(self.ema_rate)
            self.reset_parameters()
            set_requires_grad(self.ema_df, False)

        self.noise_schedule = "linear"
        if self.noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif self.noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {self.noise_schedule}')

        # init vqvae

        self.autoencoder = load_dualoctree(conf = vq_conf, ckpt = opt.vq_ckpt, opt = opt)

        trainable_params = []
        self.set_requires_grad([
            self.df.unet_lr
        ], False)

        trainable_params_num = 0
        for m in [self.df]:
            trainable_params += [p for p in m.parameters() if p.requires_grad == True]
            trainable_params_num += sum([p.numel() for p in m.parameters() if p.requires_grad == True])
        print("Trainable_params: ", trainable_params_num)
        ######## END: Define Networks ########

        if self.isTrain:

            # initialize optimizers
            self.optimizer = optim.AdamW([p for p in self.df.parameters() if p.requires_grad == True], lr=opt.lr)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1000, 0.9)

            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]

            self.print_networks(verbose=False)

        if opt.pretrain_ckpt is not None:
            self.load_ckpt(opt.pretrain_ckpt, self.df.unet_lr, self.ema_df.unet_lr, load_opt=False)

        if opt.ckpt is None and os.path.exists(os.path.join(opt.logs_dir, opt.name, "ckpt/df_steps-latest.pth")):
            opt.ckpt = os.path.join(opt.logs_dir, opt.name, "ckpt/df_steps-latest.pth")
        if opt.ckpt is not None:
            self.load_ckpt(opt.ckpt, self.df, self.ema_df, load_opt=self.isTrain)
            if self.isTrain:
                self.optimizers = [self.optimizer]


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
            self.autoencoder_module = self.autoencoder.module

        else:
            self.df_module = self.df
            self.autoencoder_module = self.autoencoder

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

        with torch.no_grad():
            self.input_data, self.doctree_in = self.autoencoder_module.extract_code(self.octree_in)

        self.df_feature_loss = torch.tensor(0., device=self.device)
        self.df_split_loss = torch.tensor(0., device=self.device)
        stage_flag = "HR"

        if stage_flag == "LR":
            times = torch.zeros(
                (self.batch_size,), device=self.device).float().uniform_(0, 1)
            split_small = self.octree2split_small(self.doctree_in.octree)
            noise = torch.randn_like(split_small)

            noise_level = self.log_snr(times)
            padded_noise_level = right_pad_dims_to(split_small, noise_level)
            alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
            noised_split_small = alpha * split_small + sigma * noise

            x_self_cond = None
            if random() < 0.5:
                with torch.no_grad():
                    x_self_cond = self.df(x_lr = noised_split_small, timesteps = noise_level, label = self.label)

            pred = self.df(x_lr = noised_split_small, timesteps = noise_level, x_self_cond = x_self_cond, label = self.label)

            self.df_split_loss = F.mse_loss(pred, split_small)
        else:
            times = torch.zeros((self.batch_size,), device = self.device).float().uniform_(0,1)
            noise_level = self.log_snr(times)

            alpha, sigma = log_snr_to_alpha_sigma(noise_level)

            noised_feature = self.input_data.clone()

            batch_id = self.doctree_in.batch_id(depth = self.small_depth)
            noise = torch.randn_like(noised_feature, device = self.device)

            batch_alpha = alpha[batch_id].unsqueeze(1)
            batch_sigma = sigma[batch_id].unsqueeze(1)
            noised_feature = noised_feature * batch_alpha + noise * batch_sigma

            output = self.df(x_hr = noised_feature, doctree = self.doctree_in, timesteps = noise_level)

            self.df_feature_loss = F.mse_loss(output, noise)

        self.loss = self.df_split_loss + self.df_feature_loss


    def get_sampling_timesteps(self, batch, device, steps):
        times = torch.linspace(1., 0., steps + 1, device=device)
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    @torch.no_grad()
    def uncond_octree(self, ema=False, ddim_steps=200, truncated_index = 0.0):

        
        small_time_pairs = self.get_sampling_timesteps(
            self.batch_size, device=self.device, steps=ddim_steps)

        shape = (self.batch_size, *self.z_shape)
        noised_split_small = torch.randn(shape, device = self.device)

        # label = torch.zeros(self.batch_size).to(self.device)
        # label += category_5_to_label[category]
        # label = label.long()
        label = None

        x_start_small = None

        small_iter = tqdm(small_time_pairs, desc='small sampling loop time step')

        for time, time_next in small_iter:

            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            log_snr, log_snr_next = map(
                partial(right_pad_dims_to, noised_split_small), (log_snr, log_snr_next))

            alpha, _ = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            noise_cond = self.log_snr(time)

            if ema:
                x_start_small = self.ema_df(x_lr = noised_split_small, timesteps = noise_cond, x_self_cond = x_start_small, label = label)
            else:
                x_start_small = self.df(x_lr = noised_split_small, timesteps = noise_cond, x_self_cond = x_start_small, label = label)

            if time[0] < TRUNCATED_TIME:
                x_start_small.sign_()

            c = -expm1(log_snr - log_snr_next)
            mean = alpha_next * (noised_split_small * (1 - c) / alpha + c * x_start_small)
            variance = (sigma_next ** 2) * c
            noise = torch.where(
                rearrange(time_next > truncated_index, 'b -> b 1 1 1 1'),
                torch.randn_like(noised_split_small),
                torch.zeros_like(noised_split_small)
            )
            noised_split_small = mean + torch.sqrt(variance) * noise
        return noised_split_small
    
    @torch.no_grad()
    def uncond(self, data, split_path, category = 'airplane', suffix = 'mesh_2t', ema = False, ddim_steps=200, ddim_eta=0., clean = False, save_index = 0):

        if ema:
            self.ema_df.eval()

        else:
            self.df.eval()

        if data != None:
            self.set_input(data)
            split_small = self.split_small
        elif split_path != None:
            split_small = torch.load(split_path)
            split_small = split_small.to(self.device)
        else:
            split_small = self.uncond_octree(ema = ema, ddim_steps = ddim_steps)
        octree_small = self.split2octree_small(split_small)

        save_dir = os.path.join(self.opt.logs_dir, self.opt.name, suffix)
        self.export_octree(octree_small, depth = self.small_depth, save_dir = os.path.join(save_dir, "octree"), index = save_index)

        batch_size = octree_small.batch_size

        doctree_small = dual_octree.DualOctree(octree_small)
        doctree_small.post_processing_for_docnn()

        doctree_small_num = doctree_small.total_num
        noised_feature = torch.randn((doctree_small_num, self.code_channel), device = self.device)

        feature_time_pairs = self.get_sampling_timesteps(
            batch_size, device=self.device, steps=ddim_steps)

        feature_start = None

        feature_iter = tqdm(feature_time_pairs, desc='feature stage sampling loop time step')

        for time, time_next in feature_iter:

            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            noise_cond = log_snr

            if ema:
                pred_noise = self.ema_df(x_hr = noised_feature, doctree = doctree_small, timesteps = noise_cond)
            else:
                pred_noise = self.df(x_hr = noised_feature, doctree = doctree_small, timesteps = noise_cond)

            alpha, sigma, alpha_next, sigma_next = alpha[0], sigma[0], alpha_next[0], sigma_next[0]

            feature_start = (noised_feature - pred_noise * sigma) / alpha.clamp(min=1e-8)

            noised_feature = feature_start * alpha_next + pred_noise * sigma_next

        samples = noised_feature

        print(samples.max())
        print(samples.min())
        print(samples.mean())
        print(samples.std())

        # decode z
        self.output = self.autoencoder_module.decode_code(samples, doctree_small)
        self.get_sdfs(self.output['neural_mpu'], batch_size, bbox = None)
        self.export_mesh(save_dir = save_dir, index = save_index, clean = clean)


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
        octree_out = create_full_octree(depth = self.input_depth, full_depth = self.full_depth, batch_size = batch_size, device = self.device)
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

        os.makedirs(save_dir, exist_ok=True)

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
                mesh.export(os.path.join(save_dir, f'{index + i}.obj'))


    def get_sdfs(self, neural_mpu, batch_size, bbox):
        # bbox used for marching cubes
        if bbox is not None:
            self.bbmin, self.bbmax = bbox[:3], bbox[3:]
        else:
            sdf_scale = self.solver.sdf_scale
            self.bbmin, self.bbmax = -sdf_scale, sdf_scale    # sdf_scale = 0.9

        self.sdfs = calc_sdf(neural_mpu, batch_size, size = self.solver.resolution, bbmin = self.bbmin, bbmax = self.bbmax)

    def export_mesh(self, save_dir, index = 0, level = 0, clean = False):
        os.makedirs(save_dir, exist_ok=True)
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

        # self.set_requires_grad([self.df.unet_hr], requires_grad=True)

        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
        self.update_EMA()

    def get_current_errors(self):

        ret = OrderedDict([
            ('loss', self.loss.data),
        ])

        if hasattr(self, 'loss_gamma'):
            ret['gamma'] = self.loss_gamma.data

        return ret

    def get_current_visuals(self):

        with torch.no_grad():
            self.img_gen_df = render_sdf_dualoctree(self.renderer, self.sdfs, level=0,
                                                bbmin = self.bbmin, bbmax = self.bbmax,
                                                mesh_scale = self.vq_conf.data.test.point_scale, render_all = True)

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

    def load_ckpt(self, ckpt, df, ema_df, load_opt=False):
        map_fn = lambda storage, loc: storage
        if type(ckpt) == str:
            state_dict = torch.load(ckpt, map_location=map_fn)
        else:
            state_dict = ckpt
            
        df.load_state_dict(state_dict['df'])
        ema_df.load_state_dict(state_dict['ema_df'])
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))

        if load_opt:
            self.start_iter = state_dict['global_step']
            print(colored('[*] training start from: %d' % self.start_iter, 'green'))
            self.optimizer.load_state_dict(state_dict['opt'])
            print(colored('[*] optimizer successfully restored from: %s' % ckpt, 'blue'))
