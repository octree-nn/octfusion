# Reference: diffusion is borrowed from the LDM repo: https://github.com/CompVis/latent-diffusion
# Specifically, functions from: https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/ddpm.py

import os
import sys
from collections import OrderedDict
from functools import partial
import copy
import time
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


# distributed
from utils.distributed import reduce_loss_dict, get_rank

# rendering
from utils.util_dualoctree import calc_sdf, octree2split_small, octree2split_large, split2octree_small, split2octree_large
from utils.util import TorchRecoder, seed_everything, category_5_to_label

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
        self.load_split_small = self.vq_conf.data.train.load_split_small

        # init diffusion networks
        df_model_params = df_conf.model.params
        unet_params = df_conf.unet.params
        self.conditioning_key = df_model_params.conditioning_key
        self.num_timesteps = df_model_params.timesteps
        self.enable_label = "num_classes" in df_conf.unet.params

        self.df = UNet3DModel(**unet_params)
        self.df.to(self.device)
        self.stage_flag = opt.stage_flag

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

        ######## END: Define Networks ########

        if opt.pretrain_ckpt is not None:
            self.load_ckpt(opt.pretrain_ckpt, self.df, self.ema_df, load_options=["unet_lr"])
        
        if self.stage_flag == "lr":
            self.set_requires_grad([
                self.df.unet_hr
            ], False)
        elif self.stage_flag == "hr":
            self.set_requires_grad([
                self.df.unet_lr
            ], False)
        
        if self.isTrain:

            # initialize optimizers
            self.optimizer = optim.AdamW([p for p in self.df.parameters() if p.requires_grad == True], lr=opt.lr)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1000, 0.9)

            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]

            self.print_networks(verbose=False)

        if opt.ckpt is None and os.path.exists(os.path.join(opt.logs_dir, opt.name, "ckpt/df_steps-latest.pth")):
            opt.ckpt = os.path.join(opt.logs_dir, opt.name, "ckpt/df_steps-latest.pth")
        
        if opt.ckpt is not None:
            if self.stage_flag == "lr":
                load_options = ["unet_lr"]
            elif self.stage_flag == "hr":
                load_options = ["unet_lr", "unet_hr"]
            if self.isTrain:
                load_options.append("opt")
            self.load_ckpt(opt.ckpt, self.df, self.ema_df, load_options)
                
        trainable_params_num = 0
        for m in [self.df]:
            trainable_params_num += sum([p.numel() for p in m.parameters() if p.requires_grad == True])
        print("Trainable_params: ", trainable_params_num)

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
            batch['split_small'] = octree2split_small(batch['octree_in'], self.full_depth)

        batch['label'] = batch['label'].cuda()
        if self.load_octree:
            batch['octree_in'] = batch['octree_in'].cuda()
            batch['split_small'] = octree2split_small(batch['octree_in'], self.full_depth)
            # batch['split_large'] = self.octree2split_large(batch['octree_in'])
        elif self.load_split_small:
            batch['split_small'] = batch['split_small'].cuda()
            batch['octree_in'] = split2octree_small(batch['split_small'], self.input_depth, self.full_depth)

    def set_input(self, input=None):
        self.batch_to_cuda(input)
        self.split_small = input['split_small']
        # self.split_large = input['split_large']
        self.octree_in = input['octree_in']
        self.batch_size = self.octree_in.batch_size

        if self.enable_label:
            self.label = input['label']
        else:
            self.label = None

    def switch_train(self):
        self.df.train()

    def switch_eval(self):
        self.df.eval()


    def forward_lr(self, split_small):
        times = torch.zeros(
            (self.batch_size,), device=self.device).float().uniform_(0, 1)
        
        noise = torch.randn_like(split_small)

        noise_level = self.log_snr(times)
        padded_noise_level = right_pad_dims_to(split_small, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
        noised_split_small = alpha * split_small + sigma * noise

        x_self_cond = None
        if random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.df(unet_type="lr", x = noised_split_small, timesteps = noise_level, label = self.label)

        pred = self.df(unet_type="lr", x = noised_split_small, timesteps = noise_level, x_self_cond = x_self_cond, label = self.label)

        return F.mse_loss(pred, split_small)
    
    def forward_hr(self, input_data, input_depth, unet_type, unet_lr):
        times = torch.zeros((self.batch_size,), device = self.device).float().uniform_(0,1)
        noise_level = self.log_snr(times)

        alpha, sigma = log_snr_to_alpha_sigma(noise_level)

        noised_feature = input_data.clone()

        batch_id = self.doctree_in.batch_id(input_depth)
        noise = torch.randn_like(noised_feature, device = self.device)

        batch_alpha = alpha[batch_id].unsqueeze(1)
        batch_sigma = sigma[batch_id].unsqueeze(1)
        noised_feature = noised_feature * batch_alpha + noise * batch_sigma

        output = self.df(unet_type=unet_type, x = noised_feature, doctree = self.doctree_in, timesteps = noise_level, unet_lr = unet_lr, label = self.label)

        return F.mse_loss(output, noise)

    def forward(self):

        self.df.train()

        c = None

        with torch.no_grad():
            self.input_data, self.doctree_in = self.autoencoder_module.extract_code(self.octree_in)

        self.df_hr_loss = torch.tensor(0., device=self.device)
        self.df_lr_loss = torch.tensor(0., device=self.device)

        if self.stage_flag == "lr":
            split_small = octree2split_small(self.doctree_in.octree, self.full_depth)
            self.df_lr_loss = self.forward_lr(split_small)
            
        elif self.stage_flag == "hr":
            self.df_hr_loss = self.forward_hr(self.input_data, self.small_depth, "hr", self.df_module.unet_lr)

        self.loss = self.df_lr_loss + self.df_hr_loss

    def get_sampling_timesteps(self, batch, device, steps):
        times = torch.linspace(1., 0., steps + 1, device=device)
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    @torch.no_grad()
    def sample_lr(self, ema=False, ddim_steps=200, label = None, truncated_index = 0.0):
        batch_size = self.vq_conf.data.test.batch_size

        seed_everything(int(time.time()))
        small_time_pairs = self.get_sampling_timesteps(
            batch_size, device=self.device, steps=ddim_steps)

        shape = (batch_size, *self.z_shape)
        noised_split_small = torch.randn(shape, device = self.device)

        x_start_small = None

        small_iter = tqdm(small_time_pairs, desc='small sampling loop time step')

        for t, t_next in small_iter:

            log_snr = self.log_snr(t)
            log_snr_next = self.log_snr(t_next)
            log_snr, log_snr_next = map(
                partial(right_pad_dims_to, noised_split_small), (log_snr, log_snr_next))

            alpha, _ = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            noise_cond = self.log_snr(t)

            if ema:
                x_start_small = self.ema_df(unet_type="lr", x = noised_split_small, timesteps = noise_cond, x_self_cond = x_start_small, label = label)
            else:
                x_start_small = self.df(unet_type="lr", x = noised_split_small, timesteps = noise_cond, x_self_cond = x_start_small, label = label)

            if t[0] < TRUNCATED_TIME:
                x_start_small.sign_()

            c = -expm1(log_snr - log_snr_next)
            mean = alpha_next * (noised_split_small * (1 - c) / alpha + c * x_start_small)
            variance = (sigma_next ** 2) * c
            noise = torch.where(
                rearrange(t_next > truncated_index, 'b -> b 1 1 1 1'),
                torch.randn_like(noised_split_small),
                torch.zeros_like(noised_split_small)
            )
            noised_split_small = mean + torch.sqrt(variance) * noise
        
        return noised_split_small
        

    @torch.no_grad()
    def sample_hr(self, doctree_lr = None, label = None, ema = False, ddim_steps=200):
        batch_size = self.vq_conf.data.test.batch_size
        
        seed_everything(self.opt.seed)

        doctree_small_num = doctree_lr.total_num
        noised_feature = torch.randn((doctree_small_num, self.code_channel), device = self.device)

        feature_time_pairs = self.get_sampling_timesteps(
            batch_size, device=self.device, steps=ddim_steps)

        feature_start = None

        feature_iter = tqdm(feature_time_pairs, desc='feature stage sampling loop time step')

        for t, t_next in feature_iter:

            log_snr = self.log_snr(t)
            log_snr_next = self.log_snr(t_next)

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            noise_cond = log_snr

            if ema:
                pred_noise = self.ema_df(unet_type="hr", x = noised_feature, doctree = doctree_lr, timesteps = noise_cond, unet_lr = self.ema_df.unet_lr, label=label)
            else:
                pred_noise = self.df(unet_type="hr", x = noised_feature, doctree = doctree_lr, timesteps = noise_cond, unet_lr = self.df.unet_lr, label=label)

            alpha, sigma, alpha_next, sigma_next = alpha[0], sigma[0], alpha_next[0], sigma_next[0]

            feature_start = (noised_feature - pred_noise * sigma) / alpha.clamp(min=1e-8)

            noised_feature = feature_start * alpha_next + pred_noise * sigma_next

        return noised_feature
    
    @torch.no_grad()
    def sample(self, split_path = None, category = 'airplane', prefix = 'results', ema = False, ddim_steps=200, clean = False, save_index = 0):

        if ema:
            self.ema_df.eval()
        else:
            self.df.eval()
        
        if self.enable_label:
            label = torch.ones(batch_size).to(self.device) * category_5_to_label[category]
            label = label.long()
        else:
            label = None
            
        save_dir = os.path.join(self.opt.logs_dir, self.opt.name, f"{prefix}_{category}")
        batch_size = self.vq_conf.data.test.batch_size
        if split_path != None:
            split_small = torch.load(split_path)
            split_small = split_small.to(self.device)
        else:
            split_small = self.sample_lr(ema=ema, ddim_steps=ddim_steps, label=label)
        
        octree_small = split2octree_small(split_small, self.input_depth, self.full_depth)
        self.export_octree(octree_small, depth = self.small_depth, save_dir = os.path.join(save_dir, "octree"), index = save_index)
        for i in range(batch_size):
            torch.save(split_small[i].unsqueeze(0), os.path.join(save_dir, f"{save_index}.pth"))
        
        if self.stage_flag == "lr":
            return
        
        doctree_small = dual_octree.DualOctree(octree_small)
        doctree_small.post_processing_for_docnn()
            
        samples = self.sample_hr(doctree_lr = doctree_small, label = label, ema = ema, ddim_steps = ddim_steps)

        print(samples.max())
        print(samples.min())
        print(samples.mean())
        print(samples.std())

        # decode z
        self.output = self.autoencoder_module.decode_code(samples, doctree_small)
        self.get_sdfs(self.output['neural_mpu'], batch_size, bbox = None)
        self.export_mesh(save_dir = save_dir, index = save_index, clean = clean)


    def export_octree(self, octree, depth, save_dir = None, index = 0):
        try:
            os.makedirs(save_dir, exist_ok=True)
        except FileExistsError:
            pass

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

    def save(self, label, global_iter):

        state_dict = {
            'df_unet_lr': self.df_module.unet_lr.state_dict(),
            'df_unet_hr': self.df_module.unet_hr.state_dict(),
            'ema_df_unet_lr': self.ema_df.unet_lr.state_dict(),
            'ema_df_unet_hr': self.ema_df.unet_hr.state_dict(),
            'opt': self.optimizer.state_dict(),
            'global_step': global_iter,
        }

        save_filename = 'df_%s.pth' % (label)
        save_path = os.path.join(self.opt.ckpt_dir, save_filename)

        ckpts = os.listdir(self.opt.ckpt_dir)
        ckpts = [ck for ck in ckpts if ck!='df_steps-latest.pth']
        ckpts.sort(key=lambda x: int(x[9:-4]))
        if len(ckpts) > self.opt.ckpt_num:
            for ckpt in ckpts[:-self.opt.ckpt_num]:
                os.remove(os.path.join(self.opt.ckpt_dir, ckpt))

        torch.save(state_dict, save_path)

    def load_ckpt(self, ckpt, df, ema_df, load_options=[]):
        map_fn = lambda storage, loc: storage
        if type(ckpt) == str:
            state_dict = torch.load(ckpt, map_location=map_fn)
        else:
            state_dict = ckpt
        
        if "unet_lr" in load_options and "df_unet_lr" in state_dict:
            df.unet_lr.load_state_dict(state_dict['df_unet_lr'])
            ema_df.unet_lr.load_state_dict(state_dict['ema_df_unet_lr'])
            print(colored('[*] weight successfully load unet_lr from: %s' % ckpt, 'blue'))
        if "unet_hr" in load_options and "df_unet_hr" in state_dict:
            df.unet_hr.load_state_dict(state_dict['df_unet_hr'])
            ema_df.unet_hr.load_state_dict(state_dict['ema_df_unet_hr'])
            print(colored('[*] weight successfully load unet_hr from: %s' % ckpt, 'blue'))

        if "opt" in load_options and "opt" in state_dict:
            self.start_iter = state_dict['global_step']
            print(colored('[*] training start from: %d' % self.start_iter, 'green'))
            self.optimizer.load_state_dict(state_dict['opt'])
            print(colored('[*] optimizer successfully restored from: %s' % ckpt, 'blue'))
