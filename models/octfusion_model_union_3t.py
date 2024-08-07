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
from models import octfusion_model_union
TRUNCATED_TIME = 0.7


class OctFusionModel(octfusion_model_union.OctFusionModel):
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
        self.large_depth = self.vq_conf.model.depth_stop
        self.small_depth = 6
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
                self.df.unet_hr,
                self.df.unet_feature,
            ], False)
        elif self.stage_flag == "hr":
            self.set_requires_grad([
                self.df.unet_lr,
                self.df.unet_feature,
            ], False)
        elif self.stage_flag == "feature":
            self.set_requires_grad([
                self.df.unet_lr,
                self.df.unet_hr,
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
            elif self.stage_flag == "feature":
                load_options = ["unet_lr", "unet_hr", "unet_feature"]

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

    def forward(self):

        self.df.train()

        c = None

        
        self.df_lr_loss = torch.tensor(0., device=self.device)
        self.df_hr_loss = torch.tensor(0., device=self.device)
        self.df_feature_loss = torch.tensor(0., device=self.device)

        if self.stage_flag == "lr":
            split_small = octree2split_small(self.octree_in, self.full_depth)
            self.df_lr_loss = self.forward_lr(split_small)
            
        elif self.stage_flag == "hr":
            split_large = octree2split_large(self.octree_in, self.small_depth)
            nnum_large = split_large.shape[0]
            split_large_padded = torch.zeros((self.doctree_in.graph[self.small_depth]['keyd'].shape[0], split_large.shape[1]), device=self.device)
            split_large_padded[-nnum_large:, :] = split_large
            
            self.df_hr_loss = self.forward_hr(split_large_padded, self.small_depth, "hr", self.df.unet_lr)
        elif self.stage_flag == "feature":
            with torch.no_grad():
                self.input_data, self.doctree_in = self.autoencoder_module.extract_code(self.octree_in)
            self.df_feature_loss = self.forward_hr(self.input_data, self.large_depth, "feature", self.df.unet_hr)

        self.loss = self.df_lr_loss + self.df_hr_loss + self.df_feature_loss

    def get_sampling_timesteps(self, batch, device, steps):
        times = torch.linspace(1., 0., steps + 1, device=device)
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

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
        
        octree_small = split2octree_small(split_small, self.small_depth, self.full_depth)
        self.export_octree(octree_small, depth = self.small_depth, save_dir = os.path.join(save_dir, "octree"), index = save_index)
        for i in range(batch_size):
            save_path = os.path.join(save_dir, "splits_small", f"{save_index}.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(split_small[i].unsqueeze(0), save_path)
        
        if self.stage_flag == "lr":
            return
        
        doctree_small = dual_octree.DualOctree(octree_small)
        doctree_small.post_processing_for_docnn()
            
        split_large = self.sample_hr(doctree_lr = doctree_small, label = label, ema = ema, ddim_steps = ddim_steps)
        octree_large = split2octree_small(split_large, self.input_depth, self.full_depth)
        self.export_octree(octree_large, depth = self.small_depth, save_dir = os.path.join(save_dir, "octree"), index = save_index)
        for i in range(batch_size):
            save_path = os.path.join(save_dir, "splits_large", f"{save_index}.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(split_small[i].unsqueeze(0), save_path)
        
        if self.stage_flag == "hr":
            return
        
        doctree_large = dual_octree.DualOctree(octree_large)
        doctree_large.post_processing_for_docnn()
        
        samples = self.sample_hr(doctree_lr = doctree_large, label = label, ema = ema, ddim_steps = ddim_steps)

        print(samples.max())
        print(samples.min())
        print(samples.mean())
        print(samples.std())

        # decode z
        self.output = self.autoencoder_module.decode_code(samples, doctree_small)
        self.get_sdfs(self.output['neural_mpu'], batch_size, bbox = None)
        self.export_mesh(save_dir = save_dir, index = save_index, clean = clean)

    def save(self, label, global_iter):

        state_dict = {
            'df_unet_lr': self.df_module.unet_lr.state_dict(),
            'df_unet_hr': self.df_module.unet_hr.state_dict(),
            'ema_df_unet_lr': self.ema_df.unet_lr.state_dict(),
            'ema_df_unet_hr': self.ema_df.unet_hr.state_dict(),
            'df_unet_feature': self.df_module.unet_feature.state_dict(),
            'ema_df_unet_feature': self.ema_df.unet_feature.state_dict(),
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
        if "unet_feature" in load_options and "df_unet_feature" in state_dict:
            df.unet_feature.load_state_dict(state_dict['df_unet_feature'])
            ema_df.unet_feature.load_state_dict(state_dict['ema_df_unet_feature'])
            print(colored('[*] weight successfully load unet_feature from: %s' % ckpt, 'blue'))

        if "opt" in load_options and "opt" in state_dict:
            self.start_iter = state_dict['global_step']
            print(colored('[*] training start from: %d' % self.start_iter, 'green'))
            self.optimizer.load_state_dict(state_dict['opt'])
            print(colored('[*] optimizer successfully restored from: %s' % ckpt, 'blue'))
