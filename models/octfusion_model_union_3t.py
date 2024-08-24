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
from utils.distributed import reduce_loss_dict, get_rank, get_world_size

# rendering
from utils.util_dualoctree import calc_sdf, octree2split_small, octree2split_large, split2octree_small, split2octree_large
from utils.util import TorchRecoder, seed_everything, category_5_to_label
from models import octfusion_model_union
TRUNCATED_TIME = 0.7


class OctFusionModel(octfusion_model_union.OctFusionModel):
    def name(self):
        return 'SDFusion-Model-Union-Two-Times'

    def initialize(self, opt):
        octfusion_model_union.OctFusionModel.network_initialize(self, opt)
        self.optimizer_initialize(opt)
        
        
    def optimizer_initialize(self, opt):
        
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
        
        if opt.pretrain_ckpt is not None:
            if self.stage_flag == "hr":
                load_options = ["unet_lr"]
            elif self.stage_flag == "feature":
                load_options = ["unet_lr", "unet_hr"]
            self.load_ckpt(opt.pretrain_ckpt, self.df, self.ema_df, load_options=load_options)

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
            batch_id = torch.arange(0, self.batch_size, device=self.device).long()
            self.df_lr_loss = self.calc_loss(self.split_small, None, batch_id, "lr", None, self.df_type[0])
            
        elif self.stage_flag == "hr":
            self.doctree_in = dual_octree.DualOctree(self.octree_in)
            self.doctree_in.post_processing_for_docnn()
            
            split_large = octree2split_large(self.octree_in, self.small_depth)
            nnum_large = split_large.shape[0]
            split_large_padded = torch.zeros((self.doctree_in.graph[self.small_depth]['keyd'].shape[0], split_large.shape[1]), device=self.device)
            split_large_padded[-nnum_large:, :] = split_large
            batch_id = self.doctree_in.batch_id(self.small_depth)
            
            # self.df_hr_loss = self.forward_hr(split_large_padded, self.small_depth, "hr", self.df.unet_lr)
            self.df_hr_loss = self.calc_loss(split_large_padded, self.doctree_in, batch_id, "hr", unet_lr=self.df_module.unet_lr, df_type=self.df_type[1])
        elif self.stage_flag == "feature":
            with torch.no_grad():
                self.input_data, self.doctree_in = self.autoencoder_module.extract_code(self.octree_in)
            # self.df_feature_loss = self.forward_hr(self.input_data, self.large_depth, "feature", self.df_module.unet_hr)
            self.df_feature_loss = self.calc_loss(self.input_data, self.doctree_in, self.doctree_in.batch_id(self.large_depth), "feature", unet_lr=self.df_module.unet_hr, df_type=self.df_type[2])

        self.loss = self.df_lr_loss + self.df_hr_loss + self.df_feature_loss

    def sample(self, split_small = None, category = 'airplane', prefix = 'results', ema = False, ddim_steps=200, clean = False, save_index = 0):

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
        if split_small == None:
            # seed_everything(self.opt.seed + save_index)
            split_small = self.sample_loop(doctree_lr=None, ema=ema, shape=(batch_size, *self.z_shape), ddim_steps=ddim_steps, label=label, unet_type="lr", unet_lr=None, df_type=self.df_type[0], truncated_index=TRUNCATED_TIME)
        
        octree_small = split2octree_small(split_small, self.octree_depth, self.full_depth)
        self.export_octree(octree_small, depth = self.small_depth, save_dir = os.path.join(save_dir, "octree"), index = save_index)
        # for i in range(batch_size):
        #     save_path = os.path.join(save_dir, "splits_small", f"{save_index}.pth")
        #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #     torch.save(split_small[i].unsqueeze(0), save_path)
        
        if self.stage_flag == "lr":
            return
        
        doctree_small = dual_octree.DualOctree(octree_small)
        doctree_small.post_processing_for_docnn()
        doctree_small_num = doctree_small.total_num
        
        # seed_everything(self.opt.seed)
        split_large = self.sample_loop(doctree_lr=doctree_small, shape=(doctree_small_num, self.split_channel), ema=ema, ddim_steps=ddim_steps, label=label, unet_type="hr", unet_lr=self.ema_df.unet_lr, df_type=self.df_type[1])
        
        split_large = split_large[-octree_small.nnum[self.small_depth]: ]
        
        octree_large = split2octree_large(octree_small, split_large, self.small_depth)
        self.export_octree(octree_large, depth = self.large_depth, save_dir = os.path.join(save_dir, "octree"), index = save_index)
        # for i in range(batch_size):
        #     save_path = os.path.join(save_dir, "splits_large", f"{save_index}.pth")
        #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #     torch.save(split_small[i].unsqueeze(0), save_path)
        if self.stage_flag == "hr":
            return
        
        doctree_large = dual_octree.DualOctree(octree_large)
        doctree_large.post_processing_for_docnn()
        doctree_large_num = doctree_large.total_num
        
        # seed_everything(self.opt.seed)
        samples = self.sample_loop(doctree_lr=doctree_large, shape=(doctree_large_num, self.code_channel), ema=ema, ddim_steps=ddim_steps, label=label, unet_type="feature", unet_lr=self.ema_df.unet_hr, df_type=self.df_type[2])

        print(samples.max())
        print(samples.min())
        print(samples.mean())
        print(samples.std())

        # decode z
        self.output = self.autoencoder_module.decode_code(samples, doctree_large)
        self.get_sdfs(self.output['neural_mpu'], batch_size, bbox = None)
        self.export_mesh(save_dir = save_dir, index = save_index, clean = clean)

    def save(self, label, global_iter):

        state_dict = {
            'df_unet_lr': self.df_module.unet_lr.state_dict(),
            'ema_df_unet_lr': self.ema_df.unet_lr.state_dict(),
            'opt': self.optimizer.state_dict(),
            'global_step': global_iter,
        }
        if self.stage_flag == "hr" or self.stage_flag == "feature":
            state_dict['df_unet_hr'] = self.df_module.unet_hr.state_dict()
            state_dict['ema_df_unet_hr'] = self.ema_df.unet_hr.state_dict()
        if self.stage_flag == "feature":
            state_dict['df_unet_feature'] = self.df_module.unet_feature.state_dict()
            state_dict['ema_df_unet_feature'] = self.ema_df.unet_feature.state_dict()

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
