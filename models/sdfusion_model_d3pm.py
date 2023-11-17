# Reference: diffusion is borrowed from the LDM repo: https://github.com/CompVis/latent-diffusion
# Specifically, functions from: https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/ddpm.py

import os
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

import torchvision.utils as vutils
import torchvision.transforms as transforms

from models.base_model import BaseModel
from models.networks.diffusion_networks.network import DiffusionUNet
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
        return 'SDFusion-Model'

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
        self.depth = self.vq_conf.model.depth
        self.full_depth = self.vq_conf.model.full_depth
        self.voxel_size = 2 ** self.depth

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
        code_channel = 8
        z_sp_dim = 2 ** self.full_depth
        self.z_shape = (code_channel, z_sp_dim, z_sp_dim, z_sp_dim)

        self.ema_df = copy.deepcopy(self.df)
        self.ema_df.to(self.device)
        if opt.isTrain:
            self.ema_rate = opt.ema_rate
            self.ema_updater = EMA(self.ema_rate)
            self.reset_parameters()
            set_requires_grad(self.ema_df, False)

        self.steps = np.arange(self.num_timesteps + 1, dtype=np.float64) / self.num_timesteps
        self.alpha_bar = np.cos((self.steps + 0.008) / 1.008 * np.pi / 2) ** 2
        self.betas = np.minimum(1 - self.alpha_bar[1:] / self.alpha_bar[:-1], 0.999)
        self.betas = torch.tensor(self.betas).to(self.device)

        self.Q = torch.zeros([self.num_timesteps, 2, 2], dtype=torch.float32).to(self.device)
        self.Q[:, 0, 0] = 1 - 0.5 * self.betas
        self.Q[:, 1, 1] = 1 - 0.5 * self.betas
        self.Q[:, 1, 0] = 0.5 * self.betas
        self.Q[:, 0, 1] = 0.5 * self.betas

        self.Q_T = self.Q.permute(0, 2, 1)

        self.Q_ = torch.ones([self.num_timesteps, 2, 2], dtype=torch.float32).to(self.device)
        self.Q_[0] = self.Q[0]
        for i in range(1, self.num_timesteps):
            self.Q_[i] = torch.matmul(self.Q_[i-1], self.Q[i])

        # self.noise_schedule = 'cosine'
        # if self.noise_schedule == "linear":
        #     self.log_snr = beta_linear_log_snr
        # elif self.noise_schedule == "cosine":
        #     self.log_snr = alpha_cosine_log_snr
        # else:
        #     raise ValueError(f'invalid noise schedule {self.noise_schedule}')

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

        self.ddim_steps = 200
        if self.opt.debug == "1":
            # NOTE: for debugging purpose
            self.ddim_steps = 7
        cprint(f'[*] setting ddim_steps={self.ddim_steps}', 'blue')

    def reset_parameters(self):
        self.ema_df.load_state_dict(self.df.state_dict())

    def make_distributed(self, opt):
        self.df = nn.parallel.DistributedDataParallel(
            self.df,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
        )

    ############################ START: init diffusion params ############################

    def set_input(self, input=None):
        self.voxels = input['voxel'].to(self.device)
        self.label = input['label']

    def switch_train(self):
        self.df.train()

    def switch_eval(self):
        self.df.eval()

    # check: ddpm.py, line 891
    def apply_model(self, x_noisy, t, cond, return_ids=False):

        """
            self.model = DiffusionWrapper(unet_config, conditioning_key)
            => which is the denoising UNet.
        """

        if isinstance(cond, dict):
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.df_module.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        key = 'c_crossattn'
        label = [self.label]
        cond = {key: label}

        # eps
        self_cond = None
        if random() < 0.5:
            with torch.no_grad():
                self_cond = self.df(x_noisy, t, self_cond, **cond).detach_()
        out = self.df(x_noisy, t, self_cond, **cond)

        if isinstance(out, tuple) and not return_ids:
            return out[0]
        else:
            return out

    def get_loss(self, pred, target, loss_type='l2', mean=True):
        if loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    # check: ddpm.py, line 871 forward
    # check: p_losses
    # check: q_sample, apply_model
    def p_losses(self, x_start, cond, t, noise=None):

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # predict noise (eps) or x0
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        # l2
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3, 4])
        loss_dict.update({f'loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3, 4))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'loss_total': loss.clone().detach().mean()})

        return x_noisy, target, loss, loss_dict

    def _at(self, a, t, x):
        # print('_at', a.shape, t.shape, x.shape)
        t_broadcast = np.expand_dims(t, tuple(range(1, x.ndim)))
        return a[t_broadcast, x]

    def _at_onehot(self, a, t, x):
        # print('_at_onehot', a.shape, t.shape, x.shape)
        return torch.matmul(x.unsqueeze(1), a[t]).squeeze(1)

    def q_probs(self, x_0, t):
        '''
        x_0: L
        t: L
        '''
        # print('q_probs', x_0.shape, t.shape)
        return self._at(self.Q_, t, x_0)

    def q_sample(self, x_0, t, noise):
        '''
        Compute logits of q(x_t | x_start).
        x_0: L
        noise: Lx2
        '''
        # print('q_sample', x_0.shape, t.shape, x_0.shape)
        logits = torch.log(self.q_probs(x_0, t) + 1e-8)
        noise = torch.clip(noise, min=1e-8, max=1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(logits+gumbel_noise, -1)

    def q_posterior_logits(self, x_start, x_t, t, x_start_logits):
        '''
        Compute logits of q(x_{t-1} | x_t, x_start).
        '''
        # print('q_posterior_logits', x_start.shape, x_t.shape, t.shape, x_start_logits)
        if x_start_logits:
            assert x_start.shape == x_t.shape + (2,), (x_start.shape, x_t.shape)
        else:
            assert x_start.shape == x_t.shape, (x_start.shape, x_t.shape)

        fact1 = self._at(self.Q_T, t, x_t)
        if x_start_logits:
            fact2 = self._at_onehot(self.Q_, t-1, torch.softmax(x_start, axis=-1))
            tzero_logits = x_start
        else:
            fact2 = self._at(self.Q_, t-1, x_start)
            tzero_logits = torch.log(torch.nn.functional.one_hot(x_start, 2) + 1e-8)

        # At t=0 we need the logits of q(x_{-1}|x_0, x_start)
        # where x_{-1} == x_start. This should be equal the log of x_0.
        out = torch.log(fact1 + 1e-8) + torch.log(fact2 + 1e-8)
        t_broadcast = np.expand_dims(t, tuple(range(1, out.ndim)))
        return torch.where(torch.tensor(t_broadcast, device=x_start.device) == 0, tzero_logits, out)

    def p_logits(self, model_out, x, t):
        '''
        Compute logits of p(x_{t-1} | x_t).
        '''
        # print('p_logits', model_out.shape, x.shape, t.shape)
        model_logits = model_out
        pred_x_start_logits = model_logits
        t_broadcast = np.expand_dims(t, tuple(range(1, model_logits.ndim)))
        model_logits = torch.where(torch.tensor(t_broadcast, device=x.device) == 0, pred_x_start_logits, self.q_posterior_logits(pred_x_start_logits, x, t, x_start_logits=True))

        return model_logits, pred_x_start_logits

    def p_sample(self, model_out, x, t, noise):
        '''
        Sample one timestep from the model p(x_{t-1} | x_t).
        '''
        # print('p_sample', model_out.shape, x.shape, t.shape, noise.shape)
        model_logits, pred_x_start_logits = self.p_logits(model_out, x, t)
        assert noise.shape == model_logits.shape, noise.shape
        nonzero_mask = torch.tensor((t != 0), dtype=x.dtype, device=x.device).reshape(x.shape[0], *([1] * (len(x.shape))))
        # For numerical precision clip the noise to a minimum value
        noise = torch.clip(noise, 1e-8, 1.)
        gumbel_noise = -torch.log(-torch.log(noise))
        sample = torch.argmax(model_logits + nonzero_mask * gumbel_noise, axis=-1)

        assert sample.shape == x.shape
        assert pred_x_start_logits.shape == model_logits.shape
        return sample, torch.softmax(pred_x_start_logits, axis=-1)

    def points2octree(self, points):
        points_in = Points(points = points.float())
        points_in.clip(min=-1, max=1)
        octree = Octree(self.depth, self.full_depth)
        octree.build_octree(points_in)
        return octree

    def forward(self):

        # self.voxel  [batch_size, voxel_size, voxel_size, voxel_size]
        batch_size = self.voxels.shape[0]

        mid = (self.voxel_size - 1.) / 2.

        octree_gt_list = []

        for b in range(batch_size):
            voxel = self.voxels[b]
            points = torch.stack(torch.where(voxel == 1), -1)
            points = (points - mid) / mid
            octree_gt = self.points2octree(points)
            octree_gt = octree_gt.to(self.device)
            octree_gt_list.append(copy.deepcopy(octree_gt))

        octree_gt = ocnn.octree.merge_octrees(octree_gt_list)
        octree_gt.construct_all_neigh()

        # self.export_octree(octree_gt, save_dir = 'airplane_input')

        doctree_gt = dual_octree.DualOctree(octree_gt)
        doctree_gt.post_processing_for_docnn()

        output_data = doctree_gt.get_input_feature(feature = 'L')

        self.voxels = self.voxels.long().view(-1)

        L = batch_size * (self.voxel_size ** 3)
        interval = self.voxel_size ** 3
        timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
        # timesteps = torch.tensor([500] * batch_size).to(self.device)
        time = torch.zeros(L, device = self.device).long()
        for i in range(batch_size):
            time[i * interval: (i+1) * interval] = timesteps[i]

        noise = torch.rand((L, 2), device = self.device)
        x_t = self.q_sample(self.voxels.view(-1), time.cpu().numpy(), noise)
        x_t = x_t.view(batch_size, self.voxel_size, self.voxel_size, self.voxel_size)

        octree_in_list = []

        for b in range(batch_size):
            voxel = x_t[b]
            points = torch.stack(torch.where(voxel == 1), -1)
            points = (points - mid) / mid
            octree_in = self.points2octree(points)
            octree_in = octree_in.to(self.device)
            octree_in_list.append(copy.deepcopy(octree_in))

        octree_in = ocnn.octree.merge_octrees(octree_in_list)
        octree_in.construct_all_neigh()

        # self.export_octree(octree_in, save_dir = 'airplane_noised')

        noised_doctree = dual_octree.DualOctree(octree_in)
        noised_doctree.post_processing_for_docnn()
        doctree_in = noised_doctree

        input_data = doctree_in.get_input_feature(feature = 'L')

        out, logits, _ = self.df(input_data, doctree_in, doctree_gt, timesteps, c_crossattn = [self.label])

        self.df_loss = F.mse_loss(out, output_data)

        self.logit_loss = 0.
        for d in logits.keys():
            logitd = logits[d]
            label_gt = octree_gt.nempty_mask(d).long()
            self.logit_loss += F.cross_entropy(logitd, label_gt)

        self.loss = self.df_loss + self.logit_loss

    def get_sampling_timesteps(self, batch, device, steps):
        times = torch.linspace(1., 0., steps + 1, device=device)
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    # check: ddpm.py, log_images(). line 1317~1327
    @torch.no_grad()
    def inference(self, batch_size=16, steps=200, category = 0, ema = False, truncated_index: float = 0.0):

        if ema:
            self.ema_df.eval()
        else:
            self.df.eval()

        shape = (batch_size, *self.z_shape)

        batch, device = shape[0], self.device
        time_pairs = self.get_sampling_timesteps(
            batch, device=device, steps=steps)

        x_start = None
        label = torch.zeros(batch_size).to(self.device)
        label += category
        label = label.long()

        noised_split = torch.randn(shape, device=device)

        _iter = tqdm(time_pairs, desc='sampling loop time step')

        for time, time_next in _iter:
            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            log_snr, log_snr_next = map(
                partial(right_pad_dims_to, noised_split), (log_snr, log_snr_next))

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            noise_cond = self.log_snr(time)

            noised_octree = self.split2octree(noised_split)

            noised_doctree = dual_octree.DualOctree(noised_octree)
            noised_doctree.post_processing_for_docnn()

            input_data = torch.randn((noised_doctree.total_num, 1), device = self.device)

            if ema:
                _, _, doctree_start = self.ema_df(input_data, noised_doctree, doctree_out = None, t = noise_cond, c_crossattn = [label])
            else:
                _, _, doctree_start = self.df(input_data, noised_doctree, doctree_out = None, t = noise_cond, c_crossattn = [label])

            octree_start = doctree_start.octree
            x_start = self.octree2split(octree_start)

            c = -expm1(log_snr - log_snr_next)
            mean = alpha_next * (noised_split * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c
            noise = torch.where(
                rearrange(time_next > truncated_index, 'b -> b 1 1 1 1'),
                torch.randn_like(noised_split),
                torch.zeros_like(noised_split)
            )
            noised_split = mean + torch.sqrt(variance) * noise

        print(noised_split.max())
        print(noised_split.min())

        split = noised_split

        octree_out = self.split2octree(split)

        self.df.train()

    @torch.no_grad()
    def uncond(self, batch_size=16, steps=200, category = 0, ema = True, truncated_index: float = 0.0, save_dir = None, index = 0):

        if ema:
            self.ema_df.eval()
        else:
            self.df.eval()

        z_shape = (batch_size, self.voxel_size, self.voxel_size, self.voxel_size)
        mid = (self.voxel_size - 1.) / 2.

        x_t = torch.randint(low=0, high=2, size = z_shape, device = self.device)

        L = batch_size * (self.voxel_size ** 3)

        for t in tqdm(range(self.num_timesteps-1, -1, -1)):
            x_t = x_t.view(batch_size, self.voxel_size, self.voxel_size, self.voxel_size)

            octree_in_list = []

            for b in range(batch_size):
                voxel = x_t[b]
                points = torch.stack(torch.where(voxel == 1), -1)
                points = (points - mid) / mid
                octree_in = self.points2octree(points)
                octree_in = octree_in.to(self.device)
                octree_in_list.append(copy.deepcopy(octree_in))

            octree_in = ocnn.octree.merge_octrees(octree_in_list)
            octree_in.construct_all_neigh()

            noised_doctree = dual_octree.DualOctree(octree_in)
            noised_doctree.post_processing_for_docnn()
            doctree_in = noised_doctree

            input_data = doctree_in.get_input_feature(feature = 'L')

            timesteps = torch.tensor([t] * batch_size).to(self.device)

            out, logits, doctree_out = self.df(input_data, doctree_in, doctree_gt = None, timesteps = timesteps, c_crossattn = [self.label])
            octree_out = doctree_out.octree

            logits = logits[self.depth]
            x, y, z, batch_id = octree_out.xyzb(depth = self.depth)

            fake_list = []
            for b in range(batch_size):
                fake = torch.zeros((self.voxel_size, self.voxel_size,self.voxel_size, 2)).float().to(self.device)
                fake[:, :, 1] = torch.log(fake[:, :, 1]+1e-8)

                batch_x = x[batch_id == b]
                batch_y = y[batch_id == b]
                batch_z = z[batch_id == b]

                fake[batch_x, batch_y, batch_z :] = logits[batch_id == b]

                fake_list.append(fake.detach().clone())

            x_0 = torch.stack(fake_list, 0)
            x_0_image = torch.argmax(x_0, -1)
            x_0_image = x_0_image.view(batch_size, self.voxe_size, self.voxe_size)

            noise = torch.rand((L, 2), device = self.device)
            x_0 = x_0.view(-1, 2)
            x_t = x_t.view(-1)
            x_t, _ = self.p_sample(x_0, x_t, np.array([t]*L), noise)


        # self.export_octree(octree_out, save_dir, index)

        # doctree_out = dual_octree.DualOctree(octree_out)
        # doctree_out.post_processing_for_docnn()

        return doctree_out

    def get_doctree_data(self, doctree):

        data = torch.zeros([doctree.total_num,2])

        num_full_depth = doctree.lnum[self.full_depth]
        num_full_depth_p1 = doctree.lnum[self.full_depth + 1]

        data[:num_full_depth] = torch.tensor([-1,-1])
        data[num_full_depth : num_full_depth + num_full_depth_p1] = torch.tensor([1,-1])
        data[num_full_depth + num_full_depth_p1 :] = torch.tensor([1,1])
        data = data.to(self.device)
        return data

    def octree2split(self, octree):

        child_full_p1 = octree.children[self.full_depth + 1]
        split_full_p1 = (child_full_p1 >= 0)
        split_full_p1 = split_full_p1.reshape(-1, 8)
        split_full = octree_pad(data = split_full_p1, octree = octree, depth = self.full_depth)
        split_full = octree2voxel(data=split_full, octree=octree, depth = self.full_depth)
        split_full = split_full.permute(0,4,1,2,3).contiguous()

        split_full = split_full.float()
        split_full = 2 * split_full - 1

        return split_full

    def split2octree(self, split):

        split[split > 0] = 1
        split[split < 0] = 0

        batch_size = split.shape[0]
        octree_out = create_full_octree(depth = self.depth, full_depth = self.full_depth, batch_size = batch_size, device = self.device)
        split_sum = torch.sum(split, dim = 1)
        nempty_mask_voxel = (split_sum > 0)
        x, y, z, b = octree_out.xyzb(self.full_depth)
        nempty_mask = nempty_mask_voxel[b,x,y,z]
        label = nempty_mask.long()
        octree_out.octree_split(label, self.full_depth)
        octree_out.octree_grow(self.full_depth + 1)
        octree_out.depth += 1

        x, y, z, b = octree_out.xyzb(depth = self.full_depth, nempty = True)
        nempty_mask_p1 = split[b,:,x,y,z]
        nempty_mask_p1 = nempty_mask_p1.reshape(-1)
        label_p1 = nempty_mask_p1.long()
        octree_out.octree_split(label_p1, self.full_depth + 1)
        octree_out.octree_grow(self.full_depth + 2)
        octree_out.depth += 1

        return octree_out

    def export_octree(self, octree, save_dir = None, index = 0):

        if not os.path.exists(save_dir): os.makedirs(save_dir)

        batch_id = octree.batch_id(depth = self.depth, nempty = True)
        data = torch.ones((len(batch_id), 1), device = self.device)
        data = octree2voxel(data = data, octree = octree, depth = self.depth, nempty = True)
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
            ('diffusion', self.df_loss.data),
            ('logit', self.logit_loss.data),
            ('total', self.loss.data),
        ])

        if hasattr(self, 'loss_gamma'):
            ret['gamma'] = self.loss_gamma.data

        return ret

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
