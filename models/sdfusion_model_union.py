# Reference: diffusion is borrowed from the LDM repo: https://github.com/CompVis/latent-diffusion
# Specifically, functions from: https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/ddpm.py

import os
import json
from collections import OrderedDict
from functools import partial

import numpy as np
# import marching_cubes as mcubes
from omegaconf import OmegaConf
from termcolor import colored, cprint
from einops import rearrange, repeat
from tqdm import tqdm
from random import random
import copy
import ocnn
from ocnn.nn import octree2voxel
from ocnn.octree import key2xyz, xyz2key

import torch
import torch.nn.functional as F
from torch.special import expm1
from torch import nn, optim

import torchvision.utils as vutils
import torchvision.transforms as transforms
import kornia
import timm
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras, FoVOrthographicCameras

from models.base_model import BaseModel
from models.networks.diffusion_networks.network_union import DiffusionUNet
from models.networks.bert_networks.network import BERTTextEncoder
from models.model_utils import load_vqvae, load_dualoctree

# ldm util
from models.networks.diffusion_networks.ldm_diffusion_util import *
from models.networks.diffusion_networks.samplers.ddim import DDIMSampler
from models.networks.diffusion_networks.condition import ConditionEncoder
# distributed
from utils.distributed import reduce_loss_dict

# rendering
from utils.util_3d import init_mesh_renderer, render_sdf,render_sdf_dualoctree
from utils.util_dualoctree import calc_sdf, calc_sdf_with_color

# Octree
from ocnn.nn import octree_pad
from models.networks.dualoctree_networks import dual_octree
from models.networks.dualoctree_networks.modules_v1 import doctree_align, doctree_align_reverse

class SDFusionModel(BaseModel):
	def name(self):
		return 'SDFusion-Model'

	def initialize(self, opt):
		BaseModel.initialize(self, opt)
		self.isTrain = opt.isTrain
		self.model_name = self.name()
		self.device = opt.device


		######## START: Define Networks ########
		assert opt.df_cfg is not None
		assert opt.vq_cfg is not None

		# init df
		df_conf = OmegaConf.load(opt.df_cfg)
		vq_conf = OmegaConf.load(opt.vq_cfg)
		unet_params = df_conf.unet.params
		self.df_conf = df_conf
		self.vq_conf = vq_conf
		self.solver = self.vq_conf.SOLVER

		# record z_shape
		self.doctree_depth = unet_params.depth
		self.full_depth = unet_params.full_depth
		self.code_channel = vq_conf.MODEL.embed_dim
		self.split_channel = 8
		self.log_snr = alpha_cosine_log_snr # beta_linear_log_snr
		self.split_threshold = 0.4
		# init diffusion networks
		conditioning_key = df_conf.model.params.conditioning_key
		self.df = DiffusionUNet(unet_params, vq_conf=vq_conf, conditioning_key = conditioning_key)
		self.df.to(self.device)

		self.ema_df = copy.deepcopy(self.df)
		self.ema_df.to(self.device)
		if opt.isTrain:
			self.ema_rate = opt.ema_rate
			self.ema_updater = EMA(self.ema_rate)
			self.reset_parameters()
			set_requires_grad(self.ema_df, False)

		# init vqvae
		self.autoencoder = load_dualoctree(conf = vq_conf, ckpt = opt.vq_ckpt, opt = opt)

		self.trainable_models = [self.df]

		# init cond model
		self.image_cond = df_conf.model.params.image_cond
		self.text_cond = df_conf.model.params.text_cond
		if conditioning_key != "None":
			self.cond_encoder = ConditionEncoder(df_conf, self.device)
			self.cond_encoder.to(self.device)
			# self.trainable_models.append(self.cond_encoder)
		######## END: Define Networks ########

		# param list
		trainable_params = []

		for m in self.trainable_models:
			trainable_params += [p for p in m.parameters() if p.requires_grad == True]

		######## END: Define Networks ########

		if self.isTrain:

			# initialize optimizers
			self.optimizer = optim.AdamW(trainable_params, lr=opt.lr)
			self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1, 0.9)
			# self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 100, 0.9)

			self.optimizers = [self.optimizer]
			self.schedulers = [self.scheduler]

			self.print_networks(verbose=False)

		if opt.ckpt is None and os.path.exists(os.path.join(opt.logs_dir, opt.name, "ckpt/df_steps-latest.pth")):
			opt.ckpt = os.path.join(opt.logs_dir, opt.name, "ckpt/df_steps-latest.pth")
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
			self.autoencoder_module = self.autoencoder
		else:
			self.df_module = self.df
			self.autoencoder_module = self.autoencoder

		self.ddim_steps = 50
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
			find_unused_parameters=True,
		)
		for p in self.autoencoder.parameters():
			p.requires_grad = False
			
		if self.image_cond or self.text_cond:
			for p in self.cond_encoder.parameters():
				p.requires_grad = False
		


	def batch_to_cuda(self, batch):
		batch['octree_in'] = batch['octree_in'].cuda()
		# batch['split_small'] = self.octree2split_small(batch['octree_in'], batch['octree_in'].full_depth)
		# batch['split_large'] = self.octree2split_large(batch['octree_in'], batch['octree_in'].full_depth + 2)

		doctree = dual_octree.DualOctree(batch['octree_in'])
		doctree.post_processing_for_docnn()
		batch['doctree_in'] = doctree

		input_data = doctree.get_input_feature()
		batch['input_data'] = input_data

	def set_input(self, input=None):
		self.batch_to_cuda(input)
		# self.split_small = input['split_small']
		# self.split_large = input['split_large']
		self.octree_in = input['octree_in']
		self.doctree_in = input['doctree_in']
		self.input_data = input['input_data']
		self.label = input['label']
		self.batch_size = self.octree_in.batch_size


		if self.image_cond:
			self.img = input['image']
			self.tocuda(["img"])
			self.pm = []
			for elev, azim in input['euler']:
				R, T = look_at_view_transform(dist=3.0, elev=elev, azim=azim, device=self.device)
				scale = 1.2
				cameras = FoVOrthographicCameras(
					R=R, T = T, device=self.device,
					max_x=scale, min_x=-scale, max_y=scale, min_y=-scale,
				)
				p_matrix = cameras.get_projection_transform().get_matrix()
				self.pm.append(p_matrix)
			self.pm = torch.stack(self.pm, dim=0)
		else:
			self.img = None

		if self.text_cond:
			self.txt = input['text']
		else:
			self.txt = None

	def switch_train(self):
		for model in self.trainable_models:
			model.train()

	def switch_eval(self):
		for model in self.trainable_models:
			model.eval()
		self.autoencoder.eval()

	def get_condition(self):
		if self.df_module.conditioning_key != "None":
			cond = dict()
			self.cond_encoder.eval()
			with torch.no_grad():
				c_mm = self.cond_encoder(self.img, self.txt)
			cond['c_crossattn'] = c_mm
			if self.image_cond:
				cond['projection_matrix'] = self.pm
		else:
			cond = {}
		return cond

	def octree_add_noise(self, octree_gt, split_threshold=0.5, times=None):

		noised_octree = create_full_octree(depth=self.doctree_depth, full_depth=self.full_depth, batch_size=self.batch_size, device=self.device)
		noised_octree_split_dict = {}
		noise_level_dict = {}

		for d in range(self.full_depth, self.doctree_depth + 1):
			gt_child_d = octree_gt.children[d]
			gt_split_d = (gt_child_d >= 0).float()
			gt_split_d = 2 * gt_split_d - 1.0
				
			noise_level_d = self.log_snr(times)
			noise_level_dict[d] = noise_level_d
			padded_noise_level_d = right_pad_dims_to(gt_split_d, noise_level_d)
			alpha_d, sigma_d = log_snr_to_alpha_sigma(padded_noise_level_d)
			
			noise_level_dict[d] = noise_level_d

			# find keys
			gt_key_d = copy.deepcopy(octree_gt.key(d, nempty=False))
			out_key_d = copy.deepcopy(noised_octree.key(d, nempty=False))  
			# gt_to_out_split_d = doctree_align_reverse(gt_split_d, out_key_d, gt_key_d)
			gt_to_out_split_d = doctree_align(gt_split_d, gt_key_d, out_key_d)  # x0

			batch_id_out = noised_octree.batch_id(d)
			batch_alpha_d = alpha_d[batch_id_out]
			batch_sigma_d = sigma_d[batch_id_out]
			
			if True:
				noised_split_d = torch.randn_like(gt_to_out_split_d)
				noised_split_d = batch_alpha_d * gt_to_out_split_d + batch_sigma_d * noised_split_d
			else:
				noised_split_d = torch.rand_like(gt_to_out_split_d) > split_threshold
				noised_split_d = noised_split_d.float() * 2.0 - 1.0
				
				gt_or_noise = torch.rand_like(gt_to_out_split_d) < (batch_alpha_d / (batch_alpha_d + batch_sigma_d))
				noised_split_d[gt_or_noise] = gt_to_out_split_d[gt_or_noise]

			noised_octree_split_dict[d] = noised_split_d
			# noise to octree
			discrete_split_d = (noised_split_d > split_threshold).long()
			noised_octree.octree_split(discrete_split_d, d)
			if d < self.doctree_depth:
				noised_octree.octree_grow(d+1)
		
		noised_doctree = dual_octree.DualOctree(noised_octree)
		noised_doctree.post_processing_for_docnn()

		noised_doctree_split_dict = self.get_noised_doctree_splits(noised_doctree, noised_octree_split_dict)

		return noised_doctree, noised_doctree_split_dict

	def get_noised_doctree_splits(self, doctree, noised_octree_splits):	
		def doctree_split_upsample(x, x1, doctree, d):
			numd = doctree.nnum[d]
			leaf_mask = doctree.node_child(d) < 0
			# upsample nodes at layer (depth-1)
			outd = x[-numd:]

			# construct the final output
			out = torch.cat([x[:-numd], outd[leaf_mask], x1], dim=0)
			return out

		noised_doctree_splits = {}
		noised_doctree_splits[self.full_depth] = noised_octree_splits[self.full_depth]
		for d in range(self.full_depth, self.doctree_depth):
			noised_doctree_splits[d + 1] = doctree_split_upsample(noised_doctree_splits[d], noised_octree_splits[d+1], doctree, d)
		return noised_doctree_splits

	def latent_add_noise(self, doctree_gt, doctree_out, latent_list, times=None):
		batch_size = doctree_gt.batch_size


		noise_level_list = []

		gt_key_depth = copy.deepcopy(doctree_gt.graph[self.doctree_depth]['keyd'])
		out_key_depth = copy.deepcopy(doctree_out.graph[self.doctree_depth]['keyd'])

		noised_latent_list = []
		for latent_k in latent_list:
			if times == None:
				times = torch.zeros((batch_size,), device=self.device).float().uniform_(0,1)
			noise_level_k = self.log_snr(times)
			noise_level_list.append(noise_level_k)
			padded_noise_level_k = right_pad_dims_to(latent_k, noise_level_k)
			alpha_k, sigma_k = log_snr_to_alpha_sigma(padded_noise_level_k)

			batch_id_out = doctree_out.batch_id(self.doctree_depth)
			batch_alpha_k = alpha_k[batch_id_out]
			batch_sigma_k = sigma_k[batch_id_out]

			gt_to_out_latent_k = doctree_align(latent_k, gt_key_depth, out_key_depth)
			noise_k = torch.randn_like(gt_to_out_latent_k)
			noised_latent_k = batch_alpha_k * gt_to_out_latent_k + batch_sigma_k * noise_k
			noised_latent_list.append(noised_latent_k)
			
		return noised_latent_list
	
	def visualize_noised_octree(self):
		for thres in [0.5]:
			for ti in [0.1, 0.2, 0.5, 0.8]:
				times = torch.ones((self.batch_size,), device=self.device).float() * ti
				noised_doctree, noised_doctree_split_dict = self.octree_add_noise(self.doctree_in.octree, split_threshold=thres, times=times)
				for i in range(self.full_depth + 1, 7):
					self.export_octree(noised_octree, i, f"logs/noised_octree_visualize/cosine/thres{thres}/time{ti}", index=[f'depth{i}_iter{bid}' for bid in range(self.batch_size)])

	
	def forward(self):

		self.df.train()

		cond = self.get_condition()

		with torch.no_grad():
			self.latent_code, self.doctree_in = self.autoencoder_module.extract_code(self.octree_in)

		# self.visualize_noised_octree()
		times = torch.zeros((self.batch_size,), device=self.device).float().uniform_(0,1)
		noised_doctree, noised_doctree_split_dict = self.octree_add_noise(self.doctree_in.octree, split_threshold=self.split_threshold, times=times)

		noised_latent_list = self.latent_add_noise(
			self.doctree_in, noised_doctree, 
			self.latent_code,
			times=times,
		)

		gt_latent_code = torch.cat(self.latent_code, dim=1)
		noised_latent_code = torch.cat(noised_latent_list, dim=1)
		noised_doctree_split_dict["latent"] = noised_latent_code

		output_data, logits, _ = self.df(
			noised_doctree_split_dict,
			doctree_in=noised_doctree, 
			doctree_out=self.doctree_in, 
			timesteps=times,
			**cond,
		)

		self.df_feature_loss = 0.
		self.df_split_loss = 0.

		for d in logits.keys():
			logitd = logits[d]
			label_gt = self.octree_in.nempty_mask(d).float()
			label_gt = label_gt * 2 - 1
			self.df_split_loss += F.mse_loss(logitd, label_gt) 

		self.df_feature_loss = F.mse_loss(gt_latent_code, output_data)

		self.loss = self.df_split_loss + self.df_feature_loss * 10.0


	def export_octree(self, octree, depth, save_dir = None, index = 0):

		if not os.path.exists(save_dir): os.makedirs(save_dir)

		batch_id = octree.batch_id(depth = depth, nempty = False)
		data = torch.ones((len(batch_id), 1), device = self.device)
		data = octree2voxel(data = data, octree = octree, depth = depth, nempty = False)
		data = data.permute(0,4,1,2,3).contiguous()

		for i in tqdm(range(self.batch_size)):
			voxel = data[i].squeeze().cpu().numpy()
			mesh = voxel2mesh(voxel)
			save_path = os.path.join(save_dir, f'octree/{index[i]}.obj')
			os.makedirs(os.path.dirname(save_path), exist_ok=True)
			mesh.export(save_path)


	def get_sampling_timesteps(self, batch, device, steps):
		times = torch.linspace(1., 0., steps + 1, device=device)
		times = repeat(times, 't -> b t', b=batch)
		times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
		times = times.unbind(dim=-1)
		return times
	
	@torch.no_grad()
	def cond(self, data, ema=False, ddim_steps=200, ddim_eta=0., scale=None, save_dir = None, save_index = None):
		if ema:
			self.ema_df.eval()
		else:
			self.df.eval()

		cond = self.get_condition()

		with torch.no_grad():
			self.latent_code, self.doctree_in = self.autoencoder_module.extract_code(self.octree_in)

		time = torch.ones((self.batch_size,), device=self.device).float()
		noised_doctree, noised_doctree_split_dict = self.octree_add_noise(self.doctree_in.octree, split_threshold=self.split_threshold, times=time)
		noised_latent_list = self.latent_add_noise(
			self.doctree_in, noised_doctree, 
			self.latent_code,
			times=time,
		)
		noised_latent_code = torch.cat(noised_latent_list, dim=1)
		noised_doctree_split_dict["latent"] = noised_latent_code

		time_pairs = self.get_sampling_timesteps(
			self.batch_size, device=self.device, steps=ddim_steps)

		iters = tqdm(time_pairs, desc='sampling loop time step')

		for time, time_next in iters:

			with torch.no_grad():
				if ema:
					output_data, output_logits, doctree_out  = self.ema_df(
						noised_doctree_split_dict,
						doctree_in=noised_doctree, 
						doctree_out=None, 
						timesteps=time,
						**cond,
					)
				else:
					output_data, output_logits, doctree_out = self.df(
						noised_doctree_split_dict,
						doctree_in=noised_doctree, 
						doctree_out=None, 
						timesteps=time,
						**cond,
					)
			
			# noised_doctree, noised_doctree_split_dict = self.octree_interp_noise(noised_doctree.octree, noised_doctree_split_dict, doctree_out.octree, output_logits, time, time_next)
			noised_doctree, noised_doctree_split_dict = self.octree_add_noise(self.doctree_in.octree, split_threshold=self.split_threshold, times=time_next)
			noised_latent_list = self.latent_add_noise(
				doctree_out, noised_doctree, 
				[output_data],
				times=time_next,
			)
			noised_latent_code = torch.cat(noised_latent_list, dim=1)
			noised_doctree_split_dict["latent"] = noised_latent_code

			# export octree per 10 iters
			# if round(time[0].item() * 1000) % 100 == 0:
			# 	self.export_octree(doctree_out.octree, depth=6, save_dir=save_dir, index=["%.2f_output_%s" % (time[i], save_index[i]) for i in range(self.batch_size)])
			# 	self.export_octree(noised_doctree.octree, depth=6, save_dir=save_dir, index=["%.2f_noise_%s" % (time[i], save_index[i]) for i in range(self.batch_size)])


		self.export_octree(doctree_out.octree, depth=self.doctree_depth, save_dir=save_dir, index=save_index)
		
		with torch.no_grad():
			self.output = self.autoencoder_module.decode_code(output_data, doctree_out)
		self.get_sdfs(self.output['neural_mpu'], self.batch_size, bbox=None)
		self.get_mesh(save_dir, save_index)

	def get_sdfs(self, neural_mpu, batch_size, bbox):
		# bbox used for marching cubes
		if bbox is not None:
			self.bbmin, self.bbmax = bbox[:3], bbox[3:]
		else:
			sdf_scale = self.solver.sdf_scale
			self.bbmin, self.bbmax = -sdf_scale, sdf_scale    # sdf_scale = 0.9

		self.sdfs, self.colors = calc_sdf_with_color(neural_mpu, batch_size, size = self.solver.resolution, bbmin = self.bbmin, bbmax = self.bbmax)


	@torch.no_grad()
	def eval_metrics(self, dataloader, thres=0.0, global_step=0):
		self.eval()

		ret = OrderedDict([
			('dummy_metrics', 0.0),
		])
		self.train()
		return ret

	def backward(self):
		if self.opt.accelerate:
			self.opt.booster.backward(self.loss, self.optimizer)
		else:
			self.loss.backward()

	def update_EMA(self):
		update_moving_average(self.ema_df, self.df, self.ema_updater)

	def optimize_parameters(self, data):
		self.set_input(data)
		self.set_requires_grad([self.df], requires_grad=True)

		if self.opt.accelerate and self.opt.plugin in ["torch_ddp_fp16", "low_level_zero"]:
			# with torch.cuda.amp.autocast():
			self.forward()
		else:
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

		return ret #, self.stage_flag


	def save(self, label, global_step, save_opt=True):

		state_dict = {
			'df': self.df_module.state_dict(),
			'ema_df': self.ema_df.state_dict(),
			'opt': self.optimizer.state_dict(),
			'global_step': global_step,
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

	def load_ckpt(self, ckpt, load_opt=False):
		map_fn = lambda storage, loc: storage
		if type(ckpt) == str:
			state_dict = torch.load(ckpt, map_location=map_fn)
		else:
			state_dict = ckpt
			
		self.df.load_state_dict(state_dict['df'])
		self.ema_df.load_state_dict(state_dict['ema_df'])
		print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))

		if load_opt:
			self.optimizer.load_state_dict(state_dict['opt'])
			print(colored('[*] optimizer successfully restored from: %s' % ckpt, 'blue'))

	def get_mesh(self, save_dir, index, level = 0):
		ngen = self.sdfs.shape[0]
		size = self.solver.resolution
		bbmin = self.bbmin
		bbmax = self.bbmax
		mesh_scale=self.vq_conf.DATA.test.point_scale
		import marching_cubes as mcubes
		for i in range(ngen):
			sdf_value = self.sdfs[i].cpu().numpy()
			color_value = self.colors[i].cpu().numpy()
			vtx_with_color, faces_with_color = np.zeros((0, 3)), np.zeros((0, 3))
			try:
				vtx_with_color, faces_with_color = mcubes.marching_cubes_color(sdf_value, color_value, level)
			except:
				pass
			if vtx_with_color.size == 0 or faces_with_color.size == 0:
				print('Warning from marching cubes: Empty mesh!')
				return
			vtx_with_color[:, :3] = vtx_with_color[:, :3] * ((bbmax - bbmin) / size) + bbmin   # [0,sz]->[bbmin,bbmax]  把vertex放缩到[bbmin, bbmax]之间
			vtx_with_color[:, :3] = vtx_with_color[:, :3] / mesh_scale
			mesh_path = os.path.join(save_dir, f'{index[i]}.obj')
			os.makedirs(save_dir, exist_ok=True)
			mcubes.export_obj(vtx_with_color, faces_with_color, mesh_path)

	def normalize(self, points):
		bbmin, bbmax = points.min(0)[0], points.max(0)[0]
		# center = points.mean(0)
		center = (bbmin + bbmax) * 0.5
		scale = 2.0 / (bbmax - bbmin).max()
		points = (points - center) * scale
		return points