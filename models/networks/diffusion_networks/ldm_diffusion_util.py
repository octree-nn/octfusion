# Reference: https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/util.py


import os
import math
import torch
import torch.nn as nn
import numpy as np
from einops import repeat
from ocnn.octree import Octree
import skimage.measure
import trimesh

# from ldm.util import instantiate_from_config ## copy from here
import importlib

from functools import partial
from inspect import isfunction

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class our_Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ma_model, current_model, ema_updater):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    # import pdb; pdb.set_trace()
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def checkpoint(func, inputs, params, flag):
    def wrapper(dummy_tensor, *args):
        return func(*args)

    # The dummy tensor is a workaround when the checkpoint is used for the first conv layer:
    # https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/11
    if flag:
        dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)
        return torch.utils.checkpoint.checkpoint(
            wrapper, dummy, *inputs, use_reentrant=False)
    else:
        return func(*inputs)

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def voxelnormalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class HybridConditioner(nn.Module):

    def __init__(self, c_concat_config, c_crossattn_config):
        super().__init__()
        self.concat_conditioner = instantiate_from_config(c_concat_config)
        self.crossattn_conditioner = instantiate_from_config(c_crossattn_config)

    def forward(self, c_concat, c_crossattn):
        c_concat = self.concat_conditioner(c_concat)
        c_crossattn = self.crossattn_conditioner(c_crossattn)
        return {'c_concat': [c_concat], 'c_crossattn': [c_crossattn]}


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def beta_linear_log_snr(t):
    return -torch.log(torch.special.expm1(1e-4 + 10 * (t ** 2)))


def alpha_cosine_log_snr(t, s: float = 0.008):
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps=1e-5)


def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))

def get_sampling_timesteps(batch, device, steps):
    times = torch.linspace(1., 0., steps + 1, device=device)
    times = repeat(times, 't -> b t', b=batch)
    times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
    times = times.unbind(dim=-1)
    return times
    
def create_full_octree(depth, full_depth,batch_size, device):
    r''' Initialize a full octree for decoding.
    '''
    octree = Octree(depth, full_depth, batch_size, device)
    for d in range(full_depth+1):
      octree.octree_grow_full(depth=d)
    octree.depth = full_depth
    return octree

def voxel2fulloctree(voxel: torch.Tensor, depth ,batch_size, device, nempty: bool = False):
  r''' Converts the input feature to the full-voxel-based representation.

  Args:
    voxel (torch.Tensor): batch_size, channel, num, num, num
    depth (int): The depth of current octree.
    nempty (bool): If True, :attr:`data` only contains the features of non-empty
        octree nodes.
  '''
  channel = voxel.shape[1]
  octree = create_full_octree(depth = depth, full_depth = depth, batch_size = batch_size, device = device)
  x, y, z, b = octree.xyzb(depth, nempty)
  key = octree.key(depth, nempty)
  data = voxel.new_zeros(key.shape[0], channel)
  data = voxel[b,:, x,y,z]

  return data

def voxel2mesh(voxel, threshold=0.4, use_vertex_normal: bool = False):
    verts, faces, vertex_normals = _voxel2mesh(voxel, threshold)
    if use_vertex_normal:
        return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=vertex_normals)
    else:
        return trimesh.Trimesh(vertices=verts, faces=faces)


def _voxel2mesh(voxels, threshold=0.5):

    top_verts = [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    top_faces = [[0, 1, 3], [1, 2, 3]]
    top_normals = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]

    bottom_verts = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    bottom_faces = [[1, 0, 3], [2, 1, 3]]
    bottom_normals = [[0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1]]

    left_verts = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]
    left_faces = [[0, 1, 3], [2, 0, 3]]
    left_normals = [[-1, 0, 0], [-1, 0, 0], [-1, 0, 0], [-1, 0, 0]]

    right_verts = [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    right_faces = [[1, 0, 3], [0, 2, 3]]
    right_normals = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]

    front_verts = [[0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1]]
    front_faces = [[1, 0, 3], [0, 2, 3]]
    front_normals = [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]

    back_verts = [[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]]
    back_faces = [[0, 1, 3], [2, 0, 3]]
    back_normals = [[0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0]]

    top_verts = np.array(top_verts)
    top_faces = np.array(top_faces)
    bottom_verts = np.array(bottom_verts)
    bottom_faces = np.array(bottom_faces)
    left_verts = np.array(left_verts)
    left_faces = np.array(left_faces)
    right_verts = np.array(right_verts)
    right_faces = np.array(right_faces)
    front_verts = np.array(front_verts)
    front_faces = np.array(front_faces)
    back_verts = np.array(back_verts)
    back_faces = np.array(back_faces)

    dim = voxels.shape[0]
    new_voxels = np.zeros((dim+2, dim+2, dim+2))
    new_voxels[1:dim+1, 1:dim+1, 1:dim+1] = voxels
    voxels = new_voxels

    scale = 2/dim
    verts = []
    faces = []
    vertex_normals = []
    curr_vert = 0
    a, b, c = np.where(voxels > threshold)

    for i, j, k in zip(a, b, c):
        if voxels[i, j, k+1] < threshold:
            verts.extend(scale * (top_verts + np.array([[i-1, j-1, k-1]])))
            faces.extend(top_faces + curr_vert)
            vertex_normals.extend(top_normals)
            curr_vert += len(top_verts)

        if voxels[i, j, k-1] < threshold:
            verts.extend(
                scale * (bottom_verts + np.array([[i-1, j-1, k-1]])))
            faces.extend(bottom_faces + curr_vert)
            vertex_normals.extend(bottom_normals)
            curr_vert += len(bottom_verts)

        if voxels[i-1, j, k] < threshold:
            verts.extend(scale * (left_verts +
                         np.array([[i-1, j-1, k-1]])))
            faces.extend(left_faces + curr_vert)
            vertex_normals.extend(left_normals)
            curr_vert += len(left_verts)

        if voxels[i+1, j, k] < threshold:
            verts.extend(scale * (right_verts +
                         np.array([[i-1, j-1, k-1]])))
            faces.extend(right_faces + curr_vert)
            vertex_normals.extend(right_normals)
            curr_vert += len(right_verts)

        if voxels[i, j+1, k] < threshold:
            verts.extend(scale * (front_verts +
                         np.array([[i-1, j-1, k-1]])))
            faces.extend(front_faces + curr_vert)
            vertex_normals.extend(front_normals)
            curr_vert += len(front_verts)

        if voxels[i, j-1, k] < threshold:
            verts.extend(scale * (back_verts +
                         np.array([[i-1, j-1, k-1]])))
            faces.extend(back_faces + curr_vert)
            vertex_normals.extend(back_normals)
            curr_vert += len(back_verts)

    return np.array(verts) - 1, np.array(faces), np.array(vertex_normals)
