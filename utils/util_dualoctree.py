# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import torch.autograd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.measure
import trimesh
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from ocnn.nn import octree2voxel, octree_pad
import copy
from models.networks.diffusion_networks.ldm_diffusion_util import create_full_octree


def get_mgrid(size, dim=3):
    r'''
    Example:
    >>> get_mgrid(3, dim=2)
        array([[0.0,  0.0],
                [0.0,  1.0],
                [0.0,  2.0],
                [1.0,  0.0],
                [1.0,  1.0],
                [1.0,  2.0],
                [2.0,  0.0],
                [2.0,  1.0],
                [2.0,  2.0]], dtype=float32)
    '''
    coord = np.arange(0, size, dtype=np.float32)
    coords = [coord] * dim
    output = np.meshgrid(*coords, indexing='ij')
    output = np.stack(output, -1)
    output = output.reshape(size**dim, dim)
    return output    # 返回[size**3, 3]的array


def lin2img(tensor):
    channels = 1
    num_samples = tensor.shape
    size = int(np.sqrt(num_samples))
    return tensor.view(channels, size, size)


def make_contour_plot(array_2d, mode='log'):
    fig, ax = plt.subplots(figsize=(2.75, 2.75), dpi=300)

    if(mode == 'log'):
        nlevels = 6
        levels_pos = np.logspace(-2, 0, num=nlevels)  # logspace
        levels_neg = -1. * levels_pos[::-1]
        levels = np.concatenate((levels_neg, np.zeros((0)), levels_pos), axis=0)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=nlevels * 2 + 1))
    elif(mode == 'lin'):
        nlevels = 10
        levels = np.linspace(-.5, .5, num=nlevels)
        colors = plt.get_cmap("Spectral")(np.linspace(0., 1., num=nlevels))
    else:
        raise NotImplementedError

    sample = np.flipud(array_2d)
    CS = ax.contourf(sample, levels=levels, colors=colors)
    cbar = fig.colorbar(CS)

    ax.contour(sample, levels=levels, colors='k', linewidths=0.1)
    ax.contour(sample, levels=[0], colors='k', linewidths=0.3)
    ax.axis('off')
    return fig


def write_sdf_summary(model, writer, global_step, alias=''):
    size = 128
    coords_2d = get_mgrid(size, dim=2)
    coords_2d = coords_2d / size - 1.0   # [0, size] -> [-1, 1]
    coords_2d = torch.from_numpy(coords_2d)
    with torch.no_grad():
        zeros = torch.zeros_like(coords_2d[:, :1])
        ones = torch.ones_like(coords_2d[:, :1])
        names = ['train_yz_sdf_slice', 'train_xz_sdf_slice', 'train_xy_sdf_slice']
        coords = [torch.cat((zeros, coords_2d), dim=-1),
                torch.cat((coords_2d[:, :1], zeros, coords_2d[:, -1:]), dim=-1),
                torch.cat((coords_2d, -0.75 * ones), dim=-1)]
        for name, coord in zip(names, coords):
            ids = torch.zeros(coord.shape[0], 1)
            coord = torch.cat([coord, ids], dim=1).cuda()
            sdf_values = model(coord)
            sdf_values = lin2img(sdf_values).squeeze().cpu().numpy()
            fig = make_contour_plot(sdf_values)
            writer.add_figure(alias + name, fig, global_step=global_step)


def calc_sdf(model, batch_size = 1, size=256, max_batch=64**3, bbmin=-1.0, bbmax=1.0):
  # generate samples
    num_samples = size ** 3
    samples = get_mgrid(size, dim=3)  # 得到的sample是[size**3, 3]大小的网格点坐标，每个坐标都在[0, size]内
    samples = samples * ((bbmax - bbmin) / size) + bbmin  # [0,sz]->[bbmin,bbmax]   把[0,size]内的坐标值放缩到[bbmin, bbmax]内，bbmin = -sdf_scale, bbmax = sdf_scale
    samples = torch.from_numpy(samples)
    sdfs = torch.zeros(batch_size, num_samples)  # [size**3,]

    for batch_idx in range(batch_size):
        head = 0
        while head < num_samples:
            tail = min(head + max_batch, num_samples)
            sample_subset = samples[head:tail, :]   # 在这个batch中的采样点的坐标，[batch_num, 3]
            idx = torch.zeros(sample_subset.shape[0], 1) + batch_idx  # [batch_num,1]
            pts = torch.cat([sample_subset, idx], dim=1).cuda()  # [batch_num, 4]，前三个是采样点坐标，最后一列都是0（因为测试的时候batch_size为1，所以batch_idx都为0）
            pred = model(pts).squeeze().detach().cpu()  # 计算batch_num个采样点的sdf值。
            sdfs[batch_idx][head:tail] = pred  # 把计算得到的batch_num个采样点的sdf值存储在sdfs里
            head += max_batch   # 然后进行下一轮batch
    sdfs = sdfs.reshape(batch_size, size, size, size).cuda()  # 最后把sdfs resize到[size, size, size]
    return sdfs

def create_mesh(model, filename, size=256, max_batch=64**3, level=0,  # 这里的model就是一个函数，根据输入的pos计算sdf值
                bbmin=-0.9, bbmax=0.9, mesh_scale=1.0, save_sdf=False, **kwargs):  # size是sdf采样分辨率的大小
    # marching cubes
    sdf_values = calc_sdf(model, size, max_batch, bbmin, bbmax)  # 返回[size, size, size]，采样点的坐标大小在[bbmin, bbmax]之间。
    vtx, faces = np.zeros((0, 3)), np.zeros((0, 3))
    try:
        vtx, faces, _, _ = skimage.measure.marching_cubes(sdf_values, level)  # marching cude得到vtx和faces。
    except:
        pass
    if vtx.size == 0 or faces.size == 0:
        print('Warning from marching cubes: Empty mesh!')
        return

    # normalize vtx
    vtx = vtx * ((bbmax - bbmin) / size) + bbmin   # [0,sz]->[bbmin,bbmax]  把vertex放缩到[bbmin, bbmax]之间
    vtx = vtx * mesh_scale                         # 然后存储的时候进行放缩：mesh_scale = point_scale = 0.5

    # save to ply and npy
    mesh = trimesh.Trimesh(vtx, faces)  # 利用Trimesh创建mesh并存储为obj文件。
    mesh.export(filename)
    if save_sdf:
        np.save(filename[:-4] + ".sdf.npy", sdf_values)


def calc_sdf_err(filename_gt, filename_pred):
    scale = 1.0e2  # scale the result for better display
    sdf_gt = np.load(filename_gt)
    sdf = np.load(filename_pred)
    err = np.abs(sdf - sdf_gt).mean() * scale
    return err


def calc_chamfer(filename_gt, filename_pred, point_num):
    scale = 1.0e5  # scale the result for better display
    np.random.seed(101)

    mesh_a = trimesh.load(filename_gt)
    points_a, _ = trimesh.sample.sample_surface(mesh_a, point_num)
    mesh_b = trimesh.load(filename_pred)
    points_b, _ = trimesh.sample.sample_surface(mesh_b, point_num)

    kdtree_a = cKDTree(points_a)
    dist_a, _ = kdtree_a.query(points_b)
    chamfer_a = np.mean(np.square(dist_a)) * scale

    kdtree_b = cKDTree(points_b)
    dist_b, _ = kdtree_b.query(points_a)
    chamfer_b = np.mean(np.square(dist_b)) * scale
    return chamfer_a, chamfer_b


def points2ply(filename, points, scale=1.0):
    xyz = points.points
    # xyz = ocnn.points_property(points, 'xyz')
    normal = points.normals
    # normal = ocnn.points_property(points, 'normal')
    has_normal = normal is not None
    xyz = xyz.numpy() * scale   #  这里同样乘以point_scale = 0.5，不知道为什么每次存储point都要乘0.5。
    if has_normal: normal = normal.numpy()

    # data types
    data = xyz
    py_types = (float, float, float)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    if has_normal:
        py_types = py_types + (float, float, float)
        npy_types = npy_types + [('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
        data = np.concatenate((data, normal), axis=1)

    # format into NumPy structured array
    vertices = []
    for idx in range(data.shape[0]):
        vertices.append(tuple(dtype(d) for dtype, d in zip(py_types, data[idx])))
    structured_array = np.array(vertices, dtype=npy_types)
    el = PlyElement.describe(structured_array, 'vertex')

    # write ply
    PlyData([el]).write(filename)

def octree2split_small(octree, full_depth):

    child_full_p1 = octree.children[full_depth + 1]
    split_full_p1 = (child_full_p1 >= 0)
    split_full_p1 = split_full_p1.reshape(-1, 8)
    split_full = octree_pad(data = split_full_p1, octree = octree, depth = full_depth)
    split_full = octree2voxel(data=split_full, octree=octree, depth = full_depth)
    split_full = split_full.permute(0,4,1,2,3).contiguous()

    split_full = split_full.float()
    split_full = 2 * split_full - 1  # scale to [-1, 1]

    return split_full

def octree2split_large(octree, small_depth):

    child_small_p1 = octree.children[small_depth + 1]
    split_small_p1 = (child_small_p1 >= 0)
    split_small_p1 = split_small_p1.reshape(-1, 8)
    split_small = octree_pad(data = split_small_p1, octree = octree, depth = small_depth)

    split_small = split_small.float()
    split_small = 2 * split_small - 1    # scale to [-1, 1]

    return split_small

def split2octree_small(split, input_depth, full_depth):

    discrete_split = copy.deepcopy(split)
    discrete_split[discrete_split > 0] = 1
    discrete_split[discrete_split < 0] = 0

    batch_size = discrete_split.shape[0]
    octree_out = create_full_octree(depth = input_depth, full_depth = full_depth, batch_size = batch_size, device = split.device)
    split_sum = torch.sum(discrete_split, dim = 1)
    nempty_mask_voxel = (split_sum > 0)
    x, y, z, b = octree_out.xyzb(full_depth)
    nempty_mask = nempty_mask_voxel[b,x,y,z]
    label = nempty_mask.long()
    octree_out.octree_split(label, full_depth)
    octree_out.octree_grow(full_depth + 1)
    octree_out.depth += 1

    x, y, z, b = octree_out.xyzb(depth = full_depth, nempty = True)
    nempty_mask_p1 = discrete_split[b,:,x,y,z]
    nempty_mask_p1 = nempty_mask_p1.reshape(-1)
    label_p1 = nempty_mask_p1.long()
    octree_out.octree_split(label_p1, full_depth + 1)
    octree_out.octree_grow(full_depth + 2)
    octree_out.depth += 1

    return octree_out

def split2octree_large(octree, split, small_depth):

    discrete_split = copy.deepcopy(split)
    discrete_split[discrete_split > 0] = 1
    discrete_split[discrete_split < 0] = 0

    octree_out = copy.deepcopy(octree)
    split_sum = torch.sum(discrete_split, dim = 1)
    nempty_mask_small = (split_sum > 0)
    label = nempty_mask_small.long()
    octree_out.octree_split(label, depth = small_depth)
    octree_out.octree_grow(small_depth + 1)
    octree_out.depth += 1

    nempty_mask_small_p1 = discrete_split[split_sum > 0]
    nempty_mask_small_p1 = nempty_mask_small_p1.reshape(-1)
    label_p1 = nempty_mask_small_p1.long()
    octree_out.octree_split(label_p1, depth = small_depth + 1)
    octree_out.octree_grow(small_depth + 2)
    octree_out.depth += 1

    return octree_out