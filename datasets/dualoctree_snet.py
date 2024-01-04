# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import ocnn
import torch
import numpy as np
import copy

from ocnn.octree import Octree, Points
from solver import Dataset
from .utils import collate_func


class TransformShape:

  def __init__(self, flags):
    self.flags = flags

    self.depth = flags.depth
    self.full_depth = flags.full_depth

    self.point_sample_num = 3000
    self.sdf_sample_num = 5000
    self.first_points_scale = 0.5  # the points are in [-0.5, 0.5]
    self.second_points_scale = flags.points_scale
    self.noise_std = 0.005

  def points2octree(self, points: Points):
    octree = Octree(self.depth, self.full_depth)
    octree.build_octree(points)
    return octree

  def points2voxel(self, points):
    scale = 2 ** (self.depth - 1)
    points = (points + 1.0) * scale
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    x ,y, z = x.long(), y.long(), z.long()
    voxel_size = 2 ** self.depth
    voxel = torch.zeros([voxel_size, voxel_size, voxel_size]).long()
    voxel[x,y,z] = 1
    return voxel

  def process_points_cloud(self, sample):
    # get the input
    points, normals = sample['points'], sample['normals']
    points = points / self.first_points_scale  # scale to [-1.0, 1.0]
    points = points * self.second_points_scale
    
    # transform points to octree
    points_gt = Points(points = torch.from_numpy(points).float(),normals = torch.from_numpy(normals).float())
    points_gt.clip(min=-1, max=1)

    return {'points': points_gt}

  def sample_sdf(self, sample):   # 这里加载的sdf的坐标也都是在[-1,1]范围内的。
    sdf = sample['sdf']
    grad = sample['grad']
    points = sample['points'] / self.points_scale  # to [-1, 1]

    points = points * self.second_points_scale
    sdf = sdf * self.second_points_scale

    rand_idx = np.random.choice(points.shape[0], size=self.sdf_sample_num)
    points = torch.from_numpy(points[rand_idx]).float()
    sdf = torch.from_numpy(sdf[rand_idx]).float()
    grad = torch.from_numpy(grad[rand_idx]).float()
    return {'pos': points, 'sdf': sdf, 'grad': grad}

  def sample_on_surface(self, points, normals):
    rand_idx = np.random.choice(points.shape[0], size=self.sdf_sample_num)
    xyz = torch.from_numpy(points[rand_idx]).float()
    grad = torch.from_numpy(normals[rand_idx]).float()
    sdf = torch.zeros(self.sdf_sample_num)
    return {'pos': xyz, 'sdf': sdf, 'grad': grad}

  def sample_off_surface(self, xyz):
    xyz = xyz / self.points_scale  # to [-1, 1]

    rand_idx = np.random.choice(xyz.shape[0], size=self.sdf_sample_num)
    xyz = torch.from_numpy(xyz[rand_idx]).float()
    # grad = torch.zeros(self.sample_number, 3)  # dummy grads
    grad = xyz / (xyz.norm(p=2, dim=1, keepdim=True) + 1.0e-6)
    sdf = -1 * torch.ones(self.sdf_sample_num)  # dummy sdfs
    return {'pos': xyz, 'sdf': sdf, 'grad': grad}

  def __call__(self, sample, idx):
    output = self.process_points_cloud(sample['point_cloud'])

    # sample ground truth sdfs
    if self.flags.load_sdf:
      sdf_samples = self.sample_sdf(sample['sdf'])
      output.update(sdf_samples)

    # sample on surface points and off surface points
    if self.flags.sample_surf_points:
      on_surf = self.sample_on_surface(sample['points'], sample['normals'])
      off_surf = self.sample_off_surface(sample['sdf']['points'])  # TODO
      sdf_samples = {
          'pos': torch.cat([on_surf['pos'], off_surf['pos']], dim=0),
          'grad': torch.cat([on_surf['grad'], off_surf['grad']], dim=0),
          'sdf': torch.cat([on_surf['sdf'], off_surf['sdf']], dim=0)}
      output.update(sdf_samples)

    return output


class ReadFile:
  def __init__(self, load_sdf=False, load_occu=False):
    self.load_occu = load_occu
    self.load_sdf = load_sdf

  def __call__(self, filename):
    filename_pc = os.path.join(filename, 'pointcloud.npz')
    raw = np.load(filename_pc)
    point_cloud = {'points': raw['points'], 'normals': raw['normals']}
    output = {'point_cloud': point_cloud}

    if self.load_occu:
      filename_occu = os.path.join(filename, 'points.npz')
      raw = np.load(filename_occu)
      occu = {'points': raw['points'], 'occupancies': raw['occupancies']}
      output['occu'] = occu

    if self.load_sdf:
      filename_sdf = os.path.join(filename, 'sdf.npz')
      raw = np.load(filename_sdf)
      sdf = {'points': raw['points'], 'grad': raw['grad'], 'sdf': raw['sdf']}
      output['sdf'] = sdf
    return output


def get_shapenet_dataset(flags):
  transform = TransformShape(flags)
  read_file = ReadFile(flags.load_sdf, flags.load_occu)
  dataset = Dataset(flags.location, flags.filelist, transform,
                    read_file=read_file, in_memory=flags.in_memory)
  return dataset, collate_func
