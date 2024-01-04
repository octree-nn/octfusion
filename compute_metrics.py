# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import argparse
import trimesh.sample
import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree



def compute_metrics(filename_ref, filename_pred, num_samples=30000):
  mesh_ref = trimesh.load(filename_ref)
  points_ref, idx_ref = trimesh.sample.sample_surface(mesh_ref, num_samples)
  normals_ref = mesh_ref.face_normals[idx_ref]
  # points_ref, normals_ref = read_ply(filename_ref)

  mesh_pred = trimesh.load(filename_pred)
  points_pred, idx_pred = trimesh.sample.sample_surface(mesh_pred, num_samples)
  normals_pred = mesh_pred.face_normals[idx_pred]

  kdtree_a = cKDTree(points_ref)
  dist_a, idx_a = kdtree_a.query(points_pred)
  chamfer_a = np.mean(dist_a)
  dot_a = np.sum(normals_pred * normals_ref[idx_a], axis=1)
  angle_a = np.mean(np.arccos(dot_a) * (180.0 / np.pi))
  consist_a = np.mean(np.abs(dot_a))

  kdtree_b = cKDTree(points_pred)
  dist_b, idx_b = kdtree_b.query(points_ref)
  chamfer_b = np.mean(dist_b)
  dot_b = np.sum(normals_ref * normals_pred[idx_b], axis=1)
  angle_b = np.mean(np.arccos(dot_b) * (180 / np.pi))
  consist_b = np.mean(np.abs(dot_b))

  return chamfer_a, chamfer_b, angle_a, angle_b, consist_a, consist_b

