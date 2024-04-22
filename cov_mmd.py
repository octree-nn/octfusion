from pprint import pprint
from metrics.evaluation_metrics import compute_cov_mmd, compute_1_nna
import torch
import time
import numpy as np
import os
import argparse
import sys
import pickle


sample_pc_path = '/data/checkpoints/xiongbj/OctFusion-Full/chair_single_pointclouds'
# sample_pc_path = '/data/checkpoints/xiongbj/3DShape2VecSet/chair_pointclouds'
ref_pc_path = '/data/checkpoints/xiongbj/DualOctreeGNN-Pytorch-HR/data/ShapeNet/test_pointclouds/03001627'

def normalize_pc_to_unit_shpere(points):
    centroid = (np.max(points, axis=0) + np.min(points, axis=0))/2
    points -= centroid
    distances = np.linalg.norm(points, axis=1)
    points /= np.max(distances)
    return points

# sample_points = os.listdir(sample_pc_path)
# sample_pcs = []

# for index, point in enumerate(sample_points[:1356]):
#     print(index, point)
#     point_path = os.path.join(sample_pc_path, point)
#     point_numpy = np.load(point_path)
#     sample_pcs.append(torch.from_numpy(point_numpy))

# sample_pcs = torch.stack(sample_pcs, dim = 0)
# print(sample_pcs.shape)

# torch.save(sample_pcs, 'chair_sample_pcs.pth')

# exit(1)

# ref_points = os.listdir(ref_pc_path)
# ref_pcs = []

# for index, point in enumerate(ref_points):
#     print(index, point)
#     point_path = os.path.join(ref_pc_path, point)
#     point_numpy = np.load(point_path)
#     ref_pcs.append(torch.from_numpy(point_numpy))

# ref_pcs = torch.stack(ref_pcs, dim = 0)
# print(ref_pcs.shape)

# torch.save(ref_pcs, 'shapenet_v1_chair_ref_pcs.pth')

# exit(1)

test_sample_pcs = torch.load('las_sample_pcs.pth')
print(test_sample_pcs.shape)

test_sample_pcs = test_sample_pcs.cuda().to(torch.float32)

test_ref_pcs = torch.load('shapenet_v1_chair_ref_pcs.pth')
print(test_ref_pcs.shape)

test_ref_pcs = test_ref_pcs.cuda().to(torch.float32)

print('##################################################################')

results = compute_cov_mmd(sample_pcs = test_sample_pcs[:256], ref_pcs=test_ref_pcs[:256], batch_size = 256)
results = {k: (v.cpu().detach().item()
              if not isinstance(v, float) else v) for k, v in results.items()}

pprint(results)

sys.exit()

results = compute_1_nna(
    sample_pcs[:ref_pcs.shape[0]], ref_pcs, batch_size = 256)
results = {k: (v.cpu().detach().item()
              if not isinstance(v, float) else v) for k, v in results.items()}

pprint(results)

print('##################################################################')
