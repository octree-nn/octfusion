import os
import numpy as np
import torch

# pc_path = '/data/xiongbj/OctFusion-Cascade/chair_pointclouds'
pc_path = '/data/xiongbj/ShapeNet/test_pointclouds/03001627'

def normalize_pc_to_unit_shpere(points):
    centroid = (np.max(points, axis=0) + np.min(points, axis=0))/2
    points -= centroid
    distances = np.linalg.norm(points, axis=1)
    points /= np.max(distances)
    return points

points = os.listdir(pc_path)
pcs = []

for index, point in enumerate(points):
    print(index, point)
    point_path = os.path.join(pc_path, point)
    point_numpy = np.load(point_path)
    point_numpy = normalize_pc_to_unit_shpere(point_numpy)
    pcs.append(torch.from_numpy(point_numpy))

pcs = torch.stack(pcs, dim = 0)
print(pcs.shape)

torch.save(pcs, 'chair_ref_pcs.pth')
