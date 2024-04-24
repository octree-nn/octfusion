import os
import sys
from compute_metrics import compute_metrics
import trimesh
import numpy as np
import torch
import pickle

gpu_ids = 3
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_ids}"

num_samples = 2048

input_obj = 'chair_mesh_2t/4165.obj'

def normalize_pc_to_unit_shpere(points):
    centroid = (np.max(points, axis=0) + np.min(points, axis=0))/2
    points -= centroid
    distances = np.linalg.norm(points, axis=1)
    points /= np.max(distances)
    return points

mesh = trimesh.load(input_obj, force='mesh')
points, idx = trimesh.sample.sample_surface(mesh, num_samples)
points = points.astype(np.float32)

points = normalize_pc_to_unit_shpere(points)

points = torch.from_numpy(points)
points = points.cuda().to(torch.float32)

sample_pc = points

ref_pcs = torch.load('chair_train_ref_pcs.pth')
ref_pcs = ref_pcs.cuda().to(torch.float32)

cd = compute_metrics(sample_pc, ref_pcs, batch_size = 256)

with open('name.pkl', 'rb') as file:
    name = pickle.load(file)

k = 3

sorted_values, sorted_indices = torch.topk(cd.view(-1), k, largest=False)

print(name[sorted_indices[0].item()])
print(name[sorted_indices[1].item()])
print(name[sorted_indices[2].item()])
