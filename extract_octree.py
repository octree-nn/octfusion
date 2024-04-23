import ocnn
import torch
import numpy as np
from ocnn.octree import Octree, Points
import os
from ocnn.nn import octree2voxel, octree_pad

input_depth = 8
small_depth = 6
full_depth = 4

def points2octree(points):
    octree = ocnn.octree.Octree(depth = input_depth, full_depth = full_depth)
    octree.build_octree(points)
    return octree

def octree2split_small(octree):
    child_full_p1 = octree.children[full_depth + 1]
    split_full_p1 = (child_full_p1 >= 0)
    split_full_p1 = split_full_p1.reshape(-1, 8)
    split_full = octree_pad(data = split_full_p1, octree = octree, depth = full_depth)
    split_full = octree2voxel(data=split_full, octree=octree, depth = full_depth)
    split_full = split_full.permute(0,4,1,2,3).contiguous()

    split_full = split_full.float()
    split_full = 2 * split_full - 1  # scale to [-1, 1]

    return split_full

def octree2split_large(octree):

    child_small_p1 = octree.children[small_depth + 1]
    split_small_p1 = (child_small_p1 >= 0)
    split_small_p1 = split_small_p1.reshape(-1, 8)
    split_small = octree_pad(data = split_small_p1, octree = octree, depth = small_depth)

    split_small = split_small.float()
    split_small = 2 * split_small - 1    # scale to [-1, 1]

    return split_small

category = '04090263'

# 02691156
# 02958343
# 03001627
# 04090263
# 04379243

dataset = f'/data/xiongbj/ShapeNet/dataset_256/{category}'
split_path_small = f'/data/xiongbj/ShapeNet/split_small/{category}'
split_path_large = f'/data/xiongbj/ShapeNet/split_large/{category}'

os.makedirs(split_path_small, exist_ok = True)
os.makedirs(split_path_large, exist_ok = True)

meshes = os.listdir(dataset)

points_scale = 0.5

for mesh in meshes:
    if mesh in ['val.lst', 'train.lst', 'test.lst']: continue
    mesh_dir = os.path.join(dataset, mesh)
    filename_pc = os.path.join(mesh_dir, 'pointcloud.npz')
    raw = np.load(filename_pc)
    points = raw['points']
    normals = raw['normals']
    points = points / points_scale
    points_gt = Points(points = torch.from_numpy(points).float(),normals = torch.from_numpy(normals).float())
    points_gt.clip(min=-1, max=1)
    points = [points_gt.cuda(non_blocking=True)]
    octrees = [points2octree(pts) for pts in points]
    octree = ocnn.octree.merge_octrees(octrees)
    octree.construct_all_neigh()
    split_small = octree2split_small(octree)
    split_large = octree2split_large(octree)
    split_small = split_small.squeeze()
    print(mesh, split_small.shape, split_small.max(), split_small.min())
    print(mesh, split_large.shape, split_large.max(), split_large.min())
    target_path_small = os.path.join(split_path_small, f'{mesh}.pth')
    target_path_large = os.path.join(split_path_large, f'{mesh}.pth')
    torch.save(split_small, target_path_small)
    torch.save(split_large, target_path_large)
