import torch
import numpy as np
import ocnn
from ocnn.octree import Octree, Points
from glob import glob
import os
import sys
from tqdm import tqdm
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(current_dir))
from utils.util_dualoctree import split2octree_large, split2octree_small, octree2split_large, octree2split_small
root_folder = "data/Objaverse"
pointcloud_dir = "data/Objaverse/objaverse-select"
split_dir = "data/Objaverse/objaverse-split"
octree_dir = "data/Objaverse/objaverse-octree"

def get_filenames(filelist):
    r''' Gets filenames from a filelist.
    '''

    filelist = os.path.join(root_folder, 'filelist', filelist)
    with open(filelist, 'r') as fid:
        lines = fid.readlines()
    filenames = [line.split()[0] for line in lines]
    return filenames

def points2octree(points):
    octree = ocnn.octree.Octree(depth = 10, full_depth = 4)
    octree.build_octree(points)
    return octree
        
filenames = get_filenames('train_obja.txt')

for filename in tqdm(filenames):
    print(filename)
    filename_pointcloud = os.path.join(pointcloud_dir, filename, "pointcloud.npz")
    filename_split = os.path.join(split_dir, filename)
    filename_octree = os.path.join(octree_dir, filename, "octree.pth")
    raw = np.load(filename_pointcloud)
    points, normals = raw['points'], raw['normals']

    # transform points to octree
    points_gt = Points(points = torch.from_numpy(points).float(), normals = torch.from_numpy(normals).float())
    octree_gt = points2octree(points_gt)
    os.makedirs(os.path.dirname(filename_octree), exist_ok = True)
    torch.save(octree_gt, filename_octree)
    
    split_small = octree2split_small(octree_gt, full_depth=4)
    split_large = octree2split_large(octree_gt, small_depth=6)
    os.makedirs(filename_split, exist_ok = True)
    torch.save(split_small.squeeze(0), os.path.join(filename_split, "split_small.pth"))
    torch.save(split_large, os.path.join(filename_split, "split_large.pth"))
    
    
