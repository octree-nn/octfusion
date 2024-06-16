from utils.render_utils import generate_image_for_fid
import trimesh
import os

filelist = '/data/checkpoints/xiongbj/DualOctreeGNN-Pytorch-HR/data/ShapeNet/filelist/train_im_5.txt'
mesh_dataset = '/data/public-datasets/ShapeNetCore.v1'
image_path = '/data/checkpoints/xiongbj/DualOctreeGNN-Pytorch-HR/data/ShapeNet/fid_images'

with open(filelist) as fid:
    lines = fid.readlines()

for i, line in enumerate(lines):
    filename = line.split()[0]
    category = filename.split('/')[0]
    category_path = os.path.join(image_path, category)
    if not os.path.exists(category_path): os.makedirs(category_path)
    mesh_path = os.path.join(mesh_dataset, filename, 'model.obj')
    mesh = trimesh.load(mesh_path, force = 'mesh')
    generate_image_for_fid(mesh,category_path, i)
    print(f'The {i} th mesh finish rendering')
