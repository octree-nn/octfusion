import trimesh
import os
import numpy as np
from tqdm import tqdm

filelist = '/data/xiongbj/ShapeNet/filelist/test_chair.txt'
mesh_dataset = '/data/xiongbj/ShapeNet/mesh'
pointcloud_path = '/data/xiongbj/ShapeNet/test_pointclouds'

def scale_to_unit_sphere(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()
    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


def scale_to_unit_cube(mesh, padding=0.0):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    vertices *= 2 / np.max(mesh.bounding_box.extents) * (1 - padding)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


def sample_pts_from_mesh(mesh_path, output_path):

    num_samples = 2048
    mesh = trimesh.load(mesh_path, force='mesh')
    mesh = scale_to_unit_cube(mesh)

    points = mesh.sample(count = num_samples)

    np.save(output_path, points.astype(np.float32))

if __name__ == '__main__':
    print('-> Run sample_pts_from_mesh.')

    with open(filelist) as fid:
        lines = fid.readlines()

    for i, line in tqdm(enumerate(lines)):
        filename = line.split()[0]
        category = filename.split('/')[0]
        category_path = os.path.join(pointcloud_path, category)
        if not os.path.exists(category_path): os.makedirs(category_path)
        mesh_path = os.path.join(mesh_dataset, filename, 'model.obj')
        output_path = os.path.join(pointcloud_path, filename + '.npy')
        sample_pts_from_mesh(mesh_path, output_path)
