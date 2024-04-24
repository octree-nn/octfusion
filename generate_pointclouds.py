import os
import trimesh
import numpy as np
from tqdm import tqdm

def scale_to_unit_sphere(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()
    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

def sample_pts_from_mesh(mesh_folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print('-> Run sample_pts_from_mesh.')
    num_samples = 2048
    filenames = os.listdir(mesh_folder)
    for mesh_name in tqdm(filenames[:2000]):
        mesh_path = os.path.join(mesh_folder, mesh_name)
        mesh = trimesh.load(mesh_path, force='mesh')
        # mesh: trimesh.Trimesh = scale_to_unit_sphere(mesh)
        points = mesh.sample(count = num_samples)
        filename_pts = os.path.join(output_folder, mesh_name[:-4]+'.npy')
        np.save(filename_pts, points.astype(np.float32))

if __name__ == '__main__':
    category = 'chair'
    mesh_folder = '/data/xiongbj/OctFusion-Cascade/chair_mesh_2t'
    output_folder = '/data/xiongbj/OctFusion-Cascade/chair_pointclouds'
    sample_pts_from_mesh(mesh_folder, output_folder)
