# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import time
import wget
import shutil
import torch
import ocnn
import trimesh
import logging
import mesh2sdf
import zipfile
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from plyfile import PlyData, PlyElement

logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, default="generate_dataset")
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=45572)
parser.add_argument('--sdf_size', type=int, default=128)
args = parser.parse_args()

size = args.sdf_size        # resolution of SDF
size = 128                 # resolution of SDF
level = 0.015            # 2/128 = 0.015625
shape_scale = 0.5    # rescale the shape into [-0.5, 0.5]
project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
root_folder = os.path.join(project_folder, 'data/ShapeNet')
file_folder = 'data/ShapeNet/'


def create_flag_file(filename):
    r''' Creates a flag file to indicate whether some time-consuming works
    have been done.
    '''

    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(filename, 'w') as fid:
        fid.write('succ @ ' + time.ctime())


def check_folder(filenames: list):
    r''' Checks whether the folder contains the filename exists.
    '''

    for filename in filenames:
        folder = os.path.dirname(filename)
        if not os.path.exists(folder):
            os.makedirs(folder)


def get_filenames(filelist):
    r''' Gets filenames from a filelist.
    '''

    filelist = os.path.join(root_folder, 'filelist', filelist)
    with open(filelist, 'r') as fid:
        lines = fid.readlines()
    filenames = [line.split()[0] for line in lines]
    return filenames


def unzip_shapenet():
    r''' Unzip the ShapeNetCore.v1
    '''

    # filename = os.path.join(root_folder, 'ShapeNetCore.v1.zip')
    filename = os.path.join(file_folder, 'ShapeNetCore.v1.zip')
    flag_file = os.path.join(root_folder, 'flags/unzip_shapenet_succ')
    if not os.path.exists(flag_file):
        print('-> Unzip ShapeNetCore.v1.zip.')
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(file_folder)
        create_flag_file(flag_file)

    folder = os.path.join(file_folder, 'ShapeNetCore.v1')
    flag_file = os.path.join(root_folder, 'flags/unzip_shapenet_all_succ')
    if not os.path.exists(flag_file):
        print('-> Unzip all zip files in ShapeNetCore.v1.')
        filenames = os.listdir(folder)
        for filename in filenames:
            if filename.endswith('.zip'):
                print('-    Unzip %s' % filename)
                zipname = os.path.join(folder, filename)
                with zipfile.ZipFile(zipname, 'r') as zip_ref:
                    zip_ref.extractall(folder)
                os.remove(zipname)
        create_flag_file(flag_file)


def download_filelist():
    r''' Downloads the filelists used for learning.
    '''

    flag_file = os.path.join(root_folder, 'flags/download_filelist_succ')
    if not os.path.exists(flag_file):
        print('-> Download the filelist.')
        url = 'https://www.dropbox.com/s/4jvam486l8961t7/shapenet.filelist.zip?dl=1'
        filename = os.path.join(root_folder, 'filelist.zip')
        wget.download(url, filename, bar=None)

        folder = os.path.join(root_folder, 'filelist')
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(path=folder)
        # os.remove(filename)
        create_flag_file(flag_file)


def run_mesh2sdf():
    r''' Converts the meshes from ShapeNet to SDFs and manifold meshes.
    '''

    print('-> Run mesh2sdf.')
    mesh_scale = 0.8
    filenames = get_filenames('all.txt')
    for i in tqdm(range(args.start, args.end), ncols=80):
        filename = filenames[i]
        filename_raw = os.path.join(
                file_folder, 'ShapeNetCore.v1', filename, 'model.obj')
        filename_obj = os.path.join(root_folder, 'mesh', filename + '.obj')
        filename_box = os.path.join(root_folder, 'bbox', filename + '.npz')
        filename_npy = os.path.join(root_folder, 'sdf', filename + '.npy')
        check_folder([filename_obj, filename_box, filename_npy])
        if os.path.exists(filename_obj): continue

        # load the raw mesh
        mesh = trimesh.load(filename_raw, force='mesh')

        # rescale mesh to [-1, 1] for mesh2sdf, note the factor **mesh_scale**
        vertices = mesh.vertices
        bbmin, bbmax = vertices.min(0), vertices.max(0)
        center = (bbmin + bbmax) * 0.5
        scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
        vertices = (vertices - center) * scale

        # run mesh2sdf
        sdf, mesh_new = mesh2sdf.compute(vertices, mesh.faces, size, fix=True,
                                                                         level=level, return_mesh=True)
        mesh_new.vertices = mesh_new.vertices * shape_scale

        # save
        np.savez(filename_box, bbmax=bbmax, bbmin=bbmin, mul=mesh_scale)
        np.save(filename_npy, sdf)
        mesh_new.export(filename_obj)

def run_mesh2sdf_mp():
    r''' Converts the meshes from ShapeNet to SDFs and manifold meshes.
        '''

    num_processes = 100
    num_meshes = args.end
    mesh_per_process = num_meshes // num_processes + 1

    print('-> Run mesh2sdf.')
    mesh_scale = 0.8
    filenames = get_filenames('all.txt')

    def process(process_id):
        for i in tqdm(range(process_id * mesh_per_process, (process_id + 1)* mesh_per_process), ncols=80):
            if i >= num_meshes: break
            filename = filenames[i]
            filename_raw = os.path.join(
                    file_folder, 'ShapeNetCore.v1', filename, 'model.obj')
            filename_obj = os.path.join(root_folder, 'mesh', filename + '.obj')
            filename_box = os.path.join(root_folder, 'bbox', filename + '.npz')
            filename_npy = os.path.join(root_folder, 'sdf', filename + '.npy')
            check_folder([filename_obj, filename_box, filename_npy])
            if os.path.exists(filename_obj): continue

            # load the raw mesh
            mesh = trimesh.load(filename_raw, force='mesh')

            # rescale mesh to [-1, 1] for mesh2sdf, note the factor **mesh_scale**
            vertices = mesh.vertices
            bbmin, bbmax = vertices.min(0), vertices.max(0)
            center = (bbmin + bbmax) * 0.5
            scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
            vertices = (vertices - center) * scale

            # run mesh2sdf
            sdf, mesh_new = mesh2sdf.compute(vertices, mesh.faces, size, fix=True,
                                                                            level=level, return_mesh=True)
            mesh_new.vertices = mesh_new.vertices * shape_scale

            # save
            np.savez(filename_box, bbmax=bbmax, bbmin=bbmin, mul=mesh_scale)
            np.save(filename_npy, sdf)
            mesh_new.export(filename_obj)

    processes = [mp.Process(target=process, args=[pid]) for pid in range(num_processes)]
    for p in processes:
            p.start()
    for p in processes:
            p.join()

def unzip_shapenet():
    r''' Unzip the ShapeNetCore.v1
    '''

    # filename = os.path.join(root_folder, 'ShapeNetCore.v1.zip')
    filename = os.path.join(file_folder, 'ShapeNetCore.v1.zip')
    flag_file = os.path.join(root_folder, 'flags/unzip_shapenet_succ')
    if not os.path.exists(flag_file):
        print('-> Unzip ShapeNetCore.v1.zip.')
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(file_folder)
        create_flag_file(flag_file)

    folder = os.path.join(file_folder, 'ShapeNetCore.v1')
    flag_file = os.path.join(root_folder, 'flags/unzip_shapenet_all_succ')
    if not os.path.exists(flag_file):
        print('-> Unzip all zip files in ShapeNetCore.v1.')
        filenames = os.listdir(folder)
        for filename in filenames:
            if filename.endswith('.zip'):
                print('-    Unzip %s' % filename)
                zipname = os.path.join(folder, filename)
                with zipfile.ZipFile(zipname, 'r') as zip_ref:
                    zip_ref.extractall(folder)
                os.remove(zipname)
        create_flag_file(flag_file)

def sample_pts_from_mesh():
    r''' Samples 10k points with normals from the ground-truth meshes.
    '''

    print('-> Run sample_pts_from_mesh.')
    num_samples = 100000
    mesh_folder = os.path.join(root_folder, 'mesh')
    output_folder = os.path.join(root_folder, 'dataset')
    filenames = get_filenames('all.txt')
    for i in tqdm(range(args.start, args.end), ncols=80):
        filename = filenames[i]
        filename_obj = os.path.join(mesh_folder, filename + '.obj')
        filename_pts = os.path.join(output_folder, filename, 'pointcloud.npz')
        check_folder([filename_pts])
        if os.path.exists(filename_pts): continue

        # sample points
        mesh = trimesh.load(filename_obj, force='mesh')
        points, idx = trimesh.sample.sample_surface(mesh, num_samples)
        normals = mesh.face_normals[idx]

        # save points
        np.savez(filename_pts, points=points.astype(np.float16),
                         normals=normals.astype(np.float16))
        

def sample_sdf():
    r''' Samples ground-truth SDF values for training.
    '''

    # constants
    depth, full_depth = 6, 4
    sample_num = 4    # number of samples in each octree node 也就是文中说的在每个八叉树的节点，采4个点并计算对应的sdf值。
    grid = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                                    [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

    print('-> Sample SDFs from the ground truth.')
    filenames = get_filenames('all.txt')
    for i in tqdm(range(args.start, args.end), ncols=80):
        filename = filenames[i]
        dataset_folder = os.path.join(root_folder, 'dataset')
        filename_sdf = os.path.join(root_folder, 'sdf', filename + '.npy')
        filename_pts = os.path.join(dataset_folder, filename, 'pointcloud.npz')
        filename_out = os.path.join(dataset_folder, filename, 'sdf.npz')
        if os.path.exists(filename_out): continue

        # load data
        pts = np.load(filename_pts)
        sdf = np.load(filename_sdf)
        sdf = torch.from_numpy(sdf)
        points = pts['points'].astype(np.float32)
        normals = pts['normals'].astype(np.float32)
        points = points / shape_scale    # rescale points to [-1, 1]

        # build octree
        points = ocnn.octree.Points(torch.from_numpy(points),torch.from_numpy(normals))
        octree = ocnn.octree.Octree(depth = depth, full_depth = full_depth)
        octree.build_octree(points)

        # sample points and grads according to the xyz
        xyzs, grads, sdfs = [], [], []
        for d in range(full_depth, depth + 1):
            xyzb = octree.xyzb(d)
            x,y,z,b = xyzb
            xyz = torch.stack((x,y,z),dim=1).float()

            # sample k points in each octree node
            xyz = xyz.unsqueeze(1) + torch.rand(xyz.shape[0], sample_num, 3)
            xyz = xyz.view(-1, 3)                                    # (N, 3)
            xyz = xyz * (size / 2 ** d)                        # normalize to [0, 2^sdf_depth] 相当于将坐标放大到[0,128]，128是sdf采样的分辨率
            xyz = xyz[(xyz < 127).all(dim=1)]            # remove out-of-bound points
            xyzs.append(xyz)

            # interpolate the sdf values
            xyzi = torch.floor(xyz)                                # the integer part (N, 3)
            corners = xyzi.unsqueeze(1) + grid         # (N, 8, 3)
            coordsf = xyz.unsqueeze(1) - corners     # (N, 8, 3), in [-1.0, 1.0]
            weights = (1 - coordsf.abs()).prod(dim=-1)    # (N, 8, 1)
            corners = corners.long().view(-1, 3)
            x, y, z = corners[:, 0], corners[:, 1], corners[:, 2]
            s = sdf[x, y, z].view(-1, 8)
            sw = torch.sum(s * weights, dim=1)
            sdfs.append(sw)

            # calc the gradient
            gx = s[:, 4] - s[:, 0] + s[:, 5] - s[:, 1] + \
                     s[:, 6] - s[:, 2] + s[:, 7] - s[:, 3]    # noqa
            gy = s[:, 2] - s[:, 0] + s[:, 3] - s[:, 1] + \
                     s[:, 6] - s[:, 4] + s[:, 7] - s[:, 5]    # noqa
            gz = s[:, 1] - s[:, 0] + s[:, 3] - s[:, 2] + \
                     s[:, 5] - s[:, 4] + s[:, 7] - s[:, 6]    # noqa
            grad = torch.stack([gx, gy, gz], dim=-1)
            norm = torch.sqrt(torch.sum(grad ** 2, dim=-1, keepdims=True))
            grad = grad / (norm + 1.0e-8)
            grads.append(grad)

        # concat the results
        xyzs = torch.cat(xyzs, dim=0).numpy()
        points = (xyzs / 64 - 1).astype(np.float16) * shape_scale    # 这里的points是sdf采样点的points，然后继续缩放到[-0.5, 0.5], 真的搞不懂为什么非要加这个0.5的shape_scale转来转去的，有啥意义。
        grads = torch.cat(grads, dim=0).numpy().astype(np.float16)
        sdfs = torch.cat(sdfs, dim=0).numpy().astype(np.float16)     # 这里的sdf还是跟之前一样，都是在[-1, 1]之间

        # save results
        # points = (points * args.scale).astype(np.float16)    # in [-scale, scale]
        np.savez(filename_out, points=points, grad=grads, sdf=sdfs)    # 也就是说sdf的值是在[-1,1]的尺度上，但是point的坐标在[-0.5, 0.5]


def sample_occu():
    r''' Samples occupancy values for evaluating the IoU following ConvONet.
    '''

    num_samples = 100000
    grid = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                                     [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

    # filenames = get_filenames('all.txt')
    filenames = get_filenames('test.txt') + get_filenames('test_unseen5.txt')
    for filename in tqdm(filenames, ncols=80):
        filename_sdf = os.path.join(root_folder, 'sdf', filename + '.npy')
        filename_occu = os.path.join(root_folder, 'dataset', filename, 'points')
        if os.path.exists(filename_occu) or (not os.path.exists(filename_sdf)):
            continue

        sdf = np.load(filename_sdf)
        factor = 127.0 / 128.0    # make sure the interpolation is well defined
        points_uniform = np.random.rand(num_samples, 3) * factor    # in [0, 1)
        points = (points_uniform - 0.5) * (2 * shape_scale)             # !!! rescale
        points = points.astype(np.float16)

        # interpolate the sdf values
        xyz = points_uniform * 128                                             # in [0, 127)
        xyzi = np.floor(xyz)                                                         # the integer part (N, 3)
        corners = np.expand_dims(xyzi, 1) + grid                 # (N, 8, 3)
        coordsf = np.expand_dims(xyz, 1) - corners             # (N, 8, 3), in [-1.0, 1.0]
        weights = np.prod(1 - np.abs(coordsf), axis=-1)    # (N, 8)

        corners = np.reshape(corners.astype(np.int64), (-1, 3))
        x, y, z = corners[:, 0], corners[:, 1], corners[:, 2]
        values = np.reshape(sdf[x, y, z], (-1, 8))
        value = np.sum(values * weights, axis=1)
        occu = value < 0
        occu = np.packbits(occu)

        # save
        np.savez(filename_occu, points=points, occupancies=occu)


def generate_test_points():
    r''' Generates points in `ply` format for testing.
    '''

    noise_std = 0.005
    point_sample_num = 3000
    # filenames = get_filenames('all.txt')
    filenames = get_filenames('test.txt') + get_filenames('test_unseen5.txt')
    for filename in tqdm(filenames, ncols=80):
        filename_pts = os.path.join(
                root_folder, 'dataset', filename, 'pointcloud.npz')
        filename_ply = os.path.join(
                root_folder, 'test.input', filename + '.ply')
        if not os.path.exists(filename_pts): continue
        check_folder([filename_ply])

        # sample points
        pts = np.load(filename_pts)
        points = pts['points'].astype(np.float32)
        noise = noise_std * np.random.randn(point_sample_num, 3)
        rand_idx = np.random.choice(points.shape[0], size=point_sample_num)
        points_noise = points[rand_idx] + noise

        # save ply
        vertices = []
        py_types = (float, float, float)
        npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        for idx in range(points_noise.shape[0]):
            vertices.append(
                    tuple(dtype(d) for dtype, d in zip(py_types, points_noise[idx])))
        structured_array = np.array(vertices, dtype=npy_types)
        el = PlyElement.describe(structured_array, 'vertex')
        PlyData([el]).write(filename_ply)


def download_dataset():
    r''' Directly downloads the dataset.
    '''

    flag_file = os.path.join(root_folder, 'flags/download_dataset_succ')
    if not os.path.exists(flag_file):
        print('-> Download the dataset.')
        url = 'https://www.dropbox.com/s/mc3lrwqpmnfq3j8/shapenet.dataset.zip?dl=1'
        filename = os.path.join(root_folder, 'shapenet.dataset.zip')
        wget.download(url, filename)

        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(path=root_folder)
        # os.remove(filename)
        create_flag_file(flag_file)


def generate_dataset_unseen5():
    r'''Creates the unseen5 dataset
    '''

    dataset_folder = os.path.join(root_folder, 'dataset')
    unseen5_folder = os.path.join(root_folder, 'dataset.unseen5')
    if not os.path.exists(unseen5_folder):
        os.makedirs(unseen5_folder)
    for folder in ['02808440', '02773838', '02818832', '02876657', '03938244']:
        curr_folder = os.path.join(dataset_folder, folder)
        if os.path.exists(curr_folder):
            shutil.move(os.path.join(dataset_folder, folder), unseen5_folder)


def copy_convonet_filelists():
    r''' Copies the filelist of ConvONet to the datasets, which are needed when
     calculating the evaluation metrics.
     '''

    with open(os.path.join(root_folder, 'filelist/lists.txt'), 'r') as fid:
        lines = fid.readlines()
    filenames = [line.split()[0] for line in lines]
    filelist_folder = os.path.join(root_folder, 'filelist')
    for filename in filenames:
        src_name = os.path.join(filelist_folder, filename)
        des_name = src_name.replace('filelist/convonet.filelist', 'dataset')    \
                                             .replace('filelist/unseen5.filelist', 'dataset.unseen5')
        if not os.path.exists(des_name):
            shutil.copy(src_name, des_name)


def convert_mesh_to_sdf():
    unzip_shapenet()
    download_filelist()
    run_mesh2sdf()


def generate_dataset():
    sample_pts_from_mesh()
    sample_sdf()
    sample_occu()
    generate_test_points()
    generate_dataset_unseen5()
    copy_convonet_filelists()



if __name__ == '__main__':
    eval('%s()' % args.run)
