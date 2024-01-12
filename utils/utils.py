from contextlib import contextmanager, ExitStack
import torch
from torch import nn
import numpy as np
import os
from scipy.spatial.transform import Rotation
import trimesh
from skimage.measure import marching_cubes
import argparse


def get_data_class_label(data_class):
    if data_class == "chairs":
      label = "03001627"
    elif data_class == "planes":
      label = "02691156"
    elif data_class == "cars":
      label = "02958343"
    elif data_class == "tables":
      label = "04379243"
    elif data_class == "rifles":
      label = "04090263"
    else:
      raise NotImplementedError
    return label

def get_sample_number_for_metric(data_class, metrics= "fid"):
    assert metrics in ["fid", "cov", "fpd"]
    if metrics == "fid" or metrics == "fpd":
      if data_class == "chairs":
        n_sam = 4744
      elif data_class == "planes":
        n_sam = 2831
      elif data_class == "cars":
        n_sam = 5247
      elif data_class == "tables":
        n_sam = 5956
      elif data_class == "rifles":
        n_sam = 1660
      else:
        raise NotImplementedError
    elif metrics == "cov":
      if data_class == "chairs":
        n_sam = 1356 * 5
      elif data_class == "planes":
        n_sam = 809 * 5
      elif data_class == "cars":
        n_sam = 1500 * 5
      elif data_class == "tables":
        n_sam = 1702 * 5
      elif data_class == "rifles":
        n_sam = 475 * 5
      else:
        raise NotImplementedError
    else:
      raise NotImplementedError  
    return n_sam



def scale_to_unit_sphere(mesh, evaluate_metric = False):
  if isinstance(mesh, trimesh.Scene):
    mesh = mesh.dump().sum()

  vertices = mesh.vertices - mesh.bounding_box.centroid
  distances = np.linalg.norm(vertices, axis=1)
  vertices /= np.max(distances)
  if evaluate_metric:
        vertices /= 2
  return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


def shapenet_v2_to_v1_orientation(mesh):
    mesh.apply_transform(get_rotation_matrix(-90, 'y'))
    # mesh.invert()
    return mesh

def get_grid_normal(grid, bouding_box_length=2):

  grid_res = grid.shape[-1]
  n = grid_res - 1
  voxel_size = bouding_box_length/n

  X_1 = torch.cat((grid[:, :, 1:, :, :], (3 * grid[:, :, n, :, :] - 3 *
                  grid[:, :, n-1, :, :] + grid[:, :, n-2, :, :]).unsqueeze_(2)), 2)
  X_2 = torch.cat(((-3 * grid[:, :, 1, :, :] + 3 * grid[:, :, 0, :, :] +
                  grid[:, :, 2, :, :]).unsqueeze_(2), grid[:, :, :n, :, :]), 2)
  grid_normal_x = (X_1 - X_2) / (2 * voxel_size)

  Y_1 = torch.cat((grid[:, :, :, 1:, :], (3 * grid[:, :, :, n, :] - 3 *
                  grid[:, :, :, n-1, :] + grid[:, :, :, n-2, :]).unsqueeze_(3)), 3)
  Y_2 = torch.cat(((-3 * grid[:, :, :, 1, :] + 3 * grid[:, :, :, 0, :] +
                  grid[:, :, :, 2, :]).unsqueeze_(3), grid[:, :, :, :n, :]), 3)
  grid_normal_y = (Y_1 - Y_2) / (2 * voxel_size)
  
  Z_1 = torch.cat((grid[:, :, :, :, 1:], (3 * grid[:, :, :, :, n] - 3 *
                  grid[:, :, :, :, n-1] + grid[:, :, :, :, n-2]).unsqueeze_(4)), 4)
  Z_2 = torch.cat(((-3 * grid[:, :, :, :, 1] + 3 * grid[:, :, :, :, 0] +
                  grid[:, :, :, :, 2]).unsqueeze_(4), grid[:, :, :, :, :n]), 4)
  grid_normal_z = (Z_1 - Z_2) / (2 * voxel_size)

  return torch.cat((grid_normal_x, grid_normal_y, grid_normal_z), 1)


def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def cast_tuple(val, repeat=1):
  return val if isinstance(val, tuple) else ((val,) * repeat)


def run(cmd,verbose=True):
  if verbose:
    print(cmd)
  os.system(cmd)


def process_mesh(mesh, sample_number=2048):
  mesh = scale_to_unit_sphere(mesh=mesh, evaluate_metric=True)
  return mesh.sample(sample_number)


def points_gradient(inputs, outputs):
  d_points = torch.ones_like(
      outputs, requires_grad=False, device=outputs.device)
  points_grad = torch.autograd.grad(
      outputs=outputs,
      inputs=inputs,
      grad_outputs=d_points,
      create_graph=True,
      retain_graph=True,
      only_inputs=True)[0]
  return points_grad


def get_voxel_coordinates(resolution=32, size=1, center=0, device=None):
  if type(center) == int:
    center = (center, center, center)
  points = np.meshgrid(
      np.linspace(center[0] - size, center[0] + size, resolution),
      np.linspace(center[1] - size, center[1] + size, resolution),
      np.linspace(center[2] - size, center[2] + size, resolution)
  )
  points = np.stack(points)
  points = np.swapaxes(points, 1, 2)
  points = points.reshape(3, -1).transpose()
  if device is not None:
    return torch.tensor(points, dtype=torch.float32, device=device)
  else:
    return torch.tensor(points, dtype=torch.float32)


def process_sdf(volume, level=0, padding=True, spacing=None, offset=-1,normalize=False):
  try:
    if padding:
      volume = np.pad(volume, 1, mode='constant', constant_values=1)
    if spacing is None:
      spacing = 2/(volume.shape[-1] - 1)
    vertices, faces, normals, _ = marching_cubes(
        volume, level=level, spacing=(spacing, spacing, spacing))
    if offset is not None:
      vertices += offset
    if normalize:
      return scale_to_unit_sphere(trimesh.Trimesh(
          vertices=vertices, faces=faces, vertex_normals=normals))   
    else:
      return trimesh.Trimesh(
          vertices=vertices, faces=faces, vertex_normals=normals)
  except Exception as e:
    print(str(e))
    return None


def ensure_directory(directory):
  if not os.path.exists(directory):
    os.makedirs(directory, exist_ok=True)


class NanException(Exception):
  pass


def get_rotation_matrix(angle, axis='y'):
    rotation = Rotation.from_euler(axis, angle, degrees=True)
    matrix = np.identity(4)
    matrix[:3, :3] = rotation.as_matrix()
    return matrix

def get_pc_rotation_matrix(angle, axis='y'):
    rotation = Rotation.from_euler(axis, angle, degrees=True)
    return rotation.as_matrix()
    # return matrix
    

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cast_list(el):
  return el if isinstance(el, list) else [el]


def exists(val):
  return val is not None


@contextmanager
def null_context():
  yield


def combine_contexts(contexts):
  @contextmanager
  def multi_contexts():
    with ExitStack() as stack:
      yield [stack.enter_context(ctx()) for ctx in contexts]

  return multi_contexts


def default(value, d):
  return value if exists(value) else d


def cycle(iterable):
  while True:
    for i in iterable:
      yield i


def cast_list(el):
  return el if isinstance(el, list) else [el]


def is_empty(t):
  if isinstance(t, torch.Tensor):
    return t.nelement() == 0
  return not exists(t)


def raise_if_nan(t):
  if torch.isnan(t):
    raise NanException


def noise(batch_size, latent_dim, device):
  return torch.randn(batch_size, latent_dim).cuda(device)


def noise_list(batch_size, layers, latent_dim, device):
  return [(noise(batch_size, latent_dim, device), layers)]


def mixed_list(batch_size, layers, latent_dim, device):
  tt = int(torch.rand(()).numpy() * layers)
  return noise_list(batch_size, tt, latent_dim, device) + noise_list(batch_size, layers - tt, latent_dim, device)


def latent_to_w(style_vectorizer, latent_descr):
  return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]


def image_noise(n, im_size, device):
  return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).cuda(device)


def volume_noise(n, vol_size, device):
  return torch.FloatTensor(n, vol_size, vol_size, vol_size, 1).uniform_(0., 1.).cuda(device)


def leaky_relu(p=0.2,):
  return nn.LeakyReLU(p, inplace=True)


def evaluate_in_chunks(max_batch_size, model, *args):
  split_args = list(
      zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
  chunked_outputs = [model(*i) for i in split_args]
  if len(chunked_outputs) == 1:
    return chunked_outputs[0]
  return torch.cat(chunked_outputs, dim=0)


def styles_def_to_tensor(styles_def):
  return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)


def set_requires_grad(model, bool):
  for p in model.parameters():
    p.requires_grad = bool

def linear_slerp(val, low, high):
  val = val.squeeze()
  return (1-val)*low + val * high


