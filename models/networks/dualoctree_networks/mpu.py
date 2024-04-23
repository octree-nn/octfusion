# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

from ocnn.octree import xyz2key
from ocnn.utils import cumsum
import torch
import torch.nn

from .utils.spmm import spmm, modulated_spmm

kNN = 8


class ABS(torch.autograd.Function):
  '''The derivative of torch.abs on `0` is `0`, and in this implementation, we
  modified it to `1`
  '''
  @staticmethod
  def forward(ctx, input):
    ctx.save_for_backward(input)
    return input.abs()

  @staticmethod
  def backward(ctx, grad_in):
    input, = ctx.saved_tensors
    sign = input < 0
    grad_out = grad_in * (-2.0 * sign.to(input.dtype) + 1.0)
    return grad_out


def linear_basis(x):
  return 1.0 - ABS.apply(x)


def get_linear_mask(dim=3):
  mask = torch.tensor([0, 1], dtype=torch.float32)
  mask = torch.meshgrid([mask]*dim)
  mask = torch.stack(mask, -1).view(-1, dim)
  return mask

#  mask = tensor([[0., 0., 0.],
#         [0., 0., 1.],
#         [0., 1., 0.],
#         [0., 1., 1.],
#         [1., 0., 0.],
#         [1., 0., 1.],
#         [1., 1., 0.],
#         [1., 1., 1.]])


def octree_linear_pts(octree, depth, pts):
  # get neigh coordinates
  scale = 2 ** depth   ## 在特定深度depth下的scale为2**depth
  mask = get_linear_mask(dim=3).to(pts.device)
  xyzf, ids = torch.split(pts, [3, 1], 1)  # 将pos[N, 4]分解为前三个（坐标）xyzf和最后一个（batch_idx）ids
  xyzf = (xyzf + 1.0) * (scale / 2.0)    # [-1, 1] -> [0, scale] 将xyz坐标放缩到[0, scale]
  xyzf = xyzf - 0.5                      # the code is defined on the center
  xyzi = torch.floor(xyzf).detach()      # the integer part (N, 3), use floor
  corners = xyzi.unsqueeze(1) + mask     # (N, 8, 3), 得到这N个点的周围8个corner的坐标，也就是 [N,8,3]，这里corner的坐标都是整数，在[0,scale]范围内
  coordsf = xyzf.unsqueeze(1) - corners  # (N, 8, 3), in [-1.0, 1.0]

  # coorers -> key
  ids = ids.detach().repeat(1, kNN).unsqueeze(-1)       # (N, 8, 1)
  key = torch.cat([corners, ids], dim=-1).view(-1, 4).short()  # (N*8, 4)
  key = xyz2key(x = key[:,0], y = key[:,1], z = key[:,2], b = key[:,3]).long()  # (N*8, )
  idx = octree.search_key(key, depth)    # (N*8, )
  # key = ocnn.octree_encode_key(key).long()                     # (N*8, )
  # idx = ocnn.octree_search_key(key, octree, depth, key_is_xyz=True)

  # corners -> flags
  valid = torch.logical_and(corners > -1, corners < scale)  # out-of-bound
  valid = torch.all(valid, dim=-1).view(-1)
  flgs = torch.logical_and(idx > -1, valid)

  # remove invalid pts
  idx = idx[flgs].long()               # (N*8, )   -> (N', )
  coordsf = coordsf.view(-1, 3)[flgs]  # (N, 8, 3) -> (N', 3)

  # bspline weights
  weights = linear_basis(coordsf)                     # (N', 3)
  weights = torch.prod(weights, axis=-1).view(-1)     # (N', )
  # Here, the scale factor `2**(depth - 6)` is used to emphasize high-resolution
  # basis functions. Tune this factor further if needed! !!! NOTE !!!
  # weights = weights * 2**(depth - 6)                 # used for shapenet
  weights = weights * (depth**2 / 50)                  # testing

  # rescale back the original scale
  # After recaling, the coordsf is in the same scale as pts
  coordsf = coordsf * (2.0 / scale)   # [-1.0, 1.0] -> [-2.0/scale, 2.0/scale] 这一步相当于，把[0,scale]的坐标重新缩小到[-1,1]的尺度上
  return {'idx': idx, 'xyzf': coordsf, 'weights': weights, 'flgs': flgs}


def get_linear_pred(pts, octree, shape_code, neighs, depth_start, depth_end):
  npt = pts.size(0)
  indices, weights, xyzfs = [], [], []
  nnum = octree.nnum
  nnum_cum = cumsum(nnum, dim=0, exclusive=True)
  # nnum_cum = ocnn.octree_property(octree, 'node_num_cum')
  ids = torch.arange(npt, device=pts.device, dtype=torch.long)
  ids = ids.unsqueeze(-1).repeat(1, kNN).view(-1)
  for d in range(depth_start, depth_end+1):
    neighd = neighs[d]
    idxd = neighd['idx']
    xyzfd = neighd['xyzf']
    weightd = neighd['weights']
    valid = neighd['flgs']
    idsd = ids[valid]

    if d < depth_end:
      child = octree.children[d]
      leaf = child[idxd] < 0  # keep only leaf nodes
      idsd, idxd, weightd, xyzfd = idsd[leaf], idxd[leaf], weightd[leaf], xyzfd[leaf]

    idxd = idxd + (nnum_cum[d] - nnum_cum[depth_start])
    indices.append(torch.stack([idsd, idxd], dim=1))
    weights.append(weightd)
    xyzfs.append(xyzfd)

  indices = torch.cat(indices, dim=0).t()
  weights = torch.cat(weights, dim=0)
  xyzfs = torch.cat(xyzfs, dim=0)

  code_num = shape_code.size(0)
  output = modulated_spmm(indices, weights, npt, code_num, shape_code, xyzfs)
  norm = spmm(indices, weights, npt, code_num, torch.ones(code_num, 1).cuda()) # 这里norm的维度为[N,]
  output = torch.div(output, norm + 1e-8).squeeze()  # 这里output的维度为[N, 1]（也就是每个查询点的sdf值）

  # whether the point has affected by the octree node in depth layer
  mask = neighs[depth_end]['flgs'].view(-1, kNN).any(axis=-1)
  return output, mask


class NeuralMPU:
  def __init__(self, full_depth, depth_stop, depth):
    self.full_depth = full_depth
    self.depth_stop = depth_stop
    self.depth = depth

  def __call__(self, pos, reg_voxs, octree_out): # reg_voxs就是dual octree的每个节点中存储的vector，其维度为4
    mpus = dict()
    neighs = dict()
    for d in range(self.full_depth, self.depth+1):
      neighs[d] = octree_linear_pts(octree_out, d, pos)

    for d in range(self.depth_stop, self.depth+1):
      fval, flgs = get_linear_pred(
          pos, octree_out, reg_voxs[d], neighs, self.full_depth, d)
      mpus[d] = (fval, flgs)
    return mpus