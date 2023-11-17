# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
# import torch_sparse
import numpy as np

from ocnn.octree import key2xyz, xyz2key
from ocnn.utils import cumsum
from ocnn.modules import InputFeature


class DualOctree:

  def __init__(self, octree):
    # prime octree
    self.octree = octree
    self.device = octree.device
    self.depth = octree.depth
    self.full_depth = octree.full_depth
    self.batch_size = octree.batch_size

    # node numbers
    self.nnum = octree.nnum
    self.nenum = octree.nnum_nempty
    self.ncum = cumsum(self.nnum, dim=0, exclusive=True)
    # self.nnum = ocnn.octree_property(octree, 'node_num')
    # self.nenum = ocnn.octree_property(octree, 'node_num_ne')
    # self.ncum = ocnn.octree_property(octree, 'node_num_cum')
    self.lnum = self.nnum - self.nenum  # leaf node numbers

    # node properties
    # xyzi = ocnn.octree_property(octree, 'xyz')
    # self.xyzi = ocnn.octree_decode_key(xyzi)
    # self.xyz = self.xyzi[:, :3]
    # self.batch = self.xyzi[:, 3]
    self.node_depth = self._node_depth()
    self.child = torch.cat([child for child in octree.children if child != None])
    # self.child = torch.cat(octree.children)
    self.key = torch.cat([key for key in octree.keys if key != None])
    # self.key = torch.cat(octree.keys)
    self.keyd = self.key | (self.node_depth << 58)
    xyzi = key2xyz(self.key)
    x,y,z,i = xyzi
    self.xyzi = torch.stack((x,y,z,i),dim=1)
    self.xyz = self.xyzi[:, :3]
    self.batch = self.xyzi[:, 3]

    # build lookup tables
    self._lookup_table()

    # build dual graph
    self._graph = [dict()] * (self.depth + 1)   # the internal graph
    self.graph = [dict()] * (self.depth + 1)    # the output graph
    self.build_dual_graph()

    self.batch_id_dict = {}
    self.calc_batch_id()
    self.total_num = len(self.batch_id_dict[self.depth])

  def calc_batch_id(self):
    leaf_batch_id = torch.tensor([]).long().to(self.device)
    for i in range(self.full_depth, self.depth + 1):
      batch_id = self.octree.batch_id(i, nempty = False)
      if i==self.full_depth:
        last_batch_id = torch.tensor([]).long().to(self.device)
      else:
        empty_mask = self.octree.children[i-1] < 0
        key = self.octree.keys[i-1]
        key = key[empty_mask]
        last_batch_id = key >> 48

      leaf_batch_id = torch.cat([leaf_batch_id, last_batch_id], dim = 0)
      batch_id = torch.cat([leaf_batch_id, batch_id], dim = 0)
      self.batch_id_dict[i] = batch_id

  def batch_id(self, depth):
    return self.batch_id_dict[depth]

  # def batch_id(self, depth, nempty: bool = False):
  #   batch_id = self.octree.batch_id(depth, nempty)

  #   if self.octree.full_depth == depth:
  #     return batch_id

  #   keys = []

  #   for i in range(self.octree.full_depth, depth):
  #     empty_mask = self.octree.children[i] < 0
  #     key = self.octree.keys[i]
  #     key = key[empty_mask]
  #     keys.append(key)

  #   keys = torch.cat(keys)
  #   leaf_batch_id = keys >> 48

  #   batch_id = torch.cat([leaf_batch_id, batch_id],dim = 0)

  #   return batch_id

  def _lookup_table(self):
    self.ngh = torch.tensor(
        [[0, 0, 1], [0, 0, -1],       # up, down
         [0, 1, 0], [0, -1, 0],       # right, left
         [1, 0, 0], [-1, 0, 0]],      # front, back
        dtype=torch.int16, device=self.device)
    self.dir_table = torch.tensor(
        [[1, 3, 5, 7], [0, 2, 4, 6],   # up, down
         [2, 3, 6, 7], [0, 1, 4, 5],   # right, left
         [4, 5, 6, 7], [0, 1, 2, 3]],  # front, back
        dtype=torch.int64, device=self.device)
    self.dir_type = torch.tensor(
        [0, 1, 2, 3, 4, 5],
        dtype=torch.int64, device=self.device)
    self.remap = torch.tensor(
        [1, 0, 3, 2, 5, 4],
        dtype=torch.int64, device=self.device)
    self.inter_row = torch.tensor(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,
         4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7],
        dtype=torch.int64, device=self.device)
    self.inter_col = torch.tensor(
        [1, 2, 4, 0, 3, 5, 0, 3, 6, 1, 2, 7,
         0, 5, 6, 1, 4, 7, 2, 4, 7, 3, 5, 6],
        dtype=torch.int64, device=self.device)
    self.inter_dir = torch.tensor(
        [0, 2, 4, 1, 2, 4, 3, 0, 4, 3, 1, 4,
         5, 0, 2, 5, 1, 2, 5, 3, 0, 5, 3, 1],
        dtype=torch.int64, device=self.device)

  def _node_depth(self):
    nd = [torch.tensor([d], dtype=torch.int64, device=self.device).expand(
          self.nnum[d]) for d in range(self.depth + 1)]
    return torch.cat(nd)

  def build_dual_graph(self):
    self._graph[self.full_depth] = self.dense_graph(self.full_depth)
    for d in range(self.full_depth + 1, self.depth + 1):
      self._graph[d] = self.sparse_graph(d, self._graph[d - 1])

  def dense_graph(self, depth=3):
    K = 6  # each node has at most K neighboring node
    bnd = 2 ** depth
    num = bnd ** 3

    ki = torch.arange(0, num, dtype=torch.int64, device=self.device)
    xi = key2xyz(ki, depth)
    xi = torch.stack(xi[:3],dim=1)
    xj = xi.unsqueeze(1) + self.ngh    # [N, K, 3]

    row = ki.unsqueeze(1).repeat(1, K).view(-1)
    zj = torch.zeros(num, K, 1, dtype=torch.int16, device=self.device)
    kj = torch.cat([xj, zj], dim=-1).view(-1, 4)
    # for full octree, the octree key is the index
    col = xyz2key(kj[:,0],kj[:,1],kj[:,2],kj[:,3],depth)

    valid = torch.logical_and(xj > -1, xj < bnd)  # out-of-bound
    valid = torch.all(valid, dim=-1).view(-1)
    row, col = row[valid], col[valid]

    edge_dir = self.dir_type.repeat(num)
    edge_dir = edge_dir[valid]

    # deal with batches
    dis = torch.arange(self.batch_size, dtype=torch.int64, device=self.device)
    dis = dis.unsqueeze(1) * num + self.ncum[depth]  # NOTE:add self.ncum[depth]
    row = row.unsqueeze(0) + dis
    col = col.unsqueeze(0) + dis
    edge_dir = edge_dir.unsqueeze(0) + torch.zeros_like(dis)
    # rowptr = torch.ops.torch_sparse.ind2ptr(row, num)
    return {'edge_idx': torch.stack([row.view(-1), col.view(-1)]),
            'edge_dir': edge_dir.view(-1)}

  def _internal_edges(self, nnum, dis=0):
    assert(nnum % 8 == 0)
    d = torch.arange(0, nnum / 8, dtype=torch.int64, device=self.device)
    d = torch.unsqueeze(d * 8 + dis, dim=1)
    row = self.inter_row.unsqueeze(0) + d
    col = self.inter_col.unsqueeze(0) + d
    edge_dir = self.inter_dir.unsqueeze(0) + torch.zeros_like(d)
    return row.view(-1), col.view(-1), edge_dir.view(-1)

  def relative_dir(self, vi, vj, depth, rescale=True):
    xi = self.xyz[vi]
    xj = self.xyz[vj]

    # get 6 neighborhoods of xi via `self.ngh`
    xn = xi.unsqueeze(1) + self.ngh

    # rescale the coord of xj
    scale = torch.ones_like(vj)
    if rescale:
      dj = self.node_depth[vj]
      scale = torch.pow(2.0, depth - dj)
      # torch._assert((scale > 1.0).all(), 'vj is larger than vi')
      xj = xj * scale.unsqueeze(-1)

    # inbox testing
    xj = xj.unsqueeze(1)
    scale = scale.view(-1, 1, 1)
    inbox = torch.logical_and(xn >= xj, xn < xj + scale.view(-1, 1, 1))
    inbox = torch.all(inbox, dim=-1)
    rel_dir = torch.argmax(inbox.byte(), dim=-1)
    return rel_dir

  def _node_property(self, prop, depth):
    return prop[self.ncum[depth]: self.ncum[depth] + self.nnum[depth]]

  def node_child(self, depth):
    return self._node_property(self.child, depth)

  def sparse_graph(self, depth, graph):
    # Add internal edges connecting sliding nodes.
    ncum_d = self.ncum[depth]  # NOTE: add ncum_d, i.e., self.ncum[depth]
    row_i, col_i, dir_i = self._internal_edges(self.nnum[depth], ncum_d)

    # mark invalid nodes of layer (depth-1)
    edge_idx, edge_dir = graph['edge_idx'], graph['edge_dir']
    row, col = edge_idx[0], edge_idx[1]
    valid_row = self.child[row] < 0
    valid_col = self.child[col] < 0
    invalid_row = torch.logical_not(valid_row)
    invalid_col = torch.logical_not(valid_col)
    valid_edges = torch.logical_and(valid_row, valid_col)
    invalid_row_vtx = torch.logical_and(invalid_row, valid_col)
    invalid_both_vtx = torch.logical_and(invalid_row, invalid_col)

    # deal with edges with invalid row vtx only
    vi, vj = row[invalid_row_vtx], col[invalid_row_vtx]
    rel_dir = self.relative_dir(vi, vj, depth - 1)
    row_o1 = self.child[vi].unsqueeze(1) * 8 + self.dir_table[rel_dir, :]
    row_o1 = row_o1.view(-1) + ncum_d  # NOTE: add ncum_d
    col_o1 = vj.unsqueeze(1).repeat(1, 4).view(-1)
    dir_o1 = rel_dir.unsqueeze(1).repeat(1, 4).view(-1)

    # deal with edges with 2 invalid nodes
    row_o2 = torch.tensor([], dtype=torch.int64, device=self.device)
    col_o2 = torch.tensor([], dtype=torch.int64, device=self.device)
    dir_o2 = torch.tensor([], dtype=torch.int64, device=self.device)
    if invalid_both_vtx.any():
      vi, vj = row[invalid_both_vtx], col[invalid_both_vtx]
      rel_dir = self.relative_dir(vi, vj, depth - 1, rescale=False)
      row_o2 = self.child[vi].unsqueeze(1) * 8 + self.dir_table[rel_dir, :]
      row_o2 = row_o2.view(-1) + ncum_d  # NOTE: add ncum_d
      dir_o2 = rel_dir.unsqueeze(1).repeat(1, 4).view(-1)
      rel_dir_col = self.remap[rel_dir]
      col_o2 = self.child[vj].unsqueeze(1) * 8 + self.dir_table[rel_dir_col, :]
      col_o2 = col_o2.view(-1) + ncum_d  # NOTE: add ncum_d

    # gather the results
    edge_idx = torch.stack([
        torch.cat([row[valid_edges], row_i, row_o1, col_o1, row_o2]),
        torch.cat([col[valid_edges], col_i, col_o1, row_o1, col_o2])])
    edge_dir = torch.cat([
        edge_dir[valid_edges], dir_i, dir_o1, self.remap[dir_o1], dir_o2])
    return {'edge_idx': edge_idx, 'edge_dir': edge_dir}

  def add_self_loops(self):
    for d in range(self.full_depth, self.depth + 1):
      edge_idx = self._graph[d]['edge_idx']
      edge_dir = self._graph[d]['edge_dir']
      row, col = edge_idx[0], edge_idx[1]
      unique_idx = torch.unique(row, sorted=True)
      dir_idx = torch.ones_like(unique_idx) * 6
      self.graph[d] = {'edge_idx': torch.stack([torch.cat([row, unique_idx]),
                                                torch.cat([col, unique_idx])]),
                       'edge_dir': torch.cat([edge_dir, dir_idx])}

  def calc_edge_type(self):
    dir_num = 7
    for d in range(self.full_depth, self.depth + 1):
      depth_num = d - self.full_depth + 1
      edge_idx = self._graph[d]['edge_idx']
      edge_dir = self._graph[d]['edge_dir']
      row, col = edge_idx[0], edge_idx[1]

      dr = self.node_depth[row] - self.full_depth
      dc = self.node_depth[col] - self.full_depth
      edge_type = (dr * depth_num + dc) * dir_num + edge_dir

      self.graph[d]['edge_type'] = edge_type

  def remap_node_idx(self):
    leaf_nodes = self.child < 0
    for d in range(self.full_depth, self.depth + 1):
      leaf_d = torch.ones(self.nnum[d], dtype=torch.bool, device=self.device)
      mask = torch.cat([leaf_nodes[:self.ncum[d]], leaf_d], dim=0)
      remap = torch.cumsum(mask.long(), dim=0) - 1
      self.graph[d]['edge_idx'] = remap[self.graph[d]['edge_idx']]

  def filter_multiple_level_edges(self):
    for d in range(self.full_depth, self.depth + 1):
      edge_idx = self.graph[d]['edge_idx']
      edge_dir = self.graph[d]['edge_dir']
      valid_edges = (self.node_depth[edge_idx] == d).all(dim=0)

      # filter edges
      edge_idx = edge_idx[:, valid_edges]
      edge_dir = edge_dir[valid_edges]

      self.graph[d] = {'edge_idx': edge_idx, 'edge_dir': edge_dir}

  def filter_coarse_to_fine_edges(self):
    for d in range(self.full_depth, self.depth + 1):
      edge_idx = self.graph[d]['edge_idx']
      edge_dir = self.graph[d]['edge_dir']

      edge_node_depth = self.node_depth[edge_idx]
      # the depth of sender nodes should be larger than receivers
      valid_edges = edge_node_depth[0] >= edge_node_depth[1]

      # filter edges
      edge_idx = edge_idx[:, valid_edges]
      edge_dir = edge_dir[valid_edges]

      self.graph[d] = {'edge_idx': edge_idx, 'edge_dir': edge_dir}

  def filter_crosslevel_edges(self):
    for d in range(self.full_depth, self.depth + 1):
      edge_idx = self.graph[d]['edge_idx']
      edge_dir = self.graph[d]['edge_dir']

      edge_node_depth = self.node_depth[edge_idx]
      valid_edges = edge_node_depth[0] == edge_node_depth[1]

      # filter edges
      edge_idx = edge_idx[:, valid_edges]
      edge_dir = edge_dir[valid_edges]

      self.graph[d] = {'edge_idx': edge_idx, 'edge_dir': edge_dir}

  def displace_edge_and_add_node_type(self):
    for d in range(self.full_depth, self.depth + 1):
      # displace edge index
      self.graph[d]['edge_idx'] -= self.ncum[d]

      # only one type of node
      zeros = torch.zeros(self.nnum[d], dtype=torch.long, device=self.device)
      self.graph[d]['node_type'] = zeros

      # used in skip connections
      self.graph[d]['keyd'] = self._node_property(self.keyd, d)

  def post_processing_for_ocnn(self):
    self.add_self_loops()
    self.filter_multiple_level_edges()
    self.displace_edge_and_add_node_type()
    self.sort_edges()

  def sort_edges(self):
    dir_num = 7
    for d in range(self.full_depth, self.depth + 1):
      edge_idx = self.graph[d]['edge_idx']
      edge_dir = self.graph[d]['edge_dir']

      edge_key = edge_idx[0] * dir_num + edge_dir
      sidx = torch.argsort(edge_key)
      self.graph[d]['edge_idx'] = edge_idx[:, sidx]
      self.graph[d]['edge_dir'] = edge_dir[sidx]

  def get_input_feature(self, all_leaf_nodes=True, feature = 'ND'):
    # the initial feature of leaf nodes in the layer self.depth
    octree_feature = InputFeature(feature = feature, nempty=False)
    data = octree_feature(self.octree)

    # data = ocnn.octree_property(self.octree, 'feature', self.depth)
    # data = data.squeeze(0).squeeze(-1).t()

    # the initial feature of leaf nodes in other layers
    if all_leaf_nodes:
      channel = data.shape[1]
      leaf_num = self.lnum[self.full_depth:self.depth].sum()
      zeros = torch.zeros(leaf_num, channel, device=self.device)

      # concat zero features with the initial features in layer depth
      data = torch.cat([zeros, data], dim=0)

    return data

  def add_node_keyd(self):
    keyd1, keyd2 = [], []
    for d in range(self.full_depth, self.depth + 1):
      keyd = self._node_property(self.keyd, d)
      leaf_mask = self._node_property(self.child, d) < 0
      keyd1.append(keyd[leaf_mask])
      keyd2.append(keyd)
      self.graph[d]['keyd'] = torch.cat(keyd1[:-1] + keyd2[-1:], dim=0)

  def add_node_xyzd(self):
    xyz1, xyz2 = [], []
    for d in range(self.full_depth, self.depth + 1):
      xyzd = self._node_property(self.xyz, d)
      xyzf = xyzd.float() / 2 ** d   # normalize to [0, 1]
      leaf_mask = self._node_property(self.child, d) < 0
      xyz1.append(xyzf[leaf_mask])
      xyz2.append(xyzf)
      self.graph[d]['xyz'] = torch.cat(xyz1[:-1] + xyz2[-1:], dim=0)

  def add_node_type(self):
    ntype1, ntype2 = [], []
    full_depth, depth = self.full_depth, self.depth
    for i, d in enumerate(range(full_depth, depth + 1)):
      ntype = d - full_depth
      ntype1.append(torch.ones(self.lnum[d], device=self.device) * ntype)
      ntype2.append(torch.ones(self.nnum[d], device=self.device) * ntype)
      node_type = torch.cat(ntype1[:i] + ntype2[i:i + 1], dim=0).long()
      self.graph[d]['node_type'] = node_type

  def add_node_mask(self):
    leaf_masks = []
    full_depth, depth = self.full_depth, self.depth
    for i, d in enumerate(range(full_depth, depth + 1)):
      mask1 = self._node_property(self.child, d) < 0
      mask2 = torch.ones(self.nnum[d], dtype=torch.bool, device=self.device)
      leaf_masks.append(mask1)
      self.graph[d]['node_mask'] = torch.cat(leaf_masks[:i] + [mask2], dim=0)

  def post_processing_for_docnn(self):
    self.add_self_loops()
    # The following 2 functions are only used in ablation study.
    # self.filter_coarse_to_fine_edges()
    # self.filter_crosslevel_edges()
    self.remap_node_idx()
    self.add_node_type()
    self.add_node_keyd()  # used in skip connects
    self.add_node_mask()
    self.sort_edges()

  def save(self, filename):
    np.save(filename + 'xyz.npy', self.xyz.cpu().numpy())
    np.save(filename + 'batch.npy', self.batch.cpu().numpy())
    np.save(filename + 'node_depth.npy', self.node_depth.cpu().numpy())
    for d in range(self.full_depth, self.depth + 1):
      edge_idx = self._graph[d]['edge_idx']
      np.save(filename + "edge_%d.npy" % d, edge_idx.t().cpu().numpy())


if __name__ == '__main__':
  octrees = ocnn.octree_samples(['octree_1', 'octree_2'])
  octree = ocnn.octree_batch(octrees).cuda()
  pdoctree = DualOctree(octree)
  pdoctree.save('data/batch_12_')
  pdoctree.add_self_loops()
  pdoctree.calc_edge_type()
  pdoctree.remap_node_idx()
  print('succ!')
