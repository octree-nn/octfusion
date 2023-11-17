# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
from .scatter import scatter_add


def spmm(index, value, m, n, matrix):
  """Matrix product of sparse matrix with dense matrix.

  Args:
      index (:class:`LongTensor`): The index tensor of sparse matrix.
      value (:class:`Tensor`): The value tensor of sparse matrix.
      m (int): The first dimension of corresponding dense matrix.
      n (int): The second dimension of corresponding dense matrix.
      matrix (:class:`Tensor`): The dense matrix.

  :rtype: :class:`Tensor`
  """

  assert n == matrix.size(-2)

  row, col = index
  matrix = matrix if matrix.dim() > 1 else matrix.unsqueeze(-1)

  out = matrix.index_select(-2, col)
  out = out * value.unsqueeze(-1)
  out = scatter_add(out, row, dim=-2, dim_size=m)

  return out


def modulated_spmm(index, value, m, n, matrix, xyzf):
  """Matrix product of sparse matrix with dense matrix.

  Args:
      index (:class:`LongTensor`): The index tensor of sparse matrix.
      value (:class:`Tensor`): The value tensor of sparse matrix.
      m (int): The first dimension of corresponding dense matrix.
      n (int): The second dimension of corresponding dense matrix.
      matrix (:class:`Tensor`): The dense matrix.

  :rtype: :class:`Tensor`
  """

  assert n == matrix.size(-2)

  row, col = index
  matrix = matrix if matrix.dim() > 1 else matrix.unsqueeze(-1)

  out = matrix.index_select(-2, col)
  ones = torch.ones((xyzf.shape[0], 1), device=xyzf.device)
  out = torch.sum(out * torch.cat([xyzf, ones], dim=1), dim=1, keepdim=True)
  out = out * value.unsqueeze(-1)
  out = scatter_add(out, row, dim=-2, dim_size=m)

  return out
