# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
from typing import Optional


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
  if dim < 0:
    dim = other.dim() + dim
  if src.dim() == 1:
    for _ in range(0, dim):
      src = src.unsqueeze(0)
  for _ in range(src.dim(), other.dim()):
    src = src.unsqueeze(-1)
  src = src.expand_as(other)
  return src


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
  index = broadcast(index, src, dim)
  if out is None:
    size = list(src.size())
    if dim_size is not None:
      size[dim] = dim_size
    elif index.numel() == 0:
      size[dim] = 0
    else:
      size[dim] = int(index.max()) + 1
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, index, src)
  else:
    return out.scatter_add_(dim, index, src)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 weights: Optional[torch.Tensor] = None,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor:
  if weights is not None:
    src = src * broadcast(weights, src, dim)
  out = scatter_add(src, index, dim, out, dim_size)
  dim_size = out.size(dim)

  index_dim = dim
  if index_dim < 0:
    index_dim = index_dim + src.dim()
  if index.dim() <= index_dim:
    index_dim = index.dim() - 1

  if weights is None:
    weights = torch.ones(index.size(), dtype=src.dtype, device=src.device)
  count = scatter_add(weights, index, index_dim, None, dim_size)
  count[count < 1] = 1
  count = broadcast(count, out, dim)
  if out.is_floating_point():
    out.true_divide_(count)
  else:
    out.div_(count, rounding_mode='floor')
  return out
