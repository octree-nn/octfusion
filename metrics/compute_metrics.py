# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import argparse
import trimesh.sample
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial import cKDTree

def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P.min(1)[0], P.min(2)[0]

try:
    from metrics.StructuralLosses.nn_distance import nn_distance
    def distChamferCUDA(x, y):
        return nn_distance(x, y)
except Exception as e:
    print(str(e))
    print("distChamferCUDA not available; fall back to slower version.")
    def distChamferCUDA(x, y):
        return distChamfer(x, y)


def compute_metrics(sample_pcs, ref_pcs, batch_size):

  N_ref = ref_pcs.shape[0]
  cd_lst = []
  for ref_b_start in range(0, N_ref, batch_size):
    ref_b_end = min(N_ref, ref_b_start + batch_size)
    ref_batch = ref_pcs[ref_b_start:ref_b_end]

    batch_size_ref = ref_batch.size(0)
    sample_batch_exp = sample_pcs.view(1, -1, 3).expand(batch_size_ref, -1, -1)
    sample_batch_exp = sample_batch_exp.contiguous()
    dl, dr = distChamferCUDA(sample_batch_exp, ref_batch)
    cd = (dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1)
    cd_lst.append(cd)
  
  cd_lst = torch.cat(cd_lst, dim=1)
  
  return cd_lst