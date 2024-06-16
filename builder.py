# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import ocnn

import datasets
import models


def get_dataset(flags):
  if flags.name.lower() == 'shapenet':
    return datasets.dualoctree_snet.get_shapenet_dataset(flags)
  else:
    raise ValueError
