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
  # if flags.name.lower() == 'completion':
  #   return datasets.get_completion_dataset(flags)
  # elif flags.name.lower() == 'noise2clean':
  #   return datasets.get_noise2clean_dataset(flags)
  # elif flags.name.lower() == 'convonet':
  #   return datasets.get_convonet_dataset(flags)
  # elif flags.name.lower() == 'deepmls':
  #   return datasets.get_deepmls_dataset(flags)

  if flags.name.lower() == 'shapenet':
    return datasets.dualoctree_snet.get_shapenet_dataset(flags)
  else:
    raise ValueError
