# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import ocnn
import torch
from ocnn.dataset import CollateBatch


def collate_func(batch):

    collate_batch = CollateBatch(merge_points=False)
    output = collate_batch(batch)
    # output = ocnn.collate_octrees(batch)

    if 'pos' in output:
        batch_idx = torch.cat([torch.ones(pos.size(0), 1) * i
                                                     for i, pos in enumerate(output['pos'])], dim=0)
        pos = torch.cat(output['pos'], dim=0)
        output['pos'] = torch.cat([pos, batch_idx], dim=1)

    for key in ['grad', 'sdf', 'occu', 'weight']:
        if key in output:
            output[key] = torch.cat(output[key], dim=0)

    if 'split_small' in output:
        output['split_small'] = torch.stack(output['split_small'])

    if 'split_large' in output:
        output['split_large'] = torch.cat(output['split_large'], dim = 0)

    return output
