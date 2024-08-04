# --------------------------------------------------------
# Dual Octree Graph Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import torch
import torch.utils.data
import numpy as np
from tqdm import tqdm
from datasets.shapenet_utils import snc_synth_id_to_label_13, snc_synth_id_to_label_5


def read_file(filename):
    points = np.fromfile(filename, dtype=np.uint8)
    return torch.from_numpy(points)     # convert it to torch.tensor


class Dataset(torch.utils.data.Dataset):

    def __init__(self, root, filelist, transform, read_file=read_file,
                in_memory=False, take: int = -1):
        super(Dataset, self).__init__()
        self.root = root
        self.filelist = filelist
        self.transform = transform
        self.in_memory = in_memory
        self.read_file = read_file
        self.take = take

        self.filenames, self.labels = self.load_filenames()
        if self.in_memory:
            print('Load files into memory from ' + self.filelist)
            self.samples = [self.read_file(os.path.join(self.root, f))
                                            for f in tqdm(self.filenames, ncols=80, leave=False)]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        sample = self.samples[idx] if self.in_memory else \
                         self.read_file(os.path.join(self.root, self.filenames[idx]))    # noqa
        output = self.transform(sample, idx)        # data augmentation + build octree
        output['label'] = self.labels[idx]
        output['filename'] = self.filenames[idx]
        return output

    def load_filenames(self):
        filenames, labels = [], []
        with open(self.filelist) as fid:
            lines = fid.readlines()
        for line in lines:
            filename = line.split()[0]
            label = filename.split('/')[0]
            if label in snc_synth_id_to_label_5:
            	label = snc_synth_id_to_label_5[label]
            else:
                label = 0
            filenames.append(filename)
            labels.append(torch.tensor(label))

        num = len(filenames)
        if self.take > num or self.take < 1:
            self.take = num

        return filenames[:self.take], labels[:self.take]
