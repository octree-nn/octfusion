
import numpy as np
from imageio import imread
from PIL import Image

from termcolor import colored, cprint

import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms

from torchvision import datasets

# from configs.paths import dataroot


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def CreateDataset(opt):
    dataset = None

    # decide resolution later at model
    if opt.dataset_mode == 'snet':
        from datasets.snet_dataset import ShapeNetDataset
        train_dataset = ShapeNetDataset()
        test_dataset = ShapeNetDataset()
        train_dataset.initialize(opt, 'train', cat=opt.cat, res=opt.res)
        test_dataset.initialize(opt, 'test', cat=opt.cat, res=opt.res)

    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    cprint("[*] Dataset has been created: %s" % (train_dataset.name()), 'blue')
    return train_dataset, test_dataset
