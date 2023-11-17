import torch.utils.data

from datasets.base_dataset import CreateDataset
from datasets.base_dataset import data_sampler
from datasets.sampler import InfSampler, DistributedInfSampler
from builder import get_dataset
from omegaconf import OmegaConf

def get_data_generator(loader):
    while True:
        for data in loader:
            yield data

def config_dataloader(opt, disable_train_data=False):
    dualoctree_conf = OmegaConf.load(opt.vq_cfg)
    flags_train, flags_test = dualoctree_conf.data.train, dualoctree_conf.data.test
    if not disable_train_data and not flags_train.disable:
      train_loader = get_dataloader(opt,flags_train, drop_last = True)

    if not flags_test.disable:
      test_loader = get_dataloader(opt,flags_test, drop_last = False)

    if not flags_test.disable:
      test_loader_for_eval = get_dataloader(opt,flags_test, eval = True, drop_last = False)

    return train_loader, test_loader, test_loader_for_eval

def get_dataloader(opt, flags, eval = False, drop_last = True):
  dataset, collate_fn = get_dataset(flags)

  if opt.distributed:
    sampler = DistributedInfSampler(dataset, shuffle=flags.shuffle)
  else:
    sampler = InfSampler(dataset, shuffle=flags.shuffle)

  if eval==False:
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=flags.batch_size, num_workers=flags.num_workers,
        sampler=sampler, collate_fn=collate_fn, pin_memory=True, drop_last = drop_last)
  else:
    data_loader = torch.utils.data.DataLoader(
      dataset, batch_size=max(int(flags.batch_size // 2), 1), num_workers=flags.num_workers,
      sampler=sampler, collate_fn=collate_fn, pin_memory=True, drop_last = drop_last)

  return data_loader
