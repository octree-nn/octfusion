import os
from termcolor import colored, cprint
import torch
import utils.util as util
import math

def create_model(opt):
    model = None

    if opt.model == "union_2t":
        from models.octfusion_model_union import OctFusionModel
        model = OctFusionModel()
    elif opt.model == "union_3t":
        from models.octfusion_model_union_3t import OctFusionModel
        model = OctFusionModel()
    elif opt.model == "vae":
        from models.octfusion_model_vae import OctFusionModel
        model = OctFusionModel()
    else:
        raise ValueError

    model.initialize(opt)
    cprint("[*] Model has been created: %s" % model.name(), 'blue')
    return model


# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor

        self.model_names = []
        self.epoch_labels = []
        self.optimizers = []

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_errors(self):
        return {}

    # define the optimizers
    def set_optimizers(self):
        pass

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('[*] learning rate = %.7f' % lr)

    def update_learning_rate_cos(self, epoch, opt):
        """Decay the learning rate with half-cycle cosine after warmup"""
        if epoch >= opt.warmup_epochs:
            lr = opt.min_lr + (opt.lr - opt.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - opt.warmup_epochs) / (opt.epochs - opt.warmup_epochs)))
            for param_group in self.optimizer.param_groups:
                if "lr_scale" in param_group:
                    param_group["lr"] = lr * param_group["lr_scale"]
                else:
                    param_group["lr"] = lr
            print('[*] learning rate = %.7f' % lr)

    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def train(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    # print network information
    def print_networks(self, verbose=False):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def tocuda(self, var_names):
        for name in var_names:
            if isinstance(name, str):
                var = getattr(self, name)
                # setattr(self, name, var.cuda(self.gpu_ids[0], non_blocking=True))
                setattr(self, name, var.cuda(self.opt.device, non_blocking=True))
