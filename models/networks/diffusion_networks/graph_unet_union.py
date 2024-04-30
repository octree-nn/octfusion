### adapted from: https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py

from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.diffusion_networks import graph_unet_lr, graph_unet_hr

class UNet3DModel(nn.Module):


    # def __init__(self, config_dict):
    def __init__(
        self,
        unet_params,
    ):
        super().__init__()
        self.unet_lr = graph_unet_lr.UNet3DModel(**unet_params)
        self.unet_hr = graph_unet_hr.UNet3DModel(**unet_params)

    def forward(self, x_lr=None, x_hr=None, x_self_cond=None, doctree=None, timesteps = None, label = None, context = None, **kwargs):
        if x_hr == None:
            return self.unet_lr(x_lr=x_lr, timesteps=timesteps, x_self_cond=x_self_cond, label=label, context=context)
        else:
            return self.unet_hr(x_hr=x_hr, doctree=doctree, timesteps=timesteps, unet_lr=self.unet_lr, label=label, context=context)
        
