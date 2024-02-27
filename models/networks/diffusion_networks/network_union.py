""" Reference: https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/ddpm.py#L1395-L1421 """

import math
import torch
import torch.nn as nn

from einops import rearrange, repeat

from .graph_unet_union import UNet3DModel

class DiffusionUNet(nn.Module):
    def __init__(self, unet_params, vq_conf=None, conditioning_key=None):
        """ init method """
        super().__init__()

        self.diffusion_net = UNet3DModel(**unet_params)
        self.conditioning_key = conditioning_key # default for lsun_bedrooms

    def forward(self, x, doctree_in, doctree_out=None, timesteps=None, c_concat: list = None, c_crossattn: list = None, projection_matrix = None):
        # x: should be latent code. shape: (bs X z_dim X d X h X w)

        if self.conditioning_key == 'None':
            out = self.diffusion_net(x, doctree_in, doctree_out, timesteps)
        # elif self.conditioning_key == 'concat':
            # xc = torch.cat([x] + c_concat, dim=1)
            # out = self.diffusion_net(xc, timesteps)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_net(x, doctree_in, doctree_out, timesteps, context=cc, projection_matrix=projection_matrix)
        # elif self.conditioning_key == 'hybrid':
            # xc = torch.cat([x] + c_concat, dim=1)
            # cc = torch.cat(c_crossattn, 1)
            # out = self.diffusion_net(xc, timesteps, context=cc)
        # elif self.conditioning_key == 'adm':
            # cc = c_crossattn[0]
            # out = self.diffusion_net(x, doctree_in, timesteps, y=cc)
        else:
            raise NotImplementedError()

        return out