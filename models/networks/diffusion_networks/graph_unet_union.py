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
        image_size,
        input_depth,
        unet_type,
        full_depth,
        input_channels,
        out_channels,
        model_channels,
        num_res_blocks,
        attention_resolutions,
        channel_mult,
        num_heads,
        use_checkpoint,
        dims,
    ):
        super().__init__()
        unet_models = []
        num_models = len(unet_type)
        for i in range(num_models):
            if unet_type[i] == "lr":
                unet_model = graph_unet_lr.UNet3DModel(
                    full_depth=full_depth,
                    in_split_channels=input_channels[i],
                    model_channels=model_channels[i],
                    out_split_channels=out_channels[i],
                    attention_resolutions=attention_resolutions,
                    channel_mult=channel_mult[i],
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    dims=dims,
                )
            elif unet_type[i] == "hr":
                unet_model = graph_unet_hr.UNet3DModel(
                    image_size=image_size[i],
                    input_depth=input_depth[i],
                    full_depth=full_depth,
                    in_channels=input_channels[i],
                    model_channels=model_channels[i],
                    lr_model_channels=model_channels[i - 1],
                    out_channels=out_channels[i],
                    num_res_blocks=num_res_blocks[i],
                    channel_mult=channel_mult[i],
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                )
            else:
                raise ValueError
            unet_models.append(unet_model)
        self.unet_lr = unet_models[0]
        self.unet_hr = unet_models[1]
        if num_models > 2:
            self.unet_feature = unet_models[2]


    def forward(self, unet_type=None, **input_data):
        if unet_type == "lr":
            return self.unet_lr(**input_data)
        elif unet_type == "hr":
            return self.unet_hr(**input_data)
        elif unet_type == "feature":
            return self.unet_feature(**input_data)
        else:
            raise ValueError
        
