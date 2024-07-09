import os
gpu_ids = 4
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_ids}"

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from models.base_model import create_model
from utils.util import seed_everything
from utils.demo_util import SDFusionOpt
from datasets.dataloader import config_dataloader, get_data_generator

seed = 2023
opt = SDFusionOpt(gpu_ids=gpu_ids, seed=seed)
device = opt.device

category_5_to_label = {
    'airplane': 0,
    'car': 1,
    'chair': 2,
    'table': 3,
    'rifle': 4,
}

category_5_to_num = {'airplane' : 2831, 'car': 5247,  'chair': 4744, 'table': 5956, 'rifle': 1660}

category = 'rifle'
label = category_5_to_label[category]
total_num = category_5_to_num[category]

# initialize SDFusion model
model = 'octfusion_small'
df_cfg = 'configs/sdfusion_snet_small_5.yaml'
ckpt_path = f'logs_home/2024-04-07T20-16-40-octfusion_small-snet-im_5-LR1e-4-release/ckpt/df_steps-latest.pth'

vq_cfg = "configs/shapenet_vae_lr.yaml"
vq_ckpt = 'saved_ckpt/graph_vae/all/all-KL-0.25-weight-0.001-depth-8-00200.model.pth'

dset="snet"
opt.init_model_args(model = model, df_cfg = df_cfg, ckpt_path=ckpt_path, vq_cfg = vq_cfg, vq_ckpt_path = vq_ckpt)
opt.init_dset_args(dataset_mode=dset, category = category)
SDFusion = create_model(opt)

ngen = 1
ddim_steps = 200
ddim_eta = 0.

for i in range(total_num):
    seed_everything(i)
    SDFusion.uncond(batch_size=ngen, category = category, ema = True, ddim_steps = ddim_steps, ddim_eta = ddim_eta, save_index = i)
    # SDFusion.uncond_interp(batch_size=ngen, category = category, ema = True, ddim_steps = ddim_steps, ddim_eta = ddim_eta, save_index = i)
