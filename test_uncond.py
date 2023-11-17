import os
gpu_ids = 3
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_ids}"


import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from models.base_model import create_model
from utils.util import seed_everything
from utils.demo_util import SDFusionOpt

seed = 2023
opt_lr = SDFusionOpt(gpu_ids=gpu_ids, seed=seed)
device = opt_lr.device

category_5_to_label = {
    'airplane': 0,
    'car': 1,
    'chair': 2,
    'table': 3,
    'rifle': 4,
}

category_5_to_num = {'airplane' : 3236, 'car': 5996,  'chair': 5422, 'table': 6807, 'rifle': 1897}

category = 'rifle'
label = category_5_to_label[category]
total_num = category_5_to_num[category]

out_dir_lr = 'airplane_lr'
if not os.path.exists(out_dir_lr): os.makedirs(out_dir_lr)

# initialize SDFusion model
lr_df_cfg = 'configs/sdfusion_snet.yaml'
lr_ckpt_path = 'logs_home/2023-11-15T16-58-56-sdfusion_split-snet-airplane-LR1e-4-release/ckpt/df_steps-latest.pth'
model = 'sdfusion_split'
lr_vq_cfg = "configs/shapenet_vqvae.yaml"
dset="snet"
opt_lr.init_model_args(model = model, df_cfg = lr_df_cfg, ckpt_path=lr_ckpt_path, vq_cfg = lr_vq_cfg)
opt_lr.init_dset_args(dataset_mode=dset)
SDFusion_lr = create_model(opt_lr)

ngen = 1
ddim_steps = 200
ddim_eta = 0.

for i in range(total_num):
    seed_everything(3)
    doctree_out = SDFusion_lr.uncond(batch_size=ngen, steps=ddim_steps, category = label, ema = True, save_dir = out_dir_lr, index = i)
