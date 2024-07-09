import os
gpu_ids = 2
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

category_5_to_num = {'airplane' : 2831, 'car': 5247, 'chair': 4744, 'table': 5956, 'rifle': 1660}

category = 'chair'
label = category_5_to_label[category]
total_num = category_5_to_num[category]

# initialize SDFusion model
model = 'octfusion_hr_large'
df_cfg = 'configs/octfusion_snet_hr_large.yaml'
ckpt_path = f'Tencent/{category}/df_steps-hr-large.pth'

vq_cfg = "configs/shapenet_vae_hr.yaml"
vq_ckpt = 'saved_ckpt/graph_vae/all/all-KL-0.25-weight-0.001-depth-9-00140.model.pth'

dset="snet"
opt.init_model_args(model = model, df_cfg = df_cfg, ckpt_path=ckpt_path, vq_cfg = vq_cfg, vq_ckpt_path = vq_ckpt)
opt.init_dset_args(dataset_mode=dset, category = category)
SDFusion = create_model(opt)

ngen = 1
ddim_steps = 200
ddim_eta = 0.
split_dir = f'{category}_split_small'

for i in range(total_num):
    split_path = os.path.join(split_dir, f'{i}.pth')
    SDFusion.uncond(data = None, split_path = split_path, category = category, ema = True, ddim_steps = ddim_steps, ddim_eta = ddim_eta, save_index = i)
