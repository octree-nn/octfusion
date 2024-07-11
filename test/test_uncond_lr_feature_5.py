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

category_5_to_num = {'airplane' : 2831, 'car': 5247, 'chair': 4744, 'table': 5956, 'rifle': 1660}

category = 'table'
label = category_5_to_label[category]
total_num = category_5_to_num[category]

# initialize SDFusion model
model = 'octfusion_lr_feature_5'
df_cfg = 'configs/octfusion_snet_lr_feature_5.yaml'
ckpt_path = 'logs_home/2024-04-23T21-23-41-octfusion_lr_feature_5-snet-im_5-LR1e-4-release/ckpt/df_steps-latest.pth'

vq_cfg = "configs/shapenet_vae_lr.yaml"
vq_ckpt = 'saved_ckpt/graph_vae/all/all-KL-0.25-weight-0.001-depth-8-00200.model.pth'

dset="snet"
opt.init_model_args(model = model, df_cfg = df_cfg, ckpt_path=ckpt_path, vq_cfg = vq_cfg, vq_ckpt_path = vq_ckpt)
opt.init_dset_args(dataset_mode=dset, category = category)
SDFusion = create_model(opt)

ngen = 1
ddim_steps = 200
ddim_eta = 0.
uncond_split_dir = f'{category}_split_small'
category_cond_split_dir = f'{category}_split_small_cond'

text_cond = 'rocking_chair'
text_cond_split_dir = f'text_cond_results/{text_cond}/split'

split_dir = category_cond_split_dir

all_splits = os.listdir(split_dir)

for i in range(total_num):
    seed_everything(0)    # rifle的话是4
    split_path = os.path.join(split_dir, f'{i}.pth')
    SDFusion.sample(data = None, split_path = split_path, category = category, suffix = 'mesh_2t_cond', ema = True, ddim_steps = ddim_steps, ddim_eta = ddim_eta, clean = False, save_index = i)
