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

all_num = 0
for key, val in category_5_to_num.items():
    all_num += val

category_5_to_num['im_5'] = all_num

category = 'chair'
total_num = category_5_to_num[category]

# initialize SDFusion model
model = 'octfusion_hr_feature'
df_cfg = 'configs/octfusion_snet_hr_feature.yaml'
ckpt_path = f'logs_home/continue-2024-02-07T11-18-49-octfusion_hr_feature-snet-im_5-LR1e-4-release/ckpt/df_steps-latest.pth'

vq_cfg = "configs/shapenet_vae_hr.yaml"
vq_ckpt = 'saved_ckpt/graph_vae/all/all-KL-0.25-weight-0.001-depth-9-00140.model.pth'

dset="snet"
opt.init_model_args(model = model, df_cfg = df_cfg, ckpt_path=ckpt_path, vq_cfg = vq_cfg, vq_ckpt_path = vq_ckpt)
opt.init_dset_args(dataset_mode=dset, category = category)
SDFusion = create_model(opt)

train_loader, test_loader = config_dataloader(opt)
total_num = len(train_loader)
test_dg = get_data_generator(test_loader)
train_dg = get_data_generator(train_loader)

ddim_steps = 200
ddim_eta = 0.
split_dir_small = f'{category}_split_small'
split_dir_large = f'{category}_split_large'

for i in range(total_num):
    # train_data = next(train_dg)
    # test_data = next(test_dg)

    split_path_small = os.path.join(split_dir_small, f'{i}.pth')
    split_path_large = os.path.join(split_dir_large, f'{i}.pth')
    SDFusion.uncond(data = None, split_path_small = split_path_small, split_path_large = split_path_large, category = category, ema = True, ddim_steps = ddim_steps, ddim_eta = ddim_eta, save_index = i)
