import os
gpu_ids = 9
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

category = 'airplane'
label = category_5_to_label[category]
total_num = category_5_to_num[category]

# initialize SDFusion model
model = 'octfusion_lr_feature'
df_cfg = 'configs/octfusion_snet_lr_feature.yaml'
ckpt_path = f'saved_ckpt/diffusion-ckpt/{category}/df_steps-lr-feature.pth'

# if category == 'chair':
#     ckpt_path = 'logs_home/2024-04-08T14-55-07-octfusion_lr_feature-snet-chair-LR1e-4-release/ckpt/df_steps-282000.pth'
# elif category == 'table':
#     ckpt_path = 'logs_home/2024-04-22T14-56-03-octfusion_lr_feature-snet-table-LR1e-4-release/ckpt/df_steps-latest.pth'

vq_cfg = "configs/shapenet_vae_lr.yaml"
vq_ckpt = 'saved_ckpt/all-KL-0.25-weight-0.001-depth-8-00200.model.pth'

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
text_cond_split_dir = f'/data1/xiongbj/weist/code/OctFusion-Union/logs/text-cond/8channel-{category}-channel256-cfg4-epoch300/text-cond/office_cfg2/split'

sketch_split_dir = f'/data1/xiongbj/weist/code/OctFusion-Union/logs/sketch-cond/8channel-image-all/results/{category}/split'
split_dir = sketch_split_dir

all_splits = os.listdir(split_dir)

for split_filename in all_splits:
    seed_everything(0)
    index = int(split_filename.split(".")[0])
    split_path = os.path.join(split_dir, split_filename)
    SDFusion.uncond(data = None, split_path = split_path, category = category, suffix = 'mesh_2t', ema = True, ddim_steps = ddim_steps, ddim_eta = ddim_eta, clean = False, save_index = index)
