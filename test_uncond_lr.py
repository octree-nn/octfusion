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

category_5_to_num = {'airplane' : 3236, 'car': 5996,  'chair': 5422, 'table': 6807, 'rifle': 1897}

category = 'rifle'
label = category_5_to_label[category]
total_num = category_5_to_num[category]

# initialize SDFusion model
model = 'sdfusion_union_two_time_8_channels'
df_cfg = 'configs/sdfusion_snet_2t.yaml'
ckpt_path = f'Tencent/{category}_lr/df_steps-latest.pth'
# ckpt_path = 'logs_home/2023-12-23T23-33-33-sdfusion_union_three_time_8_channels-snet-rifle-LR1e-4-release/ckpt/df_steps-latest.pth'

vq_cfg = "configs/shapenet_vae_128.yaml"
vq_ckpt = 'saved_ckpt/graph_vae/all/all-KL-0.25-weight-0.001-depth-8-00200.model.pth'

dset="snet"
opt.init_model_args(model = model, df_cfg = df_cfg, ckpt_path=ckpt_path, vq_cfg = vq_cfg, vq_ckpt_path = vq_ckpt)
opt.init_dset_args(dataset_mode=dset)
SDFusion = create_model(opt)

train_loader, test_loader, test_loader_for_eval = config_dataloader(opt)
total_num = len(train_loader)
test_dg = get_data_generator(test_loader)
train_dg = get_data_generator(train_loader)

ngen = 1
ddim_steps = 200
ddim_eta = 0.

for i in range(total_num):
    seed_everything(i)
    SDFusion.uncond(batch_size=ngen, steps=ddim_steps, category = category, ema = False, index = i)

    # train_data = next(train_dg)
    # test_data = next(test_dg)
    # SDFusion.uncond_withdata_small(train_data, steps = ddim_steps, category = category, ema = True, index = i)
    # SDFusion.uncond_withdata_large(train_data, steps=ddim_steps, category = category, ema = True, index = i)
