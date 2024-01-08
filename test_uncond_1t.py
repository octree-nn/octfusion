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
model = 'sdfusion_feature'
df_cfg = 'configs/sdfusion_snet_1t.yaml'
# ckpt_path = f'Tencent/{category}/df_steps-66000-large-1t.pth'
ckpt_path = 'Tencent/feature-stage/df_steps-126000.pth'

vq_cfg = "configs/shapenet_vae_1t_eval.yaml"
vq_ckpt = 'saved_ckpt/graph_vae/all/all-KL-0.25-weight-0.001-depth-9-00140.model.pth'

dset="snet"
opt.init_model_args(model = model, df_cfg = df_cfg, ckpt_path=ckpt_path, vq_cfg = vq_cfg, vq_ckpt_path = vq_ckpt)
opt.init_dset_args(dataset_mode=dset, category = category)
SDFusion = create_model(opt)

train_loader, test_loader, test_loader_for_eval = config_dataloader(opt)
# total_num = len(test_loader)
total_num = len(train_loader)
test_dg = get_data_generator(test_loader)
train_dg = get_data_generator(train_loader)

ngen = 1
ddim_steps = 200
ddim_eta = 0.

for i in range(total_num):
    seed_everything(i)
    train_data = next(train_dg)
    test_data = next(test_dg)
    SDFusion.uncond(data = train_data, split_path = None, category = category, ema = True, ddim_steps = ddim_steps, ddim_eta = ddim_eta, save_index = i)
