import os
gpu_ids = 1
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_ids}"

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

category = 'airplane'
label = category_5_to_label[category]
total_num = category_5_to_num[category]

# initialize SDFusion model
model = 'sdfusion_union_three_time'
df_cfg = 'configs/sdfusion_snet.yaml'
ckpt_path = 'Tencent/three-times/df_steps-latest.pth'

vq_cfg = "configs/shapenet_vae.yaml"
vq_ckpt = 'saved_ckpt/graph_vae/all/depth-9-00070.model.pth'

dset="snet"
opt_lr.init_model_args(model = model, df_cfg = df_cfg, ckpt_path=ckpt_path, vq_cfg = vq_cfg, vq_ckpt_path = vq_ckpt)
opt_lr.init_dset_args(dataset_mode=dset)
SDFusion_lr = create_model(opt_lr)

ngen = 1
ddim_steps = 50
ddim_eta = 0.

for i in range(total_num):
    seed_everything(i)
    doctree_out = SDFusion_lr.uncond(batch_size=ngen, steps=ddim_steps, category = label, ema = True, index = i)
