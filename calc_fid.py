import torch
from cleanfid import fid
import os
gpu_ids = 4
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_ids}"

snc_synth_id_to_category_5 = {
    '02691156': 'airplane',   '02958343': 'car',   '03001627': 'chair',
    '04379243': 'table',
    '04090263': 'rifle'
}

category_to_snc_synth_id = {v:k for (k,v) in snc_synth_id_to_category_5.items()}

category = 'table'
cond = False
synth_id = category_to_snc_synth_id[category]


synthesis_path = f'logs/{category}_union/cascade_pretrain_res110_chan124_lr2e-4/fid_images'
dataset_path = f'data/ShapeNet/fid_images/{category}'

views1 = os.listdir(synthesis_path)
views2 = os.listdir(dataset_path)

views1 = sorted(views1, key=lambda item:int(item[5:]))
views2 = sorted(views2, key=lambda item:int(item[5:]))

assert len(views1) == len(views2)
num_views = len(views1)

fid_sum = 0
fid_dict = {}

for i, (view1, view2) in enumerate(zip(views1, views2)):
    assert view1 == view2
    view1_path = os.path.join(synthesis_path, view1)
    view2_path = os.path.join(dataset_path, view2)
    fid_value = fid.compute_fid(view1_path, view2_path, batch_size = 128)
    fid_sum += fid_value
    fid_dict[view1] = fid_value
    print(f'Finish {i} th view')
    print(f'The FID of {i} th view is {fid_value}')

fid_ave = fid_sum / num_views

print('FID Value:', fid_ave)
for k, v in fid_dict.items():
    print(k, v)
