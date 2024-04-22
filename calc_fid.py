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
synth_id = category_to_snc_synth_id[category]

# synthesis_path = f'fid_{category}_uncond'
synthesis_path = f'/data/xiongbj/OctFusion-Cascade/fid_{category}_uncond_2t'
dataset_path = f'/data/xiongbj/ShapeNet/fid_images/{category}'

views1 = os.listdir(synthesis_path)
views2 = os.listdir(dataset_path)

assert len(views1) == len(views2)
num_views = len(views1)

fid_sum = 0

for i, (view1, view2) in enumerate(zip(views1, views2)):
    assert view1 == view2
    view1_path = os.path.join(synthesis_path, view1)
    view2_path = os.path.join(dataset_path, view2)
    fid_value = fid.compute_fid(view1_path, view2_path, batch_size = 128)
    fid_sum += fid_value
    print(f'Finish {i} th view')
    print(f'The FID of {i} th view is {fid_value}')

fid_ave = fid_sum / num_views

print('FID Value:', fid_ave)
