import torch
from pytorch_fid.fid_score import calculate_fid_given_paths
import os
gpu_ids = 2
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_ids}"

batch_size = 50

dims = 2048

try:
    num_cpus = len(os.sched_getaffinity(0))
except AttributeError:
    # os.sched_getaffinity is not available under Windows, use
    # os.cpu_count instead (which may not return the *available* number
    # of CPUs).
    num_cpus = os.cpu_count()

num_workers = min(num_cpus, 8) if num_cpus is not None else 0

device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

snc_synth_id_to_category_5 = {
    '02691156': 'airplane',   '02958343': 'car',   '03001627': 'chair',
    '04379243': 'table',
    '04090263': 'rifle'
}

category_to_snc_synth_id = {v:k for (k,v) in snc_synth_id_to_category_5.items()}

category = 'airplane'
synth_id = category_to_snc_synth_id[category]

synthesis_path = f'fid_{category}_uncond'
dataset_path = f'/data/checkpoints/xiongbj/DualOctreeGNN-Pytorch-HR/data/ShapeNet/fid_images/{category}'

views1 = os.listdir(synthesis_path)
views2 = os.listdir(dataset_path)

assert len(views1) == len(views2)
num_views = len(views1)

fid = 0

for i, (view1, view2) in enumerate(zip(views1, views2)):
    assert view1 == view2
    view1_path = os.path.join(synthesis_path, view1)
    view2_path = os.path.join(dataset_path, view2)
    paths = [view1_path, view2_path]
    fid_value = calculate_fid_given_paths(paths, batch_size, device, dims, num_workers)
    fid += fid_value
    print(f'Finish {i} th view')
    print(f'The FID of {i} th view is {fid_value}')

fid = fid / num_views

print('FID Value:', fid)
