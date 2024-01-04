import os
import sys
from compute_metrics import compute_metrics

num_samples = 10000

input_obj = 'rifle_lr/15.obj'

train_dataset = 'rifle_lr_gt'

train_meshes = os.listdir(train_dataset)

chamfer_min = sys.maxsize

res = []

for index, mesh in enumerate(train_meshes):
    print(index, mesh)
    mesh_path = os.path.join(train_dataset, mesh)
    filename_ref = mesh_path
    filename_gen = input_obj
    metrics = compute_metrics(filename_ref, filename_gen, num_samples)
    chamfer_a, chamfer_b = metrics[0], metrics[1]
    chamfer = 0.5 * (chamfer_a + chamfer_b)
    if len(res) < 5: res.append([chamfer, mesh])
    else:
        res.sort()
        if chamfer < res[-1][0]:
            res[-1] = [chamfer, mesh]

res.sort()
print(res)
