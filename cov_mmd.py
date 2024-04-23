from pprint import pprint
from metrics.evaluation_metrics import compute_cov_mmd, compute_1_nna
import torch
import time
import numpy as np
import os
import argparse
import sys
import pickle

gpu_ids = 5
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_ids}"

sample_pcs = torch.load('chair_sample_pcs.pth')
print(sample_pcs.shape)

sample_pcs = sample_pcs.cuda().to(torch.float32)

ref_pcs = torch.load('chair_ref_pcs.pth')
print(ref_pcs.shape)

ref_pcs = ref_pcs.cuda().to(torch.float32)

print('##################################################################')

results = compute_cov_mmd(sample_pcs[:ref_pcs.shape[0]], ref_pcs, batch_size = 256)
# results = compute_cov_mmd(sample_pcs[:256], ref_pcs[:256], batch_size = 256)
results = {k: (v.cpu().detach().item()
              if not isinstance(v, float) else v) for k, v in results.items()}

pprint(results)

print('##################################################################')
