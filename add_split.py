import os
import shutil

category = '02958343'

# 02691156
# 02958343
# 03001627
# 04090263
# 04379243

data_path = f'/data/xiongbj/ShapeNet/dataset_256/{category}'
split_path_small = f'/data/xiongbj/ShapeNet/split_small/{category}'
split_path_large = f'/data/xiongbj/ShapeNet/split_large/{category}'

splits_small = os.listdir(split_path_small)
splits_large = os.listdir(split_path_large)

for split_small, split_large in zip(splits_small, splits_large):
    split_tensor_small = os.path.join(split_path_small, split_small)
    split_tensor_large = os.path.join(split_path_large, split_large)
    index = split_small[:-4]
    print(index)
    target_path_small = os.path.join(data_path, index, 'split_small.pth')
    target_path_large = os.path.join(data_path, index, 'split_large.pth')
    shutil.copy(split_tensor_small, target_path_small)
    shutil.copy(split_tensor_large, target_path_large)
