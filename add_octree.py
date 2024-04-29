import os
import shutil

category = '02691156'

# 02691156
# 02958343
# 03001627
# 04090263
# 04379243

data_path = f'/data/xiongbj/ShapeNet/dataset_256/{category}'
octree_path = f'/data/xiongbj/ShapeNet/octree_0.8/{category}'

octrees = os.listdir(octree_path)

for octree in octrees:
    octree_tensor = os.path.join(octree_path, octree)
    index = octree[:-4]
    print(index)
    target_octree = os.path.join(data_path, index, 'octree.pth')
    shutil.copy(octree_tensor, target_octree)
