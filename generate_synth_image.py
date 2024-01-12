from utils.render_utils import generate_image_for_fid
import trimesh
import os

category = "airplane"

snc_category_to_synth_id_13 = {
    'airplane': '02691156',
    'bench': '02828884',
    'cabinet': '02933112',
    'car': '02958343',
    'chair': '03001627',
    'monitor': '03211117',
    'lamp': '03636649',
    'loudspeaker': '03691459',
    'rifle': '04090263',
    'sofa': '04256520',
    'table': '04379243',
    'telephone': '04401088',
    'vessel': '04530566',
}


synth_id = snc_category_to_synth_id_13[category]
# filelist = f'/data/checkpoints/xiongbj/DualOctreeGNN-Pytorch-HR/data/ShapeNet/filelist/train_{category}.txt'

fid_root = f'./fid_{category}_uncond'

os.makedirs(fid_root, exist_ok=True)

mesh_dir = f'{category}_mesh'

# with open(filelist) as fid:
#     lines = fid.readlines()

# for line in lines:
#     filename = line.split()[0]
#     filename = filename.split('/')[1]
#     mesh_path = os.path.join(mesh_dir, filename + '.off')
#     if not os.path.exists(mesh_path): continue
#     mesh = trimesh.load_mesh(mesh_path)
#     generate_image_for_fid(mesh,fid_root, filename)
#     print(f'The mesh {filename} finish rendering')

meshes = os.listdir(mesh_dir)

for mesh in meshes:
    name = mesh[:-4]
    mesh_path = os.path.join(mesh_dir, mesh)
    mesh = trimesh.load_mesh(mesh_path)
    generate_image_for_fid(mesh,fid_root, name)
    print(f'The mesh {name} finish rendering')