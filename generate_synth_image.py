from utils.render_utils import generate_image_for_fid
import trimesh
import os

category = "car"

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

fid_root = f'./fid_{category}_uncond_2t'

os.makedirs(fid_root, exist_ok=True)

mesh_dir = f'{category}_mesh_2t'

meshes = os.listdir(mesh_dir)

num = 0

for (index, mesh) in enumerate(meshes):
    name = mesh[:-4]
    mesh_path = os.path.join(mesh_dir, mesh)
    mesh = trimesh.load_mesh(mesh_path)
    generate_image_for_fid(mesh,fid_root, name)
    print(f'The mesh {name} finish rendering')
