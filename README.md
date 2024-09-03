# OctFusion: Octree-based Diffusion Models for 3D Shape Generation
[[`arXiv`](https://arxiv.org/abs/2408.14732)]
[[`BibTex`](#citation)]

Code release for the paper "OctFusion: Octree-based Diffusion Models for 3D Shape Generation".

![teaser](./assets/teaser.png)


## 1. Installation
1. Clone this repository
```bash
git clone https://github.com/octree-nn/octfusion.git
cd octfusion
```
2. Create a `Conda` environment.
```bash
conda create -n octfusion python=3.11 -y && conda activate octfusion
```

3. Install PyTorch with Conda
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

4. Install other requirements.
```bash
pip3 install -r requirements.txt 
```

## 2. Generation with pre-trained models

### 2.1 Download pre-trained models
We provide the pretrained models for the category-conditioned generation and sketch-conditioned generation. Please download the pretrained models from [Google Drive](https://drive.google.com/drive/folders/140U_xzAy1MobUqurN67Fm2Y-3oWrZQ1m?usp=drive_link) or [Baidu Netdisk](https://pan.baidu.com/s/15-jp9Mwtw4soch8GAC7qgQ?pwd=rhui) and put them in `saved_ckpt/diffusion-ckpt` and `saved_ckpt/vae-ckpt`.

### 2.2 Generation
1. Unconditional generation in category `airplane`, `car`, `chair`, `rifle`, `table`.
```
sh scripts/run_snet_uncond.sh generate hr $category
# Example
sh scripts/run_snet_uncond.sh generate hr airplane

```

2. Category-conditioned generation
```
sh scripts/run_snet_cond.sh generate hr $category
# Example
sh scripts/run_snet_cond.sh generate hr chair
```

## 3. Train from scratch
### 3.1 Data Preparation

1. Download `ShapeNetCore.v1.zip` (31G) from [ShapeNet](https://shapenet.org/) and place it in `data/ShapeNet/ShapeNetCore.v1.zip`. Download `filelist` from [Google Drive](https://drive.google.com/drive/folders/140U_xzAy1MobUqurN67Fm2Y-3oWrZQ1m?usp=drive_link) or [Baidu Netdisk](https://pan.baidu.com/s/15-jp9Mwtw4soch8GAC7qgQ?pwd=rhui) and place it in `data/ShapeNet/filelist`.

2. Convert the meshes in `ShapeNetCore.v1` to signed distance fields (SDFs).
We use the same data preparation as [DualOctreeGNN](https://github.com/microsoft/DualOctreeGNN.git). Note that this process is relatively slow, it may take several days to finish converting all the meshes from ShapeNet. 
```bash
python tools/repair_mesh.py --run convert_mesh_to_sdf
python tools/repair_mesh.py --run generate_dataset
```



### 3.2 Train OctFusion
1. VAE Training. We provide pretrained weights in `saved_ckpt/vae-ckpt/vae-shapenet-depth-8.pth`.
```bash
sh scripts/run_snet_vae.sh train vae im_5
```
2. Train the first stage model. We provide pretrained weights in `saved_ckpt/diffusion-ckpt/$category/df_steps-split.pth`.
```bash
sh scripts/run_snet_uncond.sh train lr $category
```

3. Load the pretrained first stage model and train the second stage. We provide pretrained weights in `saved_ckpt/diffusion-ckpt/$category/df_steps-union.pth`. 
```bash
sh scripts/run_snet_uncond.sh train hr $category
```
# <a name="citation"></a> Citation

If you find this code helpful, please consider citing:


1. arxiv version
```BibTeX
@article{xiong2024octfusion,
  author = {Xiong, Bojun and Wei, Si-Tong and Zheng, Xin-Yang and Cao, Yan-Pei and Lian, Zhouhui and Wang, Peng-Shuai},
  title = {{OctFusion}: Octree-based Diffusion Models for 3D Shape Generation},
  journal = {arXiv},
  year = {2024},
}
```

# <a name="issue"></a> Issues and FAQ
Coming soon!

# Acknowledgement
This code borrows heavely from [SDFusion](https://github.com/yccyenchicheng/SDFusion), [LAS-Diffusion](https://github.com/Zhengxinyang/LAS-Diffusion), [DualOctreeGNN](https://github.com/microsoft/DualOctreeGNN). We thank the authors for their great work. The followings packages are required to compute the SDF: [mesh2sdf](https://github.com/wang-ps/mesh2sdf).
