# OctFusion: Octree-based Diffusion Models for 3D Shape Generation
[[`arXiv`](https://arxiv.org/abs/2212.04493)]
[[`Project Page`](https://yccyenchicheng.github.io/SDFusion/)]
[[`BibTex`](#citation)]

Code release for the paper "OctFusion: Octree-based Diffusion Models for 3D Shape Generation".

![teaser](./assets/teaser.png)


## Installation
We recommend using [`conda`](https://www.anaconda.com/products/distribution) to install the required python packages. You might need to change the `cudatoolkit` version to match with your GPU driver.
```
conda create -n octfusion python=3.9 -y && conda activate octfusion
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip3 install ocnn h5py joblib termcolor scipy einops tqdm matplotlib opencv-python PyMCubes imageio trimesh omegaconf tensorboard notebook numpy tqdm yacs scipy plyfile tensorboard scikit-image trimesh wget mesh2sdf setuptools matplotlib

```

## Usage

### Pre-trained Models
We provide the pretrained models for the category-conditioned generation and sketch-conditioned generation. Please download the pretrained models from [Google Drive](https://drive.google.com/drive/folders/1mN6iZ-NHAkSyQ526bcoECiDrDSx4zL9B?usp=sharing) and put them in `saved_ckpt/`.

### Generate

#### Unconditional generation
```
sh scripts/gen_snet_uncond.sh
```

#### Category-conditioned generation
```
sh scripts/gen_snet_cond.sh
```

### Preparing the data


### How to train the SDFusion



# <a name="citation"></a> Citation

If you find this code helpful, please consider citing:

1. Conference version
```BibTeX
@inproceedings{cheng2023sdfusion,
  author={Cheng, Yen-Chi and Lee, Hsin-Ying and Tuyakov, Sergey and Schwing, Alex and Gui, Liangyan},
  title={{SDFusion}: Multimodal 3D Shape Completion, Reconstruction, and Generation},
  booktitle={CVPR},
  year={2023},
}
```
2. arxiv version
```BibTeX
@article{cheng2022sdfusion,
  author = {Cheng, Yen-Chi and Lee, Hsin-Ying and Tuyakov, Sergey and Schwing, Alex and Gui, Liangyan},
  title = {{SDFusion}: Multimodal 3D Shape Completion, Reconstruction, and Generation},
  journal = {arXiv},
  year = {2022},
}
```

# <a name="issue"></a> Issues and FAQ
Coming soon!

# Acknowledgement
This code borrows heavely from [SDFusion](https://github.com/yccyenchicheng/SDFusion), [LAS-Diffusion](https://github.com/Zhengxinyang/LAS-Diffusion), [DualOctreeGNN](https://github.com/microsoft/DualOctreeGNN). We thank the authors for their great work. The followings packages are required to compute the SDF: [freeglut3](https://freeglut.sourceforge.net/), [tbb](https://www.ubuntuupdates.org/package/core/kinetic/universe/base/libtbb-dev).

This work is supported in part by NSF under Grants 2008387, 2045586, 2106825, MRI 1725729, and NIFA award 2020-67021-32799. Thanks to NVIDIA for providing a GPU for debugging.
