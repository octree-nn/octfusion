from .render.render import render_mesh
import torch
import torchvision
from .util import ensure_directory, scale_to_unit_sphere
import os

def render_one_mesh(mesh, i, j, mydir, render_resolution=1024):
    mesh = scale_to_unit_sphere(mesh)
    image = render_mesh(mesh, index=j, resolution=render_resolution)/255
    torchvision.utils.save_image(torch.from_numpy(image.copy()).permute(
        2, 0, 1), f"{mydir}/{i}_{j}.png")

# Inception v3 input size (299, 299)
def generate_image_for_fid(mesh, mydir, i):
    render_resolution = 299
    mesh = scale_to_unit_sphere(mesh)
    for j in range(20):
        if os.path.exists(f"{mydir}/view_{j}/{i}.png"):
            continue
        ensure_directory(f"{mydir}/view_{j}")
        image = render_mesh(mesh, index=j, resolution=render_resolution)/255
        torchvision.utils.save_image(torch.from_numpy(image.copy()).permute(
                2, 0, 1), f"{mydir}/view_{j}/{i}.png")