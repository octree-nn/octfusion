import numpy as np
from utils.render.render_utils import Render, create_pose
import matplotlib
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
matplotlib.use("Agg")



FrontVector = (np.array([[0.52573, 0.38197, 0.85065],
                         [-0.20081, 0.61803, 0.85065],
                         [-0.64984, 0.00000, 0.85065],
                         [-0.20081, -0.61803,  0.85065],
                         [0.52573, -0.38197, 0.85065],
                         [0.85065, -0.61803, 0.20081],
                         [1.0515,  0.00000, -0.20081],
                         [0.85065, 0.61803, 0.20081],
                         [0.32492, 1.00000, -0.20081],
                         [-0.32492, 1.00000,  0.20081],
                         [-0.85065, 0.61803, -0.20081],
                         [-1.0515, 0.00000,  0.20081],
                         [-0.85065, -0.61803, -0.20081],
                         [-0.32492, -1.00000,  0.20081],
                         [0.32492, -1.00000, -0.20081],
                         [0.64984, 0.00000, -0.85065],
                         [0.20081, 0.61803, -0.85065],
                         [-0.52573, 0.38197, -0.85065],
                         [-0.52573, -0.38197, -0.85065],
                         [0.20081, -0.61803, -0.85065]]))*2

def render_mesh(mesh, resolution=1024, index=5, background=None, scale=1, no_fix_normal=True):
  
  camera_pose = create_pose(FrontVector[index]*scale)

  render = Render(size=resolution, camera_pose=camera_pose,
                  background=background)

  triangle_id, rendered_image, normal_map, depth_image, p_images = render.render(path=None,
                                                                                 clean=True,
                                                                                 mesh=mesh,
                                                                                 only_render_images=no_fix_normal)
  return rendered_image