"""
-------------------------------------------------------------
This script generates control points for B-spline mesh deformation,
applies the deformation, and visualizes the resulting surface in 
Cartesian, polar coordinates, and H-surface representations.
(No interpolation of cylinder points)

Author: Mitterand Ekole
Date: 22-05-2024
-------------------------------------------------------------
"""

import numpy as np
from scipy.interpolate import SmoothBivariateSpline
import pyvista as pv
from sklearn.preprocessing import StandardScaler
import open3d as o3d
import matplotlib.pyplot as plt
from utils import generate_uniform_grid_control_points, GridViz, SingleWindowGridViz, BMesh, BMeshDense, BMeshDefDense



rho_step_size = 0.1  # Step size for rho
alpha_step_size = np.pi/4  # Step size for alpha
radius = 100
center = (0, 0)

# control_points = generate_uniform_grid_control_points(rho_step_size, alpha_step_size,R=100)
# # print(control_points.shape)
control_points=np.loadtxt('/Users/ekole/Dev/gut_slam/pose_estim/logs/optimized_control_points_frame_0.txt')
control_points=control_points.reshape(20,9,3)
#control_points = generate_uniform_grid_control_points(rho_step_size, alpha_step_size, h_constant=100)


bmd = BMeshDense(radius=radius, center=center)
deformed_points = bmd.b_mesh_deformation(control_points=control_points,subsample_factor=5)
texture_img = './tex/stomach_DIFF.png'

# viz = GridViz(grid_shape=(2, 3))
viz=SingleWindowGridViz()

# viz.add_mesh_polar(deformed_points, subplot=(0, 0),texture_img=texture_img)
# viz.add_mesh_cy(deformed_points, subplot=(0, 1),texture_img=texture_img)
# viz.add_mesh_cartesian(deformed_points,subplot=(0,2),texture_img=texture_img)
# viz.add_mesh_polar(deformed_points, subplot=(1, 0))
# viz.add_mesh_cy(deformed_points, subplot=(1, 1))
# viz.add_mesh_cartesian(deformed_points,subplot=(1,2))

camera_info_cartesian=viz.visualize_and_save_cartesian(deformed_points, './rendering/cartesian_mesh.vtk', screenshot='./rendering/cartesian_mesh.png',texture_img=texture_img)
viz.save_camera_info_to_file(camera_info_cartesian, './data/cartesian_camera_info.txt')

# viz()
