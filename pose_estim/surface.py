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
from utils import generate_uniform_grid_control_points, GridViz, SingleWindowGridViz, BMesh, BMeshDense, BMeshDefDense,MeshPlotter



rho_step_size = 0.1  # Step size for rho
alpha_step_size = np.pi/4  # Step size for alpha
radius = 100
center = (0, 0)

control_points = generate_uniform_grid_control_points(rho_step_size, alpha_step_size,R=100)



bmd = BMeshDefDense(radius=radius, center=center)
points = bmd.b_mesh_deformation(control_points=control_points,subsample_factor=10,bend_amplitude=2.0,bend_frequency=0.001)
# bmd= BMeshDense(radius=radius, center=center)
# points=bmd.b_mesh_deformation(control_points=control_points,subsample_factor=10)

texture_img = './tex/colon_DIFF.png'

# viz = GridViz(grid_shape=(2, 3))

# viz.add_mesh_cartesian(points, texture_img=texture_img,subplot=(0, 0))
# viz.add_mesh_cy(points, texture_img=texture_img,subplot=(0, 1))
# viz.add_mesh_polar(points, texture_img=texture_img,subplot=(0, 2))

visualizer=SingleWindowGridViz()

camera_settings_cartesian = visualizer.visualize_and_save_cartesian(points, screenshot=None, wireframe=True,filename='./rendering/cartesian_mesh10.vtk')
#print("Cartesian Camera Settings (without wireframe):", camera_settings_cartesian)

# Visualize and save in Cartesian coordinates with wireframe
camera_settings_cartesian_wireframe = visualizer.visualize_and_save_cylindrical(points, screenshot=None, wireframe=True,filename='./rendering/cylindrical_mesh10.vtk')
#print("Cartesian Camera Settings (with wireframe):", camera_settings_cartesian_wireframe)

# Visualize and save in Polar coordinates without wireframe
camera_settings_polar = visualizer.visualize_and_save_polar(points, texture_img=None, wireframe=True,screenshot=None, filename='./rendering/polar_mesh10.vtk')
#print("Polar Camera Settings (without wireframe):", camera_settings_polar)


# Add meshes to different subplots


# viz.show()

#viz.save_plot(filename='grid_plot.png')
