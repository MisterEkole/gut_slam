import numpy as np
import pyvista as pv
from utils import *

def main():
  
    image_center = (512, 512, 0)  
    radius = 500  
    height = 1000
    vanishing_points = (0, 0, 10)
    center = image_center
    resolution = 500

   
    warp_field = WarpField(radius, height, vanishing_points, center, resolution)
    control_points = np.loadtxt('/Users/ekole/Dev/gut_slam/pose_estim/data/ControlPoints_Minimal.txt').reshape(20, 20, 3)
    #control_points=generate_control_points(radius,height,num_radial=10,num_height=10)

    warp_field.b_mesh_deformation(a=0.00051301747, b=0.0018595674, control_points=control_points)
    mesh_points = warp_field.extract_pts()
    #visualize_and_save_mesh_from_points(mesh_points,'./rendering/mesh3.ply',screenshot='./rendering/mesh3.png')
    #visualize_h_surface(mesh_points)
    visualize_3dmeshcart(mesh_points)

if __name__ == "__main__":
    main()
