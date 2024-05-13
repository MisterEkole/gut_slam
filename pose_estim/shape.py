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
    control_points = np.loadtxt('/Users/ekole/Dev/gut_slam/pose_estim/data/control_points10.txt').reshape(10, 10, 3)

    #control_points=generate_control_points(radius,height,num_radial=10,num_height=10)

    warp_field.b_mesh_deformation(a=0.00051301747, b=0.0018595674, control_points=control_points)
    mesh_points = warp_field.extract_pts()
    #mesh_points=np.loadtxt('./data/mesh_points.txt', delimiter=',')
    #np.savetxt('./data/mesh_points4.txt', mesh_points, fmt='%.6f', delimiter=',')
    # mesh_points1=np.loadtxt('./data/mesh_points1.txt', delimiter=',')
    # mesh_points2=np.loadtxt('./data/mesh_points2.txt', delimiter=',')
    # mesh_points3=np.loadtxt('./data/mesh_points3.txt', delimiter=',')
    #visualize_and_save_mesh_from_points(mesh_points1,'./rendering/mesh1.vtk',screenshot='./rendering/mesh1.png')
    # visualize_and_save_mesh_from_points(mesh_points2,'./rendering/mesh2.vtk',screenshot='./rendering/mesh2.png')
    # visualize_and_save_mesh_from_points(mesh_points3,'./rendering/mesh3.vtk',screenshot='./rendering/mesh3.png')
    #visualize_and_save_mesh_from_points(mesh_points,'./rendering/mesh4.vtk',screenshot='./rendering/mesh4.png')

    # cam_info=visualize_and_save_mesh_with_camera(mesh_points,'./rendering/mesh6.vtk','./rendering/mesh6.png')
    # print('Cam info:', cam_info)
    
    # plotter=GridViz(grid_shape=(1,3))
    # plotter.add_h_surface(mesh_points,(0,0))
    # plotter.add_mesh_cartesian(mesh_points,(0,1))
    # plotter.add_mesh_polar(mesh_points,(0,2))
    # plotter()

if __name__ == "__main__":
    main()
