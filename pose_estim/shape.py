import numpy as np
import pyvista as pv
from utils import *

def main():
  
    image_center = (512, 512, 0)  
    radius = 500  
    height = 1000
    vanishing_points = (0, 0, 10)
    center = image_center
    resolution = 100

   
    warp_field = WarpField(radius, height, vanishing_points, center, resolution)
    control_points = np.loadtxt('./data/control_points14.txt').reshape(10, 10, 3)

    warp_field.b_mesh_deformation(a=0.00051301747, b=0.0018595674, control_points=control_points)
    mesh_points = warp_field.extract_pts()
    #mesh_points=np.loadtxt('./data/mesh_points8.txt', delimiter=',')
    #np.savetxt('./data/mesh_points8.txt', mesh_points, fmt='%.6f', delimiter=',')
    
    #visualize_and_save_mesh_from_points(mesh_points,'./rendering/mesh10.ply',screenshot='./rendering/mesh10.png')
    plotter=GridViz(grid_shape=(2,2),window_size=(2300,1500))
    plotter.add_h_surface(mesh_points,(0,0))
    plotter.add_mesh_cartesian(mesh_points,(0,1))
    plotter.add_mesh_polar(mesh_points,(1,0))
    plotter.add_3dmesh_open(mesh_points,(1,1))
    plotter()

if __name__ == "__main__":
    main()
