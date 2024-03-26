
import cv2
import numpy as np
import pyvista as pv
from utils import WarpField

def main():
    image_path = '/Users/ekole/Dev/gut_slam/gut_images/FrameBuffer_0037.png'
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    image_height, image_width = image.shape[:2]
    image_center = (image_width / 2, image_height / 2, 0)
    radius = 1
    height = 10
    vanishing_pts = (0, 0, 10)
    center = image_center
    resolution = 100
    warp_field = WarpField(radius, height, vanishing_pts, center, resolution)

    # Apply deformation to the cylinder (optional)
    #warp_field.apply_shrinking(start_radius=3, end_radius=2)
    #warp_field.apply_deformation(strength=0.1,frequency=1)
    warp_field.apply_deformation_axis(strength=0.1,frequency=1)

    # Extract points from the cylinder
    cylinder_points = warp_field.extract_pts()

    # Plot the extracted 3D points from the cylinder's surface using PyVista
    point_cloud = pv.PolyData(cylinder_points)
    point_cloud['scalar'] = np.arange(point_cloud.n_points)  

    # Plot the mesh of the 3D cylinder
    cylinder_mesh = pv.PolyData(cylinder_points, warp_field.cylinder.faces)
    cylinder_mesh.compute_normals(cell_normals=False, inplace=True)  

    plotter = pv.Plotter()
    plotter.add_mesh(point_cloud, color='blue', point_size=5, render_points_as_spheres=True, label="Cylinder Points")
    plotter.add_mesh(cylinder_mesh, color='red', show_edges=True, edge_color='black', line_width=1, label="Cylinder Mesh")
    plotter.add_legend()
    plotter.add_title("3D Points and Mesh from Cylinder Surface")
    plotter.show()


  

if __name__ == "__main__":
    main()

