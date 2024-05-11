'''
-------------------------------------------------------------
Visualize 3D Mesh Wireframe on 2D Image
Author: Mitterand Ekole
Date: 02-05-2024
-------------------------------------------------------------
'''


import cv2
import numpy as np
import pyvista as pv
from utils import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from PIL import Image

def plot_on_image(image_path, points_2d):
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    
    plt.figure(figsize=(7, 5))
    plt.imshow(image)
    plt.scatter(points_2d[:, 0], points_2d[:, 1], color='red', s=10, alpha=0.6)  # scatter plot on the image
    plt.axis('off')  
    plt.show()

def scale_projected_points(projected_pts, image_width, image_height):
    # Normalize points to the range [0, 1]
    min_x, max_x = np.min(projected_pts[:, 0]), np.max(projected_pts[:, 0])
    min_y, max_y = np.min(projected_pts[:, 1]), np.max(projected_pts[:, 1])

    # Scale points based on image dimensions
    projected_pts[:, 0] = (projected_pts[:, 0] - min_x) / (max_x - min_x) * image_width
    projected_pts[:, 1] = (projected_pts[:, 1] - min_y) / (max_y - min_y) * image_height

    return projected_pts

def read_vtk_file(vtk_file):
    mesh = pv.read(vtk_file)
    points = mesh.points
    if mesh.faces.size > 0:
        faces = mesh.faces.reshape(-1, 4)[:, 1:]  # Reshape and skip the first column if using VTK POLYDATA with 'vtkCellArray' format
    else:
        faces = np.array([])  # Handle the case where no faces data is present
    return points, faces


def plot_mesh_wireframe_on_image_cmap(image, points_2d, points_3d, faces):
    plt.figure(figsize=(7, 5))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax = plt.gca()  

    scalars = points_3d[:, 2]  
    norm = colors.Normalize(vmin=np.min(scalars), vmax=np.max(scalars))
    cmap = plt.get_cmap('viridis')
    sm = cmx.ScalarMappable(norm=norm, cmap=cmap)
    #iterate through each face and draw lines between vertices
    for face in faces:
        for i in range(len(face)):
            start_point = points_2d[face[i]]
            end_point = points_2d[face[(i + 1) % len(face)]]
            z_val = (points_3d[face[i], 2] + points_3d[face[(i + 1) % len(face)], 2]) / 2
            color = sm.to_rgba(z_val)
            ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color=color)

 
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', label='Z-Value')
    ax.axis('off')  
    ax.set_title("3D Mesh Wireframe on 2D Image with Colormap")
    plt.show()


def plot_mesh_wireframe_on_image(image, points_2d, faces):
    plt.figure(figsize=(7, 5))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Iterate through each face and draw lines between vertices
    for face in faces:
        for i in range(len(face)):
            start_point = points_2d[face[i]]
            end_point = points_2d[face[(i + 1) % len(face)]]  # Connect vertices cyclically
            plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='red')
    
    plt.axis('off')  
    plt.title("3D Mesh Wireframe on 2D Image")
    plt.show()

def get_camera_parameters(image_height, image_width,rotation_vector, translation_vector, image_center):
    """
    Generate camera intrinsic matrix and set rotation and translation vectors for extrinsic parameters.
    """
   
    #focal_length_px = image_width / (2 * np.tan(np.radians(30)))
    focal_length_px=2.22*(image_width/36)

    cx, cy, _ = image_center

    intrinsic_matrix = np.array([
        [focal_length_px, 0, cx],
        [0, focal_length_px, cy],
        [0, 0, 1]
    ])

    
    rotation_matrix = np.array(rotation_vector).reshape(3,3)
    translation_matrix = np.array(translation_vector).reshape(3,1)  
    return intrinsic_matrix, rotation_matrix, translation_matrix


def main():
    image_path='/Users/ekole/Dev/gut_slam/pose_estim/rendering/mesh1.png'
    image = cv2.imread(image_path)
    yaw=np.radians(0)
    pitch=np.radians(0)
    roll=np.radians(0)
    if image is None:
        print("Error: Image not found.")
        return
 
    image_height, image_width = image.shape[:2]
    image_center = (image_height/2, image_width/2, 0)
    radius = 500 # in mm  use approriate measuements along with appropriate camera parameters to match scaling and projection
    height = 1000
    vanishing_pts = (0, 0, 10)
    center = image_center
    resolution = 500
    # warp_field = WarpField(radius, height, vanishing_pts, center, resolution)

    #init rot and trans mat
    z_vector = np.array([0, 0, 10]) #vp from vanishing pooint
    z_unit_vector = z_vector / np.linalg.norm(z_vector) 
    x_camera_vector = np.array([1, 0, 0])
    y_vector = np.cross(z_unit_vector, x_camera_vector)
    x_vector = np.cross(z_unit_vector, y_vector)
    x_vector /= np.linalg.norm(x_vector)
    y_vector /= np.linalg.norm(y_vector)
    rot_mat = np.vstack([x_vector, y_vector, z_unit_vector]).T
    trans_mat=np.array([0, 0, 1])

    intrinsic_matrix, rotation_matrix, translation_vector =get_camera_parameters(
    image_height, image_width, rot_mat, trans_mat,center)
    

    projector = Project3D_2D_cam(intrinsic_matrix, rotation_matrix, translation_vector)
    
    mesh_pts, mesh_edges=read_vtk_file('./rendering/mesh1.ply')

    projected_pts=projector.project_points(points_3d=mesh_pts)
    projected_pts=scale_projected_points(projected_pts, image_width, image_height)
  
    plot_mesh_wireframe_on_image(image, projected_pts, mesh_edges)
    plot_on_image(image_path, projected_pts)
    #plot_mesh_wireframe_on_image_cmap(image, projected_pts, mesh_pts, mesh_edges)
    #visualize_mesh_from_points(cylinder_points)

    

  

if __name__ == "__main__":
    main()

