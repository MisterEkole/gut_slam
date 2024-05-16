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

def densify_points(points, num_divisions=5):
    """
    Create additional points along the edges to densify the projected points for better visualization.
    
    Parameters:
    - points: array of projected points.
    - num_divisions: number of segments to split each edge into.

    Returns:
    - A new array with the original and interpolated points.
    """
    if points.shape[0] < 2:
        return points

    all_points = []
    for i in range(len(points) - 1):
        start_point = points[i]
        end_point = points[i + 1]
        divisions = [start_point + (end_point - start_point) * j / num_divisions for j in range(num_divisions + 1)]
        all_points.extend(divisions)

    # Ensure the last point is added
    all_points.append(points[-1])
    return np.array(all_points)


def plot_mesh_wireframe_on_image(image, points_2d, faces, num_divisions=5):
    """
    Plot a densified wireframe mesh on a 2D image.

    Parameters:
    - image: An image array on which the mesh will be plotted.
    - points_2d: 2D projected points of the mesh vertices.
    - faces: Index array of the faces, where each face is represented as a list of indices into points_2d.
    - num_divisions: Number of divisions to densify each edge of the mesh.
    """
    plt.figure(figsize=(7, 5))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Iterate through each face and draw densified lines between vertices
    for face in faces:
        for i in range(len(face)):
            start_idx = face[i]
            end_idx = face[(i + 1) % len(face)]  # Connect vertices cyclically
            edge_points = np.array([points_2d[start_idx], points_2d[end_idx]])
            #densified_edge_points = densify_points(edge_points, num_divisions=num_divisions)
            plt.plot(edge_points[:, 0], edge_points[:, 1], color='red')
    
    plt.axis('off')  
    plt.title("3D Mesh Wireframe on 2D Image")
    plt.show()

def get_camera_parameters(image_height, image_width,rotation_matrix, translation_vector, image_center):
    """
    Generate camera intrinsic matrix and set rotation and translation vectors for extrinsic parameters.
    """
   
    focal_length_px = image_height / (2 * np.tan(np.radians(60)/2))
    #focal_length_px=2.22*(image_width/36)
   

    cx, cy, _ = image_center

    intrinsic_matrix = np.array([
        [focal_length_px, 0, cx],
        [0, focal_length_px, cy],
        [0, 0, 1]
    ])

    
    #rotation_matrix = np.array(rotation_vector).reshape(3,3)
    translation_matrix = np.array(translation_vector).reshape(3,1)  
    return intrinsic_matrix, rotation_matrix, translation_matrix

def calculate_rotation_matrix(position, focal_point, view_up):
    """
    Calculate the rotation matrix to align the camera direction with the focal point.

    Parameters:
    - position: Camera position.
    - focal_point: The point where the camera is looking.
    - view_up: The up direction for the camera.
    """
    forward = (focal_point - position) / np.linalg.norm(focal_point - position)
    right = np.cross(forward, view_up)
    right = right / np.linalg.norm(right)
    corrected_up = np.cross(right, forward)
    
    rotation_matrix = np.stack([right, corrected_up, -forward], axis=1)
    return rotation_matrix


def main():
    image_path='/Users/ekole/Dev/gut_slam/pose_estim/rendering/mesh9.png'
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image not found.")
        return
 
    image_height, image_width = image.shape[:2]
    image_center = (image_height/2, image_width/2, 0)
    center = image_center
    # cam_info={'position':(4.310669534819416, 5.074316599765719, 4.636957423146306),
    #     'focal_point':(-0.14862758317728766, 0.6150194817690126, 0.17766030514960568),
    #         'view_up': (0.0, 0.0, 1.0)}
    cam_info={'position':(4.442435128936857, 4.892312186729204, 4.68719624206537),
        'focal_point':(-0.016861989059850247, 0.43301506873249845, 0.2278991240686753),
            'view_up': (0.0, 0.0, 1.0)}

    #init rot and trans mat
    # z_vector = np.array([0, 0, 10]) #vp from vanishing pooint
    # z_unit_vector = z_vector / np.linalg.norm(z_vector) 
    # x_camera_vector = np.array([1, 0, 0])
    # y_vector = np.cross(z_unit_vector, x_camera_vector)
    # x_vector = np.cross(z_unit_vector, y_vector)
    # x_vector /= np.linalg.norm(x_vector)
    # y_vector /= np.linalg.norm(y_vector)
    # rot_mat = np.vstack([x_vector, y_vector, z_unit_vector]).T
    # trans_mat=np.array([10, 10, 10])
    #trans_mat=np.array([4.310669534819416, 5.074316599765719, 4.636957423146306])

    # Calculate Rotation and Translation from Camera Information
    cam_position = np.array(cam_info['position'])
    cam_focal_point = np.array(cam_info['focal_point'])
    cam_view_up = np.array(cam_info['view_up'])
    trans_mat=cam_position
    rot_mat=calculate_rotation_matrix(cam_position,cam_focal_point,cam_view_up)

    intrinsic_matrix, rotation_matrix, translation_vector =get_camera_parameters(
    image_height, image_width, rot_mat, trans_mat,center)
    

    projector = Project3D_2D_cam(intrinsic_matrix, rotation_matrix, translation_vector)
    
    mesh_pts, mesh_edges=read_vtk_file('./rendering/mesh9.ply')

    projected_pts=projector.project_points(points_3d=mesh_pts)
    projected_pts=scale_projected_points(projected_pts, image_width, image_height)
  
    # plot_mesh_wireframe_on_image(image, projected_pts, mesh_edges)
    # plot_on_image(image_path, projected_pts)
    #load_and_plot_mesh('./rendering/textured_gut_mesh.ply')
    load_and_plot_mesh('./rendering/mesh9.ply')


    

  

if __name__ == "__main__":
    main()

