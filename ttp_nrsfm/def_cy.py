''' Deformable cylindrical mesh & projection on 2D image using Open3D and OpenCV '''

import open3d as o3d
import numpy as np
import cv2

def create_deformable_cylinder(radius=1.0, height=2.0, center=(10, 3, 28), theta_resolution=10, deformation_factor=0.1):
    """Creates a deformable cylindrical mesh using Open3D.

    Args:
        radius: Radius of the cylinder.
        height: Height of the cylinder.
        center: Center point of the cylinder's base.
        theta_resolution: Number of points around the circumference.
        deformation_factor: Magnitude of deformation.

    Returns:
        An Open3D TriangleMesh representing the deformable cylinder.
    """
    # Create a cylinder without deformation
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
   

    # Convert vertices to NumPy array for modification
    vertices = np.asarray(mesh.vertices)

    # Apply deformation along the length
    vertices[:, 2] += deformation_factor * np.sin(vertices[:, 0] / radius)

    # Update the mesh vertices with the modified values
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    # Translate to the specified center
    mesh.translate(center)

    return mesh

def create_deformable_cone(radius=1.0, height=2.0, center=(10, 3, 28), theta_resolution=10, deformation_factor=0.1):
    """Creates a deformable cone mesh using Open3D.

    Args:
        radius: Radius of the cone.
        height: Height of the cone.
        center: Center point of the cone's base.
        theta_resolution: Number of points around the circumference.
        deformation_factor: Magnitude of deformation.

    Returns:
        An Open3D TriangleMesh representing the deformable cone.
    """
    # Create a cone without deformation
    mesh = o3d.geometry.TriangleMesh.create_cone(radius=radius, height=height)

    # Convert vertices to NumPy array for modification
    vertices = np.asarray(mesh.vertices)

    # Apply deformation along the length
    vertices[:, 2] += deformation_factor * np.sin(vertices[:, 0] / radius)

    # Update the mesh vertices with the modified values
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    # Translate to the specified center
    mesh.translate(center)

    return mesh

def display_deformable_cylinder(mesh):
    """Displays the deformable cylindrical mesh using Open3D."""
    o3d.visualization.draw_geometries([mesh])
    


def project_deformable_cylinder(mesh, image, camera_matrix, distortion_coefficients):
    """Projects a deformable cylinder mesh onto a 2D image using Open3D."""
    # Define object points in the local coordinate system
    object_points = np.array([
        [-0.5, -0.5, 0.0],  # Bottom square corner
        [0.5, -0.5, 0.0],
        [0.5, 0.5, 0.0],
        [-0.5, 0.5, 0.0],
        [-0.5, -0.5, 1.0],  # Top square corner
        [0.5, -0.5, 1.0],
        [0.5, 0.5, 1.0],
        [-0.5, 0.5, 1.0]
    ])

    # Project the object points into image coordinates
    image_points, _ = cv2.projectPoints(object_points, np.eye(3), np.zeros(3), camera_matrix, distortion_coefficients)

    # Draw lines on the image
    for i in range(4):
        cv2.line(image, tuple(image_points[i].ravel().astype(int)), tuple(image_points[(i + 1) % 4].ravel().astype(int)), (0, 255, 0), 2)  # Bottom square
        cv2.line(image, tuple(image_points[i + 4].ravel().astype(int)), tuple(image_points[(i + 1) % 4 + 4].ravel().astype(int)), (0, 255, 0), 2)  # Top square
        cv2.line(image, tuple(image_points[i].ravel().astype(int)), tuple(image_points[i + 4].ravel().astype(int)), (0, 255, 0), 2)  # Side lines

    return image

# Load an image
image = cv2.imread('/Users/ekole/Dev/gut_slam/gut_images/image1.jpeg')
distortion_coefficients = np.zeros(5)

# Camera Parameters
focal_length = 35.0
image_center = (image.shape[1] / 2, image.shape[0] / 2)

# Create the camera matrix
camera_matrix = np.array([
    [focal_length, 0, image_center[0]],
    [0, focal_length, image_center[1]],
    [0, 0, 1]
])

# Create an initial deformable cylinder using Open3D
initial_cylinder_mesh = create_deformable_cylinder(radius=500, height=40, center=(1, 2, 0.5))
initial_cone_mesh = create_deformable_cone(radius=500, height=700, center=(1, 2, 0.5))

# Display the deformable cylinder using Open3D
#display_deformable_cylinder(initial_cylinder_mesh)
display_deformable_cylinder(initial_cone_mesh)

# Project the initial cylinder onto the image and get the result
result_image = project_deformable_cylinder(initial_cylinder_mesh, image.copy(), camera_matrix, distortion_coefficients)

# Display the result
cv2.imshow('Projection', result_image)
cv2.waitKey(0)
