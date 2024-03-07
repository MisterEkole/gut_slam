
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2




# def create_cylinder(radius, height, num_pixel_points=10):
#     num_points=int(2*np.pi*radius*num_pixel_points)
#     theta = np.linspace(0, 2 * np.pi, num_points) #gen array of angles, each angle is a point on circumference
#     x = radius * np.cos(theta)
#     y = radius * np.sin(theta)
#     z = np.linspace(0, height, num_points)
#     return np.column_stack((x, y, z))

def create_cylinder(radius, height, num_pixel_points=5):  #exponential grid spacing
    num_points = int(2 * np.pi * radius * num_pixel_points)
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    lambda_val = 4
    zmin = 0.01  # Replace with your desired minimum axial value
    z = np.exp(lambda_val * np.linspace(np.log(zmin)/lambda_val, np.log(height)/lambda_val, num_points))
    return np.column_stack((x, y, z))

def update_cylinder_position(event):
    if event.inaxes == ax_2d:
        new_center = np.array([event.xdata, event.ydata])
        update_cylinder(new_center)

def update_cylinder(new_center):
    global cylinder_points, projected_points, mesh
    # Adjust the cylinder center
    cylinder_points[:, :2] += new_center - np.mean(cylinder_points[:, :2], axis=0)
    
    # Project 3D points onto the updated image
    projected_points = project_points_onto_image(cylinder_points)
    
    # Update the scatter plot for 2D projection
    #scatter_2d.set_offsets(projected_points)
    scatter_2d.set_offsets(projected_points[:, :2])
    
    # Update the 3D plot for 3D projection
    #ax_3d.cla()  # Clear the previous 3D plot
    
    # Plot the cylinder as a mesh
    mesh = ax_3d.plot_trisurf(cylinder_points[:, 0], cylinder_points[:, 1], cylinder_points[:, 2], color='b', alpha=1)
    
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.zaxis.set_rotate_label(False)
    ax_3d.set_zlabel('Z', rotation=90, labelpad=10)
    ax_3d.set_title('3D Projection of Cylinder')
    # Set new limits for 3D plot based on updated cylinder points
    ax_3d.set_xlim([np.min(cylinder_points[:, 0]), np.max(cylinder_points[:, 0])])
    ax_3d.set_ylim([np.min(cylinder_points[:, 1]), np.max(cylinder_points[:, 1])])
    ax_3d.set_zlim([np.min(cylinder_points[:, 2]), np.max(cylinder_points[:, 2])])

    plt.draw()


def get_camera_parameters():
    # Camera parameters (intrinsic matrix)
    fx = 50.0  # focal length in x direction
    fy = 50.0  # focal length in y direction
    cx = 120.0  # x-coordinate of the principal point
    cy = 140.0  # y-coordinate of the principal point

    camera_matrix = np.array([[fx, 0, cx],
                             [0, fy, cy],
                             [0, 0, 1]])

    return camera_matrix

# Function to project 3D points onto a 2D image
def project_points_onto_image(points_3d):
    cam=get_camera_parameters()
    homogeneous_coords = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    homogeneous_coords = homogeneous_coords[:, :3]
    points_2d_homogeneous = np.dot(homogeneous_coords, cam.T)
    
    # Check for division by zero
    mask = (points_2d_homogeneous[:, 2] != 0)
    
    # Avoid division by zero and replace invalid values with nan
    points_2d = np.empty_like(points_2d_homogeneous[:, :2])
    points_2d[mask] = points_2d_homogeneous[mask, :2] / points_2d_homogeneous[mask, 2:]
    points_2d[~mask] = np.nan
    
    return points_2d


image_path = '/Users/ekole/Dev/gut_slam/gut_images/image3.jpeg' 
image = cv2.imread(image_path)
image_resolution=(image.shape[1], image.shape[0])
# Generate 3D points for a cylinder
radius = 5.0
height = 1.0


cylinder_points = create_cylinder(radius, height) #3d cylinder points

# Project 3D points onto the image
projected_points = project_points_onto_image(cylinder_points)


# Display the results on the image and 3D plots
fig, (ax_2d, ax_3d) = plt.subplots(1, 2, figsize=(10, 5))

# 2D Projection on Image
img = ax_2d.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
scatter_2d = ax_2d.scatter(projected_points[:, 0], projected_points[:, 1], c='r', marker='o', label='Projected Points')
ax_2d.set_title('Projected Points on Image')
ax_2d.legend()

# Create an Axes3D instance for 3D projection
ax_3d = fig.add_subplot(122, projection='3d')

# Plot the cylinder as a mesh
mesh = ax_3d.plot_trisurf(cylinder_points[:, 0], cylinder_points[:, 1], cylinder_points[:, 2], color='r', alpha=0.7)

ax_3d.set_xlabel('X')
ax_3d.set_ylabel('Y')
ax_3d.set_zlabel('Z')
ax_3d.zaxis.set_rotate_label(False)
ax_3d.set_zlabel('Z', rotation=90, labelpad=10)
ax_3d.set_title('3D Projection of Cylinder')

# Connect the mouse click event to the update_cylinder_position function
fig.canvas.mpl_connect('button_press_event', update_cylinder_position)

plt.show()
