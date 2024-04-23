import cv2
import numpy as np
import pyvista as pv
from utils import *
import matplotlib.pyplot as plt
from PIL import Image

def initialize_control_points(radius, height, M, N):
    """
    Initialize a grid of control points on the surface of a cylinder.
    
    Parameters:
    - radius: The radius of the cylinder.
    - height: The height of the cylinder.
    - M: Number of vertical control points (height divisions).
    - N: Number of circumferential control points (angular divisions).
    
    Returns:
    - control_points: A numpy array of shape (M, N, 3).
    """
    # Define the heights and angles
    heights = np.linspace(0, height, M)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # Create meshgrid for heights and angles
    heights_grid, angles_grid = np.meshgrid(heights, angles, indexing='ij')

    # Calculate x, y, z positions
    x_positions = radius * np.cos(angles_grid)
    y_positions = radius * np.sin(angles_grid)
    z_positions = heights_grid

    # Stack the positions into a 3D array
    control_points = np.stack((x_positions, y_positions, z_positions), axis=-1)

    return control_points

def initial_control_points(all_points, vp, height, width):
    M, N = height, width
    heights = np.linspace(0, 1, M)  # Assuming normalized heights for simplicity
    control_points = np.zeros((M, N, 3))  # Initialize control points array

    # Compute angles and control points
    for i, point in enumerate(all_points):
        p_minus_vp = point - vp
        theta = np.arctan2(p_minus_vp[1], p_minus_vp[0])
        h = np.linalg.norm(p_minus_vp)**2  # Squared norm as the 'height' metric
        m_index = int(h * (M - 1))  # Map height to the closest index
        n_index = int(((theta + np.pi) / (2 * np.pi)) * (N - 1))  # Map angle to index

        # Assuming control points need to be set or adjusted here:
        control_points[m_index, n_index, :] = point  # Place or modify the point at this control point location

    return control_points

def extract_points_from_image(image_path):
    # Load image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    data = np.array(img)

    # Extract points where brightness is above a threshold
    threshold = 128  
    points = np.column_stack(np.where(data > threshold))

    # Normalize points based on image dimensions
    height, width = data.shape
    points = points / np.array([height, width])

    # Append dummy z-coordinate if needed for compatibility
    points = np.hstack([points, np.zeros((points.shape[0], 1))])

    return points



def main():
    #image_path = '/Users/ekole/Dev/gut_slam/gut_images/image1.jpeg'
    image_path = '/Users/ekole/Dev/gut_slam/gut_images/FrameBuffer_0038.png'
    image = cv2.imread(image_path)
    yaw=np.radians(0)
    pitch=np.radians(0)
    roll=np.radians(0)
    if image is None:
        print("Error: Image not found.")
        return
    



    image_height, image_width = image.shape[:2]
    image_center = (image_width / 2, image_height / 2, 0)
    radius = 500  # in mm  use approriate measuements along with appropriate camera parameters to match scaling and projection
    height = 1000
    vanishing_pts = (0, 0, 10)
    center = image_center
    resolution = 100
    warp_field = WarpField(radius, height, vanishing_pts, center, resolution)

    a_values = np.zeros((image_height, image_width, 3)) 
    b_values = np.zeros((image_height, image_width))  
    

    
    for row in range(image_height):
        for col in range(image_width):
            pixel = image[row, col]
            p_minus_vp = np.array([row, col, 0]) - np.array(vanishing_pts)
            a_values[row, col] = p_minus_vp
            b_values[row, col] = np.arctan2(p_minus_vp[1], p_minus_vp[0])

           
    
    a_values=np.array(a_values)
    a_values=np.max(a_values/(np.linalg.norm(np.mean(a_values,axis=1))))
    b_values=np.array(b_values)
    b_values=np.max(b_values/(np.linalg.norm(b_values)))

    print(a_values)
    print(b_values)


    # intrinsic_matrix = np.array([[735.37, 0, image_height/2],
    #                          [0, 552.0, image_width/2],
    #                          [0, 0, 1]])

    # rot_mat = np.array([[1, 0, 0],   
    #                         [0, 1, 0],
    #                         [0, 0, 1]])

    rot_mat=euler_to_rot_mat(yaw,pitch,roll)

    trans_mat = np.array([0, 0, 0]) 

    


    intrinsic_matrix, rotation_matrix, translation_vector = Project3D_2D_cam.get_camera_parameters(
    image_height, image_width, rot_mat, trans_mat)

    projector = Project3D_2D_cam(intrinsic_matrix, rotation_matrix, translation_vector)
    
    # Apply deformation to the cylinder (optional)
   
    image_points = extract_points_from_image(image_path)
    control_point=np.random.rand(50,50,3)
    #np.savetxt('control_points.txt', control_point.reshape(-1, 3))

    #warp_field.b_spline_mesh_deformation(control_points=control_point, strength=10)
    warp_field.b_mesh_deformation(a=a_values, b=b_values, control_points=control_point)
    
    cylinder_points = warp_field.extract_pts()
    print(cylinder_points)

    projected_pts=projector.project_points(points_3d=cylinder_points)

    #print(projected_pts)
    
    # k=2.5
    # g_t=2.0
    # gamma=2.2

    # #precompute values for cylinder points and image points before doing the photometric projection


    # # for point in cylinder_points:
    # #     for row in range(image.shape[0]):
    # #         for col in range(image.shape[1]):
    # #             u = image[row, col]
    # #             x,y,z=point
    # #             L=calib_p_model(x,y,z,k,g_t,gamma)
    # #             I=get_pixel_intensity(u)
    # #             C=cost_func(I,L)
    # #             #print("Pixel intensity: ",I, "Cost function: ",C, "Light intensity: ",L)
      
       
 


  

    # plt.imshow(image)
    # plt.xlim(0, image.shape[1])
    # plt.ylim(image.shape[0], 0)  # Inverted y-axis to match image coordinate system
    # plt.scatter(projected_pts[:, 0], projected_pts[:, 1], color='red', s=10)  # Increased size for visibility
    # plt.show()


    # plt.imshow(image)
    # plt.scatter([image.shape[1] / 2], [image.shape[0] / 2], color='red', s=10)  # Center of the image
    # plt.show()

    #plot 3D cylinder

    # plotter = pv.Plotter()
    # plotter.add_mesh(warp_field.cylinder, show_edges=True, color='lightblue', edge_color='blue')
    # plotter.add_title("Deformable Cylinder Visualization")
    # plotter.show()

    #display_point_cloud(cylinder_points)
    #visualize_point_cloud(cylinder_points)
    #point_cloud_to_mesh(cylinder_points)
    visualize_mesh_from_points(cylinder_points)
   


  

if __name__ == "__main__":
    main()

