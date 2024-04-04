import cv2
import numpy as np
import pyvista as pv
from utils import *
import matplotlib.pyplot as plt

def main():
    image_path = '/Users/ekole/Dev/gut_slam/gut_images/image1.jpeg'
    image = cv2.imread(image_path)
    
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
   


    # intrinsic_matrix = np.array([[735.37, 0, image_height/2],
    #                          [0, 552.0, image_width/2],
    #                          [0, 0, 1]])

    rot_mat = np.array([[1, 0, 0],   
                            [0, 1, 0],
                            [0, 0, 1]])

    trans_mat = np.array([0, 0, 0]) 

    


    intrinsic_matrix, rotation_matrix, translation_vector = Project3D_2D_cam.get_camera_parameters(
    image_height, image_width, rot_mat, trans_mat)

    projector = Project3D_2D_cam(intrinsic_matrix, rotation_matrix, translation_vector)


    # camera_matrix = Project3D_2D.get_camera_parameters(image_height, image_width)
    # projector = Project3D_2D(camera_matrix)

    # Apply deformation to the cylinder (optional)
    #warp_field.apply_shrinking(start_radius=None, end_radius=None)
    #warp_field.apply_deformation(strength=0.1,frequency=1)
    warp_field.apply_deformation_axis(strength=5,frequency=10)
    

    # Extract points from the cylinder
    cylinder_points = warp_field.extract_pts()
    print(cylinder_points)

    projected_pts=projector.project_points(points_3d=cylinder_points)

    #print(projected_pts)
    print(warp_field.cylinder.points)
    k=2.5
    g_t=2.0
    gamma=2.2

    #precompute values for cylinder points and image points before doing the photometric projection


    # for point in cylinder_points:
    #     for row in range(image.shape[0]):
    #         for col in range(image.shape[1]):
    #             u = image[row, col]
    #             x,y,z=point
    #             L=calib_p_model(x,y,z,k,g_t,gamma)
    #             I=get_pixel_intensity(u)
    #             C=cost_func(I,L)
    #             print("Pixel intensity: ",I, "Cost function: ",C, "Light intensity: ",L)
      
       
 


  

    plt.imshow(image)
    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)  # Inverted y-axis to match image coordinate system
    plt.scatter(projected_pts[:, 0], projected_pts[:, 1], color='red', s=10)  # Increased size for visibility
    plt.show()


    # plt.imshow(image)
    # plt.scatter([image.shape[1] / 2], [image.shape[0] / 2], color='red', s=10)  # Center of the image
    # plt.show()

    #plot 3D cylinder

    # plotter = pv.Plotter()
    # plotter.add_mesh(warp_field.cylinder, show_edges=True, color='lightblue', edge_color='blue')
    # plotter.add_title("Deformable Cylinder Visualization")
    # plotter.show()

    #display_point_cloud(cylinder_points)


  

if __name__ == "__main__":
    main()

