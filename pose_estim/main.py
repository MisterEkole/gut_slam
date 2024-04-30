import cv2
import numpy as np
import pyvista as pv
from utils import *
import matplotlib.pyplot as plt
from PIL import Image


def main():
    image_path = '/Users/ekole/Dev/gut_slam/gut_images/image4.jpg'
    #image_path = '/Users/ekole/Dev/gut_slam/gut_images/FrameBuffer_0038.png'
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Image not found.")
        return

    image_height, image_width = image.shape[:2]
    image_center = (image_width / 2, image_height / 2, 0)
    radius = 500 # in mm  use approriate measuements along with appropriate camera parameters to match scaling and projection
    height = 1000
    vanishing_pts = (0, 0, 10)
    center = image_center
    resolution = 500
    warp_field = WarpField(radius, height, vanishing_pts, center, resolution)
    #warp_field.save_pts('./cylinder_points.txt')

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

    #init rot and trans mat
    z_vector = np.array([0, 0, 10]) #vp from vanishing pooint
    z_unit_vector = z_vector / np.linalg.norm(z_vector) 
    x_camera_vector = np.array([1, 0, 0])
    y_vector = np.cross(z_unit_vector, x_camera_vector)
    x_vector = np.cross(z_unit_vector, y_vector)
    x_vector /= np.linalg.norm(x_vector)
    y_vector /= np.linalg.norm(y_vector)
    rot_mat = np.vstack([x_vector, y_vector, z_unit_vector]).T
    trans_mat=np.array([0, 0, 10])



    


    intrinsic_matrix, rotation_matrix, translation_vector = Project3D_2D_cam.get_camera_parameters(
    image_height, image_width, rot_mat, trans_mat)

    projector = Project3D_2D_cam(intrinsic_matrix, rotation_matrix, translation_vector)
    
    
    #control_points=np.random.rand(5,5,3)
    #np.savetxt('control_points5.txt', control_points.reshape(-1,3))
    #print(control_point.shape)
    control_points=np.loadtxt('control_points5.txt')
  
    control_points=control_points.reshape(5,5,3)
    

    warp_field.b_mesh_deformation(a=a_values, b=b_values, control_points=control_points)
    
    
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

    #visualize_point_cloud(cylinder_points)
    #point_cloud_to_mesh(cylinder_points)
    visualize_mesh_from_points(cylinder_points)
    #visualize_and_save_mesh_from_points(cylinder_points,'./mesh.vtk',screenshot='./mesh.png')
    #visualize_mesh_on_image(cylinder_points,'projection.png')

    #plot_3d_mesh_on_image('./def_cylinder_points.txt',image_path)
   


  

if __name__ == "__main__":
    main()

