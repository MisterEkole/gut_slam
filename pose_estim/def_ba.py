''' Bundle adjustement algorithm for Pose and Deformation Estimation in GutSLAM-- this script includes
optimising the deformation parameters.

This is a single frame bundle adjustment algorithm that estimates the pose and deformation parameters

Author: Mitterand Ekole
Date: 25-03-2024
'''

import cv2
import numpy as np
from scipy.optimize import least_squares
from utils import WarpField, Project3D_2D_cam, calib_p_model, cost_func, get_pixel_intensity, reg_func
import matplotlib.pyplot as plt


def objective_function(params, points_3d, points_2d_observed, image, intrinsic_matrix, k, g_t, gamma, warp_field):
    rotation_matrix = params[:9].reshape(3, 3)
    translation_vector = params[9:12]
    deformation_strength = params[12]
    deformation_frequency = params[13]
    
    # Update deformation parameters
    warp_field.apply_deformation_axis(strength=deformation_strength, frequency=deformation_frequency)
    points_3d_deformed = warp_field.extract_pts()

    projector = Project3D_2D_cam(intrinsic_matrix, rotation_matrix, translation_vector)
    projected_2d_pts = projector.project_points(points_3d_deformed)
    reprojection_error = np.linalg.norm(projected_2d_pts - points_2d_observed, axis=1)
    lamda_reg = 1.0
    
    photometric_error = []
    for pt2d, pt3d in zip(projected_2d_pts, points_3d_deformed):
        x, y, z = pt3d
        L = calib_p_model(x, y, z, k, g_t, gamma)
        if 0 <= int(pt2d[0]) < image.shape[1] and 0 <= int(pt2d[1]) < image.shape[0]:  
            pixel_intensity = get_pixel_intensity(image[int(pt2d[1]), int(pt2d[0])])
            C = cost_func(pixel_intensity, L)
        else:
            C = 0 
        grad=np.ones((projected_2d_pts.shape[0],))
        reg=reg_func(grad)
        photometric_error.append(C+lamda_reg*reg)
    photometric_error = np.array(photometric_error)

    errors = np.concatenate([reprojection_error, photometric_error.flatten()])
    def normalize_errors(errors, target_scale=100):
        mean_error = np.mean(errors)
        scale_factor = target_scale / mean_error if mean_error else 1
        normalized_errors = errors * scale_factor
        return normalized_errors #outputs a normalised error array
    normalize_errors=normalize_errors(errors, target_scale=100)
    #print(" The error is : ", np.mean(normalize_errors)) 
    return normalize_errors  #outputs the mean error of the normalised error array

def optimize_params(points_3d, points_2d_observed, image, intrinsic_matrix, initial_params, k, g_t, gamma, warp_field):
    result = least_squares(objective_function, 
                           initial_params,
                         args=(points_3d, points_2d_observed, image, intrinsic_matrix, k, g_t, gamma, warp_field), 
                         method='lm', max_nfev=1000, gtol=1e-6)
    return result.x

def rotation_matrix_to_vector(rotation_matrix):
    """
    Convert a rotation matrix to a rotation vector using Rodrigues' formula.
    
    :param rotation_matrix: A 3x3 rotation matrix.
    :return: A rotation vector where the direction represents the axis of rotation
             and the magnitude represents the angle of rotation in radians.
    """
    rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
    return rotation_vector


def main():
    image_path = '/Users/ekole/Dev/gut_slam/gut_images/FrameBuffer_0037.png'
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return
    
    image_height, image_width = image.shape[:2]
    image_center = (image_width / 2, image_height / 2, 0)
    radius = 500  
    height = 1000
    vanishing_pts = (0, 0, 10)
    center = image_center
    resolution = 100

    warp_field = WarpField(radius, height, vanishing_pts, center, resolution)
    warp_field.apply_deformation_axis(strength=5, frequency=10)

    rot_mat = np.random.rand(3, 3)

    print("Initial Rotation Vec: \n", rotation_matrix_to_vector(rot_mat))

    #trans_mat = np.array([1, 3, 2]) 
    trans_mat = np.random.rand(3)

    print(" Initial Translation : \n", trans_mat)

    intrinsic_matrix, rotation_matrix, translation_vector = Project3D_2D_cam.get_camera_parameters(image_height, image_width, rot_mat, trans_mat)
    projector = Project3D_2D_cam(intrinsic_matrix, rotation_matrix, translation_vector)

    cylinder_points = warp_field.extract_pts()
    points_2d_observed = projector.project_points(cylinder_points)

    k = 2.5
    g_t = 2.0
    gamma = 2.2


    # Initial deformation parameters
    initial_deformation_strength = np.random.rand() 
    initial_deformation_frequency = np.random.rand() 

    print("Initial Defomation Strength: ", initial_deformation_strength)
    print("Initial Defomation Frequency: ", initial_deformation_frequency) 


    # Adjust initial_params to include deformation parameters
    initial_params = np.hstack([rotation_matrix.flatten(), translation_vector.flatten(), initial_deformation_strength, initial_deformation_frequency])

    # Optimization call
    optimized_params = optimize_params(cylinder_points, points_2d_observed, image, intrinsic_matrix, initial_params, k, g_t, gamma, warp_field)


    plt.imshow(image)
    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)
    plt.scatter(points_2d_observed[:, 0], points_2d_observed[:, 1], color='red', s=10)
    plt.show()

    print("Optimized Rotation Vector: \n", rotation_matrix_to_vector(optimized_params[:9].reshape(3, 3)))
    print("Optimized Translation Vector: \n", optimized_params[9:12])


    # Output optimized parameters
    optimized_deformation_strength = optimized_params[12]
    optimized_deformation_frequency = optimized_params[13]
    print("Optimized Deformation Strength: ", optimized_deformation_strength)
    print("Optimized Deformation Frequency: ", optimized_deformation_frequency)

  

if __name__ == "__main__":
    main()
