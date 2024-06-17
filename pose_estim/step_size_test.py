''' This script determines the step size for rho based on the desired number of steps.'''
import numpy as np

def determine_rho_step_size(rho_range, desired_steps):
    # Calculate the width of the rho range
    range_width = rho_range[1] - rho_range[0]
    
    # Calculate the step size
    step_size = range_width / desired_steps
    
    return step_size

rho_range = (0.5, 4.0)
desired_steps = 5 
rho_step_size = determine_rho_step_size(rho_range, desired_steps)
print(f"Rho step size: {rho_step_size:.2f}")

def determine_rho_range_from_radius(radius, min_fraction=0.005):
    min_rho = radius * min_fraction
    max_rho = radius
    return (min_rho, max_rho)


radius = 100
rho_range = determine_rho_range_from_radius(radius)
print(f"Derived rho range from radius {radius}: {rho_range}")



































# import cv2
# import numpy as np
# from scipy.optimize import least_squares
# from utils import *
# import matplotlib.pyplot as plt
# import os
# import time
# from tqdm import tqdm

# import cProfile, pstats, io

# def load_frames_from_video(video_path):
#     frames = []
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error opening video file {video_path}")
#         return frames

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break  
#         frames.append(frame)

#     cap.release()  
#     return frames

# def load_frames_from_directory(directory_path):
#     frames = []
#     valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
#     image_files = sorted([f for f in os.listdir(directory_path) if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(directory_path, f))])

#     for image_file in image_files:
#         image_path = os.path.join(directory_path, image_file)
#         frame = cv2.imread(image_path)
#         if frame is not None:
#             frames.append(frame)
#         else:
#             print(f"Warning: Could not read image {image_file}")

#     return frames

# def rotation_matrix_to_vector(rotation_matrix):
#     rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
#     return rotation_vector

# optimization_errors = []

# def objective_function(params, points_3d, points_2d_observed, image, intrinsic_matrix, k, g_t, gamma, b_mesh_deformation, lambda_ortho, lambda_det, texture, pbar):
#     rotation_matrix = params[:9].reshape(3, 3)
#     translation_vector = params[9:12]
#     control_points = params[12:-2].reshape(11, 11, 3)
#     lambda_ortho = params[-2]
#     lambda_det = params[-1]
#     image_height, image_width = image.shape[:2]
#     center = (image_width/2, image_height/2, 0)

#     points_3d = BMeshDeformation(radius=50, center=center)
#     deformed_points = points_3d.b_mesh_deformation(control_points)

#     projector = Project3D_2D_cam(intrinsic_matrix, rotation_matrix, translation_vector)
#     projected_2d_pts = projector.project_points(deformed_points)
#     if projected_2d_pts.shape[0] > points_2d_observed.shape[0]:
#         projected_2d_pts = projected_2d_pts[:points_2d_observed.shape[0], :]
#     elif projected_2d_pts.shape[0] < points_2d_observed.shape[0]:
#         points_2d_observed = points_2d_observed[:projected_2d_pts.shape[0], :]
#     points_2d_observed = points_2d_observed.reshape(-1, 2)
    
#     # Compute light intensity error
#     light_intensity_error = []
#     for pt2d, pt3d in zip(projected_2d_pts, deformed_points):
#         x, y, z = pt3d
#         L = calib_p_model(x, y, z, k, g_t, gamma)
#         if 0 <= int(pt2d[0]) < image.shape[1] and 0 <= int(pt2d[1]) < image.shape[0]:
#             pixel_intensity = get_pixel_intensity(image[int(pt2d[1]), int(pt2d[0])])
#             light_intensity_error.append(abs(pixel_intensity - L))
#         else:
#             light_intensity_error.append(0)

#     # Compute texture intensity error
#     texture_intensity_error = []
#     texture_height, texture_width, _ = texture.shape
#     for pt2d, pt3d in zip(projected_2d_pts, deformed_points):
#         if 0 <= int(pt2d[0]) < texture_width and 0 <= int(pt2d[1]) < texture_height:
#             u = int(pt2d[0] * (texture_width - 1))
#             v = int(pt2d[1] * (texture_height - 1))
#             texture_intensity = texture[v, u]
#             texture_intensity_error.append(abs(texture_intensity - L))
#         else:
#             texture_intensity_error.append(0)
    
#     # Compute photometric error as the sum of light intensity error and texture intensity error
#     photometric_error = np.sum(light_intensity_error) + np.sum(texture_intensity_error)

#     ortho_constraint = np.dot(rotation_matrix, rotation_matrix.T) - np.eye(3)
#     det_constraint = np.linalg.det(rotation_matrix) - 1

#     objective = photometric_error
#     objective += lambda_ortho * np.linalg.norm(ortho_constraint, 'fro')**2
#     objective += lambda_det * det_constraint**2

#     pbar.update(1)

#     return objective
# def optimize_params(points_3d, points_2d_observed, image, intrinsic_matrix, initial_params, k, g_t, gamma, texture, frame_idx, pbar):
#     global optimization_errors
#     optimization_errors = []
#     num_params = len(initial_params)
#     lower_bounds = [-np.inf] * num_params
#     upper_bounds = [np.inf] * num_params
#     lower_bounds[-2:] = [0, 0]
#     upper_bounds[-2:] = [np.inf, np.inf]

#     with tqdm(total=frame_idx, desc=f"Optimizing frame {frame_idx}") as pbar:
#         result = least_squares(
#             objective_function,
#             initial_params,
#             args=(points_3d, points_2d_observed, image, intrinsic_matrix, k, g_t, gamma, texture, 1, 1, pbar), #1,1 lambda ortho lamda det init
#             method='dogbox',
#             max_nfev=50,
#             gtol=1e-8,
#             tr_solver='lsmr'
#         )
    
#     log_errors(optimization_errors, frame_idx)
#     return result.x

# def log_errors(errors, frame_idx):
#     folder_path = './logs'
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
    
#     file_path = os.path.join(folder_path, 'optimization_errors_all_frames.txt')
   
#     with open(file_path, 'a') as f:
#         f.write(f"Frame {frame_idx}\n")
#         for idx, error in enumerate(errors):
#             f.write(f"Iteration {idx + 1}: Reprojection Error: {error['reprojection_error']:.4f}, Photometric Error: {error['photometric_error']:.4f}\n")
#         mean_reprojection_error = np.mean([error['reprojection_error'] for error in errors])
#         mean_photometric_error = np.mean([error['photometric_error'] for error in errors])
#         f.write(f"Mean Reprojection Error: {mean_reprojection_error:.4f}\n")
#         f.write(f"Mean Photometric Error: {mean_photometric_error:.4f}\n\n")

# def process_frame(image, intrinsic_matrix, initial_params, points_3d, k, g_t, gamma, b_mesh_deformation, frame_idx):
#     points_2d_observed = Project3D_2D_cam(intrinsic_matrix, initial_params[:9].reshape(3, 3), initial_params[9:12]).project_points(points_3d)
#     optimized_params = optimize_params(points_3d, points_2d_observed, image, intrinsic_matrix, initial_params, k, g_t, gamma, b_mesh_deformation, frame_idx)
#     return optimized_params

# def detect_feature_points(image):
#     orb = cv2.ORB_create()
#     kp = orb.detect(image, None)
#     kp = cv2.KeyPoint_convert(kp)
#     return np.array(kp)

# def log_optim_params(optimized_params, frame_idx):
#     folder_path = './logs'
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
#     file_path = os.path.join(folder_path, 'optimized_params_all_frames.txt')
    
#     with open(file_path, 'a') as f:
#         f.write(f"Frame {frame_idx}\n")
#         f.write("Optimized Parameters:\n")
#         f.write("Rotation Matrix: \n")
#         f.write(str(optimized_params[:9].reshape(3, 3)) + "\n")
#         f.write("Translation Vector: \n")
#         f.write(str(optimized_params[9:12]) + "\n")
#     control_points_file = os.path.join(folder_path, f'optimized_control_points_frame_{frame_idx}.txt')
#     np.savetxt(control_points_file, optimized_params[12:-2].reshape(-1, 3))

# def main():
#     image_path = '/Users/ekole/Dev/gut_slam/gut_images/Frames_S2000/0774.png'
#     texture_path = './tex/colon_DIFF.png'

#     print("Optimization started...")
#     start_time = time.time()

#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Error: Could not read image {image_path}")
#         return

#     texture = cv2.imread(texture_path)
#     if texture is None:
#         print(f"Error: Could not read texture image {texture_path}")
#         return

#     image_height, image_width = image.shape[:2]
#     intrinsic_matrix = np.array([
#         [1181.7734, 0, 385.0671],
#         [0, 1181.7734, 282.0998],
#         [0, 0, 1]
#     ])

#     initial_rotation_matrix = np.eye(3)
#     initial_translation_vector = np.array([0, 0, 100])
#     initial_control_points = np.zeros((11, 11, 3))
#     initial_lambda_ortho = 1
#     initial_lambda_det = 1

#     initial_params = np.hstack([
#         initial_rotation_matrix.ravel(),
#         initial_translation_vector.ravel(),
#         initial_control_points.ravel(),
#         initial_lambda_ortho,
#         initial_lambda_det
#     ])

#     points_3d = detect_feature_points(image)
#     optimized_params = process_frame(image, intrinsic_matrix, initial_params, points_3d, 1, 1, 1, texture, 0)
#     log_optim_params(optimized_params, 0)
    
#     end_time = time.time()
#     print(f"Optimization completed in {end_time - start_time:.2f} seconds.")
    
# if __name__ == '__main__':
#     main()
