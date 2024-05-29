'''
-------------------------------------------------------------
Updated Bundle Adjustment Algorithm for Single Frame Pose and Control 
Point Estimation in GutSLAM

Author: Mitterand Ekole
Date: 28-05-2024
-------------------------------------------------------------
'''

import cv2
import numpy as np
from scipy.optimize import least_squares
from utils import *
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

import cProfile,pstats,io

def load_frames_from_video(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break  
        frames.append(frame)

    cap.release()  
    return frames

def load_frames_from_directory(directory_path):
    frames = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    image_files = sorted([f for f in os.listdir(directory_path) if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(directory_path, f))])

    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        frame = cv2.imread(image_path)
        if frame is not None:
            frames.append(frame)
        else:
            print(f"Warning: Could not read image {image_file}")

    return frames

def rotation_matrix_to_vector(rotation_matrix):
    rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
    return rotation_vector

optimization_errors = []

def objective_function(params, points_3d, points_2d_observed, image, intrinsic_matrix, k, g_t, gamma, b_mesh_deformation, lambda_ortho, lambda_det, pbar):
    rotation_matrix = params[:9].reshape(3, 3)
    translation_vector = params[9:12]
    control_points = params[12:-2].reshape(11, 11, 3)
    lambda_ortho = params[-2]
    lambda_det = params[-1]
    a = 0.00051301747 
    b = 0.0018595674

    deformed_points = points_3d.b_mesh_deformation(a, b, control_points)

    projector = Project3D_2D_cam(intrinsic_matrix, rotation_matrix, translation_vector)
    projected_2d_pts = projector.project_points(deformed_points)
    if projected_2d_pts.shape[0] > points_2d_observed.shape[0]:
        projected_2d_pts = projected_2d_pts[:points_2d_observed.shape[0], :]
    elif projected_2d_pts.shape[0] < points_2d_observed.shape[0]:
        points_2d_observed = points_2d_observed[:projected_2d_pts.shape[0], :]
    points_2d_observed = points_2d_observed.reshape(-1, 2)
    
    reprojection_error = np.linalg.norm(projected_2d_pts - points_2d_observed, axis=1)
    photometric_error = []
    for pt2d, pt3d in zip(projected_2d_pts, deformed_points):
        x, y, z = pt3d
        L = calib_p_model(x, y, z, k, g_t, gamma)
        if 0 <= int(pt2d[0]) < image.shape[1] and 0 <= int(pt2d[1]) < image.shape[0]:
            pixel_intensity = get_pixel_intensity(image[int(pt2d[1]), int(pt2d[0])])
            C = cost_func(pixel_intensity, L)
        else:
            C = 0
        photometric_error.append(float(C))

    photometric_error = np.array(photometric_error, dtype=float)
    
    reprojection_error /= (np.linalg.norm(reprojection_error) + 1e-8)
    photometric_error /= (np.linalg.norm(photometric_error) + 1e-8)

    global optimization_errors
    optimization_errors.append(
        {
            'reprojection_error': np.mean(reprojection_error),
            'photometric_error': np.mean(photometric_error),
        }
    )

    ortho_constraint = np.dot(rotation_matrix, rotation_matrix.T) - np.eye(3)
    det_constraint = np.linalg.det(rotation_matrix) - 1

    objective = np.sum(reprojection_error**2) + np.sum(photometric_error**2)
    objective += lambda_ortho * np.linalg.norm(ortho_constraint, 'fro')**2
    objective += lambda_det * det_constraint**2

    pbar.update(1)

    return objective



def optimize_params(points_3d, points_2d_observed, image, intrinsic_matrix, initial_params, k, g_t, gamma, frame_idx):
    global optimization_errors
    optimization_errors = []
    num_params = len(initial_params)
    lower_bounds = [-np.inf] * num_params
    upper_bounds = [np.inf] * num_params
    lower_bounds[-2:] = [0, 0]
    upper_bounds[-2:] = [np.inf, np.inf]

    with tqdm(total=frame_idx, desc=f"Optimizing frame {frame_idx}") as pbar:
        result = least_squares(
            objective_function,
            initial_params,
            args=(points_3d, points_2d_observed, image, intrinsic_matrix, k, g_t, gamma, 1, 1,frame_idx, pbar),
            method='dogbox',
            max_nfev=50,
            gtol=1e-8,
            tr_solver='lsmr'
        )
    
    log_errors(optimization_errors, frame_idx)
    return result.x

def log_errors(errors, frame_idx):
    folder_path = './logs'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    file_path = os.path.join(folder_path, 'optimization_errors_all_frames.txt')
   
    with open(file_path, 'a') as f:
        f.write(f"Frame {frame_idx}\n")
        for idx, error in enumerate(errors):
            f.write(f"Iteration {idx + 1}: Reprojection Error: {error['reprojection_error']:.4f}, Photometric Error: {error['photometric_error']:.4f}\n")
        mean_reprojection_error = np.mean([error['reprojection_error'] for error in errors])
        mean_photometric_error = np.mean([error['photometric_error'] for error in errors])
        f.write(f"Mean Reprojection Error: {mean_reprojection_error:.4f}\n")
        f.write(f"Mean Photometric Error: {mean_photometric_error:.4f}\n\n")

def process_frame(image, intrinsic_matrix, initial_params, points_3d, k, g_t, gamma, b_mesh_deformation, frame_idx):
    points_2d_observed = Project3D_2D_cam(intrinsic_matrix, initial_params[:9].reshape(3, 3), initial_params[9:12]).project_points(points_3d)
    optimized_params = optimize_params(points_3d, points_2d_observed, image, intrinsic_matrix, initial_params, k, g_t, gamma, b_mesh_deformation, frame_idx)
    return optimized_params

def detect_feature_points(image):
    orb = cv2.ORB_create()
    kp = orb.detect(image, None)
    kp = cv2.KeyPoint_convert(kp)
    return kp

def log_optim_params(optimized_params, frame_idx):
    folder_path = './logs'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, 'optimized_params_all_frames.txt')
    
    with open(file_path, 'a') as f:
        f.write(f"Frame {frame_idx}\n")
        f.write("Optimized Parameters:\n")
        f.write("Rotation Matrix: \n")
        f.write(str(optimized_params[:9].reshape(3, 3)) + "\n")
        f.write("Translation Vector: \n")
        f.write(str(optimized_params[9:12]) + "\n")
    control_points_file = os.path.join(folder_path, f'optimized_control_points_frame_{frame_idx}.txt')
    np.savetxt(control_points_file, optimized_params[12:-2].reshape(-1, 3))

def main():
    image_path = './rendering/mesh.png'
    # texture_img='./tex/colon_DIFF.png'
    # texture=pv.read_texture(texture_img)
    print("Optimization started...")
    start_time = time.time()

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    image_height, image_width = image.shape[:2]
    image_center = (image_width / 2, image_height / 2, 0)
    radius = 50  # Adjusted to match rho_max
    height = 100
    vanishing_pts = (0, 0, 10)
    center = image_center
    # rho_step_size = 5
    # alpha_step_size = 2*np.pi / 10

    control_points=np.loadtxt('./data/control_points.txt')

    points_2d_observed=detect_feature_points(image)

    z_vector = np.array([0, 0, 10])
    z_unit_vector = z_vector / np.linalg.norm(z_vector)
    x_camera_vector = np.array([1, 0, 0])
    y_vector = np.cross(z_unit_vector, x_camera_vector)
    x_vector = (np.cross(z_unit_vector, y_vector))
    x_vector /= np.linalg.norm(x_vector)
    y_vector /= np.linalg.norm(y_vector)
    rot_mat = np.vstack([x_vector, y_vector, z_unit_vector]).T
   
    trans_mat = np.array([0, 0, 10])

    intrinsic_matrix, rotation_matrix, translation_vector = Project3D_2D_cam.get_camera_parameters(image_height, image_width, rot_mat, trans_mat,center)
    k = 2.5
    g_t = 2.0
    gamma = 2.2
    init_lambda_ortho = 1
    init_lambda_det = 1
    M=5
    N=5
 
    cylinder_points = generate_cylinder_points(radius, height, M,N)
    points_3d=BMeshDeformation(height,center,cylinder_points)
    

    initial_params = np.hstack([rotation_matrix.flatten(), translation_vector.flatten(),control_points.ravel(),init_lambda_ortho, init_lambda_det])
    optimized_params = optimize_params(points_3d, points_2d_observed, image, intrinsic_matrix, initial_params, k, g_t, gamma,frame_idx=0)

    log_optim_params(optimized_params, 0)

    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time} seconds")

# pr=cProfile.Profile()
if __name__ == "__main__":
    # pr.enable()
    main()
    # pr.disable()
    # s=io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')

    # ps.print_stats()
    # with open('profiling_results.txt', 'w') as f:
    #     f.write(s.getvalue())

