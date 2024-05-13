'''
-------------------------------------------------------------
Independent Multiframe Pose Optimization and Control Point
Adjustment via Bundle Adjustment

Author: Mitterand Ekole
Date: 04-04-2024
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


# Function to load frames from a video file
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


# Function to load frames from a directory of images
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
    """
    Convert a rotation matrix to a rotation vector using Rodrigues' formula.
    
    :param rotation_matrix: A 3x3 rotation matrix.
    :return: A rotation vector where the direction represents the axis of rotation
             and the magnitude represents the angle of rotation in radians.
    """
    rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
    return rotation_vector


optimization_errors=[]


''' Objective function with ortho and det constrains on Rot Mat using Lagrange Multipliers'''
def objective_function(params, points_3d, points_2d_observed, image, intrinsic_matrix, k, g_t, gamma, warp_field, lambda_ortho, lambda_det,pbar):
    # Unpacking parameters
    rotation_matrix = params[:9].reshape(3, 3)
    translation_vector = params[9:12]
    control_points=params[12:-2].reshape(10,10,3)
    lambda_ortho = params[-2]
    lambda_det = params[-1]
    a=0.00051301747 
    b=0.0018595674
   

    
    warp_field.b_mesh_deformation(a=a, b=b, control_points=control_points)
    points_3d_deformed = warp_field.extract_pts()
    
    # Project points
    projector = Project3D_2D_cam(intrinsic_matrix, rotation_matrix, translation_vector)
    projected_2d_pts = projector.project_points(points_3d_deformed)
    if projected_2d_pts.shape[0] > points_2d_observed.shape[0]:
        projected_2d_pts = projected_2d_pts[:points_2d_observed.shape[0], :]
    elif projected_2d_pts.shape[0] < points_2d_observed.shape[0]:
        points_2d_observed = points_2d_observed[:projected_2d_pts.shape[0], :]
    points_2d_observed = points_2d_observed.reshape(-1, 2)
    
    # Compute reprojection and photometric errors
    reprojection_error = np.linalg.norm(projected_2d_pts - points_2d_observed, axis=1)
    photometric_error = []
    for pt2d, pt3d in zip(projected_2d_pts, points_3d_deformed):
        x, y, z = pt3d
        L = calib_p_model(x, y, z, k, g_t, gamma)
        if 0 <= int(pt2d[0]) < image.shape[1] and 0 <= int(pt2d[1]) < image.shape[0]:
            pixel_intensity = get_pixel_intensity(image[int(pt2d[1]), int(pt2d[0])])
            C = cost_func(pixel_intensity, L)
        else:
            C = 0
        photometric_error.append(float(C))
    photometric_error = np.array(photometric_error, dtype=float)
    
 

     #Normalize each error type to the same scale
    reprojection_error /= (np.linalg.norm(reprojection_error) + 1e-8)
    photometric_error /= (np.linalg.norm(photometric_error) + 1e-8)

    global optimization_errors
    optimization_errors.append(
        {
            'reprojection_error': np.mean(reprojection_error),
            'photometric_error': np.mean(photometric_error),
        }
    )

    # Constraints with Lagrange Multipliers
    ortho_constraint = np.dot(rotation_matrix, rotation_matrix.T) - np.eye(3)
    det_constraint = np.linalg.det(rotation_matrix) - 1

    # Objective function with Lagrange multipliers
    objective = np.sum(reprojection_error**2) + np.sum(photometric_error**2)
    objective += lambda_ortho * np.linalg.norm(ortho_constraint, 'fro')**2  # Frobenius norm for matrix norm
    objective += lambda_det * det_constraint**2
    pbar.update(1)

    return objective


def optimize_params(points_3d, points_2d_observed, image, intrinsic_matrix, initial_params, k, g_t, gamma, warp_field, frame_idx,a,b):
    global optimization_errors
    optimization_errors = []
    # lower_bounds = [-np.inf]*14 + [0, 0]  # Assuming non-negative values for the Lagrange multipliers
    # upper_bounds = [np.inf]*14 + [np.inf, np.inf]
    num_params = len(initial_params)
    lower_bounds = [-np.inf] * num_params
    upper_bounds = [np.inf] * num_params
    lower_bounds[-2:] = [0, 0]  # Setting non-negative bounds for lambda parameters
    upper_bounds[-2:] = [np.inf, np.inf]
    
    with tqdm(total=frame_idx, desc=f"Optimizing frame {frame_idx}") as pbar:
        result = least_squares(
            objective_function,
            initial_params,
            args=(points_3d, points_2d_observed, image, intrinsic_matrix, k, g_t, gamma, warp_field, 1, 1,pbar),#1,1 lambda ortho, lambda det init
            method='dogbox',
            #bounds=(lower_bounds, upper_bounds),
            max_nfev=50,
            gtol=1e-8,
            tr_solver='lsmr'
        )


    log_errors(optimization_errors, frame_idx)
    return result.x




def log_errors(errors, frame_idx):
   
    with open('./optimization_errors_all_frames.txt', 'a') as f:
        f.write(f"Frame {frame_idx}\n")  # Indicate the start of a new frame
        for idx, error in enumerate(errors):
            f.write(f"Iteration {idx + 1}: Reprojection Error: {error['reprojection_error']:.4f}, Photometric Error: {error['photometric_error']:.4f}\n")
        mean_reprojection_error = np.mean([error['reprojection_error'] for error in errors])
        mean_photometric_error = np.mean([error['photometric_error'] for error in errors])
        f.write(f"Mean Reprojection Error: {mean_reprojection_error:.4f}\n")
        f.write(f"Mean Photometric Error: {mean_photometric_error:.4f}\n\n")  


def process_frame(image, intrinsic_matrix, initial_params, points_3d, k, g_t, gamma, warp_field, frame_idx,control_points):
    points_2d_observed = Project3D_2D_cam(intrinsic_matrix, initial_params[:9].reshape(3, 3), initial_params[9:12]).project_points(points_3d)
    optimized_params = optimize_params(points_3d, points_2d_observed, image, intrinsic_matrix, initial_params, k, g_t, gamma, warp_field,control_points)
    return optimized_params
def detect_feature_points(image):
    orb = cv2.ORB_create()
    kp=orb.detect(image,None)
    kp=cv2.KeyPoint_convert(kp)
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
    frames_directory = '/Users/ekole/Synth_Col_Data/Frames_S2'
    print("Optimization started...")
    start_time = time.time()

    frames = load_frames_from_directory(frames_directory)
    total_optim_time = 0

    # Constants for optimization--- light model parameters
    k = 2.5
    g_t = 2.0
    gamma = 2.2

   
    control_points = np.loadtxt('./data/control_points6.txt').reshape(10, 10, 3)

    
    init_lambda_ortho, init_lambda_det = 1, 1

    optimized_params = None

    for frame_idx, image in enumerate(frames):
        print("Processing Frame:", frame_idx)

        if image is None:
            print("Warning: Failed to read frame at index", frame_idx)
            continue

        image_height, image_width = image.shape[:2]

       
        points_2d_observed = detect_feature_points(image)

     
        warp_field = WarpField(radius=500, height=1000, vanishing_pts=(0, 0, 10), center=(image_width / 2, image_height / 2, 0), resolution=100)
        warp_field.b_mesh_deformation(a=0.43613728652325934, b=0.0018595670614189284, control_points=control_points)
        cylinder_points = warp_field.extract_pts()

       
        z_vector = np.array([0, 0, 10])
        z_unit_vector = z_vector / np.linalg.norm(z_vector)
        x_camera_vector = np.array([1, 0, 0])
        y_vector = np.cross(z_unit_vector, x_camera_vector)
        x_vector = np.cross(z_unit_vector, y_vector)
        x_vector /= np.linalg.norm(x_vector)
        y_vector /= np.linalg.norm(y_vector)
        rot_mat = np.vstack([x_vector, y_vector, z_unit_vector]).T
        trans_mat = np.array([0, 0, 10])

    
        intrinsic_matrix, rotation_matrix, translation_vector = Project3D_2D_cam.get_camera_parameters(image_height, image_width, rot_mat, trans_mat,center=(image_width / 2, image_height / 2, 0))

       
        initial_params = np.hstack([rotation_matrix.flatten(), translation_vector.flatten(), control_points.ravel(), init_lambda_ortho, init_lambda_det])

 
        optim_start_time = time.time()
        optimized_params = optimize_params(cylinder_points, points_2d_observed, image, intrinsic_matrix, initial_params, k, g_t, gamma, warp_field, frame_idx, a=0.43613728652325934, b=0.0018595670614189284)
        optim_end_time = time.time()

        optim_time = optim_end_time - optim_start_time
        total_optim_time += optim_time

  
        log_optim_params(optimized_params, frame_idx)
       

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()

