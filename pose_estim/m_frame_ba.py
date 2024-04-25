''' Multiframe bundle adjustment for pose and deformation estimation
Author: Mitterand Ekole
Date: 04-04-2024

'''
import cv2
import numpy as np
from scipy.optimize import least_squares
from utils import WarpField, Project3D_2D_cam, Points_Processor,calib_p_model, cost_func, get_pixel_intensity, reg_func, visualize_point_cloud
import matplotlib.pyplot as plt
import os
import time


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
def objective_function(params, points_3d, points_2d_observed, image, intrinsic_matrix, k, g_t, gamma, warp_field, lambda_ortho, lambda_det,control_points):
    if not isinstance(points_2d_observed, np.ndarray):
        points_2d_observed = np.array(points_2d_observed)

    
    rotation_matrix = params[:9].reshape(3, 3)
    translation_vector = params[9:12]
    a_params = params[12]
    b_params = params[13]
    
    warp_field.b_mesh_deformation(a=a_params, b=b_params, control_points=control_points)
    points_3d_deformed = warp_field.extract_pts()
    
    
    projector = Project3D_2D_cam(intrinsic_matrix, rotation_matrix, translation_vector)
    projected_2d_pts = projector.project_points(points_3d_deformed)
    if projected_2d_pts.shape[0] > points_2d_observed.shape[0]:
        projected_2d_pts = projected_2d_pts[:points_2d_observed.shape[0], :]
    elif projected_2d_pts.shape[0] < points_2d_observed.shape[0]:
        points_2d_observed = points_2d_observed[:projected_2d_pts.shape[0], :]
    
    points_2d_observed = points_2d_observed.reshape(-1, 2)
    
    reprojection_error = np.linalg.norm(projected_2d_pts - points_2d_observed, axis=1)
    photometric_error = []
    for pt2d, pt3d in zip(projected_2d_pts, points_3d_deformed):
        if np.isnan(pt2d).any():
            pt2d=np.where(np.isnan(pt2d),1,pt2d) #replace nan with 1 in arrays
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

    return objective

def optimize_params(points_3d, points_2d_observed, image, intrinsic_matrix, initial_params, k, g_t, gamma, warp_field, frame_idx,control_points):
    global optimization_errors
    optimization_errors = []
    lower_bounds = [-np.inf]*14 + [0, 0]  # Assuming non-negative values for the Lagrange multipliers
    upper_bounds = [np.inf]*14 + [np.inf, np.inf]

    result = least_squares(
        objective_function,
        initial_params,
        args=(points_3d, points_2d_observed, image, intrinsic_matrix, k, g_t, gamma, warp_field, 1, 1,control_points),  
        method='dogbox',  # Trust Region Reflective algorithm
        bounds=(lower_bounds, upper_bounds),  # Apply bounds
        max_nfev=1000, 
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

     
def process_frame(image, intrinsic_matrix, initial_params, points_3d, k, g_t, gamma, warp_field, frame_idx):
    points_2d_observed = Project3D_2D_cam(intrinsic_matrix, initial_params[:9].reshape(3, 3), initial_params[9:12]).project_points(points_3d)
    optimized_params = optimize_params(points_3d, points_2d_observed, image, intrinsic_matrix, initial_params, k, g_t, gamma, warp_field)
    return optimized_params
def detect_feature_points(image):
    orb = cv2.ORB_create()
    kp=orb.detect(image,None)
    kp=cv2.KeyPoint_convert(kp)
    return kp

def log_optim_params(optimized_params, frame_idx):
    with open('./optimized_params_all_frames.txt', 'a') as f:
        f.write(f"Frame {frame_idx}\n")
        f.write("Optimized Parameters:\n")
        f.write("Rotation Matrix: \n")
        f.write(str(optimized_params[:9].reshape(3, 3)) + "\n")
        f.write("Translation Vector: \n")
        f.write(str(optimized_params[9:12]) + "\n")
        f.write("Optimized a_val: ")
        f.write(str(optimized_params[12]) + "\n")
        f.write("Optimized b_val: ")
        f.write(str(optimized_params[13]) + "\n\n")
        f.write("Lambda Ortho: ")
        f.write(str(optimized_params[14]) + "\n")
        f.write("Lambda Det: ")
        f.write(str(optimized_params[15]) + "\n\n")



def main():
    frames_directory = '/Users/ekole/Synth_Col_Data/Frames_S1'
    print("Optimization started...")
    start_time=time.time()
    #frames_directory = '/Users/ekole/Dev/gut_slam/gut_images'
    frames = load_frames_from_directory(frames_directory)
    total_optim_time = 0
   
    
    # Assuming the first frame's parameters are applicable as the starting point for the next frames
    for frame_idx, image in enumerate(frames):
        image_height, image_width = image.shape[:2]
        image_center = (image_width / 2, image_height / 2, 0)
        radius = 500  
        height = 1000
        vanishing_pts = (0, 0, 10)
        center = image_center
        resolution = 100
        a_values = np.zeros((image_height, image_width, 3)) 
        b_values = np.zeros((image_height, image_width))  
        for row in range(image_height):
            for col in range(image_width):
                pixel = image[row, col]
                p_minus_vp = np.array([row, col, 0]) - np.array(vanishing_pts)
                a_values[row, col] = p_minus_vp
                b_values[row, col] = np.arctan2(p_minus_vp[1], p_minus_vp[0])

        a_values = a_values / np.linalg.norm(np.mean(a_values, axis=1), axis=1, keepdims=True)
        b_values = b_values / np.linalg.norm(b_values)
       
    

        points_2d_observed=detect_feature_points(image)
       
        # Initialize or update warp field for each frame
        warp_field = WarpField(radius, height, vanishing_pts, center, resolution)
        cylinder_points = warp_field.extract_pts()
        control_points=np.loadtxt('control_points.txt')
        control_points=control_points.reshape(30,30,3) 
        
        if frame_idx == 0:
            a_init=np.mean(a_values.ravel())
            b_init=np.mean(b_values.ravel())
            
            warp_field.b_mesh_deformation(a=a_init, b=b_init, control_points=control_points)
        else:
           
            warp_field.b_mesh_deformation(a=optimized_a, b=optimized_b, control_points=control_points)

        if frame_idx == 0:
            z_vector = np.array([0, 0, 10]) #vp from vanishing pooint
            z_unit_vector = z_vector / np.linalg.norm(z_vector) 

            # Define X vector of the camera
            x_camera_vector = np.array([1, 0, 0])

            # Calculate Y vector
            y_vector = np.cross(z_unit_vector, x_camera_vector)

            # Recalculate X vector based on new Y
            x_vector = np.cross(z_unit_vector, y_vector)

            # Normalize X and Y vectors to ensure they are unit vectors
            x_vector /= np.linalg.norm(x_vector)
            y_vector /= np.linalg.norm(y_vector)

            # Construct rotation matrix
            rot_mat = np.vstack([x_vector, y_vector, z_unit_vector]).T
            trans_mat=np.array([0, 0, 10])
        else:
            # Use the optimized parameters from the previous frame for initialization
            rot_mat = optimized_params[:9].reshape(3, 3)
            trans_mat = optimized_params[9:12]

        intrinsic_matrix, rotation_matrix, translation_vector = Project3D_2D_cam.get_camera_parameters(image_height, image_width, rot_mat, trans_mat)
        k = 2.5
        g_t = 2.0
        gamma = 2.2

        if frame_idx == 0:
            a_init=np.mean(a_values.ravel())
            b_init=np.mean(b_values.ravel())
            
            initial_lamda_ortho = 0
            initial_lamda_det = 0
        else:
            a_init = optimized_a
            b_init =optimized_b
            initial_lamda_ortho = optimized_lamda_ortho
            initial_lamda_det = optimized_lamda_det

        initial_params = np.hstack([rotation_matrix.flatten(), translation_vector.flatten(), a_init, b_init,initial_lamda_ortho, initial_lamda_det])
        optim_start_time = time.time()
        optimized_params = optimize_params(cylinder_points, points_2d_observed, image, intrinsic_matrix, initial_params, k, g_t, gamma, warp_field, frame_idx,control_points=control_points)

        optim_end_time = time.time()

        optim_time=optim_end_time-optim_start_time
        total_optim_time += optim_time

        optimized_a = optimized_params[12]
        optimized_b = optimized_params[13]
        optimized_lamda_ortho = optimized_params[14]
        optimized_lamda_det = optimized_params[15]

        log_optim_params(optimized_params, frame_idx)
        print("Optimizing Frame: ", frame_idx)
     

        # plt.imshow(image)
        # plt.xlim(0, image.shape[1])
        # plt.ylim(image.shape[0], 0)  # Inverted y-axis to match image coordinate system
        # plt.scatter(points_2d_observed[:, 0], points_2d_observed[:, 1], color='red', s=10)  # Increased size for visibility
        # plt.title(f"Frame {frame_idx}")
        # plt.show()

        # Optional: Visualize  the optimized results for each frame
        # print(f"Optimization time for frame {frame_idx}: {optim_time:.2f} seconds")

        # print(f"Frame {frame_idx}: Optimized Parameters:")
        # print("Rotation Matrix: \n",optimized_params[:9].reshape(3, 3))
        # #print("Optimized Rotation Vector: \n", rotation_matrix_to_vector(optimized_params[:9].reshape(3, 3)))
        # print("Optimized Translation Vector: \n", optimized_params[9:12])
        # print("Optimized Deformation Strength: ", optimized_deformation_strength)
        # print("Optimized Deformation Frequency: ", optimized_deformation_frequency)
    end_time=time.time()
    total_time=end_time-start_time
    print(f"Total execution time: {total_time:.2f} seconds")
    #visualize_point_cloud(cylinder_points)

if __name__ == "__main__":
    main()





