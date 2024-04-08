''' Multiframe bundle adjustment for pose and deformation estimation
Author: Mitterand Ekole
Date: 04-04-2024

'''
import cv2
import numpy as np
from scipy.optimize import least_squares
from utils import WarpField, Project3D_2D_cam, Points_Processor,calib_p_model, cost_func, get_pixel_intensity, reg_func, euler_to_rot_mat
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
            break  # Break the loop when there are no frames left to read
        frames.append(frame)

    cap.release()  # Release the video capture object
    return frames

# Function to load frames from a directory of images
def load_frames_from_directory(directory_path):
    frames = []
    image_files = sorted([f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))])

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

def objective_function(params, points_3d, points_2d_observed, image, intrinsic_matrix, k, g_t, gamma, warp_field):
    if not isinstance(points_2d_observed, np.ndarray):
        points_2d_observed = np.array(points_2d_observed)

    rotation_matrix = params[:9].reshape(3, 3)
    translation_vector = params[9:12]
    deformation_strength = params[12]
    deformation_frequency = params[13]
    
    # Update deformation parameters
    warp_field.apply_deformation_axis(strength=deformation_strength, frequency=deformation_frequency)
    points_3d_deformed = warp_field.extract_pts()

    projector = Project3D_2D_cam(intrinsic_matrix, rotation_matrix, translation_vector)
    projected_2d_pts = projector.project_points(points_3d_deformed)
    #print("Before error line:", type(points_2d_observed), points_2d_observed.shape)

    # Resize projected_2d_pts to match points_2d_observed
    if projected_2d_pts.shape[0] > points_2d_observed.shape[0]:
        projected_2d_pts = projected_2d_pts[:points_2d_observed.shape[0], :]
    elif projected_2d_pts.shape[0] < points_2d_observed.shape[0]:
        points_2d_observed = points_2d_observed[:projected_2d_pts.shape[0], :]
    
        
    reprojection_error = (np.linalg.norm(projected_2d_pts - points_2d_observed, axis=1))/100
    lamda_reg = 1.0
    
    photometric_error = []
    for pt2d, pt3d in zip(projected_2d_pts, points_3d_deformed):
        x, y, z = pt3d
        L = calib_p_model(x, y, z, k, g_t, gamma)
        if 0 <= int(pt2d[0]) < image.shape[1] and 0 <= int(pt2d[1]) < image.shape[0]:  #check if pts2d is within boundary of image, then proceed to compute pixel intensity
            pixel_intensity = get_pixel_intensity(image[int(pt2d[1]), int(pt2d[0])])
            C = cost_func(pixel_intensity, L)
        else:
            C = 0 
        grad=np.ones((projected_2d_pts.shape[0],))
        reg=reg_func(grad)
        
        photometric_error.append(C)
    photometric_error = np.array(photometric_error+lamda_reg*reg)


    ''' adding rotation matrix constraints'''
    #Orthogonality constraint
    ortho_pen_scale=10
    ortho_penalty = ortho_pen_scale * np.linalg.norm(np.dot(rotation_matrix, rotation_matrix.T) - np.eye(3))
    ortho_penalty*=ortho_pen_scale

    #derterminant constraint
    det_pen_scale=10
    det_penalty = det_pen_scale * ((np.linalg.det(rotation_matrix) - 1)**2)
    det_penalty*=det_pen_scale

    errors = np.concatenate([reprojection_error, photometric_error.flatten()])
    errors=np.append(errors, [ortho_penalty, det_penalty])
    
    def normalize_errors(errors, target_scale=100):
        mean_error = np.mean(errors)
        scale_factor = target_scale / mean_error if mean_error else 1
        normalized_errors = errors * scale_factor
        return normalized_errors #outputs a normalised error array
    

    normalize_errors=normalize_errors(errors, target_scale=100)
    # print(" The photometric error : ", np.mean(photometric_error)) 
    # print("Reprojection error: ", np.mean(reprojection_error))
    return normalize_errors #outputs the mean error of the normalised error array

def optimize_params(points_3d, points_2d_observed, image, intrinsic_matrix, initial_params, k, g_t, gamma, warp_field):
    # result = least_squares(objective_function, 
    #                        initial_params,
    #                      args=(points_3d, points_2d_observed, image, intrinsic_matrix, k, g_t, gamma, warp_field), 
    #                      method='lm', max_nfev=2000, gtol=1e-6)
    result = least_squares(objective_function, 
                       initial_params,
                       args=(points_3d, points_2d_observed, image, intrinsic_matrix, k, g_t, gamma, warp_field), 
                       method='trf',  # Trust Region Reflective algorithm supports bounds
                       bounds=([-np.inf]*9 + [-10, -10, -10] + [0, 0],  # Lower bounds
                               [np.inf]*9 + [10, 10, 10] + [np.inf, np.inf]),  # Upper bounds
                       max_nfev=1000, 
                       gtol=1e-8)
    return result.x


def process_frame(image, intrinsic_matrix, initial_params, points_3d, k, g_t, gamma, warp_field):
    points_2d_observed = Project3D_2D_cam(intrinsic_matrix, initial_params[:9].reshape(3, 3), initial_params[9:12]).project_points(points_3d)
    optimized_params = optimize_params(points_3d, points_2d_observed, image, intrinsic_matrix, initial_params, k, g_t, gamma, warp_field)
    return optimized_params
def detect_feature_points(image):
    orb = cv2.ORB_create()
    kp=orb.detect(image,None)
    kp=cv2.KeyPoint_convert(kp)
    return kp

def main():
    frames_directory = '/Users/ekole/Synth_Col_Data/Frames_S1'
    start_time=time.time()
    #cframes_directory = '/Users/ekole/Dev/gut_slam/gut_images'
    frames = load_frames_from_directory(frames_directory)
    total_optim_time = 0
    yaw=np.radians(0)
    pitch=np.radians(0)
    roll=np.radians(0)
    
    # Assuming the first frame's parameters are applicable as the starting point for the next frames
    for frame_idx, image in enumerate(frames):
        image_height, image_width = image.shape[:2]
        image_center = (image_width / 2, image_height / 2, 0)
        radius = 500  
        height = 1000
        vanishing_pts = (0, 0, 10)
        center = image_center
        resolution = 100

        points_2d_observed=detect_feature_points(image)
      

        

        # Initialize or update warp field for each frame
        warp_field = WarpField(radius, height, vanishing_pts, center, resolution)
        cylinder_points = warp_field.extract_pts()
        if frame_idx == 0:
            warp_field.apply_deformation_axis(strength=5, frequency=10)
        else:
            # Adjust strength and frequency based on optimized parameters from the previous frame
            warp_field.apply_deformation_axis(strength=optimized_deformation_strength, frequency=optimized_deformation_frequency)

        if frame_idx == 0:
        
           
           
            # rot_mat = np.array([[0.5, 0.8, 3.5],   
            #                  [2, 1, 2],
            #                 [1.5, 0, 10]]) #vp
            
            rot_mat=euler_to_rot_mat(yaw,pitch,roll)
      
            trans_mat=np.array([0, 0, 10])
        else:
            # Use the optimized parameters from the previous frame for initialization
            rot_mat = optimized_params[:9].reshape(3, 3)
            trans_mat = optimized_params[9:12]

        intrinsic_matrix, rotation_matrix, translation_vector = Project3D_2D_cam.get_camera_parameters(image_height, image_width, rot_mat, trans_mat)
        
        
        #points_2d_observed = Project3D_2D_cam(intrinsic_matrix, rotation_matrix, translation_vector).project_points(cylinder_points)

        k = 2.5
        g_t = 2.0
        gamma = 2.2

        if frame_idx == 0:
            initial_deformation_strength = 0.1
            initial_deformation_frequency = 1
        else:
            initial_deformation_strength = optimized_deformation_strength
            initial_deformation_frequency = optimized_deformation_frequency

        initial_params = np.hstack([rotation_matrix.flatten(), translation_vector.flatten(), initial_deformation_strength, initial_deformation_frequency])
        optim_start_time = time.time()
        optimized_params = optimize_params(cylinder_points, points_2d_observed, image, intrinsic_matrix, initial_params, k, g_t, gamma, warp_field)

        optim_end_time = time.time()

        optim_time=optim_end_time-optim_start_time
        total_optim_time += optim_time

        optimized_deformation_strength = optimized_params[12]
        optimized_deformation_frequency = optimized_params[13]

        # plt.imshow(image)
        # plt.xlim(0, image.shape[1])
        # plt.ylim(image.shape[0], 0)  # Inverted y-axis to match image coordinate system
        # plt.scatter(points_2d_observed[:, 0], points_2d_observed[:, 1], color='red', s=10)  # Increased size for visibility
        # plt.title(f"Frame {frame_idx}")
        # plt.show()

        # Optional: Visualize  the optimized results for each frame
        print(f"Optimization time for frame {frame_idx}: {optim_time:.2f} seconds")

        print(f"Frame {frame_idx}: Optimized Parameters:")
        print("Rotation Matrix: \n",optimized_params[:9].reshape(3, 3))
        #print("Optimized Rotation Vector: \n", rotation_matrix_to_vector(optimized_params[:9].reshape(3, 3)))
        print("Optimized Translation Vector: \n", optimized_params[9:12])
        print("Optimized Deformation Strength: ", optimized_deformation_strength)
        print("Optimized Deformation Frequency: ", optimized_deformation_frequency)
    end_time=time.time()
    total_time=end_time-start_time
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()