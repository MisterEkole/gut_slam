import argparse
from calib import get_calibration
from utils import get_intensity, unprojec_cam_model, get_intrinsic_matrix, get_canonical_intensity
from p_model import calib_p_model, cost_func, reg_func
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import os
from multiprocessing import Pool, cpu_count
import multiprocessing as mp

# Define a global variable to hold the arguments
global global_args


def compute_img_depths_single(args):
    img, iters, downsample_factor, output_dir = args

    img = cv2.resize(img, None, fx=1/downsample_factor, fy=1/downsample_factor, interpolation=cv2.INTER_AREA)
    k, g_t, gamma = get_calibration(img)
    
    energy_function = np.zeros((img.shape[0], img.shape[1]))
    gradient = np.ones((img.shape[0], img.shape[1]))
    depth_map=1/np.sqrt(np.cos(0)) #depth map init with canonical intensity
    depth_map = np.full((img.shape[0],img.shape[1]), np.cos(0))
    errors = []
    
    regularization_lambda = 1.0
    alpha = 0.001
    prev_energy_function = np.zeros((img.shape[0], img.shape[1]))

    patch_size = 5
    
    for i in tqdm(range(iters)):
        for row in range(0, img.shape[0], patch_size):
            for col in range(0, img.shape[1], patch_size):
                patch_img = img[row:row+patch_size, col:col+patch_size]
                patch_depth_map = depth_map[row:row+patch_size, col:col+patch_size]
                patch_energy_function = energy_function[row:row+patch_size, col:col+patch_size]
                patch_gradient = gradient[row:row+patch_size, col:col+patch_size]

                for patch_row in range(patch_img.shape[0]):
                    for patch_col in range(patch_img.shape[1]):
                        d = patch_depth_map[patch_row, patch_col]
                        u = patch_img[patch_row, patch_col]
                        x, y, z = unprojec_cam_model(u, d)
                        L = calib_p_model(x, y, d, k, g_t, gamma)
                        I = get_intensity(u)
                        C = cost_func(I, L)
                        R = reg_func(patch_gradient[patch_row, patch_col])
                        patch_energy_function[patch_row, patch_col] = C + regularization_lambda * R

                        # Perform gradient descent for every pixel patch with variable step size
                        if i > 0:
                            patch_gradient[patch_row, patch_col] = patch_energy_function[patch_row, patch_col] - prev_energy_function[row+patch_row, col+patch_col]

                            # Introduce variable step size based on the gradient magnitude
                            step_size = alpha / np.sqrt(np.sum(patch_gradient[patch_row, patch_col] ** 2) + 1e-8)
                            patch_depth_map[patch_row, patch_col] -= step_size * patch_gradient[patch_row, patch_col]

                depth_map[row:row+patch_size, col:col+patch_size] = patch_depth_map
                energy_function[row:row+patch_size, col:col+patch_size] = patch_energy_function
                gradient[row:row+patch_size, col:col+patch_size] = patch_gradient

        prev_energy_function = energy_function.copy()

        error = np.sum(energy_function) / (img.shape[0] * img.shape[1])  
        errors.append(error)

    return depth_map, errors

def compute_img_depths_parallel(args_global):
    img, iters, downsample_factor, args = args_global
    return compute_img_depths_single(img, iters, downsample_factor, args.output_dir)



def save_depth_map(depth_map, img_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Normalize depth map to the range [0, 1]
    depth_map_normalized = cv2.normalize(depth_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # Scale depth map to the range [0, 255] and convert to uint8
    depth_map_scaled = (depth_map_normalized * 255).astype(np.uint8)

    # Apply colormap for better visualization
    depth_map_colored = cv2.applyColorMap(depth_map_scaled, cv2.COLORMAP_JET)
    depth_map_path = os.path.join(output_dir, f'depthmap_{img_name}.png')
    cv2.imwrite(depth_map_path, depth_map_colored)

def save_point_cloud(point_cloud, img_name, output_dir):
    pcl_output = os.path.join(output_dir, f'pcl_{img_name}')
    os.makedirs(pcl_output, exist_ok=True)
    pcl_file_path = os.path.join(pcl_output, f'point_cloud_{img_name}.txt')
    np.savetxt(pcl_file_path, point_cloud)

def save_error_plot(errors, batch_start, output_dir):
    plt.figure()
    plt.plot(errors)
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Error Plot')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'error_plot_batch_{batch_start}.png'))
    plt.close()

def generate_point_cloud(depth_map, k, g_t, gamma):
    point_cloud = []
    for row in range(depth_map.shape[0]):
        for col in range(depth_map.shape[1]):
            d = depth_map[row, col]
            K=get_intrinsic_matrix()
            if d > 0:
                u, v = [row, col]
                x = (u - K[0, 2]) * d / K[0, 0]  # adjust to camera intrinsic
                y = (v - K[1, 2]) * d / K[1, 1]
                z = d
                point_cloud.append([x, y, z])
    return np.array(point_cloud)

def parseargs():
    parser = argparse.ArgumentParser(description="Photometric Dense 3D Reconstruction--Gut SLAM")
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset folder')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for processing')
    parser.add_argument('--iters', type=int, default=50, help='Number of iterations')
    parser.add_argument('--downsample-factor', type=int, default=10, help='Downsample factor')
    parser.add_argument('--output_dir', type=str, default='./image_output', help='Depth map output directory')
    return parser.parse_args()

if __name__=='__main__':
    args = parseargs()
    dataset_path = args.dataset_path
    batch_size = args.batch_size
    image_files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]
    num_processes = 4
    pool = Pool(num_processes)  # Create a multiprocessing pool
    for batch_start in range(0, len(image_files), batch_size):
        batch_images = []
        for i in range(batch_size):
            if batch_start + i < len(image_files):
                img_path = os.path.join(dataset_path, image_files[batch_start + i])
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check if the file is an image
                    #print("Loading image:", img_path)  # Print the image path for debugging
                    img = cv2.imread(img_path)
                    if img is None:
                        print("Failed to load image:", img_path)  # Print an error message if image loading fails
                        continue
                    #print("Image shape:", img.shape)  # Print the shape of the loaded image for debugging
                    batch_images.append((img, image_files[batch_start + i]))
                else:
                    print("Skipping non-image file:", img_path)

        if len(batch_images) > 0:
            print(f"Processing batch starting from image {batch_start + 1}...")
            batch_results = pool.map(compute_img_depths_single, [(img, args.iters, args.downsample_factor, args) for img, img_name in batch_images])

            for i, (depth_map, errors) in enumerate(batch_results):
                save_depth_map(depth_map, batch_images[i][1], args.output_dir)
                k, g_t, gamma = get_calibration(batch_images[i][0])
                point_cloud = generate_point_cloud(depth_map, k, g_t, gamma)
                save_point_cloud(point_cloud, batch_images[i][1], args.output_dir)
                save_error_plot(errors, batch_start, args.output_dir)
        

    pool.close()
    pool.join()

    print("Reconstruction Complete!")





