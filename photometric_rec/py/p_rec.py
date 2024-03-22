''' Patch based reconstruction'''
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


''' compute depth map using gradient descent with variable step size'''
def compute_img_depths(img, iters=50, downsample_factor=10, display_interval=100):
    img = cv2.resize(img, None, fx=1/downsample_factor, fy=1/downsample_factor, interpolation=cv2.INTER_AREA)
    k, g_t, gamma = get_calibration(img)
    
    energy_function = np.zeros((img.shape[0], img.shape[1]))
    gradient = np.ones((img.shape[0], img.shape[1]))
    depth_map=1/np.sqrt(np.cos(0)) #depth map init with canonical intensity
    depth_map=np.full((img.shape[0],img.shape[1]),np.cos(0))
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

                        # Perform gradient descent for every pixel with variable step size
                        if i > 0:
                            patch_gradient[patch_row, patch_col] = patch_energy_function[patch_row, patch_col] - prev_energy_function[row+patch_row, col+patch_col]

                            # Introduce variable step size based on the gradient magnitude
                            step_size = alpha / np.sqrt(np.sum(patch_gradient[patch_row, patch_col] ** 2) + 1e-8)
                            patch_depth_map[patch_row, patch_col] -= step_size * patch_gradient[patch_row, patch_col]

                depth_map[row:row+patch_size, col:col+patch_size] = patch_depth_map
                energy_function[row:row+patch_size, col:col+patch_size] = patch_energy_function
                gradient[row:row+patch_size, col:col+patch_size] = patch_gradient

        prev_energy_function = energy_function.copy()

        error = np.sum(energy_function) / (img.shape[0] * img.shape[1])  # Expressed in percentage
        print(f"Iteration {i+1}, Error: {error}")
        errors.append(error)

        if (i + 1) % display_interval == 0 or i == iters - 1:
            display_depth_map(depth_map, i, args.output_dir)

    return depth_map, errors


def display_depth_map(depth_map, iteration, img_output):
    depth_map_img = cv2.normalize(src=depth_map, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    os.makedirs(img_output, exist_ok=True)
    output_path = f'{img_output}/depthmap_{iteration}.png'
    
    cv2.imwrite(output_path, depth_map_img)

    # Display depth map
    plt.imshow(depth_map_img, cmap='gray')
    plt.title(f'Depth Map - Iteration {iteration}')
    plt.colorbar()
    plt.show()

    # Display heatmap variation of the depth map estimate
    sns.heatmap(depth_map, cmap='viridis', annot=False)
    plt.title(f'Depth Map Estimation(cm) - Iteration {iteration}')
    plt.show()

    return output_path


def save_depth_map_heatmap(depth_map, iteration):
    # Save the heatmap image
    plt.figure(figsize=(10, 8))
    sns.heatmap(depth_map, cmap='viridis', annot=False)
    plt.title(f'Depth Map Heatmap - Iteration {iteration}')
    plt.savefig(f'depthmap_heatmap_{iteration}.png')  # Save the heatmap image
    plt.close()


def parseargs():
    parser = argparse.ArgumentParser(description="Photometric Dense 3D Reconstruction--Gut SLAM")
    parser.add_argument('--image_path', type=str, help='Path to the image file')
    parser.add_argument('--iters', type=int, default=50, help='Number of iterations')
    parser.add_argument('--downsample-factor', type=int, default=10, help='Downsample factor')
    parser.add_argument('--display-interval', type=int, default=100, help='Display interval')
    parser.add_argument('--output_dir', type=str, default='./image_output', help='Depth map output directory')

    return parser.parse_args()

if __name__=='__main__':
    args = parseargs()
    img = cv2.imread(args.image_path)

    depth_map, errors = compute_img_depths(img, args.iters, args.downsample_factor, args.display_interval)

    # Plotting error
    plt.plot(range(1, args.iters+1), errors)
    plt.title("Error Plot")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.show()

    print("Reconstruction Complete!")
