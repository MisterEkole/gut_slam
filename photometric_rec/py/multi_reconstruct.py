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
import torch
from torch.multiprocessing import set_start_method

# Set multiprocessing start method to 'spawn' for CUDA compatibility
try:
    set_start_method('spawn')
except RuntimeError:
    pass

def generate_synthetic_image(size=(512, 512), intensity_range=(0, 255)):
    synthetic_image = np.random.randint(intensity_range[0], intensity_range[1] + 1, size, dtype=np.uint8)
    return synthetic_image

def compute_img_depths_parallel(args):
    img_path, output_dir, pcl_output = args
    img = cv2.imread(img_path)
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    iters = 50
    downsample_factor = 10
    display_interval = 100

    img = cv2.resize(img, None, fx=1 / downsample_factor, fy=1 / downsample_factor, interpolation=cv2.INTER_AREA)
    k, g_t, gamma = get_calibration(img)

    energy_function = np.zeros((img.shape[0], img.shape[1]))
    gradient = np.ones((img.shape[0], img.shape[1]))
    depth_map = 1 / np.sqrt(np.cos(0))  # depth map init with canonical intensity
    depth_map = np.full((img.shape[0], img.shape[1]), np.cos(0))
    errors = []

    regularization_lambda = 1.0
    alpha = 0.001
    prev_energy_function = np.zeros((img.shape[0], img.shape[1]))

    for i in tqdm(range(iters)):
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                d = depth_map[row, col]
                u = img[row, col]
                x, y, z = unprojec_cam_model(u, d)
                L = calib_p_model(x, y, d, k, g_t, gamma)
                I = get_intensity(u)
                C = cost_func(I, L)
                R = reg_func(gradient[row, col])
                energy_function[row, col] = C + regularization_lambda * R

                # Perform gradient descent for every pixel with variable step size
                if i > 0:
                    gradient[row, col] = energy_function[row, col] - prev_energy_function[row, col]

                    # Introduce variable step size based on the gradient magnitude
                    step_size = alpha / np.sqrt(np.sum(gradient[row, col] ** 2) + 1e-8)
                    depth_map[row, col] -= step_size * gradient[row, col]

        prev_energy_function = energy_function.copy()

        error = (np.sum(energy_function) / (img.shape[0] * img.shape[1])) * 100  # expressed in percentage
        errors.append(error)

        with open(os.path.join(output_dir, f"errors_{img_name}.txt"), "w") as f:
            for e in errors:
                f.write(str(e) + "\n")

        if (i + 1) % display_interval == 0 or i == iters - 1:
            display_point_cloud_and_depth_map(depth_map, i, img_name, output_dir, pcl_output)

    save_depth_map_heatmap(depth_map, img_name, output_dir)

    return depth_map


def display_point_cloud_and_depth_map(depth_map, iteration, img_name, output_dir, pcl_output):
    display_depth_map(depth_map, iteration, img_name, output_dir)
    display_point_cloud(depth_map, img_name, output_dir, pcl_output)


def display_depth_map(depth_map, iteration, img_name, output_dir):
    depth_map_img = cv2.normalize(src=depth_map, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                  dtype=cv2.CV_8UC1)
    output_path = os.path.join(output_dir, f'depthmap_{img_name}_{iteration}.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, depth_map_img)

    # Optionally: display the depth map or heatmap here if needed.


def save_depth_map_heatmap(depth_map, img_name, output_dir):
    # Save the heatmap image
    plt.figure(figsize=(10, 8))
    sns.heatmap(depth_map, cmap='viridis', annot=False)
    plt.title(f'Depth Map Heatmap - {img_name}')
    output_path = os.path.join(output_dir, f'depthmap_heatmap_{img_name}.png')
    plt.savefig(output_path)  # Save the heatmap image
    plt.close()


def display_point_cloud(depth_map, img_name, output_dir, pcl_output):
    point_cloud = generate_point_cloud(depth_map)
    output_path = os.path.join(pcl_output, f'point_cloud_{img_name}.txt')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savetxt(output_path, point_cloud)
    print(f"Point cloud saved to {output_path}")


def generate_point_cloud(depth_map):
    point_cloud = []
    for row in range(depth_map.shape[0]):
        for col in range(depth_map.shape[1]):
            d = depth_map[row, col]
            K = get_intrinsic_matrix()
            if d > 0:
                u, v = [row, col]
                x = (u - K[0, 2]) * d / K[0, 0]  # adjust to camera intrinsic
                y = (v - K[1, 2]) * d / K[1, 1]
                z = d
                point_cloud.append([x, y, z])
    return np.array(point_cloud)


def parseargs():
    parser = argparse.ArgumentParser(description="Photometric Dense 3D Reconstruction--Gut SLAM")
    parser.add_argument('--image_dir', type=str, help='Path to the directory containing image files')
    parser.add_argument('--output_dir', type=str, default='./image_output', help='Depth map output directory')
    parser.add_argument('--pcl_output', type=str, default='./pcl_output', help='Point cloud output directory')

    return parser.parse_args()


if __name__ == '__main__':
    args = parseargs()
    image_dir = args.image_dir
    output_dir = args.output_dir
    pcl_output = args.pcl_output

    img_paths = [os.path.join(image_dir, img_file) for img_file in os.listdir(image_dir) if img_file.endswith('.jpg')]

    # Use multiprocessing to parallelize reconstruction
    with Pool(cpu_count()) as pool:
        depth_maps = pool.map(compute_img_depths_parallel, [(img_path, output_dir, pcl_output) for img_path in img_paths])

    # Optionally, you can plot the errors over iterations
    plt.figure(figsize=(10, 6))
    for i, depth_map in enumerate(depth_maps):
        img_name = os.path.splitext(os.path.basename(img_paths[i]))[0]
        errors = np.loadtxt(os.path.join(output_dir, f"errors_{img_name}.txt"))
        plt.plot(errors, label=f"{img_name}")

    plt.xlabel('Iteration')
    plt.ylabel('Reconstruction Error (%)')
    plt.title('Reconstruction Error Over Iterations')
    plt.legend()
    plt.show()
