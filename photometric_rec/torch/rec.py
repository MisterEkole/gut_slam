import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm

from calib import get_calibration
from p_model import calib_p_model, cost_func, reg_func
from utils import (
    get_intensity,
    unprojec_cam_model,
    get_intrinsic_matrix,
    get_canonical_intensity,
   
)


def generate_synthetic_image(size=(512, 512), intensity_range=(0, 255)):
    return np.random.randint(intensity_range[0], intensity_range[1] + 1, size, dtype=np.uint8)


def get_synthetic_image_data():
    return generate_synthetic_image()


def compute_img_depths(
    img, iters=50, downsample_factor=10, display_interval=100, device="cuda"
):
    img_resized = cv2.resize(
        img, None, fx=1 / downsample_factor, fy=1 / downsample_factor, interpolation=cv2.INTER_AREA
    )

    # Convert to PyTorch tensors and send to device
    img_tensor = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0).to(device)
    k, g_t, gamma = get_calibration(img_tensor)

    depth_map = torch.full((img_tensor.shape[2], img_tensor.shape[3]), np.cos(0)).to(device)
    errors = []
    regularization_lambda = 1.0
    alpha = 0.001
    prev_energy_function = torch.zeros_like(depth_map).to(device)

    for i in tqdm(range(iters)):
        energy_function = torch.zeros_like(depth_map).to(device)

        for row in range(img_tensor.shape[2]):
            for col in range(img_tensor.shape[3]):
                d = depth_map[row, col]
                u = img_tensor[0, 0, row, col]  

              
                x, y, z = unprojec_cam_model(u, d)
                L = calib_p_model(x, y, d, k, g_t, gamma)
                I = get_intensity(u)
                C = cost_func(I, L)
                R = reg_func(depth_map[row, col]) 
                energy_function[row, col] = C + regularization_lambda * R

        gradient = energy_function - prev_energy_function
        step_size = alpha / torch.sqrt(torch.sum(gradient ** 2) + 1e-3)
        depth_map -= step_size * gradient
        prev_energy_function = energy_function.clone()

        error = (torch.sum(energy_function) / (img_tensor.shape[2] * img_tensor.shape[3])) * 100
        print(f"Iteration {i+1}, Error: {error.item():.2f}%") 
        errors.append(error.item())

        with open("./errors.txt", "w") as f:
            for e in errors:
                f.write(str(e) + "\n")

        if (i + 1) % display_interval == 0 or i == iters - 1:
            display_depth_map(depth_map.cpu().numpy(), i, args.output_dir)
            display_point_cloud(depth_map.cpu().numpy(), k, g_t, gamma, args.pcl_output)

    return depth_map.cpu().numpy()

# def compute_img_depths_vectorized(
#     img, iters=50, downsample_factor=10, display_interval=100, device="cuda"
# ):
#     img_resized = cv2.resize(
#         img, None, fx=1 / downsample_factor, fy=1 / downsample_factor, interpolation=cv2.INTER_AREA
#     )

#     img_tensor = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0).to(device)
#     k, g_t, gamma = get_calibration(img_tensor)

#     depth_map = torch.full((img_tensor.shape[2], img_tensor.shape[3]), np.cos(0)).to(device)
#     errors = []
#     regularization_lambda = 1.0
#     alpha = 0.001
#     prev_energy_function = torch.zeros_like(depth_map).to(device)

#     for i in tqdm(range(iters)):
#         energy_function = torch.zeros_like(depth_map).to(device)

#         u_values = img_tensor[0, 0].view(-1)  # Flatten u values
#         d_values = depth_map.view(-1)  # Flatten depth_map values

#         x, y, z = unprojec_cam_model(u_values, d_values)
#         L_values = calib_p_model(x, y, d_values, k, g_t, gamma)
#         I_values = get_intensity(u_values)
#         C_values = cost_func(I_values, L_values)
#         R_values = reg_func(depth_map.view(-1))  # Assuming reg_func is vectorized

#         energy_function.view(-1)[:] = C_values + regularization_lambda * R_values

#         gradient = energy_function - prev_energy_function
#         step_size = alpha / torch.sqrt(torch.sum(gradient ** 2) + 1e-3)
#         depth_map -= step_size * gradient
#         prev_energy_function = energy_function.clone()

#         error = (torch.sum(energy_function) / (img_tensor.shape[2] * img_tensor.shape[3])) * 100
#         print(f"Iteration {i+1}, Error: {error.item():.2f}%")
#         errors.append(error.item())

#         with open("./errors.txt", "w") as f:
#             for e in errors:
#                 f.write(str(e) + "\n")

#         if (i + 1) % display_interval == 0 or i == iters - 1:
#             display_depth_map(depth_map.cpu().numpy(), i, args.output_dir)
#             display_point_cloud(depth_map.cpu().numpy(), k, g_t, gamma, args.pcl_output)

#     return depth_map.cpu().numpy()



def display_depth_map(depth_map, iteration, img_output):
    depth_map_img = cv2.normalize(src=depth_map, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    os.makedirs(img_output, exist_ok=True)
    output_path=f'{img_output}/depthmap_{iteration}.png'
    
    cv2.imwrite(output_path, depth_map_img)
    #cv2.imwrite(f'/Users/ekole/Dev/gut_slam/gut_images/depthmap_{iteration}.png', depth_map_img)



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

def display_point_cloud(depth_map,k,g_t,gamma, pcl_output):
    point_cloud=generate_point_cloud(depth_map,k,g_t,gamma)
    output_path=f'{pcl_output}/point_cloud.txt'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savetxt(output_path,point_cloud)
    print("Point cloud saved to point_cloud.txt")
   
def generate_point_cloud(depth_map, k, g_t, gamma):
    point_cloud = []
    for row in range(depth_map.shape[0]):
        for col in range(depth_map.shape[1]):
            d = depth_map[row, col]
            K=get_intrinsic_matrix()
            if d > 0:
                u, v = [row, col]
                x = (u - K[0, 2]) * d / K[0, 0] #adjust to camera intrinsic
                y = (v - K[1, 2]) * d / K[1, 1]
                z = d
                point_cloud.append([x, y, z])
    return np.array(point_cloud)



def compute_energy_func(depth_map, img, k, g_t, gamma, regularization_lambda):
    energy_func = np.zeros((img.shape[0], img.shape[1]))

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            d = depth_map[row, col]
            u = img[row, col]
            x, y, z = unprojec_cam_model(u, d)
            L = calib_p_model(x, y, d, k, g_t, gamma)
            I = get_intensity(u)
            C = cost_func(I, L)
            R = reg_func(depth_map[row, col])
            energy_func[row, col] = C + regularization_lambda * R

    return np.sum(energy_func)

def parse_args():
    parser = argparse.ArgumentParser(description="Photometric Dense 3D Reconstruction--Gut SLAM")
    parser.add_argument("--image_path", type=str, help="Path to the image file")
    parser.add_argument("--iters", type=int, default=50, help="Number of iterations")
    parser.add_argument(
        "--downsample-factor", type=int, default=10, help="Downsample factor"
    )
    parser.add_argument(
        "--display-interval", type=int, default=100, help="Display interval"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./image_output", help="Depth map output directory"
    )
    parser.add_argument(
        "--pcl_output", type=str, default="./pcl_output", help="Point cloud output directory"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for computation (e.g., 'cuda', 'cpu')"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args() 
    img = cv2.imread(args.image_path)

    depth_map = compute_img_depths(img, args.iters, args.downsample_factor, args.display_interval, args.device)

    print("Reconstruction Complete!")
