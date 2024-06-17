'''
-------------------------------------------------------------
Photometric  3D Reconstruction from Single Image
Author: Mitterrand Ekole
Date: 19-02-2024
-------------------------------------------------------------
'''

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

def generate_synthetic_image(size=(512, 512), intensity_range=(0, 255)):
    synthetic_image = np.random.randint(intensity_range[0], intensity_range[1] + 1, size, dtype=np.uint8)
    return synthetic_image

def get_synthetic_image_data():
    synthetic_image = generate_synthetic_image()
    return synthetic_image

''' compute depth map using gradient descent with variable step size'''
def compute_img_depths(img, iters=50, downsample_factor=10, display_interval=100):
    img = cv2.resize(img, None, fx=1/downsample_factor, fy=1/downsample_factor, interpolation=cv2.INTER_AREA)
    k, g_t, gamma = get_calibration(img)
    
    energy_function = np.zeros((img.shape[0], img.shape[1]))
    gradient = np.ones((img.shape[0], img.shape[1]))
    #depth_map = np.ones((img.shape[0], img.shape[1]))
    depth_map=1/np.sqrt(np.cos(0)) #depth map init with canonical intensity
    depth_map=np.full((img.shape[0],img.shape[1]),np.cos(0))
    errors = []
    #per_pixel_errors=[]

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

        #error = np.sum(energy_function)
    
        error=(np.sum(energy_function) / (img.shape[0] * img.shape[1])) #expressed in percentage
        #per_pixel_error=np.mean(np.abs(energy_function))
        print(f"Iteration {i+1}, Error: {error}")
        errors.append(error)
        # per_pixel_errors.append(per_pixel_error)

        with open("./errors.txt", "w") as f:
            for e in errors:
                f.write(str(e) + "\n")
            # for p in per_pixel_errors:
            #     f.write(str(p) + "\n")

        if (i + 1) % display_interval == 0 or i == iters - 1:
            display_depth_map(depth_map, i, args.output_dir)
        display_point_cloud(depth_map,k,g_t,gamma, pcl_output)

    return depth_map


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


def parseargs():
    parser=argparse.ArgumentParser(description="Photometric Dense 3D Reconstruction--Gut SLAM")
    parser.add_argument('--image_path', type=str, help='Path to the image file')
    parser.add_argument('--iters', type=int, default=50, help='Number of iterations')
    parser.add_argument('--downsample-factor', type=int, default=10, help='Downsample factor')
    parser.add_argument('--display-interval', type=int, default=100, help='Display interval')
    parser.add_argument('--output_dir', type=str, default='./image_output' ,help='Depth map output directory')
    parser.add_argument('--pcl_output', type=str, default='./pcl_output' ,help='Point cloud output directory')


    return parser.parse_args()

if __name__=='__main__':
    args=parseargs()
    img=cv2.imread(args.image_path)
    image_output=args.output_dir
    pcl_output=args.pcl_output
    k = 2.5
    g_t = 2.0
    gamma = 2.2

    for i in range(args.iters):
        depth_map=compute_img_depths(img, args.iters, args.downsample_factor, args.display_interval)
        #display_depth_map(depth_map, i, image_output)
        display_point_cloud(depth_map, pcl_output, k, g_t, gamma)
        break
    print("Reconstruction Complete!")
        
