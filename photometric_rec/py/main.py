from calib import get_calibration
from utils import get_intensity, unprojec_cam_model, get_intrinsic_matrix
from p_model import calib_p_model, cost_func, reg_func

from tqdm import tqdm
import numpy as np
import cv2
import tracemalloc
from multiprocessing import current_process, cpu_count, Process, Pool
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import minimize
# Globals for parallel processing
# global k, g_t, gamma, regularization_lambda, alpha, img, depth_map, gradient, energy_function



''' compute depth map using gradient descent with variable step size'''
def compute_img_depths(img, iters=10, downsample_factor=5, display_interval=100):
    img = cv2.resize(img, None, fx=1/downsample_factor, fy=1/downsample_factor, interpolation=cv2.INTER_AREA)
    k, g_t, gamma = get_calibration(img)
    
    energy_function = np.zeros((img.shape[0], img.shape[1]))
    gradient = np.ones((img.shape[0], img.shape[1]))
    depth_map = np.ones((img.shape[0], img.shape[1]))
    errors = []

    regularization_lambda = 1e-2
    alpha = 0.05
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

        error = np.sum(energy_function)
        print(f"Iteration {i+1}, Error: {error}")
        errors.append(error)

        with open("./errors.txt", "w") as f:
            for e in errors:
                f.write(str(e) + "\n")

        if (i + 1) % display_interval == 0 or i == iters - 1:
            display_depth_map(depth_map, i)

    return depth_map


def display_depth_map(depth_map, iteration):
    depth_map_img = cv2.normalize(src=depth_map, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.imwrite(f'/Users/ekole/Dev/gut_slam/gut_images/depthmap_{iteration}.png', depth_map_img)

    # Display depth map
    plt.imshow(depth_map_img, cmap='gray')
    plt.title(f'Depth Map - Iteration {iteration}')
    plt.colorbar()
    plt.show()

    # Display heatmap variation of the depth map estimate
    sns.heatmap(depth_map, cmap='viridis', annot=False)
    plt.title(f'Depth Map Heatmap - Iteration {iteration}')
    plt.show()


def display_point_cloud(depth_map,k,g_t,gamma):
    point_cloud=generate_point_cloud(depth_map,k,g_t,gamma)
    np.savetxt("point_cloud.txt",point_cloud)
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
'''
Optimize depth map using L-BFGS-B, trust-constr, or other optimization algorithms
method: L-BFGS-B, trust-constr, etc.
''' 

''' Consumes more memory and is slower'''
def optimize_depth_map(img, iters=50, regularization_lambda=0.5, alpha=0.1, downsample_factor=10):
    img=cv2.resize(img,None,fx=1/downsample_factor,fy=1/downsample_factor,interpolation=cv2.INTER_AREA)
    k, g_t, gamma = get_calibration(img)
    depth_map = np.ones((img.shape[0], img.shape[1]))

    def objective(depth_map):
        return compute_energy_func(depth_map.reshape(img.shape[0], img.shape[1]), img, k, g_t, gamma, regularization_lambda)

    bounds = [(0, np.inf)] * (img.shape[0] * img.shape[1])  # Non-negative depth values

    # Initial guess for depth_map
    initial_depth_map = np.zeros((img.shape[0], img.shape[1]))

    #minimize with LFGBS-B
    erros=[]
    for i in tqdm(range(iters)):
      result=minimize(objective,initial_depth_map.flatten(),method='L-BFGS-B',bounds=bounds,options={'maxiter':iters})
      optimize_depth_map=result.x.reshape(img.shape[0],img.shape[1])
      error=compute_energy_func(optimize_depth_map,img,k,g_t,gamma,regularization_lambda)
      erros.append(error)
      
      print(f"iteation {i+1}/{iters} , Error:{error}")

      with open("./errors.txt","w") as f:
        for error in erros:
          f.write(str(error)+"\n")

    depth_map_img=cv2.normalize(src=optimize_depth_map,dst=None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC1)
    cv2.imwrite(f'/Users/ekole/Dev/gut_slam/gut_images/depthmap_{i}.png',depth_map_img)

    return optimize_depth_map

''' Parallel processing of iterations '''
# Function to optimize a single iteration
def optimize_iteration(args):
    i, initial_depth_map, img, k, g_t, gamma, regularization_lambda = args
    bounds = [(0, np.inf)] * (img.shape[0] * img.shape[1])  # Non-negative depth values
    def objective(depth_map):
        return compute_energy_func(depth_map.reshape(img.shape[0], img.shape[1]), img, k, g_t, gamma, regularization_lambda)

    result = minimize(objective, initial_depth_map.flatten(), method='L-BFGS-B', bounds=bounds, options={'maxiter': 1})
    optimized_depth_map = result.x.reshape(img.shape[0], img.shape[1])
    error = compute_energy_func(optimized_depth_map, img, k, g_t, gamma, regularization_lambda)

    print(f"Iteration {i+1}, Error: {error}")

    with open("./errors.txt", "w") as f:
        f.write(str(error) + "\n")
    depth_map_img = cv2.normalize(src=optimized_depth_map, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.imwrite(f'/Users/ekole/Dev/gut_slam/gut_images/depthmap_{i}.png', depth_map_img)

    return optimized_depth_map, error

def optimize_depth_map_parallel(img, iters=500, regularization_lambda=0.5, alpha=0.1, num_processes=4):
    img = 1/255 * img
    k, g_t, gamma = get_calibration(img)
    depth_map = np.zeros((img.shape[0], img.shape[1]))

    pool = Pool(processes=num_processes)

    # Initial guess for depth_map
    initial_depth_map = np.zeros((img.shape[0], img.shape[1]))

    # Create argument list for parallel processing
    args_list = [(i, initial_depth_map, img, k, g_t, gamma, regularization_lambda) for i in range(iters)]

    # Parallel processing of iterations
    results = pool.map(optimize_iteration, args_list)

    # Close the pool to free up resources
    pool.close()
    pool.join()

    optimized_depth_maps, errors = zip(*results)

    # Save results or perform additional analysis if needed

    return optimized_depth_maps[-1]

if __name__=='__main__':
  img = cv2.imread("/Users/ekole/Dev/gut_slam/gut_images/image2.jpeg")
  #print(f"CPU Count: {cpu_count()}")
  #print(optimize_depth_map(img))
  #print(optimize_depth_map_parallel(img))
  print(compute_img_depths(img))
  