import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import networkx as nx
from networkx.algorithms.flow import min_cost_flow
from calib import get_calibration
from utils import unprojec_cam_model, get_intrinsic_matrix, get_intensity
from p_model import calib_p_model, cost_func, reg_func
from tqdm import tqdm

def compute_img_depths(img, iters=100, downsample_factor=2, display_interval=100):
    img = cv2.resize(img, None, fx=1/downsample_factor, fy=1/downsample_factor, interpolation=cv2.INTER_AREA)
    k, g_t, gamma = get_calibration(img)
    depth_map=1/np.sqrt(np.cos(0)) #depth map init with canonical intensity
    depth_map=np.full((img.shape[0],img.shape[1]),np.cos(0))
    
    #depth_map = np.zeros_like(img, dtype=np.float32)  # Initialize depth map

    for i in tqdm(range(iters)):
        if (i + 1) % display_interval == 0 or i == iters - 1:
            display_depth_map(depth_map, i, args.output_dir)
            save_depth_map_heatmap(depth_map, i)
            rms_error = compute_rms_error(img, depth_map)
            photometric_error = compute_photometric_error(img, depth_map)
            print(f"Iteration {i+1}, RMS Error: {rms_error}, Photometric Error: {photometric_error}")

    display_point_cloud(depth_map, k, g_t, gamma, args.pcl_output)

    return depth_map


def display_depth_map(depth_map, iteration, img_output):
    depth_map_img = cv2.normalize(src=depth_map, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    os.makedirs(img_output, exist_ok=True)
    output_path = f'{img_output}/depthmap_{iteration}.png'
    cv2.imwrite(output_path, depth_map_img)

    plt.imshow(depth_map_img, cmap='gray')
    plt.title(f'Depth Map - Iteration {iteration}')
    plt.colorbar()
    plt.show()

    depth_map=np.squeeze(depth_map)

    sns.heatmap(depth_map, cmap='viridis', annot=False) 
    plt.title(f'Depth Map Estimation (cm) - Iteration {iteration}')
    plt.show()

    return output_path

def save_depth_map_heatmap(depth_map, iteration):
    plt.figure(figsize=(10, 8))
    sns.heatmap(depth_map, cmap='viridis', annot=False)
    plt.title(f'Depth Map Heatmap - Iteration {iteration}')
    plt.savefig(f'depthmap_heatmap_{iteration}.png')
    plt.close()

def compute_rms_error(img, depth_map):
    I = img.flatten()
    D = depth_map.flatten()
    rms_error = np.sqrt(np.mean((I - D) ** 2))
    return rms_error

def compute_photometric_error(img, depth_map):
    k, g_t, gamma = get_calibration(img)
    photometric_error = 0

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            d = depth_map[row, col]
            u = np.array([row, col])
            x, y, z = unprojec_cam_model(u, d)
            L = calib_p_model(x, y, d, k, g_t, gamma)
            I = get_intensity(u)
            C = cost_func(I, L)
            photometric_error += C

    photometric_error /= (img.shape[0] * img.shape[1])
    return photometric_error

def display_point_cloud(depth_map, k, g_t, gamma, pcl_output):
    point_cloud = generate_point_cloud(depth_map, k, g_t, gamma)
    output_path = f'{pcl_output}/point_cloud.txt'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savetxt(output_path, point_cloud)
    print("Point cloud saved to point_cloud.txt")

def generate_point_cloud(depth_map, k, g_t, gamma):
    point_cloud = []
    for row in range(depth_map.shape[0]):
        for col in range(depth_map.shape[1]):
            d = depth_map[row, col]
            K = get_intrinsic_matrix()
            if d > 0:
                u, v = [row, col]
                x = (u - K[0, 2]) * d / K[0, 0]
                y = (v - K[1, 2]) * d / K[1, 1]
                z = d
                point_cloud.append([x, y, z])
    return np.array(point_cloud)

def construct_graph(img):
    rows, cols = img.shape[:2]
    G = nx.grid_2d_graph(rows, cols)
    return G

def compute_edge_weight(img, u, v):
    weight = np.abs(img[u] - img[v])  
    return np.mean(weight)

def add_edge_weights(G, img):
    for u, v in G.edges:
        weight = compute_edge_weight(img, u, v)
        G[u][v]['weight'] = float(weight)

def optimize_depth_map_graph(img):
    G = construct_graph(img)
    add_edge_weights(G, img)

    mcg = nx.DiGraph()
    mcg.add_nodes_from(['s', 't'])
    for (u, v) in G.edges:
        mcg.add_edge(u, v, capacity=1, weight=G[u][v]['weight'])
        mcg.add_edge(v, u, capacity=1, weight=G[u][v]['weight'])
    for node in G.nodes:
        mcg.add_edge('s', node, capacity=1, weight=0)
        mcg.add_edge(node, 't', capacity=1, weight=0)

    flow_dict = min_cost_flow(mcg, 's', 't')
    depth_map = np.zeros_like(img, dtype=np.float32)

    for u, neighbors in flow_dict.items():
        for v, flow in neighbors.items():
            if u != 's' and v != 't' and flow > 0:
                depth_map[u] += flow

    return depth_map

def parseargs():
    parser = argparse.ArgumentParser(description="Photometric Dense 3D Reconstruction--Gut SLAM")
    parser.add_argument('--image_path', type=str, help='Path to the image file')
    parser.add_argument('--output_dir', type=str, default='./image_output', help='Depth map output directory')
    parser.add_argument('--pcl_output', type=str, default='./pcl_output', help='Point cloud output directory')

    return parser.parse_args()

if __name__ == '__main__':
    args = parseargs()
    img = cv2.imread(args.image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    depth_map = compute_img_depths(img_gray)

    print("Reconstruction Complete!")
