#python script to read point cloud data from file and render it using open3d

import open3d as o3d
import logging
logging.basicConfig(level=logging.ERROR)

import numpy as np

def load_and_render_point_cloud(file_path):
    # Load point cloud data from file
    points = np.loadtxt(file_path, delimiter=' ')
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])

if __name__ == "__main__":
    # Specify the path to your point cloud file (replace 'your_point_cloud_file.txt' with your actual file path)
    file_path = '/Users/ekole/Dev/gut_slam/photometric_rec/py/point_cloud1.txt'

    # Load and render the point cloud
    load_and_render_point_cloud(file_path)