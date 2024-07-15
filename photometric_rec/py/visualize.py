import open3d as o3d
import numpy as np

def load_point_cloud(file_path):
    # Load point cloud data from file
    points = np.loadtxt(file_path, delimiter=' ')
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud

def visualize_and_save_point_cloud(point_cloud, filename):
    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)
    
    # Capture events to update the view
    vis.run()

    # Update the renderer and save the image
    vis.update_renderer()
    vis.capture_screen_image(filename)
    vis.destroy_window()

if __name__ == "__main__":
    file_path = '/Users/ekole/Dev/gut_slam/photometric_rec/py/2.2/point_cloud.txt'

    # Load original point cloud
    original_pc = load_point_cloud(file_path)

    # Visualize and save the point cloud with updated orientation
    visualize_and_save_point_cloud(original_pc, 'point_cloud_plot.png')