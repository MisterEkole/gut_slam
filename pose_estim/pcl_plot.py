import open3d as o3d
import pyvista as pv
import numpy as np

def read_vtk_mesh(file_path):
    mesh = pv.read(file_path)
    return mesh

def vtk_to_point_cloud(mesh):
    points = np.asarray(mesh.points)
    return points

def densify_points(points, factor=2):
    """
    Densify the points by adding new points between existing points.
    The factor determines the number of points added between each pair of original points.
    """
    new_points = []
    num_points = len(points)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            line_points = np.linspace(points[i], points[j], factor + 2)
            new_points.extend(line_points)
    return np.array(new_points)

def save_point_cloud_as_image(pcd, image_path, vis):
    # Capture and save the current screen image
    vis.capture_screen_image(image_path)

def main(file_path, densify_factor=2, image_path="point_cloud_image.png"):
    # Read the VTK mesh
    mesh = read_vtk_mesh(file_path)

    # Convert VTK mesh to point cloud
    points = vtk_to_point_cloud(mesh)

    # Densify the points
    densified_points = densify_points(points, densify_factor)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(densified_points)

    # Create visualizer and add point cloud
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    # Run the visualizer
    vis.run()

    # Save the point cloud as an image
    save_point_cloud_as_image(pcd, image_path, vis)

    # Destroy the visualizer window
    vis.destroy_window()

if __name__ == "__main__":
    file_path = "./rendering/cylindrical_mesh.vtk"  # Replace with the path to your VTK file
    densify_factor = 1  # Adjust the densification factor as needed
    image_path = "point_cloud_image.png"  # Path to save the point cloud image
    main(file_path, densify_factor, image_path)