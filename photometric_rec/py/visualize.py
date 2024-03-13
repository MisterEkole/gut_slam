#python script to read point cloud data from file and render it using open3d



import open3d as o3d
import logging
logging.basicConfig(level=logging.ERROR)

import numpy as np

def load_point_cloud(file_path):
    # Load point cloud data from file
    points = np.loadtxt(file_path, delimiter=' ')
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud

def downsample_point_cloud(point_cloud, voxel_size=0.05):
    # Downsample the point cloud using voxel grid
    downsampled_pc = point_cloud.voxel_down_sample(voxel_size)
    return downsampled_pc

def estimate_vertex_normals(point_cloud):
    # Estimate vertex normals
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return point_cloud

# def compute_and_visualize_convex_hull(point_cloud):
#     # Compute convex hull
#     convex_hull = point_cloud.compute_convex_hull()

#     # Visualize the point cloud and convex hull
#     o3d.visualization.draw_geometries([point_cloud, convex_hull[0]])

def compute_and_visualize_convex_hull(point_cloud):
    # Compute convex hull
    convex_hull, _ = point_cloud.compute_convex_hull()

    # Create a TriangleMesh from the convex hull
    #mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(convex_hull.vertices),triangles=o3d.utility.Vector3iVector(np.array(convex_hull.simplices)))

    mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(convex_hull.vertices), triangles=o3d.utility.Vector3iVector(convex_hull.triangles))

    # Paint the mesh in red
    mesh.paint_uniform_color([0, 0, 0])

    # Visualize the point cloud and convex hull
    o3d.visualization.draw_geometries([point_cloud, mesh])
# def visualize_point_cloud(point_cloud, bounding_volume=True):
#     # Create an AxisAlignedBoundingBox
#     #bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(np.asarray(point_cloud.points))
#     bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(point_cloud.points))

#     # Visualize the point cloud with or without bounding volume
#     if bounding_volume:
#         o3d.visualization.draw_geometries([point_cloud, bounding_box])
#     else:
#         o3d.visualization.draw_geometries([point_cloud])

def visualize_point_cloud(point_cloud, bounding_volume=True):
    # Create an AxisAlignedBoundingBox
    bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(point_cloud.points))

    # Create a LineSet from the edges of the bounding box
    lines = [
        [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
        [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]
    ]
    colors = [[1, 0, 0] for _ in range(len(lines))]  # red color
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(bounding_box.get_box_points()),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud with or without bounding volume
    if bounding_volume:
        o3d.visualization.draw_geometries([point_cloud, line_set])
    else:
        o3d.visualization.draw_geometries([point_cloud])


if __name__ == "__main__":
    file_path = '/Users/ekole/Dev/gut_slam/photometric_rec/py/image_output/pcl_FrameBuffer_0416.png/point_cloud_FrameBuffer_0416.png.txt'

    # Load original point cloud
    original_pc = load_point_cloud(file_path)

    # Visualize the original point cloud with bounding volume
    visualize_point_cloud(original_pc)

    # Downsample point cloud
    downsampled_pc = downsample_point_cloud(original_pc)

    # Visualize the downsampled point cloud with bounding volume
    visualize_point_cloud(downsampled_pc)

    # Estimate vertex normals
    point_cloud_with_normals = estimate_vertex_normals(downsampled_pc)

    # Visualize the point cloud with vertex normals and bounding volume
    visualize_point_cloud(point_cloud_with_normals)

    #compute and visualise convex hull

    compute_and_visualize_convex_hull(original_pc)
