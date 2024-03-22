import numpy as np
import open3d as o3d
from scipy.optimize import minimize

def rigid_transform_3D(A, B):
    """
    Compute the rigid transformation between point clouds A and B.

    Parameters:
    - A: Nx3 numpy array representing the source point cloud
    - B: Nx3 numpy array representing the target point cloud

    Returns:
    - R: 3x3 rotation matrix
    - T: 1x3 translation vector
    """
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    H = np.dot((A - centroid_A).T, B - centroid_B)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(U, Vt)
    T = centroid_B - np.dot(R, centroid_A)

    return R, T

def deformable_transform_3D(X, deformation):
    """
    Apply non-rigid deformation to the point cloud X.

    Parameters:
    - X: Nx3 numpy array representing the source point cloud
    - deformation: Mx3 numpy array representing the non-rigid deformation field

    Returns:
    - Y: Nx3 numpy array representing the deformed point cloud
    """
    Y = X + deformation
    return Y

def cost_function(params, X, Y):
    """
    Define the cost function for optimization.

    Parameters:
    - params: 1D array containing the concatenation of rotation matrix and translation vector
    - X: Nx3 numpy array representing the source point cloud
    - Y: Nx3 numpy array representing the target point cloud

    Returns:
    - cost: Scalar value representing the cost of the transformation
    """
    num_points = X.shape[0]
    R = params[:9].reshape((3, 3))
    T = params[9:]
    transformed_X = np.dot(X, R.T) + T
    # if transformed_X.shape[0]==Y.shape[0]:
    #     cost= np.sum(np.linalg.norm(transformed_X - Y, axis=1) ** 2) / num_points
    # else:
    #     print("Error: The number of points in the transformed point cloud is not equal to the number of points in the target point cloud")

    
    cost = np.sum(np.linalg.norm(transformed_X - Y, axis=1) ** 2) / num_points

    return cost

def icp_non_rigid(X, Y, max_iterations=100):
    """
    Perform non-rigid ICP for point cloud registration.

    Parameters:
    - X: Nx3 numpy array representing the source point cloud
    - Y: Nx3 numpy array representing the target point cloud
    - max_iterations: Maximum number of iterations for optimization

    Returns:
    - deformed_X: Nx3 numpy array representing the aligned source point cloud
    """
    # Initialize transformation parameters
    params_init = np.zeros(12)
    
    # Optimize using scipy minimize
    result = minimize(cost_function, params_init, args=(X, Y), method='L-BFGS-B', options={'maxiter': max_iterations})

    # Obtain optimized transformation
    R_opt = result.x[:9].reshape((3, 3))
    T_opt = result.x[9:]
    deformed_X = deformable_transform_3D(X, T_opt)

    return deformed_X

def visualize_point_clouds(X, Y, deformed_X):
    """
    Visualize the original and aligned point clouds.

    Parameters:
    - X: Nx3 numpy array representing the source point cloud
    - Y: Nx3 numpy array representing the target point cloud
    - deformed_X: Nx3 numpy array representing the aligned source point cloud
    """
    source_cloud = o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(X)

    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(Y)

    deformed_cloud = o3d.geometry.PointCloud()
    deformed_cloud.points = o3d.utility.Vector3dVector(deformed_X)

    o3d.visualization.draw_geometries([source_cloud, target_cloud, deformed_cloud])

def save_point_cloud(point_cloud, filename):
    """
    Save the point cloud to a file.

    Parameters:
    - point_cloud: Nx3 numpy array representing the point cloud
    - filename: Name of the file to save the point cloud (e.g., 'result.ply')
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.io.write_point_cloud(filename, pcd)


def read_point_cloud_from_txt(file_path):
    """
    Read 3D point cloud from a text file.

    Parameters:
    - file_path: Path to the text file containing point cloud data

    Returns:
    - point_cloud: Nx3 numpy array representing the point cloud
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        point_cloud = np.array([list(map(float, line.strip().split())) for line in lines])

    return point_cloud


if __name__ == "__main__":
  
    X = read_point_cloud_from_txt('/Users/ekole/Dev/gut_slam/photometric_rec/py/pcl_output/point_cloud1.txt')
    Y = read_point_cloud_from_txt('/Users/ekole/Dev/gut_slam/photometric_rec/py/pcl_output/point_cloud1.txt')

    # Perform non-rigid ICP
    deformed_X = icp_non_rigid(X, Y)

    # Visualize the result
    visualize_point_clouds(X, Y, deformed_X)

    # Save the deformed point cloud
    save_point_cloud(deformed_X, "deformed_point_cloud.ply")