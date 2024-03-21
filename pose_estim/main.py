from utils import *


if __name__ =="__main__":
    observed_point_cloud1 = read_point_cloud("/Users/ekole/Dev/gut_slam/photometric_rec/py/pcl_output/point_cloud2.txt")
    observed_point_cloud2 = read_point_cloud("/Users/ekole/Dev/gut_slam/photometric_rec/py/pcl_output/point_cloud2.txt")
    mesh_vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    control_points = np.array([[0.2, 0.2, 0], [0.8, 0.2, 0]])

    intrinsic_params = {
        'focal_length': 100,  
        'principal_point': np.array([320, 240]), 
        'distortion_coeffs': np.ones(4)  
    }

    estimator = PoseDeformationEstimator(observed_point_cloud1, observed_point_cloud2, mesh_vertices, control_points, intrinsic_params)
    
    # Initial guess for camera pose and deformation parameters
    initial_parameters = np.zeros(6 + control_points.size)

    # Optimize parameters using Gauss-Newton
    optimized_params = estimator.gauss_newton(initial_parameters)

    # Extract optimized camera pose and deformation parameters
    optimized_pose = optimized_params[:6]
    optimized_deformation = optimized_params[6:]

    print("Optimized Camera Pose:", optimized_pose)
    print("Optimized Deformation Parameters:", optimized_deformation)

    # Deform mesh using optimized parameters
    estimator.warp_field.displacement_vectors = optimized_deformation.reshape(-1, 3)
    deformed_mesh = estimator.project_mesh()

    # Save deformed point cloud and estimated pose point cloud to files
    save_point_cloud("deformed_point_cloud.txt", deformed_mesh)
    estimated_pose_point_cloud = estimator.project_mesh()
    save_point_cloud("estimated_pose_point_cloud.txt", estimated_pose_point_cloud)

    # Visualize deformed point cloud and estimated pose point cloud
    visualize_point_cloud(deformed_mesh)
    visualize_point_cloud(estimated_pose_point_cloud)