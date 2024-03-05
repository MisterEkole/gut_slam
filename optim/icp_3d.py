import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

def icp_3d(source_points, target_points, max_iterations=100, tolerance=1e-4):
    """
    Perform ICP (Iterative Closest Point) algorithm for 3D point cloud alignment.

    Parameters:
    - source_points: numpy array of shape (N, 3) representing the source point cloud.
    - target_points: numpy array of shape (N, 3) representing the target point cloud.
    - max_iterations: Maximum number of iterations for the ICP algorithm.
    - tolerance: Convergence criterion. The algorithm stops if the change is smaller than this value.

    Returns:
    - transformation_matrix: 3x3 transformation matrix.
    - aligned_points: Transformed source points.
    """

    # Initialize transformation matrix
    transformation_matrix = np.identity(3)

    for iteration in range(max_iterations):
        # Find the nearest neighbors using KDTree
        tree = KDTree(target_points)
        distances, indices = tree.query(source_points)

        # Filter out points with large distances
        valid_indices = distances < tolerance
        if not np.any(valid_indices):
            break  # No valid points found, exit the loop

        source_matched = source_points[valid_indices]
        target_matched = target_points[indices[valid_indices]]

        # Calculate the centroid of the matched points
        centroid_source = np.mean(source_matched, axis=0)
        centroid_target = np.mean(target_matched, axis=0)

        # Center the matched points
        centered_source = source_matched - centroid_source
        centered_target = target_matched - centroid_target

        # Calculate the covariance matrix H
        H = np.dot(centered_source.T, centered_target)

        # Singular Value Decomposition (SVD) to find the rotation matrix
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # Update the transformation matrix
        transformation_matrix = np.dot(transformation_matrix, R)

        # Update the source points
        source_points = np.dot(source_points - centroid_source, R.T) + centroid_target

        # Check for convergence
        if np.linalg.norm(np.eye(3) - R) < tolerance:
            break

    aligned_points = source_points

    return transformation_matrix, aligned_points

# Generate some random points for source and target point clouds
np.random.seed(42)
source_points_3d = np.random.rand(10, 3)
rotation_matrix_3d = np.array([[0.8, -0.5, 0.2],
                              [0.3, 0.9, -0.1],
                              [-0.1, 0.2, 0.95]])
translation_vector_3d = np.array([1.0, 2.0, -3.0])
target_points_3d = np.dot(source_points_3d, rotation_matrix_3d.T) + translation_vector_3d

# Add some noise to the source points
source_points_3d += 0.1 * np.random.randn(10, 3)

# Apply ICP algorithm for 3D
transformation_matrix_3d, aligned_points_3d = icp_3d(source_points_3d, target_points_3d)

# Plot the results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(target_points_3d[:, 0], target_points_3d[:, 1], target_points_3d[:, 2], label='Target Points')
ax.scatter(source_points_3d[:, 0], source_points_3d[:, 1], source_points_3d[:, 2], label='Source Points', alpha=0.5)
ax.scatter(aligned_points_3d[:, 0], aligned_points_3d[:, 1], aligned_points_3d[:, 2], label='Aligned Points', marker='x')
ax.legend()
ax.set_title('ICP Algorithm for 3D Point Cloud Alignment')
plt.show()
