import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

def icp(source_points, target_points, max_iterations=1000, tolerance=1e-4):
    """
    Perform ICP (Iterative Closest Point) algorithm for 2D point cloud alignment.

    Parameters:
    - source_points: numpy array of shape (N, 2) representing the source point cloud.
    - target_points: numpy array of shape (N, 2) representing the target point cloud.
    - max_iterations: Maximum number of iterations for the ICP algorithm.
    - tolerance: Convergence criterion. The algorithm stops if the change is smaller than this value.

    Returns:
    - transformation_matrix: 2x2 transformation matrix.
    - aligned_points: Transformed source points.
    """

    # Initialize transformation matrix
    transformation_matrix = np.identity(2)

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
        if np.abs(np.trace(R) - 2) < tolerance:
            break

    aligned_points = source_points

    return transformation_matrix, aligned_points

# Generate some random points for source and target point clouds
np.random.seed(100)
source_points = np.random.rand(10, 2)
theta = np.pi / 4  # Rotation angle in radians
rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
translation_vector = np.array([1.0, 2.0])
target_points = np.dot(source_points, rotation_matrix.T) + translation_vector

# Add some noise to the source points
source_points += 0.1 * np.random.randn(10, 2)

# Apply ICP algorithm
transformation_matrix, aligned_points = icp(source_points, target_points)

# Plot the results
plt.scatter(target_points[:, 0], target_points[:, 1], label='Target Points')
plt.scatter(source_points[:, 0], source_points[:, 1], label='Source Points', alpha=0.5)
plt.scatter(aligned_points[:, 0], aligned_points[:, 1], label='Aligned Points', marker='x')
plt.legend()
plt.title('ICP Algorithm for 2D Point Cloud Alignment')
plt.show()