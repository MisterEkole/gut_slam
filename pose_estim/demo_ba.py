import numpy as np
from utils import Project3D_2D  
from scipy.optimize import least_squares

from scipy.spatial.transform import Rotation as R

class BundleAdjustment:
    def __init__(self, initial_camera_pose, camera_matrix, points_2d, points_3d):
        self.projector = Project3D_2D(camera_matrix)  # Camera projection
        self.initial_camera_pose = np.array(initial_camera_pose)  # Initial camera pose [rx, ry, rz, tx, ty, tz]
        self.points_2d = np.array(points_2d)
        self.points_3d = np.array(points_3d)
    
    def project_points(self, camera_pose, points_3d):
        # Decompose the camera pose
        rotation_vector = camera_pose[:3]
        translation_vector = camera_pose[3:]
        # Convert rotation vector to a rotation matrix
        rotation_matrix = R.from_euler('xyz', rotation_vector).as_matrix()
        
        # Apply the camera pose to the points
        points_3d_homogeneous = np.hstack((points_3d, np.ones((len(points_3d), 1))))
        projected_2d_homogeneous = np.dot(points_3d_homogeneous, np.dot(rotation_matrix.T, self.projector.camera_matrix.T) + translation_vector.reshape(-1, 1))
        points_2d = projected_2d_homogeneous[:, :2] / projected_2d_homogeneous[:, 2, np.newaxis]
        
        return points_2d

    def reprojection_error(self, params):
        camera_pose = params[:6]
        points_3d = params[6:].reshape((-1, 3))
        error = []
        for i, point_3d in enumerate(points_3d):
            projected_2d = self.project_points(camera_pose, np.array([point_3d]))
            obs_2d = self.points_2d[i]
            error.append(projected_2d - obs_2d)
        return np.array(error).ravel()

    def optimize(self):
        init_params = np.hstack((self.initial_camera_pose, self.points_3d.ravel()))
        result = least_squares(self.reprojection_error, init_params, method='lm')
        optimized_camera_pose = result.x[:6]
        optimized_points_3d = result.x[6:].reshape((self.points_3d.shape))
        return optimized_camera_pose, optimized_points_3d


def generate_synthetic_data():
    
    true_camera_pose = np.array([0.1, -0.1, 0.1, 1.0, 2.0, 3.0]) 
    
    
    fx, fy = 800, 800  
    cx, cy = 400, 300  
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]])
    
    
    points_3d = np.random.rand(10, 3) * 10 - 5  # Random points in the range [-5, 5]
    points_3d[:, 2] += 15  # Ensure points are in front of the camera (Z > 0)
    
    rotation_matrix = R.from_euler('xyz', true_camera_pose[:3]).as_matrix()
    translated_points_3d = np.dot(points_3d, rotation_matrix.T) + true_camera_pose[3:]

    # Extend points to homogeneous coordinates for projection
    points_3d_homogeneous = np.hstack((translated_points_3d, np.ones((len(translated_points_3d), 1))))
    
    # Correct projection onto 2D
    points_2d_homogeneous = np.dot(points_3d_homogeneous, camera_matrix)
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2][:, np.newaxis]
    
    return camera_matrix, true_camera_pose, points_2d, points_3d


def main():
    camera_matrix, true_camera_pose, points_2d, points_3d = generate_synthetic_data()
    
    # Initialize the BundleAdjustment class with the synthetic data
    # Note: Assuming the initial camera pose is slightly off the true pose
    initial_camera_pose = true_camera_pose + np.random.normal(0, 0.1, 6)
    ba = BundleAdjustment(initial_camera_pose, camera_matrix, points_2d, points_3d)
    
    # Optimize camera pose and 3D points
    optimized_camera_pose, optimized_points_3d = ba.optimize()
    
    print("True Camera Pose:", true_camera_pose)
    print("Initial Camera Pose:", initial_camera_pose)
    print("Optimized Camera Pose:", optimized_camera_pose)
    print("Difference (Optimized - True):", optimized_camera_pose - true_camera_pose)

if __name__ == "__main__":
    main()