''' Utils for pose estimation 
Author: Mitterand Ekole
Date: 16-03-2024
'''

import numpy as np
import open3d as o3d

class WarpField:
    def __init__(self, mesh_vertices, control_points):
        self.mesh_vertices = mesh_vertices
        self.control_points = control_points
        self.num_vertices = len(mesh_vertices)
        self.num_control_points = len(control_points)
        self.displacement_vectors = np.zeros((self.num_control_points, 3))

    def deform_mesh(self):
        deformed_mesh = np.zeros_like(self.mesh_vertices)
        for i in range(self.num_vertices):
            weighted_sum = np.zeros(3)
            total_weight = 0
            for j in range(self.num_control_points):
                distance = np.linalg.norm(self.mesh_vertices[i] - self.control_points[j])
                weight = self.weight_function(distance)
                weighted_sum += weight * self.displacement_vectors[j]
                total_weight += weight
            if total_weight > 0:
                deformed_mesh[i] = self.mesh_vertices[i] + weighted_sum / total_weight
            else:
                deformed_mesh[i] = self.mesh_vertices[i]
        return deformed_mesh
    
    def weight_function(self, distance):
        max_distance = 1.0
        if distance <= max_distance:
            return 1 - distance / max_distance
        else:
            return 0
        

class PoseDeformationEstimator:
    def __init__(self, observed_point_cloud1, observed_point_cloud2, mesh_vertices, control_points, intrinsic_params):
        self.observed_point_cloud1 = observed_point_cloud1
        self.observed_point_cloud2 = observed_point_cloud2
        self.warp_field = WarpField(mesh_vertices, control_points)
        self.camera_pose = np.zeros(6)  
        self.intrinsic_params = intrinsic_params

    def project_mesh(self):
        translated_mesh = self.warp_field.mesh_vertices + self.camera_pose[:3]
        rotated_mesh = self.rotate_points(translated_mesh, self.camera_pose[3:])
        image_points = self.project_to_image(rotated_mesh)
        return image_points

    def project_to_image(self, points_3d):
        focal_length = self.intrinsic_params['focal_length']
        principal_point = self.intrinsic_params['principal_point']
        distortion_coeffs = self.intrinsic_params['distortion_coeffs']

        points_2d = points_3d[:, :2] / points_3d[:, 2].reshape(-1, 1) * focal_length + principal_point

        r = np.linalg.norm(points_2d - principal_point, axis=1)
        k1, k2, p1, p2 = distortion_coeffs
        radial_distortion = 1 + k1 * r ** 2 + k2 * r ** 4
        tangential_distortion_x = 2 * p1 * (points_2d[:, 0] - principal_point[0]) * (points_2d[:, 1] - principal_point[1])
        tangential_distortion_y = 2 * p2 * (points_2d[:, 1] - principal_point[1]) * (points_2d[:, 0] - principal_point[0])
        points_2d[:, 0] += tangential_distortion_x
        points_2d[:, 1] += tangential_distortion_y
        
        return points_2d

    def rotate_points(self, points, angles):
        roll, pitch, yaw = angles
        rotation_matrix = self.euler_to_rotation_matrix(roll, pitch, yaw)
        rotated_points = np.dot(points, rotation_matrix.T)
        return rotated_points

    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
        rotation_matrix = np.dot(R_z, np.dot(R_y, R_x))
        return rotation_matrix
    

    def cost_function(self, parameters):
        num_camera_pose_params = 6
        num_deformation_params = self.warp_field.num_control_points * 3
        pose_params = parameters[:num_camera_pose_params]
        deformation_params = parameters[num_camera_pose_params:num_camera_pose_params + num_deformation_params]
        
        self.warp_field.displacement_vectors = deformation_params.reshape(-1, 3)
        projected_mesh = self.project_mesh()
        
        #error1 = np.mean(np.linalg.norm(projected_mesh - self.observed_point_cloud1, axis=1))
        #error2 = np.mean(np.linalg.norm(projected_mesh - self.observed_point_cloud2, axis=1))
        error=np.mean(np.linalg.norm(self.observed_point_cloud1 - self.observed_point_cloud2, axis=1))
        return error
   


    def gauss_newton(self, initial_parameters, max_iterations=100, tolerance=1e-5):
        parameters = initial_parameters.copy()
        prev_cost = float('inf')
        for i in range(max_iterations):
            jacobian = self.numerical_jacobian(self.cost_function, parameters)
            residuals = self.numerical_residuals(self.cost_function, parameters)
            update = np.linalg.lstsq(jacobian, residuals, rcond=None)[0]
            parameters -= update
            cost = self.cost_function(parameters)
            if abs(cost - prev_cost) < tolerance:
                break
            prev_cost = cost
        return parameters

    def numerical_jacobian(self, func, params, epsilon=1e-5):
        num_params = len(params)
        num_residuals = self.observed_point_cloud1.shape[0] + self.observed_point_cloud2.shape[0]
        jacobian = np.zeros((num_residuals, num_params))
        for i in range(num_params):
            original_value = params[i]
            params[i] = original_value + epsilon
            cost_plus = func(params)
            params[i] = original_value - epsilon
            cost_minus = func(params)
            jacobian[:, i] = (cost_plus - cost_minus) / epsilon
            params[i] = original_value
        return jacobian

    def numerical_residuals(self, func, params, epsilon=1e-5):
        num_params = len(params)
        residuals = np.zeros(num_params)
        for i in range(num_params):
            original_value = params[i]
            params[i] = original_value + epsilon
            cost_plus = func(params)
            residuals[i] = (func(params) - cost_plus) / epsilon
            params[i] = original_value
        return residuals
    

def read_point_cloud(filename):
    cloud = np.loadtxt(filename)
    return cloud

def save_point_cloud(filename, point_cloud):
    np.savetxt(filename, point_cloud)

def visualize_point_cloud(point_cloud):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.visualization.draw_geometries([pcd])
