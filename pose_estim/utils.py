''' Utils for pose  and deformation estimation 
Author: Mitterand Ekole
Date: 16-03-2024
'''

import numpy as np
import open3d as o3d
import pyvista as pv

class WarpField:
    """
        Initialize the WarpField class with cylinder parameters.

        Parameters:
        - radius: The radius of the base of the cylinder.
        - height: The height of the cylinder.
        - vanishing_point: A tuple (x, y, z) representing the vanishing point which influences the cylinder's orientation.
        - center: A tuple (x, y, z) representing the center of the base of the cylinder.
        - resolution: The number of points around the circumference of the cylinder.
        """
    def __init__(self, radius, height,vanishing_pts,center=(0,0,0),resolution=100):
        self.radius = radius
        self.height = height
        self.center = np.array(center)
        self.resolution = resolution
        self.vanishing_pts=np.array(vanishing_pts)
        self.cylinder=self.create_cylinder()

    def create_cylinder(self):
        'Generate cylindrical mesh considering vp and center'
        direction=self.vanishing_pts-self.center
        cylinder = pv.Cylinder(radius=self.radius, height=self.height, direction=direction, center=self.center, resolution=self.resolution)
        return cylinder

    def apply_deformation(self, strength=0.1, frequency=1):
        'Apply non-rigid deformation to the cylinder mesh'
        # Get the points of the cylinder mesh
        points = self.cylinder.points
        #points[:,0]+=strength*np.sin(frequency*points[:,0]) #apply deformation to x,y,z
        #points[:,1]+=strength*np.cos(frequency*points[:,1])
        points[:,2]+=strength*np.sin(frequency*points[:,2])
        self.cylinder.points = points
    
    # def apply_deformation(self, strength=0.1, frequency=1):
    #     # Get the points of the cylinder mesh
    #     points = self.cylinder.points
    #     # Convert to cylindrical coordinates (r, phi, z)
    #     x, y, z = points[:, 0], points[:, 1], points[:, 2]
    #     r = np.sqrt(x**2 + y**2)
    #     phi = np.arctan2(y, x)
    #     # Apply a twist deformation as a function of height (z)
    #     twist_phi = phi + strength * np.sin(frequency * z)
    #     # Convert back to Cartesian coordinates, keeping r constant to preserve the radius
    #     points[:, 0] = r * np.cos(twist_phi)
    #     points[:, 1] = r * np.sin(twist_phi)
    #     # Update the cylinder points
    #     self.cylinder.points = points

    # def apply_deformation(self, strength=0.1, frequency=1):
    #     # Get the points of the cylinder mesh
    #     points = self.cylinder.points
        
    #     # Normalize the z-coordinates to the range [0, 1] for consistent bending regardless of the actual height
    #     z_normalized = (points[:, 2] - points[:, 2].min()) / self.height
        
    #     # Apply a sine-based bending deformation
    #     # The deformation causes the cylinder to bend along the x-axis
    #     # You can adjust this to bend along a different axis or in a different manner as needed
    #     points[:, 0] += strength * np.sin(frequency * np.pi * z_normalized)
        
    #     self.cylinder.points = points

    def extract_pcd(self):
        pcd = self.cylinder.points
        return pcd
    
    def save_point_cloud(self, filename):
        point_cloud = self.extract_pcd()
        np.savetxt(filename, point_cloud, delimiter=',')

    def densify_point_cloud(self, target_count):
        """
        Densify the cylinder's point cloud to a specified target count.
        This method interpolates new points to increase the density of the point cloud.
        """
        # Extract current point cloud
        points = self.extract_pcd()

       
        # Calculate factor to increase the number of points
        current_count = len(points)
        factor = np.ceil(target_count / current_count).astype(int)

        # Initialize new points array
        densified_points = np.empty((0, 3), dtype=np.float64)

        # Simple densification: repeat each point 'factor' times ~alternative: linear interpolation
        
        for point in points:
            repeated_points = np.tile(point, (factor, 1))
            densified_points = np.vstack((densified_points, repeated_points))

        
        self.cylinder.points = densified_points[:target_count]
    
    
class PointCloudPreparer:
    def __init__(self, target_num_points=None, normalize=False):
        """
        Initialize the PointCloudPreparer.

        Parameters:
        - target_num_points: The target number of points for downsampling. If None, downsampling is not performed.
        - normalize: A boolean indicating whether the point clouds should be normalized.
        """
        self.target_num_points = target_num_points
        self.normalize = normalize
    
    def read_point_cloud_from_txt(self, file_path):
        """
        Reads a point cloud from a text file, attempting to automatically detect the delimiter.

        Parameters:
        - file_path: The path to the .txt file containing the point cloud data.

        Returns:
        - A NumPy array containing the point cloud.
        """
        # Attempt to read with common delimiters and select the one that works
        for delimiter in [',', ' ']:
            try:
                return np.loadtxt(file_path, delimiter=delimiter)
            except ValueError:
                continue
        # If neither delimiter worked, raise an error
        raise ValueError(f"Failed to automatically detect delimiter and read point cloud data from {file_path}.")


    def downsample(self, pc):
        """Downsample the point cloud to a specific number of points."""
        if self.target_num_points is not None and pc.shape[0] > self.target_num_points:
            indices = np.random.choice(pc.shape[0], self.target_num_points, replace=False)
            return pc[indices]
        return pc

    def normalize_points(self, pc):
        """Normalize the point cloud to have zero mean and fit within the range [-1, 1]."""
        pc -= np.mean(pc, axis=0)  # Center to zero
        max_abs_val = np.max(np.abs(pc))
        pc /= max_abs_val  # Scale to [-1, 1]
        return pc

    def prepare(self, pc):
        """
        Apply downsampling and normalization to the point cloud based on the initialization parameters.
        """
        pc = self.downsample(pc)
        if self.normalize:
            pc = self.normalize_points(pc)
        return pc
    
    def align_point_clouds(self, source_pc, target_pc, threshold=1.0):
        """
        Align two point clouds using the ICP algorithm.

        Parameters:
        - source_pc: The source point cloud as a NumPy array.
        - target_pc: The target point cloud as a NumPy array to align to the source.
        - threshold: The distance threshold to consider for point matches.

        Returns:
        - Aligned version of target_pc as a NumPy array.
        """
        # Convert numpy arrays to Open3D point clouds
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(source_pc)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(target_pc)

        # Perform ICP alignment
        trans_init = np.identity(4)  # Initial transformation
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        
        # Apply the transformation to the target point cloud
        target.transform(reg_p2p.transformation)

        # Return the aligned point cloud as a numpy array
        return np.asarray(target.points)

    
def visualize_point_clouds(pc1, pc2):
    """
    Visualizes two point clouds using Open3D.

    Parameters:
    - pc1: The first point cloud as a NumPy array.
    - pc2: The second point cloud as a NumPy array.
    """
    # Convert the NumPy arrays to Open3D point cloud objects
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc1)
    pcd1.paint_uniform_color([1, 0, 0])  # Red color for the first point cloud

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pc2)
    pcd2.paint_uniform_color([0, 0, 1])  # Blue color for the second point cloud

    #visualize the prepared point clouds using Open3D
    o3d.visualization.draw_geometries([pcd1, pcd2])

