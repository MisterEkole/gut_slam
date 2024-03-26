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

    def apply_deformation_axis(self, strength=0.1, frequency=1):
        'Apply non-rigid deformation to the cylinder mesh'
        # Get the points of the cylinder mesh
        points = self.cylinder.points
        points[:,0]+=strength*np.sin(frequency*points[:,0]) #apply deformation to x,y,z
        points[:,1]+=strength*np.cos(frequency*points[:,1])
        points[:,2]+=strength*np.sin(frequency*points[:,2])
        self.cylinder.points = points
    
    def apply_deformation(self, strength=0.1, frequency=1):
        # Get the points of the cylinder mesh
        points = self.cylinder.points
        # Convert to cylindrical coordinates (r, phi, z)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        # Apply a twist deformation as a function of height (z)
        twist_phi = phi + strength * np.sin(frequency * z)
        # Convert back to Cartesian coordinates, keeping r constant to preserve the radius
        points[:, 0] = r * np.cos(twist_phi)
        points[:, 1] = r * np.sin(twist_phi)
        # Update the cylinder points
        self.cylinder.points = points
    
    def apply_shrinking(self, start_radius=None, end_radius=None):
        """
        Apply a linear shrinking deformation along the length of the cylinder.

        Parameters:
        - start_radius: The radius at the base of the cylinder. If None, uses the original radius.
        - end_radius: The desired radius at the top of the cylinder. If None, slightly less than the start_radius.
        """
        # If no radii are provided, use the cylinder's current radius and a slightly smaller value for the end_radius
        if start_radius is None:
            start_radius = self.radius
        if end_radius is None:
            end_radius = self.radius * 0.9  # Default shrink to 90% of the original radius

        points = self.cylinder.points
        # Extract z-coordinates
        z = points[:, 2]
        
        # Normalize z-coordinates to range [0, 1]
        z_normalized = (z - z.min()) / (z.max() - z.min())
        
        # Linear interpolation between start_radius and end_radius based on z position
        new_radii = start_radius * (1 - z_normalized) + end_radius * z_normalized
        
        # Convert Cartesian (x, y, z) to cylindrical (r, phi, z) coordinates
        r = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        phi = np.arctan2(points[:, 1], points[:, 0])
        
        # Adjust r according to new_radii
        r_shrunk = r / r.max() * new_radii
        
        # Convert back to Cartesian coordinates
        points[:, 0] = r_shrunk * np.cos(phi)
        points[:, 1] = r_shrunk * np.sin(phi)
        
        # Update the cylinder points
        self.cylinder.points = points


    def extract_pts(self):
        pcd = self.cylinder.points
        return pcd
    
    def save_pts(self, filename):
        points = self.extract_pts()
        np.savetxt(filename, points, delimiter=',')

    def densify_pts(self, target_count):
        """
        Densify the cylinder's point cloud to a specified target count.
        This method interpolates new points to increase the density of the point cloud.
        """
        # Extract current point cloud
        points = self.extract_pts()

       
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
    
    
class Points_Processor:
    def __init__(self, target_num_points=None, normalize=False):
        """
        Initialize the PointProcessor.

        Parameters:
        - target_num_points: The target number of points for downsampling. If None, downsampling is not performed.
        - normalize: A boolean indicating whether the point clouds should be normalized.
        """
        self.target_num_points = target_num_points
        self.normalize = normalize
    
    def read_point_from_txt(self, file_path):
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
    
    
    
    def align_points(self, source_pc, target_pc, threshold=1.0):  #optional method
        """
        Align two 3D points  using the ICP algorithm.

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


class Project3D_2D:
   

    def __init__(self, camera_matrix):
        """
        Initializes the projector with a camera matrix.

        Parameters:
        - camera_matrix: A 3x4 matrix representing the intrinsic and extrinsic camera parameters.
        """
        self.camera_matrix = np.array(camera_matrix)
    
    def get_camera_parameters(image_height, image_width):
        fx=735.37
        fy=552.0
        cx=image_height/2
        cy=image_width/2

        camera_matrix=np.array([[fx, 0, cx],
                             [0, fy, cy],
                             [0, 0, 1]])
        
        return camera_matrix
  

    


    def project_points(self, points_3d):
        """
        Projects 3D points to 2D using the camera matrix and homogeneous coordinates.

        Parameters:
        - points_3d: A Nx3 numpy array of 3D points.

        Returns:
        - A Nx2 numpy array of 2D projected points.
        """
        num_points = points_3d.shape[0]
       
        homogeneous_3d = np.hstack((points_3d, np.ones((num_points, 1))))
        homogeneous_3d=homogeneous_3d[:,:3]
        
        # Project the points using the camera matrix
        #points_2d_homogeneous = np.dot(self.camera_matrix, homogeneous_3d.T).T
        points_2d_homogeneous=np.dot(homogeneous_3d,self.camera_matrix)

         # Check for division by zero
        mask = (points_2d_homogeneous[:, 2] != 0)
        
        # Convert from homogeneous coordinates to 2D
        #points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, [2]]
        points_2d = np.empty_like(points_2d_homogeneous[:, :2])
        points_2d[mask] = points_2d_homogeneous[mask, :2] / points_2d_homogeneous[mask, 2:]
        points_2d[~mask] = np.nan
        
        return points_2d

    

