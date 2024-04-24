''' Utils for pose  and deformation estimation 
Author: Mitterand Ekole
Date: 16-03-2024
'''

import numpy as np
import open3d as o3d
import pyvista as pv
import scipy
from scipy.optimize import least_squares
from scipy.interpolate import make_interp_spline, BSpline, RectBivariateSpline,SmoothBivariateSpline
import scipy.special
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
    def __init__(self, radius, height, vanishing_pts, center=(0,0,0), resolution=100):
        self.radius = radius
        self.height = height
        self.center = np.array(center)
        self.resolution = resolution
        self.vanishing_pts = np.array(vanishing_pts)
        self.cylinder = self.create_cylinder()

    def create_cylinder(self):
        'Generate cylindrical mesh considering vp and center'
        direction = self.vanishing_pts - self.center
        cylinder = pv.Cylinder(radius=self.radius, height=self.height, direction=direction, center=self.center, resolution=self.resolution)
        return cylinder

    def apply_deformation_axis(self, strength=0.1, frequency=1):
        'Apply non-rigid deformation to the cylinder mesh'
        points = self.cylinder.points
        points[:, 0] += strength * np.sin(frequency * points[:, 0])
        points[:, 1] += strength * np.cos(frequency * points[:, 1])
        points[:, 2] += strength * np.sin(frequency * points[:, 2])
        self.cylinder.points = points

    def apply_deformation(self, strength=0.1, frequency=1):
        points = self.cylinder.points
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        twist_phi = phi + strength * np.sin(frequency * z)
        points[:, 0] = r * np.cos(twist_phi)
        points[:, 1] = r * np.sin(twist_phi)
        self.cylinder.points = points

    def apply_shrinking(self, start_radius=None, end_radius=None):
        if start_radius is None:
            start_radius = self.radius
        if end_radius is None:
            end_radius = self.radius * 0.9
        points = self.cylinder.points
        z = points[:, 2]
        z_normalized = (z - z.min()) / (z.max() - z.min())
        new_radii = start_radius * (1 - z_normalized) + end_radius * z_normalized
        r = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        phi = np.arctan2(points[:, 1], points[:, 0])
        r_shrunk = r / r.max() * new_radii
        points[:, 0] = r_shrunk * np.cos(phi)
        points[:, 1] = r_shrunk * np.sin(phi)
        self.cylinder.points = points

    def extract_pts(self):
        pcd = self.cylinder.points
        return pcd

    def save_pts(self, filename):
        points = self.extract_pts()
        np.savetxt(filename, points, delimiter=',')

    def densify_pts(self, target_count):
        points = self.extract_pts()
        current_count = len(points)
        factor = np.ceil(target_count / current_count).astype(int)
        densified_points = np.empty((0, 3), dtype=np.float64)
        for point in points:
            repeated_points = np.tile(point, (factor, 1))
            densified_points = np.vstack((densified_points, repeated_points))
        self.cylinder.points = densified_points[:target_count]

    def b_spline_mesh_deformation(self, control_points, strength=0.3):
        M, N, _ = control_points.shape
        heights = np.linspace(0, self.height, M)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        cp_x = control_points[:, :, 0]
        cp_y = control_points[:, :, 1]
        cp_z = control_points[:, :, 2]
        spline_x = RectBivariateSpline(heights, angles, cp_x, s=10 )
        spline_y = RectBivariateSpline(heights, angles, cp_y, s=10)
        spline_z = RectBivariateSpline(heights, angles, cp_z, s=10)
        points=self.cylinder.points
        for i, point in enumerate(self.cylinder.points):
            h = point[2] 
            theta = np.arctan2(point[1] - self.center[1], point[0] - self.center[0]) % (2 * np.pi)
            new_x = spline_x(h, theta, grid=False)
            new_y = spline_y(h, theta, grid=False)
            new_z = spline_z(h, theta, grid=False)
           
            point[0] = new_x*strength
            point[1] = new_y*strength
            point[2] = new_z

        self.cylinder.points = points 
    def b_mesh_deformation(self, a,b,control_points):
        M, N, _ = control_points.shape
        heights = np.linspace(0, self.height, M)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)

        heights, angles=np.meshgrid(heights,angles)
        heights=heights.ravel()
        angles=angles.ravel()

        

        cp_x = control_points[:, :, 0].ravel()
        cp_y = control_points[:, :, 1].ravel()
        cp_z = control_points[:, :, 2].ravel()
        # spline_x = RectBivariateSpline(heights, angles, cp_x, s=50 )
        # spline_y = RectBivariateSpline(heights, angles, cp_y, s=0.2)
        # spline_z = RectBivariateSpline(heights, angles, cp_z, s=20)

        spline_x = SmoothBivariateSpline(heights, angles, cp_x, s=M*N)
        spline_y = SmoothBivariateSpline(heights, angles, cp_y, s=M*N)
        spline_z = SmoothBivariateSpline(heights, angles, cp_z, s=M*N)
        deformed_pts=[]
        for point in self.cylinder.points:
            h=point[2]
            theta=np.arctan2(point[1]-self.center[1],point[0]-self.center[0]) % (2*np.pi)
            
            deformed_x=0
            deformed_y=0
            deformed_z=0
            alpha_max=np.mean(b)


            for i in range(M):
              
                B_i=(b/(2*np.pi))**i*(1-b/(2*np.pi))**(N-i) #influence def of control points
              
              
                
                for j in range(N):
                    B_j=(a / alpha_max) * (1 - a / alpha_max)**(N - j) #influence deformation of control points
               

                B_i /= np.linalg.norm(B_i)
                B_j /= np.linalg.norm(B_j)
                new_x = spline_x(h, theta, grid=False)
                new_y = spline_y(h, theta, grid=False)
                new_z = spline_z(h, theta, grid=False)
                 
                deformed_x += B_i * B_j*new_x*0.8
    
                deformed_y += B_i * B_j *new_y*0.8
                    
                deformed_z += B_i *B_j *new_z*0.8
                    

            deformed_pts.append([deformed_x,deformed_y,deformed_z])
        self.cylinder.points=deformed_pts
   




    
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
    @staticmethod
    def get_camera_parameters(image_height, image_width):
        fx=2.22*(image_width/36)
        #fx=735.37
        fy=fx
        cx=image_height/2
        cy=image_width/2

        camera_matrix=np.array([[fx, 0, cx],
                             [0, fy, cy],
                             [0, 0, 1]])
        
        return camera_matrix

    def project_points(self,points_3d):
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
        #points_2d_homogeneous=np.dot(self.camera_matrix,homogeneous_3d.T)
        points_2d_homogeneous=np.dot(homogeneous_3d,self.camera_matrix.T)

         # Check for division by zero
        mask = (points_2d_homogeneous[:, 2] != 0)
        
        # Convert from homogeneous coordinates to 2D
        #points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, [2]]
        points_2d = np.empty_like(points_2d_homogeneous[:, :2])
        points_2d[mask] = points_2d_homogeneous[mask, :2] / points_2d_homogeneous[mask, 2:]
        points_2d[~mask] = np.nan
        
        return points_2d
    


class Project3D_2D_cam:
    def __init__(self, intrinsic_matrix, rotation_matrix, translation_vector):
        """
        Initializes the projector with camera intrinsic and extrinsic parameters.

        Parameters:
        - intrinsic_matrix: A 3x3 numpy array representing the intrinsic camera parameters.
        - rotation_matrix: A 3x3 numpy array representing the rotation part of the extrinsic parameters.
        - translation_vector: A 3x1 numpy array representing the translation part of the extrinsic parameters.
        """
        self.intrinsic_matrix = np.array(intrinsic_matrix)
        self.rotation_matrix = np.array(rotation_matrix)
        self.translation_vector = np.array(translation_vector).reshape(3, 1)

    def project_points(self, points_3d):
        num_points = points_3d.shape[0]
        # Convert points to homogeneous coordinates for extrinsic transformation
        homogeneous_3d = np.hstack((points_3d, np.ones((num_points, 1))))

        # Apply extrinsic parameters (rotation and translation)
        camera_coords = (self.rotation_matrix @ homogeneous_3d[:, :3].T + self.translation_vector).T

        # Apply intrinsic parameters (note: only take the X, Y, Z without the homogeneous '1')
        image_points_homogeneous = (self.intrinsic_matrix @ camera_coords.T).T  # Use camera_coords directly

        # Normalize by the third (z) coordinate to project onto the image plane
        points_2d = image_points_homogeneous[:, :2] / image_points_homogeneous[:, [2]]

        return points_2d
    @staticmethod
    def get_camera_parameters(image_height, image_width, rotation_vector, translation_vector):
        """
        Generates camera intrinsic matrix and sets rotation and translation vectors for extrinsic parameters.
        """
        #fx = 200
        #fx = image_width / (2 * np.tan(90 / 2 * np.pi / 180))
        fx=2.22*(image_width/36)
        fy = fx
        cx = image_width / 2
        cy = image_height / 2

        intrinsic_matrix = np.array([[fx, 0, cx],
                                     [0, fy, cy],
                                     [0, 0, 1]])
        
        # Assuming rotation_vector is already in the form of a rotation matrix or can be converted to one
        rotation_matrix = np.array(rotation_vector).reshape(3, 3)
        translation_matrix = np.array(translation_vector).reshape(3, 1)

        return intrinsic_matrix, rotation_matrix, translation_matrix


# =============================================================================
# =============================================================================
# Utils  for Photometric model
# =============================================================================
# =============================================================================

def light_spread_func(x,k):
    return np.power(np.abs(x),k)

def calib_p_model(x,y,z,k,g_t,gamma):
    ''' Computes Ical based on calib model'''
    mu=light_spread_func(z,k)
    fr_thetha=1/np.pi
    cent_to_pix=np.linalg.norm(np.array([x,y,z]))
    thetha=2*(np.arccos(np.linalg.norm(np.array([x,y]))/cent_to_pix))/np.pi
    L=(mu/cent_to_pix)*fr_thetha*np.cos(thetha)*g_t
    L=np.power(np.abs(L),gamma)
    return L

def cost_func(I,L,sigma=1e-3):
    '''Computes the cost function for the photometric model'''
    if np.linalg.norm(I-L)<sigma:
        norm=np.linalg.norm(I-L)/(2*sigma)
    else:
        norm=np.abs(I-L)+(sigma/2)
    return norm  #return photometric error

def reg_func(grad,sigma=1e-3):
    '''Computes the regularization function for the photometric model'''
    g=np.exp(-np.linalg.norm(grad))
    if np.linalg.norm(grad)<sigma:
        norm=np.power(np.linalg.norm(grad),2)/(2*sigma)
    else:
        norm=np.abs(grad)+(sigma/2)
    return g*norm

def get_pixel_intensity(pixel):
    '''Computes the intensity of a pixel'''
    r,g,b=pixel
    return (float(r)+float(g)+float(b))/(255*3)



# =============================================================================
# =============================================================================
# Utils for bundle adjustment
# =============================================================================
# =============================================================================
# =============================================================================
def reprojection_error(projected_2d_pts, points_2d):
    ''' Compute reprojection error between projected points and observed points'''
    geodesic_error = np.linalg.norm(projected_2d_pts - points_2d)**2
    return geodesic_error


''' euler angles to rot_matrix)'''

def euler_to_rot_mat(yaw, pitch, roll):
    Rz_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    Ry_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rx_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
   
    rot_mat = Rz_yaw @ Ry_pitch @ Rx_roll #xyz order
    return rot_mat
##=============================================================================
##=============================================================================
## Objective Func--1 for bundle adjustment
##=============================================================================
##=============================================================================

def objective_function(params, points_3d, points_2d_observed, image, intrinsic_matrix, k, g_t, gamma, warp_field):
    if not isinstance(points_2d_observed, np.ndarray):
        points_2d_observed = np.array(points_2d_observed)

    rotation_matrix = params[:9].reshape(3, 3)
    translation_vector = params[9:12]
    deformation_strength = params[12]
    deformation_frequency = params[13]
    
    # Update deformation parameters
    warp_field.apply_deformation_axis(strength=deformation_strength, frequency=deformation_frequency)
    points_3d_deformed = warp_field.extract_pts()

    projector = Project3D_2D_cam(intrinsic_matrix, rotation_matrix, translation_vector)
    projected_2d_pts = projector.project_points(points_3d_deformed)
    #print("Before error line:", type(points_2d_observed), points_2d_observed.shape)

    #Resize projected_2d_pts to match points_2d_observed
    if projected_2d_pts.shape[0] > points_2d_observed.shape[0]:
        projected_2d_pts = projected_2d_pts[:points_2d_observed.shape[0], :]
    elif projected_2d_pts.shape[0] < points_2d_observed.shape[0]:
        points_2d_observed = points_2d_observed[:projected_2d_pts.shape[0], :]
    
    points_2d_observed=points_2d_observed.reshape(-1, 2)
    
        
    reprojection_error = (np.linalg.norm(projected_2d_pts - points_2d_observed, axis=1))
    lamda_reg = 1.0
    
    photometric_error = []
    for pt2d, pt3d in zip(projected_2d_pts, points_3d_deformed):
        x, y, z = pt3d
        L = calib_p_model(x, y, z, k, g_t, gamma)
        if 0 <= int(pt2d[0]) < image.shape[1] and 0 <= int(pt2d[1]) < image.shape[0]:  #check if pts2d is within boundary of image, then proceed to compute pixel intensity
            pixel_intensity = get_pixel_intensity(image[int(pt2d[1]), int(pt2d[0])])
            C = cost_func(pixel_intensity, L)
        else:
            C = 0 
      
        
        photometric_error.append(C)
    grad=np.mean(np.ones((projected_2d_pts.shape[0],)))
    reg=reg_func(grad)
    photometric_error = np.array(photometric_error+lamda_reg*reg)


    ''' adding rotation matrix constraints'''
    #Orthogonality constraint
    ortho_pen_scale=10
    ortho_penalty = ortho_pen_scale * np.linalg.norm(np.dot(rotation_matrix, rotation_matrix.T) - np.eye(3))
    ortho_penalty*=ortho_pen_scale

    #derterminant constraint
    det_pen_scale=10
    det_penalty = det_pen_scale * ((np.linalg.det(rotation_matrix) - 1)**2)
    det_penalty*=det_pen_scale

    errors = np.concatenate([reprojection_error, photometric_error.flatten()])
    errors=np.append(errors, [ortho_penalty, det_penalty])
    
    def normalize_errors(errors, target_scale=100):
        mean_error = np.mean(errors)
        scale_factor = target_scale / mean_error if mean_error else 1
        normalized_errors = errors * scale_factor
        return normalized_errors #outputs a normalised error array
    

    #normalize_errors=normalize_errors(errors, target_scale=100)
    #print(" The photometric error : ", np.mean(photometric_error)) 
    #print("Reprojection error: ", np.mean(reprojection_error))
    return errors #outputs the mean error of the normalised error array



''' objective func with ortho and det constraints for single frame with penalty scale factor approach'''
# def objective_function(params, points_3d, points_2d_observed, image, intrinsic_matrix, k, g_t, gamma, warp_field):
#     if not isinstance(points_2d_observed, np.ndarray):
#         points_2d_observed = np.array(points_2d_observed)

#     rotation_matrix = params[:9].reshape(3, 3)
#     translation_vector = params[9:12]
#     deformation_strength = params[12]
#     deformation_frequency = params[13]
    
#     # Update deformation parameters
#     warp_field.b_spline_deformation(strength=deformation_strength, frequency=deformation_frequency)
#     points_3d_deformed = warp_field.extract_pts()

#     projector = Project3D_2D_cam(intrinsic_matrix, rotation_matrix, translation_vector)
#     projected_2d_pts = projector.project_points(points_3d_deformed)
#     # print("Projected 2D points shape: ", projected_2d_pts.shape)
#     # print("Observed 2D points shape: ", points_2d_observed.shape)

#     # Resize projected_2d_pts to match points_2d_observed
#     if projected_2d_pts.shape[0] > points_2d_observed.shape[0]:
#         projected_2d_pts = projected_2d_pts[:points_2d_observed.shape[0], :]
#     elif projected_2d_pts.shape[0] < points_2d_observed.shape[0]:
#         points_2d_observed = points_2d_observed[:projected_2d_pts.shape[0], :]
    
    
    
#     points_2d_observed = points_2d_observed.reshape(-1, 2)
    
#     # Compute reprojection error
#     reprojection_error = np.linalg.norm(projected_2d_pts - points_2d_observed, axis=1)
    
#     # Compute photometric error
#     photometric_error = []
#     for pt2d, pt3d in zip(projected_2d_pts, points_3d_deformed):
#         x, y, z = pt3d
#         L = calib_p_model(x, y, z, k, g_t, gamma)
#         if 0 <= int(pt2d[0]) < image.shape[1] and 0 <= int(pt2d[1]) < image.shape[0]:  # Check if pt2d is within image boundary
#             pixel_intensity = get_pixel_intensity(image[int(pt2d[1]), int(pt2d[0])])
#             C = cost_func(pixel_intensity, L)
#         else:
#             C = 0
#         photometric_error.append(float(C))
#     photometric_error = np.array(photometric_error, dtype=float)

#     # Normalize each error type to the same scale
#     reprojection_error /= (np.linalg.norm(reprojection_error) + 1e-8)
#     photometric_error /= (np.linalg.norm(photometric_error) + 1e-8)

#     # Compute rotation matrix constraints (penalties)
#     ortho_penalty = 100 * np.linalg.norm(np.dot(rotation_matrix, rotation_matrix.T) - np.eye(3))
#     det_penalty = 100 * (abs(np.linalg.det(rotation_matrix) - 1))**2

#     # Normalize penalties
#     total_penalty = ortho_penalty + det_penalty
#     total_penalty /= (total_penalty + 1e-8)

#     # Combine errors

#     global optimization_errors
#     optimization_errors.append(
#         {
#             'reprojection_error': np.mean(reprojection_error),
#             'photometric_error': np.mean(photometric_error),
#         }
#     )
#     errors = np.concatenate([reprojection_error, photometric_error, np.array([total_penalty])])
  

#     return errors


''' Optim params func for single frame ba with penalty scale factor approach'''
# def optimize_params(points_3d, points_2d_observed, image, intrinsic_matrix, initial_params, k, g_t, gamma, warp_field, frame_idx):
#     global optimization_errors
#     optimization_errors=[]
#     # result = least_squares(objective_function, 
#     #                        initial_params,
#     #                      args=(points_3d, points_2d_observed, image, intrinsic_matrix, k, g_t, gamma, warp_field), 
#     #                      method='lm', max_nfev=2000, gtol=1e-6)
#     result = least_squares(objective_function, 
#                        initial_params,
#                        args=(points_3d, points_2d_observed, image, intrinsic_matrix, k, g_t, gamma, warp_field), 
#                        method='trf',  # Trust Region Reflective algorithm s
#                        bounds=([-np.inf]*9 + [-np.inf, -np.inf, -np.inf] + [0, 0],  # Lower bounds for def params, rot and translation no bounds
#                                [np.inf]*9 + [np.inf, np.inf, np.inf] + [np.inf, np.inf]),  # Upper bounds for def params, rot and translation no bounds
#                        #max_nfev=5000, 
#                        gtol=1e-8,
#                        tr_solver='lsmr'
#                        #verbose=2
#                        )
    
#     log_errors(optimization_errors, frame_idx)

#     return result.x


''' optim params func with multi frame ba using  penalty_scale approach'''
# def optimize_params(points_3d, points_2d_observed, image, intrinsic_matrix, initial_params, k, g_t, gamma, warp_field, frame_idx):
#     global optimization_errors
#     optimization_errors=[]
#     # result = least_squares(objective_function, 
#     #                        initial_params,
#     #                      args=(points_3d, points_2d_observed, image, intrinsic_matrix, k, g_t, gamma, warp_field), 
#     #                      method='lm', max_nfev=2000, gtol=1e-6)
#     result = least_squares(objective_function, 
#                        initial_params,
#                        args=(points_3d, points_2d_observed, image, intrinsic_matrix, k, g_t, gamma, warp_field), 
#                        method='trf',  # Trust Region Reflective algorithm s
#                        bounds=([-np.inf]*9 + [-np.inf, -np.inf, -np.inf] + [-np.inf, -np.inf],  # Lower bounds for def params, rot and translation no bounds
#                                [np.inf]*9 + [np.inf, np.inf, np.inf] + [np.inf, np.inf]),  # Upper bounds for def params, rot and translation no bounds
#                        max_nfev=1000, 
#                        gtol=1e-6,
#                        tr_solver='lsmr'
#                        #verbose=2
#                        )
    
#     log_errors(optimization_errors, frame_idx)

#     return result.x

''' Objective Function with Ortho and Det Contraints on Rot Mat using a penalty_scale faactor approach'''
# def objective_function(params, points_3d, points_2d_observed, image, intrinsic_matrix, k, g_t, gamma, warp_field):
#     if not isinstance(points_2d_observed, np.ndarray):
#         points_2d_observed = np.array(points_2d_observed)

#     rotation_matrix = params[:9].reshape(3, 3)
#     translation_vector = params[9:12]
#     deformation_strength = params[12]
#     deformation_frequency = params[13]
    
#     # Update deformation parameters
#     warp_field.b_spline_deformation(strength=deformation_strength, frequency=deformation_frequency)
#     points_3d_deformed = warp_field.extract_pts()

#     projector = Project3D_2D_cam(intrinsic_matrix, rotation_matrix, translation_vector)
#     projected_2d_pts = projector.project_points(points_3d_deformed)
#     # print("Projected 2D points shape: ", projected_2d_pts.shape)
#     # print("Observed 2D points shape: ", points_2d_observed.shape)

#     # Resize projected_2d_pts to match points_2d_observed
#     if projected_2d_pts.shape[0] > points_2d_observed.shape[0]:
#         projected_2d_pts = projected_2d_pts[:points_2d_observed.shape[0], :]
#     elif projected_2d_pts.shape[0] < points_2d_observed.shape[0]:
#         points_2d_observed = points_2d_observed[:projected_2d_pts.shape[0], :]
    
    
    
#     points_2d_observed = points_2d_observed.reshape(-1, 2)
    
#     # Compute reprojection error
#     reprojection_error = np.linalg.norm(projected_2d_pts - points_2d_observed, axis=1)
    
#     # Compute photometric error
#     photometric_error = []
#     for pt2d, pt3d in zip(projected_2d_pts, points_3d_deformed):
#         x, y, z = pt3d
#         L = calib_p_model(x, y, z, k, g_t, gamma)
#         if 0 <= int(pt2d[0]) < image.shape[1] and 0 <= int(pt2d[1]) < image.shape[0]:  # Check if pt2d is within image boundary
#             pixel_intensity = get_pixel_intensity(image[int(pt2d[1]), int(pt2d[0])])
#             C = cost_func(pixel_intensity, L)
#         else:
#             C = 0
#         photometric_error.append(float(C))
#     photometric_error = np.array(photometric_error, dtype=float)

#     # Normalize each error type to the same scale
#     reprojection_error /= (np.linalg.norm(reprojection_error) + 1e-8)
#     photometric_error /= (np.linalg.norm(photometric_error) + 1e-8)

#     # Compute rotation matrix constraints (penalties)
#     ortho_penalty = 10 * np.linalg.norm(np.dot(rotation_matrix, rotation_matrix.T) - np.eye(3))
#     det_penalty = 10 * (abs(np.linalg.det(rotation_matrix))-1)**2

#     # Normalize penalties
#     total_penalty = ortho_penalty + det_penalty
#     total_penalty /= (total_penalty + 1e-8)

#     # Combine errors

#     global optimization_errors
#     optimization_errors.append(
#         {
#             'reprojection_error': np.mean(reprojection_error),
#             'photometric_error': np.mean(photometric_error),
#         }
#     )
#     errors = np.concatenate([reprojection_error, photometric_error, np.array([total_penalty])])
  

#     return errors



##=============================================================================
##=============================================================================
## Visualization Utils 
##=============================================================================
##=============================================================================

def visualize_point_cloud(points):
    """
    Visualizes a point cloud using Open3D.
    
    Parameters:
    - points: A NumPy array of shape (N, 3) containing the XYZ coordinates of the points.
    """
    # Convert the NumPy array of points into an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Optionally, estimate normals to improve the visualization. This can be useful
    # for visualizing the point cloud with lighting effects.
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Visualization", point_show_normal=True)


def point_cloud_to_mesh(points):
    """
    Reconstructs a mesh from a point cloud and visualizes it using Open3D.
    
    Parameters:
    - points: A NumPy array of shape (N, 3) containing the XYZ coordinates of the points.
    """
    # Create a point cloud object from the points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estimate normals if they are not already present
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    
    # Use the Ball Pivoting algorithm (BPA) to reconstruct the mesh
    radii = [10, 100, 100, 100]  # Set radii for ball pivoting, adjust based on your point cloud density
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                   pcd,
                   o3d.utility.DoubleVector(radii))
    
    # Optionally, simplify the mesh
    dec_mesh = bpa_mesh.simplify_quadric_decimation(target_number_of_triangles=1000)
    
    # Visualize the mesh
    o3d.visualization.draw_geometries([dec_mesh], window_name="Mesh Visualization")


def visualize_mesh_from_points(points):
    """
    Creates and visualizes a mesh from a given set of points using PyVista.
    
    Parameters:
    - points: A NumPy array of shape (N, 3) containing the XYZ coordinates of the points.
    """
    # Create a PyVista point cloud object
    cloud = pv.PolyData(points)
    mesh = cloud.delaunay_2d()
    
    # Visualize the mesh
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='blue', show_edges=True)
    #plotter.add_points(points, color='red')  # Optionally add the original points on top
    plotter.show()




##=============================================================================
##=============================================================================
## Class Warpfield V1
##=============================================================================
##=============================================================================

# class WarpField:
#     """
#         Initialize the WarpField class with cylinder parameters.

#         Parameters:
#         - radius: The radius of the base of the cylinder.
#         - height: The height of the cylinder.
#         - vanishing_point: A tuple (x, y, z) representing the vanishing point which influences the cylinder's orientation.
#         - center: A tuple (x, y, z) representing the center of the base of the cylinder.
#         - resolution: The number of points around the circumference of the cylinder.
#         """
#     def __init__(self, radius, height,vanishing_pts,center=(0,0,0),resolution=100):
#         self.radius = radius
#         self.height = height
#         self.center = np.array(center)
#         self.resolution = resolution
#         self.vanishing_pts=np.array(vanishing_pts)
#         self.cylinder=self.create_cylinder()

#     def create_cylinder(self):
#         'Generate cylindrical mesh considering vp and center'
#         direction=self.vanishing_pts-self.center
#         cylinder = pv.Cylinder(radius=self.radius, height=self.height, direction=direction, center=self.center, resolution=self.resolution)
#         return cylinder

#     def apply_deformation_axis(self, strength=0.1, frequency=1):
#         'Apply non-rigid deformation to the cylinder mesh'
#         # Get the points of the cylinder mesh
#         points = self.cylinder.points
#         points[:,0]+=strength*np.sin(frequency*points[:,0]) #apply deformation to x,y,z
#         points[:,1]+=strength*np.cos(frequency*points[:,1])
#         points[:,2]+=strength*np.sin(frequency*points[:,2])
#         self.cylinder.points = points
    
#     #@staticmethod
#     def apply_deformation(self, strength=0.1, frequency=1):
#         # Get the points of the cylinder mesh
#         points = self.cylinder.points
#         # Convert to cylindrical coordinates (r, phi, z)
#         x, y, z = points[:, 0], points[:, 1], points[:, 2]
#         r = np.sqrt(x**2 + y**2)
#         phi = np.arctan2(y, x)
#         # Apply a twist deformation as a function of height (z)
#         twist_phi = phi + strength * np.sin(frequency * z)
#         # Convert back to Cartesian coordinates, keeping r constant to preserve the radius
#         points[:, 0] = r * np.cos(twist_phi)
#         points[:, 1] = r * np.sin(twist_phi)
#         # Update the cylinder points
#         self.cylinder.points = points
    
#     def apply_shrinking(self, start_radius=None, end_radius=None):
#         """
#         Apply a linear shrinking deformation along the length of the cylinder.

#         Parameters:
#         - start_radius: The radius at the base of the cylinder. If None, uses the original radius.
#         - end_radius: The desired radius at the top of the cylinder. If None, slightly less than the start_radius.
#         """
#         # If no radii are provided, use the cylinder's current radius and a slightly smaller value for the end_radius
#         if start_radius is None:
#             start_radius = self.radius
#         if end_radius is None:
#             end_radius = self.radius * 0.9  # Default shrink to 90% of the original radius

#         points = self.cylinder.points
       
#         z = points[:, 2]
        
#         # Normalize z-coordinates to range [0, 1]
#         z_normalized = (z - z.min()) / (z.max() - z.min())
        
#         # Linear interpolation between start_radius and end_radius based on z position
#         new_radii = start_radius * (1 - z_normalized) + end_radius * z_normalized
        
#         # Convert Cartesian (x, y, z) to cylindrical (r, phi, z) coordinates
#         r = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
#         phi = np.arctan2(points[:, 1], points[:, 0])
        
#         # Adjust r according to new_radii
#         r_shrunk = r / r.max() * new_radii
        
#         # Convert back to Cartesian coordinates
#         points[:, 0] = r_shrunk * np.cos(phi)
#         points[:, 1] = r_shrunk * np.sin(phi)
        
#         # Update the cylinder points
#         self.cylinder.points = points


#     def extract_pts(self):
#         pcd = self.cylinder.points
#         return pcd
    
#     def save_pts(self, filename):
#         points = self.extract_pts()
#         np.savetxt(filename, points, delimiter=',')
    

#     def densify_pts(self, target_count):
#         """
#         Densify the cylinder's point cloud to a specified target count.
#         This method interpolates new points to increase the density of the point cloud.
#         """
#         # Extract current point cloud
#         points = self.extract_pts()

       
#         # Calculate factor to increase the number of points
#         current_count = len(points)
#         factor = np.ceil(target_count / current_count).astype(int)

#         # Initialize new points array
#         densified_points = np.empty((0, 3), dtype=np.float64)

#         # Simple densification: repeat each point 'factor' times ~alternative: linear interpolation
        
#         for point in points:
#             repeated_points = np.tile(point, (factor, 1))
#             densified_points = np.vstack((densified_points, repeated_points))

        
#         self.cylinder.points = densified_points[:target_count]
#     def b_spline_deformation(self, strength=None, frequency=None):
#         """
#         Apply B-Spline deformation to cylinder mesh to mimic complex deformations in the human gut.

#         Parameters:
#         - strength: Magnitude of the deformation.
#         - frequency: Influences the degree of spread of control points, affecting curvature."""

#         num_control_points=max(3,abs(frequency*2+1))
#         heights=np.linspace(0, self.height, int(num_control_points))
#         angles=np.linspace(0, 2*np.pi, int(num_control_points), endpoint=False)

#         #control point swirl around cylinder axis

#         control_points_x=self.center[0]+np.cos(angles)*strength+self.radius
#         control_points_y=self.center[1]+np.sin(angles)*strength+self.radius
#         control_points_z=self.center[2]+heights


#         #interpolating b-spline through control points
#         spline_x=make_interp_spline(control_points_z, control_points_x, bc_type='clamped')
#         spline_y=make_interp_spline(control_points_z, control_points_y, bc_type='clamped')

#         points=self.cylinder.points

#         for i, point in enumerate(points):
#             new_x=spline_x(point[2])
#             new_y=spline_y(point[2])

#             #apply deformation based on cylinder axis to simulate "bulging" effect

#             dist_from_axis=np.sqrt((point[0]-self.center[0])**2+(point[1]-self.center[1])**2)
#             deformation_scale=strength*dist_from_axis/self.radius

#             points[i][0]=(new_x-point[0])+deformation_scale
#             points[i][1]=(new_y-point[1])+deformation_scale
#         self.cylinder.points=points