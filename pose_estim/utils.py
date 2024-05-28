'''
-------------------------------------------------------------
Utils for GutSLAM Pose and Deformation Estimation
Author: Mitterand Ekole
Date: 16-03-2024
-------------------------------------------------------------
'''


import numpy as np
import open3d as o3d
import pyvista as pv
import scipy
from scipy.optimize import least_squares
from scipy.interpolate import make_interp_spline, BSpline, RectBivariateSpline,SmoothBivariateSpline,interp2d,LSQBivariateSpline
import scipy.special
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from sklearn.preprocessing import StandardScaler
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

   
    def b_mesh_deformation(self, a, b, control_points):
        M, N, _ = control_points.shape
        heights = np.linspace(0, self.height, M)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=True)

        heights, angles = np.meshgrid(heights, angles)
        heights = heights.ravel()
        angles = angles.ravel()
        cp_x = control_points[:, :, 0].ravel()
        cp_y = control_points[:, :, 1].ravel()
        cp_z = control_points[:, :, 2].ravel()
   
        spline_x = SmoothBivariateSpline(heights, angles, cp_x, s=M*N)
        spline_y = SmoothBivariateSpline(heights, angles, cp_y, s=M*N)
        spline_z = SmoothBivariateSpline(heights, angles, cp_z, s=M*N)
  
        pts = []
        for point in self.cylinder.points:
            h = point[2]
            theta = np.arctan2(point[1] - self.center[1], point[0] - self.center[0]) % (2 * np.pi)
        
            x = y = z = 0 
            B_i = np.zeros(M)
            B_j = np.zeros(N)

            for i in range(M):
                B_i[i] = (b / (2 * np.pi)) ** i * (1 - b / (2 * np.pi)) ** (M - i)
               
               
                for j in range(N):
                    B_j[j] = (a / np.max(a+b)) * (1 - a / np.max(a+b)) ** (N - j)
    
                B_i /= np.linalg.norm(B_i, ord=2) 
                B_j /= np.linalg.norm(B_j, ord=2) 
      
            for i in range(M):
                for j in range(N):
                    weight = B_i[i] * B_j[j]  

               
                    x += weight * spline_x.ev(h, theta)
                    y += weight * spline_y.ev(h, theta)
                    z += weight * spline_z.ev(h, theta)

            pts.append([x, y, z])  

        self.cylinder.points = np.array(pts)  
    def get_mesh_pts(self):
        return self.cylinder.points.copy()
    def extract_mesh_pts(self):
        return self.get_mesh_pts()





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
    def get_camera_parameters(image_height, image_width, rotation_vector, translation_vector,image_center):
        """
        Generates camera intrinsic matrix and sets rotation and translation vectors for extrinsic parameters.
        """
        #fx = 200
        fx = image_height / (2 * np.tan(np.radians(30)/2))
        #fx=2.22*(image_width/36)
        fy = fx
        # cx = image_width / 2
        # cy = image_height / 2
        cx,cy,_=image_center
      
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
## Generate Control Points with Uniform Grid
## Generate Cylinder Points
##=============================================================================
##=============================================================================
def generate_cylinder_points(rho_max, height, M, N):
    """
    Generates points on a cylinder surface.

    Parameters:
    rho_max (float): Maximum value of rho (radius)
    height (float): Height of the cylinder
    M (int): Number of divisions along the height and rho
    N (int): Number of divisions along the angular direction

    Returns:
    numpy.ndarray: Array of shape (M*N*M, 3) containing points in cylindrical coordinates (rho, alpha, z)
    """
    rho_values = np.linspace(0, rho_max, M)
    alpha_values = np.linspace(0, 2 * np.pi, N)
    z_values = np.linspace(0, height, M)
    
    cylinder_points = np.array([[rho, alpha, z] 
                                for rho in rho_values 
                                for alpha in alpha_values 
                                for z in z_values])
    
    return cylinder_points

def polar_to_cartesian(rho, alpha, z):
    x = rho * np.cos(alpha)
    y = rho * np.sin(alpha)
    return x, y, z

# Function to generate a uniform grid of control points
def generate_uniform_grid_control_points(rho_step_size, alpha_step_size, h_constant=None, h_variable_range=None, rho_range=(0, 50), alpha_range=(0, 2 * np.pi)):
    rho_values = np.arange(rho_range[0], rho_range[1] + rho_step_size, rho_step_size)
    alpha_values = np.arange(alpha_range[0], alpha_range[1] + alpha_step_size, alpha_step_size)
    
    control_points = []
    for rho in rho_values:
        for alpha in alpha_values:
            if h_constant is not None:
                h = h_constant
            else:
                h = np.random.uniform(*h_variable_range)
            x, y, z = polar_to_cartesian(rho, alpha, h)
            control_points.append((x, y, z))

    return np.array(control_points).reshape(len(rho_values), len(alpha_values), 3)
##=============================================================================
##=============================================================================
## Updated B-Spline Surface Model
##=============================================================================
##=============================================================================
class BMeshDeformation:
    def __init__(self, height, center, cylinder_points):
        self.height = height
        self.center = center
        self.cylinder_points = cylinder_points

    def b_mesh_deformation(self, a, b, control_points):
        M, N, _ = control_points.shape
        heights = np.linspace(0, self.height, M)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=True)

        heights, angles = np.meshgrid(heights, angles)
        heights = heights.ravel()
        angles = angles.ravel()
        cp_x = control_points[:, :, 0].ravel()
        cp_y = control_points[:, :, 1].ravel()
        cp_z = control_points[:, :, 2].ravel()

        spline_x = SmoothBivariateSpline(heights, angles, cp_x, s=M * N)
        spline_y = SmoothBivariateSpline(heights, angles, cp_y, s=M * N)
        spline_z = SmoothBivariateSpline(heights, angles, cp_z, s=M * N)

        pts = []
        for point in self.cylinder_points:
            h = point[2]
            theta = np.arctan2(point[1] - self.center[1], point[0] - self.center[0]) % (2 * np.pi)

            x = y = z = 0 
            B_i = np.zeros(M)
            B_j = np.zeros(N)

            for i in range(M):
                B_i[i] = (b / (2 * np.pi)) ** i * (1 - b / (2 * np.pi)) ** (M - i)
                for j in range(N):
                    B_j[j] = (a / np.max(a+b)) * (1 - a / np.max(a+b)) ** (N - j)
                    
                B_i /= np.linalg.norm(B_i, ord=2) 
                B_j /= np.linalg.norm(B_j, ord=2) 
                
            for i in range(M):
                for j in range(N):
                    weight = B_i[i] * B_j[j]  
                    x += weight * spline_x.ev(h, theta)
                    y += weight * spline_y.ev(h, theta)
                    z += weight * spline_z.ev(h, theta)
            pts.append([x, y, z])  

        return np.array(pts)
##=============================================================================
##=============================================================================
## Visualization Utils 
##=============================================================================
##=============================================================================
class GridViz:
    def __init__(self, grid_shape, window_size=(2300, 1500)):
        self.plotter = pv.Plotter(shape=grid_shape, window_size=window_size)

    def add_mesh_cartesian(self, points, subplot,texture_img=None):
        scaler = StandardScaler()
        points = scaler.fit_transform(points)
        cloud = pv.PolyData(points)
        mesh = cloud.delaunay_3d()
        scalars = mesh.points[:, 2]
        self.plotter.subplot(*subplot)
        #self.plotter.add_points(mesh.points, color='green', point_size=5)

        if texture_img is not None:
            mesh.texture_map_to_plane(inplace=True)
            texture=pv.read_texture(texture_img)
            #self.plotter.add_mesh(mesh, texture=texture, show_edges=False, show_scalar_bar=False)
            self.apply_texture_with_scalars(mesh, scalars=scalars,texture=texture)
        else:
            #self.plotter.add_mesh(mesh,scalars=scalars,cmap='viridis',show_edges=False,show_scalar_bar=False)
            self.plotter.add_points(mesh.points,color='green',point_size=5)
        self.plotter.add_axes(line_width=5, interactive=True)

    def add_mesh_polar(self, points, subplot,texture_img=None):
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
    
        rho = np.sqrt(x**2 + y**2)
        alpha = np.arctan2(y, x)
        h = z
    
        points = np.vstack((rho, alpha, h)).T
        scaler = StandardScaler()
        points = scaler.fit_transform(points)
        cloud = pv.PolyData(points)
        mesh = cloud.delaunay_2d()
        mesh = mesh.smooth(n_iter=600)
        scalars = mesh.points[:, 2]
        self.plotter.subplot(*subplot)
        #self.plotter.add_points(mesh.points, color='green', point_size=5)
        if texture_img is not None:
            mesh.texture_map_to_plane(inplace=True)
            texture=pv.read_texture(texture_img)
            #self.plotter.add_mesh(mesh, texture=texture, show_edges=False, show_scalar_bar=False)
            self.apply_texture_with_scalars(mesh, scalars=scalars,texture=texture)
        else:
            #self.plotter.add_mesh(mesh,scalars=scalars,cmap='viridis',show_edges=False,show_scalar_bar=False)
            self.plotter.add_points(mesh.points,color='green',point_size=5)
        self.plotter.add_axes(interactive=True, xlabel='r', ylabel='theta', zlabel='h', line_width=5)

    def add_h_surface(self, points, subplot, texture_img=None):
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
    
        rho = np.sqrt(x**2 + y**2)
        alpha = np.arctan2(y, x)
        h = z
    
        points = np.vstack((rho, alpha, h)).T
        scaler = StandardScaler()
        points = scaler.fit_transform(points)
        cloud = pv.PolyData(points)
        mesh = cloud.delaunay_2d()
        mesh = mesh.smooth(n_iter=600)
        scalars = mesh.points[:, 2]
        
        self.plotter.subplot(*subplot)
        #self.plotter.add_points(mesh.points, color='blue', point_size=5)
        if texture_img is not None:
            mesh.texture_map_to_plane(inplace=True)
            texture=pv.read_texture(texture_img)
            #self.plotter.add_mesh(mesh, texture=texture, show_edges=False, show_scalar_bar=False)
            self.apply_texture_with_scalars(mesh, scalars=scalars,texture=texture)
        else:
            #self.plotter.add_mesh(mesh,scalars=scalars,cmap='viridis',show_edges=False,show_scalar_bar=False)
            self.plotter.add_points(mesh.points,color='green',point_size=5)
        self.plotter.add_axes(interactive=True, xlabel='rho', ylabel='alpha', zlabel='h',line_width=5)
    
    def add_mesh_open(self,points, subplot,texture_img=None):
        scaler = StandardScaler()
        points = scaler.fit_transform(points)
        cloud = pv.PolyData(points)
        mesh = cloud.delaunay_2d()
        direction=(0,0,5)
        extrude_mesh=mesh.extrude(vector=direction,capping=False)
        new_scalars = np.interp(np.linspace(0, 1, num=extrude_mesh.n_points), 
                                np.linspace(0, 1, num=len(points[:, 2])), 
                                points[:, 2])
        
        c_pts=extrude_mesh.points
        self.plotter.subplot(*subplot)

        if texture_img is not None:
            extrude_mesh.texture_map_to_plane(inplace=True)
            texture=pv.read_texture(texture_img)
            self.apply_texture_with_scalars(extrude_mesh,scalars=new_scalars,texture=texture)
        else:
            self.plotter.add_points(c_pts,color='green',point_size=5)
            #self.plotter.add_mesh(extrude_mesh,scalars=new_scalars,cmap='viridis',show_edges=False,show_scalar_bar=False)
        self.plotter.add_axes(line_width=5, interactive=True)


        
    
    def apply_texture_with_scalars(self, mesh, scalars, texture):
        texture_image = texture.to_image()
        width, height, _ = texture_image.dimensions
        texture_array = texture_image.point_data.active_scalars.reshape((height,width,-1))

        normalized_scalars = (scalars - scalars.min()) / (scalars.max() - scalars.min())
        if mesh.active_texture_coordinates is None or len(mesh.active_texture_coordinates) == 0:
            mesh.texture_map_to_plane(inplace=True)
      
        texture_coordinates = mesh.active_texture_coordinates
        for i, (u, v) in enumerate(texture_coordinates):
            x = int(u * (width - 1))
            y = int(v * (height - 1))
            x = np.clip(x, 0, width - 1)
            y = np.clip(y, 0, height - 1)
            factor = normalized_scalars[i]
            texture_array[y, x] = texture_array[y, x] * factor
        texture_array = np.clip(texture_array, 0, 255).astype(np.uint8)
        modified_texture=pv.Texture(texture_array.reshape((height, width, -1)))
        self.plotter.add_mesh(mesh, texture=modified_texture, show_edges=False, show_scalar_bar=False)
    def __call__(self):
        self.plotter.show()



'''loads a mesh file and plot it'''
def load_and_plot_mesh(file_path):
    mesh = pv.read(file_path)
    z_values = mesh.points[:, 2]
    z_min, z_max = z_values.min(), z_values.max()
    z_normalized = (z_values - z_min) / (z_max - z_min)

    # Create custom colors mimicking human gut color
    base_color = np.array([1.0, 0.5, 0.5])  # Pinkish-red
    variation_strength = 0.1
    colors = np.tile(base_color, (mesh.n_points, 1))
    #colors += variation_strength * np.random.uniform(-1, 1, colors.shape)
    

    #introduce variation in color intensity based on normalized z val
    intensity_variation = np.random.uniform(0.8, 1.2, mesh.n_points).reshape(-1, 1)
    height_based_variation = (z_normalized * 0.5 + 0.5).reshape(-1, 1)
    colors *= intensity_variation * height_based_variation

    colors = np.clip(colors, 0, 1)

    mesh.point_data['colors'] = colors

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars='colors', rgb=True)
    plotter.set_background("white")
    plotter.show()

def visualize_and_save_mesh_from_points(points, filename, screenshot=None):
    """
    Creates, visualizes, and saves a mesh from a given set of points using PyVista.

    
    Parameters:
    - points: A NumPy array of shape (N, 3) containing the XYZ coordinates of the points.
    - filename: String, the path and file name to save the mesh. The format is inferred from the extension.
    - screenshot: Optional string, the path and file name to save a screenshot of the plot.
    """
    scaler=StandardScaler()
    points=scaler.fit_transform(points)
   
    cloud = pv.PolyData(points)

    mesh = cloud.delaunay_2d()
    mesh = mesh.smooth(n_iter=300)
    scalars = mesh.points[:, 2]
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars=scalars, cmap='viridis', show_edges=True, show_scalar_bar=False)
    #plotter.add_mesh(mesh, color="white", show_edges=False)
    #plotter.add_scalar_bar(title="Scene Deformation", label_font_size=10, title_font_size=10)
    plotter.show(screenshot=screenshot)

    cam_pos=plotter.camera_position
    print('Camera position:', cam_pos)
    mesh.save(filename)



##=============================================================================
##=============================================================================
## Function to Compute a & b constant params for B-spline from the given image
##=============================================================================
##=============================================================================

def compute_a_b_values(image_path):
   
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return None, None
    image_height, image_width = image.shape[:2]
    image_center = (image_width / 2, image_height / 2, 0)
    
    vanishing_pts = (0, 0, 10)
   
    a_values = np.zeros((image_height, image_width, 3), dtype=np.float32)
    b_values = np.zeros((image_height, image_width), dtype=np.float32)

    for row in range(image_height):
        for col in range(image_width):
            p_minus_vp = np.array([col, row, 0]) - np.array(vanishing_pts) 
            a_values[row, col] = p_minus_vp
            b_values[row, col] = np.arctan2(p_minus_vp[1], p_minus_vp[0])

   
    a_values = a_values / np.linalg.norm(a_values, axis=(0, 1))
    a_values=np.mean(a_values.ravel())
    b_values = b_values / np.linalg.norm(b_values)
    b_values=np.mean(b_values.ravel())

    return a_values, b_values


''' Objective function with texture info'''
def objective_function(params, points_3d, points_2d_observed, image, intrinsic_matrix, k, g_t, gamma, b_mesh_deformation, lambda_ortho, lambda_det, texture, pbar):
    rotation_matrix = params[:9].reshape(3, 3)
    translation_vector = params[9:12]
    control_points = params[12:-2].reshape(11, 11, 3)
    lambda_ortho = params[-2]
    lambda_det = params[-1]
    a = 0.00051301747 
    b = 0.0018595674

    deformed_points = points_3d.b_mesh_deformation(a, b, control_points)

    projector = Project3D_2D_cam(intrinsic_matrix, rotation_matrix, translation_vector)
    projected_2d_pts = projector.project_points(deformed_points)
    if projected_2d_pts.shape[0] > points_2d_observed.shape[0]:
        projected_2d_pts = projected_2d_pts[:points_2d_observed.shape[0], :]
    elif projected_2d_pts.shape[0] < points_2d_observed.shape[0]:
        points_2d_observed = points_2d_observed[:projected_2d_pts.shape[0], :]
    points_2d_observed = points_2d_observed.reshape(-1, 2)
    
    # Compute light intensity error
    light_intensity_error = []
    for pt2d, pt3d in zip(projected_2d_pts, deformed_points):
        x, y, z = pt3d
        L = calib_p_model(x, y, z, k, g_t, gamma)
        if 0 <= int(pt2d[0]) < image.shape[1] and 0 <= int(pt2d[1]) < image.shape[0]:
            pixel_intensity = get_pixel_intensity(image[int(pt2d[1]), int(pt2d[0])])
            light_intensity_error.append(abs(pixel_intensity - L))
        else:
            light_intensity_error.append(0)

    # Compute texture intensity error
    texture_intensity_error = []
    for pt2d, pt3d in zip(projected_2d_pts, deformed_points):
        if 0 <= int(pt2d[0]) < texture.width and 0 <= int(pt2d[1]) < texture.height:
            u = int(pt2d[0] * (texture.width - 1))
            v = int(pt2d[1] * (texture.height - 1))
            texture_intensity = texture.point_data.active_scalars[v, u]
            texture_intensity_error.append(abs(texture_intensity - L))
        else:
            texture_intensity_error.append(0)
    
    # Compute photometric error as the sum of light intensity error and texture intensity error
    photometric_error = np.sum(light_intensity_error) + np.sum(texture_intensity_error)

    ortho_constraint = np.dot(rotation_matrix, rotation_matrix.T) - np.eye(3)
    det_constraint = np.linalg.det(rotation_matrix) - 1

    objective = photometric_error
    objective += lambda_ortho * np.linalg.norm(ortho_constraint, 'fro')**2
    objective += lambda_det * det_constraint**2

    pbar.update(1)

    return objective