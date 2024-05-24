'''
-------------------------------------------------------------
Utilities for Pose and Deformation Estimation
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
                    #print(np.max(a+b))
                
                    

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
## Visualization Utils 
##=============================================================================
##=============================================================================

class GridViz:
    def __init__(self, grid_shape, window_size=(1300, 800)):
        # Initialize the plotter with a specified grid shape
        self.plotter = pv.Plotter(shape=grid_shape,window_size=window_size)

    def add_mesh_cartesian(self, points, subplot):
        # Visualize a 3D mesh from points in Cartesian coordinates
        scaler = StandardScaler()
        points = scaler.fit_transform(points)
        cloud = pv.PolyData(points)

        mesh = cloud.delaunay_3d()
        scalars = mesh.points[:, 2]  
        self.plotter.subplot(*subplot)
        #self.plotter.add_mesh(mesh, scalars=scalars, cmap='viridis', show_edges=True, show_scalar_bar=False)
        #self.plotter.add_mesh(mesh, color='white', show_edges=False)
        self.plotter.add_points(mesh.points, color='green', point_size=5)
        self.plotter.add_axes(line_width=5, interactive=True)
        

    def add_mesh_polar(self, points, subplot):
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
    
        rho = np.sqrt(x**2 + y**2)
        alpha = np.arctan2(y, x)
        h = z
    
        points = np.vstack((rho, alpha, h)).T
        #polar_points=np.column_stack((rho,alpha))
        # Visualize a 3D mesh from points in Polar coordinates
        scaler = StandardScaler()
        points = scaler.fit_transform(points)
        #points = np.column_stack((polar_points, np.zeros(polar_points.shape[0])))
        cloud = pv.PolyData(points)

        mesh = cloud.delaunay_2d()
        mesh = mesh.smooth(n_iter=600)
        scalars = mesh.points[:, 2]
        self.plotter.subplot(*subplot)
        #self.plotter.add_mesh(mesh, scalars=scalars, cmap='viridis', show_edges=True, show_scalar_bar=False,style='wireframe')
        #self.plotter.add_mesh(mesh, color='red', show_edges=False, render_points_as_spheres=True)
        self.plotter.add_points(mesh.points, color='green', point_size=5)
        self.plotter.add_axes(interactive=True, xlabel='r', ylabel='theta', zlabel='h', line_width=5)

    def add_h_surface(self, points, subplot):
        # Visualize an H-Surface from points
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
    
        rho = np.sqrt(x**2 + y**2)
        alpha = np.arctan2(y, x)
        h = z
    
        points = np.vstack((rho, alpha, h)).T
        #points[:, 2] *= 1
        cloud = pv.PolyData(points)
        mesh = cloud.delaunay_2d()
        mesh = mesh.smooth(n_iter=600)
        scalars = mesh.points[:, 2]
        
        self.plotter.subplot(*subplot)
        #self.plotter.add_mesh(mesh, show_edges=False, cmap='viridis', scalars=scalars, show_scalar_bar=False)
        #self.plotter.add_mesh(mesh, color='white', show_edges=False)
        self.plotter.add_points(mesh.points, color='blue', point_size=5)
        self.plotter.add_axes(interactive=True, xlabel='rho', ylabel='alpha', zlabel='h',line_width=5)

    def add_3dmesh_open(self, points, subplot):
        """
        Creates and visualizes a 3D open-ended cylinder in Cartesian coordinates from a given set of points using PyVista.
        
        Parameters:
        - points: A NumPy array of shape (N, 3) containing the XYZ coordinates of the points.
        - subplot: A tuple specifying the subplot location (row, col).
        """
        # Standardize the points
        scaler = StandardScaler()
        points = scaler.fit_transform(points)
        cloud = pv.PolyData(points)
        surface=cloud.delaunay_2d()
        #surface = surface.smooth(n_iter=600)
        direction = (0, 0, 5)
        extrude_mesh = surface.extrude(vector=direction, capping=False)
        new_scalars = np.interp(np.linspace(0, 1, num=extrude_mesh.n_points), 
                                np.linspace(0, 1, num=len(points[:, 2])), 
                                points[:, 2])
        
        c_pts=extrude_mesh.points


        self.plotter.subplot(*subplot)
        #self.plotter.add_mesh(extrude_mesh, scalars=new_scalars, cmap='viridis', show_edges=False, show_scalar_bar=False)
        #self.plotter.add_mesh(extrude_mesh, color='white', show_edges=False)
        self.plotter.add_points(c_pts, color='red', point_size=5)
        self.plotter.add_axes(line_width=5, interactive=True)

    def __call__(self):
        # Display the plot when the instance is called
        self.plotter.show()

def visualize_3dmeshcart(points):
    """
    Creates and visualizes a  3D mesh in cartesian coord from a given set of points using PyVista.
    
    Parameters:
    - points: A NumPy array of shape (N, 3) containing the XYZ coordinates of the points.
    """
    # Create a PyVista point cloud object
    scaler=StandardScaler()
    points=scaler.fit_transform(points)
    cloud = pv.PolyData(points)
    
    mesh = cloud.delaunay_3d()
    scalars = mesh.points[:, 2]  # Use Z-coordinates for coloring
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars=scalars, cmap='viridis', show_edges=True, show_scalar_bar=False)
    plotter.add_axes()
    #plotter.add_points(points, color='red', point_size=5)
    plotter.show()

def visualize_3dmeshopen(points):
    """
    Creates and visualizes a 3D open-ended cylinder in Cartesian coordinates from a given set of points using PyVista.
    
    Parameters:
    - points: A NumPy array of shape (N, 3) containing the XYZ coordinates of the points.
    """
    # Standardize the points
    scaler = StandardScaler()
    points = scaler.fit_transform(points)
    cloud=pv.PolyData(points)
    surface=cloud.delaunay_2d()
    surface=surface.smooth(n_iter=600)
    direction=(0,0,20)
    heights=points[:,2].ptp()
    extrude_mesh=surface.extrude(vector=direction, capping=False)
    new_scalars = np.interp(np.linspace(0, 1, num=extrude_mesh.n_points), 
                        np.linspace(0, 1, num=len(points[:, 2])), 
                        points[:, 2])


    plotter = pv.Plotter()
    plotter.add_mesh(extrude_mesh, scalars=new_scalars, cmap='viridis',show_edges=True, show_scalar_bar=False,style='wireframe')
    plotter.add_axes()
    plotter.show()

def visualize_3dmeshpol(points):
    """
    Creates and visualizes a  3D mesh in Polar coord from a given set of points using PyVista.
    
    Parameters:
    - points: A NumPy array of shape (N, 3) containing the XYZ coordinates of the points.
    """
    # Create a PyVista point cloud object
    scaler=StandardScaler()
    points=scaler.fit_transform(points)
    cloud = pv.PolyData(points)
    mesh=cloud.delaunay_2d()
    mesh=mesh.smooth(n_iter=600)
    scalars = mesh.points[:, 2]  # Use Z-coordinates for coloring
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars=scalars, cmap='viridis', show_edges=True, show_scalar_bar=False)
    #plotter.add_points(mesh.points, color='blue', point_size=5)
    plotter.add_axes(interactive=True,xlabel='r', ylabel='theta', zlabel='h')
    
    #plotter.add_points(points, color='red', point_size=5)
    plotter.show()

def visualize_h_surface(points):
    """
    Visualizes the points as an H-Surface(heighted surface) using PyVista.
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    rho = np.sqrt(x**2 + y**2)
    alpha = np.arctan2(y, x)
    h = z
    
    points = np.vstack((rho, alpha, h)).T
    #points[:, 2] *= 3
    scaler=StandardScaler()
    points=scaler.fit_transform(points)
    cloud = pv.PolyData(points)
    mesh = cloud.delaunay_2d()
    mesh=mesh.smooth(n_iter=600)
    scalars = mesh.points[:, 2]

    plotter = pv.Plotter()
    #plotter.add_mesh(mesh, show_edges=True, cmap='viridis', scalars=scalars,show_scalar_bar=False)
    plotter.add_points(mesh.points, color='blue', point_size=5) 
    plotter.add_axes(interactive=True,xlabel='rho', ylabel='alpha', zlabel='h')
    plotter.show()


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

def visualize_and_save_mesh_with_camera(points, filename, screenshot=None):
    """
    Creates, visualizes, and saves a mesh from a given set of points using PyVista and captures the camera settings.
    
    Parameters:
    - points: A NumPy array of shape (N, 3) containing the XYZ coordinates of the points.
    - filename: String, the path and file name to save the mesh.
    - screenshot: Optional string, the path and file name to save a screenshot of the plot.

    Returns:
    - camera_settings: Dictionary containing the camera's position, focal point, and view up vector.
    """
    cloud = pv.PolyData(points)
    mesh = cloud.delaunay_2d()
    mesh.smooth(n_iter=600)
    scalars = mesh.points[:, 2]
    
    plotter = pv.Plotter()

    #plotter.add_mesh(mesh, color="white", show_edges=True)
    plotter.add_mesh(mesh, scalars=scalars, cmap='viridis', show_edges=False, show_scalar_bar=False)

    # Set the camera position, focal point, and view up vector manually
    camera_position = (10, 10, 10)  
    focal_point = (0, 0, 0) 
    view_up = (0, 0, 1)  
    plotter.camera.position = camera_position
    plotter.camera.focal_point = focal_point
    plotter.camera.view_up = view_up
    
    plotter.show(screenshot=screenshot)
    mesh.save(filename)

    # Retrieve the camera settings to ensure consistent usage
    camera_settings = {
        "position": plotter.camera.position,
        "focal_point": plotter.camera.focal_point,
        "view_up": plotter.camera.view_up
    }

    return camera_settings




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
 # def apply_texture_with_scalars(self, mesh, scalars, texture):
    #     texture_image = texture.to_image()
    #     width, height, _ = texture_image.dimensions
    #     texture_array = texture_image.point_data.active_scalars.reshape((height, width, -1))

        
    #     normalized_scalars = (scalars - scalars.min()) / (scalars.max() - scalars.min())

    #     # Adjust brightness of the texture based on the normalized scalars
    #     for i, scalar in enumerate(normalized_scalars):
    #         x, y = i % width, i // width
    #         texture_array[y, x] = texture_array[y, x] * scalar
    #     texture_array=np.clip(texture_array,0,255).astype(np.uint8)

    #     # Update the texture with the modified data
    #     new_texture = pv.Texture(texture_array)
    #     self.plotter.add_mesh(mesh, texture=new_texture, show_edges=False, show_scalar_bar=False)
    # Function to generate a uniform grid of control points
# def polar_to_cartesian(rho, alpha, z):
#     x = rho * np.cos(alpha)
#     y = rho * np.sin(alpha)
#     return x, y, z


# def generate_uniform_grid_control_points(rho_step_size, alpha_step_size,z_step_size, h_constant=None, h_variable_range=None, rho_range=(0, 50), alpha_range=(0, 2 * np.pi)):
#     rho_values = np.arange(rho_range[0], rho_range[1] + rho_step_size, rho_step_size)
#     alpha_values = np.arange(alpha_range[0], alpha_range[1] + alpha_step_size, alpha_step_size)
#     z_values=np.arange(0,h_constant+z_step_size,z_step_size)
    
#     control_points = []
#     for z in z_values:
#         for rho in rho_values:
#             for alpha in alpha_values:
#                 if h_constant is not None:
#                     h = h_constant
#                 else:
#                     h = np.random.uniform(*h_variable_range)
#                 x, y, z = polar_to_cartesian(rho, alpha, h)
#                 control_points.append((x, y, z))

#     #return np.array(control_points).reshape(len(rho_values), len(alpha_values), 3)
#     return np.array(control_points).reshape(len(z_values)*len(rho_values), len(alpha_values), 3)