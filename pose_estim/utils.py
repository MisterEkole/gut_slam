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
from scipy.spatial.transform import Rotation as R
import scipy.special
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from sklearn.preprocessing import StandardScaler
import csv

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

def euler_to_rotation_matrix(euler_angles):
    # Convert Euler angles to rotation matrix
    r = R.from_euler('xyz', euler_angles)
    return r.as_matrix()

##=============================================================================
##=============================================================================
## Generate Control Points with Uniform Grid
## Generate Cylinder Points
##=============================================================================
##=============================================================================

def polar_to_cartesian(rho, alpha,h):
    x = rho *h * np.cos(alpha)
    y = rho *h *np.sin(alpha)
    z=h
    
    return x, y,z

def polar_to_cartesian_with_vp(rho, alpha, vp_x, vp_y, z):
  x = rho * np.cos(alpha) + vp_x
  y = rho * np.sin(alpha) + vp_y
  return x, y, z

# Function to read frame index, translation vector, and Euler angles from CSV
def read_csv(csv_file):
    frame_data = {}
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            frame_idx = int(row['Frame'])
            translation = np.array([float(row['Location X']), float(row['Location Y']), float(row['Location Z'])])
            euler_angles = np.array([float(row['Rotation X']), float(row['Rotation Y']), float(row['Rotation Z'])])
            frame_data[frame_idx] = {'translation': translation, 'euler_angles': euler_angles}
    return frame_data

# Function to generate a uniform grid of control points
def generate_uniform_grid_control_points(rho_step_size, alpha_step_size, h_constant=None, R=None, rho_range=(0.1, 2.0), alpha_range=(0, 2 * np.pi)):
    
    rho_values = np.arange(rho_range[0], rho_range[1] + rho_step_size, rho_step_size)
    if rho_values[-1] > rho_range[1]:  
        rho_values = rho_values[:-1]

    alpha_values = np.arange(alpha_range[0], alpha_range[1] + alpha_step_size, alpha_step_size)
    if alpha_values[-1] > alpha_range[1]:  
        alpha_values = alpha_values[:-1]

    control_points = []

    for rho in rho_values:
        for alpha in alpha_values:
            if h_constant is not None:
                h = h_constant
            else:
                h = R / rho
           
          
            control_points.append((rho, alpha, h))

    return np.array(control_points).reshape(len(rho_values), len(alpha_values), 3)
##=============================================================================
##=============================================================================
## Updated B-Spline Surface Model
##=============================================================================
##=============================================================================


''' Sparse B-spline-mesh with interpolation of h'''
class BMesh:
    def __init__(self, radius, center):
        self.radius = radius
        self.center = center

    def b_mesh_deformation(self,control_points):
        M, N, _ = control_points.shape

        rho = control_points[:, :, 0].flatten()
        alpha = control_points[:, :, 1].flatten()
        cp_h = control_points[:, :, 2].flatten()

        spline_z = SmoothBivariateSpline(rho, alpha, cp_h, s=M * N)

        pts = []
        for i in range(M):
            for j in range(N):
                h = rho[i * N + j]
                theta = alpha[i * N + j]
                x = control_points[i, j, 0]
                y = control_points[i, j, 1]
                h = spline_z.ev(h, theta)
                pts.append([x, y, h])

        return np.array(pts)
 
''' Dense b-spline mesh with interpolated h'''
class BMeshDense:
    def __init__(self, radius, center):
        self.radius = radius
        self.center = center

    def b_mesh_deformation(self, control_points, subsample_factor=2):
        M, N, _ = control_points.shape

        rho = control_points[:, :, 0].flatten()
        alpha = control_points[:, :, 1].flatten()
        cp_h = control_points[:, :, 2].flatten()

        spline_z = SmoothBivariateSpline(rho, alpha, cp_h, s=M * N)

        pts = []
        for i in range(M - 1):
            for j in range(N - 1):
           
                rho1, rho2 = control_points[i, j, 0], control_points[i + 1, j, 0]
                alpha1, alpha2 = control_points[i, j, 1], control_points[i, j + 1, 1]

               
                for k in range(subsample_factor):
                    for l in range(subsample_factor):
                        frac_k = k / subsample_factor
                        frac_l = l / subsample_factor

                        new_rho = rho1 + frac_k * (rho2 - rho1)
                        new_alpha = alpha1 + frac_l * (alpha2 - alpha1)
                        new_h = spline_z.ev(new_rho, new_alpha)

                        pts.append([new_rho, new_alpha, new_h])


        return np.array(pts)



''' Dense b_mesh with bending, twisting and random disturbances deformations'''
class BMeshDefDense:
    def __init__(self, radius, center):
        self.radius = radius
        self.center = center

    def twist_deformation(self, points, twist_rate):
        """
        Apply a twisting deformation to the points.
        """
        twisted_points = np.copy(points)
        for point in twisted_points:
            rho, alpha, z = point
            twist_angle = twist_rate * z
            twisted_alpha = alpha + twist_angle
            point[1] = twisted_alpha
        return twisted_points

    def bend_deformation(self, points, bend_amplitude, bend_frequency):
        """
        Apply a bending deformation to the points. For each points a sine function is applied to radial coordinate
        bend_amplitude:determines max deviation from original pos
        bend_frequency: control how frequently the bending occurs
        """
        bent_points = np.copy(points)
        for point in bent_points:
            rho, alpha, z = point
            bend = bend_amplitude * np.sin(bend_frequency * z)
            bent_rho = rho + bend
            point[0] = bent_rho
        return bent_points
    def apply_random_disturbances(self, points, disturbance_amplitude):
        """
        Apply random disturbances to the height (z-coordinate) of the points.
        """
        disturbed_points = np.copy(points)
        disturbances = np.random.uniform(-disturbance_amplitude, disturbance_amplitude, size=disturbed_points.shape)
        disturbed_points+= disturbances
        return disturbed_points
    
    def apply_sinusoidal_disturbances(self, points, amplitude, frequency):
        disturbed_points = np.copy(points)
        for i in range(disturbed_points.shape[0]):
            disturbed_points[i, 0] += amplitude * np.sin(frequency * points[i, 0])  # Sinusoidal disturbance on rho
            disturbed_points[i, 1] += amplitude * np.sin(frequency * points[i, 1])  # Sinusoidal disturbance on alpha
            disturbed_points[i, 2] += amplitude * np.sin(frequency * points[i, 2])  # Sinusoidal disturbance on z
        return disturbed_points

    def b_mesh_deformation(self, control_points, subsample_factor=2, disturbance_amplitude=None, bend_amplitude=None,bend_frequency=None):
        M, N, _ = control_points.shape

        rho = control_points[:, :, 0].flatten()
        alpha = control_points[:, :, 1].flatten()
        cp_h = control_points[:, :, 2].flatten()

        spline_z = SmoothBivariateSpline(rho, alpha, cp_h, s=M * N)

        pts = []
        for i in range(M - 1):
            for j in range(N - 1):
                rho1, rho2 = control_points[i, j, 0], control_points[i + 1, j, 0]
                alpha1, alpha2 = control_points[i, j, 1], control_points[i, j + 1, 1]

                for k in range(subsample_factor):
                    for l in range(subsample_factor):
                        frac_k = k / subsample_factor
                        frac_l = l / subsample_factor

                        new_rho = rho1 + frac_k * (rho2 - rho1)
                        new_alpha = alpha1 + frac_l * (alpha2 - alpha1)
                        new_h = spline_z.ev(new_rho, new_alpha)

                        pts.append([new_rho, new_alpha, new_h])

        pts = np.array(pts)
        #pts=self.bend_deformation(pts, bend_amplitude, bend_frequency)
        pts=self.apply_random_disturbances(pts,disturbance_amplitude=4.0)
        #pts=self.apply_sinusoidal_disturbances(pts, amplitude=10.0, frequency=2.0)
        #pts=self.twist_deformation(pts,twist_rate=1.0)

        return pts

##=============================================================================
##=============================================================================
## Visualization Utils 
##=============================================================================
##=============================================================================
''' GridViz accepts points in polar coords before converting to appropriate coord system for visualisation,applies texture to mesh'''
class GridViz:
    def __init__(self, grid_shape, window_size=(2300, 1500)):
        self.plotter = pv.Plotter(shape=grid_shape, window_size=window_size)

    def add_mesh_cartesian(self, points, subplot, texture_img=None, cmap='YlOrRd',label=None): #pale flesh tone color
        rho = points[:, 0]
        alpha = points[:, 1]
        h = points[:, 2]
        x, y, z = polar_to_cartesian(rho, alpha, h)
        points = np.vstack((x, y, z)).T
        scaler = StandardScaler()
        points = scaler.fit_transform(points)
        # cloud = pv.PolyData(points)
        # mesh = cloud.delaunay_3d()
        nx = int(np.sqrt(points.shape[0]))
        ny = nx
        grid = pv.StructuredGrid()
        grid.points = points
        grid.dimensions = (nx, ny, 1)
        mesh=grid
        scalars = mesh.points[:, 2]
        self.plotter.subplot(*subplot)
        if texture_img is not None:
            self.apply_texture_with_geometry(mesh, texture_img, rho, alpha, h)
            #self.apply_color_texture_with_geometry(mesh,cmap)
        else:
            self.plotter.add_points(points, cmap='viridis', scalars=scalars, point_size=5, show_scalar_bar=False)
            #self.plotter.add_points(mesh.points, cmap='viridis',show_scalar_bar=False,scalars=scalars, point_size=5)
        self.plotter.add_axes(line_width=5, interactive=True)
        if label:
            self.plotter.add_text(label, position='upper_left')

    def add_mesh_polar(self, points, subplot, texture_img=None,cmap='YlOrRd',label=None):
        scaler = StandardScaler()
        points = scaler.fit_transform(points)
        cloud = pv.PolyData(points)
        mesh = cloud.delaunay_2d()
        mesh = mesh.smooth(n_iter=500)
        scalars = mesh.points[:, 2]
        self.plotter.subplot(*subplot)
        if texture_img is not None:
            rho, alpha, h = self.extract_polar_coordinates(points)
            self.apply_texture_with_geometry(mesh, texture_img, rho, alpha, h)
            #self.apply_color_texture_with_geometry(mesh,cmap)
        else:
            #self.plotter.add_points(mesh.points, cmap='viridis',show_scalar_bar=False,scalars=scalars, point_size=5)
            self.plotter.add_points(points, cmap='viridis',show_scalar_bar=False,scalars=scalars, point_size=5)
        self.plotter.add_axes(interactive=True, xlabel='rho', ylabel='alpha', zlabel='h', line_width=5)
        if label:
            self.plotter.add_text(label, position='upper_left')

    def add_mesh_cy(self, points, subplot, texture_img=None,cmap='YlOrRd',label=None):
        rho = points[:, 0]
        alpha = points[:, 1]
        h = points[:, 2]
        x = rho * np.cos(alpha)
        y = rho * np.sin(alpha)
        r = np.sqrt(x**2 + y**2)
        theta = alpha
        points = np.vstack((r, theta, h)).T
        scaler = StandardScaler()
        points = scaler.fit_transform(points)
        cloud = pv.PolyData(points)
        mesh = cloud.delaunay_2d()
        mesh = mesh.smooth(n_iter=500)
        scalars = mesh.points[:, 2]
        self.plotter.subplot(*subplot)
        if texture_img is not None:
            self.apply_texture_with_geometry(mesh, texture_img, rho, alpha, h)
            #self.apply_color_texture_with_geometry(mesh,cmap)
        else:
            self.plotter.add_points(mesh.points, cmap='viridis',show_scalar_bar=False,scalars=scalars, point_size=5)
            #self.plotter.add_points(points, cmap='viridis',show_scalar_bar=False,scalars=scalars, point_size=5)
        self.plotter.add_axes(interactive=True, xlabel='r', ylabel='theta', zlabel='h', line_width=5)
        if label:
            self.plotter.add_text(label, position='upper_left')

    def apply_texture_with_geometry(self, mesh, texture_img, rho, alpha, h):
        texture = pv.read_texture(texture_img)
        texture_image=texture.to_image()
        width,height=texture_image.dimensions[:2]
        texture_array = texture_image.point_data.active_scalars.reshape((height, width, -1))
        scalars=mesh.points[:,2]
        normalized_scalars = (scalars - scalars.min()) / (scalars.max() - scalars.min())
        
        # Normalize the texture coordinates to be between 0 and 1
        u = (alpha - alpha.min()) / (alpha.max() - alpha.min())
        v = (h - h.min()) / (h.max() - h.min())
        
        texture_coords = np.c_[u, v]
        mesh.active_texture_coordinates = texture_coords

        #adujust texture intensity based on normalized scalars
        for i, (u, v) in enumerate(texture_coords):
            x = int(u * (width - 1))
            y = int(v * (height - 1))
            x = np.clip(x, 0, width - 1)
            y = np.clip(y, 0, height - 1)
            factor = normalized_scalars[i]
            texture_array[y, x] = texture_array[y, x] * factor

        #overide the texture with the modified texture array
        modified_texture = pv.numpy_to_texture(texture_array)
        self.plotter.add_mesh(mesh, texture=modified_texture, show_edges=False, show_scalar_bar=False)

    def apply_color_texture_with_geometry(self, mesh, cmap):
        z = mesh.points[:, 2]
        z_range = z.max() - z.min()
        if z_range == 0:
            z_normalized = np.ones_like(z)  # If all z values are the same, set normalized z to 1
        else:
            z_normalized = (z - z.min()) / z_range
        
        # Get the colormap
        colormap = plt.get_cmap(cmap)
        colors = colormap(z_normalized)[:, :3]  # Get RGB values from colormap
        mesh.point_data['colors'] = colors
        self.plotter.add_mesh(mesh, scalars='colors', rgb=True, show_edges=False, show_scalar_bar=False)

    def extract_polar_coordinates(self, points):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        rho = np.sqrt(x**2 + y**2)
        alpha = np.arctan2(y, x)
        return rho, alpha, z
    
    def save_plot(self, filename):
        self.plotter.screenshot(filename)
    
    def show(self):
        self.plotter.show()





''' renders a mesh in cartesian, polar and cylindrical coordinates in a single window, saves camera info to file and mesh to file'''

class SingleWindowGridViz:
    def __init__(self):
        pass  # Removed plotter initialization from here

    def visualize_and_save_cartesian(self, points, filename, screenshot=None, texture_img=None, cmap='YlOrRd', wireframe=False):
        rho = points[:, 0]
        alpha = points[:, 1]
        h = points[:, 2]
        x, y, z = polar_to_cartesian(rho, alpha, h)
        points = np.vstack((x, y, z)).T
        points = self.standard_scale(points)
        mesh = self.create_structured_grid(points)
        camera_settings = self.visualize_mesh(mesh, filename, screenshot, texture_img, 'cartesian', wireframe)
        return camera_settings

    def visualize_and_save_polar(self, points, filename, screenshot=None, texture_img=None, cmap='YlOrRd', wireframe=False):
        points = self.standard_scale(points)
        mesh = self.create_delaunay_mesh(points)
        camera_settings = self.visualize_mesh(mesh, filename, screenshot, texture_img, 'polar', wireframe)
        return camera_settings

    def visualize_and_save_cylindrical(self, points, filename, screenshot=None, texture_img=None, cmap='YlOrRd', wireframe=False):
        rho = points[:, 0]
        alpha = points[:, 1]
        h = points[:, 2]
        x = rho * np.cos(alpha)
        y = rho * np.sin(alpha)
        r = np.sqrt(x**2 + y**2)
        theta = alpha
        points = np.vstack((r, theta, h)).T
        points = self.standard_scale(points)
        mesh = self.create_delaunay_mesh(points)
        camera_settings = self.visualize_mesh(mesh, filename, screenshot, texture_img, 'cylindrical', wireframe)
        return camera_settings

    def standard_scale(self, points):
        scaler = StandardScaler()
        return scaler.fit_transform(points)

    def create_structured_grid(self, points):
        nx = int(np.sqrt(points.shape[0]))
        ny = nx
        grid = pv.StructuredGrid()
        grid.points = points
        grid.dimensions = (nx, ny, 1)
        return grid

    def create_delaunay_mesh(self, points):
        cloud = pv.PolyData(points)
        mesh = cloud.delaunay_2d()
        return mesh.smooth(n_iter=500)

    def visualize_mesh(self, mesh, filename, screenshot, texture_img, cmap, coordinate_system, wireframe=False):
        plotter = pv.Plotter()  # Create a new plotter instance for each visualization
        scalars = mesh.points[:, 2]

        if texture_img:
            if coordinate_system == 'cartesian':
                rho, alpha, h = self.extract_polar_coordinates(mesh.points)
            elif coordinate_system == 'cylindrical':
                rho, alpha, h = self.extract_cylindrical_coordinates(mesh.points)
            else:
                rho, alpha, h = mesh.points[:, 0], mesh.points[:, 1], mesh.points[:, 2]
            self.apply_texture_with_geometry(mesh, texture_img, rho, alpha, h, plotter)
        #else:
            #self.apply_color_texture_with_scalars(mesh, scalars, cmap, plotter)

        if wireframe:
            plotter.add_mesh(mesh, show_edges=True, show_scalar_bar=False, style='wireframe')
        else:
            plotter.add_mesh(mesh, show_edges=False, show_scalar_bar=False)

        self.set_camera(plotter)
        plotter.show(screenshot=screenshot)
        mesh.save(filename)

        camera_settings = {
            "position": plotter.camera.position,
            "focal_point": plotter.camera.focal_point,
            "view_up": plotter.camera.view_up
        }

        plotter.close()  # Ensure the plotter is closed after showing the plot
        return camera_settings

    def apply_texture_with_geometry(self, mesh, texture_img, rho, alpha, h, plotter):
        texture = pv.read_texture(texture_img)
        texture_image = texture.to_image()
        width, height = texture_image.dimensions[:2]
        texture_array = texture_image.point_data.active_scalars.reshape((height, width, -1))
        scalars = mesh.points[:, 2]
        normalized_scalars = (scalars - scalars.min()) / (scalars.max() - scalars.min())

        if rho is not None and alpha is not None:
            u = (alpha - alpha.min()) / (alpha.max() - alpha.min())
            v = (h - h.min()) / (h.max() - h.min())
        else:
            u, v = mesh.active_texture_coordinates.T

        texture_coords = np.c_[u, v]
        mesh.active_texture_coordinates = texture_coords

        for i, (u, v) in enumerate(texture_coords):
            x = int(u * (width - 1))
            y = int(v * (height - 1))
            x = np.clip(x, 0, width - 1)
            y = np.clip(y, 0, height - 1)
            factor = normalized_scalars[i]
            texture_array[y, x] = texture_array[y, x] * factor

        modified_texture = pv.numpy_to_texture(texture_array)
        plotter.add_mesh(mesh, texture=modified_texture, show_edges=False, show_scalar_bar=False)

    def apply_color_texture_with_scalars(self, mesh, scalars, cmap, plotter):
        z_range = scalars.max() - scalars.min()
        if z_range == 0:
            z_normalized = np.ones_like(scalars)
        else:
            z_normalized = (scalars - scalars.min()) / z_range

        colormap = plt.get_cmap(cmap)
        colors = colormap(z_normalized)[:, :3]
        mesh.point_data['colors'] = colors
        plotter.add_mesh(mesh, scalars='colors', rgb=True, show_edges=False, show_scalar_bar=False)

    def extract_polar_coordinates(self, points):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        rho = np.sqrt(x ** 2 + y ** 2)
        alpha = np.arctan2(y, x)
        return rho, alpha, z
    
    def extract_cylindrical_coordinates(self, points):
        rho, alpha, h = points[:, 0], points[:, 1], points[:, 2]
        x = rho * np.cos(alpha)
        y = rho * np.sin(alpha)
        r = np.sqrt(x**2 + y**2)
        theta = alpha
        return r, theta, h

    def set_camera(self, plotter):
        camera_position = (10, 10, 10)
        focal_point = (0, 0, 0)
        view_up = (0, 0, 1)
        plotter.camera.position = camera_position
        plotter.camera.focal_point = focal_point
        plotter.camera.view_up = view_up

    def save_camera_info_to_file(self, camera_settings, filename):
        with open(filename, 'w') as file:
            file.write(f"Camera Position: {camera_settings['position']}\n")
            file.write(f"Focal Point: {camera_settings['focal_point']}\n")
            file.write(f"View Up: {camera_settings['view_up']}\n")



''' loads a vtk mesh file an visualises in appropriate coordinate system'''

class MeshPlotter:
    def __init__(self):
        self.plotter = pv.Plotter()

    def visualize_cartesian(self, mesh_file, texture_img=None, cmap='YlOrRd', screenshot=None, wireframe=False):
        mesh = pv.read(mesh_file)
        mesh.points = self.standard_scale(mesh.points)
        camera_settings = self.visualize_mesh(mesh, screenshot, texture_img, cmap, 'cartesian', wireframe)
        return camera_settings

    def visualize_polar(self, mesh_file, texture_img=None, cmap='YlOrRd', screenshot=None, wireframe=False):
        mesh = pv.read(mesh_file)
        mesh.points = self.standard_scale(mesh.points)
        camera_settings = self.visualize_mesh(mesh, screenshot, texture_img, cmap, 'polar', wireframe)
        return camera_settings

    def visualize_cylindrical(self, mesh_file, texture_img=None, cmap='YlOrRd', screenshot=None, wireframe=False):
        mesh = pv.read(mesh_file)
        rho = mesh.points[:, 0]
        alpha = mesh.points[:, 1]
        h = mesh.points[:, 2]
        x = rho * np.cos(alpha)
        y = rho * np.sin(alpha)
        mesh.points = np.vstack((x, y, h)).T
        mesh.points = self.standard_scale(mesh.points)
        camera_settings = self.visualize_mesh(mesh, screenshot, texture_img, cmap, 'cylindrical', wireframe)
        return camera_settings

    def standard_scale(self, points):
        scaler = StandardScaler()
        return scaler.fit_transform(points)

    def visualize_mesh(self, mesh, screenshot, texture_img, cmap, coordinate_system, wireframe=False):
        scalars = mesh.points[:, 2]

        if texture_img:
            if coordinate_system == 'cartesian':
                rho, alpha, h = self.extract_polar_coordinates(mesh.points)
            else:
                rho, alpha, h = None, None, None
            self.apply_texture_with_geometry(mesh, texture_img, rho, alpha, h)
        else:
            self.apply_color_texture_with_scalars(mesh, scalars, cmap)
            self.plotter.add_mesh(mesh, show_edges=False, show_scalar_bar=False, style='wireframe' if wireframe else 'surface')
        
        self.set_camera()
        self.plotter.show(screenshot=screenshot)

        camera_settings = {
            "position": self.plotter.camera.position,
            "focal_point": self.plotter.camera.focal_point,
            "view_up": self.plotter.camera.view_up
        }

        return camera_settings

    def apply_texture_with_geometry(self, mesh, texture_img, rho, alpha, h):
        texture = pv.read_texture(texture_img)
        texture_image = texture.to_image()
        width, height = texture_image.dimensions[:2]
        texture_array = texture_image.point_data.active_scalars.reshape((height, width, -1))
        scalars = mesh.points[:, 2]
        normalized_scalars = (scalars - scalars.min()) / (scalars.max() - scalars.min())

        if rho is not None and alpha is not None:
            u = (alpha - alpha.min()) / (alpha.max() - alpha.min())
            v = (h - h.min()) / (h.max() - h.min())
        else:
            u, v = mesh.active_texture_coordinates.T

        texture_coords = np.c_[u, v]
        mesh.active_texture_coordinates = texture_coords

        for I, (u, v) in enumerate(texture_coords):
            x = int(u * (width - 1))
            y = int(v * (height - 1))
            x = np.clip(x, 0, width - 1)
            y = np.clip(y, 0, height - 1)
            factor = normalized_scalars[I]
            texture_array[y, x] = texture_array[y, x] * factor

        modified_texture = pv.numpy_to_texture(texture_array)
        self.plotter.add_mesh(mesh, texture=modified_texture, show_edges=False, show_scalar_bar=False)

    def apply_color_texture_with_scalars(self, mesh, scalars, cmap):
        z_range = scalars.max() - scalars.min()
        if z_range == 0:
            z_normalized = np.ones_like(scalars)
        else:
            z_normalized = (scalars - scalars.min()) / z_range

        colormap = plt.get_cmap(cmap)
        colors = colormap(z_normalized)[:, :3]
        mesh.point_data['colors'] = colors
        self.plotter.add_mesh(mesh, scalars='colors', rgb=True, show_edges=False, show_scalar_bar=False)

    def extract_polar_coordinates(self, points):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        rho = np.sqrt(x ** 2 + y ** 2)
        alpha = np.arctan2(y, x)
        return rho, alpha, z

    def set_camera(self):
        camera_position = (10, 10, 10)
        focal_point = (0, 0, 0)
        view_up = (0, 0, 1)
        self.plotter.camera.position = camera_position
        self.plotter.camera.focal_point = focal_point
        self.plotter.camera.view_up = view_up

    def save_camera_info_to_file(self, camera_settings, filename):
        with open(filename, 'w') as file:
            file.write(f"Camera Position: {camera_settings['position']}\n")
            file.write(f"Focal Point: {camera_settings['focal_point']}\n")
            file.write(f"View Up: {camera_settings['view_up']}\n")



