"""
-------------------------------------------------------------
This script generates control points for B-spline mesh deformation,
applies the deformation, and visualizes the resulting surface in 
Cartesian, polar coordinates, and H-surface representations.
(No interpolation of cylinder points)

Author: Mitterand Ekole
Date: 22-05-2024
-------------------------------------------------------------
"""

import numpy as np
from scipy.interpolate import SmoothBivariateSpline
import pyvista as pv
from sklearn.preprocessing import StandardScaler
import open3d as o3d

def polar_to_cartesian(rho, alpha,h):
    x = rho *h * np.cos(alpha)
    y = rho *h *np.sin(alpha)
    z=h
    
    return x, y,z

def polar_to_cartesian_with_vp(rho, alpha, vp_x, vp_y, z):
  x = rho * np.cos(alpha) + vp_x
  y = rho * np.sin(alpha) + vp_y
  return x, y, z

# def generate_uniform_grid_control_points(rho_step_size, alpha_step_size, h_constant=None, R=None, rho_range=(2.0/4.0, 4.0), alpha_range=(0, 2 * np.pi)):
#     rho_values = rho_range[0]+np.arange(0, (rho_step_size+1))* (rho_range[1]-rho_range[0])/rho_step_size
#     alpha_values = np.arange(alpha_range[0], alpha_range[1] + alpha_step_size, alpha_step_size)

#     control_points = []
    
  
    
#     for rho in rho_values:
#         for alpha in alpha_values:
#             if h_constant is not None:
#                 h = h_constant
#             else:
#                 # %h_start, h_end = h_variable_range
#                 # %h_values = np.arange(h_start, h_end + h_step_size, h_step_size)
#                 # %h = h_values[len(control_points) % len(h_values)]
#                 h = R/(rho)
          
#             control_points.append((rho, alpha, h))


#     return np.array(control_points).reshape(len(rho_values), len(alpha_values), 3)

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


# class GridViz:
#     def __init__(self, grid_shape, window_size=(2300, 1500)):
#         self.plotter = pv.Plotter(shape=grid_shape, window_size=window_size)

#     def add_mesh_cartesian(self, points, subplot, texture_img=None):
#         rho = points[:, 0]
#         alpha = points[:, 1]
#         h = points[:, 2]
#         x, y, z = polar_to_cartesian(rho, alpha, h)
#         #x,y,z=polar_to_cartesian_with_vp(rho=rho,alpha=alpha,z=h,vp_x=10,vp_y=10)
#         points = np.vstack((x, y, z)).T
#         scaler = StandardScaler()
#         points = scaler.fit_transform(points)
#         # cloud = pv.PolyData(points)
#         # mesh = cloud.delaunay_2d()
#         nx = int(np.sqrt(points.shape[0]))
#         ny = nx
#         grid = pv.StructuredGrid()
#         grid.points = points
#         grid.dimensions = (nx, ny, 1)
#         mesh=grid
#         #mesh = mesh.smooth(n_iter=600)
#         scalars = mesh.points[:, 2]
#         self.plotter.subplot(*subplot)
#         if texture_img is not None:
#             mesh.texture_map_to_plane(inplace=True)
#             texture = pv.read_texture(texture_img)
#             #self.apply_texture_with_scalars(mesh, scalars=scalars, texture=texture)
#             self.apply_texture_with_geometry(mesh, texture=texture)
#         else:
#             self.plotter.add_points(points, cmap='viridis', scalars=scalars, point_size=5, show_scalar_bar=False)
#         self.plotter.add_axes(line_width=5, interactive=True)

#         "Polar Coord Plots"

#     def add_mesh_polar(self, points, subplot, texture_img=None):
#         scaler = StandardScaler()
#         points = scaler.fit_transform(points)
#         cloud = pv.PolyData(points)
#         mesh = cloud.delaunay_2d()
#         mesh = mesh.smooth(n_iter=600)
#         scalars = mesh.points[:, 2]
#         self.plotter.subplot(*subplot)
#         if texture_img is not None:
#             mesh.texture_map_to_plane(inplace=True)
#             texture = pv.read_texture(texture_img)
#             self.apply_texture_with_geometry(mesh, texture=texture)
#         else:
#             self.plotter.add_points(points, color='green', point_size=5)
#         self.plotter.add_axes(interactive=True, xlabel='rho', ylabel='alpha', zlabel='h', line_width=5)

#         ''' Cylindrical Coord Plots'''
#     def add_mesh_cy(self, points, subplot, texture_img=None):
#         rho = points[:, 0]
#         alpha = points[:, 1]
#         h = points[:, 2]
#         x=rho*np.cos(alpha)
#         y=rho*np.sin(alpha)
#         r=np.sqrt(x**2+y**2)
#         theta=alpha
#         points = np.vstack((r, theta, h)).T
#         scaler = StandardScaler()
#         points = scaler.fit_transform(points)
#         cloud = pv.PolyData(points)
#         mesh = cloud.delaunay_2d()
#         mesh = mesh.smooth(n_iter=600)
#         scalars = mesh.points[:, 2]
#         self.plotter.subplot(*subplot)
#         if texture_img is not None:
#             mesh.texture_map_to_plane(inplace=True)
#             texture = pv.read_texture(texture_img)
#             #self.apply_texture_with_scalars(mesh, scalars=scalars, texture=texture)
#             self.apply_texture_with_geometry(mesh, texture=texture)
#         else:
#             self.plotter.add_points(points, color='green', point_size=5)
#         self.plotter.add_axes(interactive=True, xlabel='r', ylabel='theta', zlabel='h', line_width=5)

#     def apply_texture_with_scalars(self, mesh, scalars, texture):
#         texture_image = texture.to_image()
#         width, height, _ = texture_image.dimensions
#         texture_array = texture_image.point_data.active_scalars.reshape((height, width, -1))
#         normalized_scalars = (scalars - scalars.min()) / (scalars.max() - scalars.min())
#         if mesh.active_texture_coordinates is None or len(mesh.active_texture_coordinates) == 0:
#             mesh.texture_map_to_plane(inplace=True)
#         texture_coordinates = mesh.active_texture_coordinates
#         for i, (u, v) in enumerate(texture_coordinates):
#             x = int(u * (width - 1))
#             y = int(v * (height - 1))
#             x = np.clip(x, 0, width - 1)
#             y = np.clip(y, 0, height - 1)
#             factor = normalized_scalars[i]
#             texture_array[y, x] = texture_array[y, x] * factor
#         texture_array = np.clip(texture_array, 0, 255).astype(np.uint8)
#         modified_texture = pv.Texture(texture_array.reshape((height, width, -1)))
#         self.plotter.add_mesh(mesh, texture=modified_texture, show_edges=False, show_scalar_bar=False)

#     def apply_texture_with_geometry(self, mesh, texture):
#         if mesh.active_texture_coordinates is None or len(mesh.active_texture_coordinates) == 0:
#             mesh.texture_map_to_plane(inplace=True)
#         self.plotter.add_mesh(mesh, texture=texture, show_edges=False, show_scalar_bar=False)


#     def __call__(self):
#         self.plotter.show()
class GridViz:
    def __init__(self, grid_shape, window_size=(2300, 1500)):
        self.plotter = pv.Plotter(shape=grid_shape, window_size=window_size)

    def add_mesh_cartesian(self, points, subplot, texture_img=None):
        rho = points[:, 0]
        alpha = points[:, 1]
        h = points[:, 2]
        x, y, z = polar_to_cartesian(rho, alpha, h)
        points = np.vstack((x, y, z)).T
        scaler = StandardScaler()
        points = scaler.fit_transform(points)
        cloud = pv.PolyData(points)
        mesh = cloud.delaunay_3d()
        # nx = int(np.sqrt(points.shape[0]))
        # ny = nx
        # grid = pv.StructuredGrid()
        # grid.points = points
        # grid.dimensions = (nx, ny, 1)
        # mesh = grid
        scalars = mesh.points[:, 2]
        self.plotter.subplot(*subplot)
        if texture_img is not None:
            self.apply_texture_with_geometry(mesh, texture_img, rho, alpha, h)
        else:
            self.plotter.add_points(points, cmap='viridis', scalars=scalars, point_size=5, show_scalar_bar=False)
        self.plotter.add_axes(line_width=5, interactive=True)

    def add_mesh_polar(self, points, subplot, texture_img=None):
        scaler = StandardScaler()
        points = scaler.fit_transform(points)
        cloud = pv.PolyData(points)
        mesh = cloud.delaunay_2d()
        mesh = mesh.smooth(n_iter=600)
        scalars = mesh.points[:, 2]
        self.plotter.subplot(*subplot)
        if texture_img is not None:
            rho, alpha, h = self.extract_polar_coordinates(points)
            self.apply_texture_with_geometry(mesh, texture_img, rho, alpha, h)
        else:
            self.plotter.add_points(points, color='green', point_size=5)
        self.plotter.add_axes(interactive=True, xlabel='rho', ylabel='alpha', zlabel='h', line_width=5)

    def add_mesh_cy(self, points, subplot, texture_img=None):
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
        mesh = mesh.smooth(n_iter=600)
        scalars = mesh.points[:, 2]
        self.plotter.subplot(*subplot)
        if texture_img is not None:
            self.apply_texture_with_geometry(mesh, texture_img, rho, alpha, h)
        else:
            self.plotter.add_points(points, color='green', point_size=5)
        self.plotter.add_axes(interactive=True, xlabel='r', ylabel='theta', zlabel='h', line_width=5)

    def apply_texture_with_geometry(self, mesh, texture_img, rho, alpha, h):
        texture = pv.read_texture(texture_img)
        
        # Normalize the texture coordinates to be between 0 and 1
        u = (alpha - alpha.min()) / (alpha.max() - alpha.min())
        v = (h - h.min()) / (h.max() - h.min())
        
        texture_coords = np.c_[u, v]

        # Apply the texture coordinates to the mesh
        mesh.active_texture_coordinates = texture_coords
        
        # Add the mesh with the texture to the plotter
        self.plotter.add_mesh(mesh, texture=texture, show_edges=False, show_scalar_bar=False)

    def extract_polar_coordinates(self, points):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        rho = np.sqrt(x**2 + y**2)
        alpha = np.arctan2(y, x)
        return rho, alpha, z

    def __call__(self):
        self.plotter.show()
class SingleWindowGridViz:
    def __init__(self):
        pass
    
    def visualize_and_save_mesh_with_camera(self, points, filename, screenshot=None, texture_img=None):
        rho = points[:, 0]
        alpha = points[:, 1]
        h = points[:, 2]
        x,y,z=polar_to_cartesian(rho,alpha,h)
        points = np.vstack((x, y, z)).T
    
        scaler = StandardScaler()
        points = scaler.fit_transform(points)
        cloud = pv.PolyData(points)
        mesh = cloud.delaunay_2d()
      
        mesh = mesh.smooth(n_iter=600)
        scalars = mesh.points[:, 2]
        
        plotter = pv.Plotter()
        if texture_img is not None:
            mesh.texture_map_to_plane(inplace=True)
            texture = pv.read_texture(texture_img)
            self.apply_texture_with_scalars(plotter, mesh, scalars, texture)
        else:
            plotter.add_mesh(mesh, scalars=scalars, cmap='viridis', show_edges=False, show_scalar_bar=False)
        
        camera_position = (10, 10, 10)  
        focal_point = (0, 0, 0) 
        view_up = (0, 0, 1)  
        plotter.camera.position = camera_position
        plotter.camera.focal_point = focal_point
        plotter.camera.view_up = view_up
        
        plotter.show(screenshot=screenshot)
        mesh.save(filename)

        camera_settings = {
            "position": plotter.camera.position,
            "focal_point": plotter.camera.focal_point,
            "view_up": plotter.camera.view_up
        }

        return camera_settings
    
    def apply_texture_with_scalars(self, plotter, mesh, scalars, texture):
        texture_image = texture.to_image()
        width, height, _ = texture_image.dimensions
        texture_array = texture_image.point_data.active_scalars.reshape((height, width, -1))

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
        modified_texture = pv.Texture(texture_array.reshape((height, width, -1)))
        plotter.add_mesh(mesh, texture=modified_texture, show_edges=False, show_scalar_bar=False)

# B-Spline Surface Generation
    
''' B-mesh with interpolation of h '''
class BMeshDef:
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
 
''' dense mesh with interpolated h'''
class BMeshDefDense:
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

# class BMeshDefDenseNoise:
#     def __init__(self, radius, center, noise_scale=0.05, frequency=1.0):
#         self.radius = radius
#         self.center = center
#         self.noise_scale = noise_scale  # Scale of the random noise
#         self.frequency = frequency  # Frequency of the sinusoidal noise

#     def sinusoidal_noise(self, rho, alpha, h):
#         # Generate sinusoidal noise based on rho, alpha, and h
#         noise_rho = self.noise_scale * np.sin(self.frequency * rho)
#         noise_alpha = self.noise_scale * np.cos(self.frequency * alpha)
#         noise_h = self.noise_scale * np.sin(self.frequency * h)
#         return noise_h

#     def b_mesh_deformation(self, control_points, subsample_factor=2):
#         M, N, _ = control_points.shape

#         rho = control_points[:, :, 0].flatten()
#         alpha = control_points[:, :, 1].flatten()
#         cp_h = control_points[:, :, 2].flatten()

#         spline_z = SmoothBivariateSpline(rho, alpha, cp_h, s=M * N)

#         pts = []
#         for i in range(M - 1):
#             for j in range(N - 1):
#                 rho1, rho2 = control_points[i, j, 0], control_points[i + 1, j, 0]
#                 alpha1, alpha2 = control_points[i, j, 1], control_points[i, j + 1, 1]

#                 for k in range(subsample_factor):
#                     for l in range(subsample_factor):
#                         frac_k = k / subsample_factor
#                         frac_l = l / subsample_factor

#                         new_rho = rho1 + frac_k * (rho2 - rho1)
#                         new_alpha = alpha1 + frac_l * (alpha2 - alpha1)
#                         new_h = spline_z.ev(new_rho, new_alpha)

#                         # Apply sinusoidal noise
#                         noise_h = self.sinusoidal_noise(new_rho, new_alpha, new_h)
#                         # new_rho += noise_rho
#                         # new_alpha += noise_alpha
#                         new_h += noise_h

#                         pts.append([new_rho, new_alpha, new_h])

#         return np.array(pts)

rho_step_size = 0.1  # Step size for rho
alpha_step_size = np.pi/6  # Step size for alpha
radius = 100
center = (0, 0)

control_points = generate_uniform_grid_control_points(rho_step_size, alpha_step_size,R=100)
#control_points = generate_uniform_grid_control_points(rho_step_size, alpha_step_size, h_constant=100)


bmd = BMeshDefDense(radius=radius, center=center)
deformed_points = bmd.b_mesh_deformation(control_points=control_points,subsample_factor=5)
texture_img = './tex/colon_DIFF.png'

viz = GridViz(grid_shape=(2, 3))

viz.add_mesh_polar(deformed_points, subplot=(0, 0),texture_img=texture_img)
viz.add_mesh_cy(deformed_points, subplot=(0, 1),texture_img=texture_img)
viz.add_mesh_cartesian(deformed_points,subplot=(0,2),texture_img=texture_img)
viz.add_mesh_polar(deformed_points, subplot=(1, 0))
viz.add_mesh_cy(deformed_points, subplot=(1, 1))
viz.add_mesh_cartesian(deformed_points,subplot=(1,2))


viz()
