'''
-------------------------------------------------------------
This script generates control points for B-spline mesh deformation,
applies the deformation, and visualizes the resulting surface in 
Cartesian, polar coordinates, and H-surface representations.

Author: Mitterand Ekole
Date: 22-05-2024
-------------------------------------------------------------
'''

import numpy as np
from scipy.interpolate import SmoothBivariateSpline
import pyvista as pv
from sklearn.preprocessing import StandardScaler

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


class SingleWindowGridViz:
    def __init__(self):
        pass
    
    def visualize_and_save_mesh_with_camera(self, points, filename, screenshot=None, texture_img=None):
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
        mesh=mesh.smooth(n_iter=600)
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

M = 10
N = 5
# Generate a uniform grid of control points
rho_step_size = 5  # Step size for rho
alpha_step_size = 2*np.pi / 10 # Step size for alpha
height = 100
center = (0, 0)

cylinder_points = np.array([[rho, alpha, z] for rho in np.linspace(0, 10, M) for alpha in np.linspace(0, 2 * np.pi, N) for z in np.linspace(0, 100, M)])
h_constant = 10
control_points = generate_uniform_grid_control_points(rho_step_size, alpha_step_size, h_constant=None,h_variable_range=(50,100))
# control_points=np.loadtxt('/Users/ekole/Dev/gut_slam/pose_estim/logs/optimized_control_points_frame_0.txt')
# control_points=control_points.reshape(11,11,3)


bmd = BMeshDeformation(height, center, cylinder_points)
deformed_points = bmd.b_mesh_deformation(0.00051301747, 0.0018595674, control_points)
texture_img='./tex/colon_DIFF.png'

viz = GridViz(grid_shape=(1, 3))
#singleViz=SingleWindowGridViz()
#viz.add_mesh_cartesian(deformed_points, subplot=(0, 0),texture_img=texture_img)
viz.add_mesh_polar(deformed_points, subplot=(0, 0),texture_img=texture_img)
viz.add_h_surface(deformed_points, subplot=(0, 1),texture_img=texture_img)
viz.add_mesh_open(deformed_points, subplot=(0, 2),texture_img=texture_img)

#singleViz.visualize_and_save_mesh_with_camera(deformed_points,'./rendering/def_mesh.vtk',screenshot='./rendering/mesh.png',texture_img=texture_img)
viz()