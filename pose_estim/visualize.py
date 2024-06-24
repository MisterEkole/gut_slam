import numpy as np
import pyvista as pv
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from utils import generate_uniform_grid_control_points, BMeshDense, BMeshDefDense

def polar_to_cartesian(rho, alpha, h):
    x = rho*h * np.cos(alpha)
    y = rho*h * np.sin(alpha)
    z = h
    return x, y, z

class DrawPlot:
    def __init__(self, window_size=(500, 500)):
        self.window_size = window_size
        self.plotters = []

    def create_plotter(self, off_screen=False):
        plotter = pv.Plotter(window_size=self.window_size, off_screen=off_screen)
        self.plotters.append(plotter)
        return plotter

    def add_mesh_cartesian(self, points, texture_img=None, cmap='YlOrRd', label=None, wireframe=False, off_screen=False): 
        plotter = self.create_plotter(off_screen=off_screen)
        rho = points[:, 0]
        alpha = points[:, 1]
        h = points[:, 2]
        x, y, z = polar_to_cartesian(rho, alpha, h)
        points = np.vstack((x, y, z)).T
        scaler = StandardScaler()
        points = scaler.fit_transform(points)
        nx = int(np.sqrt(points.shape[0]))
        ny = nx
        grid = pv.StructuredGrid()
        grid.points = points
        grid.dimensions = (nx, ny, 1)
        mesh = grid
        scalars = mesh.points[:, 2]

        if texture_img is not None:
            self.apply_texture_with_geometry(mesh, texture_img, rho, alpha, h)
        else:
            if wireframe:
                plotter.add_mesh(mesh, style='wireframe')
            else:
                plotter.add_points(points, cmap='viridis', scalars=scalars, point_size=8, show_scalar_bar=False, render_points_as_spheres=True)

        plotter.add_axes(line_width=5, interactive=True)
        if label:
            plotter.add_text(label, position='upper_left')

    def add_mesh_polar(self, points, texture_img=None, cmap='YlOrRd', label=None, wireframe=False, off_screen=False):
        plotter = self.create_plotter(off_screen=off_screen)
        scaler = StandardScaler()
        points = scaler.fit_transform(points)
        cloud = pv.PolyData(points)
        mesh = cloud.delaunay_2d()
        mesh = mesh.smooth(n_iter=500)
        scalars = mesh.points[:, 2]

        if texture_img is not None:
            rho, alpha, h = self.extract_polar_coordinates(points)
            self.apply_texture_with_geometry(mesh, texture_img, rho, alpha, h)
        else:
            if wireframe:
                plotter.add_mesh(mesh, style='wireframe')
            else:
                plotter.add_points(mesh.points, cmap='viridis', show_scalar_bar=False, scalars=scalars, point_size=8, render_points_as_spheres=True)

        plotter.add_axes(interactive=True, xlabel='rho', ylabel='alpha', zlabel='h', line_width=5)
        if label:
            plotter.add_text(label, position='upper_left')

    def add_mesh_cy(self, points, texture_img=None, cmap='YlOrRd', label=None, wireframe=False, off_screen=False):
        plotter = self.create_plotter(off_screen=off_screen)
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

        if texture_img is not None:
            self.apply_texture_with_geometry(mesh, texture_img, rho, alpha, h)
        else:
            if wireframe:
                plotter.add_mesh(mesh, style='wireframe')
            else:
                plotter.add_points(mesh.points, cmap='viridis', show_scalar_bar=False, scalars=scalars, point_size=8,render_points_as_spheres=True)

        plotter.add_axes(interactive=True, xlabel='r', ylabel='theta', zlabel='h', line_width=5)
        if label:
            plotter.add_text(label, position='upper_left')

    def apply_texture_with_geometry(self, mesh, texture_img, rho, alpha, h):
        texture = pv.read_texture(texture_img)
        texture_image = texture.to_image()
        width, height = texture_image.dimensions[:2]
        texture_array = texture_image.point_data.active_scalars.reshape((height, width, -1))
        scalars = mesh.points[:, 2]
        normalized_scalars = (scalars - scalars.min()) / (scalars.max() - scalars.min())

        u = (alpha - alpha.min()) / (alpha.max() - alpha.min())
        v = (h - h.min()) / (h.max() - h.min())

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
        plotter = self.plotters[-1]
        plotter.add_mesh(mesh, texture=modified_texture, show_edges=False, show_scalar_bar=False)

    def plot_control_points(self, control_points, label=None, off_screen=False):
        plotter = self.create_plotter(off_screen=off_screen)
        rho = control_points[:, :, 0].flatten()
        alpha = control_points[:, :, 1].flatten()
        h = control_points[:, :, 2].flatten()
        
        # Points in rho-alpha-h space
        points = np.vstack((rho, alpha, h)).T

        plotter.add_points(points, color='green', point_size=8, render_points_as_spheres=True)
        plotter.add_axes(interactive=True, xlabel='rho', ylabel='alpha', zlabel='h', line_width=5)
        if label:
            plotter.add_text(label, position='upper_left')

    def extract_polar_coordinates(self, points):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        rho = np.sqrt(x**2 + y**2)
        alpha = np.arctan2(y, x)
        return rho, alpha, z

    def save_plot(self, filename):
        plotter = self.plotters[-1]
        plotter.screenshot(filename)

    def show(self):
        for plotter in self.plotters:
            plotter.show()


rho_step_size = 0.1  # Step size for rho
alpha_step_size = np.pi / 4  # Step size for alpha
radius = 100
center = (0, 0)

# Generate control points
#control_points = generate_uniform_grid_control_points(rho_step_size, alpha_step_size, h_constant=10)
control_points=generate_uniform_grid_control_points(rho_step_size, alpha_step_size, R=100)

# Initialize the DrawPlot instance
plotter = DrawPlot(window_size=(2500, 1500))

# # Step 1: Plot and save control points
# plotter.plot_control_points(control_points, label=None, off_screen=True)
# plotter.save_plot(filename='./plots/control_points2.png')

# Step 2: Apply B-spline mesh deformation
bmd = BMeshDense(radius=radius, center=center)
deformed_points = bmd.b_mesh_deformation(control_points=control_points, subsample_factor=30)
texture_img = './tex/colon_DIFF.png'

# Plot and save the deformed mesh
# plotter.add_mesh_cartesian(deformed_points, label=None, wireframe=True ,off_screen=True)
# plotter.save_plot(filename='./plots/deformed_mesh3.png')

# Step 3: Visualize in different coordinate systems
# Cartesian
plotter.add_mesh_cartesian(deformed_points, label=None, wireframe=False, off_screen=True,texture_img=texture_img)
plotter.save_plot(filename='./plots/cartesian_mesh4.png')

# Polar
plotter.add_mesh_polar(deformed_points, label=None, wireframe=False, off_screen=True,texture_img=texture_img)
plotter.save_plot(filename='./plots/polar_mesh4.png')

# Cylindrical
plotter.add_mesh_cy(deformed_points, label=None, wireframe=False, off_screen=True,texture_img=texture_img)
plotter.save_plot(filename='./plots/cylindrical_mesh4.png')