

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv
from utils import generate_uniform_grid_control_points, BMeshDense

rho_step_size = 0.1  # Step size for rho
alpha_step_size = np.pi / 4  # Step size for alpha
radius = 100
center = (0, 0)

#control_points = np.loadtxt('./logs/optimized_control_points_frame_0.txt')
control_points=generate_uniform_grid_control_points(rho_step_size, alpha_step_size, R=100)
control_points = control_points.reshape(20, 9, 3)

bmd = BMeshDense(radius=radius, center=center)
points = bmd.b_mesh_deformation(control_points=control_points, subsample_factor=5)

# Extract rho, alpha, h coordinates
rho = points[:, 0]
alpha = points[:, 1]
h = points[:, 2]

# Convert to Cartesian coordinates
x = rho*h * np.cos(alpha)
y = rho*h * np.sin(alpha)
z = h

# Plot in Cartesian Coordinates
fig_cartesian = plt.figure()
ax_cartesian = fig_cartesian.add_subplot(111, projection='3d')
ax_cartesian.scatter(x, y, z, c='r', marker='o')
ax_cartesian.set_title('Point Cloud in Cartesian Coordinates')
ax_cartesian.set_xlabel('X')
ax_cartesian.set_ylabel('Y')
ax_cartesian.set_zlabel('Z')
plt.show()

# Plot in Cylindrical Coordinates
fig_cylindrical = plt.figure()
ax_cylindrical = fig_cylindrical.add_subplot(111, projection='3d')
ax_cylindrical.scatter(rho, alpha, h, c='b', marker='o')
ax_cylindrical.set_title('Point Cloud in Cylindrical Coordinates')
ax_cylindrical.set_xlabel('Rho')
ax_cylindrical.set_ylabel('Alpha')
ax_cylindrical.set_zlabel('H')
plt.show()

# Convert to Polar Coordinates and plot (assuming polar coordinates mean 2D representation)
fig_polar = plt.figure()
ax_polar = fig_polar.add_subplot(111, projection='3d')
ax_polar.scatter(alpha, rho, c='g', marker='o')
ax_polar.set_title('Point Cloud in Polar Coordinates')
plt.show()

# # Optional: Visualization using PyVista for interactive exploration
# # Plot in Cartesian coordinates using PyVista
# point_cloud = pv.PolyData(np.column_stack((x, y, z)))
# plotter = pv.Plotter()
# plotter.add_mesh(point_cloud, color='red', point_size=5, render_points_as_spheres=True)
# plotter.show(title='Point Cloud in Cartesian Coordinates')

# # Plot in Cylindrical coordinates using PyVista
# cylindrical_points = np.column_stack((rho, alpha, h))
# cylindrical_point_cloud = pv.PolyData(cylindrical_points)
# plotter = pv.Plotter()
# plotter.add_mesh(cylindrical_point_cloud, color='blue', point_size=5, render_points_as_spheres=True)
# plotter.show(title='Point Cloud in Cylindrical Coordinates')