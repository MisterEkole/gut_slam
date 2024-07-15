import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import BMeshDense, BMeshDefDense

def read_vtk_mesh(file_path):
    import pyvista as pv  
    mesh = pv.read(file_path)
    return mesh

def vtk_to_point_cloud(mesh):
    points = np.asarray(mesh.points)
    return points

def polar_to_cartesian(rho, alpha, h):
    x = rho * np.cos(alpha)
    y = rho * np.sin(alpha)
    z = h
    return x, y, z

def convert_to_cartesian(points):
    """
    Convert points from polar to Cartesian coordinates.
    Assumes points are in the form [rho, alpha, h].
    """
    cartesian_points = np.array([polar_to_cartesian(p[0], p[1], p[2]) for p in points])
    return cartesian_points

def plot_cartesian_point_cloud(points, dpi=300):
    fig = plt.figure(dpi=dpi)
    
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o', s=1, label='Reconstructed Point Cloud')
    
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.legend()

    plt.show()

def plot_cartesian_point_cloud_gt(points, dpi=300):
    fig = plt.figure(dpi=dpi)
    
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='y', marker='o', s=1, label='Ground Truth Point Cloud')
    
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.legend()

    plt.show()

def plot_polar_point_cloud(points, dpi=300):
    fig = plt.figure(dpi=dpi)
    
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='^', s=1, label='Reconstructed Point Cloud')
    
    ax.set_xlabel('Rho')
    ax.set_ylabel('Alpha')
    ax.set_zlabel('H')
    ax.legend()

    plt.show()

def main(cartesian_file_path, polar_file_path, densify_factor=2, dpi=300):
    # Load Cartesian mesh
    cartesian_mesh = read_vtk_mesh(cartesian_file_path)
    cartesian_points = vtk_to_point_cloud(cartesian_mesh)
    
    # Load polar mesh
    polar_mesh = read_vtk_mesh(polar_file_path)
    polar_points = vtk_to_point_cloud(polar_mesh)


    # Plot both point clouds in separate figures
    plot_cartesian_point_cloud(cartesian_points, dpi)
    plot_cartesian_point_cloud_gt(cartesian_points, dpi)
    plot_polar_point_cloud(polar_points, dpi)

if __name__ == "__main__":
    cartesian_file_path = "./rendering/cartesian_mesh9.vtk"
    polar_file_path = "./rendering/polar_mesh9.vtk"
    densify_factor = 1
    dpi = 150
    main(cartesian_file_path, polar_file_path, densify_factor, dpi)