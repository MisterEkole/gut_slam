import pyvista as pv
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
            self.plotter.add_mesh(mesh, show_edges=False, show_scalar_bar=True, style='wireframe' if wireframe else 'surface')
        
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

def main():
    # Example usage of the MeshPlotter class
    mesh_file = "./rendering/cartesian_mesh1.vtk"  
    #texture_img = '/Users/ekole/Dev/gut_slam/pose_estim/tex/colon_DIFF.png' 
    #texture_img='/Users/ekole/Dev/gut_slam/pose_estim/tex/gallbladder_DIFF.png'
    texture_img=None

    cmap = 'viridis'
    screenshot = './rendering/mesh_render5.png'
    wireframe = False

    plotter = MeshPlotter()
    camera_settings = plotter.visualize_cartesian(mesh_file, texture_img, cmap, screenshot, wireframe)

    # Save camera settings to a file
    plotter.save_camera_info_to_file(camera_settings, "./logs/camera_settings.txt")

if __name__ == "__main__":
    main()