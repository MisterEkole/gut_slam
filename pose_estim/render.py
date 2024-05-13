''' Mesh rendering to img script'''

import pyvista as pv


def capture_mesh_to_image(points, filename):
    """
    Creates a 3D mesh from points and captures a 2D projection as an image.
    
    Parameters:
    - points: A NumPy array of shape (N, 3) containing the XYZ coordinates of the points.
    - filename: String, the file name to save the screenshot.
    """
    cloud = pv.PolyData(points)
    mesh = cloud.delaunay_2d()
    mesh = mesh.smooth(n_iter=300)
    scalars = mesh.points[:, 2]
    
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars=scalars, cmap='viridis', show_edges=False)
    
    
    plotter.camera.position = (0, 0, 10)
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.up = (0, 1, 0)

    plotter.show_axes = False
    plotter.background_color = 'white'
    
    
    plotter.show(screenshot=filename)

def render_and_save(mesh_file, output_filename):
    plotter = pv.Plotter(off_screen=True, window_size=[500, 400])
    mesh = pv.read(mesh_file)
    centroid = mesh.center
    scalar=mesh.points[:,2]  

    plotter.add_mesh(mesh, scalars=scalar, cmap='viridis', show_edges=True)
    

    camera_position = (0.6, 1.5, 0.9)  #adjust cam position to get appropriate rendering
    camera_focal_point = centroid

    camera_view_up = (0, 0, 1)
    plotter.camera.position = camera_position
    plotter.camera.focal_point = camera_focal_point
    plotter.camera.view_up = camera_view_up
    plotter.show(auto_close=True)  
    plotter.screenshot(output_filename)
    plotter.close()

def render_mesh_from_file(mesh_file, output_image, camera_position, camera_focal_point, camera_view_up):
    """
    Render a mesh from a file and save a 2D image from a specified camera view.
    
    Parameters:
        mesh_file (str): Path to the mesh file (.vtk or .ply).
        output_image (str): Path where the output image will be saved.
        camera_position (tuple): The (x, y, z) position of the camera.
        camera_focal_point (tuple): The (x, y, z) focal point of the camera.
        camera_view_up (tuple): The upward direction for the camera.
    """
    # Load the mesh
    mesh = pv.read(mesh_file)
    
    # Create a plotting object and add the mesh
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh, color='white', show_edges=True)

    # Set camera settings
    plotter.camera.position = camera_position
    plotter.camera.focal_point = camera_focal_point
    plotter.camera.view_up = camera_view_up

    # Show the plotter, capture the image, and close
    plotter.show(auto_close=False)
    plotter.screenshot(output_image)
    plotter.close()

def render_mesh_from_front(mesh_file, output_image):
    # Read the mesh file
    mesh = pv.read(mesh_file)

    # Create a plotting object
    plotter = pv.Plotter(off_screen=True)
    scalars=mesh.points[:,2]
    #plotter.add_mesh(mesh, color='w', show_edges=True)
    plotter.add_mesh(mesh, scalars=scalars, cmap='viridis', show_edges=True, show_scalar_bar=False)

    # Calculate the center of the mesh
    center = mesh.center

    # Set the camera position (front view)
    camera_position = [center[0], center[1] - max(mesh.bounds[3] - mesh.bounds[2], mesh.bounds[5] - mesh.bounds[4]), center[2]]
    camera_focal_point = center
    camera_view_up = [0, 0, 1]

    # Apply the camera settings
    plotter.camera.position = camera_position
    plotter.camera.focal_point = camera_focal_point
    plotter.camera.view_up = camera_view_up

    # Render the image and save it
    plotter.show(auto_close=False)
    plotter.screenshot(output_image)
    plotter.close()


if __name__ == "__main__":
 
    mesh_file = '/Users/ekole/Dev/gut_slam/pose_estim/rendering/mesh5.ply'  # Update this path
    output_image = './rendering/img_o.png'  # Update this path
    camera_position = (1, 1, 1)  
    camera_focal_point = (1, 0, 0)  
    camera_view_up = (0, 0, 1)  
    
    #render_mesh_from_file(mesh_file, output_image, camera_position, camera_focal_point, camera_view_up)
    render_mesh_from_front(mesh_file,output_image)
