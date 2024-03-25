from utils import *
import open3d as o3d

''' init warp field instance creation and pcd save'''
warp_field = WarpField(radius=1.0, height=2.0, vanishing_pts=(0, 0, 10), center=(10, 10, 10), resolution=500)
warp_field.apply_deformation(strength=0.3, frequency=2)
warp_field.densify_point_cloud(target_count=2000)
warp_field.save_point_cloud('deformed_cylinder_point_cloud.txt')

# # Extract and print the point cloud data
# point_cloud = warp_field.extract_pcd()
# #print(point_cloud)  # This will display the 3D points of the deformed cylinder
# # Optionally, visualize the deformed cylinder using PyVista
plotter = pv.Plotter()
plotter.add_mesh(warp_field.cylinder, color='lightblue', show_edges=False)
plotter.show()

if __name__ == "__main__":
    pcd_preparer = PointCloudPreparer(target_num_points=1000, normalize=True) #instance of PointCloudPreparer
    scene_point_cloud_path = '/Users/ekole/Dev/gut_slam/photometric_rec/py/pcl_output/point_cloud1.txt'
    def_cyl_path = '/Users/ekole/Dev/gut_slam/pose_estim/deformed_cylinder_point_cloud.txt'

    pc1 = pcd_preparer.read_point_cloud_from_txt(def_cyl_path)
    pc2= pcd_preparer.read_point_cloud_from_txt(scene_point_cloud_path)

    prepared_pc1 = pcd_preparer.prepare(pc1)
    prepared_pc2 = pcd_preparer.prepare(pc2)

    #visualize the prepared point clouds using Open3D
    
    #visualize_point_clouds(prepared_pc1, prepared_pc2)
    #visualize_point_clouds(prepared_pc1, prepared_pc2)   

    # # Align the source point cloud to the target using ICP
 

    aligned_source_pc = pcd_preparer.align_point_clouds(prepared_pc1, prepared_pc2, threshold=1.0)

    # Convert the aligned source point cloud back to Open3D PointCloud
    aligned_source_o3d = o3d.geometry.PointCloud()
    aligned_source_o3d.points = o3d.utility.Vector3dVector(aligned_source_pc)

    # Convert the prepared target point cloud back to Open3D PointCloud
    prepared_pc2_o3d = o3d.geometry.PointCloud()
    prepared_pc2_o3d.points = o3d.utility.Vector3dVector(prepared_pc1)

    # Visualization of the aligned point clouds
    o3d.visualization.draw_geometries([aligned_source_o3d, prepared_pc2_o3d])



   
    
    