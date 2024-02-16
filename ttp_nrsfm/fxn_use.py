""" Demo usage of some utils functions """

from ttp_nrsfm.utils import *


''' Example usage: getNeighborsVis '''
# Example usage:
# Q, alpha, signo = register_to_ground_truth(Q, Qg)


# Example usage:
# IDX = getNeighborsVis(m, Ng, visb)

'''Example usage: reconstruct_cylindrical_mesh'''
# # Example usage:
# p = np.array([x, y, 1])  # Replace x and y with image coordinates
# vp = np.array([vx, vy, vz])  # Replace vx, vy, vz with vanishing point coordinates
# p_0 = np.array([p0x, p0y, p0z])  # Replace p0x, p0y, p0z with cylinder axis origin coordinates
# R = cylinder_radius  # Replace cylinder_radius with the actual radius value

# result = reconstruct_cylindrical_mesh(p, vp, p_0, R)
# print("P:", result[0])
# print("Theta:", result[1])
# print("h:", result[2])
# print("u_d:", result[3])

'''Example usage: reconstruct_cylindrical_geometry'''

# # Example usage:
# p = np.array([x, y, 1])  # Replace x and y with image coordinates
# vp = np.array([vx, vy, vz])  # Replace vx, vy, vz with vanishing point coordinates
# p_0 = np.array([p0x, p0y, p0z])  # Replace p0x, p0y, p0z with cylinder axis origin coordinates
# h = cylinder_height  # Replace cylinder_height with the actual height value
# theta = cylinder_theta  # Replace cylinder_theta with the actual theta value

# result = reconstruct_cylindrical_geometry(p, vp, p_0, h, theta)
# print("P:", result[0])
# print("R:", result[1])
# print("u_d:", result[2])

''' Example usage: project_cylindrical_mesh '''
# # Example usage:
# theta = np.array([...])  # Azimuthal angles of the cylindrical mesh
# h = np.array([...])  # Heights of the cylindrical mesh
# nappe = np.array([...])  # Radial distances of the cylindrical mesh
# K = np.array([[focal_length, 0, principal_point_x],
#               [0, focal_length, principal_point_y],
#               [0, 0, 1]])  # Camera intrinsic matrix
# camTcyl = np.array([...])  # Transformation matrix from cylindrical mesh coordinates to camera coordinates
# zmin = some_zmin_value  # Minimum depth value for clipping

# result = project_cylindrical_mesh(theta, h, nappe, K, camTcyl, zmin)
# print(result)

''' Example usage: create_cylindrical_mesh '''
# # Example usage:
# nb_pts_radiaux = 20
# nb_pts_axiaux = 10
# rayon = 5.0
# hauteur = 10.0

# theta, h, nappe, pts3d = create_cylindrical_mesh(nb_pts_radiaux, nb_pts_axiaux, rayon, hauteur)
# print("theta:\n", theta)
# print("h:\n", h)
# print("nappe:\n", nappe)
# print("pts3d:\n", pts3d)
''' Example usage: create_I_model(3model from mesh) '''
# # Example usage:
# theta = np.linspace(0, 2 * np.pi, 36)
# pts2d = np.random.rand(3 * len(theta), len(theta))  # Replace with actual pts2d values
# I_new = np.random.rand(len(theta) - 1, 3)  # Replace with actual I_new values

# create_I_model(pts2d, I_new, theta)

''' Example usage: adjust_cylindrical_mesh '''
# # Example usage:
# theta = np.linspace(0, 2 * np.pi, 36)
# h = np.linspace(0, 10, 10)
# nappe = np.ones((10, 36))
# K = np.array([[focal_length, 0, principal_point_x],
#               [0, focal_length, principal_point_y],
#               [0, 0, 1]])
# vp = np.array([vp_x, vp_y, vp_z])
# p_0 = np.array([p0_x, p0_y, p0_z])
# R = cylinder_radius

# zmin = some_zmin_value

# result_pts2d, result_pts3d, result_camTcyl = adjust_cylindrical_mesh(theta, h, nappe, K, vp, p_0, R)
# print("pts2d:\n", result_pts2d)
# print("pts3d:\n", result_pts3d)
# print("camTcyl:\n", result_camTcyl)