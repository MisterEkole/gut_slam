import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import SmoothBivariateSpline
from utils import generate_uniform_grid_control_points, polar_to_cartesian

class Project3D_2D_cam:
    def __init__(self, intrinsic_matrix, rotation_matrix, translation_vector):
        self.intrinsic_matrix = np.array(intrinsic_matrix)
        self.rotation_matrix = np.array(rotation_matrix)
        self.translation_vector = np.array(translation_vector).reshape(3, 1)

    def project_points(self, points_3d):
        num_points = points_3d.shape[0]
        homogeneous_3d = np.hstack((points_3d, np.ones((num_points, 1))))
        camera_coords = (self.rotation_matrix @ homogeneous_3d[:, :3].T + self.translation_vector).T
        image_points_homogeneous = (self.intrinsic_matrix @ camera_coords.T).T
        points_2d = image_points_homogeneous[:, :2] / image_points_homogeneous[:, [2]]
        return points_2d

    @staticmethod
    def get_camera_parameters(image_height, image_width, rotation_vector, translation_vector, image_center):
        fx = image_height / (2 * np.tan(np.radians(30)/2))
        fy = fx
        cx, cy, _ = image_center
        intrinsic_matrix = np.array([[fx, 0, cx],
                                     [0, fy, cy],
                                     [0, 0, 1]])
        rotation_matrix = np.array(rotation_vector).reshape(3, 3)
        translation_matrix = np.array(translation_vector).reshape(3, 1)
        return intrinsic_matrix, rotation_matrix, translation_matrix

class BMeshDefDense:
    def __init__(self, radius, center):
        self.radius = radius
        self.center = center

    def twist_deformation(self, points, twist_rate):
        twisted_points = np.copy(points)
        for point in twisted_points:
            rho, alpha, z = point
            twist_angle = twist_rate * z
            twisted_alpha = alpha + twist_angle
            point[1] = twisted_alpha
        return twisted_points

    def bend_deformation(self, points, bend_amplitude, bend_frequency):
        bent_points = np.copy(points)
        for point in bent_points:
            rho, alpha, z = point
            bend = bend_amplitude * np.sin(bend_frequency * z)
            bent_rho = rho + bend
            point[0] = bent_rho
        return bent_points

    def apply_random_disturbances(self, points, disturbance_amplitude):
        disturbed_points = np.copy(points)
        disturbances = np.random.uniform(-disturbance_amplitude, disturbance_amplitude, size=disturbed_points.shape)
        disturbed_points += disturbances
        return disturbed_points
    
    def apply_sinusoidal_disturbances(self, points, amplitude, frequency):
        disturbed_points = np.copy(points)
        for i in range(disturbed_points.shape[0]):
            disturbed_points[i, 0] += amplitude * np.sin(frequency * points[i, 0])
            disturbed_points[i, 1] += amplitude * np.sin(frequency * points[i, 1])
            disturbed_points[i, 2] += amplitude * np.sin(frequency * points[i, 2])
        return disturbed_points

    def b_mesh_deformation(self, control_points, subsample_factor=2, disturbance_amplitude=None, bend_amplitude=None, bend_frequency=None):
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
        pts = self.apply_sinusoidal_disturbances(pts, amplitude=10.0, frequency=0.5)
        return pts

# Create control points for the B-spline mesh
rho_step_size = 0.1  # Step size for rho
alpha_step_size = np.pi/4  # Step size for alpha
radius = 100
center = (0, 0)



control_points = generate_uniform_grid_control_points(rho_step_size, alpha_step_size,R=100)
bmesh = BMeshDefDense(radius=10, center=[0, 0, 0])
deformed_points = bmesh.b_mesh_deformation(control_points)

rho = deformed_points[:, 0]
alpha = deformed_points[:, 1]
h = deformed_points[:, 2]
x, y, z = polar_to_cartesian(rho, alpha, h)
deformed_points = np.vstack((x, y, z)).T


intrinsic_matrix, rotation_matrix, translation_vector = Project3D_2D_cam.get_camera_parameters(
    image_height=800, image_width=800, rotation_vector=np.eye(3), translation_vector=[0, 0, 20], image_center=[400, 400, 0]
)

camera = Project3D_2D_cam(intrinsic_matrix, rotation_matrix, translation_vector)
projected_points = camera.project_points(deformed_points)


fig = plt.figure(figsize=(12, 6))

# Plot the 3D deformed mesh
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(deformed_points[:, 0], deformed_points[:, 1], deformed_points[:, 2], c='r', label='Deformed Mesh')
ax1.set_title('3D Deformed B-Spline Mesh')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend()

# Plot the 2D projection
ax2 = fig.add_subplot(122)
ax2.scatter(projected_points[:, 0], projected_points[:, 1], c='b', label='Projected Points')
ax2.set_title('2D Projection of Deformed Mesh')
ax2.set_xlabel('Image X')
ax2.set_ylabel('Image Y')
ax2.legend()

# Plot the camera and projection lines
camera_position = -rotation_matrix.T @ translation_vector
camera_box = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]]) - 0.5
camera_box = camera_box @ np.diag([2, 2, 4]) + camera_position.T


ax1.plot(camera_box[[0, 1, 2, 3, 0], 0], camera_box[[0, 1, 2, 3, 0], 1], camera_box[[0, 1, 2, 3, 0], 2], 'k-')
ax1.plot(camera_box[[4, 5, 6, 7, 4], 0], camera_box[[4, 5, 6, 7, 4], 1], camera_box[[4, 5, 6, 7, 4], 2], 'k-')
ax1.plot(camera_box[[0, 4], 0], camera_box[[0, 4], 1], camera_box[[0, 4], 2], 'k-')
ax1.plot(camera_box[[1, 5], 0], camera_box[[1, 5], 1], camera_box[[1, 5], 2], 'k-')
ax1.plot(camera_box[[2, 6], 0], camera_box[[2, 6], 1], camera_box[[2, 6], 2], 'k-')
ax1.plot(camera_box[[3, 7], 0], camera_box[[3, 7], 1], camera_box[[3, 7], 2], 'k-')
ax1.text(camera_position[0, 0], camera_position[1, 0], camera_position[2, 0], 'Camera', color='k')

# for point in deformed_points:
#     ax1.plot([camera_position[0, 0], point[0]], [camera_position[1, 0], point[1]], [camera_position[2, 0], point[2]], 'k--', alpha=0.3)

plt.tight_layout()
plt.show()