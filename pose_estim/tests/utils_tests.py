import unittest
from pose_estim.utils import *
import numpy as np

class TestProject3D2DCam(unittest.TestCase):

    def setUp(self):
        # Example intrinsic matrix
        self.intrinsic_matrix = np.array([[1000, 0, 320],
                                          [0, 1000, 240],
                                          [0, 0, 1]])
        # Example rotation matrix (identity, for simplicity)
        self.rotation_matrix = np.eye(3)
        # Example translation vector (zero, for simplicity)
        self.translation_vector = np.zeros((3, 1))

        # Instantiate the Project3D_2D_cam object
        self.projector = Project3D_2D_cam(self.intrinsic_matrix, self.rotation_matrix, self.translation_vector)

    def test_project_points_dimension(self):
        # Example 3D points (4 points in this example)
        points_3d = np.array([[0, 0, 10],
                              [1, 1, 10],
                              [-1, -1, 10],
                              [0.5, 0.5, 5]])

        # Project the points
        projected_points_2d = self.projector.project_points(points_3d)

        # Check the shape of the projected points
        self.assertEqual(projected_points_2d.shape, (4, 2))


''' Test case to check output shape in B-Spline classess'''
class TestBSplineClasses(unittest.TestCase):

    def setUp(self):
        self.radius = 5.0
        self.center = (0, 0, 0)
        self.control_points = np.array([[[i, j, np.sin(i) + np.cos(j)] for j in range(4)] for i in range(4)])

        self.bmesh = BMesh(self.radius, self.center)
        self.bmesh_dense = BMeshDense(self.radius, self.center)
        self.bmesh_def_dense = BMeshDefDense(self.radius, self.center)

    def test_bmesh_deformation(self):
    
        deformed_points = self.bmesh.b_mesh_deformation(self.control_points) 
        # Check the shape of the deformed points
        self.assertEqual(deformed_points.shape, (16, 3))

    def test_bmesh_dense_deformation(self):
        # Perform BMeshDense deformation with subsample_factor
        deformed_points = self.bmesh_dense.b_mesh_deformation(self.control_points, subsample_factor=2)
        
        # Expected number of points: (M-1) * (N-1) * subsample_factor^2
        expected_num_points = (self.control_points.shape[0] - 1) * (self.control_points.shape[1] - 1) * (2 ** 2)
        
        # Check the shape of the deformed points
        self.assertEqual(deformed_points.shape, (expected_num_points, 3))

    def test_bmesh_def_dense_deformation(self):
        # Perform BMeshDefDense deformation with subsample_factor
        deformed_points = self.bmesh_def_dense.b_mesh_deformation(self.control_points, subsample_factor=2, disturbance_amplitude=3, bend_amplitude=5.0, bend_frequency=0.5)
        
        # Expected number of points: (M-1) * (N-1) * subsample_factor^2
        expected_num_points = (self.control_points.shape[0] - 1) * (self.control_points.shape[1] - 1) * (2 ** 2)
        
        # Check the shape of the deformed points
        self.assertEqual(deformed_points.shape, (expected_num_points, 3))

''' Test case to check if all methods can run without raising an exception in GridViz class'''
class TestGridViz(unittest.TestCase):

    def setUp(self):
       
        self.grid_shape = (1, 1)
        self.window_size = (800, 600)
        self.viz = GridViz(self.grid_shape, self.window_size)
        self.points = np.array([[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9],
                                [10, 11, 12]])
        self.subplot = (0, 0)

    def test_add_mesh_cartesian(self):
       
        try:
            self.viz.add_mesh_cartesian(self.points, self.subplot)
        except Exception as e:
            self.fail(f"add_mesh_cartesian raised an exception: {e}")

    def test_add_mesh_polar(self):
      
        try:
            self.viz.add_mesh_polar(self.points, self.subplot)
        except Exception as e:
            self.fail(f"add_mesh_polar raised an exception: {e}")

    def test_add_mesh_cy(self):
        
        try:
            self.viz.add_mesh_cy(self.points, self.subplot)
        except Exception as e:
            self.fail(f"add_mesh_cy raised an exception: {e}")

    def test_extract_polar_coordinates(self):
      
        rho, alpha, h = self.viz.extract_polar_coordinates(self.points)
        self.assertEqual(rho.shape, (4,))
        self.assertEqual(alpha.shape, (4,))
        self.assertEqual(h.shape, (4,))

if __name__ == '__main__':
    unittest.main()