import unittest
import cvxpy as cp
import numpy as np
from ttp_nrsfm.mdh_nrsfm_socp import MDH_NrSfM

class TestMDH_NrSfM(unittest.TestCase):
    def setUp(self):
        # Define some sample data for testing
        self.IDX = np.array([[0, 1], [1, 2], [2, 0]])
        self.m = [np.random.rand(2, 3) for _ in range(3)]
        self.vis = [np.ones(3) for _ in range(3)]
        self.max_depth_heuristic = 1.0

    def test_input_data_types(self):
        # Test input data types
        mu, D = MDH_NrSfM(self.IDX, self.m, self.vis, self.max_depth_heuristic)
        self.assertIsInstance(mu, np.ndarray)
        self.assertIsInstance(D, np.ndarray)
        self.assertIsInstance(self.IDX, np.ndarray)
        self.assertIsInstance(self.m, list)
        self.assertIsInstance(self.vis, list)
        self.assertIsInstance(self.max_depth_heuristic, float)

    def test_output_data_types(self):
        # Test output data types
        mu, D = MDH_NrSfM(self.IDX, self.m, self.vis, self.max_depth_heuristic)
        self.assertTrue(np.issubdtype(mu.dtype, np.number))
        self.assertTrue(np.issubdtype(D.dtype, np.number))

    def test_matrix_construction_logic(self):
        # Test matrix construction logic
        mu, D = MDH_NrSfM(self.IDX, self.m, self.vis, self.max_depth_heuristic)
        self.assertEqual(mu.shape, (3, 3)) 
        self.assertEqual(D.shape, (3, 1))   

    def test_optimization_problem_formulation(self):
        # Test optimization problem formulation
        mu, D = MDH_NrSfM(self.IDX, self.m, self.vis, self.max_depth_heuristic)
        self.assertTrue(np.all(mu >= 0))  # Check if all depth values are non-negative
        self.assertTrue(np.all(D >= 0))   # Check if all distance values are non-negative

if __name__ == '__main__':
    unittest.main()
