''' Bundle Adjustment Algorithm for Pose and Deformation Estimation in GutSLAM
Author: Mitterand Ekole
Date: 25-03-2024
'''

import numpy as np
from utils import *
from scipy.optimize import least_squares

class BundleAdjustment:
    def __init__(self, camera_models, points_2d, points_3d):
        self.camera_models=camera_models
        self.points_2d=np.array(points_2d)
        self.points_3d=np.array(points_3d)

        self.num_cams=len(camera_models)
        self.num_pts=len(points_2d)
        self.num_observations=len(points_3d)
    

    def project_points(self, points_3d, cam_idx):
        projected_points = []
        for point in points_3d:
            projected_points.append(Project3D_2D.project_points(self.camera_models[cam_idx], point))
        return np.array(projected_points)

    def calculate_reprojection_error(self, camera_params,points_3d):
        error=[]
        for i in range(self.num_observations):
            cam_idx=i%self.num_cams
            points_3d=points_3d[self.num_observations[i]]

            projected_2d=self.project_points([points_3d],cam_idx)
            obs_2d=self.points_2d[i]
            error.append(projected_2d-obs_2d)

        return np.array(error).ravel()
    

    def optimize(self, init_cam_params,init_pts_3d):
        x0=np.hstack=((init_cam_params.ravel(), init_pts_3d.ravel()))

        #lamba func for optimiser that only takes parameter vector

        fun=lambda x: self.calculate_reprojection_error(x[:len(init_cam_params)], x[len(init_cam_params):].reshape(-1,3))

        results=least_squares(fun, x0, method='lm')

        #extract optimized cam pose

        optimized_camera_params = results.x[:len(init_cam_params)].reshape(self.num_cameras, -1)
        optimized_points_3d = results.x[len(init_cam_params):].reshape(-1, 3)

        return optimized_camera_params, optimized_points_3d