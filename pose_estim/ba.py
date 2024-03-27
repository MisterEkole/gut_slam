''' Bundle Adjustment Algorithm for Pose and Deformation Estimation in GutSLAM
Author: Mitterand Ekole
Date: 25-03-2024
'''

import numpy as np
from utils import *
from scipy.optimize import least_squares

class BundleAdjustment:
    def __init__(self, camera_models, points_2d, points_3d):
        self.camera_models=Project3D_2D.get_camera_parameters()
        self.points_2d=np.array(points_2d)
        self.points_3d=np.array(points_3d)

        self.num_cams=len(camera_models)
        self.num_pts=len(points_2d)
        self.num_observations=len(points_3d)  #TODO: check if projected 3d points to 2d or 3D points from cylinder's surface
    

    def project_points(self, points_3d):
        projected_points = []
        for point in points_3d:
            projected_points.append(Project3D_2D.project_points(point))
        return projected_points

    def reprojection_error(self, camera_params,points_3d):
        error=[]
        for i in range(self.num_observations):
            
            points_3d=points_3d[self.num_observations[i]]

            projected_2d=self.project_points([points_3d])
            obs_2d=self.points_2d[i]
            error.append(projected_2d-obs_2d)

        return np.array(error).ravel()
    

    def optimize(self, init_cam_params,init_pts_3d):
        init_params=np.hstack=((init_cam_params.ravel(), init_pts_3d.ravel()))
        
        results = least_squares(self.reprojection_error,init_params, method='lm')

        optim_cam_params=results.x[:len(init_cam_params)].reshape(self.num_cams,-1)
        optim_3d_pts=results.x[len(init_cam_params):].reshape(-1,3)


        return optim_cam_params,optim_3d_pts