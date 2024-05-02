'''
-------------------------------------------------------------
Photometric Model for Photometric Reconstruction
Author: Mitterrand Ekole
Date: 19-02-2024
-------------------------------------------------------------
'''


import numpy as np

def light_spread_func(x,k):
    return np.power(np.abs(x),k)

def calib_p_model(x,y,z,k,g_t,gamma):
    """Calculates the expected light intensity at a pixel based on its location,  
        light properties, and calibration parameters.

    Args:
        x & y: The x-coord & y-coord of the pixel in the image.
        z : The depth (distance from the sensor) of the corresponding 3D point.
        k : Parameter controlling the spread of light (used in light_spread_func).
        g_t: Overall gain or transmission factor.
        gamma (float): Exponent for gamma correction.

    Returns:
        float: The calculated light intensity (or radiance) at the pixel.
    """
    
    mu=light_spread_func(z,k)
    fr_theta=1/np.pi  #Lambertian BRDF
    cen_to_pix=np.linalg.norm(np.array([x,y,z])) #distance from the center of the image to the pixel

    theta=2*(np.arccos(np.linalg.norm(np.array([x,y]))/cen_to_pix))/np.pi #compute angle of incidence and then thetha
    L=(mu/cen_to_pix)*fr_theta*np.cos(theta)*g_t
    L=np.power(np.abs(L),gamma)

    return L

def cost_func(I, L,sigma=1e-3):
    """Computes the cost function for the photometric model.
        huber norm of pixel intensity
    """
    if np.linalg.norm(I-L)<sigma:
        norm=np.linalg.norm(I-L)/(2*sigma)
    else:
        norm=np.abs(I-L)+(sigma/2)

    return norm

def reg_func(grad, sigma=1e-3):
    """Computes the regularization function for the photometric model.
        huber norm of gradient
    """
    g=np.exp(-np.linalg.norm(grad))  #multiply by the exponential of the norm of the gradient
    if np.linalg.norm(grad)<sigma:
        norm=np.power(np.linalg.norm(grad),2)/(2*sigma)
    else:
        norm=np.abs(grad)+(sigma/2)

    return g* norm



