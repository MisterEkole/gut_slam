'''
-------------------------------------------------------------
Utils for Photometric  3D Reconstruction from Single Image
Author: Mitterrand Ekole
Date: 10-02-2024
-------------------------------------------------------------
'''

import numpy as np
from PIL import Image

def get_pixel_rgb(image_path, x, y):
    """
    Gets the RGB values of a pixel in an image.

    Args:
        image_path (str): The path to the image file.
        x (int): The x-coordinate of the pixel.
        y (int): The y-coordinate of the pixel.

    Returns:
        tuple: A tuple containing the RGB values (red, green, blue) of the pixel.
              Returns None if the image cannot be opened or coordinates are invalid.
    """

    try:
        
        image = Image.open(image_path)

        # Convert to RGB format (in case the image isn't already in RGB)
        image = image.convert('RGB')

        # Access the pixel data        
        pixel_data = image.load()[x, y]

        return pixel_data  # Return the R, G, B values

    except FileNotFoundError:
        print(f"Error: Image not found at '{image_path}'")
        return None
    except IndexError:
        print(f"Error: Coordinates ({x}, {y}) are outside the image dimensions.")
        return None
    

def get_cam_params():
    #return cx,cy,fx,fy and distortion coeff k1-4 for kanala brandt camera projection model
    return 735.37, 552.80, 717.21, 717.48, -0.13893, -1.2396e-3, 9.1258e-4, -4.0716e-5
def get_intrinsic_matrix():
    cx, cy, fx, fy, k1, k2, k3, k4 = get_cam_params()
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    return K

def unprojec_cam_model(u,d):
    
    #kanalabrandt unprojection model
    #map 2D pixel val u to 3D point d
    # if isinstance(u,(tuple, list,np.ndarray)):
    #     u0, u1 = u[0], u[1] if isinstance(u, (tuple, list)) else u
    # else:
    #     #u is scalar, assume u is first component
    #     u0, u1 = u,0
       

    cx,cy,fx,fy,k1,k2,k3,k4=get_cam_params()
    mx=(u[0]-cx)/fx
    #mx=(u0-cx)/fx
   # my=(u1-cy)/fy
    my=(u[1]-cy)/fy
    r=np.sqrt(mx**2+my**2)**0.5

    #find roots for model
    coeff=np.array([k4,0,k3,0,k2,0,k1,0,1,-r])
    roots=np.roots(coeff)
    theta=np.real(roots[0])

    #get 3D point

    x=np.sin(theta)*(mx/r)
    y=np.sin(theta)*(my/r)
    z=np.cos(theta)

    return mx,my, theta

#another way of getting pixel intensity
def get_intensity(pixel):
  r,g,b = pixel
  return (float(r) + float(g) + float(b)) / (255*3)

def get_canonical_intensity(img, k, g_t, gamma):
    # Compute canonical intensity using the given formula
    intensity = np.cos(0) / (1.0 ** 2)  # Replace 0 and 1.0 with actual values based on your data
    return intensity
