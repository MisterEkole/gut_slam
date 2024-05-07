import numpy as np
from utils import *

# control_points=np.random.rand(15,15,3)
# np.savetxt('control_points.txt', control_points.reshape(-1,3))

img_path='/Users/ekole/Dev/gut_slam/gut_images/FrameBuffer_0038.png'
a_b_val=compute_a_b_values(img_path)
print(a_b_val)


# Assuming you have `height` and `radius` defined, and `M` and `N` are the grid dimensions for the control points
M = 10  # Number of vertical divisions
N = 20  # Number of circular divisions
height=1000
radius=500
heights = np.linspace(0, height, M)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
heights_grid, angles_grid = np.meshgrid(heights, angles)

# Compute the x, y, z coordinates of the control points
cp_x = radius * np.cos(angles_grid)
cp_y = radius * np.sin(angles_grid)
cp_z = heights_grid

control_points = np.stack((cp_x, cp_y, cp_z), axis=-1)