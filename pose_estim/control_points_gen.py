import numpy as np
from utils import *

# control_points=np.random.rand(15,15,3)
# np.savetxt('control_points.txt', control_points.reshape(-1,3))

img_path='/Users/ekole/Dev/gut_slam/gut_images/FrameBuffer_0038.png'
a_b_val=compute_a_b_values(img_path)
print(a_b_val)
