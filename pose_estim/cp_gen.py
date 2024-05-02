import numpy as np

control_points=np.random.rand(10,10,3)
np.savetxt('control_points6.txt', control_points.reshape(-1,3))