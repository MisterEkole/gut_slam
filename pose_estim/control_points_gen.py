import numpy as np


radius = 500  
height = 1000 
M = 20  
N = 20 

# Generate control points for minimal deformation (points on cylinder surface)
control_points_minimal = np.zeros((M, N, 3))
for i in range(M):
    h = i * height / (M - 1)
    for j in range(N):
        angle = 2 * np.pi * j / N
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        control_points_minimal[i, j, :] = [x, y, h]

# Generate control points for significant deformation (distorted points)
control_points_significant = np.zeros((M, N, 3))
for i in range(M):
    h = i * height / (M - 1) + np.random.uniform(-0.1, 0.1) * height  # slight random offset in height
    for j in range(N):
        perturbation = np.random.uniform(0.8, 1.0)  # random scaling factor
        angle = 2 * np.pi * j / N + np.random.uniform(-0.5, 0.5)  # slight random offset in angle
        x = radius * perturbation * np.cos(angle)
        y = radius * perturbation * np.sin(angle)
        control_points_significant[i, j, :] = [x, y, h]

# Flatten arrays and convert to space-separated strings
flat_minimal = control_points_minimal.reshape(-1, 3)
flat_significant = control_points_significant.reshape(-1, 3)

# Save to text files
np.savetxt('./data/ControlPoints_Minimal.txt', flat_minimal, fmt='%e', delimiter=' ')
np.savetxt('./data/ControlPoints_Significant.txt', flat_significant, fmt='%e', delimiter=' ')
