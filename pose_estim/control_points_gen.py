import numpy as np

radius = 500  
height = 1000 
M = 10  
N = 10

# Generate control points for minimal deformation (points on cylinder surface)
control_points_minimal = np.zeros((M, N, 3))
for i in range(M):
    h = i * height / (M - 1)
    for j in range(N):
        angle = 2 * np.pi * j / N
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        control_points_minimal[i, j, :] = [x, y, h]

# Generate control points for significant deformation using sinusoidal perturbation
control_points_significant = np.zeros((M, N, 3))
amplitude = 150  # Amplitude of the sinusoidal deformation
frequency = 5    # Frequency of the sinusoidal deformation
for i in range(M):
    h = i * height / (M - 1)
    for j in range(N):
        angle = 2 * np.pi * j / N
        x = radius * np.cos(angle) + amplitude * np.sin(frequency * h / height * 2 * np.pi)
        y = radius * np.sin(angle) + amplitude * np.cos(frequency * h / height * 2 * np.pi)
        control_points_significant[i, j, :] = [x, y, h]

# Flatten arrays and convert to space-separated strings
flat_minimal = control_points_minimal.reshape(-1, 3)
flat_significant = control_points_significant.reshape(-1, 3)

# Save to text files
np.savetxt('./data/ControlPoints_Minimal10.txt', flat_minimal, fmt='%e', delimiter=' ')
np.savetxt('./data/ControlPoints_Significant10.txt', flat_significant, fmt='%e', delimiter=' ')


