# import numpy as np

# radius = 500  
# height = 1000 
# M = 10  
# N = 10

# # Generate control points for minimal deformation (points on cylinder surface)
# control_points_minimal = np.zeros((M, N, 3))
# for i in range(M):
#     h = i * height / (M - 1)
#     for j in range(N):
#         angle = 2 * np.pi * j / N
#         x = radius * np.cos(angle)
#         y = radius * np.sin(angle)
#         control_points_minimal[i, j, :] = [x, y, h]

# # Generate control points for significant deformation using sinusoidal perturbation
# control_points_significant = np.zeros((M, N, 3))
# amplitude = 150  # Amplitude of the sinusoidal deformation
# frequency = 5    # Frequency of the sinusoidal deformation
# for i in range(M):
#     h = i * height / (M - 1)
#     for j in range(N):
#         angle = 2 * np.pi * j / N
#         x = radius * np.cos(angle) + amplitude * np.sin(frequency * h / height * 2 * np.pi)
#         y = radius * np.sin(angle) + amplitude * np.cos(frequency * h / height * 2 * np.pi)
#         control_points_significant[i, j, :] = [x, y, h]

# # Flatten arrays and convert to space-separated strings
# flat_minimal = control_points_minimal.reshape(-1, 3)
# flat_significant = control_points_significant.reshape(-1, 3)

# # Save to text files
# np.savetxt('./data/ControlPoints_Minimal10.txt', flat_minimal, fmt='%e', delimiter=' ')
# np.savetxt('./data/ControlPoints_Significant10.txt', flat_significant, fmt='%e', delimiter=' ')
# import numpy as np

# def generate_cylindrical_control_points(n_points, max_radius):
#     """
#     Generate control points in a cylindrical coordinate frame.

#     Parameters:
#     - n_points (int): The total number of control points to generate.
#     - max_radius (float): The maximum value for the radial distance (rho).

#     Returns:
#     - control_points (numpy.ndarray): An array of shape (n_points, 3) containing the control points.
#     """
#     # Generate alpha values between 0 and 2*pi
#     alpha = np.random.uniform(0, 2 * np.pi, n_points)
    
#     # Set h to be constant for all points
#     h = np.ones(n_points)
    
#     # Generate rho values randomly between 0 and max_radius
#     rho = np.random.uniform(0, max_radius, n_points)
    
#     # Combine alpha, rho, and h into a single array
#     control_points = np.vstack((alpha, rho, h)).T
    
#     return control_points

# def generate_cylindrical_grid(n_alpha, n_rho, max_radius, h_value=1):
#     """
#     Generate a structured grid of control points in a cylindrical coordinate frame.

#     Parameters:
#     - n_alpha (int): Number of angular divisions.
#     - n_rho (int): Number of radial divisions.
#     - max_radius (float): Maximum value for the radial distance (rho).
#     - h_value (float): Constant height value for all points.

#     Returns:
#     - control_points (numpy.ndarray): An array of shape (n_alpha * n_rho, 3) containing the control points.
#     """
#     # Generate grid of alpha and rho values
#     alpha = np.linspace(0, 2 * np.pi, n_alpha)
#     rho = np.linspace(0, max_radius, n_rho)
    
#     # Create meshgrid
#     alpha_grid, rho_grid = np.meshgrid(alpha, rho)
    
#     # Flatten the grid arrays
#     alpha_flat = alpha_grid.flatten()
#     rho_flat = rho_grid.flatten()
    
#     # Set h to be constant for all points
#     h_flat = np.full_like(alpha_flat, h_value)
    
#     # Combine alpha, rho, and h into a single array
#     control_points = np.vstack((alpha_flat, rho_flat, h_flat)).T
    
#     return control_points


# n_points = 100  # Number of control points (10x10 grid)
# max_radius = 50  # Maximum value for rho

# n_alpha = 10  # Number of angular divisions
# n_rho = 10  # Number of radial divisions

# #control_points = generate_cylindrical_control_points(n_points, max_radius)
# control_points=generate_cylindrical_grid(n_alpha, n_rho, max_radius)

# # Save the control points to a text file
# output_file_path = "./data/ControlPoints_Cylindrical.txt"
# np.savetxt(output_file_path, control_points, fmt='%.6e',delimiter=' ')



# import numpy as np

# def generate_control_points(num_points, max_radius, constant_height):
#     control_points = []
    
#     # Generate num_points values of rho and alpha
#     rhos = np.linspace(0, max_radius, num_points)
#     alphas = np.linspace(0, 2 * np.pi, num_points)
    
#     for rho, alpha in zip(rhos, alphas):
#         x = rho * np.cos(alpha)
#         y = rho * np.sin(alpha)
#         z = constant_height
#         control_points.append((x, y, z))
    
#     return control_points

# def save_control_points(control_points, filename):
#     with open(filename, 'w') as f:
#         for point in control_points:
#             f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")

# # Parameters
# num_points = 100  # Number of control points to generate
# max_radius = 500.0  # Maximum radial distance
# constant_height = 10  # Constant height value

# # Generate and save control points
# control_points = generate_control_points(num_points, max_radius, constant_height)
# save_control_points(control_points, './data/generated_control_points.txt')


import numpy as np

# Define the parameters for control point generation
# Define the grid size for MxN control points
M = 10  
N = 10 
rho_range = (0, 10)   
alpha_range = (0, 360/N) 
h_range = (0, 10)    
width=512
height=512
vpx=0
vpy=0
vpz=10


 
corners_3d = [(0, 0, 0), (width, 0, 0), (0, height, 0), (width, height, 0)]
rho_max = max(np.sqrt((vpx - x)**2 + (vpy - y)**2 + vpz**2) for x, y, z in corners_3d)

control_points = []

for i in range(M):
    for j in range(N):
        rho = np.random.uniform(0, rho_max)
        alpha = np.random.uniform(*alpha_range)
        h = np.random.uniform(*h_range)
        control_points.append((rho, alpha, h))

# Save the control points to a .txt file
with open("./data/control_points.txt", "w") as file:
    for point in control_points:
        file.write(f"{point[0]} {point[1]} {point[2]}\n")

control_points
