
import numpy as np
M = 10  
N = 10 
h_range = (0, 1000)    


rho_step_size = 5  # Step size for rho
alpha_step_size = 2*np.pi / 10 # Step size for alpha

# Function to convert polar coordinates to Cartesian coordinates
def polar_to_cartesian(rho, alpha, z):
    x = rho * np.cos(alpha)
    y = rho * np.sin(alpha)
    return x, y, z


def generate_uniform_grid_control_points(rho_step_size, alpha_step_size, h_constant=None, h_variable_range=None, rho_range=(0, 50), alpha_range=(0, 2 * np.pi)): #default rho range can be changed, from 0 to rho max where rho max is the radius of the cylinder
    rho_values = np.arange(rho_range[0], rho_range[1] + rho_step_size, rho_step_size)
    alpha_values = np.arange(alpha_range[0], alpha_range[1] + alpha_step_size, alpha_step_size)
    
    control_points = []
    for rho in rho_values:
        for alpha in alpha_values:
            if h_constant is not None:
                h = h_constant
            else:
                h = np.random.uniform(*h_variable_range)
            x, y, z = polar_to_cartesian(rho, alpha, h)
            control_points.append((x, y, z))

    return np.array(control_points).reshape(len(rho_values), len(alpha_values), 3)
control_points=generate_uniform_grid_control_points(rho_step_size,alpha_step_size,h_constant=None,h_variable_range=(50,100))
control_points=control_points.reshape(-1,3)
np.savetxt('./data/control_points1.txt',control_points)
print(control_points)

print(control_points.shape)




















# import numpy as np

# M = 10  
# N = 10 
# rho_range = (0, 500) 
# x_range=(0,500)  
# y_range=(0,500)
# alpha_range = (0, 2*np.pi) 
# h_range = (0, 1000)    
# width = 512
# height = 512
# vpx = 0
# vpy = 0
# vpz = 10
# #corners of 3d space
# corners_3d = [(0, 0, 0), (width, 0, 0), (0, height, 0), (width, height, 0)]
# rho_max = max(np.sqrt((vpx - x)**2 + (vpy - y)**2 + (vpz - z)**2) for x, y, z in corners_3d)

# def generate_control_points_constant_z(M, N, z_constant):
#     control_points = []
#     for i in range(M):
#         for j in range(N):
#             #rho = np.random.uniform(0, rho_max)
#             rho=np.random.uniform(*rho_range)
#             alpha = np.random.uniform(*alpha_range)
#             control_points.append((rho, alpha, z_constant))
#     return control_points

# def generate_control_points_variable_h(M, N):
#     control_points = []
#     for i in range(M):
#         for j in range(N):
#             #rho = np.random.uniform(0,rho_max)
#             rho=np.random.uniform(*rho_range)
#             alpha = np.random.uniform(*alpha_range)
#             h = np.random.uniform(*h_range)
#             control_points.append((rho, alpha, h))
#     return control_points
# def generate_uniform_control_points_constant_z(M, N, z_constant):
#     control_points = []
#     rho_values = np.linspace(*rho_range, M)
#     alpha_values = np.linspace(*alpha_range, N)
#     for rho in rho_values:
#         for alpha in alpha_values:
#             control_points.append((rho, alpha, z_constant))
#     return control_points

# def generate_uniform_control_points_variable_h(M, N):
#     control_points = []
#     rho_values = np.linspace(*rho_range, M)
#     alpha_values = np.linspace(*alpha_range, N)
#     for rho in rho_values:
#         for alpha in alpha_values:
#             h = np.random.uniform(*h_range)
#             control_points.append((rho, alpha, h))
#     return control_points

# def lin_control_points_constant_z(M, N, z_constant):
#     control_points = []
#     rho_values = np.linspace(0, rho_max, M)
#     alpha_values = np.linspace(0, 2 * np.pi, N)
#     for rho in rho_values:
#         for alpha in alpha_values:
#             control_points.append((rho, alpha, z_constant))
#     return control_points

# def lin_control_points_variable_h(M, N):
#     control_points = []
#     rho_values = np.linspace(0, rho_max, M)
#     alpha_values = np.linspace(0, 2 * np.pi, N)
#     h_values = np.linspace(*h_range, M * N)
#     idx = 0
#     for rho in rho_values:
#         for alpha in alpha_values:
#             h = h_values[idx]
#             control_points.append((rho, alpha, h))
#             idx += 1
#     return control_points


# # Choose method to generate control points
# method = "constant_z" 
# #method="variable_h" # Change to "variable_h" for the other method

# if method == "constant_z":
#     z_constant = 0
#     #control_points = generate_uniform_control_points_constant_z(M, N, z_constant)
#     control_points=generate_control_points_constant_z(M,N,z_constant)
    
# else:
#     #control_points = generate_uniform_control_points_variable_h(M, N)
#     control_points=generate_control_points_variable_h(M,N)
    

# with open("./data/control_points13.txt", "w") as file:
#     for point in control_points:
#         file.write(f"{point[0]} {point[1]} {point[2]}\n")

# control_points