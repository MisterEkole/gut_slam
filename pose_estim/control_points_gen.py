import numpy as np
M = 10  
N = 10 
h_range = (0, 1000)    


rho_step_size = 5  # Step size for rho
alpha_step_size = 2*np.pi / 10 # Step size for alpha
h_step_size=10 #h_step size

def generate_uniform_grid_control_points(rho_step_size, alpha_step_size, h_constant=None, h_variable_range=None, h_step_size=None, rho_range=(0, 100), alpha_range=(0, 2 * np.pi)):
    rho_values = np.arange(rho_range[0], rho_range[1] + rho_step_size, rho_step_size)
    alpha_values = np.arange(alpha_range[0], alpha_range[1] + alpha_step_size, alpha_step_size)

    control_points = []
    for rho in rho_values:
        for alpha in alpha_values:
            if h_constant is not None:
                h = h_constant
            else:
                h_start, h_end = h_variable_range
                h_values = np.arange(h_start, h_end + h_step_size, h_step_size)
                h = h_values[len(control_points) % len(h_values)]
          
            control_points.append((rho, alpha, h))

    return np.array(control_points).reshape(len(rho_values), len(alpha_values), 3)
control_points=generate_uniform_grid_control_points(rho_step_size,alpha_step_size,h_constant=None,h_variable_range=(50,100))
control_points=control_points.reshape(-1,3)
np.savetxt('./data/control_points1.txt',control_points)
print(control_points)

print(control_points.shape)




















