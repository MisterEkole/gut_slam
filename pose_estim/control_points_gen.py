
import numpy as np

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


with open("./data/control_points.txt", "w") as file:
    for point in control_points:
        file.write(f"{point[0]} {point[1]} {point[2]}\n")

control_points
