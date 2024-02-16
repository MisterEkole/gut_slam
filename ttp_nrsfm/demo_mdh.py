from mdh_nrsfm_socp import *
from nrsfm_socp import *
from mdhnrsfm import *


import numpy as np

# Number of images (camera views)
M = 2

# Number of 3D points
N = 4

# Simulated camera projections (2D measurements)
m = []
for _ in range(M):
    m_i = np.random.rand(2, N) 
    m.append(m_i)

# Visibility matrix (for simplicity, all points visible in all images)
vis = [np.ones(N) for _ in range(M)]

# Sample inter-point distance information (partially defined)

IDX = np.array([[0, 1], [1, 2]])
# Arbitrary maximum depth heuristic
max_depth_heuristic = 2.0

# Sample execution using defined data
mu, D = MDH_NrSfM(IDX, m, vis, max_depth_heuristic, solver='ECOS')  
#mu, D= NrSfM(IDX, m, vis, solver='MOSEK')

# Display reconstructed depths and calculated distances
print("Estimated Depths (mu):\n", mu)
print("Calculated Distances (D):\n", D)

# import numpy as np

# def generate_synthetic_data(M, N):
#     # Generate synthetic data (replace this with your actual data)
#     m = [np.random.rand(2, N) for _ in range(M)]
#     vis = [np.random.choice([True, False], size=N) for _ in range(M)]

#     # Generate synthetic neighborhood matrix (replace this with your actual data)
#     K = 5  # Number of neighbors
#     IDX = np.random.randint(0, N, size=(N, K + 1))

#     return IDX, m, vis

# def main():
#     # Generate synthetic data
#     M = 3  # Number of views
#     N = 100  # Number of tracked points
#     IDX, m, vis = generate_synthetic_data(M, N)

#     # Call NrSfM function
#     mu, D = NrSfM(IDX, m, vis)

#     # Display the results
#     print("Depth matrix (mu):")
#     print(mu)
#     print("\nMaximum distance matrix (D):")
#     print(D)

# if __name__ == "__main__":
#     main()
