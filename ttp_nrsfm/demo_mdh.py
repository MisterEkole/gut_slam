from mdh_nrsfm_socp import *


# Generate synthetic data
np.random.seed(42)
M = 1  # Number of cameras
N = 10  # Number of points
K = 3  # Number of dimensions (e.g., 2D points)

# Generate random camera matrices
m = [np.random.rand(K, N) for _ in range(M)]

# Generate random visibility matrices
vis = [np.random.randint(0, 2, size=N) for _ in range(M)]

# Generate random depth indices with corrected parameters
IDX = np.random.randint(0, N, size=(N, M + 1))

# Set maximum depth heuristic
max_depth_heuristic = 10.0

# Call MDH_NrSfM function
mu_est, D_est = MDH_NrSfM(IDX, m, vis, max_depth_heuristic)

# Print results
print("Estimated Depth Matrix:")
print(mu_est)

print("\nEstimated Distance Matrix:")
print(D_est)