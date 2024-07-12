import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_vector(ax, origin, vector, color, label, linestyle='-', linewidth=1.5):
    """Utility function to plot a vector with proper labeling and styling."""
    ax.quiver(*origin, *vector, color=color, arrow_length_ratio=0.1, linestyle=linestyle, label=label, linewidth=linewidth, alpha=0.9)
    # Position the label at the tip of the vector slightly offset
    ax.text(*(origin + vector + 0.1), f'{label}', color=color, fontsize=12, ha='center')

def draw_camera(ax, center, size=0.1):
    """Utility function to draw a camera box at a given center."""
    # Define the corners of the cube based on the center and size
    r = size / 2
    corners = np.array([
        [center[0] - r, center[1] - r, center[2] - r],
        [center[0] + r, center[1] - r, center[2] - r],
        [center[0] + r, center[1] + r, center[2] - r],
        [center[0] - r, center[1] + r, center[2] - r],
        [center[0] - r, center[1] - r, center[2] + r],
        [center[0] + r, center[1] - r, center[2] + r],
        [center[0] + r, center[1] + r, center[2] + r],
        [center[0] - r, center[1] + r, center[2] + r]
    ])
    # List of sides' polygons of cube
    verts = [[corners[i] for i in [0, 1, 2, 3]], [corners[i] for i in [4, 5, 6, 7]], 
             [corners[i] for i in [0, 1, 5, 4]], [corners[i] for i in [2, 3, 7, 6]], 
             [corners[i] for i in [0, 3, 7, 4]], [corners[i] for i in [1, 2, 6, 5]]]
    # Create a 3D polygon collection
    ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
    # Add label for the camera
    ax.text(center[0], center[1], center[2] - 2 * r, 'Camera', color='red', fontsize=12, ha='center')

# Define vectors and normalize where necessary
v = np.array([1, 2, 3])
v_hat = v / np.linalg.norm(v)
camera_center = np.array([0, 0, 0])  # Center of the camera box

x = np.array([1, 0, 0])
y = np.cross(v_hat, x)
x_prime = np.cross(v_hat, y)

x_hat = x_prime / np.linalg.norm(x_prime)
y_hat = y / np.linalg.norm(y)

# Create a 3D plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Draw the camera at the origin
draw_camera(ax, camera_center)

# Plot vectors originating from the camera center
plot_vector(ax, camera_center, v_hat, 'green', 'v̂ (Normalized vanishing point)')
plot_vector(ax, camera_center, x_hat, 'red', 'x̂ (Final Camera X-axis)')
plot_vector(ax, camera_center, y_hat, 'purple', 'ŷ (Final Camera Y-axis)')

# Adding dotted lines for component vectors
for vec, color in [(v_hat, 'green'), (x_hat, 'red'), (y_hat, 'purple')]:
    for i in range(3):
        # Component lines from the origin along each axis
        line = np.zeros_like(vec)
        line[i] = vec[i]
        ax.plot(*zip(camera_center, camera_center + line), color=color, linestyle='dotted', alpha=0.5)

# Setting the plot limits and labels
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# Adding a grid for better orientation
ax.grid(True)

# Adding a legend and a title
ax.legend(loc='upper right')
ax.set_title('Camera Pose Initialization Vectors')

# Display the plot
plt.show()


















# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# def plot_vector(ax, origin, vector, color, label, linestyle='-', linewidth=1.5):
#     """Utility function to plot a vector with proper labeling and styling."""
#     ax.quiver(*origin, *vector, color=color, arrow_length_ratio=0.1, linestyle=linestyle, label=label, linewidth=linewidth, alpha=0.9)
#     # Position the label at the tip of the vector slightly offset
#     ax.text(*(origin + vector + 0.1), f'{label}', color=color, fontsize=12, ha='center')

# def draw_camera(ax, center, size=0.1):
#     """Utility function to draw a camera box at a given center."""
#     # Define the corners of the cube based on the center and size
#     r = size / 2
#     corners = np.array([
#         [center[0] - r, center[1] - r, center[2] - r],
#         [center[0] + r, center[1] - r, center[2] - r],
#         [center[0] + r, center[1] + r, center[2] - r],
#         [center[0] - r, center[1] + r, center[2] - r],
#         [center[0] - r, center[1] - r, center[2] + r],
#         [center[0] + r, center[1] - r, center[2] + r],
#         [center[0] + r, center[1] + r, center[2] + r],
#         [center[0] - r, center[1] + r, center[2] + r]
#     ])
#     # List of sides' polygons of cube
#     verts = [[corners[i] for i in [0, 1, 2, 3]], [corners[i] for i in [4, 5, 6, 7]], 
#              [corners[i] for i in [0, 1, 5, 4]], [corners[i] for i in [2, 3, 7, 6]], 
#              [corners[i] for i in [0, 3, 7, 4]], [corners[i] for i in [1, 2, 6, 5]]]
#     # Create a 3D polygon collection
#     ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

# # Define vectors and normalize where necessary
# v = np.array([1, 2, 3])
# v_hat = v / np.linalg.norm(v)
# camera_center = np.array([0, 0, 0])  # Center of the camera box

# x = np.array([1, 0, 0])
# y = np.cross(v_hat, x)
# x_prime = np.cross(v_hat, y)

# x_hat = x_prime / np.linalg.norm(x_prime)
# y_hat = y / np.linalg.norm(y)

# # Create a 3D plot
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')

# # Draw the camera at the origin
# draw_camera(ax, camera_center)

# # Plot vectors originating from the camera center
# plot_vector(ax, camera_center, v_hat, 'green', 'v̂ (Normalized vanishing point)')
# plot_vector(ax, camera_center, x_hat, 'red', 'x̂ (Final Camera X-axis)')
# plot_vector(ax, camera_center, y_hat, 'purple', 'ŷ (Final Camera Y-axis)')

# # Setting the plot limits and labels
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# ax.set_zlim([-1, 1])
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')

# # Adding a grid for better orientation
# ax.grid(True)

# # Adding a legend and a title
# ax.legend(loc='upper right')
# ax.set_title('Camera Pose Initialisaiton Vectors')

# # Display the plot
# plt.show()











# # import numpy as np
# # import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D

# # def plot_vector(ax, origin, vector, color, label, linestyle='-', linewidth=1.5):
# #     """Utility function to plot a vector with proper labeling and styling."""
# #     ax.quiver(*origin, *vector, color=color, arrow_length_ratio=0.1, linestyle=linestyle, label=label, linewidth=linewidth, alpha=0.9)
# #     # Position the label at the tip of the vector slightly offset
# #     ax.text(*(origin + vector + 0.1), f'{label}', color=color, fontsize=12, ha='center')

# # # Define vectors and normalize where necessary
# # v = np.array([1, 2, 3])
# # v_hat = v / np.linalg.norm(v)

# # x = np.array([1, 0, 0])
# # y = np.cross(v_hat, x)
# # x_prime = np.cross(v_hat, y)

# # x_hat = x_prime / np.linalg.norm(x_prime)
# # y_hat = y / np.linalg.norm(y)

# # # Create a 3D plot
# # fig = plt.figure(figsize=(10, 10))
# # ax = fig.add_subplot(111, projection='3d')

# # # Plot original vectors and transformations
# # plot_vector(ax, (0, 0, 0), v_hat, 'green', 'v̂ (Normalized vanishing point)')
# # plot_vector(ax, (0, 0, 0), x, 'gray', 'x (Initial X-axis)')
# # plot_vector(ax, (0, 0, 0), y, 'blue', 'y = v̂ × x')
# # plot_vector(ax, (0, 0, 0), x_prime, 'orange', "x' = v̂ × y")
# # plot_vector(ax, (0, 0, 0), x_hat, 'red', 'x̂ (Final Camera X-axis)')
# # plot_vector(ax, (0, 0, 0), y_hat, 'purple', 'ŷ (Final Camera Y-axis)', linewidth=2)

# # # Enhance visibility by adding dashed lines to show components
# # ax.plot((0, v_hat[0]), (0, v_hat[1]), (0, v_hat[2]), 'green', linestyle='dotted', alpha=0.5)
# # ax.plot((0, x_hat[0]), (0, x_hat[1]), (0, x_hat[2]), 'red', linestyle='dotted', alpha=0.5)
# # ax.plot((0, y_hat[0]), (0, y_hat[1]), (0, y_hat[2]), 'purple', linestyle='dotted', alpha=0.5)

# # # Setting the plot limits and labels
# # ax.set_xlim([-1, 1])
# # ax.set_ylim([-1, 1])
# # ax.set_zlim([-1, 1])
# # ax.set_xlabel('X axis')
# # ax.set_ylabel('Y axis')
# # ax.set_zlabel('Z axis')

# # # Adding a grid for better orientation
# # ax.grid(True)

# # # Adding a legend and a title
# # ax.legend(loc='upper right')
# # ax.set_title('Camera Pose Initialization Vectors')

# # # Display the plot
# # plt.show()
