import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch

# Load images
polar_mesh_img = mpimg.imread('./plots/polar_mesh2.png')
cylindrical_mesh_img = mpimg.imread('./plots/cylindrical_mesh2.png')
cartesian_mesh_img = mpimg.imread('./plots/cartesian_mesh2.png')

# Create a figure
fig, axs = plt.subplots(2, 2, figsize=(15, 15))

#Plot manually drawn grid of control points
axs[0, 0].set_title('Grid of Control Points')
# Manually drawing the control points grid
control_points = [[i, j] for i in range(10) for j in range(10)]
for point in control_points:
    axs[0, 0].plot(point[0], point[1], 'ko')  # 'go' means green color, round points
axs[0, 0].set_xlim(-1, 10)
axs[0, 0].set_ylim(-1, 10)
axs[0, 0].axis('off')

# Plot Polar Mesh
axs[0, 1].imshow(polar_mesh_img)
axs[0, 1].set_title('Polar Coordinates')
axs[0, 1].axis('off')

# Plot Cylindrical Mesh
axs[1, 0].imshow(cartesian_mesh_img)
axs[1, 0].set_title('Cartesian Coordinates')
axs[1, 0].axis('off')

# Plot cylindrical Mesh
axs[1, 1].imshow(cylindrical_mesh_img)
axs[1, 1].set_title('Cylindrical Coordinates')
axs[1, 1].axis('off')




# Add transformation arrows
def add_arrow(ax, start,end,text=None,text_offset=(0, 0)):
    arrow = FancyArrowPatch(start, end, mutation_scale=20, color='blue', arrowstyle='-|>')
    ax.add_patch(arrow)
    if text:
        mid = ((start[0] + end[0]) / 2 + text_offset[0], (start[1] + end[1]) / 2 + text_offset[1])
        ax.text(mid[0], mid[1], text, fontsize=12, ha='center', va='center', backgroundcolor='white')

# Add arrows between plots
arrow_props = dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=2, mutation_scale=20)

# From Control Points to Polar Mesh
fig.add_artist(FancyArrowPatch((0.4, 0.8), (0.6, 0.8), transform=fig.transFigure, **arrow_props))
fig.text(0.5, 0.78, "B-spline Meshing", fontsize=12, ha='center', va='center', backgroundcolor='white')

# From Polar Mesh to Cylindrical Mesh
fig.add_artist(FancyArrowPatch((0.8, 0.6), (0.8, 0.45), transform=fig.transFigure, **arrow_props))
fig.text(0.77, 0.5, "Polar to\n Cylindrical Coord", fontsize=12, ha='center', va='center', backgroundcolor='white', rotation='vertical')
# From Cylindrical Mesh to Cartesian Mesh
fig.add_artist(FancyArrowPatch((0.6, 0.2), (0.4, 0.2), transform=fig.transFigure, **arrow_props))
fig.text(0.5, 0.28, "Cylindrical to Cartesian Coord", fontsize=12, ha='center', va='center', backgroundcolor='white')




# Adjust layout
plt.tight_layout()

# Save the combined figure
plt.savefig('./plots/combined_lifecycle_diagram_improved.png')
plt.show()





































# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# # Load images
# control_points_img = mpimg.imread('./data/control_points.png')
# deformed_mesh_img = mpimg.imread('./data/deformed_mesh.png')
# cartesian_mesh_img = mpimg.imread('./data/cartesian_mesh.png')
# polar_mesh_img = mpimg.imread('./data/polar_mesh.png')
# cylindrical_mesh_img = mpimg.imread('./data/cylindrical_mesh.png')

# # Create a figure
# fig, axs = plt.subplots(3, 2, figsize=(15, 15))

# # Plot images
# axs[0, 0].imshow(control_points_img)
# axs[0, 0].set_title('Control Points')
# axs[0, 0].axis('off')

# axs[1, 0].imshow(deformed_mesh_img)
# axs[1, 0].set_title('B-Spline Mesh Deformation')
# axs[1, 0].axis('off')

# axs[2, 0].imshow(cartesian_mesh_img)
# axs[2, 0].set_title('Cartesian Mesh')
# axs[2, 0].axis('off')

# axs[0, 1].imshow(polar_mesh_img)
# axs[0, 1].set_title('Polar Mesh')
# axs[0, 1].axis('off')

# axs[1, 1].imshow(cylindrical_mesh_img)
# axs[1, 1].set_title('Cylindrical Mesh')
# axs[1, 1].axis('off')

# # Adjust layout
# plt.tight_layout()

# # Save the combined figure
# plt.savefig('./data/combined_lifecycle.png')
# plt.show()

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# from matplotlib.patches import FancyArrowPatch

# # Load images
# polar_mesh_img = mpimg.imread('./plots/polar_mesh.png')
# cylindrical_mesh_img = mpimg.imread('./plots/cylindrical_mesh.png')
# cartesian_mesh_img = mpimg.imread('./plots/cartesian_mesh.png')

# # Create a figure
# fig, axs = plt.subplots(2, 2, figsize=(15, 15))

# # Plot manually drawn grid of control points
# axs[0, 0].set_title('Grid of Control Points')
# # Manually drawing the control points grid
# control_points = [[i, j] for i in range(10) for j in range(10)]
# for point in control_points:
#     axs[0, 0].plot(point[0], point[1], 'go')  # 'go' means green color, round points
# axs[0, 0].set_xlim(-1, 10)
# axs[0, 0].set_ylim(-1, 10)
# axs[0, 0].axis('off')

# # Plot Polar Mesh
# axs[0, 1].imshow(polar_mesh_img)
# #axs[0, 1].set_title('Polar Mesh')
# axs[0, 1].axis('off')

# # Plot Cylindrical Mesh
# axs[1, 0].imshow(cylindrical_mesh_img)
# #axs[1, 0].set_title('Cylindrical Mesh')
# axs[1, 0].axis('off')

# # Plot Cartesian Mesh
# axs[1, 1].imshow(cartesian_mesh_img)
# #axs[1, 1].set_title('Cartesian Mesh')
# axs[1, 1].axis('off')

# # Add transformation arrows
# def add_arrow(ax, start, end):
#     arrow = FancyArrowPatch(start, end, mutation_scale=20, color='blue', arrowstyle='-|>')
#     ax.add_patch(arrow)

# # Add arrows between plots
# add_arrow(axs[0, 0], (1, 1), (0.8, 0.2))
# add_arrow(axs[0, 1], (0.2, 0.8), (0.8, 0.2))
# add_arrow(axs[1, 0], (1, 1), (0.8, 0.2))

# # Adjust layout
# plt.tight_layout()

# # Save the combined figure
# plt.savefig('./plots/combined_lifecycle_diagram.png')
# plt.show()