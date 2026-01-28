import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Create figure and 3D axis
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Define the constrained weight space bounds
w1_min, w1_max = 0.2, 0.3
w2_min, w2_max = 0.6, 0.7
w3_min, w3_max = 0.4, 0.5

# Create an irregular "wonky" shape by defining vertices - scaled to fill more space
np.random.seed(42)
vertices = np.array([
    [0.15, 0.35, 0.25],
    [0.65, 0.40, 0.30],
    [0.60, 0.85, 0.28],
    [0.20, 0.42, 0.75],
    [0.62, 0.38, 0.72],
    [0.58, 0.82, 0.70],
    [0.18, 0.80, 0.68],
    [0.40, 0.60, 0.50],  # center point
])

# Use scipy to create a convex hull for a wonky but closed shape
from scipy.spatial import ConvexHull
hull = ConvexHull(vertices)

# Create the faces from the convex hull
faces = []
for simplex in hull.simplices:
    faces.append([vertices[simplex[0]], vertices[simplex[1]], vertices[simplex[2]]])

# Create the irregular shape
wonky_shape = Poly3DCollection(faces, alpha=0.3, facecolor='#66a0ed', edgecolor='#004aad', linewidths=2.5)
ax.add_collection3d(wonky_shape)

# Set labels
ax.set_xlabel('$w_1$', fontsize=20, labelpad=10)
ax.set_ylabel('$w_2$', fontsize=20, labelpad=10)
ax.set_zlabel('$w_3$', fontsize=20, labelpad=10)

# Set tick parameters
ax.tick_params(axis='both', labelsize=16, pad=8)

# Set limits to 0-1 range
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

# Set custom ticks with fewer values to avoid overlap
ticks = [0, 0.5, 1]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks(ticks)

# Adjust viewing angle
ax.view_init(elev=25, azim=120)

plt.tight_layout()
plt.savefig('weight_space_example.pdf', dpi=300, bbox_inches='tight')
plt.show()
