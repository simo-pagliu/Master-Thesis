import numpy as np
import matplotlib.pyplot as plt

def constraints(v):
    x, y = v[0], v[1]
    c1 = x + y - 1.5      # x + y <= 1.5
    c2 = -x + y - 0.3     # y <= x + 0.3
    c3 = x + 0.5*y - 0.8  # x + 0.5*y >= 0.8
    c4 = -0.5*x + y - 0.2 # y >= 0.5*x + 0.2
    return max(c1, c2, -c3, -c4)

# Plot constraints
x = np.linspace(0, 1, 400)
y1 = 1.5 - x
y2 = x + 0.3
y3 = (0.8 - x) / 0.5
y4 = 0.2 + 0.5*x

plt.figure(figsize=(8, 8))
plt.plot(x, y1, label='x + y <= 1.5')
plt.plot(x, y2, label='y <= x + 0.3')
plt.plot(x, y3, label='x + 0.5*y >= 0.8')
plt.plot(x, y4, label='y >= 0.5*x + 0.2')

# Initial point
initial_point = np.array([0.7, 0.6], dtype=float)
plt.plot(initial_point[0], initial_point[1], 'ro', label='Initial Point')

tolerance = 1e-3
grid_size = 0.1  # Size of grid cells for exclusion zones
delta = grid_size * 1.1 # Step size

# Track valid and boundary points
valid_points = [initial_point]
boundary_points = []

# Grid to track used cells
used_cells = set()
used_cells.add(tuple((initial_point / grid_size).astype(int)))
i = 0
nx = int(np.ceil(1.0 / grid_size))
ny = nx
total_cells = nx * ny

max_iters = 1000
while i < max_iters and len(used_cells) < total_cells:
    i += 1
    # Randomly select a valid point to expand from
    x_a = valid_points[np.random.randint(len(valid_points))]
    y_a = constraints(x_a)

    # Generate a random direction
    direction = np.random.randn(2)
    direction = direction / np.linalg.norm(direction)
    x_b = x_a + delta * direction
    x_b = np.clip(x_b, 0, 1)  # Ensure x_b is in [0, 1]
    y_b = constraints(x_b)

    j = 0
    while abs(y_b) > tolerance and j < 1000:
        j += 1
        if y_b > 0:  # b is invalid
            # Move toward a
            direction = x_a - x_b
            norm = np.linalg.norm(direction)
            if norm == 0:
                break
            direction = direction / norm
            new_modulus = abs(y_a) / (abs(y_b - y_a) + 1e-10)
            x_b = x_a + direction * new_modulus
            x_b = np.clip(x_b, 0, 1)
            y_b = constraints(x_b)
        else:  # b is valid
            # Add to valid points if not in a used cell (clamp index at grid edges)
            cell_idx = (x_b / grid_size).astype(int)
            cell_idx = np.minimum(cell_idx, np.array([nx - 1, ny - 1], dtype=int))
            cell = tuple(cell_idx)
            if cell not in used_cells:
                valid_points.append(x_b.copy())
                used_cells.add(cell)
            # Move further in the same direction
            x_b = x_b + delta * direction
            x_b = np.clip(x_b, 0, 1)
            y_b = constraints(x_b)

    if j < 1000:
        boundary_points.append(x_b.copy())

# Plot valid and boundary points
for x_v in valid_points:
    plt.plot(x_v[0], x_v[1], 'bo')
# for x_b in boundary_points:
#     plt.plot(x_b[0], x_b[1], 'go')

plt.legend()
plt.axis('equal')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title('Improved Sampling in [0, 1] Space')
plt.show()
