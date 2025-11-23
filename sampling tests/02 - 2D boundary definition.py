import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def is_valid(v):
    x, y = v[0], v[1]
    c1 = x + y - 1.25
    c2 = -x + y + 0.5
    c3 = -0.75*x - y
    c4 = 2.3*x + y - 3.0
    return all(c <= 0 for c in [c1, c2, c3, c4])

# Plot constraints
x = np.linspace(0, 2, 400)
y1 = 1.25 - x
y2 = x - 0.5
y3 = -0.75*x
y4 = -2.3*x + 3.0

plt.figure(figsize=(8, 8))
plt.plot(x, y1, label='y <= -x +1.25')
plt.plot(x, y2, label='y <= x - 0.5')
plt.plot(x, y3, label='y >= -0.75x')
plt.plot(x, y4, label='y <= -2.3x + 3.0')

initial_point = np.array([1, 0], dtype=float)
plt.plot(initial_point[0], initial_point[1], 'ro', label='Initial Point')

delta = 0.1

n_dims = initial_point.size

points = []
# For each dimension
point_under_eval = initial_point
for n in range(10):  # 10 iterations to expand the search
    for i in range(n_dims):
        # For each direction (forward or backward in that dimension)
        for s in (+1, -1):
            # As long as the point is still valid, keep moving in that direction
            while is_valid(point_under_eval):
                point_under_eval[i] += s * delta
            # Now we know a couple of points, one valid, one invalid
            valid = point_under_eval
            valid[i] -= s * delta
            invalid = point_under_eval
            # Find boundary by bisection
            mid = valid
            for _ in range(50):  # 10 iterations of bisection
                mid[i] = (valid[i] + invalid[i]) / 2
                if is_valid(mid):
                    valid[i] = mid[i]
                else:
                    invalid[i] = mid[i]
            # Use the last valid point found
            plt.plot(valid[0], valid[1], 'bo')
            point_under_eval[i] -= s * delta


plt.legend()
plt.axis('equal')
plt.show()
