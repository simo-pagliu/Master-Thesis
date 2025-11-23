import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def space_limits(v):
    x, y = v[0], v[1]
    c1 = x + y - 1.25
    c2 = -x + y + 0.5
    c3 = -0.75*x - y
    c4 = 2.3*x + y - 3.0
    return [c1, c2, c3, c4]

# Plot the space limits
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

initial_point = [1, 0]
plt.plot(initial_point[0], initial_point[1], 'ro', label='Initial Point')

cons = space_limits(initial_point)
if all(c <= 0 for c in cons):
    print("Initial point is within the feasible region.")
    p0 = np.asarray(initial_point, dtype=float)
    d = 0.3

    # Track all validated points to avoid revisiting
    validated_points = set()
    validated_points.add(tuple(p0))

    # Queue for points to expand
    queue = deque([p0])

    while queue:
        current = queue.popleft()
        n = current.size

        # Generate new points by moving +/- d along each axis
        for i in range(n):
            for s in (+1, -1):
                p = current.copy()
                p[i] += s * d
                p_tuple = tuple(p)

                if p_tuple not in validated_points:
                    if all(c <= 0 for c in space_limits(p)):
                        plt.plot(p[0], p[1], 'bo')
                        validated_points.add(p_tuple)
                        queue.append(p)
                    else:
                        plt.plot(p[0], p[1], 'rx')

plt.legend()
plt.axis('equal')
plt.show()
