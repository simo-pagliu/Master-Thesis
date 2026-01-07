import numpy as np
import matplotlib.pyplot as plt

# Define points for piecewise linear functions to ensure they start at 0 and end at 1
x_points = np.array([3, 10, 15, 20, 27])

# Define y points for two slightly different piecewise linear functions
y1_points = np.array([0, 0.2, 0.4, 0.8, 1])
y2_points = np.array([0, 0.3, 0.5, 0.82, 1])

# Plot the piecewise linear functions
plt.figure(figsize=(8, 8))
plt.plot(x_points, y1_points, label='$v_{f1}(x)$', color='#66a0ed', linewidth=2.5)
plt.plot(x_points, y2_points, label='$v_{f2}(x)$', color='#004aad', linewidth=2.5)

# Add labels and title
plt.xlabel('$x$', fontsize=20)
plt.ylabel('$v_f(x)$', fontsize=20)
# plt.title('Comparison of Two Slightly Different Piecewise Linear Value Functions (0 to 1)')
plt.legend(fontsize=20)
plt.grid(True)
plt.tick_params(axis='both', labelsize=16)
plt.ylim(-0.05, 1.05)
plt.savefig('./LaTex_Project/ad_hoc_plots/value_functions_example.png', dpi=300)
plt.show()
