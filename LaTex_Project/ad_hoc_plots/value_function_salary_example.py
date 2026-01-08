import numpy as np
import matplotlib.pyplot as plt

# Define points for salary value function
# Lower threshold below 1500, steep increase to 2000, less steep to 3000
x_points = np.array([1000, 1500, 2000, 3000])
y_points = np.array([0, 0, 0.7, 1])

# Plot the piecewise linear function
plt.figure(figsize=(8, 8))
plt.plot(x_points, y_points, label='Salary Value Function', color='#66a0ed', linewidth=2.5)

# Add labels and title
plt.xlabel('Salary', fontsize=20)
plt.ylabel('$v_f(x)$', fontsize=20)
# plt.title('Comparison of Two Slightly Different Piecewise Linear Value Functions (0 to 1)')
# plt.legend(fontsize=20)
plt.grid(True)
plt.tick_params(axis='both', labelsize=16)
plt.ylim(-0.05, 1.05)
plt.savefig('./LaTex_Project/ad_hoc_plots/value_function_example.png', dpi=300)
plt.show()
