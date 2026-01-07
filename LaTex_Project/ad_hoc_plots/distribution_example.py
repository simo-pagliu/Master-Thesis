import numpy as np
import matplotlib.pyplot as plt

# Create a wonky/irregular distribution (multimodal with different peaks)
x = np.linspace(0, 30, 300)

# Create a custom wonky distribution using a mix of distributions
prob = (0.3 * np.exp(-((x - 8)**2) / 20) +  # First peak
        0.5 * np.exp(-((x - 18)**2) / 15) +  # Second peak (taller)
        0.2 * np.exp(-((x - 25)**2) / 10))   # Third peak (smaller)

# Normalize to make it a proper probability distribution
prob = prob / np.trapz(prob, x)

# Plot the distribution
plt.figure(figsize=(8, 8))
plt.plot(x, prob, color='#66a0ed', linewidth=2.5)
plt.fill_between(x, prob, alpha=0.3, color='#66a0ed')

# Add labels
plt.xlabel('$x$', fontsize=20)
plt.ylabel('$p(x)$', fontsize=20)
plt.grid(True)
plt.tick_params(axis='both', labelsize=16)
plt.ylim(0, max(prob) * 1.1)
plt.xlim(0, 30)
plt.savefig('./LaTex_Project/ad_hoc_plots/distribution_example.png', dpi=300)
plt.show()
