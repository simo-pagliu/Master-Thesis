import tkinter as tk
import matplotlib.pyplot as plt

import numpy as np
import scipy.optimize as opt

from ui import WBT_ui
from best_worst_tradeoff import bwt

# --- Input Data ---
criteria = [
    {"name": "Salary", "min": 1200, "max": 2000},
    {"name": "Paid Leave", "min": 2, "max": 10},
    {"name": "Bonus", "min": 0, "max": 5000},
]

# --- Run ---
root = tk.Tk()
app = WBT_ui(root, criteria)
root.mainloop()

# After the window closes, you can access:
best_comparison_results = app.best_comparison_results
worst_comparison_results = app.worst_comparison_results

print("Best comparison results:", best_comparison_results)
print("Worst comparison results:", worst_comparison_results)
criteria_count = len(best_comparison_results)
print("Criteria count:", criteria_count)

# Value function (linear from 0 to 1)
def value_function(x):
    return x

value_functions = [value_function] * criteria_count

result = bwt(value_functions, worst_comparison_results, best_comparison_results)
print("Full Optimization Output:", result)

# Plotting the results
criteria_names = [c["name"] for c in criteria]
weights = result.x[:-1]  # Extract weights, excluding z
plt.close('all')
plt.figure(figsize=(8, 6))
plt.bar(criteria_names, weights)
plt.xlabel('Criteria')
plt.ylabel('Weights')
plt.title('Criterion Weights')
plt.show()
