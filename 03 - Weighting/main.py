################################################
# Import 3rd party libraries
################################################
import tkinter as tk
import os
import matplotlib.pyplot as plt
import numpy as np
################################################
# Import local modules
################################################
from ui import WBT_ui
from best_worst_tradeoff import bwt
from auxiliary import import_criteria

################################################
# User data input
################################################
DEBUG = True
working_directory = "03 - Weighting"

################################################
# Generate needed aux. vars
################################################
criteria = import_criteria(os.path.join(working_directory, "criteria.csv"))
criteria_count = len(criteria)

################################################
# Show all value functions before starting UI
################################################
# Group criteria by their 'group' attribute
groups = {}
for key, values in criteria.items():
    group_name = values.get('group', 'Ungrouped')
    if group_name not in groups:
        groups[group_name] = []
    groups[group_name].append((key, values))

# Create one figure per group
for group_name, group_criteria in groups.items():
    n_criteria = len(group_criteria)
    n_cols = min(3, n_criteria)
    n_rows = (n_criteria + n_cols - 1) // n_cols
    
    # Create figure with subplots for this group
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    fig.suptitle(f'Value Functions - Group: {group_name}', fontsize=16, fontweight='bold')
    
    # Flatten axes array for easier iteration
    if n_criteria == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Plot each value function in this group
    for idx, (key, values) in enumerate(group_criteria):
        ax = axes[idx]
        x_values = np.linspace(values['min'], values['max'], 100)
        y_values = [values['value_function'](x) for x in x_values]
        ax.plot(x_values, y_values, linewidth=2, color='#2b78c8')
        ax.set_title(f'{key}', fontweight='bold')
        ax.set_xlabel(f'Data Range ({values.get("unit", "")})')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    
    # Hide any unused subplots
    for idx in range(n_criteria, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()

################################################
# Run the UI
################################################
root = tk.Tk()
app = WBT_ui(root, criteria)
root.mainloop()

################################################
# Run BWT model
################################################
#result = bwt(criteria, app.worst_comparison_results, app.best_comparison_results)

################################################
# Run Auxiliary functions
################################################
#plot_results(result, app.reordered_criteria_names)
#save_to_file(result, app.reordered_criteria_names, working_directory)

################################################
# DEBUG OUTPUT
################################################
# if DEBUG:
#     print("\033[93mDebugging Information:\033[0m")
#     print("Criteria:\n", criteria)
#     print("--------------------------------")
#     print("Criteria count:\n", criteria_count)
#     print("--------------------------------")
#     print("Best comparison results: \n", app.best_comparison_results)
#     print("--------------------------------")
#     print("Worst comparison results: \n", app.worst_comparison_results)
#     print("--------------------------------")
#     print("Full Optimization Output: \n", result)
#     print("\033[93mEnd Debugging Information\033[0m")
