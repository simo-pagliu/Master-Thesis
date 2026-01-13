import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys

# Add parent directory to path for imports
PARENT_DIR = os.path.dirname(SCRIPT_DIR := os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)

from pile_bwt import bwt
from auxiliary import load_criteria, load_value_functions_with_confidence

# Get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UP_MAVT_DIR = os.path.dirname(SCRIPT_DIR)

# Find all weight CSV files
weight_file = glob.glob(os.path.join(SCRIPT_DIR, "BWT_results_2_weights.csv"))[0]


# Extract the elicitation number from filename
filename = os.path.basename(weight_file)
elicit_num = filename.split('_')[2]  # BWT_results_X_weights.csv -> X

print(f"Processing {filename}...")

# Load BWT results CSV and criteria data
bwt_results_file = os.path.join(SCRIPT_DIR, f"BWT_results_{elicit_num}.csv")
criteria_file = os.path.join(UP_MAVT_DIR, "criteria.csv")
dict_data = load_criteria(criteria_file, bwt_results_file)

# Load value functions and attach them to dict_data
elicit_dir = os.path.join(UP_MAVT_DIR, "elicitation_results", elicit_num)
vf_file = os.path.join(elicit_dir, "IT", "value_functions.csv")

try:
    vf_map, conf_map = load_value_functions_with_confidence(vf_file)
    
    # Attach value functions to criteria in dict_data
    for group_name, group_data in dict_data.items():
        for crit_name, crit in group_data['criteria'].items():
            if crit_name in vf_map:
                crit['value_function'] = vf_map[crit_name]
    
    print(f"  Loaded value functions from {vf_file}")
except Exception as e:
    print(f"  Warning: Could not load value functions: {e}")

# Compute optimal weights using BWT
optimal_weights = None
try:
    result = bwt(dict_data)
    if result and 'criteria_weights' in result:
        # result['criteria_weights'] is a dict of dicts: {group: {criterion: weight}}
        # Flatten it to {criterion: weight}
        optimal_weights = {}
        for group_name, group_weights in result['criteria_weights'].items():
            optimal_weights.update(group_weights)
        print(f"  BWT optimization successful, z = {result['z']:.6f}")
    else:
        print(f"  BWT optimization failed or returned no result")
except Exception as e:
    print(f"  Error running BWT: {e}")

# Read all weights from the file
# Format: each row is [criterion_name, value1, value2, value3, ...]
weights_dict = {}
with open(weight_file, 'r', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 2:
            criterion = row[0]
            # All columns after the first are weight values
            weights = [float(v) for v in row[1:]]
            weights_dict[criterion] = weights

# Compute min, max, and range for each criterion
criteria_names = []
min_weights = []
max_weights = []
ranges = []

for criterion in sorted(weights_dict.keys()):
    weights = weights_dict[criterion]
    min_w = min(weights)
    max_w = max(weights)
    criteria_names.append(criterion)
    min_weights.append(min_w)
    max_weights.append(max_w)
    ranges.append(max_w - min_w)

# Create horizontal bar plot
fig, ax = plt.subplots(figsize=(10, max(6, len(criteria_names) * 0.4)))

y_pos = np.arange(len(criteria_names))

# Plot horizontal bars from min to max
for i, (criterion, min_w, max_w, rng) in enumerate(zip(criteria_names, min_weights, max_weights, ranges)):
    # Draw the bar from min to max
    ax.barh(i, rng, left=min_w, height=0.6, 
            color='#2b78c8', alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add text showing the range to the right of the bar
    ax.text(max_w + 0.01, i, f'[{min_w:.3f}, {max_w:.3f}]', 
            ha='left', va='center', fontsize=8, fontweight='bold')

# Plot optimal weights as red points if available
if optimal_weights is not None:
    for i, criterion in enumerate(criteria_names):
        if criterion in optimal_weights:
            opt_weight = optimal_weights[criterion]
            ax.plot(opt_weight, i, 'ro', markersize=8, markeredgecolor='darkred', 
                    markeredgewidth=1.5, label='Optimal' if i == 0 else '', zorder=5)

ax.set_yticks(y_pos)
ax.set_yticklabels(criteria_names, fontsize=9)
ax.set_xlabel('Weight Value', fontsize=11, fontweight='bold')
ax.set_title(f'Weight Space Ranges - Elicitation {elicit_num}', 
                fontsize=13, fontweight='bold')
ax.set_xlim(0, 1)
ax.grid(axis='x', alpha=0.3)

if optimal_weights is not None:
    ax.legend(loc='upper right')

plt.tight_layout()
plt.show()

# Create second figure: comparison of declared ratios (a values) vs optimal weight ratios
if optimal_weights is not None:
    # Read BWT results to get declared ratios (a values) and comparison pairs
    comparisons = []
    declared_ratios = []
    optimal_ratios = []
    labels = []
    
    try:
        with open(bwt_results_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                comp_type = row.get('Type', '')
                ref = row.get('Reference', '')
                other = row.get('Other', '')
                group = row.get('Group', '')
                
                # Get declared ratio (a value)
                try:
                    a_val = float(row.get('a', 0))
                except Exception:
                    continue
                
                # Compute optimal ratio from weights
                if ref in optimal_weights and other in optimal_weights:
                    w_ref = optimal_weights[ref]
                    w_other = optimal_weights[other]
                    
                    # For best comparisons: ratio is w_ref / w_other
                    # For worst comparisons: ratio is w_other / w_ref (inverted)
                    if comp_type == 'best':
                        opt_ratio = w_ref / w_other if w_other != 0 else 0
                    else:  # worst
                        opt_ratio = w_other / w_ref if w_ref != 0 else 0
                    
                    # Determine label prefix based on group and comparison type
                    if group == "Between-groups-B":
                        prefix = "BB" if comp_type == "best" else "BW"
                    elif group == "Between-groups-W":
                        prefix = "WB" if comp_type == "best" else "WW"
                    else:
                        prefix = "B" if comp_type == "best" else "W"
                    
                    comparisons.append((comp_type, ref, other))
                    declared_ratios.append(a_val)
                    optimal_ratios.append(opt_ratio)
                    labels.append(f"{prefix}: {ref[:15]} vs {other[:15]}")
    except Exception as e:
        print(f"  Could not create ratio comparison plot: {e}")
        comparisons = []
    
    if comparisons:
        # Create grouped bar chart
        n_comparisons = len(comparisons)
        fig2, ax2 = plt.subplots(figsize=(max(10, n_comparisons * 0.5), 8))
        
        x_pos = np.arange(n_comparisons)
        width = 0.35
        
        # Plot declared ratios and optimal ratios side by side
        bars1 = ax2.bar(x_pos - width/2, declared_ratios, width, 
                        label='Declared Ratio (a)', color='#2b78c8', alpha=0.7, edgecolor='black')
        bars2 = ax2.bar(x_pos + width/2, optimal_ratios, width, 
                        label='Optimal Weight Ratio', color='#c8782b', alpha=0.7, edgecolor='black')
        
        ax2.set_xlabel('Comparisons', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Ratio Value', fontsize=11, fontweight='bold')
        ax2.set_title(f'Declared vs Optimal Weight Ratios - Elicitation {elicit_num}', 
                        fontsize=13, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()

