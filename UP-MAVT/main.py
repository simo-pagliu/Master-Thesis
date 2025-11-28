#################################################################################
# Main script to run UP-MAVT analysis
#
# Simone Pagliuca, 2025-2026
#
# Description:
# TBC....
#################################################################################

#################################################################################
# Import third party libraries
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
#################################################################################

#################################################################################
# Import internal modules
from pile_bwt import bwt, constraints_func
from up_mavt import startup, mc_simulation
from aggregation_methods import weighted_sum
from weight_sampling import create_weight_samples
#################################################################################

#################################################################################
# USER INPUTS
# Weight elicitation files (one per elicitation / run)
file_path_weight_elicitations = ["wbt_results_1.csv"]

# Value function files (one per elicitation / run)
file_path_value_functions = ["value_functions_1.csv"]

# Criteria definitions file
file_path_criteria = "criteria.csv"

# Montecarlo Parameters
n_runs = 10000
PLOTS = True  # Toggle plots
UPDATE_EVERY = 10  # Update plots every N runs
#################################################################################


#################################################################################
# Startup: Load all data
dict_data_list, crit_index, vf_list, alternatives = startup(file_path_criteria, file_path_weight_elicitations, file_path_value_functions)
n_alternatives = len(alternatives)
print(f"Loaded {len(dict_data_list)} elicitation(s) with {n_alternatives} alternatives.")
#################################################################################

#################################################################################
# PILE-BWT Method
# 
# Work in progress while we try to fix issues with constraints and solver
# For now it runs the BWT optimization and then tries to sample more values from the space
print("Running BWT for each elicitation...")
bwt_results = []

for i, dict_data in enumerate(dict_data_list):
    print(f"Running BWT for elicitation {i+1}...")  # Debugging: Print elicitation index
    bwt_result = bwt(dict_data)
    bwt_results.append(bwt_result)
    
print(bwt_result["solver_result"]["x"])


constraint_value = constraints_func(bwt_result["solver_result"]["x"], dict_data)
print(f"\033[91mConstraint values for last BWT result: {constraint_value}\033[0m")  # Debugging
# Create files of valid sets of weights
# We have a list of errors, one per each eliciation
# We have to create tables of possible weights to sample from in the MC simulation
weight_list = create_weight_samples(bwt_results, dict_data_list, file_path_weight_elicitations, crit_index)
# print(np.shape(weight_list))

# Debugging: Inspect weight_list before passing to mc_simulation
print("Inspecting weight_list...")
for i, weights in enumerate(weight_list):
    print(f"Elicitation {i+1}: {len(weights)} rows, first row shape: {len(weights[0]) if weights else 'N/A'}")
#################################################################################

#################################################################################
# Auxiliary functions for plotting live results
# This is used to monitor the progress of the MC simulation 
# So we can stop the simulation when we see results converging 
# rather than relying on a fixed number of runs
def update_plots(rank_probs, distributions, i, n_runs, n_alternatives):
    # Heatmap
    ax.clear()
    sns.heatmap(
        rank_probs,
        annot=True,
        fmt=".2f",
        xticklabels=[f"{j+1}th" for j in range(n_alternatives)],
        yticklabels=[f"Alt {j}" for j in range(n_alternatives)],
        cmap="YlGnBu",
        ax=ax,
        vmin=0,
        vmax=1,
        cbar=False,
    )
    ax.set_title(f"Ranking Probabilities (Run {i+1}/{n_runs})")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Alternative")

    # Histograms
    for alt_idx in range(n_alternatives):
        axes[alt_idx].clear()
        data = distributions[alt_idx, :i+1]
        # show probability on the y-axis: use weights so bar heights sum to 1 (probability mass)
        if data.size > 0:
            weights = np.ones_like(data) / data.size
            axes[alt_idx].hist(data, bins=30, weights=weights, alpha=0.7)
            # Auto-scale x-axis to the data range with a small padding so values are visible
            dmin = float(np.nanmin(data))
            dmax = float(np.nanmax(data))
            if dmin == dmax:
                pad = max(0.5, abs(dmin) * 0.1)
            else:
                pad = (dmax - dmin) * 0.1
            axes[alt_idx].set_xlim(dmin - pad, dmax + pad)
        else:
            axes[alt_idx].hist([], bins=30, alpha=0.7)
        axes[alt_idx].set_title(f"Distribution of Values for Alternative {alt_idx}")
        axes[alt_idx].set_xlabel("Value")
        axes[alt_idx].set_ylabel("Probability")
        # Format y-axis as percentage
        axes[alt_idx].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
        axes[alt_idx].set_ylim(0, max(axes[alt_idx].get_ylim()[1], 0.1))  # Ensure some space for visibility
    # Avoid expensive tight_layout on every update; call it only on the first update
    if i == 0:
        try:
            fig_hist.tight_layout()
        except Exception:
            pass

    # Draw both figures
    fig.canvas.draw()
    fig_hist.canvas.draw()
    fig.canvas.flush_events()
    fig_hist.canvas.flush_events()
#################################################################################

#################################################################################
# Preparation for the Montecarlo Simulation
print("Starting Monte Carlo simulation...")
# Call the generator (yields results one by one)
mc_code = mc_simulation(alternatives, vf_list, weight_list, weighted_sum, sim_runs=n_runs, strict=True, crit_index=crit_index)
# Setup plots if enabled
if PLOTS:
    
    # Heatmap setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    initial_data = np.zeros((n_alternatives, n_alternatives))
    sns.heatmap(initial_data, cmap="YlGnBu", ax=ax, vmin=0, vmax=1)
    cbar = ax.collections[0].colorbar

    # Histogram setup
    fig_hist, axes = plt.subplots(n_alternatives, 1, figsize=(8, 4 * n_alternatives))
    axes = axes.reshape(-1)  # Ensure axes is iterable
    
# Create csv to save results LIVE (avoids excessive memory usage)
output_file = "./results/results.csv"

# If results.csv exists, don't overwrite — find next available filename results_1.csv, results_2.csv, ...
if os.path.exists(output_file):
    base, ext = os.path.splitext(output_file)
    idx = 1
    candidate = f"{base}_{idx}{ext}"
    while os.path.exists(candidate):
        idx += 1
        candidate = f"{base}_{idx}{ext}"
    output_file = candidate
header = ["Run"] + [f"Alternative_{i+1}" for i in range(n_alternatives)]
with open(output_file, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    
# Prepare to collect useful information from the simulation 
# Alternative ranking probabilities
# and distributions of values of alternatives
full_sets = []
rank_counts = np.zeros((n_alternatives, n_alternatives)) 
#################################################################################

#################################################################################
# MONTECARLO MAIN LOOP
# This is the main loop that runs the simulation
# We limit it to n_runs but we can stop it whenever we want
for i, r in enumerate(mc_code):
    # print(r)
    # Compute the ranking for this run
    ranking_run = np.argsort(r)[::-1]
    for pos, alt in enumerate(ranking_run):
        rank_counts[alt, pos] += 1
    rank_probs = rank_counts / (i + 1)
    full_sets.append(r.copy())
    distributions = np.array(full_sets).T  # Shape: (n_alternatives, n_runs)
    
    # Add results to csv
    with open(output_file, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([i+1] + r)

    # Update plots if enabled
    if PLOTS and (i % UPDATE_EVERY == 0 or i == n_runs - 1):
        update_plots(rank_probs, distributions, i, n_runs, n_alternatives)

    # Print progress to console
    print(f"[RUNNING] {i+1}/{n_runs}", end='\r', flush=True)

if PLOTS:
    plt.ioff()
    plt.show()

print("\n[COMPLETED] Monte Carlo simulation finished.")
#################################################################################