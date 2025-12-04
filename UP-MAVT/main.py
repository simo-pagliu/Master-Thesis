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
from weight_sampling import obtain_weight_space_description
#################################################################################

#################################################################################
# USER INPUTS
# Weight elicitation files (one per elicitation / run)
file_path_weight_elicitations = ["wbt_results_1.csv", "wbt_results_2.csv"]

# Value function files (one per elicitation / run)
file_path_value_functions = ["value_functions_1.csv", "value_functions_2.csv"]

# Criteria definitions file
file_path_criteria = "criteria.csv"

# Montecarlo Parameters
n_runs = 10000
PLOTS = True  # Toggle plots
plot_bins = 50  # Number of bins for histograms
STRICT = False  # Toggle strict mode
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
list_of_weight_space_points = obtain_weight_space_description(bwt_results, dict_data_list, file_path_weight_elicitations, crit_index)
# print(np.shape(weight_list))

# Debugging: Inspect weight_list before passing to mc_simulation
print(f"Imported weight spaces for {len(list_of_weight_space_points)} elicitations.")
#################################################################################

#################################################################################
# Auxiliary functions for plotting live results
# This is used to monitor the progress of the MC simulation 
# So we can stop the simulation when we see results converging 
# rather than relying on a fixed number of runs
def update_plots(rank_probs, distributions, i, n_runs, n_alternatives, strict=False, n_elicitations=1, lists_of_full_sets=None, rank_counts_per_el=None):
    # General Heatmap (all elicitations aggregated)
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

    # Per-elicitation heatmaps (strict mode)
    if strict and axes_el is not None and lists_of_full_sets is not None and rank_counts_per_el is not None:
        for e in range(n_elicitations):
            axes_el[e].clear()
            runs_e = len(lists_of_full_sets[e])
            if runs_e > 0:
                rank_probs_e = rank_counts_per_el[e] / runs_e
            else:
                rank_probs_e = np.zeros((n_alternatives, n_alternatives))
            sns.heatmap(
                rank_probs_e,
                annot=True,
                fmt=".2f",
                xticklabels=[f"{j+1}th" for j in range(n_alternatives)],
                yticklabels=[f"Alt {j}" for j in range(n_alternatives)],
                cmap="YlGnBu",
                ax=axes_el[e],
                vmin=0,
                vmax=1,
                cbar=False,
            )
            axes_el[e].set_title(f"Elicitation {e+1} Ranking Prob (runs={runs_e})")
            axes_el[e].set_xlabel("Rank")
            axes_el[e].set_ylabel("Alternative")

    # Histograms
    for alt_idx in range(n_alternatives):
        axes[alt_idx].clear()
        if not strict:
            data = distributions[alt_idx, :i+1]
            # show probability on the y-axis: use weights so bar heights sum to 1 (probability mass)
            if data.size > 0:
                weights = np.ones_like(data) / data.size
                axes[alt_idx].hist(data, bins=plot_bins, weights=weights, alpha=0.7)
                # Force x-axis to [0,1] since values are normalized
                axes[alt_idx].set_xlim(0, 1)
            else:
                axes[alt_idx].hist([], bins=plot_bins, alpha=0.7)
                axes[alt_idx].set_xlim(0, 1)
        else:
            # Strict: overlay histograms, one per elicitation, using distinct colors
            # Create a discrete list of colors from the 'tab10' colormap in a
            # backwards-compatible way (works across Matplotlib versions).
            cmap = plt.get_cmap('tab10')
            color_list = list(cmap(np.linspace(0, 1, max(1, n_elicitations))))
            any_data = False
            for e in range(n_elicitations):
                runs_e = len(lists_of_full_sets[e])
                if runs_e == 0:
                    continue
                data_e = np.array(lists_of_full_sets[e]).T[alt_idx]
                if data_e.size > 0:
                    any_data = True
                    weights = np.ones_like(data_e) / data_e.size
                    axes[alt_idx].hist(data_e, bins=plot_bins, weights=weights, alpha=0.5, color=color_list[e], label=f"E{e+1}")
            if not any_data:
                axes[alt_idx].hist([], bins=plot_bins, alpha=0.7)
            else:
                axes[alt_idx].legend(title='Elicitation')
            # Ensure all histograms show values on [0,1]
            axes[alt_idx].set_xlim(0, 1)
        axes[alt_idx].set_title(f"Distribution of Values for Alternative {alt_idx}")
        axes[alt_idx].set_xlabel("Value")
        axes[alt_idx].set_ylabel("Probability")
        # Format y-axis as percentage
        axes[alt_idx].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
        axes[alt_idx].set_ylim(0, max(axes[alt_idx].get_ylim()[1], 0.1))  # Ensure some space for visibility
    # Avoid expensive tight_layout on every update; call it only on the first update
    # if i == 0:
    #     try:
    #         fig_hist.tight_layout()
    #     except Exception:
    #         pass

    # Draw figures
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig_hist.canvas.draw()
    fig_hist.canvas.flush_events()
    if strict and fig_el is not None:
        fig_el.canvas.draw()
        fig_el.canvas.flush_events()
#################################################################################

#################################################################################
# Preparation for the Montecarlo Simulation
print("Starting Monte Carlo simulation...")
# Call the generator (yields results one by one)
mc_code = mc_simulation(alternatives, vf_list, list_of_weight_space_points, dict_data_list, weighted_sum, sim_runs=n_runs, strict=STRICT, crit_index=crit_index)
# Number of elicitation files
n_elicitations = len(dict_data_list)
# Setup plots if enabled
if PLOTS:
    
    # Heatmap setup (general)
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    initial_data = np.zeros((n_alternatives, n_alternatives))
    sns.heatmap(initial_data, cmap="YlGnBu", ax=ax, vmin=0, vmax=1)
    cbar = ax.collections[0].colorbar
    # Ensure top and bottom margins aren't cut off by the window manager
    try:
        fig.tight_layout()
        fig.subplots_adjust(top=0.95, bottom=0.08)
    except Exception:
        pass

    # Heatmaps per elicitation (new figure) - one subplot per elicitation
    if STRICT and n_elicitations > 0:
        fig_el, axes_el = plt.subplots(n_elicitations, 1, figsize=(8, 4 * n_elicitations))
        if n_elicitations == 1:
            axes_el = np.array([axes_el])
        # Add spacing so titles and labels don't overlap between subplots
        try:
            fig_el.tight_layout()
            # slightly reduce vertical spacing and ensure top/bottom margins
            fig_el.subplots_adjust(hspace=0.35, top=0.95, bottom=0.08)
        except Exception:
            pass
    else:
        fig_el = None
        axes_el = None

    # Histogram setup (one plot per alternative). In strict mode we will overlay one histogram per elicitation
    fig_hist, axes = plt.subplots(n_alternatives, 1, figsize=(8, 4 * n_alternatives))
    axes = axes.reshape(-1)  # Ensure axes is iterable
    # Prevent overlapping text between stacked histograms and ensure margins
    try:
        fig_hist.tight_layout()
        # slightly reduce vertical spacing and set bottom/top margins
        fig_hist.subplots_adjust(hspace=0.35, top=0.95, bottom=0.08)
    except Exception:
        pass
    
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
if STRICT:
    header = ["Run", "Elicitation"] + [f"Alternative_{i+1}" for i in range(n_alternatives)]
else:
    header = ["Run"] + [f"Alternative_{i+1}" for i in range(n_alternatives)]
with open(output_file, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    
# Prepare to collect useful information from the simulation 
# Alternative ranking probabilities
# and distributions of values of alternatives
full_sets = []
rank_counts = np.zeros((n_alternatives, n_alternatives))
# If strict mode: maintain per-elicitation collections and rank counts
if STRICT:
    lists_of_full_sets = [[] for _ in range(n_elicitations)]
    rank_counts_per_el = [np.zeros((n_alternatives, n_alternatives)) for _ in range(n_elicitations)]
else:
    lists_of_full_sets = None
    rank_counts_per_el = None
#################################################################################

#################################################################################
# MONTECARLO MAIN LOOP
# This is the main loop that runs the simulation
# We limit it to n_runs but we can stop it whenever we want
for i, r in enumerate(mc_code):
    # r can be either a list of alternative values (non-strict)
    # or a tuple (elicitation_idx, run_results) when strict
    if STRICT:
        elic_idx, run_results = r
    else:
        run_results = r

    # Compute the ranking for this run
    ranking_run = np.argsort(run_results)[::-1]
    for pos, alt in enumerate(ranking_run):
        rank_counts[alt, pos] += 1
        if STRICT:
            rank_counts_per_el[elic_idx][alt, pos] += 1

    rank_probs = rank_counts / (i + 1)

    # Store run results in appropriate structure
    if STRICT:
        lists_of_full_sets[elic_idx].append(run_results.copy())
        # For the combined distributions used by the general histogram, concatenate all elicitation lists
        all_runs_combined = [item for sub in lists_of_full_sets for item in sub]
        full_sets = all_runs_combined
        distributions = np.array(full_sets).T if len(full_sets) > 0 else np.zeros((n_alternatives, 0))
    else:
        full_sets.append(run_results.copy())
        distributions = np.array(full_sets).T  # Shape: (n_alternatives, n_runs)

    # Add results to csv (include elicitation index in strict mode)
    with open(output_file, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if STRICT:
            writer.writerow([i+1, elic_idx+1] + run_results)
        else:
            writer.writerow([i+1] + run_results)

    # Update plots if enabled
    if PLOTS and (i % UPDATE_EVERY == 0 or i == n_runs - 1):
        update_plots(rank_probs, distributions, i, n_runs, n_alternatives, strict=STRICT, n_elicitations=n_elicitations, lists_of_full_sets=lists_of_full_sets, rank_counts_per_el=rank_counts_per_el)

    # Print progress to console
    print(f"[RUNNING] {i+1}/{n_runs}", end='\r', flush=True)

if PLOTS:
    # Final layout adjustments to ensure nothing is clipped, applied for both modes
    try:
        if 'fig' in globals() and fig is not None:
            fig.tight_layout()
            fig.subplots_adjust(top=0.95, bottom=0.08)
        if 'fig_hist' in globals() and fig_hist is not None:
            fig_hist.tight_layout()
            fig_hist.subplots_adjust(hspace=0.35, top=0.95, bottom=0.08)
        if 'fig_el' in globals() and fig_el is not None:
            fig_el.tight_layout()
            fig_el.subplots_adjust(hspace=0.35, top=0.95, bottom=0.08)
    except Exception:
        pass
    plt.ioff()
    plt.show()

print("\n[COMPLETED] Monte Carlo simulation finished.")
#################################################################################