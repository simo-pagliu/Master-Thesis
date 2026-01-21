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
import pandas as pd
import csv
#################################################################################

#################################################################################
# Import internal modules
from pile_bwt import bwt, constraints_func, define_weight_spaces
from up_mavt import mc_simulation
from aggregation_methods import weighted_sum, harmonic_mean, geometric_mean
from auxiliary import (
    startup,
    combine_alternatives_by_country,
    load_criteria_file,
    verify_criteria_consistency,
    convert_qualitative_indicators_in_folders,
    remap_bwt_results_for_country
)
#################################################################################

# Ensure all relative file accesses resolve relative to this script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

#################################################################################
# USER INPUTS
# Country selection (IT, FR, CH, PO)
selected_country = "IT"

# Elicitation run numbers (folders containing results)
elicitation_numbers = [1, 2, 4]

# Folders Containing quick elicitation results (only QIs)
QI_elicitation_numbers = [3, 5, 6, 7]

# Weight space generation parameters
required_weight_solutions = 1  # Target number of unique weight combinations

# Montecarlo Parameters
n_runs = 10000  # Number of Montecarlo simulation runs
PLOTS = True  # Toggle plots
plot_bins = 50  # Number of bins for histograms
STRICT = True  # Toggle strict mode
RANDOM_WEIGHT_ANALYSIS = True
UPDATE_EVERY = 100  # Update plots every N runs
opinion_weights = np.ones(len(elicitation_numbers))/len(elicitation_numbers)  # Equal weights for each elicitation
#################################################################################

if RANDOM_WEIGHT_ANALYSIS:
    STRICT = False  # Disable strict mode when running random weight analysis

# Convert qualitative indicators (and update criteria.csv accordingly) before loading criteria files
folders_to_convert = sorted(set(elicitation_numbers + (QI_elicitation_numbers or [])))
if folders_to_convert:
    print(f"Converting QIs to 0-1 scale in: {folders_to_convert}")
    convert_qualitative_indicators_in_folders(folders_to_convert)

#################################################################################
# Load and verify criteria from elicitation results
# Build paths for elicitation results
elicitation_criteria_paths = []
weight_elicitations = []
value_functions = []

for elicit_num in elicitation_numbers:
    elicit_dir = os.path.join(SCRIPT_DIR, "elicitation_results", str(elicit_num))
    
    # Criteria file path
    crit_path = os.path.join(elicit_dir, "criteria.csv")
    elicitation_criteria_paths.append(crit_path)
    
    # Weight elicitation file (BWT results): use country-adjusted comparisons when possible
    # This remaps declared comparison values into the country's value-function domain.
    weight_file = remap_bwt_results_for_country(elicit_num, selected_country, script_dir=SCRIPT_DIR)
    weight_elicitations.append(weight_file)
    
    # Value functions file for selected country
    vf_file = os.path.join(elicit_dir, selected_country, "value_functions.csv")
    value_functions.append(vf_file)

print(f"Loading elicitation results for country: {selected_country}")
print(f"Elicitations: {elicitation_numbers}")

# Load and verify all criteria are the same
print("Loading and verifying criteria files...")
criteria_dfs = [load_criteria_file(path) for path in elicitation_criteria_paths]
criteria_verified = verify_criteria_consistency(criteria_dfs)
print(f"✓ All criteria files are consistent. Using common criteria with {len(criteria_verified)} criteria.")

# Create a temporary criteria.csv file in SCRIPT_DIR for the startup function
file_path_criteria = os.path.join(SCRIPT_DIR, "criteria.csv")
criteria_verified.to_csv(file_path_criteria, index=False)

print(f"Combining alternatives for {selected_country} from {len(elicitation_numbers)} elicitation(s)...")
elicitation_dirs = [os.path.join(SCRIPT_DIR, "elicitation_results", str(num)) for num in elicitation_numbers]
qi_elicitation_dirs = [os.path.join(SCRIPT_DIR, "elicitation_results", str(num)) for num in QI_elicitation_numbers] if QI_elicitation_numbers else None
combine_alternatives_by_country(elicitation_dirs, selected_country, SCRIPT_DIR, qi_elicitation_dirs=qi_elicitation_dirs)

# Use country-specific combined alternatives file
file_path_alternatives = os.path.join(SCRIPT_DIR, f"alternatives_{selected_country}.csv")

# Startup: Load all data
dict_data_list, crit_index, vf_list, conf_list, alternatives = startup(file_path_criteria, weight_elicitations, value_functions, file_path_alternatives)
n_alternatives = len(alternatives)
print(f"Loaded {len(dict_data_list)} elicitation(s) with {n_alternatives} alternatives.")

# Extract alternative names from the combined alternatives file
alternative_names = pd.read_csv(file_path_alternatives)['name'].tolist()
#################################################################################


def _make_unique_column_names(names):
    """Make a list of CSV column names unique while preserving order."""
    seen = {}
    unique = []
    for idx, raw in enumerate(names):
        name = str(raw).strip()
        if name == "":
            name = f"Alternative_{idx+1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        unique.append(name)
    return unique

#################################################################################
# PILE-BWT Method + Weight Space Definition
# 
# Runs BWT optimization for each elicitation and generates/loads weight spaces
print("Running BWT for each elicitation and defining weight spaces...")
bwt_results = []

for i, dict_data in enumerate(dict_data_list):
    print(f"Running BWT for elicitation {i+1}...")
    bwt_result = bwt(dict_data)
    bwt_results.append(bwt_result)

print(f"BWT solver result: {bwt_results[-1]['solver_result']['x']}")
constraint_value = constraints_func(bwt_results[-1]["solver_result"]["x"], dict_data_list[-1])
print(f"Constraint values for last BWT result: {constraint_value}")

# Define or load weight spaces using constraint-based optimization
list_of_weight_space_points = define_weight_spaces(
    dict_data_list, 
    elicitation_numbers, 
    SCRIPT_DIR, 
    required_solutions=required_weight_solutions,
    country=selected_country
)

print(f"Imported weight spaces for {len(list_of_weight_space_points)} elicitations.")
#################################################################################

#################################################################################
# Auxiliary functions for plotting live results
# This is used to monitor the progress of the MC simulation 
# So we can stop the simulation when we see results converging 
# rather than relying on a fixed number of runs
def update_plots(rank_probs, distributions, i, n_runs, n_alternatives, strict=False, n_elicitations=1, lists_of_full_sets=None, rank_counts_per_el=None):
    # General Heatmap is shows the aggregated ranking probabilities
    # Only shown when not in strict mode
    if not strict and ax is not None:
        ax.clear()
        sns.heatmap(
            rank_probs.T,
            annot=True,
            fmt=".2f",
            xticklabels=alternative_names,
            yticklabels=[f"{j+1}th" for j in range(n_alternatives)],
            cmap="YlGnBu",
            ax=ax,
            vmin=0,
            vmax=1,
            cbar=False,
        )
        ax.set_title(f"Ranking Probabilities (Run {i+1}/{n_runs})")
        ax.set_xlabel("Alternative")
        ax.set_ylabel("Rank")

    # Histograms of alternative values
    # Only show in strict mode
    # Used to judge the consensus
    if strict and axes is not None:
        for alt_idx in range(n_alternatives):
            axes[alt_idx].clear()
            if lists_of_full_sets is not None:
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
                axes[alt_idx].set_xlim(0, 1)
            axes[alt_idx].set_title(f"Distribution of Values for {alternative_names[alt_idx]}")
            axes[alt_idx].set_xlabel("Value")
            axes[alt_idx].set_ylabel("Probability")
            # Format y-axis as percentage
            axes[alt_idx].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
            axes[alt_idx].set_ylim(0, max(axes[alt_idx].get_ylim()[1], 0.1))  # Ensure some space for visibility

    # Draw figures
    if 'fig' in globals() and fig is not None:
        fig.canvas.draw()
        fig.canvas.flush_events()
    if 'fig_hist' in globals() and fig_hist is not None:
        fig_hist.canvas.draw()
        fig_hist.canvas.flush_events()
#################################################################################

#################################################################################
# Preparation for the Montecarlo Simulation
print("Starting Monte Carlo simulation...")
# Call the generator (yields results one by one)
mc_code = mc_simulation(alternatives, opinion_weights, vf_list, conf_list, list_of_weight_space_points, dict_data_list, geometric_mean, sim_runs=n_runs, strict=STRICT, crit_index=crit_index, random_weight_analysis=RANDOM_WEIGHT_ANALYSIS)
# Number of elicitation files
n_elicitations = len(dict_data_list)
# Setup plots if enabled
if PLOTS:
    
    # Heatmap setup (general)
    plt.ion()
    # Create aggregated heatmap only when not running in strict mode
    if not STRICT:
        fig, ax = plt.subplots(figsize=(8, 6))
        initial_data = np.zeros((n_alternatives, n_alternatives))
        sns.heatmap(
            initial_data.T,
            cmap="YlGnBu",
            ax=ax,
            xticklabels=alternative_names,
            yticklabels=[f"{j+1}th" for j in range(n_alternatives)],
            vmin=0,
            vmax=1,
        )
        try:
            cbar = ax.collections[0].colorbar
        except Exception:
            cbar = None
        # Ensure top and bottom margins aren't cut off by the window manager
        try:
            fig.tight_layout()
            fig.subplots_adjust(top=0.95, bottom=0.08)
        except Exception:
            pass
    else:
        # aggregated heatmap disabled in strict mode
        fig = None
        ax = None

    # Histogram setup: only create histogram figures when running in strict mode
    if STRICT:
        # Use 3 rows x 2 columns layout for better A4 paper compatibility
        fig_hist, axes = plt.subplots(3, 2, figsize=(12, 10))
        axes = axes.reshape(-1)  # Ensure axes is iterable
        # Prevent overlapping text between stacked histograms and ensure margins
        try:
            fig_hist.tight_layout()
            # Increase vertical and horizontal spacing to prevent text overlap
            fig_hist.subplots_adjust(hspace=0.40, wspace=0.30, top=0.95, bottom=0.05, left=0.08, right=0.95)
        except Exception:
            pass
    else:
        fig_hist = None
        axes = None
    
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
    if len(alternative_names) != n_alternatives:
        raise ValueError(
            f"Mismatch between loaded alternatives ({n_alternatives}) and names in CSV ({len(alternative_names)})."
        )
    header = ["Run", "Elicitation"] + _make_unique_column_names(alternative_names)
else:
    if len(alternative_names) != n_alternatives:
        raise ValueError(
            f"Mismatch between loaded alternatives ({n_alternatives}) and names in CSV ({len(alternative_names)})."
        )
    header = ["Run"] + _make_unique_column_names(alternative_names)
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
    if STRICT:
        print(f"[RUNNING] {i+1}/{n_runs*n_elicitations}", end='\r', flush=True)
    else:
        print(f"[RUNNING] {i+1}/{n_runs}", end='\r', flush=True)

if PLOTS:
    # Final layout adjustments to ensure nothing is clipped, applied for both modes
    try:
        if 'fig' in globals() and fig is not None:
            fig.tight_layout()
            fig.subplots_adjust(top=0.95, bottom=0.08)
        if 'fig_hist' in globals() and fig_hist is not None:
            fig_hist.tight_layout()
            fig_hist.subplots_adjust(hspace=0.40, wspace=0.30, top=0.95, bottom=0.05, left=0.08, right=0.95)
    except Exception:
        pass
    plt.ioff()
    plt.show()

print("\n[COMPLETED] Monte Carlo simulation finished.")
#################################################################################