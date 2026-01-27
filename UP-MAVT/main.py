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
from plotting import create_heatmap, create_histograms, update_plots, finalize_layout
#################################################################################

# Ensure all relative file accesses resolve relative to this script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

#################################################################################
# USER INPUTS
# Country selection (IT, FR, CH, PO)
selected_country = "IT"

# Elicitation run numbers (folders containing results)
elicitation_numbers = [1, 2, 4, 9]

# Folders Containing quick elicitation results (only QIs)
QI_elicitation_numbers = [3, 5, 6, 7, 8]

# Weight space generation parameters
required_weight_solutions = 1  # Target number of unique weight combinations

# Montecarlo Parameters
n_runs = 10000  # Number of Montecarlo simulation runs
PLOTS = True  # Toggle plots
plot_bins = 50  # Number of bins for histograms
STRICT = True  # Toggle strict mode
RANDOM_WEIGHT_ANALYSIS = False
UPDATE_EVERY = 100  # Update plots every N runs
opinion_weights = np.ones(len(elicitation_numbers))/len(elicitation_numbers)  # Equal weights for each elicitation
#################################################################################

if RANDOM_WEIGHT_ANALYSIS:
    STRICT = False  # Disable strict mode when running random weight analysis

# NOTE: Automatic conversion disabled - criteria.csv must be manually prepared
folders_to_convert = sorted(set(elicitation_numbers + (QI_elicitation_numbers or [])))
if folders_to_convert:
    print(f"⚠ Warning: Ensure qualitative indicators are properly converted in: {folders_to_convert}")
    print(f"⚠ Warning: criteria.csv files should be manually prepared if needed")
    # convert_qualitative_indicators_in_folders(folders_to_convert)  # Disabled

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

# Use or create criteria.csv in SCRIPT_DIR
file_path_criteria = os.path.join(SCRIPT_DIR, "criteria.csv")
if os.path.exists(file_path_criteria):
    print(f"⚠ Using existing criteria.csv (not overwriting manual changes)")
else:
    print(f"Creating new criteria.csv from elicitation data")
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
# Preparation for the Montecarlo Simulation
print("Starting Monte Carlo simulation...")
# Call the generator (yields results one by one)
mc_code = mc_simulation(alternatives, opinion_weights, vf_list, conf_list, list_of_weight_space_points, dict_data_list, geometric_mean, sim_runs=n_runs, strict=STRICT, crit_index=crit_index, random_weight_analysis=RANDOM_WEIGHT_ANALYSIS)
# Number of elicitation files
n_elicitations = len(dict_data_list)
# Setup plots if enabled
fig = None
ax = None
fig_hist = None
axes = None
if PLOTS:
    plt.ion()
    fig, ax = create_heatmap(alternative_names, n_alternatives, STRICT)
    fig_hist, axes = create_histograms(n_alternatives, n_elicitations, STRICT, plot_bins)
    
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
        update_plots(
            rank_probs,
            distributions,
            i,
            n_runs,
            n_alternatives,
            alternative_names,
            strict=STRICT,
            n_elicitations=n_elicitations,
            lists_of_full_sets=lists_of_full_sets,
            rank_counts_per_el=rank_counts_per_el,
            fig=fig,
            ax=ax,
            fig_hist=fig_hist,
            axes=axes,
            plot_bins=plot_bins,
        )

    # Print progress to console
    if STRICT:
        print(f"[RUNNING] {i+1}/{n_runs*n_elicitations}", end='\r', flush=True)
    else:
        print(f"[RUNNING] {i+1}/{n_runs}", end='\r', flush=True)

if PLOTS:
    finalize_layout(fig, fig_hist)
    plt.ioff()
    plt.show()

print("\n[COMPLETED] Monte Carlo simulation finished.")
#################################################################################