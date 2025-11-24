import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv

from auxiliary import load_alternatives, load_criteria, load_criteria_definitions, load_value_functions
from pile_bwt import bwt
from up_mavt import mc_simulation
from aggregation_methods import weighted_sum
from weight_sampling import create_weight_samples

# Load data from weight elicitation
file_path_weight_elicitations = ["wbt_results_1.csv", "wbt_results_2.csv"]
# Value function files (one per elicitation / run)
file_path_value_functions = ["value_functions_1.csv", "value_functions_2.csv"]

# Load data of criteria definiton and value functions
# Not great that I have both definiton of criteria and value functions in the same file
# So here there should be a loading of the criteria definitions only
file_path_criteria = "criteria.csv"

# Load value functions - build a deterministic canonical order of criteria
# Load criteria once to establish canonical ordering (group order, then criterion order)
first_dict = load_criteria_definitions(file_path_criteria)
crit_names = [crit_name for group_data in first_dict.values() for crit_name in group_data['criteria'].keys()]
num_criteria = len(crit_names)

# mapping from criterion name to its index in crit_names
crit_index = {name: idx for idx, name in enumerate(crit_names)}

# Initialize list of lists: each index corresponds to a criterion in `crit_names`
vf_list = [[] for _ in range(num_criteria)]

# Build vf_list by reading the separate value function CSVs (one per elicitation)
for vp in file_path_value_functions:
    vf_map = load_value_functions(vp)
    for idx, crit_name in enumerate(crit_names):
        vf_list[idx].append(vf_map[crit_name])

# # Quick sanity-check: print one evaluation per criterion using the canonical min/max
# for idx, crit_name in enumerate(crit_names):
#     # retrieve min/max from the canonical (first) loaded dict
#     for group_data in first_dict.values():
#         if crit_name in group_data['criteria']:
#             crit_data = group_data['criteria'][crit_name]
#             min_val = crit_data['min_value']
#             max_val = crit_data['max_value']
#             break
#     vfs = vf_list[idx]
#     # print evaluation of the first value function for this criterion
#     print(f"Criterion {idx} ('{crit_name}'): ", vfs[0](min_val), vfs[0](max_val))
    
    
# Debugging: Verify file loading
print("Weight elicitation files:", file_path_weight_elicitations)

# Construct list of dict_data for each elicitation
# Load criteria.csv which contains only criteria definitions
# Adds info about value functions from vf_list for each elicitation
print("Loading criteria definitions and attaching value functions...")
dict_data_list = []
# Debugging: Verify loop execution
print("Starting loop over weight elicitation files...")
for i, fp in enumerate(file_path_weight_elicitations):
    print(f"Processing file {i+1}/{len(file_path_weight_elicitations)}: {fp}")  # Debugging: Track loop progress
    try:
        dict_data = load_criteria(file_path_criteria, fp)
        # print(f"Successfully loaded data for {fp}")  # Debugging: Confirm successful load
    except Exception as e:
        # print(f"Error loading file {fp}: {e}")  # Debugging: Catch and print any errors
        continue

    # Attach value functions
    for gname, gdata in dict_data.items():
        for crit_name, crit in gdata['criteria'].items():
            idx = crit_index[crit_name]
            crit['value_function'] = vf_list[idx][i]
    dict_data_list.append(dict_data)
    # print(f"Appended data for {fp} to dict_data_list")  # Debugging: Confirm append

# Debugging: Verify final dict_data_list
# print("Final dict_data_list contains:", len(dict_data_list), "entries")

# Run BWT for each elicitation results
# and collect errors from the optimization problems
print("Running BWT for each elicitation...")
bwt_results = []
import matplotlib.pyplot as plt
for i, dict_data in enumerate(dict_data_list):
    print(f"Running BWT for elicitation {i+1}...")  # Debugging: Print elicitation index
    bwt_result = bwt(dict_data)
    # print(f"BWT result for elicitation {i+1}: {bwt_result}")  # Debugging: Print BWT result
    bwt_results.append(bwt_result)
    # Plot the weights found
    # plt.figure()
    # weights = bwt_result["solver_result"]["x"][:-1]  # Exclude z
    # plt.bar(range(len(weights)), weights)
    # plt.xlabel("Criterion Index")
    # plt.ylabel("Weight")
    # plt.title(f"Weights from BWT for Elicitation {i+1}")
    # plt.show()
    

print(bwt_result["solver_result"]["x"])

from pile_bwt import constraints_func
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

# Load alternatives definitons
alternatives = load_alternatives("alternatives.csv")

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
        else:
            axes[alt_idx].hist([], bins=30, alpha=0.7)
        axes[alt_idx].set_xlim(-0.2, 1.2)
        axes[alt_idx].set_title(f"Distribution of Values for Alternative {alt_idx}")
        axes[alt_idx].set_xlabel("Value")
        axes[alt_idx].set_ylabel("Probability")
        # Format y-axis as percentage
        axes[alt_idx].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
        axes[alt_idx].set_ylim(0, max(axes[alt_idx].get_ylim()[1], 0.1))  # Ensure some space for visibility
    fig_hist.tight_layout()

    # Draw both figures
    fig.canvas.draw()
    fig_hist.canvas.draw()
    fig.canvas.flush_events()
    fig_hist.canvas.flush_events()

# Initialize
n_alternatives = len(alternatives)
rank_counts = np.zeros((n_alternatives, n_alternatives))
n_runs = 10000
PLOTS = True  # Toggle plots
UPDATE_EVERY = 10  # Update plots every N runs

# Generator
mc_code = mc_simulation(alternatives, vf_list, weight_list, weighted_sum, sim_runs=n_runs, strict=True, crit_index=crit_index)

if PLOTS:
    plt.ion()
    # Heatmap setup
    fig, ax = plt.subplots(figsize=(8, 6))
    initial_data = np.zeros((n_alternatives, n_alternatives))
    sns.heatmap(initial_data, cmap="YlGnBu", ax=ax, vmin=0, vmax=1)
    cbar = ax.collections[0].colorbar

    # Histogram setup
    fig_hist, axes = plt.subplots(n_alternatives, 1, figsize=(8, 4 * n_alternatives))
    axes = axes.reshape(-1)  # Ensure axes is iterable

# Create csv to save results
# Prepare live results CSV (overwrite if exists) with header
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

full_sets = []
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

    if PLOTS and (i % UPDATE_EVERY == 0 or i == n_runs - 1):
        update_plots(rank_probs, distributions, i, n_runs, n_alternatives)

    print(f"[RUNNING] {i+1}/{n_runs}", end='\r', flush=True)

if PLOTS:
    plt.ioff()
    plt.show()

print("\n[COMPLETED] Monte Carlo simulation finished.")