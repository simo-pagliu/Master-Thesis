import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from aux import load_alternatives, load_criteria
from pivotal_bwt import bwt
from up_mavt import mc_simulation
from aggregation_methods import weighted_sum
from weight_sampling import run_mcmc_sampling
import os
import csv

# Load data
file_path_elicitations = ["wbt_results.csv", "wbt_results.csv"]
file_path_criteria = "criteria.csv"
sim_runs = 10000
batches = 100
sampling_bins = 100
num_criteria = sum(len(group_data['criteria']) for group_data in load_criteria(file_path_criteria, file_path_elicitations[0]).values())
vf_list = [[] for _ in range(num_criteria)]
for fp in file_path_elicitations:
    dict_data = load_criteria(file_path_criteria, fp)
    for idx, crit_data in enumerate(
        crit_data for group_data in dict_data.values() for crit_data in group_data['criteria'].values()
    ):
        vf_list[idx].append(crit_data['value_function'])

# Run BWT and collect errors
err_list = []
for fp in file_path_elicitations:
    dict_data = load_criteria(file_path_criteria, fp)
    results = bwt(dict_data)
    err_list.append(results["z"])

# Collect valid samples and write to CSV if not already present
num_groups = len(load_criteria(file_path_criteria, file_path_elicitations[0]).keys())
weight_list = []
    
for i, (fp, err_v) in enumerate(zip(file_path_elicitations, err_list)):
    # Define output file name
    output_file = os.path.splitext(fp)[0] + "_weights_" + str(i) + ".csv"

    if not os.path.exists(output_file):
        # Generate weights
        weights = run_mcmc_sampling(dict_data, num_criteria, num_groups, err_v)
        weight_list.append(weights)

        # Write weights to CSV
        with open(output_file, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([f"Criterion_{i+1}" for i in range(num_criteria)])  # Header row
            writer.writerows(weights)
    else:
        # Read the weights back from the existing CSV file
        weights = []
        with open(output_file, mode='r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                weights.append([float(value) for value in row])
        weight_list.append(weights)

# Load alternatives
alternatives = load_alternatives("alternatives.csv")
print(np.shape(weight_list))
# Live plotting wrapper
def mc_simulation_with_live_plot(alternatives, vf_list, error_list, aggregation_method, sim_runs, batches, strict, weight_fixed):
    plt.ion()
    fig, ax = plt.subplots()
    lines = []
    x_data = np.arange(sim_runs)
    y_data = np.full((len(alternatives), sim_runs), np.nan)  # Initialize with NaN

    for i in range(len(alternatives)):
        line, = ax.plot([], [], label=f"Alternative {i+1}", lw=1)
        lines.append(line)

    ax.set_xlim(0, sim_runs)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Simulation Runs")
    ax.set_ylabel("Value")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    sim_generator = mc_simulation(
        alternatives, vf_list, error_list, aggregation_method, sim_runs, batches, strict, weight_fixed
    )

    all_results = []
    run_idx = 0

    for batch_results, _ in sim_generator:
        if batch_results.shape[0] == sim_runs:  # Final yield (all results for this weight set)
            all_results.append(batch_results)
        else:  # Intermediate batch
            batch_size = batch_results.shape[0]
            for alt_idx in range(len(alternatives)):
                # Fill the slice from run_idx to run_idx + batch_size
                y_data[alt_idx, run_idx:run_idx + batch_size] = batch_results[:, alt_idx]
                lines[alt_idx].set_data(x_data[:run_idx + batch_size], y_data[alt_idx, :run_idx + batch_size])
            run_idx += batch_size
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)

    plt.ioff()
    plt.show()
    return all_results, None



# Run with live plot
results, rel_errs = mc_simulation_with_live_plot(
    alternatives, vf_list, weight_list, weighted_sum, sim_runs, batches, strict=True, weight_fixed=10
)

num_weight_sets = len(results)
num_alternatives = len(alternatives)

# per-set and overall accumulators
ranking_probs_per_set = np.zeros((num_alternatives, num_alternatives, num_weight_sets))
overall_ranking_probs = np.zeros((num_alternatives, num_alternatives))

for set_idx, set_results in enumerate(results):
    # Compute ranks so 1 = best (highest score)
    ranks = np.argsort(np.argsort(-set_results, axis=1), axis=1) + 1  # shape: (sim_runs, num_alternatives)

    for alt_idx in range(num_alternatives):
        for rank in range(1, num_alternatives + 1):
            p = np.mean(ranks[:, alt_idx] == rank)
            ranking_probs_per_set[alt_idx, rank-1, set_idx] = p
            overall_ranking_probs[alt_idx, rank-1] += p  # accumulate across weight sets

# Average over weight sets to get overall probabilities
overall_ranking_probs /= num_weight_sets
