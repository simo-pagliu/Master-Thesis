
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
    
print(bwt_result["solver_result"])



print("\033[93mTesting with BWT results...\033[0m")
x_weights = bwt_result["solver_result"]["x"][:-1]
x_temp = np.concatenate([x_weights, [0]])
constraint_value = constraints_func(x_temp, dict_data)
error_b = min(max([abs(cv) for cv in constraint_value]), 10)
for c in constraint_value:
    c_py = float(c)
    print(c_py)  # Debugging: print as native Python float
print(f"\033[91mMaximum constraint violation (error_b): {error_b}\033[0m")  # Debugging
known_max_error = error_b
print("\n")
print(x_weights)

num_criteria = len(x_weights)
used_cells = set()
grid_size = np.int64(1000)
n_dimensions = num_criteria
# print("Number of dimensions for sampling:", n_dimensions)
nx = np.array([grid_size] * n_dimensions, dtype=np.int64)

initial_point = bwt_result["solver_result"]["x"][:num_criteria]
print(f"Initial point for sampling: {initial_point}")
# print("Initial point for sampling:", initial_point)
initial_cell = tuple(np.clip((np.array(initial_point) * grid_size).astype(np.int64), 0, grid_size - 1))

# For each dimension (num_criteria) determine the minimum and maximum values according to the cell dimensions
limits = []
for dim in range(n_dimensions):
    dim_values = initial_point[dim]
    min_val = dim_values - 1/grid_size
    max_val = dim_values + 1/grid_size
    limits.append((min_val, max_val))
    print(f"Dimension {dim}: min = {min_val}, max = {max_val}")

i = 0
count_valid_points = 0
valid_points = [initial_point]
# while i < 1000:
try:
    while True:
        i += 1
        print(f"  Iteration {i}, valid points found: {count_valid_points}", end="\r", flush=True)

        # Compute error_a for x_a_center
        x_temp = np.concatenate([initial_point, [0]])
        constraints = constraints_func(x_temp, dict_data)
        error_a = min(max([abs(cv) for cv in constraints]), 10)

        # Generate a random point x_b (already sums to 1)
        x_b = []
        for lim_min, lim_max in limits:
            rand = np.random.uniform(lim_min, lim_max)
            x_b.append(rand)

        # Compute error_b and constraint values for x_b
        x_temp = np.concatenate([x_b, [0]])
        constraint_value = constraints_func(x_temp, dict_data)
        error_b = min(max([abs(cv) for cv in constraint_value]), 10)

        condition_1 = all(cv <= 0 for cv in constraint_value)
        condition_2 = sum(x_b) = 1.0
        condition_3 = error_b <= known_max_error

        if condition_1 and condition_2 and condition_3:
            valid_points.append(x_b)
            count_valid_points += 1

except KeyboardInterrupt:
    print(f"\nKeyboard interrupt received. Stopped after {i} iterations with {count_valid_points} valid points.")
                            
# save valid points to CSV
output_file = "test_space_sampling_weights.csv"
with open(output_file, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([f"Criterion_{i+1}" for i in range(num_criteria)])  # Header
    for point in valid_points:
        writer.writerow(point)