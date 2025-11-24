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
for i, dict_data in enumerate(dict_data_list):
    print(f"Running BWT for elicitation {i+1}...")  # Debugging: Print elicitation index
    bwt_result = bwt(dict_data)
    # print(f"BWT result for elicitation {i+1}: {bwt_result}")  # Debugging: Print BWT result
    bwt_results.append(bwt_result)
print(bwt_result["solver_result"]["x"])

from pile_bwt import constraints_func
constraint_value = constraints_func(bwt_result["solver_result"]["x"], dict_data)
print(f"\033[91mConstraint values for last BWT result: {(constraint_value)}\033[0m")  # Debugging

# Compute the maximum absolute violation of constraints
max_violation = max([abs(cv) for cv in constraint_value])
print(f"\033[91mMaximum absolute constraint violation: {max_violation}\033[0m")  # Debugging


###
x_temp = np.concatenate([bwt_result["solver_result"]["x"][:-1], [0]])
print(x_temp)
constraints = constraints_func(x_temp, dict_data)
error_a = max([abs(cv) for cv in constraints])
print(f"\033[91mMaximum absolute constraint violation with z=0: {error_a}\033[0m")  # Debugging
###