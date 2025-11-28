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

x_weights = bwt_result["solver_result"]["x"][:-1]
x_temp = np.concatenate([x_weights, [0]])
constraint_value = constraints_func(x_temp, dict_data)
error_b = min(max([abs(cv) for cv in constraint_value]), 10)
for c in constraint_value:
    c_py = float(c)
    print(c_py)  # Debugging: print as native Python float
print(f"\033[91mMaximum constraint violation (error_b): {error_b}\033[0m")  # Debugging

##################################
# from pile_bwt import bwt_2
# bwt_result_2 = bwt_2(dict_data)
# print(bwt_result_2["solver_result"])
# input("Press Enter to continue...")

##################################
# from pile_bwt import bwt_3
# bwt_result_3 = bwt_3(dict_data)
# print(bwt_result_3["solver_result"])
# input("Press Enter to continue...")

##################################
# from pile_bwt import bwt_4
# result = bwt_4(dict_data, n=100, iters=3, polish=False)
# print(result["solver_result"])
# for w in result["solver_result"]["x"][:-1]:
#     print(float(w))
# constraint_value = constraints_func(result["solver_result"]["x"], dict_data)
# error_b = min(max([abs(cv) for cv in constraint_value]), 10)
# print(f"\033[91mMaximum constraint violation (error_b) for bwt_4: {error_b}\033[0m")  # Debugging
# for c in constraint_value:
#     c_py = float(c)
#     print(c_py)  # Debugging: print as native Python float
    
# input("Press Enter to continue...")









# Create files of valid sets of weights
# We have a list of errors, one per each eliciation
# We have to create tables of possible weights to sample from in the MC simulation
# weight_list = create_weight_samples(bwt_results, dict_data_list, file_path_weight_elicitations, crit_index)
# print(np.shape(weight_list))