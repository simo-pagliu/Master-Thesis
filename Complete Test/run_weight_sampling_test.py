import os
import numpy as np
from auxiliary import load_criteria, load_criteria_definitions, load_value_functions
from pile_bwt import bwt
from weight_sampling import create_weight_samples

file_path_weight_elicitations = ["wbt_results_1.csv"]
file_path_value_functions = ["value_functions_1.csv"]
file_path_criteria = "criteria.csv"

# Build canonical ordering
first_dict = load_criteria_definitions(file_path_criteria)
crit_names = [crit_name for group_data in first_dict.values() for crit_name in group_data['criteria'].keys()]
num_criteria = len(crit_names)
crit_index = {name: idx for idx, name in enumerate(crit_names)}

# Load value functions
vf_list = [[] for _ in range(num_criteria)]
for vp in file_path_value_functions:
    vf_map = load_value_functions(vp)
    for idx, crit_name in enumerate(crit_names):
        vf_list[idx].append(vf_map[crit_name])

# Build dict_data_list and compute bwt_results
dict_data_list = []
bwt_results = []
for i, fp in enumerate(file_path_weight_elicitations):
    dict_data = load_criteria(file_path_criteria, fp)
    for gname, gdata in dict_data.items():
        for crit_name, crit in gdata['criteria'].items():
            idx = crit_index[crit_name]
            crit['value_function'] = vf_list[idx][i]
    dict_data_list.append(dict_data)
    bwt_results.append(bwt(dict_data))

# Now run the sampling
weights = create_weight_samples(bwt_results, dict_data_list, file_path_weight_elicitations, crit_index)
print('Sampling finished. Produced', [len(w) for w in weights], 'samples per elicitation')
