# up-mavt.py
#
# this module containes the basic functions of the UP-MAVT framework

#################################################################################
# Import third party libraries
import numpy as np
#################################################################################

#################################################################################
# Import internal modules
from auxiliary import load_alternatives, load_criteria, load_criteria_definitions, load_value_functions
from weight_sampling import weight_sampler
#################################################################################

#################################################################################
# Data Loading and Preprocessing
def startup(file_path_criteria, file_path_weight_elicitations, file_path_value_functions):
    # Load value functions and criteria and establish canonical ordering
    # This is done to ensure that criteria are consistently ordered across different elicitation files
    # And that we do not mix up value functions in other steps
    first_dict = load_criteria_definitions(file_path_criteria)
    crit_names = [crit_name for group_data in first_dict.values() for crit_name in group_data['criteria'].keys()]

    # Number of criteria is saved as a variable since we are going to use it multiple times
    num_criteria = len(crit_names)

    # mapping from criterion name to its index in crit_names
    crit_index = {name: idx for idx, name in enumerate(crit_names)}

    # Initialize list of lists: each index corresponds to a criterion in `crit_names`
    # And built it by reading the separate value function CSVs (one per elicitation)
    vf_list = [[] for _ in range(num_criteria)]
    for vp in file_path_value_functions:
        vf_map = load_value_functions(vp)
        for idx, crit_name in enumerate(crit_names):
            vf_list[idx].append(vf_map[crit_name])    
            
    # Construct list of dict_data for each elicitation
    # Load criteria.csv which contains criteria definitions
    # Adds info about value functions from vf_list for each elicitation
    print("Loading criteria definitions and attaching value functions...")
    dict_data_list = []
    print("Starting loop over weight elicitation files...")
    for i, fp in enumerate(file_path_weight_elicitations):
        print(f"Processing file {i+1}/{len(file_path_weight_elicitations)}: {fp}")  # Debugging: Track loop progress
        dict_data = load_criteria(file_path_criteria, fp)
        
        # Attach value functions
        for gname, gdata in dict_data.items():
            for crit_name, crit in gdata['criteria'].items():
                idx = crit_index[crit_name]
                crit['value_function'] = vf_list[idx][i]
        dict_data_list.append(dict_data)

    # Debugging: Verify final dict_data_list
    # print("Final dict_data_list contains:", len(dict_data_list), "entries")

    # Load data for alternatives
    alternatives = load_alternatives("alternatives.csv")
    return dict_data_list, crit_index, vf_list, alternatives
#################################################################################

#################################################################################
# Sampling of data from their distributions
def sample_to_values(data, value_function):
    if isinstance(data, dict):
        data_values = list(data.values())[0]
        if list(data.keys())[0] == "Normal":
            mu = data_values[0]
            sigma = data_values[1]
            sample = np.random.normal(mu, sigma)

        elif list(data.keys())[0] == "Uniform":
            a = data_values[0]
            b = data_values[1]
            sample = np.random.uniform(a, b)

        elif list(data.keys())[0] == "Discrete":
            values = data_values[0]
            sample = np.random.choice(values)

        elif list(data.keys())[0] == "Triangular":
            a = data_values[0]
            b = data_values[1]
            c = data_values[2]
            sample = np.random.triangular(a, b, c)
        
        elif list(data.keys())[0] == "Uncertain Triangular":
            a = data_values[0]
            b = data_values[1]
            c = data_values[2]
            middle = np.random.choice(b)
            sample = np.random.triangular(a, middle, c)
        elif list(data.keys())[0] == "Special_1":
            a_possible = data_values[0]
            x_low = data_values[1]
            x_high = data_values[2]
            x = np.random.uniform(x_low, x_high)
            a = np.random.choice(a_possible)
            prob_0 = a * (1-x)
            prob_1 = a * x + (1-a)*(1-x)
            prob_2 = (1-a)*x
            sample = np.random.choice([0,1,2], p=[prob_0, prob_1, prob_2])
            
        else:
            raise ValueError(f"Unsupported distribution type in dictionary.{data.keys()}")
    else:
        try:
            sample = float(data)  # Ensure the data is numeric
        except ValueError:
            raise TypeError(f"Invalid data type for sampling: {data}")
    value = float(value_function(sample))
    # if value < 0 or value > 1:
        # print(f"Sampled value: {sample} from data: {data}, value is {value}")
    return value
#################################################################################

#################################################################################
# Montecarlo generator function
def mc_simulation(alternatives, vf_list, list_of_weight_space_points, dict_data_list, aggregation_method, sim_runs, strict, crit_index):
    # weight_list = np.array(weight_list)
    for mc_run in range(sim_runs):
        # For each montecarlo run we iterate over all experts
        for elicitation_idx in range(len(list_of_weight_space_points)):
            run_results = []
            # Select the weight space for this elicitation
            weight_space_points = list_of_weight_space_points[elicitation_idx]
            # Load the dict_data for this elicitation
            dict_data = dict_data_list[elicitation_idx]
            # For each expert we choose a random set of weights
            sampled_weights = weight_sampler(dict_data, weight_space_points)
            # Normalization to ensure weights sum to 1 (should be already the case)
            sampled_weights = sampled_weights / np.sum(sampled_weights)
            # For each alternative we compute its value
            for a, alt in enumerate(alternatives):
                intermediate_results = []
                # For each criterion
                # iterate over criterion name + data so we can align with vf_list by name
                for c_idx, (crit_name, criterion_data) in enumerate(alt.items()):
                    # determine the index into vf_list for this criterion
                    idx = crit_index.get(crit_name, None)

                    # choose the value function: strict -> same expert index, otherwise random
                    if strict:
                        v_f = vf_list[idx][elicitation_idx]
                    else:
                        v_f = np.random.choice(vf_list[idx])

                    # compute the value of the criterion by sampling from its distribution
                    sampled_value = sample_to_values(criterion_data, v_f)
                    # Get the weight for this criterion from the sampled weight set
                    weight = sampled_weights[idx]
                    # save this pair of weight and value
                    intermediate_results.append([weight, sampled_value])
                # Once all criterion have been evaluated we aggregate to get the overall alternative value
                # print(intermediate_results[2][1])
                alternative_value = aggregation_method(intermediate_results)
                # And save the alternative value to go then to the next alternative
                run_results.append(alternative_value)
            # Now we save the result from this run
            # So each run is consistent with the set of weight,
            # this means that we can later compute the ranking of the alternatives per run
            if strict:
                # Yield both the elicitation index and the results so the caller
                # can aggregate and plot per-elicitation statistics
                yield elicitation_idx, run_results
            else:
                yield run_results
    return