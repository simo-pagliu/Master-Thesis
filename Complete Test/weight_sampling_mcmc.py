import os
import pytensor.tensor as pt
import pymc as pm
import csv

def create_weight_samples(err_list, dict_data_list, file_path_weight_elicitations, crit_index):
    weight_list = []    
    for i, (fp, err_v) in enumerate(zip(file_path_weight_elicitations, err_list)):
        # Define output file name
        output_file = os.path.splitext(fp)[0] + "_weights" + ".csv"

        if not os.path.exists(output_file):
            # Load dict_data for this elicitation and attach value functions from vf_list
            dict_data = dict_data_list[i]
            # Credo che la lunghezza di dict_data sia il numero di gruppi
            num_groups = len(dict_data)
            num_criteria = len(crit_index)

            # Generate a set of possible weights
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
            return weight_list
        
def pytensor_constraints_func(x, dict_data, Z_max):
    cons = []
    group_indices = {}
    current_index = 0
    num_criteria = sum(len(group_data['criteria']) for group_data in dict_data.values())

    # Map group and criterion names to their indices in x
    for group_name, group_data in dict_data.items():
        criteria_in_group = list(group_data['criteria'].keys())
        group_indices[group_name] = {
            'start': current_index,
            'end': current_index + len(criteria_in_group),
            'criteria': criteria_in_group
        }
        current_index += len(criteria_in_group)

    tot_number = current_index
    z_index = tot_number  # Z is the last element in x
    Z = x[z_index]  # Extract Z from x

    # INTRA-GROUP CONSTRAINTS
    for group_name, group_data in dict_data.items():
        criteria_in_group = group_indices[group_name]['criteria']
        w_start = group_indices[group_name]['start']
        w_end = group_indices[group_name]['end']
        w = x[w_start:w_end]  # Weights for this group

        for crit, comparisons in group_data['criteria'].items():
            v_f = comparisons['value_function']
            # BEST COMPARISONS
            for other_crit, value in comparisons['best_comparisons'].items():
                i = criteria_in_group.index(crit)
                j = criteria_in_group.index(other_crit)
                v_f_other = group_data['criteria'][crit]['value_function']
                cons.append(Z - pt.abs(w[i] / w[j] - 1.0 / v_f_other(value)))
            # WORST COMPARISONS
            for other_crit, value in comparisons['worst_comparisons'].items():
                i = criteria_in_group.index(crit)
                j = criteria_in_group.index(other_crit)
                v_f_other = group_data['criteria'][other_crit]['value_function']
                cons.append(Z - pt.abs(v_f_other(value) - w[i] / w[j]))

    # INTER-GROUP CONSTRAINTS
    intraB = {}
    intraW = {}
    for group_data in dict_data.values():
        intraB.update(group_data['intraB'])
        intraW.update(group_data['intraW'])

    w_G = x[tot_number:tot_number + len(dict_data)]  # Group weights

    # BEST-BEST and BEST-WORST COMPARISONS
    for crit_name, comparison in intraB.items():
        ref_crit = comparison['reference']
        other_crit = comparison['other']
        ref_group = next(g for g, gd in dict_data.items() if ref_crit in gd['criteria'])
        other_group = next(g for g, gd in dict_data.items() if other_crit in gd['criteria'])
        v_f_other = dict_data[ref_group]['criteria'][ref_crit]['value_function']
        cons.append(Z - pt.abs(w_G[0] / w_G[1] - 1.0 / v_f_other(comparison['value'])))

    # WORST-BEST and WORST-WORST COMPARISONS
    for crit_name, comparison in intraW.items():
        ref_crit = comparison['reference']
        other_crit = comparison['other']
        ref_group = next(g for g, gd in dict_data.items() if ref_crit in gd['criteria'])
        other_group = next(g for g, gd in dict_data.items() if other_crit in gd['criteria'])
        v_f_other = dict_data[other_group]['criteria'][other_crit]['value_function']
        cons.append(Z - pt.abs(v_f_other(comparison['value']) - w_G[1] / w_G[-1]))

    # NEW CONSTRAINT: Z must be smaller than Z_max
    cons.append(Z_max - Z)

    # Ensure non-negativity for criteria and group weights (cons <= 0 when satisfied)
    num_groups = len(dict_data)
    for i in range(tot_number + num_groups):  # criteria + group weights (Z is last element)
        cons.append(-x[i])  # -w_i <= 0  <=>  w_i >= 0

    # Ensure Z is non-negative as well
    cons.append(-Z)

    # Enforce that all criteria weights sum to 1 within a small tolerance
    sum_criteria_weights = pt.sum(x[:num_criteria])
    tol = 0 #1e-8
    cons.append(pt.abs(sum_criteria_weights - 1.0) - tol)

    return pt.stack(cons)

def run_mcmc_sampling(dict_data, num_criteria, num_groups, first_run_error):
    import pymc as pm
    import numpy as np

    # Run MCMC to get posterior samples
    with pm.Model() as model:
        # Define your model (weights, Z, constraints, etc.)
        w_crit = pm.Dirichlet("w_crit", a=np.ones(num_criteria))
        w_group = pm.Dirichlet("w_group", a=np.ones(num_groups))
        Z = pm.Uniform("Z", lower=0, upper=first_run_error)
        x = pm.math.concatenate([w_crit, w_group, Z[None]])
        cons = pytensor_constraints_func(x, dict_data, Z_max=first_run_error)
        pm.Potential("constraints", pt.switch(pt.any(pt.gt(cons, 1e-6)), -np.inf, 0.0))

        trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.9, random_seed=42)

    import numpy as np
    import pandas as pd

    # Extract the samples for each variable
    w_crit_samples = trace.posterior["w_crit"].to_numpy()  # shape: (chains, draws, num_criteria)
    w_group_samples = trace.posterior["w_group"].to_numpy()  # shape: (chains, draws, num_groups)
    Z_samples = trace.posterior["Z"].to_numpy()  # shape: (chains, draws)

    # Reshape to (total_samples, ...)
    w_crit_samples = w_crit_samples.reshape(-1, num_criteria)
    w_group_samples = w_group_samples.reshape(-1, num_groups)
    Z_samples = Z_samples.reshape(-1, 1)

    # Combine into a single array: shape = (total_samples, num_criteria + num_groups + 1)
    posterior_samples = np.concatenate([w_crit_samples, w_group_samples, Z_samples], axis=1)

    # Analyze posterior samples directly without saving to a file
    count_better = sum(1 for sample in posterior_samples if sample[-1] < first_run_error)
    count_sum_one = sum(1 for sample in posterior_samples if abs(sum(sample[:num_criteria]) - 1) < 1e-6)

    total_samples = len(posterior_samples)
    print(f"Number of samples with z < {first_run_error}: {count_better} out of {total_samples}")
    print(f"Number of samples with sum of criteria weights == 1: {count_sum_one} out of {total_samples}")

    # apply constraints to each row in posterior_samples.csv and count how many satisfy them
    count_constraints = 0
    for sample in posterior_samples:
        weights = sample[:num_criteria]
        # Example constraints: weight[0] >= weight[1], weight[2] + weight[3] <= 0.5
        if weights[0] >= weights[1] and (weights[2] + weights[3] <= 0.5):
            count_constraints += 1
    print(f"Number of samples satisfying constraints: {count_constraints} out of {total_samples}")

    # Select only those that satisfy all these three conditions and delete the others
    valid_samples = []
    for sample in posterior_samples:
        weights = sample[:num_criteria]
        z_value = sample[-1]
        if (abs(sum(weights) - 1) < 1e-6 and
            z_value < first_run_error and
            weights[0] >= weights[1] and
            (weights[2] + weights[3] <= 0.5)):
            valid_samples.append(sample)

    # Convert to numpy array for further analysis if needed
    valid_samples = np.array(valid_samples)

    # valid_samples now contains only the samples that satisfy all conditions
    return valid_samples