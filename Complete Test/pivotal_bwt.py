import numpy as np
import scipy.optimize as opt

def constraints_func(x, dict_data, z_known=None):
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

    # Total number of weights (excluding z)
    tot_number = current_index
    z_index = tot_number  # z is the last element in x

    # Iterate over groups for intra-group constraints
    for group_name, group_data in dict_data.items():
        criteria_in_group = group_indices[group_name]['criteria']
        w_start = group_indices[group_name]['start']
        w_end = group_indices[group_name]['end']
        w = x[w_start:w_end]  # Weights for this group
        z = x[z_index]        # z is the last element

        # INTRA-GROUP BEST COMPARISONS
        for crit, comparisons in group_data['criteria'].items():
            v_f = comparisons['value_function']
            for other_crit, value in comparisons['best_comparisons'].items():
                i = criteria_in_group.index(crit)
                j = criteria_in_group.index(other_crit)
                v_f_other = group_data['criteria'][crit]['value_function']  # Use value function of the reference criterion
                cons.append(z - abs(w[i] / w[j] - 1 / v_f_other(value)))

            # INTRA-GROUP WORST COMPARISONS
            for other_crit, value in comparisons['worst_comparisons'].items():
                i = criteria_in_group.index(crit)
                j = criteria_in_group.index(other_crit)
                v_f_other = group_data['criteria'][other_crit]['value_function']  # Use value function of the other criterion
                cons.append(z - abs(v_f_other(value) - w[i] / w[j]))

    # Aggregate inter-group comparisons
    intraB = {}
    intraW = {}
    for group_data in dict_data.values():
        intraB.update(group_data['intraB'])
        intraW.update(group_data['intraW'])

    w_G = x[tot_number:tot_number + len(dict_data)]  # Weights for groups

    # BEST-BEST and BEST-WORST COMPARISONS
    for crit_name, comparison in intraB.items():
        ref_crit = comparison['reference']
        other_crit = comparison['other']
        ref_group = next(group_name for group_name, group_data in dict_data.items() if ref_crit in group_data['criteria'])
        other_group = next(group_name for group_name, group_data in dict_data.items() if other_crit in group_data['criteria'])
        v_f_other = dict_data[ref_group]['criteria'][ref_crit]['value_function']  # Use value function of the reference criterion
        cons.append(z - abs(w_G[0] / w_G[1] - 1 / v_f_other(comparison['value'])))

    # WORST-BEST and WORST-WORST COMPARISONS
    for crit_name, comparison in intraW.items():
        ref_crit = comparison['reference']
        other_crit = comparison['other']
        ref_group = next(group_name for group_name, group_data in dict_data.items() if ref_crit in group_data['criteria'])
        other_group = next(group_name for group_name, group_data in dict_data.items() if other_crit in group_data['criteria'])
        v_f_other = dict_data[other_group]['criteria'][other_crit]['value_function']  # Use value function of the other criterion
        cons.append(z - abs(v_f_other(comparison['value']) - w_G[1] / w_G[-1]))

    # Non-negativity constraints for weights
    for wi in x[:-1]:
        cons.append(wi)

    if z_known is not None:
        # z shall be lower than or equal to known z
        cons.append(z_known - x[-1])

    # Sum of weights = 1
    cons.append(np.sum(x[:num_criteria]) - 1)

    return cons

# Minimization problem, objective is the auxiliary variable z
def objective(x):
    return x[-1]  # z is the last variable in the array

def bwt(dict_data):
    # Count the number of groups and criteria
    num_groups = len(dict_data)
    num_criteria = sum(len(group_data['criteria']) for group_data in dict_data.values())

    # Total number of weights (criteria weights + group weights + auxiliary variable z)
    tot_number = num_criteria + num_groups

    # Initial guess: for all w_i + z (auxiliary variable)
    x0 = np.ones(tot_number + 1) / (tot_number + 1)

    # Bounds: 
    # Positive weights: w_i >= 0, 
    # Positive auxiliary variable z >= 0
    bounds = [(0, None) for _ in range(tot_number)] + [(0, None)]

    # Solve optimization problem
    print("Starting optimization...")
    result = opt.minimize(
        objective, # Function to minimize
        x0, # Initial Guess
        method='SLSQP', # Method
        constraints={'type': 'ineq', 'fun': constraints_func, 'args': (dict_data,)},
        bounds=bounds
    )

    # Extract weights for criteria and groups
    weights = result.x[:num_criteria]
    weights_dict = {}
    current_index = 0
    for group_name, group_data in dict_data.items():
        group_criteria = list(group_data['criteria'].keys())
        weights_dict[group_name] = {
            crit: weights[current_index + i] for i, crit in enumerate(group_criteria)
        }
        current_index += len(group_criteria)

    # Extract weights for groups
    group_weights = result.x[num_criteria:num_criteria + num_groups]
    group_weights_dict = {
        group_name: group_weights[i] for i, group_name in enumerate(dict_data.keys())
    }

    # Add auxiliary variable z
    z = result.x[-1]

    return {
        'solver_result': result,
        'criteria_weights': weights_dict,
        'group_weights': group_weights_dict,
        'z': z
    }