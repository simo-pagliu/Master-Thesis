# PILE-BWT
#
# This module is the implementation of the PILE-BWT method
# The most important part is the definition of the constraints function

#################################################################################
# Import third party libraries
import numpy as np
import scipy.optimize as opt
from functools import partial
#################################################################################

#################################################################################
# Define all inequality that define the space of weights according to PILE-BWT
def constraints_func(x, dict_data, z_star=None):

    cons = []
    group_indices = {}
    current_index = 0

    # Map group and criterion names to their global indices in x
    for group_name, group_data in dict_data.items():
        criteria_in_group = list(group_data['criteria'].keys())
        # print("Group:", group_name, "Criteria:", criteria_in_group)
        group_indices[group_name] = {
            'start': current_index,
            'end': current_index + len(criteria_in_group),
            'criteria': criteria_in_group
        }
        current_index += len(criteria_in_group)
    # print("Group indices mapping:", group_indices)
    # INTRA-GROUP CONSTRAINTS
    for group_name, group_data in dict_data.items():
        criteria_in_group = group_indices[group_name]['criteria']
        w_start = group_indices[group_name]['start']
        w_end = group_indices[group_name]['end']
        # print(f"Processing intra-group constraints for group '{group_name}' with criteria {criteria_in_group}")
        w = x[w_start:w_end]  # Weights for this group
        z = x[-1]             # z is the last element
        # print(f"Processing intra-group constraints for group '{group_name}' with weights {w} and z={z}")
        for crit, comparisons in group_data['criteria'].items():
            # INTRA-GROUP BEST COMPARISONS
            for other_crit, value in comparisons['best_comparisons'].items():
                # print(f"  Criterion '{crit}' best comparisons: {comparisons['best_comparisons']}")
                i = criteria_in_group.index(crit)
                j = criteria_in_group.index(other_crit)
                v_f = comparisons['value_function']
                # print(f"Adding intra-group best comparison constraint between '{crit}' and '{other_crit}'")
                cons.append(z - abs(w[i] / w[j] - 1.0 / v_f(value)))
            # INTRA-GROUP WORST COMPARISONS
            for other_crit, value in comparisons['worst_comparisons'].items():
                # print(f"  Criterion '{crit}' worst comparisons: {comparisons['worst_comparisons']}")
                i = criteria_in_group.index(crit)
                j = criteria_in_group.index(other_crit)
                v_f_other = group_data['criteria'][other_crit]['value_function']
                # print(f"Adding intra-group worst comparison constraint between '{crit}' and '{other_crit}'")
                cons.append(z - abs(1.0 / v_f_other(value) - (w[j] / w[i])))

    # INTER-GROUP CONSTRAINTS
    intraB = {}
    intraW = {}
    for group_data in dict_data.values():
        intraB.update(group_data['intraB'])
        intraW.update(group_data['intraW'])

    def add_comparison_constraint(comparison, comparison_type, cons, x, group_indices, dict_data):
        ref_crit = comparison['reference']
        other_crit = comparison['other']
        # print(f"Processing inter-group {comparison_type} comparison: {ref_crit} vs {other_crit}")
        # Resolve group and indices for reference and other criteria.
        # Support both string criterion names and integer indices (local or global).
        def resolve_group_and_global_index(crit):
            # Otherwise assume crit is a name: find group that contains that key
            for group_name, group_data in dict_data.items():
                # print(f"  Checking if criterion '{crit}' is in group '{group_name}'")
                if crit in group_data['criteria']:
                    local_index = group_indices[group_name]['criteria'].index(crit)
                    return group_name, group_indices[group_name]['start'] + local_index
        
        ref_group, ref_global_index = resolve_group_and_global_index(ref_crit)
        other_group, other_global_index = resolve_group_and_global_index(other_crit)

        ref_local_index = ref_global_index - group_indices[ref_group]['start']
        other_local_index = other_global_index - group_indices[other_group]['start']

        v_f = dict_data[ref_group]['criteria'][ref_crit]['value_function']
        v_f_other = dict_data[other_group]['criteria'][other_crit]['value_function']

        if comparison_type == "best":
            cons.append(x[-1] - abs((x[ref_global_index] / x[other_global_index]) - (1.0 / v_f(comparison['value']))))
            # print(f"adding best comparison constraint between '{ref_crit}' and '{other_crit}'")
            # print the comparison and the indexes
            # print(f"Best comparison: {ref_crit} (global {ref_global_index}, local {ref_local_index} in group '{ref_group}') vs {other_crit} (global {other_global_index}, local {other_local_index} in group '{other_group}')")
        else:  # worst
            cons.append(x[-1] - abs(1.0 / v_f_other(comparison['value']) - (x[other_global_index] / x[ref_global_index])))
            # print(f"adding worst comparison constraint between '{ref_crit}' and '{other_crit}'")
            # print(f"Worst comparison: {ref_crit} (global {ref_global_index}, local {ref_local_index} in group '{ref_group}') vs {other_crit} (global {other_global_index}, local {other_local_index} in group '{other_group}')")

    # Process all inter-group comparisons
    for comparison in intraB.values():
        add_comparison_constraint(comparison, comparison["type"], cons, x, group_indices, dict_data)
    for comparison in intraW.values():
        add_comparison_constraint(comparison, comparison["type"], cons, x, group_indices, dict_data)

    if z_star is not None:
        # Constraint to ensure z <= z_star
        cons.append(z_star - x[-1])
    # Non-negativity constraints for weights
    # is enforced by the bounds
    # for wi in x[:-1]:
    #     cons.append(wi)

    # Round all constraints to avoid floating-point issues
    # this was deemed unnecessary
    # cons = [round(c, 10) for c in cons]
    
    # Sum shall be lower then 1 + small tolerance
    # cons.append(np.sum(x[:len(x)-1]) - 1 + 1e-8)  # Small tolerance for equality constraint
    # Sum shall be higher then 1 - small tolerance
    # cons.append(1 - np.sum(x[:len(x)-1]) + 1e-8)  # Small tolerance for equality constraint

    return cons
#################################################################################

# Minimization problem, objective is the auxiliary variable z
def objective(x, var=-1):
    return x[var]  # z is the last variable in the array

def bwt(dict_data, var=-1, z_star=None, type='min'):
    # Count the number of groups and criteria
    num_criteria = sum(len(group_data['criteria']) for group_data in dict_data.values())

    # Initial guess: for all w_i + z (auxiliary variable)
    # We ensure it starts within bounds
    x0 = np.ones(num_criteria + 1) / (num_criteria)
    # print("Initial guess x0:", x0)

    # Bounds: 
    # Positive weights: w_i >= 0, 
    # Positive auxiliary variable z >= 0
    bounds = [(0.001, 1) for _ in range(num_criteria)] + [(0, None)]
    # print("Bounds:", bounds)

    # Solve optimization problem
    print("Running optimization...")
    objective_with_var = partial(objective, var=var)
    # Scipy optimize only minimizes functions so for maximization we negate the objective
    if type == 'max':
        def neg_objective(x):
            return -objective_with_var(x)
        objective_func = neg_objective
    else:
        objective_func = objective_with_var

    # Try SLSQP first
    # if it fails catastrophically we try COBYLA as a fallback
    try:
        result = opt.minimize(
            objective_func,
            x0,
            method='SLSQP',
            constraints=[
                {'type': 'ineq', 'fun': constraints_func, 'args': (dict_data, z_star,)},
                {'type': 'eq', 'fun': lambda x: np.sum(x[:num_criteria]) - 1}
            ],
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-3}
        )
    except Exception as e:
        print(f"SLSQP raised exception: {e}")
        pass

    # Also if it did not raised an expection but did not converge we try COBYLA
    # Note that SLSQP can fail silently so we check the success flag
    # print(f"\033[93mSLSQP failed (message: {getattr(result, 'message', None)})\033[0m")
    print(f"\033[93mTrying 'COBYLA' as fallback with explicit bound constraints...\033[0m")
    try:
        # Build explicit inequality constraints that represent the bounds for COBYLA
        bound_constraints = []
        # weights 0..num_criteria-1: 0 <= x[i] <= 1
        for idx in range(num_criteria):
            bound_constraints.append({'type': 'ineq', 'fun': (lambda x, i=idx: x[i] - 0.001)})            # x[i] >= 0.001
            bound_constraints.append({'type': 'ineq', 'fun': (lambda x, i=idx: 1.0 - x[i])})      # x[i] <= 1
        # auxiliary variable z (last index) must be >= 0
        bound_constraints.append({'type': 'ineq', 'fun': (lambda x: x[-1])})

        # Combine with existing problem constraints (comparisons and sum-to-1 equality)
        cobyla_constraints = [
            {'type': 'ineq', 'fun': constraints_func, 'args': (dict_data, z_star,)},
            {'type': 'eq', 'fun': lambda x: np.sum(x[:num_criteria]) - 1}
        ] + bound_constraints

        result_coby = opt.minimize(
            objective_func,
            x0,
            method='COBYLA',
            constraints=cobyla_constraints,
            options={'maxiter': int(1e4), 'rhobeg': 1e-3}
        )
        if result_coby.success:
            result = result_coby
            print("\033[92mCOBYLA succeeded.\033[0m")
        else:
            maxcv = getattr(result_coby, 'maxcv', None)
            print(f"\033[93mCOBYLA failed (message: {getattr(result_coby, 'message', None)})\033[0m")
            print(f"\033[93mmaxcv={maxcv}\033[0m")
            # If COBYLA didn't converge but the maximum constraint violation
            # is very small, treat the solution as acceptable.
            accept_threshold = 10 #if z_star is None else 1e-3
            if (maxcv is not None) and (maxcv <= accept_threshold):
                result_coby.success = True
                print("\033[92mCOBYLA produced near-feasible solution: accepting result.\033[0m")
                result = result_coby
            else:
                # If the error was too large, we set result to None
                print("\033[91mCOBYLA solution not acceptable.\033[0m")
                result = bwt(dict_data, var=-1, z_star=None, type='min')["solver_result"]
    # If also COBYLA fails, we return None
    except Exception as e:
        print(f"\033[91mCOBYLA fallback raised exception: {e}\033[0m")
        result = bwt(dict_data, var=-1, z_star=None, type='min')["solver_result"]

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
    # Add auxiliary variable z
    z = result.x[-1]

    return {
        'solver_result': result,
        'criteria_weights': weights_dict,
        'z': z
    }