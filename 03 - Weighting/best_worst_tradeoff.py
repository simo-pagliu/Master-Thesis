import numpy as np
import scipy.optimize as opt

# Define the constraints
def constraints(x, v_f, x_w, x_b, criteria_count):
    w = x[:criteria_count]  # w_0, w_1, w_2, w_3
    z = x[-1]               # z

    cons = []
    # BEST
    # all best comparisons are made against the best criterion ---> v_f[0]
    for i in range(1, criteria_count):  # Exclude best (index 0)
        cons.append(z - abs(w[0] - w[i] / v_f[0](x_b[i])))

    # Constraints for worst: |w[criteria_count-1] * v_f[i](x_w[i]) - w[i]| <= z
    # worst comparisons are made based on the worst criterion by adjusting another one --> v_f[i]
    for i in range(criteria_count - 1):  # Exclude worst (index criteria_count-1)
        cons.append(z - abs(w[-1] * v_f[i](x_w[i]) - w[i]))



    # Sum of w_i = 1
    cons.append(np.sum(w) - 1)

    # w_i >= 0
    for wi in w:
        cons.append(wi)

    return cons

def constraints_nl(x, v_f, x_w, x_b, criteria_count):
    w = x[:criteria_count]  # w_0, w_1, w_2, w_3
    z = x[-1]               # z

    cons = []

    # BEST 
    # all best comparisons are made against the best criterion ---> v_f[0]
    for i in range(1, criteria_count):  # Exclude best (index 0)
        cons.append(z - abs( 1/v_f[0](x_b[i]) - w[0]/w[i]))

    # WORST
    # worst comparisons are made based on the worst\ criterion by adjusting another one --> v_f[i]
    for i in range(criteria_count - 1):  # Exclude worst (index criteria_count-1)
        cons.append(z - abs( v_f[i](x_w[i]) - w[i]/w[-1] ))

    # Sum of w_i = 1
    cons.append(np.sum(w) - 1)

    # w_i >= 0
    for wi in w:
        cons.append(wi)

    return cons

# Minimization problem, objective is the auxiliary variable z
def objective(x):
    return x[-1]  # z is the last variable in the array
    
def bwt(criteria, worst_comparison_results, best_comparison_results, linear=True):
    # Extract needed variables
    value_functions = [c["value_function"] for key, c in criteria.items()]
    criteria_count = len(value_functions)

    # Initial guess: for all w_i + z (auxiliary variable)
    x0 = np.ones(criteria_count + 1)/criteria_count

    # Bounds: 
    # Positive weights: w_i >= 0, 
    # Positive auxiliary variable z >= 0
    bounds = [(0, None) for _ in range(criteria_count)] + [(0, None)]

    # Linear or Non-Linear constraints
    if linear:
        constraints_func = constraints
    else:
        constraints_func = constraints_nl

    # Solve optimization problem
    print("Starting optimization...")
    result = opt.minimize(
        objective, # Function to minimize
        x0, # Initial Guess
        method='SLSQP', # Method
        # Constraints dictionary:
            # They are inequality constraints in type
            # Function defining the constraints (see above) in fun
            # Args required by the constraints function are passed in args
        constraints={'type': 'ineq', 'fun': constraints_func, 'args': (value_functions, worst_comparison_results, best_comparison_results, criteria_count)
            },
        bounds=bounds
    )

    # Modify result.x to only contain weights is a dictionary with criteria names as keys and weights as values
    weights = result.x[:criteria_count]
    weights_dict = {key: weights[i] for i, key in enumerate(criteria.keys())}
    result.x = weights_dict

    return result

def consistency_check(criteria, result,worst_comparison_results, best_comparison_results):
    # Compute values of best and worst comparisons
    keys = list(result.x.keys())
    BC_values = []
    WC_values = []

    # BEST
    for i in range(1, len(keys)):  # Exclude best (index 0)
        BC_values.append(criteria[keys[0]]["value_function"](best_comparison_results[i]))

    # WORST
    for i in range(len(keys) - 1):  # Exclude worst (index criteria_count-1)
        WC_values.append(criteria[keys[i]]["value_function"](worst_comparison_results[i]))

    # Create a table of results with columns: Criterion, Weight, BC_value, WC_value
    results_table = {}
    for i, key in enumerate(keys):
        wc_value = WC_values[i] if i < len(WC_values) else None
        bc_value = BC_values[i-1] if i > 0 else None
        results_table[key] = {
            "Weight": result.x[key],
            "BC_value": bc_value,
            "WC_value": wc_value
        }

    return results_table

# def consistency_check_liang(criteria, result,worst_comparison_results, best_comparison_results):
#     """
#     Consistency check based on Liang et al. (2022) 
#     DOI: https://doi.org/10.1016/j.ins.2022.07.097
#     """

#     # Extract the keys from result.x
#     keys = list(result.x.keys())

#     # Ordinal Ratio calculation
#     Ordinal_Ratio = []
#     for j, key_j in enumerate(keys):
#         OR_j = 0
#         for k, key_k in enumerate(keys):
#             gamma = \
#                 criteria[0]["value_function"](best_comparison_results[k]) \
#                 - criteria[0]["value_function"](best_comparison_results[j])
#             delta = \
#                 1/criteria[k]["value_function"](worst_comparison_results[j]) \
#                 - 1/criteria[k]["value_function"](worst_comparison_results[k])
            
#             if gamma * delta < 0:
#                 f_gamma_delta = 1
#             elif gamma * delta == 0 and (gamma != 0 or delta != 0):
#                 f_gamma_delta = 0.5
#             else:
#                 f_gamma_delta = 0
            
#             OR_j += f_gamma_delta
#         OR_j = OR_j / (len(keys) - 1)
#         Ordinal_Ratio.append(OR_j)


#     # Create result dictionary
#     output = {
#         "Ordinal_Ratio": Ordinal_Ratio
#     }
#     return output