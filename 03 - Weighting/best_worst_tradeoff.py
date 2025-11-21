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
    return 