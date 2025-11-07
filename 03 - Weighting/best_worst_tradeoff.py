import numpy as np
import scipy.optimize as opt

# Define the constraints
def constraints(x, v_f, x_w, x_b, criteria_count):
    w = x[:criteria_count]  # w_0, w_1, w_2, w_3
    z = x[-1]               # z

    cons = []
    # Constraints for worst: |w[criteria_count-1] * v_f[i](x_w[i]) - w[i]| <= z
    # worst comparisons are made based on the worst criterion by adjusting another one --> v_f[i]
    for i in range(criteria_count - 1):  # Exclude worst (index criteria_count-1)
        cons.append(z - abs(w[-1] * v_f[i](x_w[i]) - w[i]))

    # Constraints for best: |w[0] - w[i] / v_f[0](x_b[i])| <= z
    # all best comparisons are made against the best criterion ---> v_f[0]
    for i in range(1, criteria_count):  # Exclude best (index 0)
        cons.append(z - abs(w[0] - w[i] / v_f[0](x_b[i])))

    # Sum of w_i = 1
    cons.append(np.sum(w) - 1)

    # w_i >= 0
    for wi in w:
        cons.append(wi)

    return cons

# Minimization problem, objective is the auxiliary variable z
def objective(x):
    return x[-1]  # z is the last variable in the array
    
def bwt(criteria, worst_comparison_results, best_comparison_results):
    # Extract needed variables
    value_functions = [c["value_function"] for c in criteria]
    criteria_count = len(value_functions)

    # Initial guess: for all w_i + z (auxiliary variable)
    x0 = np.ones(criteria_count + 1)

    # Bounds: 
    # Positive weights: w_i >= 0, 
    # Positive auxiliary variable z >= 0
    bounds = [(0, None) for _ in range(criteria_count)] + [(0, None)]

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
        constraints={'type': 'ineq', 'fun': constraints, 'args': (value_functions, worst_comparison_results, best_comparison_results, criteria_count)
            },
        bounds=bounds
    )
    return result