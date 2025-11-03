import numpy as np
import scipy.optimize as opt

# Define the objective function: minimize z
def objective(x):
    return x[-1]  # z is the last variable

# Define the constraints
def constraints(x, v_f, x_w, x_b, criteria_count):
    w = x[:criteria_count]  # w_0, w_1, w_2, w_3
    z = x[-1]               # z

    cons = []
    # Constraints for worst: |w[criteria_count-1] * v_f[i](x_w[i]) - w[i]| <= z
    for i in range(criteria_count - 1):  # Exclude worst (index criteria_count-1)
        if not np.isnan(x_w[i]):
            cons.append(z - abs(w[criteria_count-1] * v_f[i](x_w[i]) - w[i]))

    # Constraints for best: |w[0] - w[i] / v_f[i](x_b[i])| <= z
    for i in range(1, criteria_count):  # Exclude best (index 0)
        if not np.isnan(x_b[i]):
            cons.append(z - abs(w[0] - w[i] / v_f[i](x_b[i])))

    # Sum of w_i = 1
    cons.append(np.sum(w) - 1)

    # w_i >= 0
    for wi in w:
        cons.append(wi)

    return cons

def bwt(value_functions, worst_comparison_results, best_comparison_results):
    criteria_count = len(value_functions) # aggiungi controllo di lunghezza
    # Initial guess: criteria_count w_i's, z
    x0 = np.ones(criteria_count + 1)  # criteria_count w_i's, z

    # Bounds: w_i >= 0, z >= 0
    bounds = [(0, None) for _ in range(criteria_count)] + [(0, None)]

    # Solve
    print("Starting optimization...")
    result = opt.minimize(
        objective,
        x0,
        method='SLSQP',
        constraints={'type': 'ineq', 'fun': constraints, 'args': (value_functions, worst_comparison_results, best_comparison_results, criteria_count)},
        bounds=bounds
    )
    return result