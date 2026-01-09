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
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
import scipy.optimize as opt
#################################################################################

#################################################################################
# Import internal modules
from up_mavt import startup
from constraints import constraints_func
#################################################################################

#################################################################################
# USER INPUTS
# Weight elicitation files (one per elicitation / run)
file_path_weight_elicitations = ["./weight_spaces/BWT_results_1.csv"]

# Value function files (one per elicitation / run)
file_path_value_functions = ["value_functions_1.csv"]

# Criteria definitions file
file_path_criteria = "criteria.csv"
#################################################################################

for vf_file, w_file in zip(file_path_value_functions, file_path_weight_elicitations):

    elicitation_number = file_path_value_functions.index(vf_file) + 1
    print(f"\n\n=== Processing Elicitation #{elicitation_number} ===")

    #################################################################################
    # Startup: Load all data
    dict_data_list, crit_index, vf_list, alternatives = startup(file_path_criteria, file_path_weight_elicitations, file_path_value_functions)
    dict_data = dict_data_list[0]  # Use the first elicitation for testing
    n_alternatives = len(alternatives)
    print(f"Loaded {len(dict_data_list)} elicitation(s) with {n_alternatives} alternatives.")
    #################################################################################

    def objective(x, var=-1):
        return x[var]  # z is the last variable
    def max_constraint_violation(x, dict_data, z_star=None):
        cons = constraints_func(x, dict_data, z_star)
        return max(0, -min(cons)) if cons else 0
    def global_optimization(dict_data, var=-1, z_star=None):
        num_criteria = sum(len(group_data['criteria']) for group_data in dict_data.values())
        # Use a finite upper bound for z (last variable) because SciPy requires finite bounds
        z_upper = 1000.0
        bounds = [(0.001, 1) for _ in range(num_criteria)] + [(0.0, z_upper)]
        def objective_for_de(x):
            # Penalize constraint violations and deviation from sum-to-one for weights
            penalty = 0.0
            # Large penalty multiplier to strongly discourage infeasible solutions
            PEN = 1e6
            penalty += PEN * max_constraint_violation(x, dict_data, z_star)
            penalty += PEN * abs(np.sum(x[:num_criteria]) - 1.0)
            return x[var] + penalty
        # Use penalty-based objective because some SciPy versions don't accept
        # constraint dicts for `differential_evolution`.
        result_de = opt.differential_evolution(
            objective_for_de,
            bounds,
            maxiter=1000,
            popsize=20,
            tol=1e-3,
            polish=True,
            seed=42
        )
        return result_de

    def find_all_solutions(dict_data, z_star=None, num_restarts=150, eps=0.001, rng_seed=42):
        num_criteria = sum(len(group_data['criteria']) for group_data in dict_data.values())
        # Finite bound for z to satisfy optimizer requirements
        z_upper = 1000.0
        bounds = [(0.001, 1) for _ in range(num_criteria)] + [(0.0, z_upper)]
        # Step 1: Find global optimum
        global_result = global_optimization(dict_data, z_star=z_star)
        max_violation_opt = max_constraint_violation(global_result.x, dict_data, z_star)
        solutions = [global_result.x]
        # Step 2: Multi-start local optimization to find all solutions within tolerance
        rng = np.random.RandomState(rng_seed)
        for _ in range(num_restarts):
            x0_random = rng.uniform(0.001, 1, size=num_criteria + 1)
            x0_random[-1] = 0.1  # Initial guess for z
            res = opt.minimize(
                objective,
                x0_random,
                method='SLSQP',
                constraints=[
                    {'type': 'ineq', 'fun': constraints_func, 'args': (dict_data, z_star)},
                    {'type': 'eq', 'fun': lambda x: np.sum(x[:num_criteria]) - 1}
                ],
                bounds=bounds,
                options={'maxiter': 1000}
            )
            if max_constraint_violation(res.x, dict_data, z_star) <= max_violation_opt + eps:
                solutions.append(res.x)
        return {
            'global_optimum': global_result,
            'all_solutions': solutions,
            'max_violation_opt': max_violation_opt
        }

    ##################################################################

    print("Starting DE-based solution search...")

    temporary_save_file = 'temp.csv'
    import pandas as pd
    tot_number_solutions = 0
    required_solutions = 100
    # Example usage:
    while tot_number_solutions < required_solutions:
        s = int(np.random.uniform(0, 1e6))
        print(f'Running with RNG seed: {s}')
        results = find_all_solutions(dict_data, z_star=None, rng_seed=s)
        # Build weights array (n_solutions x n_criteria), drop last element (z)
        raw_weights = np.array([sol[:-1] for sol in results['all_solutions'][1:]])
        # Round to 3 decimals (0.001) and deduplicate
        round_dec = 3
        rounded = np.round(raw_weights, round_dec)
        # Find unique rounded rows preserving first occurrence order
        _, uniq_idx = np.unique(rounded, axis=0, return_index=True)
        uniq_idx = np.sort(uniq_idx)
        unique_weights = rounded[uniq_idx]
        # Retrieve corresponding z values from original solutions (keep original z without rounding)
        orig_sols = results['all_solutions']#[1:]
        unique_z = [orig_sols[i][-1] for i in uniq_idx]
        n_solutions, n_criteria = unique_weights.shape
        crit_names = [f'Criterion {i+1}' for i in range(n_criteria)]
        df = pd.DataFrame(unique_weights, columns=crit_names)
        # Filter out solutions with z values too high (e.g., z > min(z) + 0.1)
        # Only save those within a reasonable threshold of the best found z
        z_values = np.array(unique_z)
        threshold = z_values.min() + 0.1
        filtered_indices = np.where(z_values <= threshold)[0]
        unique_weights = unique_weights[filtered_indices]
        unique_z = z_values[filtered_indices]
        n_solutions = unique_weights.shape[0]
        print(f'Total solutions found: {len(results["all_solutions"]) - 1}; Unique (rounded to 0.001): {n_solutions}')
        # Save unique rounded solutions to CSV (rounded to 0.001 resolution)
        # Append to file if it exists, otherwise create it
        file_exists = os.path.exists(temporary_save_file)
        with open(temporary_save_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                header = [f'Criterion {i+1}' for i in range(n_criteria)] + ['z']
                writer.writerow(header)
            # write rows corresponding to unique_weights + their z values
            for w, z in zip(unique_weights, unique_z):
                row = list(w) + [z]
                writer.writerow(row)
        tot_number_solutions += len(unique_weights)
        print(f'Appended {len(unique_weights)} unique solutions to {temporary_save_file}, total so far: {tot_number_solutions}.')


    # read csv possible_solutions_unique_0.001.csv and keep only unique rows, print how many removed
    with open(temporary_save_file, mode='r') as file:
        csv_reader = csv.reader(file)
        rows = list(csv_reader)

    def _is_float(s):
        try:
            float(s)
            return True
        except Exception:
            return False

    # detect & remove header if present (non-numeric entries in first row)
    if rows and any(not _is_float(cell.strip()) for cell in rows[0]):
        rows = rows[1:]

    initial_count = len(rows)
    seen = set()
    unique_rows = []
    for row in rows:
        tup = tuple(float(v) for v in row)
        if tup in seen:
            continue
        seen.add(tup)
        unique_rows.append(list(tup))

    removed = initial_count - len(unique_rows)
    possible_solutions = unique_rows

    print(f"Removed {removed} duplicate rows. Kept {len(possible_solutions)} unique rows.")        

    # Check if all rows of the csv sum to 1 (exclude last column)
    removed = 0
    for row in possible_solutions:
        weight_sum = sum(row[:-1])
        if not np.isclose(weight_sum, 1.0, atol=1e-3):
            # Remove the row if it does not sum to 1
            possible_solutions.remove(row)
            removed += 1
    print(f"Removed {removed} rows that did not sum to 1.")


    # Check that all remaining rows satisfy the constraints
    violations = 0
    for row in possible_solutions:
        x_temp = np.concatenate((row[:-1], [0]))
        constraints_satisfied = constraints_func(x_temp, dict_data)
        if not constraints_satisfied:
            violations += 1
    print(f"Found {violations} rows that violate the constraints.")

    # Save filtered solutions to the final weights CSV file
    import pandas as pd
    df = pd.DataFrame(possible_solutions)
    df.columns = [f'Criterion {i+1}' for i in range(df.shape[1] - 1)] + ['z']
    
    # Save to file
    output_file = "./weights_" + str(elicitation_number) + ".csv"
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} filtered solutions to {output_file}")

    # Plot
    plt.figure(figsize=(12, 6))
    sns.stripplot(data=df, orient='h', color='b', size=3, jitter=0.25, alpha=0.6)
    plt.xlabel('Weight value')
    plt.title('Weight distributions per criterion')
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.show()

#################################################################################
# DELETE temporary file
if os.path.exists(temporary_save_file):
    os.remove(temporary_save_file)
#################################################################################