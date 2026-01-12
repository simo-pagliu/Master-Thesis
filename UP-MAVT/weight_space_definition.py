"""
Weight space definition module using constraint-based optimization.

Provides functionality to define and load weight spaces for decision analysis.
"""

import numpy as np
import pandas as pd
import os
import csv
import scipy.optimize as opt
from constraints import constraints_func

def weight_sampler(dict_data, weight_space):
    weight_set_found = False
    while weight_set_found is False:
        weight_set_candidate = []
        for criteria_weight in weight_space:
            # Randomly sample a weight
            w = np.random.choice(criteria_weight)
            weight_set_candidate.append(w)
        # Check if they sum to 1
        total_weight = sum(weight_set_candidate)
        if not np.isclose(total_weight, 1.0, atol=1e-3):
            continue
        else:
            # Check if they satisfy the constraints
            x_temp = np.concatenate((weight_set_candidate, [0]))
            constraints_satisfied = constraints_func(x_temp, dict_data)
            # print(f"Sampled weights: {weight_set_candidate}, sum: {total_weight}, constraints satisfied: {constraints_satisfied}")
            if not constraints_satisfied:
                continue
            else:
                weight_set_found = True
    return weight_set_candidate

def max_constraint_violation(x, dict_data, z_star=None):
    """Calculate maximum constraint violation for a solution."""
    cons = constraints_func(x, dict_data, z_star)
    return max(0, -min(cons)) if cons else 0


def find_all_solutions(dict_data, z_star=None, num_restarts=150, eps=0.001, rng_seed=42):
    """Find all feasible weight solutions using differential evolution + multi-start local optimization."""
    num_criteria = sum(len(group_data['criteria']) for group_data in dict_data.values())
    z_upper = 1000.0
    bounds = [(0.001, 1) for _ in range(num_criteria)] + [(0.0, z_upper)]
    
    # Step 1: Find global optimum using differential evolution
    def objective_for_de(x):
        penalty = 0.0
        PEN = 1e6
        penalty += PEN * max_constraint_violation(x, dict_data, z_star)
        penalty += PEN * abs(np.sum(x[:num_criteria]) - 1.0)
        return x[-1] + penalty
    
    result_de = opt.differential_evolution(
        objective_for_de,
        bounds,
        maxiter=1000,
        popsize=20,
        tol=1e-3,
        polish=True,
        seed=rng_seed
    )
    max_violation_opt = max_constraint_violation(result_de.x, dict_data, z_star)
    solutions = [result_de.x]
    
    # Step 2: Multi-start local optimization
    def objective(x, var=-1):
        return x[var]
    
    rng = np.random.RandomState(rng_seed)
    for _ in range(num_restarts):
        x0_random = rng.uniform(0.001, 1, size=num_criteria + 1)
        x0_random[-1] = 0.1
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
    
    return solutions


def define_weight_spaces(dict_data_list, elicitation_numbers, script_dir, required_solutions=100):
    """
    Define weight spaces for each elicitation.
    
    If BWT_results_N_weights.csv exists, load it. Otherwise, generate it.
    
    Args:
        dict_data_list: List of loaded criteria/constraint data
        elicitation_numbers: List of elicitation numbers (e.g., [1, 2])
        script_dir: Directory containing elicitation results
        required_solutions: Target number of unique weight combinations to generate
    
    Returns:
        list_of_weight_space_points: List of weight space point lists for each elicitation
    """
    list_of_weight_space_points = []
    weight_spaces_dir = os.path.join(script_dir, "weight_spaces")
    
    for i, dict_data in enumerate(dict_data_list):
        elicit_num = elicitation_numbers[i]
        output_file = os.path.join(weight_spaces_dir, f"BWT_results_{elicit_num}_weights.csv")
        
        if os.path.exists(output_file):
            # Load existing weights file
            print(f"Loading existing weight space for elicitation {elicit_num} from {output_file}")
            # File format: each row is a criterion, columns are possible weight values
            weight_space_points = []
            with open(output_file, mode='r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    # First element is criterion name, rest are weight values
                    criterion_name = row[0]
                    values = [float(v) for v in row[1:]]
                    weight_space_points.append(values)
            print(f"✓ Loaded weight space with {len(weight_space_points)} criteria for elicitation {elicit_num}")
        else:
            # Generate weight space
            print(f"Generating weight space for elicitation {elicit_num}...")
            temporary_save_file = os.path.join(weight_spaces_dir, f'temp_weights_{elicit_num}.csv')
            tot_number_solutions = 0
            
            while tot_number_solutions < required_solutions:
                s = int(np.random.uniform(0, 1e6))
                print(f'  Running with RNG seed: {s}')
                all_solutions = find_all_solutions(dict_data, z_star=None, rng_seed=s)
                
                # Extract and deduplicate weights
                raw_weights = np.array([sol[:-1] for sol in all_solutions[1:]])
                rounded = np.round(raw_weights, 3)
                _, uniq_idx = np.unique(rounded, axis=0, return_index=True)
                uniq_idx = np.sort(uniq_idx)
                unique_weights = rounded[uniq_idx]
                unique_z = np.array([all_solutions[j][-1] for j in (uniq_idx + 1)])
                
                # Filter by z threshold
                z_threshold = unique_z.min() + 0.1
                filtered_idx = np.where(unique_z <= z_threshold)[0]
                unique_weights = unique_weights[filtered_idx]
                unique_z = unique_z[filtered_idx]
                
                n_solutions = unique_weights.shape[0]
                print(f'  Found {len(all_solutions) - 1} solutions; {n_solutions} unique within threshold')
                
                # Append to temporary file
                file_exists = os.path.exists(temporary_save_file)
                with open(temporary_save_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    if not file_exists:
                        num_criteria = unique_weights.shape[1]
                        header = [f'Criterion {i+1}' for i in range(num_criteria)] + ['z']
                        writer.writerow(header)
                    for w, z in zip(unique_weights, unique_z):
                        row = list(w) + [z]
                        writer.writerow(row)
                
                tot_number_solutions += n_solutions
                print(f'  Total solutions so far: {tot_number_solutions}')
            
            # Read and deduplicate final results
            with open(temporary_save_file, mode='r') as file:
                csv_reader = csv.reader(file)
                rows = list(csv_reader)
            
            def _is_float(s):
                try:
                    float(s)
                    return True
                except Exception:
                    return False
            
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
            print(f"  Removed {removed} duplicate rows. Kept {len(possible_solutions)} unique rows.")
            
            # Filter by constraint satisfaction and weight sum
            removed = 0
            for row in possible_solutions[:]:
                weight_sum = sum(row[:-1])
                if not np.isclose(weight_sum, 1.0, atol=1e-3):
                    possible_solutions.remove(row)
                    removed += 1
            print(f"  Removed {removed} rows that did not sum to 1.")
            
            # Save to final file in transposed format (rows = criteria, columns = values)
            # First, extract weight columns (excluding z)
            weights_only = [row[:-1] for row in possible_solutions]
            weights_array = np.array(weights_only)
            num_criteria = weights_array.shape[1]
            
            # Get criterion names from dict_data
            crit_names = [crit_name for group_data in dict_data.values() for crit_name in group_data['criteria'].keys()]
            
            # Transpose: each row is one criterion with all its possible values
            with open(output_file, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for crit_idx, crit_name in enumerate(crit_names):
                    # Extract all unique values for this criterion across all solutions
                    criterion_values = sorted(list(set(weights_array[:, crit_idx])))
                    writer.writerow([crit_name] + criterion_values)
            
            print(f"✓ Saved weight space with {num_criteria} criteria to {output_file}")
            
            # Clean up temporary file
            if os.path.exists(temporary_save_file):
                os.remove(temporary_save_file)
            
            # Build weight_space_points for return (list of value lists, one per criterion)
            weight_space_points = []
            for crit_idx in range(num_criteria):
                criterion_values = sorted(list(set(weights_array[:, crit_idx])))
                weight_space_points.append(criterion_values)
        
        list_of_weight_space_points.append(weight_space_points)
    
    return list_of_weight_space_points