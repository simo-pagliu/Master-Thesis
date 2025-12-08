import os
import numpy as np
import csv
from pile_bwt import constraints_func, bwt

def obtain_weight_space_description(bwt_results, dict_data_list, file_path_weight_elicitations, crit_index):
    weight_list = []
    # Recover criterion names ordered by their index for consistent plotting/serialization
    crit_names = [name for name, idx in sorted(crit_index.items(), key=lambda item: item[1])]
    for i, (fp, bwt_result) in enumerate(zip(file_path_weight_elicitations, bwt_results)):
        # Define output file name
        output_file = os.path.splitext(fp)[0] + "_weights" + ".csv"
        if not os.path.exists(output_file):
            print(f"Generating weight samples for elicitation {i+1}...")
            # Load dict_data for this elicitation
            dict_data = dict_data_list[i]
            num_criteria = len(crit_names)

            # Generate weights
            weight_space_points = define_weight_space(dict_data, num_criteria, bwt_result, crit_names=crit_names)

            # Save points to a TXT file, each column is a criterion
            with open(output_file, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for p, criteria_points in enumerate(weight_space_points):
                    writer.writerow([crit_names[p], *criteria_points])
                
        else:
            print(f"Weight samples for elicitation {i+1} already exist. Reading from file...")
            # Read the weights back from the existing TXT file
            weight_space_points = []
            with open(output_file, mode='r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    # Convert string values to float, skipping the first element (header)
                    points = [float(value) for value in row[1:]]
                    weight_space_points.append(points)
        weight_list.append(weight_space_points)
    return weight_list

def define_weight_space(dict_data, num_criteria, bwt_result, plot = True, increment=0.001, crit_names=None):
    known_max_error = bwt_result["z"]
    weight_ranges = []
    # Provide human-readable labels for plots; fall back to generic numbering if names are missing
    if crit_names is None:
        crit_names = [f'Criterion {i+1}' for i in range(num_criteria)]

    for i in range(num_criteria):
        min_run = bwt(dict_data, i, known_max_error, 'min')['solver_result']
        max_run = bwt(dict_data, i, known_max_error, 'max')['solver_result']
        weight_ranges.append([min_run['x'], max_run['x']])

        if min_run['success'] is False:
            print(f"Warning: Minimization for criterion {i+1} ({crit_names[i]}) did not converge.")
            # print(min_run)
        if max_run['success'] is False:
            print(f"Warning: Maximization for criterion {i+1} ({crit_names[i]}) did not converge.")
            # print(max_run)

    if plot:
        import matplotlib.pyplot as plt

        mins = [max(0, weight_ranges[i][0][i]) for i in range(num_criteria)]
        maxs = [min(1, weight_ranges[i][1][i]) for i in range(num_criteria)]
        widths = [maxs[i] - mins[i] for i in range(num_criteria)]

        fig_height = max(2, 0.6 * num_criteria)
        fig, ax = plt.subplots(figsize=(8, fig_height))

        y = np.arange(num_criteria)
        ax.barh(y, widths, left=mins, height=0.5, color='C0', edgecolor='k')

        ax.set_yticks(y)
        ax.set_yticklabels(crit_names)
        ax.invert_yaxis()

        span = (max(maxs) - min(mins)) if num_criteria > 0 else 1.0
        margin = span * 0.05 if span > 0 else 0.01
        xmin = max(0.0, min(mins) - margin)
        xmax = max(maxs) + margin
        ax.set_xlim(xmin, xmax)

        ax.set_xlabel('Weight Value')
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    # For each criterion
    # Each range is discretized into "increment" increments
    # If max-min < increment we just take the midpoint
    valid_points = []
    # determine decimals from increment (e.g. 0.001 -> 3 decimals)
    decimals = max(0, int(np.ceil(-np.log10(increment)))) if increment > 0 else 6

    for i in range(num_criteria):
        w_min = max(0, weight_ranges[i][0][i])
        w_max = min(1, weight_ranges[i][1][i])

        if w_max - w_min < increment:
            mid = round((w_min + w_max) / 2.0, decimals)
            valid_points.append([mid])
        else:
            # generate points by integer multiples of `increment` to avoid
            # floating-point rounding artifacts that can produce doubled steps
            # (e.g. 0.002 instead of 0.001). Preserve previous behaviour by
            # rounding UP the first and last values to the nearest increment.
            eps = 1e-12

            def round_up_to_increment(val, inc, decimals):
                idx = int(np.ceil((val - eps) / inc))
                return round(idx * inc, decimals), idx

            start_val, start_idx = round_up_to_increment(w_min, increment, decimals)
            end_val, end_idx = round_up_to_increment(w_max, increment, decimals)

            # Ensure at least one point
            if end_idx < start_idx:
                start_idx = end_idx

            idxs = np.arange(start_idx, end_idx + 1)
            pts = np.round(idxs * increment, decimals)

            # remove near-duplicates using a much smaller tolerance than one step
            dedup_atol = 10 ** (-(decimals + 2))
            unique_pts = []
            for p in pts:
                if not unique_pts or not np.isclose(p, unique_pts[-1], atol=dedup_atol):
                    unique_pts.append(float(p))

            # Force first and last to the rounded-up values (preserve previous behaviour)
            if unique_pts:
                unique_pts[0] = float(start_val)
                unique_pts[-1] = float(end_val)

            valid_points.append(unique_pts)

    # Print total number of points
    total_points = int(np.prod([len(vp) for vp in valid_points])) if valid_points else 0
    print(f"Total weight combinations to sample: {total_points} (values rounded to {decimals} decimals)")

    return valid_points


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
            print(f"Sampled weights: {weight_set_candidate}, sum: {total_weight}, constraints satisfied: {constraints_satisfied}")
            if not constraints_satisfied:
                continue
            else:
                weight_set_found = True
    return weight_set_candidate