import os
import numpy as np
import csv
from pile_bwt import constraints_func, bwt

def create_weight_samples(bwt_results, dict_data_list, file_path_weight_elicitations, crit_index):
    weight_list = []
    for i, (fp, bwt_result) in enumerate(zip(file_path_weight_elicitations, bwt_results)):
        # Define output file name
        output_file = os.path.splitext(fp)[0] + "_weights" + ".csv"
        if not os.path.exists(output_file):
            print(f"Generating weight samples for elicitation {i+1}...")
            # Load dict_data for this elicitation
            dict_data = dict_data_list[i]
            num_criteria = len(crit_index)

            # Generate weights
            weights = space_sampling(dict_data, num_criteria, bwt_result, output_file)

            # Filter out rows that are all zeros
            weights = [row for row in weights if not np.all(np.isclose(row, 0))]
            weight_list.append(weights)
        else:
            print(f"Weight samples for elicitation {i+1} already exist. Reading from file...")
            # Read the weights back from the existing CSV file
            weights = []
            with open(output_file, mode='r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip header
                for row in reader:
                    weights.append([float(value) for value in row])

            # Validate weights for consistent shape
            if weights and not all(len(row) == len(weights[0]) for row in weights):
                raise ValueError(f"Inconsistent row shapes in weights for elicitation {i+1}.")

            # Append validated weights
            weight_list.append(weights)
    return weight_list

def space_sampling(dict_data, num_criteria, bwt_result, output_file):
    used_cells = set()
    grid_size = np.int64(1000)
    n_dimensions = num_criteria
    # print("Number of dimensions for sampling:", n_dimensions)
    nx = np.array([grid_size] * n_dimensions, dtype=np.int64)

    known_max_error = bwt_result["z"]
    initial_point = bwt_result["solver_result"]["x"][:num_criteria]
    # print("Initial point for sampling:", initial_point)
    initial_cell = tuple(np.clip((np.array(initial_point) * grid_size).astype(np.int64), 0, grid_size - 1))
    used_cells.add(initial_cell)
    valid_points = [np.array(initial_cell, dtype=np.float64) / grid_size]

    print("Starting space sampling...")
    print(f"Grid size: {grid_size}, Number of dimensions: {n_dimensions}")
    print(f"Initial point: {initial_point}, Initial cell: {initial_cell}")

    # Open the correct CSV file for this elicitation
    with open(output_file, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f"Criterion_{i+1}" for i in range(num_criteria)])  # Header row

        i = 0
        # while i < max_iters:
        while True:
            i += 1
            # print(f"Iteration {i}, Used cells: {len(used_cells)}, Valid points: {len(valid_points)}", end="\r")
            x_a_center = valid_points[np.random.randint(len(valid_points))]

            # Compute error_a and y_a for x_a_center
            x_temp = np.concatenate([x_a_center, [0]])
            constraints = constraints_func(x_temp, dict_data)
            error_a = min(max([abs(cv) for cv in constraints]), 10)
            # y_a = min(constraints[:num_criteria])

            # Generate a random point x_b (already sums to 1)
            x_b = np.random.dirichlet(np.ones(num_criteria), size=1)[0]

            # Compute error_b and y_b for x_b
            x_temp = np.concatenate([x_b, [0]])
            constraints = constraints_func(x_temp, dict_data)
            error_b = min(max([abs(cv) for cv in constraints]), 10)
            # y_b = min(constraints[:num_criteria])

            j = 0
            max_attempts = 1000
            while j < max_attempts:
                j += 1
                condition_2 = sum(x_b) <= 1.0 + 1/grid_size
                condition_3 = abs(error_b) <= known_max_error*1.1
                print(f"abs(error_b): {abs(error_b)}", end="\r")

                if condition_2 and condition_3:
                    cell_idx = (x_b * grid_size).astype(int)
                    cell_idx = np.minimum(cell_idx, nx - 1)
                    cell = tuple(cell_idx)
                    if cell not in used_cells:
                        cell_center = np.array(cell_idx, dtype=np.float64) / grid_size
                        valid_points.append(cell_center)
                        used_cells.add(cell)
                        writer.writerow(cell_center[:num_criteria].tolist())
                        break
                elif error_b > 100 * known_max_error:
                    used_cells.add(cell)
                    # Random new point
                    x_b = np.random.dirichlet(np.ones(num_criteria), size=1)[0]
                    continue
                else:
                    direction = x_b - x_a_center
                    norm = np.linalg.norm(direction)
                    if norm == 0:
                        break

                    # Line search
                    alpha = 0.01
                    x_b = x_a_center + alpha * direction / norm
                    x_b = np.clip(x_b, 0, 1)
                    s = x_b.sum()
                    if (not np.isfinite(s)) or s <= 0:
                        break
                    x_b = x_b / s
                    x_temp = np.concatenate([x_b, [0]])
                    constraints = constraints_func(x_temp, dict_data)
                    error_b = min(max([abs(cv) for cv in constraints]), 10)
                    if sum(x_b) <= 1.0 + 1/grid_size and abs(error_b) <= known_max_error:
                        break
                    


                    
    print(f"\nSpace sampling completed with {len(valid_points)} valid samples found.")
    return np.array(valid_points)
