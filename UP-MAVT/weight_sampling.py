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

def space_sampling(dict_data, num_criteria, bwt_result):
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
    print(f"Total cells to explore: {(grid_size) ** n_dimensions:.2e}")
    print(f"Initial point: {initial_point}")
    print(f"Initial cell: {initial_cell}")
                              
    print(f"\nSpace sampling completed with {len(valid_points)} valid samples found.")
    return np.array(valid_points)
