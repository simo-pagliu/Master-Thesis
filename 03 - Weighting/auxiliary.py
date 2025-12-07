import matplotlib.pyplot as plt
import numpy as np
import csv
import os

def value_function_plot(criteria):
    criteria_import = import_criteria(criteria)
    plt.figure(figsize=(10, 6))
    for key, values in criteria_import.items():
        x_values = np.linspace(values['min'], values['max'], 100)
        y_values = [values['value_function'](x) for x in x_values]
        plt.plot(x_values, y_values, label=key)
        plt.title(f'{key} Value Functions')
        plt.xlabel('Data Range')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()
        plt.show()

def import_criteria(file_path):
    criteria = {}
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Read raw min/max values with safe defaults
            try:
                raw_min = float(row.get("min", "0"))
            except Exception:
                raw_min = 0.0
            try:
                raw_max = float(row.get("max", raw_min + 1.0))
            except Exception:
                raw_max = raw_min + 1.0

            # Determine type (positive/negative). If negative, invert the
            # meaning of min/max for elicitation so that UI shows the
            # criterion range with the logical "min" coming from the CSV's
            # max column and vice-versa.
            type_str = str(row.get("type", "")).strip().lower()
            if type_str in ("negative", "neg", "-"):
                lo = raw_max
                hi = raw_min
            else:
                lo = raw_min
                hi = raw_max

            # Ensure a non-zero range
            if hi == lo:
                hi = lo + 1.0

            criteria[row["name"]] = {
                "min": lo,
                "max": hi,
                "unit": row.get("unit"),
                "group": row.get("group"),
                "type": type_str,
            }
    return criteria


def plot_results(result, criteria_names):
    plt.close('all')
    plt.figure(figsize=(8, 6))

    # result.x may be either a numpy array (with z as last element) or a dict
    if isinstance(getattr(result, 'x', None), dict):
        weights_map = result.x
        # Use provided ordering if available, otherwise use dict key order
        names = criteria_names if criteria_names else list(weights_map.keys())
        weights = [weights_map.get(n, 0.0) for n in names]
    else:
        # Assume optimize result with weights + z
        weights = getattr(result, 'x', [])
        if len(weights) > 0:
            weights = weights[:-1]
        names = criteria_names if criteria_names else [f'C{i}' for i in range(len(weights))]

    plt.bar(names, weights)
    plt.xlabel('Criteria')
    plt.ylabel('Weights')
    plt.title('Criterion Weights')
    plt.show()

# Save weights to a file
def save_to_file(result, criteria_names, working_directory):
    # Support both dict-style and optimize-result-style outputs
    if isinstance(getattr(result, 'x', None), dict):
        weights_map = result.x
        names = criteria_names if criteria_names else list(weights_map.keys())
        weights_to_save = {n: f"{weights_map.get(n, 0.0):.2f}" for n in names}
    else:
        weights = getattr(result, 'x', [])
        if len(weights) > 0:
            weights = weights[:-1]
        names = criteria_names if criteria_names else [f'C{i}' for i in range(len(weights))]
        weights_to_save = {names[i]: f"{weights[i]:.2f}" for i in range(len(names))}

    # Check for existing file and change name if necessary
    file_index = 1
    while os.path.exists(os.path.join(working_directory, f"criterion_weights_{file_index}.csv")):
        file_index += 1
    with open(os.path.join(working_directory, f"criterion_weights_{file_index}.csv"), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Criterion", "Weight"])
        for name, weight in weights_to_save.items():
            writer.writerow([name, weight])
    print(f"Criterion weights saved to criterion_weights_{file_index}.csv")
