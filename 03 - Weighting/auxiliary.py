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
            criteria[row["name"]] = {
                "min": float(row["min"]),
                "max": float(row["max"]),
                "unit": row["unit"],
                "value_function": eval("lambda x: " + row["value_function"])
            }
    return criteria


def plot_results(result, criteria_names):
    weights = result.x[:-1]  # Extract weights, excluding z
    plt.close('all')
    plt.figure(figsize=(8, 6))
    plt.bar(criteria_names, weights)
    plt.xlabel('Criteria')
    plt.ylabel('Weights')
    plt.title('Criterion Weights')
    plt.show()

# Save weights to a file
def save_to_file(result, criteria_names, working_directory):
    weights = result.x[:-1] 
    weights_to_save = {criteria_names[i]: f"{weights[i]:.2f}" for i in range(len(criteria_names))}
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
