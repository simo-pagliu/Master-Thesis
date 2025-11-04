import matplotlib.pyplot as plt
import numpy as np
import csv
import os

def value_function_plot(criteria):
    criteria_import = import_criteria(criteria)
    plt.figure(figsize=(10, 6))
    for i in range(len(criteria_import)):
        x_values = np.linspace(criteria_import[i]['min'], criteria_import[i]['max'], 100)
        y_values = [criteria_import[i]['value_function'](x) for x in x_values]
        plt.plot(x_values, y_values, label=criteria_import[i]['name'])
        plt.title(f'{criteria_import[i]["name"]} Value Functions')
        plt.xlabel('Data Range')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()
        plt.show()

def import_criteria(file_path):
    criteria = []
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            criterion = {
                "name": row["name"],
                "min": float(row["min"]),
                "max": float(row["max"]),
                "unit": row["unit"],
                "value_function": eval("lambda x: " + row["value_function"])
            }
            criteria.append(criterion)
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
