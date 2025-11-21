import csv
import ast
def load_alternatives(file_path):
    """Load alternatives from a CSV file and parse distributions as dictionaries."""
    alternatives = []
    with open(file_path, mode='r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            alternative = {}
            for key, value in row.items():
                if key != 'name':
                    alternative[key] = ast.literal_eval(value)  # Parse the string as a dictionary
            alternatives.append(alternative)
    return alternatives


def load_criteria(file_path_criteria, file_path_elicitation):
    import csv

    # Parse criteria file
    with open(file_path_criteria, mode='r') as infile:
        reader = csv.reader(infile)
        next(reader)  # Skip header
        criteria = {}
        for rows in reader:
            crit_name = rows[0]
            group = rows[1]
            min_value = float(rows[2])
            max_value = float(rows[3])
            value_function = eval(rows[6])
            criteria[crit_name] = {
                "group": group,
                "min_value": min_value,
                "max_value": max_value,
                "value_function": value_function,
                "best_comparisons": {},
                "worst_comparisons": {}
            }

    # Parse elicitation results file
    with open(file_path_elicitation, mode='r') as infile:
        reader = csv.reader(infile)
        next(reader)  # Skip header
        elicitation_results = {}
        intraB = {}
        intraW = {}

        for rows in reader:
            type_comparison = rows[0]
            reference = rows[1]
            other = rows[2]
            value = float(rows[3])
            group = rows[4]

            if group == "Between-groups-B":
                intraB[f"{type_comparison}_{reference}_{other}"] = {
                    "type": type_comparison,
                    "reference": reference,
                    "other": other,
                    "value": value
                }
            elif group == "Between-groups-W":
                intraW[f"{type_comparison}_{reference}_{other}"] = {
                    "type": type_comparison,
                    "reference": reference,
                    "other": other,
                    "value": value
                }
            else:
                if type_comparison == "best":
                    criteria[reference]["best_comparisons"][other] = value
                elif type_comparison == "worst":
                    criteria[reference]["worst_comparisons"][other] = value

    # Combine results into a structured dictionary
    grouped_criteria = {}
    for crit_name, crit_data in criteria.items():
        group_name = crit_data.pop("group")
        if group_name not in grouped_criteria:
            grouped_criteria[group_name] = {
                "criteria": {},
                "intraB": intraB,
                "intraW": intraW
            }
        grouped_criteria[group_name]["criteria"][crit_name] = crit_data

    return grouped_criteria
