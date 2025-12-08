# auxiliary.py
#
# This module containes all functions used to load data from CSV files so that they can be used in other modules.
#
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
            # print(f"Loaded alternative: {row['name']} with data: {alternative}")
    return alternatives


def load_criteria(file_path_criteria, file_path_elicitation):
    # Backwards-compatible wrapper: load criteria definitions then parse elicitation results
    grouped = load_criteria_definitions(file_path_criteria)
    # parse elicitation file and populate comparisons
    with open(file_path_elicitation, mode='r') as infile:
        reader = csv.reader(infile)
        next(reader)  # Skip header
        intraB = {}
        intraW = {}

        for rows in reader:
            # skip empty rows or rows with only whitespace
            if not rows or all((cell is None) or (str(cell).strip() == '') for cell in rows):
                continue
            # Expect at least 5 columns: type, reference, other, value, group
            if len(rows) < 5:
                raise ValueError(f"Invalid elicitation row (expected 5 columns) in {file_path_elicitation}: {rows}")

            type_comparison = rows[0].strip()
            reference = rows[1].strip()
            other = rows[2].strip()
            try:
                value = float(rows[3])
            except Exception:
                raise ValueError(f"Invalid numeric value for comparison in {file_path_elicitation}: {rows[3]}")
            group = rows[4].strip()

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
                    grouped[next_group_for_ref(grouped, reference)]["criteria"][reference]["best_comparisons"][other] = value
                elif type_comparison == "worst":
                    grouped[next_group_for_ref(grouped, reference)]["criteria"][reference]["worst_comparisons"][other] = value

    # attach intra comparisons to groups (same for all groups)
    for g in grouped:
        grouped[g]["intraB"] = intraB
        grouped[g]["intraW"] = intraW

    return grouped


def next_group_for_ref(grouped, reference):
    """Helper to find which group contains a given criterion name."""
    for gname, gdata in grouped.items():
        if reference in gdata.get("criteria", {}):
            return gname
    raise KeyError(f"Reference criterion '{reference}' not found in any group")


def load_criteria_definitions(file_path_criteria):
    """Load criteria definitions (name, group, min, max, units) from criteria CSV.
    Returns grouped_criteria: {group_name: {"criteria": {crit_name: {min_value, max_value, units, best_comparisons, worst_comparisons}}}}
    """
    import csv

    with open(file_path_criteria, mode='r') as infile:
        reader = csv.reader(infile)
        next(reader)  # Skip header
        criteria = {}
        for rows in reader:
            crit_name = rows[0]
            group = rows[1] if len(rows) > 1 else "Default"
            min_value = float(rows[2]) if len(rows) > 2 and rows[2] != '' else None
            max_value = float(rows[3]) if len(rows) > 3 and rows[3] != '' else None
            units = rows[4] if len(rows) > 4 else ''
            criteria[crit_name] = {
                "group": group,
                "min_value": min_value,
                "max_value": max_value,
                "units": units,
                "best_comparisons": {},
                "worst_comparisons": {}
            }

    grouped_criteria = {}
    for crit_name, crit_data in criteria.items():
        group_name = crit_data.pop("group")
        if group_name not in grouped_criteria:
            grouped_criteria[group_name] = {"criteria": {}}
        grouped_criteria[group_name]["criteria"][crit_name] = crit_data

    return grouped_criteria


def load_value_functions(file_path_value_functions):
    """Load value functions from CSV using the `value_function` column when available."""
    import csv

    vfs = {}
    with open(file_path_value_functions, mode='r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            if not row:
                continue
            crit_name = row.get('name') or ''
            expr = row.get('value_function')
            if not expr:
                # fallback for legacy two-column files: use the second column's value
                values = list(row.values())
                expr = values[1] if len(values) > 1 else None
            if not expr:
                raise ValueError(f"No value function expression found for criterion '{crit_name}' in {file_path_value_functions}")

            fn = eval(expr)
            vfs[crit_name] = fn

    return vfs
