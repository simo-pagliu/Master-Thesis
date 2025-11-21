import csv

# Load criteria data
criteria = {}
with open("criteria.csv", mode="r") as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        name = row["name"]
        criteria[name] = {
            "group": row["group"],
            "min": float(row["min"]),
            "max": float(row["max"]),
            "value_function": eval(row["value_function"]),
        }

# Load WBT results
wbt_results = []
with open("wbt_results.csv", mode="r") as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        wbt_results.append({
            "type": row["Type"],
            "reference": row["Reference"],
            "other": row["Other"],
            "value": float(row["Value"]),
            "group": row["Group"],
        })

# Validate WBT results
for result in wbt_results:
    ref = result["reference"]
    other = result["other"]
    value = result["value"]

    if result["type"] == "best":
        # For best comparisons, use the value function of the reference criterion
        if ref in criteria:
            ref_crit = criteria[ref]
            normalized_value = ref_crit["value_function"](value)

            if not (0 <= normalized_value <= 1):
                print(f"Value {value} for {ref} is out of range after normalization: {normalized_value}")

    elif result["type"] == "worst":
        # For worst comparisons, use the value function of the other criterion
        if other in criteria:
            other_crit = criteria[other]
            normalized_value = other_crit["value_function"](value)

            if not (0 <= normalized_value <= 1):
                print(f"Value {value} for {other} is out of range after normalization: {normalized_value}")