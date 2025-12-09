
#################################################################################
# Import third party libraries
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
#################################################################################

#################################################################################
# Import internal modules
from pile_bwt import bwt, constraints_func
from up_mavt import startup, mc_simulation
from aggregation_methods import weighted_sum
from weight_sampling import obtain_weight_space_description
#################################################################################

# Ensure all relative file accesses resolve relative to this script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

#################################################################################
# USER INPUTS
# Weight elicitation files (one per elicitation / run)
file_path_weight_elicitations = ["wbt_results_1.csv"]

# Value function files (one per elicitation / run)
file_path_value_functions = ["value_functions_1.csv"]

# Criteria definitions file
file_path_criteria = "criteria.csv"

# Montecarlo Parameters
n_runs = 10000
PLOTS = True  # Toggle plots
plot_bins = 50  # Number of bins for histograms
STRICT = False  # Toggle strict mode
UPDATE_EVERY = 100  # Update plots every N runs
opinion_weights = np.ones(len(file_path_weight_elicitations))/len(file_path_weight_elicitations)  # Equal weights for each elicitation
#################################################################################


#################################################################################
# Startup: Load all data
dict_data_list, crit_index, vf_list, alternatives = startup(file_path_criteria, file_path_weight_elicitations, file_path_value_functions)
n_alternatives = len(alternatives)
print(f"Loaded {len(dict_data_list)} elicitation(s) with {n_alternatives} alternatives.")

# Extract alternative names for plotting (the CSV contains a `name` column)
alternative_names = []
try:
    alt_path = os.path.join(SCRIPT_DIR, "alternatives.csv")
    with open(alt_path, mode='r') as altf:
        reader = csv.DictReader(altf)
        for idx, row in enumerate(reader):
            name = row.get('name') if row is not None else None
            if name is None or name == '':
                name = f"Alt {idx}"
            alternative_names.append(name)
    # Fallback to generic names if the file didn't contain names or had fewer rows
    if len(alternative_names) < n_alternatives:
        alternative_names = [f"Alt {i+1}" for i in range(n_alternatives)]
except Exception:
    alternative_names = [f"Alt {i+1}" for i in range(n_alternatives)]
#################################################################################

# read wbt_results.csv
bwt_results = []
for file_path in file_path_weight_elicitations:
    # read the csv
    import csv
    with open(file_path, mode='r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            bwt_results.append(row)

# Based on min and max of criteria.csv as well as type (positive or negative polarity), create linear value functions for each criteria
linear_vf_list = []
# The loaded `dict_data_list` groups criteria by group name. Flatten the criteria from
# the first elicitation and create linear value functions per criterion.
first_grouped = dict_data_list[0]
for gname, gdata in first_grouped.items():
    for crit_name, crit in gdata['criteria'].items():
        crit_min = crit.get('min', crit.get('min_value'))
        crit_max = crit.get('max', crit.get('max_value'))
        crit_type = crit.get('type')

        if crit_min is None or crit_max is None:
            raise ValueError(f"Missing min/max for criterion: {crit_name}")

        if crit_type is None:
            # Fallback: assume increasing if max > min
            crit_type = 'positive' if crit_max > crit_min else 'negative'

        if crit_type == 'positive':
            def make_vf_positive(cmin, cmax):
                def vf_positive(x):
                    if x <= cmin:
                        return 0.0
                    elif x >= cmax:
                        return 1.0
                    else:
                        return (x - cmin) / (cmax - cmin)
                return vf_positive
            linear_vf_list.append({'name': crit_name, 'function': make_vf_positive(crit_min, crit_max)})
        elif crit_type == 'negative':
            def make_vf_negative(cmin, cmax):
                def vf_negative(x):
                    if x <= cmin:
                        return 1.0
                    elif x >= cmax:
                        return 0.0
                    else:
                        return (cmax - x) / (cmax - cmin)
                return vf_negative
            linear_vf_list.append({'name': crit_name, 'function': make_vf_negative(crit_min, crit_max)})
        else:
            raise ValueError(f"Unknown criteria type: {crit_type}")
print("Created linear value functions for all criteria.")

# Find values of wbt comparisons with linear value functions, if the type is best then the value of the comparison is computed with the best v_f, if it is of type worst, is done with the "other"
for row in bwt_results:
    comparison_type = row[0]
    crit1_name = row[1]
    crit2_name = row[2]
    value = float(row[3])
    # find the corresponding linear value functions
    vf1 = next((vf for vf in linear_vf_list if vf['name'] == crit1_name), None)
    vf2 = next((vf for vf in linear_vf_list if vf['name'] == crit2_name), None)
    if vf1 is None or vf2 is None:
        raise ValueError(f"Could not find linear value function for criteria: {crit1_name} or {crit2_name}")
    # compute the value of the comparison
    if comparison_type == 'best':
        # value = vf1(x) / (vf1(x) + vf2(x)) => we can set vf1(x) = value * (vf1(x) + vf2(x)) => vf1(x) = value * vf1(x) + value * vf2(x) => (1 - value) * vf1(x) = value * vf2(x) => vf1(x) / vf2(x) = value / (1 - value)
        ratio = value / (1 - value)
    elif comparison_type == 'worst':
        # value = vf2(x) / (vf1(x) + vf2(x)) => we can set vf2(x) = value * (vf1(x) + vf2(x)) => vf2(x) = value * vf1(x) + value * vf2(x) => (1 - value) * vf2(x) = value * vf1(x) => vf2(x) / vf1(x) = value / (1 - value)
        ratio = (1 - value) / value
    else:
        raise ValueError(f"Unknown comparison type: {comparison_type}")
    # keep original bwt row intact; do not append ratio to file rows

# Get the value functions for each criteria from value_functions_1.csv
custom_vf_list = []
for file_path in file_path_value_functions:
    import ast
    with open(file_path, mode='r') as vf_file:
        reader = csv.DictReader(vf_file)
        for row in reader:
            if not row:
                continue
            # prefer 'elicited_points' field (current format), fall back to 'points'
            crit_name = row.get('name') or row.get('criteria')
            pts_field = row.get('elicited_points') or row.get('points') or row.get('elicited_points')
            if pts_field is None:
                # if there is a 'value_function' lambda, skip (we only need points here)
                continue
            try:
                points = ast.literal_eval(pts_field)
            except Exception:
                # fallback: attempt to parse semicolon-separated pairs
                points = []
                for point_str in pts_field.split(';'):
                    x_str, y_str = point_str.split(',')
                    points.append((float(x_str), float(y_str)))

            # ensure points is a list of (x,y) tuples sorted by x
            points = [(float(p[0]), float(p[1])) for p in points]
            points.sort(key=lambda t: t[0])

            # create piecewise linear function
            def make_piecewise_func(pts):
                def piecewise_func(x):
                    for i in range(len(pts) - 1):
                        x0, y0 = pts[i]
                        x1, y1 = pts[i + 1]
                        if x0 <= x <= x1:
                            # linear interpolation
                            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
                    if x < pts[0][0]:
                        return pts[0][1]
                    else:
                        return pts[-1][1]
                return piecewise_func

            custom_vf_list.append({'name': crit_name, 'function': make_piecewise_func(points)})
print("Loaded custom value functions for all criteria.")
# Evaluate the appropriate value function for each comparison VALUE.
# For 'best' use the reference criterion VF; for 'worst' use the other criterion VF.
evaluations = []
for row in bwt_results:
    comparison_type = row[0].strip().lower()
    ref = row[1]
    other = row[2]
    raw_value = float(row[3])

    # choose which criterion's VF to evaluate with linear VF
    if comparison_type == 'best':
        target = ref
    elif comparison_type == 'worst':
        target = other
    else:
        raise ValueError(f"Unknown comparison type: {comparison_type}")

    # find linear value function for the target
    lin_vf_entry = next((vf for vf in linear_vf_list if vf['name'] == target), None)
    if lin_vf_entry is None:
        raise ValueError(f"No linear value function found for target criterion: {target}")

    lin_v = lin_vf_entry['function'](raw_value)

    # find custom (elicited) value function for the same target
    cust_vf_entry = next((vf for vf in custom_vf_list if vf['name'] == target), None)

    # if custom VF exists, invert it numerically to find x such that cust_vf(x) ~= lin_v
    corrected_x = None
    if cust_vf_entry is not None:
        # find criterion min/max from dict_data_list grouped structure
        crit_meta = None
        for gname, gdata in dict_data_list[0].items():
            if target in gdata.get('criteria', {}):
                crit_meta = gdata['criteria'][target]
                break
        if crit_meta is None:
            raise ValueError(f"Missing criteria metadata for: {target}")
        cmin = crit_meta.get('min', crit_meta.get('min_value'))
        cmax = crit_meta.get('max', crit_meta.get('max_value'))

        # sample the domain and search for a match
        found = False
        xs = np.linspace(cmin, cmax, 2000)
        for x in xs:
            v = cust_vf_entry['function'](x)
            if np.isclose(v, lin_v, atol=1e-4):
                corrected_x = float(x)
                found = True
                break
        if not found:
            # if no exact match, pick the x that minimizes the absolute difference
            vals = np.array([cust_vf_entry['function'](x) for x in xs], dtype=float)
            idx = int(np.argmin(np.abs(vals - lin_v)))
            corrected_x = float(xs[idx])
        vf_source = 'custom'
    else:
        # no custom VF: fall back to returning the original raw value
        corrected_x = raw_value
        vf_source = 'linear_fallback'

    evaluations.append((row[0], ref, other, corrected_x, row[4], target, vf_source, lin_v))

# Write evaluated results: replace Value with corrected_x
with open("wbt_results_alt.csv", mode='w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['Type', 'Reference', 'Other', 'Value', 'Group'])
    for rec in evaluations:
        writer.writerow(rec[:5])