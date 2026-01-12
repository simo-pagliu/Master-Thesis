# auxiliary.py
#
# This module containes all functions used to load data from CSV files so that they can be used in other modules.
#
import csv
import ast
import math
import json
import os
import pandas as pd

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
            # optional "type" column (e.g. 'positive'/'negative') in position 5
            crit_type = rows[5].strip() if len(rows) > 5 and rows[5] != '' else None
            criteria[crit_name] = {
                "group": group,
                # keep old keys for backward compatibility
                "min_value": min_value,
                "max_value": max_value,
                "units": units,
                # add convenience keys used elsewhere: 'min','max','type'
                "min": min_value,
                "max": max_value,
                "type": crit_type,
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
    """Load value functions from CSV by parsing `elicited_points` and building
    safe piecewise-linear interpolators. The CSV may contain a `value_function`
    column in older files, but we no longer rely on evaluating executable strings.
    Returns a dict mapping criterion name -> callable(x) in [0.001, 1.0]."""
    import csv
    import ast
    vfs = {}
    with open(file_path_value_functions, mode='r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            if not row:
                continue
            crit_name = (row.get('name') or '').strip()
            # Prefer an explicit `elicited_points` column (JSON or Python list literal)
            pts_raw = row.get('elicited_points') or ''
            pts = None
            if pts_raw:
                try:
                    pts = ast.literal_eval(pts_raw)
                except Exception:
                    try:
                        pts = json.loads(pts_raw)
                    except Exception:
                        pts = None

            # If no elicited_points, skip this row (we no longer evaluate executable strings)
            if not pts:
                # do not attempt to eval legacy `value_function` strings; skip
                continue
            # Build interpolator/fit from pts (list of [x,y]) when available
            if pts and isinstance(pts, (list, tuple)) and len(pts) > 0:
                # parse metadata to choose fit type
                meta_raw = row.get('elicitation_meta') or ''
                meta = {}
                if meta_raw:
                    try:
                        meta = json.loads(meta_raw)
                    except Exception:
                        try:
                            meta = ast.literal_eval(meta_raw)
                        except Exception:
                            meta = {}

                fit_type = str(meta.get('fit_type') or meta.get('type') or '').strip()
                if fit_type.lower() in ('piecewise linear', 'piecewise-linear', 'piecewise', 'linear'):
                    fit_type = 'piecewise'
                elif fit_type.lower() in ('pchip', 'pchipinterpolator'):
                    fit_type = 'pchip'
                elif fit_type.lower() in ('gaussian', 'normal'):
                    fit_type = 'gaussian'
                elif fit_type.lower() in ('sigmoid', 'logistic'):
                    fit_type = 'sigmoid'
                else:
                    if not fit_type:
                        fit_type = 'piecewise'

                try:
                    xs = [float(p[0]) for p in pts]
                    ys = [0.001 if float(p[1]) == 0.0 else float(p[1]) for p in pts]
                except Exception:
                    xs, ys = None, None
                if xs is None or ys is None or len(xs) == 0:
                    continue

                import numpy as _np

                # PCHIP (preferred when available)
                if fit_type == 'pchip':
                    try:
                        from scipy.interpolate import PchipInterpolator
                        pchip = PchipInterpolator(xs, ys, extrapolate=True)
                        def vf_pchip(x, _p=pchip):
                            try:
                                val = float(_p(float(x)))
                            except Exception:
                                val = 0.001
                            return float(_np.clip(val, 0.001, 1.0))
                        try:
                            vf_pchip._xs = _np.array(xs, dtype=float)
                            vf_pchip._ys = _np.array(ys, dtype=float)
                        except Exception:
                            pass
                        vfs[crit_name] = vf_pchip
                    except Exception:
                        # fallback to piecewise
                        fit_type = 'piecewise'

                if fit_type == 'piecewise':
                    xs_arr = _np.array(xs, dtype=float)
                    ys_arr = _np.array(ys, dtype=float)
                    def make_piecewise_from_arrays(xs_arr, ys_arr):
                        def vf(x):
                            try:
                                xv = float(x)
                            except Exception:
                                xv = float(xs_arr[0])
                            if xv <= xs_arr[0]:
                                x0, x1 = xs_arr[0], xs_arr[1] if xs_arr.size > 1 else xs_arr[0]
                                y0, y1 = ys_arr[0], ys_arr[1] if ys_arr.size > 1 else ys_arr[0]
                                slope = (y1 - y0) / (x1 - x0) if (x1 != x0) else 0.0
                                val = y0 + slope * (xv - x0)
                            elif xv >= xs_arr[-1]:
                                x0, x1 = xs_arr[-2] if xs_arr.size > 1 else xs_arr[-1], xs_arr[-1]
                                y0, y1 = ys_arr[-2] if ys_arr.size > 1 else ys_arr[-1], ys_arr[-1]
                                slope = (y1 - y0) / (x1 - x0) if (x1 != x0) else 0.0
                                val = y1 + slope * (xv - xs_arr[-1])
                            else:
                                idx = int(_np.searchsorted(xs_arr, xv) - 1)
                                x0, x1 = xs_arr[idx], xs_arr[idx+1]
                                y0, y1 = ys_arr[idx], ys_arr[idx+1]
                                slope = (y1 - y0) / (x1 - x0) if (x1 != x0) else 0.0
                                val = y0 + slope * (xv - x0)
                            if _np.isnan(val):
                                val = 0.001
                            return float(_np.clip(val, 0.001, 1.0))
                        try:
                            vf._xs = xs_arr
                            vf._ys = ys_arr
                        except Exception:
                            pass
                        return vf
                    # return a fresh closure with nodes attached
                    def build_pw():
                        return make_piecewise_from_arrays(xs_arr, ys_arr)
                    vfs[crit_name] = build_pw()

                if fit_type == 'gaussian':
                    try:
                        mu = float(meta.get('mu'))
                    except Exception:
                        mu = float(_np.mean(xs))
                    try:
                        sigma = float(meta.get('sigma'))
                    except Exception:
                        sigma = float((max(xs) - min(xs)) / 6.0) if max(xs) != min(xs) else 1.0
                    amplitude = float(meta.get('amplitude', 1.0))
                    def make_gaussian(mu, sigma, amplitude):
                        def vf(x):
                            try:
                                xv = float(x)
                                val = amplitude * math.exp(-0.5 * ((xv - mu) / sigma) ** 2)
                            except Exception:
                                val = 0.001
                            return float(_np.clip(val, 0.001, 1.0))
                        return vf
                    vfs[crit_name] = make_gaussian(mu, sigma, amplitude)

                if fit_type == 'sigmoid':
                    try:
                        k = float(meta.get('k'))
                    except Exception:
                        k = float(meta.get('slope', 1.0))
                    try:
                        x0 = float(meta.get('x0'))
                    except Exception:
                        x0 = float(_np.median(xs))
                    direction = str(meta.get('direction') or meta.get('shape') or '').lower()
                    inc = True
                    if 'dec' in direction or 'decrease' in direction or 'negative' in direction:
                        inc = False
                    def make_sigmoid(k, x0, inc=True):
                        def vf(x):
                            try:
                                xv = float(x)
                                s = 1.0 / (1.0 + math.exp(-k * (xv - x0)))
                                val = s if inc else (1.0 - s)
                            except Exception:
                                val = 0.001
                            return float(_np.clip(val, 0.001, 1.0))
                        return vf
                    vfs[crit_name] = make_sigmoid(k, x0, inc)
            else:
                # no pts and no usable expr: skip
                continue
    return vfs

def combine_alternatives_by_country(elicitation_dirs, selected_country, output_dir, qi_elicitation_dirs=None):
    """
    Combine alternatives from multiple elicitations for a specific country.
    
    For quantitative indicators (same values across all elicitations), keep value as is.
    For qualitative indicators (different values), create a discrete distribution.
    Also includes qualitative indicators from dedicated folders (qi_elicitation_dirs).
    
    Args:
        elicitation_dirs: List of paths to elicitation result directories (e.g., ["elicitation_results/1", "elicitation_results/2"])
        selected_country: Country code (e.g., "IT", "FR")
        output_dir: Directory to save the combined alternatives file
        qi_elicitation_dirs: List of paths to QI-only elicitation directories (default: None)
    """
    import json
    import os
    import pandas as pd
    
    all_alternatives = []
    qi_alternatives_dfs = []
    
    # Load alternatives from each elicitation
    for elicit_dir in elicitation_dirs:
        alt_path = os.path.join(elicit_dir, selected_country, "alternatives.csv")
        if not os.path.exists(alt_path):
            raise FileNotFoundError(f"Alternatives file not found: {alt_path}")
        
        df = pd.read_csv(alt_path)
        all_alternatives.append(df)
    
    # Load qualitative indicators from dedicated folders if provided
    if qi_elicitation_dirs:
        for qi_dir in qi_elicitation_dirs:
            qi_path = os.path.join(qi_dir, selected_country, "alternatives.csv")
            if os.path.exists(qi_path):
                qi_df = pd.read_csv(qi_path)
                qi_alternatives_dfs.append(qi_df)
    
    if not all_alternatives:
        raise ValueError("No alternatives files loaded")
    
    # Get list of alternative names from first file
    alt_names = all_alternatives[0]['name'].tolist()
    criteria_columns = [col for col in all_alternatives[0].columns if col != 'name']
    
    # Build combined alternatives
    combined_data = []
    
    # Known qualitative indicators that should be merged from folder 3
    qualitative_indicators = [
        'Design Maturity',
        'Licensing Status', 
        'Design Complexity',
        'Construction Complexity',
        'Supplier Availbility'
    ]
    
    for alt_name in alt_names:
        row_data = {'name': alt_name}
        
        for criterion in criteria_columns:
            # Collect values for this criterion across all elicitations
            values = []
            parsed_values = []
            
            for df in all_alternatives:
                if alt_name in df['name'].values:
                    cell_value = df[df['name'] == alt_name][criterion].iloc[0]
                    
                    # Handle NaN and empty values
                    if pd.isna(cell_value) or str(cell_value).strip() == '':
                        continue
                    
                    values.append(cell_value)
                    
                    # Parse the value to extract numeric values
                    try:
                        parsed = json.loads(cell_value)
                        if isinstance(parsed, dict) and 'Discrete' in parsed:
                            # Extract values from Discrete distribution
                            discrete_vals = parsed['Discrete'][0] if parsed['Discrete'] else []
                            parsed_values.extend(discrete_vals)
                        elif isinstance(parsed, dict):
                            # For other distributions, just keep the original
                            parsed_values.append(cell_value)
                    except (json.JSONDecodeError, TypeError):
                        # If it's not JSON, keep the raw value
                        parsed_values.append(cell_value)
            
            # If this is a qualitative indicator and QI data exists, add those values too
            if criterion in qualitative_indicators and qi_alternatives_dfs:
                for qi_df in qi_alternatives_dfs:
                    if alt_name in qi_df['name'].values:
                        qi_value = qi_df[qi_df['name'] == alt_name][criterion].iloc[0]
                        if not pd.isna(qi_value) and str(qi_value).strip() != '':
                            values.append(qi_value)
                            try:
                                parsed_qi = json.loads(qi_value)
                                if isinstance(parsed_qi, dict) and 'Discrete' in parsed_qi:
                                    discrete_vals = parsed_qi['Discrete'][0] if parsed_qi['Discrete'] else []
                                    parsed_values.extend(discrete_vals)
                                else:
                                    parsed_values.append(qi_value)
                            except (json.JSONDecodeError, TypeError):
                                parsed_values.append(qi_value)
            
            if not values:
                # No values found for this criterion
                row_data[criterion] = ''
                continue
            
            if len(values) == 1:
                # Only one elicitation has a value, keep it as is
                row_data[criterion] = values[0]
            else:
                # Multiple values - check if they're all the same
                if len(set(str(v) for v in values)) == 1:
                    # All values are the same, keep as is
                    row_data[criterion] = values[0]
                else:
                    # Values differ - need to create a discrete distribution
                    # Extract numeric values from parsed_values
                    numeric_values = []
                    for pv in parsed_values:
                        try:
                            if isinstance(pv, (int, float)):
                                numeric_values.append(float(pv))
                            elif isinstance(pv, str):
                                numeric_values.append(float(pv))
                        except (ValueError, TypeError):
                            pass
                    
                    if numeric_values:
                        # Sort and remove duplicates
                        numeric_values = sorted(list(set(numeric_values)))
                        row_data[criterion] = json.dumps({'Discrete': [numeric_values]})
                    else:
                        # If no numeric values extracted, keep the first value
                        row_data[criterion] = values[0]
        
        combined_data.append(row_data)
    
    # Create combined dataframe
    combined_df = pd.DataFrame(combined_data)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save combined alternatives
    output_file = os.path.join(output_dir, f"alternatives_{selected_country}.csv")
    combined_df.to_csv(output_file, index=False)
    
    print(f"✓ Combined alternatives for {selected_country} saved to {output_file}")
    return combined_df


# ============================================================================
# Data Conversion and Normalization Functions
# ============================================================================

def interpolate_value_function(vf_points, x):
    """Piecewise linear interpolation of value function points."""
    vf_points = sorted(vf_points, key=lambda p: p[0])
    
    if x <= vf_points[0][0]:
        return vf_points[0][1]
    if x >= vf_points[-1][0]:
        return vf_points[-1][1]
    
    for i in range(len(vf_points) - 1):
        x1, v1 = vf_points[i]
        x2, v2 = vf_points[i + 1]
        if x1 <= x <= x2:
            if x2 == x1:
                return v1
            t = (x - x1) / (x2 - x1)
            return v1 + t * (v2 - v1)
    
    return vf_points[-1][1]

def detect_raw_scale_in_data(alt_rows, fieldnames):
    """
    Detect which qualitative indicators have raw 1-6 scale data by examining
    actual values in alternatives (values > 1 indicate 1-6 scale).
    """
    qualitative_indicators = [
        'Design Maturity',
        'Licensing Status',
        'Supplier Availbility',
        'Design Complexity',
        'Construction Complexity'
    ]
    
    raw_scale = []
    for indicator in qualitative_indicators:
        if indicator not in fieldnames:
            continue
        
        max_val = 0
        for row in alt_rows:
            try:
                dist = ast.literal_eval(row[indicator])
                if 'Discrete' in dist:
                    val = float(dist['Discrete'][0][0])
                    max_val = max(max_val, val)
            except:
                pass
        
        if max_val > 1.0:
            raw_scale.append(indicator)
    
    return raw_scale


def convert_qualitative_indicators_in_folders(folder_numbers):
    """
    Convert qualitative indicators in folders to 0-1 values.
    - Converts values on 1-6 scale using their value functions
    - Replaces all qualitative VFs with linear ones [[0,0],[1,1]]
    
    Args:
        folder_numbers: List of folder numbers to process (e.g., [1, 2])
    """
    from pathlib import Path
    
    qualitative_indicators = [
        'Design Maturity',
        'Licensing Status',
        'Supplier Availbility',
        'Design Complexity',
        'Construction Complexity'
    ]
    
    for folder_num in folder_numbers:
        base_dir = Path(__file__).parent / 'elicitation_results' / str(folder_num)
        
        for country in ['IT', 'CH', 'FR', 'PO']:
            country_folder = base_dir / country
            if not country_folder.exists():
                continue
            
            vf_file = country_folder / 'value_functions.csv'
            alt_file = country_folder / 'alternatives.csv'
            
            if not vf_file.exists() or not alt_file.exists():
                continue
            
            # Load value functions
            vf_dict = {}
            with open(vf_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row['name']
                    points = json.loads(row['elicited_points'])
                    vf_dict[name] = points
            
            # Read alternatives
            with open(alt_file, 'r') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                alt_rows = list(reader)
            
            # Detect which qualitative indicators are on 1-6 scale
            raw_scale_indicators = detect_raw_scale_in_data(alt_rows, fieldnames)
            
            # Convert qualitative indicators
            for indicator in qualitative_indicators:
                if indicator not in fieldnames:
                    continue
                if indicator not in vf_dict:
                    continue
                
                if indicator not in raw_scale_indicators:
                    continue
                
                vf_points = vf_dict[indicator]
                
                for row in alt_rows:
                    try:
                        dist_dict = ast.literal_eval(row[indicator])
                        
                        if 'Discrete' in dist_dict:
                            discrete_list = dist_dict['Discrete']
                            converted_list = []
                            
                            for item in discrete_list:
                                if isinstance(item, list):
                                    converted_item = []
                                    for val_str in item:
                                        raw_val = float(val_str)
                                        converted_val = interpolate_value_function(vf_points, raw_val)
                                        converted_item.append(str(converted_val))
                                    converted_list.append(converted_item)
                                else:
                                    raw_val = float(item)
                                    converted_val = interpolate_value_function(vf_points, raw_val)
                                    converted_list.append(str(converted_val))
                            
                            row[indicator] = json.dumps({"Discrete": converted_list})
                    except:
                        pass
            
            # Write converted alternatives
            with open(alt_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(alt_rows)
            
            # Replace qualitative VFs with linear ones
            with open(vf_file, 'r') as f:
                reader = csv.DictReader(f)
                vf_fieldnames = reader.fieldnames
                vf_rows = list(reader)
            
            for vf_row in vf_rows:
                if vf_row['name'] in qualitative_indicators:
                    vf_row['elicited_points'] = json.dumps([[0.0, 0.0], [1.0, 1.0]])
            
            with open(vf_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=vf_fieldnames)
                writer.writeheader()
                writer.writerows(vf_rows)


def load_criteria_file(path):
    """Load a criteria CSV file."""
    return pd.read_csv(path)


def verify_criteria_consistency(criteria_list):
    """Verify that all criteria dataframes are identical."""
    if not criteria_list:
        raise ValueError("No criteria files provided")
    
    first_crit = criteria_list[0]
    for i, crit in enumerate(criteria_list[1:], start=1):
        if not first_crit.equals(crit):
            raise ValueError(f"Criteria file {i} differs from the first. All elicitations must have identical criteria.")
    
    return first_crit