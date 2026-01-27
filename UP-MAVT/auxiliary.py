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
import numpy as np


def _linear_utility_from_range(x, lo, hi, crit_type):
    """Map a raw value x into a [0,1] utility using a linear normalization over [lo,hi]."""
    try:
        xv = float(x)
        lo = float(lo)
        hi = float(hi)
    except Exception:
        return 0.001

    if hi == lo:
        return 1.0

    t = (xv - lo) / (hi - lo)
    if (crit_type or '').strip().lower() == 'negative':
        t = 1.0 - t

    # keep consistent with value functions used elsewhere
    if t < 0.001:
        return 0.001
    if t > 1.0:
        return 1.0
    return float(t)


def _invert_piecewise(points, u):
    """Invert a monotone piecewise-linear function defined by (x,y) points.

    Returns x such that y(x) == u (clamped to the y-range).
    """
    if not points:
        raise ValueError('Cannot invert empty value function points')

    pts = [(float(p[0]), float(p[1])) for p in points]
    pts = sorted(pts, key=lambda p: p[0])

    # clamp u to [min_y, max_y]
    ys = [p[1] for p in pts]
    min_y = min(ys)
    max_y = max(ys)
    try:
        uu = float(u)
    except Exception:
        uu = min_y
    if uu < min_y:
        uu = min_y
    if uu > max_y:
        uu = max_y

    # Find a segment where uu lies between y0 and y1
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if y0 == y1:
            # plateau: if uu equals this plateau, return left x
            if uu == y0:
                return x0
            continue
        lo_y = min(y0, y1)
        hi_y = max(y0, y1)
        if lo_y <= uu <= hi_y:
            t = (uu - y0) / (y1 - y0)
            return x0 + t * (x1 - x0)

    # If not found (numerical corner), snap to closest endpoint
    # Prefer the x associated with uu at extremes.
    if abs(uu - pts[0][1]) <= abs(uu - pts[-1][1]):
        return pts[0][0]
    return pts[-1][0]


def _eval_piecewise(points, x):
    """Evaluate a piecewise-linear function defined by (x,y) points.

    - Clamps x to the x-range of the points.
    - Assumes points are monotone in x (not necessarily in y).
    - Returns a float y.
    """
    if not points:
        raise ValueError('Cannot evaluate empty value function points')

    pts = [(float(p[0]), float(p[1])) for p in points]
    pts = sorted(pts, key=lambda p: p[0])

    try:
        xv = float(x)
    except Exception:
        xv = pts[0][0]

    # clamp x to domain
    if xv <= pts[0][0]:
        return float(pts[0][1])
    if xv >= pts[-1][0]:
        return float(pts[-1][1])

    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if x0 <= xv <= x1:
            if x1 == x0:
                return float(y0)
            t = (xv - x0) / (x1 - x0)
            return float(y0 + t * (y1 - y0))

    # Fallback (numerical corner): return closest endpoint
    if abs(xv - pts[0][0]) <= abs(xv - pts[-1][0]):
        return float(pts[0][1])
    return float(pts[-1][1])


def _load_criteria_ranges(criteria_csv_path):
    """Return mapping: criterion -> (min,max,type)."""
    ranges = {}
    with open(criteria_csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get('name') or '').strip()
            if not name:
                continue
            try:
                lo = float(row.get('min'))
            except Exception:
                lo = None
            try:
                hi = float(row.get('max'))
            except Exception:
                hi = None
            crit_type = (row.get('type') or '').strip().lower() or None
            ranges[name] = (lo, hi, crit_type)
    return ranges


def _load_vf_points(value_functions_csv_path):
    """Return mapping: criterion -> list of [x,y] points."""
    vf_points = {}
    with open(value_functions_csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get('name') or '').strip()
            pts_raw = row.get('elicited_points')
            if not name or not pts_raw:
                continue
            try:
                pts = json.loads(pts_raw)
            except Exception:
                pts = ast.literal_eval(pts_raw)
            vf_points[name] = pts
    return vf_points


def remap_bwt_results_for_country(
    elicit_num,
    country,
    script_dir=None,
    output_dir=None,
):
    """Create a country-adjusted BWT results CSV.

    The BWT CSV stores a single `Value` for each comparison. In this codebase, that `Value`
    is interpreted as:
      - if Type == 'best'  -> value is on the Reference criterion scale
      - if Type == 'worst' -> value is on the Other criterion scale

    Requested policy:
    - Compare the general (folder-level) `criteria.csv` with the country-specific
      `<country>/criteria.csv` for the same elicitation.
    - When the range differs, linearly rescale `Value` from the global domain into
      the country domain.

    Compatibility note:
    - Qualitative criteria may be stored in a raw 1–6 scale in BWT elicitation, while the
      country value functions may have been converted to a 0–1 domain. For these, we still
      remap from [1,6] into the country value-function x-range.

    After any remap, we recompute `a` as: a = 1 / vf_country(Value_new).
    """

    if script_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    if output_dir is None:
        output_dir = os.path.join(script_dir, 'weight_spaces')

    elicit_num = int(elicit_num)
    country = str(country).strip().upper()

    bwt_in = os.path.join(script_dir, 'weight_spaces', f'BWT_results_{elicit_num}.csv')
    bwt_out = os.path.join(output_dir, f'BWT_results_{elicit_num}_{country}.csv')

    criteria_csv = os.path.join(script_dir, 'elicitation_results', str(elicit_num), 'criteria.csv')
    country_criteria_csv = os.path.join(script_dir, 'elicitation_results', str(elicit_num), country, 'criteria.csv')
    vf_csv = os.path.join(script_dir, 'elicitation_results', str(elicit_num), country, 'value_functions.csv')

    ranges_global = _load_criteria_ranges(criteria_csv)
    ranges_country = _load_criteria_ranges(country_criteria_csv) if os.path.exists(country_criteria_csv) else {}
    vf_points = _load_vf_points(vf_csv)

    qualitative_indicators = {
        'Design Maturity': 'positive',
        'Licensing Status': 'positive',
        'Supplier Availbility': 'positive',
        'Design Complexity': 'negative',
        'Construction Complexity': 'positive',
    }

    with open(bwt_in, 'r', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames or 'Value' not in fieldnames:
        raise ValueError(f"Unexpected BWT file schema in {bwt_in}")

    for row in rows:
        typ = (row.get('Type') or '').strip().lower()
        ref = (row.get('Reference') or '').strip()
        other = (row.get('Other') or '').strip()

        crit_for_value = ref if typ == 'best' else other
        if not crit_for_value:
            continue
        if crit_for_value not in vf_points:
            continue

        try:
            x_old = float(row.get('Value'))
        except Exception:
            continue

        pts = vf_points[crit_for_value]
        xs = [float(p[0]) for p in pts]
        vf_lo = min(xs)
        vf_hi = max(xs)

        # Source range: what `Value` is expressed in.
        if crit_for_value in qualitative_indicators:
            src_lo, src_hi = 1.0, 6.0
        else:
            src_lo, src_hi, _crit_type = ranges_global.get(crit_for_value, (None, None, None))
            if src_lo is None or src_hi is None:
                continue

        # Target range:
        # - qualitative: always remap into the country VF x-domain (typically [0,1])
        # - quantitative: ONLY remap when country criteria ranges differ from global criteria ranges
        do_remap = False
        if crit_for_value in qualitative_indicators:
            tgt_lo, tgt_hi = vf_lo, vf_hi
            do_remap = True
        else:
            c_lo, c_hi, _c_type = ranges_country.get(crit_for_value, (None, None, None))
            if c_lo is not None and c_hi is not None:
                if not (np.isclose(float(src_lo), float(c_lo), atol=1e-9) and np.isclose(float(src_hi), float(c_hi), atol=1e-9)):
                    tgt_lo, tgt_hi = float(c_lo), float(c_hi)
                    do_remap = True

        if not do_remap:
            continue

        src_span = float(src_hi) - float(src_lo)
        tgt_span = float(tgt_hi) - float(tgt_lo)
        if abs(src_span) <= 1e-12 or abs(tgt_span) <= 1e-12:
            continue

        t = (x_old - float(src_lo)) / src_span
        
        # Warn if value is outside the source range (but allow extrapolation)
        if t < 0.0 or t > 1.0:
            print(f"  ⚠ Warning: Value {x_old} for '{crit_for_value}' is outside source range [{src_lo}, {src_hi}] (extrapolating)")

        x_new = float(tgt_lo + t * tgt_span)
        row['Value'] = str(float(x_new))

        try:
            vf_val = float(_eval_piecewise(pts, x_new))
        except Exception:
            vf_val = None
        if vf_val is not None:
            if vf_val <= 0.0:
                vf_val = 0.001
            row['a'] = str(float(1.0 / vf_val))

    with open(bwt_out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return bwt_out


def compute_bwt_weights_for_country_file(
    *,
    criteria_csv,
    bwt_results_csv,
    value_functions_csv,
    output_csv,
):
    """Solve BWT once and write a single-weight-per-criterion CSV (same row format as existing *_weights.csv files)."""
    from pile_bwt import bwt

    # Build dict_data (criteria + comparisons)
    dict_data = load_criteria(criteria_csv, bwt_results_csv)

    # Attach country-specific value functions
    vf_map, _conf_map = load_value_functions_with_confidence(value_functions_csv)
    for gdata in dict_data.values():
        for crit_name, crit in gdata['criteria'].items():
            crit['value_function'] = vf_map[crit_name]

    result = bwt(dict_data)
    weights_dict = result['criteria_weights']

    # Flatten in the canonical dict_data order
    crit_names = [crit_name for group_data in dict_data.values() for crit_name in group_data['criteria'].keys()]
    weights_flat = []
    for gname, gdata in dict_data.items():
        for crit_name in gdata['criteria'].keys():
            weights_flat.append(weights_dict[gname][crit_name])

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        for crit_name, w in zip(crit_names, weights_flat):
            writer.writerow([crit_name, float(w)])

    return output_csv

def load_alternatives(file_path):
    """Load alternatives from a CSV file and parse distributions as dictionaries."""
    alternatives = []
    with open(file_path, mode='r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            alternative = {}
            for key, value in row.items():
                if key != 'name':
                    # Skip empty values
                    if value is None or str(value).strip() == '':
                        continue
                    
                    # Try to parse as JSON first (for QI_dist and other distributions)
                    try:
                        alternative[key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        # Fall back to literal_eval for backward compatibility
                        try:
                            alternative[key] = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                            # If both fail, keep as string
                            alternative[key] = value
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


def load_value_functions_with_confidence(file_path_value_functions):
    """Load value functions from CSV and extract confidence levels.
    Returns two dicts:
    - vfs: mapping criterion name -> callable(x) in [0.001, 1.0]
    - confidences: mapping criterion name -> confidence level (0-4)
    """
    import csv
    import ast
    vfs = {}
    confidences = {}
    with open(file_path_value_functions, mode='r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            if not row:
                continue
            crit_name = (row.get('name') or '').strip()
            # Extract confidence level
            try:
                conf = float(row.get('confidence', 2.0))
            except Exception:
                conf = 2.0
            
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
                        confidences[crit_name] = conf
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
                    confidences[crit_name] = conf

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
                    confidences[crit_name] = conf

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
                    confidences[crit_name] = conf
            else:
                # no pts and no usable expr: skip
                continue
    return vfs, confidences

def setup_qi_country_folders(qi_elicitation_dirs):
    """
    Create country subfolders in QI elicitation directories with filtered data.
    
    For each QI elicitation folder:
    - Create country subfolders (IT, CH, FR, PO)
    - Filter alternatives data to only include columns for that country
    - Remove columns tagged with 'CC' (cross-country/not country-specific)
    - Convert from indicator-format to alternative-format
    
    Note: Value functions are not copied as they become linear after conversion.
    
    Args:
        qi_elicitation_dirs: List of paths to QI-only elicitation directories
    """
    import os
    from pathlib import Path
    
    if not qi_elicitation_dirs:
        return
    
    for qi_dir in qi_elicitation_dirs:
        qi_dir_path = Path(qi_dir)
        
        # Check if alternatives.csv exists at root level
        root_alt_file = qi_dir_path / 'alternatives.csv'
        
        if not root_alt_file.exists():
            print(f"ℹ No alternatives.csv at root of {qi_dir}, skipping folder setup")
            continue
        
        # Load root alternatives file (QI format: indicator x alternative)
        import pandas as pd
        root_alt_df = pd.read_csv(root_alt_file)
        
        # Extract all alternative columns (exclude 'indicator' and 'confidence')
        all_alt_cols = [col for col in root_alt_df.columns 
                       if col not in ['indicator', 'confidence']]
        
        # Detect country codes dynamically from folder structure
        # Check if country folders already exist
        existing_countries = [d.name for d in qi_dir_path.iterdir() if d.is_dir() and d.name in ['IT', 'CH', 'FR', 'PO']]
        countries_to_create = existing_countries if existing_countries else ['IT', 'CH', 'FR', 'PO']
        
        # Create country subfolders with filtered data
        for country in countries_to_create:
            country_dir = qi_dir_path / country
            country_dir.mkdir(exist_ok=True)
            
            # Filter columns: keep indicator and non-CC alternatives
            cols_to_keep = ['indicator']
            for col in all_alt_cols:
                if 'CC' not in col:
                    cols_to_keep.append(col)
            
            country_alt_df = root_alt_df[cols_to_keep].copy()
            
            # Filter rows: remove starting points and keep only country-specific data
            # Keep rows that either:
            # 1. Don't have a country tag (generic indicators)
            # 2. Have the country tag (-IT, -FR, etc.) without "starting point"
            # 3. Remove all "starting point" rows
            mask = ~country_alt_df['indicator'].str.contains('starting point', case=False, na=False)
            
            # Additionally, for country-specific rows, only keep those matching this country
            for idx, row in country_alt_df[mask].iterrows():
                indicator = str(row['indicator'])
                # If indicator has a country tag (e.g., "- IT", "- FR"), check if it matches
                if ' - ' in indicator:
                    parts = indicator.rsplit(' - ', 1)
                    if len(parts) == 2:
                        tag = parts[1].strip()
                        # If tag is a known country code, only keep if it matches our country
                        if tag in ['IT', 'CH', 'FR', 'PO']:
                            if tag != country:
                                mask[idx] = False
            
            country_alt_df = country_alt_df[mask].copy()
            
            # Save filtered alternatives.csv
            country_alt_file = country_dir / 'alternatives.csv'
            country_alt_df.to_csv(country_alt_file, index=False)
            print(f"✓ Created {qi_dir}/{country}/alternatives.csv ({len(country_alt_df)} rows)")


def transform_qi_format_to_alternative_format(qi_df):
    """
    Transform QI data from indicator format (indicators x alternatives) 
    to alternative format (alternatives x criteria).
    Also strips country tags from column names and cleans whitespace.
    """
    # Rename 'indicator' column to 'name' for consistency
    if 'indicator' in qi_df.columns:
        # Transpose: make alternatives into rows, indicators into columns
        transformed_df = qi_df.set_index('indicator').T.reset_index()
        transformed_df.rename(columns={'index': 'name'}, inplace=True)
        
        # Strip whitespace from 'name' column (alternatives)
        transformed_df['name'] = transformed_df['name'].str.strip()
        
        # Strip country tags from column names (e.g., "Licensing Status - IT" -> "Licensing Status")
        new_columns = {}
        for col in transformed_df.columns:
            if col == 'name':
                new_columns[col] = col
            else:
                # Remove country tags like " - IT", " - FR", etc.
                import re
                cleaned = re.sub(r'\s*-\s*(IT|CH|FR|PO)\s*$', '', col)
                new_columns[col] = cleaned
        
        transformed_df.rename(columns=new_columns, inplace=True)
        return transformed_df
    return qi_df


def combine_alternatives_by_country(elicitation_dirs, selected_country, output_dir, qi_elicitation_dirs=None):
    """
    Combine alternatives from multiple elicitations for a specific country.
    
    For quantitative indicators: keep value as is (assumed identical across elicitations).
    For qualitative indicators (QI): create a QI_dist with one entry per elicitation source.
    QI indicators are automatically detected from qi_elicitation_dirs folders.
    
    Args:
        elicitation_dirs: List of paths to elicitation result directories (e.g., ["elicitation_results/1", "elicitation_results/2"])
        selected_country: Country code (e.g., "IT", "FR")
        output_dir: Directory to save the combined alternatives file
        qi_elicitation_dirs: List of paths to QI-only elicitation directories (default: None)
    """
    import json
    import os
    import pandas as pd
    
    # Setup country folders in QI elicitation directories if needed
    if qi_elicitation_dirs:
        setup_qi_country_folders(qi_elicitation_dirs)
    
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
    qi_alternatives_dfs = []
    qi_columns_set = set()  # Track which columns appear in QI folders
    if qi_elicitation_dirs:
        for qi_dir in qi_elicitation_dirs:
            qi_path = os.path.join(qi_dir, selected_country, "alternatives.csv")
            if os.path.exists(qi_path):
                qi_df = pd.read_csv(qi_path)
                # Transform QI format to alternative format
                qi_df = transform_qi_format_to_alternative_format(qi_df)
                qi_alternatives_dfs.append(qi_df)
                # Track columns that appear in QI folders
                qi_columns_set.update(col for col in qi_df.columns if col != 'name')
    
    if not all_alternatives:
        raise ValueError("No alternatives files loaded")
    
    # Get list of alternative names from first file
    alt_names = all_alternatives[0]['name'].tolist()
    criteria_columns = [col for col in all_alternatives[0].columns if col != 'name']
    
    # Dynamically determine QI indicators: detect from regular elicitations too
    # Known QI indicators that might appear in the data
    known_qi_indicators = {'Design Maturity', 'Licensing Status', 'Supplier Availbility', 
                          'Design Complexity', 'Construction Complexity'}
    
    # Combine QI columns from both QI folders and regular elicitations
    for col in criteria_columns:
        if col in known_qi_indicators:
            qi_columns_set.add(col)
    
    qualitative_indicators = list(qi_columns_set)
    if qualitative_indicators:
        print(f"ℹ Detected QI indicators: {qualitative_indicators}")
    else:
        print("ℹ No QI indicators detected")
    
    # Build combined alternatives
    combined_data = []
    
    for alt_name in alt_names:
        row_data = {'name': alt_name}
        
        for criterion in criteria_columns:
            is_qi = criterion in qualitative_indicators
            
            # Collect values from main elicitations with their elicitation index
            values_list = []  # List of (value_str, elicit_idx) tuples for QI
            
            for elicit_idx, df in enumerate(all_alternatives):
                if alt_name in df['name'].values:
                    cell_value = df[df['name'] == alt_name][criterion].iloc[0]
                    
                    # Handle NaN and empty values
                    if pd.isna(cell_value) or str(cell_value).strip() == '':
                        continue
                    
                    values_list.append((cell_value, elicit_idx))
            
            # If this is a qualitative indicator and QI data exists, add those values too
            if is_qi and qi_alternatives_dfs:
                for qi_idx, qi_df in enumerate(qi_alternatives_dfs):
                    if alt_name in qi_df['name'].values:
                        qi_value = qi_df[qi_df['name'] == alt_name][criterion].iloc[0]
                        if not pd.isna(qi_value) and str(qi_value).strip() != '':
                            # Use a special index to indicate QI folder (e.g., len(all_alternatives) + qi_idx)
                            qi_elicit_idx = len(all_alternatives) + qi_idx
                            values_list.append((qi_value, qi_elicit_idx))
            
            if not values_list:
                # No values found for this criterion
                continue
            
            if is_qi:
                # For qualitative indicators, ALWAYS create QI_dist with one entry per elicitation source
                # Each entry is [value, confidence] where confidence defaults to 2.0
                qi_dist_entries = []
                
                for val_str, elicit_idx in values_list:
                    try:
                        # Try to parse as JSON first (for Discrete distributions)
                        parsed_val = json.loads(val_str)
                        if isinstance(parsed_val, dict) and 'Discrete' in parsed_val:
                            # Extract the numeric value from Discrete - take the first value only
                            numeric_vals = parsed_val['Discrete'][0]
                            if isinstance(numeric_vals, list) and len(numeric_vals) > 0:
                                # If it's a list, take first element
                                num_val = float(numeric_vals[0])
                            else:
                                # If it's a single value, use it
                                num_val = float(numeric_vals)
                            qi_dist_entries.append([num_val, 2.0])  # One entry per source
                        else:
                            # Try to convert to float
                            qi_dist_entries.append([float(val_str), 2.0])
                    except (json.JSONDecodeError, ValueError, TypeError):
                        # Try direct float conversion
                        try:
                            qi_dist_entries.append([float(val_str), 2.0])
                        except ValueError:
                            pass  # Skip non-numeric values
                
                if qi_dist_entries:
                    # Keep all entries in order (one per elicitation source), don't deduplicate
                    row_data[criterion] = json.dumps({'QI_dist': qi_dist_entries})
                else:
                    # If we couldn't parse anything, keep first value
                    row_data[criterion] = values_list[0][0]
            else:
                # For quantitative indicators, use Discrete distribution with unique values
                numeric_values = []
                for val_str, _ in values_list:
                    try:
                        parsed_val = json.loads(val_str)
                        if isinstance(parsed_val, dict) and 'Discrete' in parsed_val:
                            # Extract numeric values from Discrete
                            numeric_vals = parsed_val['Discrete'][0]
                            numeric_values.extend([float(v) for v in numeric_vals])
                        else:
                            numeric_values.append(float(val_str))
                    except (json.JSONDecodeError, ValueError, TypeError):
                        try:
                            numeric_values.append(float(val_str))
                        except ValueError:
                            pass
                
                if numeric_values:
                    numeric_values = sorted(list(set(numeric_values)))
                    row_data[criterion] = json.dumps({'Discrete': [numeric_values]})
                else:
                    row_data[criterion] = values_list[0][0]
        
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


#################################################################################
# Data Loading and Preprocessing
def startup(file_path_criteria, file_path_weight_elicitations, file_path_value_functions, file_path_alternatives="alternatives.csv"):
    # Load value functions and criteria and establish canonical ordering
    # This is done to ensure that criteria are consistently ordered across different elicitation files
    # And that we do not mix up value functions in other steps
    first_dict = load_criteria_definitions(file_path_criteria)
    crit_names = [crit_name for group_data in first_dict.values() for crit_name in group_data['criteria'].keys()]

    # Number of criteria is saved as a variable since we are going to use it multiple times
    num_criteria = len(crit_names)
    print(f"Number of criteria identified: {num_criteria}")

    # mapping from criterion name to its index in crit_names
    crit_index = {name: idx for idx, name in enumerate(crit_names)}

    # Initialize list of lists: each index corresponds to a criterion in `crit_names`
    # And built it by reading the separate value function CSVs (one per elicitation)
    vf_list = [[] for _ in range(num_criteria)]
    conf_list = [[] for _ in range(num_criteria)]
    for vp in file_path_value_functions:
        vf_map, conf_map = load_value_functions_with_confidence(vp)
        for idx, crit_name in enumerate(crit_names):
            vf_list[idx].append(vf_map[crit_name])
            conf_list[idx].append(conf_map[crit_name])    
            
    # Construct list of dict_data for each elicitation
    # Load criteria.csv which contains criteria definitions
    # Adds info about value functions from vf_list for each elicitation
    print("Loading criteria definitions and attaching value functions...")
    dict_data_list = []
    print("Starting loop over weight elicitation files...")
    for i, fp in enumerate(file_path_weight_elicitations):
        print(f"Processing file {i+1}/{len(file_path_weight_elicitations)}: {fp}")  # Debugging: Track loop progress
        dict_data = load_criteria(file_path_criteria, fp)
        
        # Attach value functions
        for gname, gdata in dict_data.items():
            for crit_name, crit in gdata['criteria'].items():
                idx = crit_index[crit_name]
                vf_raw = vf_list[idx][i]
                # If the loaded VF only contains a single node, assume linear behaviour
                # across the criterion min/max and wrap it accordingly.
                try:
                    xs_attr = getattr(vf_raw, '_xs', None)
                    if xs_attr is not None and len(xs_attr) == 1:
                        lo = crit.get('min', crit.get('min_value') or 0.0)
                        hi = crit.get('max', crit.get('max_value') or (lo + 1.0))
                        def make_linear(lo, hi):
                            def vf(x):
                                try:
                                    xv = float(x)
                                except Exception:
                                    xv = lo
                                if hi == lo:
                                    val = 1.0
                                else:
                                    val = (xv - lo) / (hi - lo)
                                import numpy as _np
                                return float(_np.clip(max(val, 0.001), 0.001, 1.0))
                            try:
                                import numpy as _np
                                vf._xs = _np.array([lo, hi], dtype=float)
                                vf._ys = _np.array([0.001, 1.0], dtype=float)
                            except Exception:
                                pass
                            return vf
                        crit['value_function'] = make_linear(lo, hi)
                    else:
                        crit['value_function'] = vf_raw
                except Exception:
                    crit['value_function'] = vf_raw
        dict_data_list.append(dict_data)

    # Debugging: Verify final dict_data_list
    # print("Final dict_data_list contains:", len(dict_data_list), "entries")

    # Load data for alternatives
    alternatives = load_alternatives(file_path_alternatives)
    return dict_data_list, crit_index, vf_list, conf_list, alternatives
#################################################################################

#################################################################################
# Data Conversion and Normalization Functions
#################################################################################

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
        
        # Detect available country folders dynamically
        countries = [d.name for d in base_dir.iterdir() if d.is_dir() and len(d.name) == 2 and d.name.isupper()]
        if not countries:
            countries = ['IT', 'CH', 'FR', 'PO']  # Fallback
        
        for country in countries:
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

        # Update folder-level criteria.csv to reflect value-space conversion.
        # After conversion, these qualitative indicators are values in [0, 1]
        # and higher is always better.
        criteria_file = base_dir / 'criteria.csv'
        if criteria_file.exists():
            with open(criteria_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                criteria_fieldnames = reader.fieldnames
                criteria_rows = list(reader)

            if criteria_fieldnames is not None:
                # Normalize qualitative indicators to 0-1 domain and set type polarity
                for crit_row in criteria_rows:
                    name = (crit_row.get('name') or '').strip()
                    if name in qualitative_indicators:
                        # All qualitative indicators are converted to 0..1 scale
                        crit_row['min'] = '0'
                        crit_row['max'] = '1'
                        # Polarity: keep negative for complexity, positive otherwise
                        if name in ('Design Complexity', 'Construction Complexity'):
                            crit_row['type'] = 'negative'
                        else:
                            crit_row['type'] = 'positive'

                with open(criteria_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=criteria_fieldnames)
                    writer.writeheader()
                    writer.writerows(criteria_rows)

#################################################################################