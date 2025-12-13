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
