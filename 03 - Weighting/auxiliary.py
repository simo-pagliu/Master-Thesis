import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import glob
import json
import ast
import warnings

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
    # Load value functions from value_functions.csv in the same directory
    base_dir = os.path.dirname(os.path.abspath(file_path))
    vf_path = os.path.join(base_dir, 'value_functions.csv')
    vfs = import_value_functions(vf_path)
    # Attach to criteria if names match; provide a default linear mapping otherwise
    for name, meta in criteria.items():
        if name in vfs:
                vf_raw = vfs[name]
                # If only a single elicited node was present, assume linear behaviour
                # between criterion min and max (0.001..1.0)
                try:
                    xs_attr = getattr(vf_raw, '_xs', None)
                    # If only a single node is present, wrap into a linear mapping
                    if xs_attr is not None and len(xs_attr) == 1:
                        lo = meta.get('min', 0.0)
                        hi = meta.get('max', lo + 1.0)
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
                                return float(np.clip(max(val, 0.001), 0.001, 1.0))
                            try:
                                vf._xs = np.array([lo, hi], dtype=float)
                                vf._ys = np.array([0.001, 1.0], dtype=float)
                            except Exception:
                                pass
                            return vf
                        meta['value_function'] = make_linear(lo, hi)
                    else:
                        # If the loaded VF exposes node arrays that look like
                        # normalized coordinates (within [0,1]), wrap the
                        # original callable so it receives normalized input and
                        # attach scaled node arrays in criterion units.
                        if xs_attr is not None:
                            try:
                                xs_arr = np.array(xs_attr, dtype=float)
                                if xs_arr.size > 1 and xs_arr.min() >= -1e-9 and xs_arr.max() <= 1.0000001:
                                    lo = meta.get('min', 0.0)
                                    hi = meta.get('max', lo + 1.0)
                                    orig_vf = vf_raw
                                    def make_wrapped(orig_vf, lo, hi, xs_arr):
                                        def vf_wrapped(x):
                                            try:
                                                xv = float(x)
                                            except Exception:
                                                xv = lo
                                            # normalize to 0..1 domain relative to criterion
                                            if hi == lo:
                                                xn = 0.0
                                            else:
                                                xn = (xv - lo) / (hi - lo)
                                            # Map normalized xn into original node domain
                                            xs_min = float(xs_arr.min())
                                            xs_max = float(xs_arr.max())
                                            if xs_max > xs_min:
                                                x_for_orig = xs_min + xn * (xs_max - xs_min)
                                            else:
                                                x_for_orig = xn
                                            try:
                                                val = float(orig_vf(x_for_orig))
                                            except Exception:
                                                val = 0.001
                                            return float(np.clip(val, 0.001, 1.0))
                                        # attach scaled node arrays for callers that inspect them
                                        try:
                                            # scale original normalized nodes to criterion units
                                            scaled_xs = np.array(xs_arr * (hi - lo) + lo, dtype=float)
                                            vf_wrapped._xs = scaled_xs
                                            vf_wrapped._ys = np.array(getattr(orig_vf, '_ys', None) or [], dtype=float)
                                        except Exception:
                                            pass
                                        return vf_wrapped
                                    meta['value_function'] = make_wrapped(orig_vf, lo, hi, xs_arr)
                                else:
                                    meta['value_function'] = vf_raw
                            except Exception:
                                meta['value_function'] = vf_raw
                        else:
                            meta['value_function'] = vf_raw
                except Exception:
                    meta['value_function'] = vfs[name]
        else:
            # Default linear mapping between min -> 0.001 and max -> 1.0
            lo = meta.get('min', 0.0)
            hi = meta.get('max', lo + 1.0)
            def make_default(lo, hi):
                def vf(x):
                    try:
                        xv = float(x)
                    except Exception:
                        xv = lo
                    if hi == lo:
                        val = 1.0
                    else:
                        val = (xv - lo) / (hi - lo)
                    return float(np.clip(max(val, 0.001), 0.001, 1.0))
                return vf
            meta['value_function'] = make_default(lo, hi)
    return criteria



def import_value_functions(vf_csv_path):
    """
    Read a value functions CSV and return a dict mapping criterion name -> callable

    - Reads `elicited_points` (Python list literal) and `elicitation_meta` (JSON).
    - Does NOT evaluate any `value_function` column (deprecated).
    - Builds a simple piecewise-linear interpolator using if/else (no external deps).
    - If `elicitation_meta` does not indicate a piecewise/monotonic fit, a warning
      is emitted but the piecewise-linear interpolator is still used.
    - Ensures outputs are clipped to [0.001, 1.0] and any explicit 0.0 points
      are increased to 0.001.
    """
    vfs = {}
    if not os.path.exists(vf_csv_path):
        raise FileNotFoundError(vf_csv_path)

    with open(vf_csv_path, mode='r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('name')
            if not name:
                continue

            # Parse elicited_points (expect a Python list literal)
            pts_raw = row.get('elicited_points', '')
            try:
                points = ast.literal_eval(pts_raw) if pts_raw else []
            except Exception:
                warnings.warn(f"Could not parse elicited_points for '{name}', falling back to empty points")
                points = []

            # Parse elicitation_meta (expect JSON)
            meta_raw = row.get('elicitation_meta', '')
            intended = False
            try:
                meta = json.loads(meta_raw) if meta_raw else {}
                # Basic heuristic: if fit_type explicitly Piecewise Linear or mode suggests manual/monotonic
                if meta.get('fit_type') == 'Piecewise Linear' or meta.get('mode') in ('Monotonic', 'Manual'):
                    intended = True
            except Exception:
                meta = {}

            if not intended:
                warnings.warn(f"Value-function metadata for '{name}' does not indicate piecewise linear elicitation; using piecewise-linear fallback.")

            # Build piecewise interpolator using simple if/else logic
            # Normalize/clean points: ensure sorted by x and convert y==0 to 0.001
            try:
                pts = sorted([(float(p[0]), float(p[1])) for p in points], key=lambda x: x[0])
            except Exception:
                pts = []

            # If no points, create a trivial constant function returning 0.001
            if len(pts) == 0:
                def vf_constant(x, _v=0.001):
                    return float(_v)
                vf_constant._xs = np.array([0.0])
                vf_constant._ys = np.array([0.001])
                vfs[name] = vf_constant
                continue

            # If a single point, return constant function with clipping
            if len(pts) == 1:
                y0 = 0.001 if pts[0][1] == 0.0 else pts[0][1]
                def vf_single(x, _y=y0):
                    return float(np.clip(_y, 0.001, 1.0))
                vf_single._xs = np.array([pts[0][0]])
                vf_single._ys = np.array([y0])
                vfs[name] = vf_single
                continue

            xs = np.array([p[0] for p in pts], dtype=float)
            ys = np.array([0.001 if p[1] == 0.0 else p[1] for p in pts], dtype=float)

            def make_piecewise(xs, ys):
                def vf(x):
                    try:
                        xv = float(x)
                    except Exception:
                        xv = float(xs[0])

                    # Left extrapolation
                    if xv <= xs[0]:
                        x0, x1 = xs[0], xs[1]
                        y0, y1 = ys[0], ys[1]
                        slope = (y1 - y0) / (x1 - x0) if (x1 != x0) else 0.0
                        val = y0 + slope * (xv - x0)
                    # Right extrapolation
                    elif xv >= xs[-1]:
                        x0, x1 = xs[-2], xs[-1]
                        y0, y1 = ys[-2], ys[-1]
                        slope = (y1 - y0) / (x1 - x0) if (x1 != x0) else 0.0
                        val = y1 + slope * (xv - xs[-1])
                    else:
                        # Find segment
                        idx = np.searchsorted(xs, xv) - 1
                        x0, x1 = xs[idx], xs[idx + 1]
                        y0, y1 = ys[idx], ys[idx + 1]
                        slope = (y1 - y0) / (x1 - x0) if (x1 != x0) else 0.0
                        val = y0 + slope * (xv - x0)

                    if np.isnan(val):
                        val = 0.001
                    # Clip to allowed range
                    return float(np.clip(val, 0.001, 1.0))
                return vf
            v = make_piecewise(xs, ys)
            # attach node arrays so callers can invert the VF if needed
            try:
                v._xs = np.array(xs, dtype=float)
                v._ys = np.array(ys, dtype=float)
            except Exception:
                v._xs = np.array(xs)
                v._ys = np.array(ys)
            vfs[name] = v

    return vfs



def plot_results(result, criteria_names):
    plt.close('all')
    # larger figure and bottom margin to avoid clipping rotated x-labels
    plt.figure(figsize=(10, 6))
    try:
        plt.gcf().subplots_adjust(bottom=0.5)
    except Exception:
        pass

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

    # thinner bars and increased spacing for readability
    n = len(names)
    if n == 0:
        return
    bar_width = 0.35
    spacing = 1.6
    x = np.arange(n) * spacing
    plt.bar(x, weights, width=bar_width)
    plt.xticks(x, names, rotation=25, ha='right', fontsize=10)
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
