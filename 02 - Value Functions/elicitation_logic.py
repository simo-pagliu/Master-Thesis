# elicitation_logic.py
import pandas as pd
import numpy as np
import json
import os
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import PchipInterpolator

class ElicitationProcess:
    def __init__(self):
        self.df = None
        self.points = []
        self.current_attribute_index = 0
        # thresholds and tail values used for piecewise behaviour
        self.lower_threshold = None
        self.upper_threshold = None
        # values for tails (left/right) when outside the fitted interval
        self.left_tail_value = None
        self.right_tail_value = None
        # path to the CSV file currently loaded
        self.file_path = None

    def load_data(self, file_path):
        """Load and validate the CSV file."""
        self.df = pd.read_csv(file_path)
        self.file_path = file_path
        # choose a results VF file for this loaded dataset (value_functions_1.csv, _2, ...)
        out_dir = os.path.dirname(self.file_path) or '.'
        i = 1
        while True:
            candidate = os.path.join(out_dir, f'value_functions_{i}.csv')
            if not os.path.exists(candidate):
                self.results_vf_path = candidate
                break
            i += 1
        required_columns = ["name", "min", "max"]
        if not all(col in self.df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {', '.join(required_columns)}")
        return self.df

    def load_quantitative_folder(self, folder_path):
        """Load the `quantitative` folder which contains `criteria.csv` and `guideline.txt`.

        The guideline uses format B: for each criterion a line like
          Criterion Name: [IT,FR,PO,CH]
        or with explicit ranges
          Criterion Name: [IT:10-20,CH:21-23,FR,PO]

        This function builds a DataFrame where per-country criteria are expanded
        into rows named 'Criterion Name - XX' and default criteria remain single-row.
        """
        folder = os.fspath(folder_path)
        crit_path = os.path.join(folder, 'criteria.csv')
        guide_path = os.path.join(folder, 'guideline.txt')
        if not os.path.exists(crit_path):
            raise FileNotFoundError(f"criteria.csv not found in {folder}")

        base_df = pd.read_csv(crit_path)
        # normalize name and group columns
        if 'name' not in base_df.columns:
            raise ValueError('criteria.csv must contain a name column')

        # parse guideline if present (also read Specifics: [IT,CH,...])
        guidelines = {}
        country_codes = []
        if os.path.exists(guide_path):
            with open(guide_path, 'r', encoding='utf-8') as gf:
                for ln in gf:
                    ln = ln.strip()
                    if not ln or ln.startswith('#'):
                        continue
                    low = ln.lower()
                    if low.startswith('specifics:') or low.startswith('specifics>'):
                        # accept 'Specifics: [IT,CH]' or 'Specifics> [IT,CH]'
                        if ':' in ln:
                            _, rhs = ln.split(':', 1)
                        else:
                            _, rhs = ln.split('>', 1)
                        rhs = rhs.strip()
                        if rhs.startswith('[') and rhs.endswith(']'):
                            rhs = rhs[1:-1]
                        country_codes = [p.strip() for p in rhs.split(',') if p.strip()]
                        continue
                    if ':' not in ln:
                        continue
                    key, rest = ln.split(':', 1)
                    key = key.strip()
                    rest = rest.strip()
                    if rest.startswith('[') and rest.endswith(']'):
                        inner = rest[1:-1]
                        # support comma-separated tokens, or colon-separated where
                        # a final numeric range follows country codes like
                        #    IT:PO:FR:CH:0-4.25  -> expand to IT:0-4.25, PO:0-4.25, ...
                        parts = []
                        if ',' in inner:
                            parts = [p.strip() for p in inner.split(',') if p.strip()]
                        elif inner.count(':') >= 2:
                            pieces = [p.strip() for p in inner.split(':') if p.strip()]
                            last = pieces[-1]
                            if '-' in last:
                                rng = last
                                codes = pieces[:-1]
                                parts = [f"{c}:{rng}" for c in codes]
                            else:
                                parts = pieces
                        else:
                            parts = [inner.strip()]
                        guidelines[key] = parts

        # expand into rows
        rows = []
        # helper to get base row for a criterion
        def get_base_row(name):
            try:
                r = base_df[base_df['name'].str.strip().str.lower() == name.strip().lower()]
                if r.shape[0] > 0:
                    return r.iloc[0].to_dict()
            except Exception:
                pass
            return None

        # If Specifics not provided, fall back to common set
        if not country_codes:
            country_codes = ['IT', 'CH', 'FR', 'PO']

        seen = set()
        for _, brow in base_df.iterrows():
            cname = str(brow.get('name'))
            # decide if this criterion has a guideline entry
            parts = guidelines.get(cname)
            if parts:
                # expand per token
                for token in parts:
                    token = token.strip()
                    if not token:
                        continue
                    if ':' in token:
                        # explicit range CC:low-high OR default:low-high
                        cc, rng = token.split(':', 1)
                        cc = cc.strip()
                        rng = rng.strip()
                        # parse numeric range
                        if '-' in rng:
                            a, b = [s.strip() for s in rng.split('-', 1)]
                            try:
                                lo = float(a)
                                hi = float(b)
                            except Exception:
                                lo = float(brow.get('min', 0))
                                hi = float(brow.get('max', 1))
                        else:
                            lo = float(brow.get('min', 0))
                            hi = float(brow.get('max', 1))

                        if cc.lower() == 'default':
                            # single folder-global entry using provided range
                            row = {
                                'name': cname,
                                'group': brow.get('group'),
                                'min': lo,
                                'max': hi,
                                'unit': brow.get('unit') if 'unit' in brow else ''
                            }
                            rows.append(row)
                        else:
                            # per-country explicit range
                            row = {
                                'name': f"{cname} - {cc}",
                                'group': brow.get('group'),
                                'min': lo,
                                'max': hi,
                                'unit': brow.get('unit') if 'unit' in brow else ''
                            }
                            rows.append(row)
                            seen.add((cname, cc))
                    else:
                        # token without ':' could be a country code or a plain range
                        if '-' in token:
                            # treat as default numeric range (e.g. "10-20")
                            try:
                                a, b = [s.strip() for s in token.split('-', 1)]
                                lo = float(a)
                                hi = float(b)
                                row = {
                                    'name': cname,
                                    'group': brow.get('group'),
                                    'min': lo,
                                    'max': hi,
                                    'unit': brow.get('unit') if 'unit' in brow else ''
                                }
                                rows.append(row)
                            except Exception:
                                # not a numeric range; fall through
                                pass
                        else:
                            cc = token.strip()
                            if cc.upper() in country_codes:
                                # use base min/max
                                row = {
                                    'name': f"{cname} - {cc}",
                                    'group': brow.get('group'),
                                    'min': float(brow.get('min', 0)),
                                    'max': float(brow.get('max', 1)),
                                    'unit': brow.get('unit') if 'unit' in brow else ''
                                }
                                rows.append(row)
                                seen.add((cname, cc))
                            else:
                                # unknown token; ignore
                                pass
            else:
                # not per-country -> keep single row
                row = {
                    'name': cname,
                    'group': brow.get('group'),
                    'min': float(brow.get('min', 0)),
                    'max': float(brow.get('max', 1)),
                    'unit': brow.get('unit') if 'unit' in brow else ''
                }
                rows.append(row)

        # Some guideline tokens may reference criteria not present in base_df; add them if possible
        for gname, parts in guidelines.items():
            # if gname already handled continue
            if any(r['name'].strip().lower() == gname.strip().lower() for r in rows):
                continue
            base = get_base_row(gname)
            if base is None:
                # create entries only for explicit ranges
                for token in parts:
                    if ':' in token:
                        cc, rng = token.split(':', 1)
                        cc = cc.strip()
                        try:
                            lo, hi = [float(x) for x in rng.split('-', 1)]
                        except Exception:
                            continue
                        rows.append({'name': f"{gname} - {cc}", 'group': '', 'min': lo, 'max': hi, 'unit': ''})
                continue

        df_rows = pd.DataFrame(rows)
        # ensure required columns
        for c in ['name', 'min', 'max']:
            if c not in df_rows.columns:
                df_rows[c] = ''

        # set process state
        self.df = df_rows
        self.file_path = folder
        # choose a results_vf_path inside the folder (global file)
        self.results_vf_path = os.path.join(folder, 'value_functions.csv')

        # If a value_functions.csv exists in the quantitative folder, merge its
        # elicited entries into the loaded dataframe so the UI shows saved defaults.
        if os.path.exists(self.results_vf_path):
            vf_df = pd.read_csv(self.results_vf_path)
            if 'name' in vf_df.columns:
                for _, vf_row in vf_df.iterrows():
                    raw_name = str(vf_row.get('name', '')).strip()
                    alt_name = raw_name.replace('/', ' - ')
                    alt_name_rev = raw_name.replace(' - ', '/')

                    # find matching index in self.df
                    mask = (self.df['name'] == raw_name) | (self.df['name'] == alt_name) | (self.df['name'] == alt_name_rev)
                    matches = self.df[mask]
                    if matches.shape[0] == 0:
                        # try case-insensitive match
                        try:
                            mask = (self.df['name'].str.lower() == raw_name.lower()) | (self.df['name'].str.lower() == alt_name.lower())
                            matches = self.df[mask]
                        except Exception:
                            matches = self.df[[]]
                    if matches.shape[0] > 0:
                        idx = matches.index[0]
                        if 'elicited_points' in vf_df.columns:
                            self.df.at[idx, 'elicited_points'] = vf_row.get('elicited_points', '')
                        if 'confidence' in vf_df.columns:
                            self.df.at[idx, 'confidence'] = vf_row.get('confidence', '')
                        if 'elicitation_meta' in vf_df.columns:
                            self.df.at[idx, 'elicitation_meta'] = vf_row.get('elicitation_meta', '')

        # Do not create per-country files here; saving will write a single
        # `value_functions.csv` in the quantitative folder using `Base/CC`
        # naming for country-specific rows.
        return self.df

    def get_saved_state_for_current_attribute(self):
        """Return saved elicited points, function string and meta for the current attribute if present."""
        if self.df is None:
            return None
        row = self.df.iloc[self.current_attribute_index]
        points = None
        func = None
        meta = None
        # Prefer defaults from an external `value_functions.csv` or any
        # `value_functions_*.csv` in the same folder as the loaded CSV. If
        # present, its rows (by name) override the per-row columns in the
        # loaded dataframe row for display only. Also respect the session
        # `results_vf_path` if set. Otherwise fall back to the values stored
        # in the loaded dataframe.
        try:
            out_dir = os.path.dirname(self.file_path) or '.'
            candidates = []
            # canonical name
            candidates.append(os.path.join(out_dir, 'value_functions.csv'))
            # session results path (may be value_functions_1.csv etc.)
            rv = getattr(self, 'results_vf_path', None)
            if rv:
                candidates.append(rv)
            # also consider any value_functions_*.csv present (pick latest if multiple)
            try:
                import glob
                patt = os.path.join(out_dir, 'value_functions_*.csv')
                found = sorted(glob.glob(patt))
                # add reversed so newer (lexicographically later) are preferred
                for f in reversed(found):
                    if f not in candidates:
                        candidates.append(f)
            except Exception:
                pass

            name = str(row.get('name')).strip()
            for ext_path in candidates:
                try:
                    if not ext_path or not os.path.exists(ext_path):
                        continue
                    vf_df = pd.read_csv(ext_path)
                    # try exact match then case-insensitive
                    match = None
                    if 'name' in vf_df.columns:
                        matches = vf_df[vf_df['name'] == name]
                        if matches.shape[0] == 0:
                            try:
                                matches = vf_df[vf_df['name'].str.lower() == name.lower()]
                            except Exception:
                                matches = pd.DataFrame()
                        if matches.shape[0] > 0:
                            match = matches.iloc[-1]
                    if match is not None:
                        if 'elicited_points' in match and pd.notna(match.get('elicited_points')):
                            try:
                                points = json.loads(match.get('elicited_points'))
                            except Exception:
                                points = None
                        if 'value_function' in match and pd.notna(match.get('value_function')):
                            func = match.get('value_function')
                        if 'elicitation_meta' in match and pd.notna(match.get('elicitation_meta')):
                            try:
                                meta = json.loads(match.get('elicitation_meta'))
                            except Exception:
                                meta = None
                        # top-level confidence column overrides meta.confidence
                        try:
                            if 'confidence' in match and pd.notna(match.get('confidence')):
                                try:
                                    confv = match.get('confidence')
                                    meta = meta or {}
                                    meta['confidence'] = int(confv)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        return {'points': points, 'value_function': func, 'meta': meta}
                except Exception:
                    # try next candidate
                    continue
        except Exception:
            pass

        except Exception:
            pass

        # fallback: read from the loaded dataframe row
        if 'elicited_points' in self.df.columns and pd.notna(row.get('elicited_points')):
            try:
                points = json.loads(row.get('elicited_points'))
            except Exception:
                points = None
        if 'value_function' in self.df.columns and pd.notna(row.get('value_function')):
            func = row.get('value_function')
        if 'elicitation_meta' in self.df.columns and pd.notna(row.get('elicitation_meta')):
            try:
                meta = json.loads(row.get('elicitation_meta'))
            except Exception:
                meta = None
        # also read a top-level 'confidence' column if present
        try:
            if 'confidence' in self.df.columns and pd.notna(row.get('confidence')):
                try:
                    confv = row.get('confidence')
                    # coerce to int if possible
                    meta = meta or {}
                    meta['confidence'] = int(confv)
                except Exception:
                    # ignore non-numeric confidence values
                    pass
        except Exception:
            pass
        return {'points': points, 'value_function': func, 'meta': meta}

    def save_current_state(self, degree=2, meta=None):
        """Save current points, computed value function string and meta into the loaded CSV file.

        This writes into columns: 'elicited_points' (JSON array), 'value_function' (string),
        'elicitation_meta' (JSON string). If the file doesn't exist or wasn't loaded, raises.
        """
        if self.df is None or self.file_path is None:
            raise RuntimeError("No CSV loaded to save to")

        idx = self.current_attribute_index
        # write points as JSON
        self.df.at[idx, 'elicited_points'] = json.dumps(self.points)

        # We no longer persist executable/lambda expressions to CSV for security and portability.
        # Only save elicited points and elicitation_meta (as JSON).
        self.df.at[idx, 'elicitation_meta'] = json.dumps(meta or {})

        # persist confidence as a top-level column if provided in meta
        conf_val = None
        if isinstance(meta, dict) and 'confidence' in meta:
            conf_val = int(meta.get('confidence'))
        # leave blank if not provided
        if conf_val is None:
            self.df.at[idx, 'confidence'] = ''
        else:
            self.df.at[idx, 'confidence'] = int(conf_val)

        # persist a dedicated value_functions.csv containing only the requested columns
        # Decide target folder: if the current attribute is country-specific ("Name - CC")
        # attempt to save to the original per-folder CSV (e.g. `ghg/value_functions.csv`).
        # Otherwise use the session results path or a new file in the loaded folder.
        default_out_dir = os.path.dirname(self.file_path) or os.path.dirname(__file__) or '.'

        # Read guideline to know which criteria are declared per-country and which country codes to accept
        guideline_path = os.path.join(os.path.dirname(__file__), 'quantitative', 'guideline.txt')
        per_country_criteria = set()
        country_codes = set()
        if os.path.exists(guideline_path):
            with open(guideline_path, 'r', encoding='utf-8') as gf:
                for ln in gf:
                    ln = ln.strip()
                    low = ln.lower()
                    if low.startswith('percountrycriteria:'):
                        _, rhs = ln.split(':', 1)
                        per_country_criteria = set([p.strip() for p in rhs.split(',') if p.strip()])
                    elif low.startswith('specifics:') or low.startswith('specifics>'):
                        # accept formats like: Specifics: [IT,CH,FR,PO]
                        try:
                            _, rhs = ln.split(':', 1) if ':' in ln else ln.split('>', 1)
                        except Exception:
                            continue
                        rhs = rhs.strip()
                        # strip brackets if present
                        if rhs.startswith('[') and rhs.endswith(']'):
                            rhs = rhs[1:-1]
                        country_codes = set([p.strip() for p in rhs.split(',') if p.strip()])

        # determine if current attribute looks like a per-country specific entry
        cur_row = self.df.iloc[self.current_attribute_index]
        attr_name = str(cur_row.get('name', '')).strip()

        qfolder = os.path.join(os.path.dirname(__file__), 'quantitative')

        out_path = getattr(self, 'results_vf_path', None)

        # If attribute is of form 'Base - CC' and Base is declared per-country, save to quantitative/value_functions_{CC}.csv
        if ' - ' in attr_name:
            base_name, suffix = [p.strip() for p in attr_name.rsplit(' - ', 1)]
            if base_name in per_country_criteria and suffix in country_codes:
                out_path = os.path.join(qfolder, f'value_functions_{suffix}.csv')

        # otherwise, if no explicit out_path, create a session file in the default output dir
        if not out_path:
            i = 1
            while True:
                candidate = os.path.join(default_out_dir, f'value_functions_{i}.csv')
                if not os.path.exists(candidate):
                    out_path = candidate
                    break
                i += 1

        # ensure the required columns exist in the dataframe (include group and confidence)
        cols = ['name', 'group', 'elicited_points', 'confidence', 'elicitation_meta']

        # Build a single output containing both global and country-specific rows.
        # Country-specific rows are named as 'Base/CC'. Global rows are written once
        # with their base name.
        rows_out = []
        for _, row in self.df.iterrows():
            nm = str(row.get('name', '')).strip()
            grp = row.get('group', '')
            ep = row.get('elicited_points', '') if 'elicited_points' in row.index else ''
            conf = row.get('confidence', '') if 'confidence' in row.index else ''
            meta = row.get('elicitation_meta', '') if 'elicitation_meta' in row.index else ''

            if ' - ' in nm:
                base, suff = [p.strip() for p in nm.rsplit(' - ', 1)]
                if suff in country_codes:
                    out_name = f"{base}/{suff}"
                else:
                    # unknown suffix: skip
                    continue
            else:
                out_name = nm

            rows_out.append({'name': out_name, 'group': grp, 'elicited_points': ep, 'confidence': conf, 'elicitation_meta': meta})

        out_file = os.path.join(qfolder, 'value_functions.csv')
        df_out = pd.DataFrame(rows_out, columns=cols)
        df_out = df_out.fillna('')
        df_out.to_csv(out_file, index=False)
        self.results_vf_path = out_file
        return out_file

    def get_value_function_string(self, degree=2):
        """Return a string representation of the fitted polynomial inside thresholds, or empty string.

        The returned string is a Python lambda expression like 'lambda x: 1.23*x**2 + 4.56*x + 7.89'
        or an empty string if fitting is not possible.
        """
        # reuse fitting selection logic
        if self.lower_threshold is not None and self.upper_threshold is not None:
            lo = min(self.lower_threshold, self.upper_threshold)
            hi = max(self.lower_threshold, self.upper_threshold)
            sel = [(px, py) for (px, py) in self.points if lo <= px <= hi]
        else:
            sel = list(self.points)

        if len(sel) < 2:
            return ''

        x = np.array([p[0] for p in sel])
        y = np.array([p[1] for p in sel])
        deg = max(1, min(degree, len(x) - 1))
        try:
            coeffs = np.polyfit(x, y, deg=deg)
        except Exception:
            return ''

        # build expression string
        terms = []
        n = len(coeffs)
        for i, c in enumerate(coeffs):
            power = n - i - 1
            # format coefficient
            terms.append(f'({c:.12g})*x**{power}' if power != 0 else f'({c:.12g})')
        expr = ' + '.join(terms)
        # simplify **1 and **0
        expr = expr.replace('**1)', ' )').replace('*x**0', '')
        return f'lambda x: {expr}'

    def get_current_attribute(self):
        """Return the current attribute being processed."""
        return self.df.iloc[self.current_attribute_index]

    def add_point(self, x, y):
        """Add a point to the elicitation process."""
        self.points.append((x, y))

    def fit_polynomial(self, degree=2):
        """Fit a polynomial curve to the collected points."""
        # If thresholds are defined, only use points within [lower, upper] to fit
        if self.lower_threshold is not None and self.upper_threshold is not None:
            lo = min(self.lower_threshold, self.upper_threshold)
            hi = max(self.lower_threshold, self.upper_threshold)
            sel = [(px, py) for (px, py) in self.points if lo <= px <= hi]
        else:
            sel = list(self.points)

        if len(sel) < 2:
            return None, None

        x = np.array([p[0] for p in sel])
        y = np.array([p[1] for p in sel])

        # ensure degree is less than number of points
        deg = max(1, min(degree, len(x) - 1))
        try:
            coeffs = np.polyfit(x, y, deg=deg)
        except Exception:
            return None, None

        x_fit = np.linspace(min(x), max(x), 200)
        y_fit = np.polyval(coeffs, x_fit)
        return x_fit, y_fit

    def fit_curve(self, fit_type='Monotone Spline (PCHIP)', degree=2, params=None):
        """Generalized fitter supporting multiple curve types.

        fit_type: one of 'Piecewise Linear', 'Polynomial', 'Monotone Spline (PCHIP)',
                  'Gaussian', 'Sigmoid', 'Exponential', 'Logarithmic'
        degree: used for polynomial fits
        params: dict of parameter values (optional). If provided and contains
                the necessary parameters, those are used directly. Otherwise
                a best-fit is attempted with scipy.optimize.curve_fit.

        Returns (x_fit, y_fit) or (None, None) on failure.
        """
        if params is None:
            params = {}

        # determine fit interval
        # If the user has added 0 or 1 points, fit between their x positions (anchors).
        # Otherwise, fit across the attribute range (min/max).
        anchor_xs = [px for (px, py) in self.points if float(py) == 0.0 or float(py) == 1.0]
        try:
            if anchor_xs:
                lo = float(min(anchor_xs))
                hi = float(max(anchor_xs))
            else:
                attr = self.get_current_attribute()
                lo = float(attr['min'])
                hi = float(attr['max'])
        except Exception:
            # fallback to point range
            if len(self.points) > 0:
                xs_only = [p[0] for p in self.points]
                lo = float(min(xs_only))
                hi = float(max(xs_only))
            else:
                return None, None, None

        # select points inside [lo, hi]
        sel = [(px, py) for (px, py) in self.points if lo <= px <= hi]

        if not sel:
            return None, None

        x = np.array([p[0] for p in sel], dtype=float)
        y = np.array([p[1] for p in sel], dtype=float)

        # determine plotting range: prefer attribute min/max if available
        try:
            attr = self.get_current_attribute()
            xmin = float(attr['min'])
            xmax = float(attr['max'])
        except Exception:
            xmin = float(np.min(x))
            xmax = float(np.max(x))

        if xmin == xmax:
            xmin -= 0.5
            xmax += 0.5

        x_fit = np.linspace(xmin, xmax, 200)

        ft = fit_type or 'Monotone Spline (PCHIP)'
        ft = ft.strip()

        try:
            if ft == 'Piecewise Linear':
                    # simple linear interpolation between sorted points
                    pts = sorted(sel, key=lambda t: t[0])
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    y_model = np.interp(x_fit, xs, ys)
                    # piecewise: constant tails outside [lo, hi]
                    left_tail = getattr(self, 'left_tail_value', None)
                    right_tail = getattr(self, 'right_tail_value', None)
                    if left_tail is None:
                        left_tail = float(ys[0])
                    if right_tail is None:
                        right_tail = float(ys[-1])
                    y_fit = np.empty_like(y_model)
                    y_fit[x_fit < lo] = left_tail
                    y_fit[x_fit > hi] = right_tail
                    mask = (x_fit >= lo) & (x_fit <= hi)
                    y_fit[mask] = y_model[mask]
                    # enforce [0,1]
                    y_fit = np.clip(y_fit, 0.0, 1.0)
                    return x_fit, y_fit, {}

            if ft == 'Polynomial':
                deg = max(1, min(int(degree), len(x) - 1))

                # If monotonicity requested, perform a constrained polynomial fit
                monotonic = bool(params.get('monotonic')) if params is not None else False
                increasing = True if params is None else bool(params.get('increasing', True))

                # standard unconstrained fit as initial guess
                try:
                    coeffs0 = np.polyfit(x, y, deg=deg)
                except Exception:
                    coeffs0 = None

                if monotonic and coeffs0 is not None:
                    # perform constrained least-squares: minimize sum squares with derivative sign constraints
                    # derivative coefficients for polynomial c: dcoeffs = c[:-1] * np.arange(n,0,-1)
                    def obj(c):
                        try:
                            y_model = np.polyval(c, x)
                            return float(np.sum((y_model - y) ** 2))
                        except Exception:
                            return 1e12

                    def deriv_at(c, xi):
                        c = np.asarray(c)
                        n = len(c) - 1
                        if n <= 0:
                            return 0.0
                        dcoeffs = c[:-1] * np.arange(n, 0, -1)
                        return float(np.polyval(dcoeffs, xi))

                    # choose grid points where derivative constraint applies
                    grid_x = np.linspace(lo, hi, min(30, max(5, int(len(x) * 3))))
                    sign = 1.0 if increasing else -1.0
                    cons = []
                    for xi in grid_x:
                        cons.append({'type': 'ineq', 'fun': (lambda xi: (lambda c: sign * deriv_at(c, xi)))(xi)})

                    # run minimize
                    try:
                        res = minimize(obj, coeffs0, method='SLSQP', constraints=cons, options={'maxiter': 1000, 'ftol': 1e-9})
                        if res.success:
                            coeffs = res.x
                        else:
                            coeffs = coeffs0
                    except Exception:
                        coeffs = coeffs0
                else:
                    # no monotonic constraint requested
                    if coeffs0 is None:
                        try:
                            coeffs = np.polyfit(x, y, deg=deg)
                        except Exception:
                            return None, None, None
                    else:
                        coeffs = coeffs0

                # compute polynomial values and piecewise tails
                try:
                    y_model = np.polyval(coeffs, x_fit)
                except Exception:
                    return None, None, None
                left_tail = getattr(self, 'left_tail_value', None)
                right_tail = getattr(self, 'right_tail_value', None)
                if left_tail is None:
                    # default to polynomial value at leftmost fit point
                    left_tail = float(np.polyval(coeffs, lo))
                if right_tail is None:
                    right_tail = float(np.polyval(coeffs, hi))
                y_fit = np.empty_like(y_model)
                y_fit[x_fit < lo] = left_tail
                y_fit[x_fit > hi] = right_tail
                mask = (x_fit >= lo) & (x_fit <= hi)
                y_fit[mask] = y_model[mask]
                y_fit = np.clip(y_fit, 0.0, 1.0)
                return x_fit, y_fit, {}

            # Monotone spline option using PCHIP + monotone projection (PAV)
            if 'PCHIP' in ft or 'Monotone Spline' in ft:
                # ensure points sorted by x
                pts = sorted(sel, key=lambda t: t[0])
                xs = np.array([p[0] for p in pts], dtype=float)
                ys = np.array([p[1] for p in pts], dtype=float)

                if len(xs) < 2:
                    return None, None, None

                # Pool Adjacent Violators Algorithm to enforce monotonic y (increasing)
                def pav(y):
                    # returns isotonic regression (non-decreasing)
                    y = y.astype(float)
                    n = len(y)
                    levels = y.copy()
                    weights = np.ones(n, dtype=float)
                    i = 0
                    while i < n - 1:
                        if levels[i] <= levels[i+1]:
                            i += 1
                            continue
                        # merge blocks i and i+1
                        total_weight = weights[i] + weights[i+1]
                        avg = (levels[i] * weights[i] + levels[i+1] * weights[i+1]) / total_weight
                        levels[i] = avg
                        weights[i] = total_weight
                        # remove i+1 by shifting
                        levels = np.delete(levels, i+1)
                        weights = np.delete(weights, i+1)
                        n -= 1
                        # move back if needed
                        if i > 0:
                            i -= 1
                    # expand levels back to original length according to weights
                    # Here weights represent block sizes; reconstruct by repeating
                    out = []
                    for lv, w in zip(levels, weights):
                        out.extend([lv] * int(round(w)))
                    # if lengths mismatch, pad/truncate
                    out = np.array(out, dtype=float)
                    if out.shape[0] < len(y):
                        # pad with last value
                        pad = np.full(len(y) - out.shape[0], out[-1])
                        out = np.concatenate([out, pad])
                    if out.shape[0] > len(y):
                        out = out[:len(y)]
                    return out

                increasing = True
                try:
                    increasing = bool(params.get('increasing', True))
                except Exception:
                    increasing = True

                if not increasing:
                    ys_proc = -ys
                else:
                    ys_proc = ys

                try:
                    ys_iso = pav(ys_proc)
                except Exception:
                    # fallback: simple cumulative min/max smoothing
                    if increasing:
                        ys_iso = np.maximum.accumulate(ys_proc)
                    else:
                        ys_iso = np.minimum.accumulate(ys_proc)

                if not increasing:
                    ys_iso = -ys_iso

                # build PCHIP interpolator on the isotonic y
                try:
                    interpolator = PchipInterpolator(xs, ys_iso, extrapolate=False)
                    y_model = interpolator(x_fit)
                except Exception:
                    # fallback to linear interpolation
                    y_model = np.interp(x_fit, xs, ys_iso)

                left_tail = getattr(self, 'left_tail_value', None)
                right_tail = getattr(self, 'right_tail_value', None)
                if left_tail is None:
                    left_tail = float(ys_iso[0])
                if right_tail is None:
                    right_tail = float(ys_iso[-1])
                y_fit = np.empty_like(y_model)
                y_fit[x_fit < lo] = left_tail
                y_fit[x_fit > hi] = right_tail
                mask = (x_fit >= lo) & (x_fit <= hi)
                # replace NaNs from PCHIP extrapolate=False using interpolation inside mask
                y_model = np.where(np.isnan(y_model), np.interp(x_fit, xs, ys_iso), y_model)
                y_fit[mask] = y_model[mask]
                y_fit = np.clip(y_fit, 0.0, 1.0)
                return x_fit, y_fit, {}

            # Non-linear fits: define model functions
            def gaussian_fn(xv, a, mu, sigma, c):
                return a * np.exp(-((xv - mu) ** 2) / (2 * sigma ** 2)) + c

            def sigmoid_fn(xv, L, k, x0, c):
                return L / (1.0 + np.exp(-k * (xv - x0))) + c

            def exponential_fn(xv, a, b, c):
                return a * np.exp(b * xv) + c

            def logarithmic_fn(xv, a, b, c, d):
                # ensure argument positive where possible
                return a * np.log(b * xv + c) + d

            # helper to try curve_fit when params not fully provided
            if ft == 'Gaussian':
                names = ['amplitude', 'mu', 'sigma', 'offset']
                if all(n in params for n in names):
                    a = params['amplitude']; mu = params['mu']; sigma = params['sigma']; c = params['offset']
                    return x_fit, gaussian_fn(x_fit, a, mu, sigma, c), params
                # initial guesses
                a0 = float(np.max(y) - np.min(y))
                mu0 = float(np.sum(x * y) / np.sum(y)) if np.sum(y) != 0 else float(np.mean(x))
                sigma0 = float(np.std(x) if np.std(x) > 0 else (xmax - xmin) / 6.0)
                c0 = float(np.min(y))
                p0 = [a0, mu0, sigma0, c0]
                try:
                    popt, _ = curve_fit(gaussian_fn, x, y, p0=p0, maxfev=20000)
                    # assemble piecewise with constant tails
                    y_model = gaussian_fn(x_fit, *popt)
                    left_tail = getattr(self, 'left_tail_value', None)
                    right_tail = getattr(self, 'right_tail_value', None)
                    if left_tail is None:
                        left_tail = float(gaussian_fn(lo, *popt))
                    if right_tail is None:
                        right_tail = float(gaussian_fn(hi, *popt))
                    y_fit = np.empty_like(y_model)
                    y_fit[x_fit < lo] = left_tail
                    y_fit[x_fit > hi] = right_tail
                    mask = (x_fit >= lo) & (x_fit <= hi)
                    y_fit[mask] = y_model[mask]
                    y_fit = np.clip(y_fit, 0.0, 1.0)
                    params_out = dict(zip(names, [float(v) for v in popt]))
                    return x_fit, y_fit, params_out
                except Exception:
                    return None, None, None

            if ft == 'Sigmoid':
                names = ['L', 'k', 'x0', 'offset']
                if all(n in params for n in names):
                    L = params['L']; k = params['k']; x0 = params['x0']; c = params['offset']
                    return x_fit, sigmoid_fn(x_fit, L, k, x0, c), params
                L0 = float(np.max(y) - np.min(y))
                k0 = 1.0 / (xmax - xmin) if xmax != xmin else 1.0
                x0_ = float((xmin + xmax) / 2.0)
                c0 = float(np.min(y))
                p0 = [L0, k0, x0_, c0]
                try:
                    popt, _ = curve_fit(sigmoid_fn, x, y, p0=p0, maxfev=20000)
                    y_model = sigmoid_fn(x_fit, *popt)
                    left_tail = getattr(self, 'left_tail_value', None)
                    right_tail = getattr(self, 'right_tail_value', None)
                    if left_tail is None:
                        left_tail = float(sigmoid_fn(lo, *popt))
                    if right_tail is None:
                        right_tail = float(sigmoid_fn(hi, *popt))
                    y_fit = np.empty_like(y_model)
                    y_fit[x_fit < lo] = left_tail
                    y_fit[x_fit > hi] = right_tail
                    mask = (x_fit >= lo) & (x_fit <= hi)
                    y_fit[mask] = y_model[mask]
                    y_fit = np.clip(y_fit, 0.0, 1.0)
                    params_out = dict(zip(names, [float(v) for v in popt]))
                    return x_fit, y_fit, params_out
                except Exception:
                    return None, None, None

            if ft == 'Exponential':
                names = ['a', 'b', 'c']
                if all(n in params for n in names):
                    a = params['a']; b = params['b']; c = params['c']
                    return x_fit, exponential_fn(x_fit, a, b, c), params
                a0 = 1.0
                b0 = 0.0
                c0 = float(np.min(y))
                p0 = [a0, b0, c0]
                try:
                    popt, _ = curve_fit(exponential_fn, x, y, p0=p0, maxfev=20000)
                    y_model = exponential_fn(x_fit, *popt)
                    left_tail = getattr(self, 'left_tail_value', None)
                    right_tail = getattr(self, 'right_tail_value', None)
                    if left_tail is None:
                        left_tail = float(exponential_fn(lo, *popt))
                    if right_tail is None:
                        right_tail = float(exponential_fn(hi, *popt))
                    y_fit = np.empty_like(y_model)
                    y_fit[x_fit < lo] = left_tail
                    y_fit[x_fit > hi] = right_tail
                    mask = (x_fit >= lo) & (x_fit <= hi)
                    y_fit[mask] = y_model[mask]
                    y_fit = np.clip(y_fit, 0.0, 1.0)
                    params_out = dict(zip(names, [float(v) for v in popt]))
                    return x_fit, y_fit, params_out
                except Exception:
                    return None, None, None

            if ft == 'Logarithmic':
                names = ['a', 'b', 'c', 'd']
                if all(n in params for n in names):
                    a = params['a']; b = params['b']; c = params['c']; d = params['d']
                    try:
                        return x_fit, logarithmic_fn(x_fit, a, b, c, d), params
                    except Exception:
                        return None, None, None
                # initial guess: map x into positive domain
                b0 = 1.0 / (xmax - xmin) if xmax != xmin else 1.0
                c0 = 1.0
                a0 = 1.0
                d0 = float(np.min(y))
                p0 = [a0, b0, c0, d0]
                try:
                    popt, _ = curve_fit(logarithmic_fn, x, y, p0=p0, maxfev=20000)
                    y_model = logarithmic_fn(x_fit, *popt)
                    left_tail = getattr(self, 'left_tail_value', None)
                    right_tail = getattr(self, 'right_tail_value', None)
                    if left_tail is None:
                        left_tail = float(logarithmic_fn(lo, *popt))
                    if right_tail is None:
                        right_tail = float(logarithmic_fn(hi, *popt))
                    y_fit = np.empty_like(y_model)
                    y_fit[x_fit < lo] = left_tail
                    y_fit[x_fit > hi] = right_tail
                    mask = (x_fit >= lo) & (x_fit <= hi)
                    y_fit[mask] = y_model[mask]
                    y_fit = np.clip(y_fit, 0.0, 1.0)
                    params_out = dict(zip(names, [float(v) for v in popt]))
                    return x_fit, y_fit, params_out
                except Exception:
                    return None, None, None

        except Exception:
            return None, None, None

        return None, None, None

    def next_attribute(self):
        """Move to the next attribute."""
        # guard when no dataframe loaded
        if self.df is None:
            return False
        if self.current_attribute_index < len(self.df) - 1:
            self.current_attribute_index += 1
            self.points = []  # Reset points for the new attribute
            return True
        return False

    def prev_attribute(self):
        """Move to the previous attribute."""
        # guard when no dataframe loaded
        if self.df is None:
            return False
        if self.current_attribute_index > 0:
            self.current_attribute_index -= 1
            self.points = []  # Reset points for the previous attribute
            return True
        return False
