# elicitation_logic.py
import pandas as pd
import numpy as np
import json
from scipy.optimize import curve_fit

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
        required_columns = ["name", "min", "max"]
        if not all(col in self.df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {', '.join(required_columns)}")
        return self.df

    def get_saved_state_for_current_attribute(self):
        """Return saved elicited points, function string and meta for the current attribute if present."""
        if self.df is None:
            return None
        row = self.df.iloc[self.current_attribute_index]
        points = None
        func = None
        meta = None
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
        try:
            self.df.at[idx, 'elicited_points'] = json.dumps(self.points)
        except Exception:
            self.df.at[idx, 'elicited_points'] = ''

        # compute value function string
        vf = self.get_value_function_string(degree)
        self.df.at[idx, 'value_function'] = vf if vf is not None else ''

        # meta
        try:
            self.df.at[idx, 'elicitation_meta'] = json.dumps(meta or {})
        except Exception:
            self.df.at[idx, 'elicitation_meta'] = ''

        # persist back to the same CSV
        self.df.to_csv(self.file_path, index=False)

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

    def fit_curve(self, fit_type='Piecewise Linear', degree=2, params=None):
        """Generalized fitter supporting multiple curve types.

        fit_type: one of 'Piecewise Linear', 'Polynomial', 'Gaussian', 'Sigmoid',
                  'Exponential', 'Logarithmic'
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

        ft = fit_type or 'Piecewise Linear'
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
                coeffs = np.polyfit(x, y, deg=deg)
                y_model = np.polyval(coeffs, x_fit)
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
        if self.current_attribute_index < len(self.df) - 1:
            self.current_attribute_index += 1
            self.points = []  # Reset points for the new attribute
            return True
        return False

    def prev_attribute(self):
        """Move to the previous attribute."""
        if self.current_attribute_index > 0:
            self.current_attribute_index -= 1
            self.points = []  # Reset points for the previous attribute
            return True
        return False
