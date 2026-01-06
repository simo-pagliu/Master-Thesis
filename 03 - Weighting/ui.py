import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider
from tkinter import ttk, messagebox
import os
import csv

class WBT_ui:
    def __init__(self, root, criteria):
        self.root = root
        self.root.title("Criteria Plotter")
        # Normalize `criteria` to a list of criterion dicts. Accept either:
        # - a dict keyed by criterion name -> dict (preferred),
        # - a list of dicts each containing at least a 'name' key,
        # - or a filepath/string (attempt to import via auxiliary.import_criteria).
        self.criteria = criteria
        if isinstance(criteria, dict):
            lst = []
            for k, v in criteria.items():
                if isinstance(v, dict):
                    if 'name' not in v:
                        v['name'] = k
                    lst.append(v)
                else:
                    lst.append({'name': k, 'value': v})
            self.criteria = lst
        elif isinstance(criteria, str):
            # try to import criteria from a file path using auxiliary.import_criteria
            try:
                from auxiliary import import_criteria

                imported = import_criteria(criteria)
                if isinstance(imported, dict):
                    lst = []
                    for k, v in imported.items():
                        if isinstance(v, dict):
                            if 'name' not in v:
                                v['name'] = k
                            lst.append(v)
                        else:
                            lst.append({'name': k, 'value': v})
                    self.criteria = lst
                elif isinstance(imported, list):
                    self.criteria = imported
            except Exception:
                # fallback: wrap the string as a single criterion name
                self.criteria = [{'name': str(criteria)}]
        else:
            # ensure list entries have 'name'
            try:
                self.criteria = [dict(c) if not isinstance(c, dict) else c for c in criteria]
            except Exception:
                self.criteria = [{'name': str(criteria)}]
        self.best_criterion = tk.StringVar()
        self.worst_criterion = tk.StringVar()
        self.current_comparison = 0
        self.comparisons = []
        self.best_comparison_results = None
        self.worst_comparison_results = None
        self.reordered_criteria_names = None
        # Track per-context Best-vs-Worst VF thresholds (y in [0,1])
        self._bw_vf_thresholds = {}
        # Track per-context cross-comparison rankings from best phase (criterion_name -> rank_value)
        self._best_rankings = {}
        # Set style for larger fonts
        plt.rcParams.update({'font.size': 12})
        # Build groups (preserve order of appearance)
        self.groups = []
        self.group_map = {}
        for c in self.criteria:
            g = c.get('group', 'Ungrouped')
            if g not in self.group_map:
                self.group_map[g] = []
                self.groups.append(g)
            self.group_map[g].append(c)

        # Track selected best/worst per group for later between-groups elicitation
        self.group_selected = {}

        # Map criteria by name for quick lookup
        self.criteria_by_name = {c.get('name'): c for c in self.criteria}

        # Start with the first group
        self.current_group_idx = 0
        # Initial selection UI for the first group
        self.setup_selection_ui()

    def setup_selection_ui(self):
        # Clear previous widgets
        for widget in self.root.winfo_children():
            widget.destroy()
        # Selection frame for current group
        selection_frame = ttk.Frame(self.root, padding="10")
        selection_frame.pack(fill=tk.X)
        group_name = self.groups[self.current_group_idx] if self.groups else 'Ungrouped'
        ttk.Label(selection_frame, text=f"Group: {group_name}", font=(None, 11, 'bold')).grid(row=0, column=0, columnspan=2)
        group_vals = [c['name'] for c in self.group_map.get(group_name, [])]

        ttk.Label(selection_frame, text="Best Criterion:").grid(row=1, column=0)
        self.best_dropdown = ttk.Combobox(selection_frame, textvariable=self.best_criterion, values=group_vals)
        self.best_dropdown.grid(row=1, column=1)
        ttk.Label(selection_frame, text="Worst Criterion:").grid(row=2, column=0)
        self.worst_dropdown = ttk.Combobox(selection_frame, textvariable=self.worst_criterion, values=group_vals)
        self.worst_dropdown.grid(row=2, column=1)
        # Expert confidence selector for this group (0..4)
        conf_options = [
            "0: Not confident at all (pure guess)",
            "1: Low confidence",
            "2: Moderately confident",
            "3: High confidence",
            "4: Extremely confident (certain)",
        ]
        self.group_confidence_var = tk.StringVar(value=conf_options[2])
        ttk.Label(selection_frame, text="Group confidence:").grid(row=3, column=0)
        self.group_confidence_cb = ttk.Combobox(selection_frame, textvariable=self.group_confidence_var, values=conf_options, width=40)
        self.group_confidence_cb.grid(row=3, column=1)
        ttk.Button(selection_frame, text="Continue", command=self.start_comparisons).grid(row=4, columnspan=2)

        # Show a single overview plot with all criteria at their max and their ranges
        try:
            # Overview plot for the current group
            group_name = self.groups[self.current_group_idx] if self.groups else 'Ungrouped'
            group_criteria = self.group_map.get(group_name, [])
            names = [c.get('name') for c in group_criteria]
            display_names = [f"{c.get('name')} \n ({c.get('unit')})" if c.get('unit') else c.get('name') for c in group_criteria]
            n = len(names)
            mins = []
            maxs = []
            for c in group_criteria:
                try:
                    lo = float(c.get('min', 0.0))
                except Exception:
                    lo = 0.0
                try:
                    hi = float(c.get('max', lo + 1.0))
                except Exception:
                    hi = lo + 1.0
                if hi == lo:
                    hi = lo + 1.0
                mins.append(lo)
                maxs.append(hi)

            # For the selection screen the bars are purely representational and
            # should appear full regardless of underlying value functions.
            vals = [1.0] * n

            fig = plt.Figure(figsize=(10, 5))
            # leave extra room at the bottom so rotated x-labels aren't clipped
            try:
                fig.subplots_adjust(bottom=0.4)
            except Exception:
                pass
            ax = fig.add_subplot(1, 1, 1)

            # space bars farther apart and make them thinner so x-labels don't overlap
            bar_width = 0.4
            spacing = 1.6
            x = np.arange(n) * spacing
            bars = ax.bar(x, vals, width=bar_width, color='#9ecae1')
            ax.set_xticks(x)
            ax.set_xticklabels(display_names, rotation=25, ha='right', fontsize=10)
            ax.set_ylim(0, 1)
            ax.set_title(f'Overview — {group_name} (value functions at max)')
            for j in range(n):
                low = min(mins[j], maxs[j])
                high = max(mins[j], maxs[j])
                ax.text(x[j], 0.02, f"{low:.2f}", ha='center', va='bottom', fontsize=10, color='black')
                ax.text(x[j], 0.98, f"{high:.2f}", ha='center', va='top', fontsize=10, color='black')

            # Embed the overview plot inside the selection frame to the right
            selection_frame.grid_columnconfigure(2, weight=1)
            plot_frame = ttk.Frame(selection_frame)
            # rowspan increased to cover the added confidence row
            plot_frame.grid(row=0, column=2, rowspan=5, sticky='nsew', padx=(8, 0))
            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(0, 12))
        except Exception:
            # If plotting fails for any reason, silently continue (UI still usable)
            pass

    def plot_initial_comparisons(self):
        # Initial plotting removed: UI now focuses solely on elicitation and
        # immediate saving of tradeoff comparisons to CSV.
        return

    def start_comparisons(self):
        best = self.best_criterion.get()
        worst = self.worst_criterion.get()
        if not best or not worst:
            messagebox.showerror("Error", "Please select both best and worst criteria.")
            return
        if best == worst:
            messagebox.showerror("Error", "Best and Worst must be different criteria.")
            return
        self.best = best
        self.worst = worst
        # Use dicts keyed by criterion name for in-memory results (works across groups)
        self.best_comparison_results = {c['name']: np.nan for c in self.criteria}
        self.worst_comparison_results = {c['name']: np.nan for c in self.criteria}

        # Build comparisons only for the current group
        group_name = self.groups[self.current_group_idx] if self.groups else 'Ungrouped'
        group_criteria = self.group_map.get(group_name, [])
        group_names = [c['name'] for c in group_criteria]

        # remember the selected best/worst for this group
        # store group-level confidence chosen in selection screen (default to 2)
        try:
            conf_str = self.group_confidence_var.get()
            conf_val = int(conf_str.split(':', 1)[0])
        except Exception:
            conf_val = 2
        self.group_selected[group_name] = {'best': best, 'worst': worst, 'confidence': conf_val}

        # set current context for plotting/saving
        self._context_criteria = group_criteria
        self._context_group_name = group_name
        self._context_scope = 'intra-group'

        # Generate comparisons with ordering:
        # 1) explicit Best vs Worst
        # 2) remaining Best vs others (excluding Worst)
        # 3) Worst vs others (excluding Best)
        self.comparisons = []
        if worst in group_names and best in group_names:
            self.comparisons.append(("best", best, worst))
        for name in group_names:
            if name != best and name != worst:
                self.comparisons.append(("best", best, name))
        for name in group_names:
            if name != worst and name != best:
                self.comparisons.append(("worst", worst, name))
        self.current_comparison = 0
        # Use a single persistent results file for all groups: 'BWT_results.csv'
        ui_dir = os.path.dirname(os.path.abspath(__file__))
        self._results_fn = os.path.join(ui_dir, 'BWT_results.csv')
        # Create the file with header if it doesn't exist
        if not os.path.exists(self._results_fn):
            with open(self._results_fn, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['Type', 'Reference', 'Other', 'Value', 'Group', 'Confidence', 'a'])

        # Try to restore BW threshold for this group from CSV (if exists)
        try:
            self._restore_bw_threshold_from_csv(group_name, best, worst)
        except Exception:
            pass
        self.show_next_comparison()

    def prepare_intergroup_phase(self, phase):
        # phase: 'B' for best-from-groups, 'W' for worst-from-groups
        # gather candidates from stored group selections
        key = 'best' if phase == 'B' else 'worst'
        candidates = []
        for g in self.groups:
            sel = self.group_selected.get(g)
            if not sel or key not in sel or sel[key] is None:
                messagebox.showerror('Error', f"Group '{g}' has no selected {key}; cannot run between-groups elicitation")
                return
            candidates.append(sel[key])

        # quick selection dialog to choose best/worst among the candidates, with a plot
        dlg = tk.Toplevel(self.root)
        dlg.title(f"Between-groups {'Best' if phase == 'B' else 'Worst'} selection")
        # Left: selection controls; Right: small overview plot of candidate ranges
        sel_frame = ttk.Frame(dlg, padding=8)
        sel_frame.grid(row=0, column=0, sticky='n')
        plot_frame = ttk.Frame(dlg, padding=4)
        plot_frame.grid(row=0, column=1, sticky='nsew')
        dlg.grid_columnconfigure(1, weight=1)

        ttk.Label(sel_frame, text=f"Select Best/Worst among group winners:").grid(row=0, column=0, columnspan=2)
        best_var = tk.StringVar()
        worst_var = tk.StringVar()
        ttk.Label(sel_frame, text='Best:').grid(row=1, column=0)
        best_cb = ttk.Combobox(sel_frame, textvariable=best_var, values=candidates)
        best_cb.grid(row=1, column=1)
        ttk.Label(sel_frame, text='Worst:').grid(row=2, column=0)
        worst_cb = ttk.Combobox(sel_frame, textvariable=worst_var, values=candidates)
        worst_cb.grid(row=2, column=1)

        # Build a small overview plot for the candidate criteria (show ranges)
        try:
            crit_list = [self.criteria_by_name[n] for n in candidates]
            base_names = [c['name'] for c in crit_list]
            display_names = [f"{c.get('name')} \n ({c.get('unit')})" if c.get('unit') else c.get('name') for c in crit_list]
            mins = []
            maxs = []
            for c in crit_list:
                try:
                    lo = float(c.get('min', 0.0))
                except Exception:
                    lo = 0.0
                try:
                    hi = float(c.get('max', lo + 1.0))
                except Exception:
                    hi = lo + 1.0
                if hi == lo:
                    hi = lo + 1.0
                mins.append(lo)
                maxs.append(hi)

            # Represent value-function at max for each candidate (or fallback to linear)
            vals = []
            for i, c in enumerate(crit_list):
                lo = mins[i]
                hi = maxs[i]
                val_at_max = maxs[i]
                vf = c.get('value_function')
                if callable(vf):
                    try:
                        v = float(vf(val_at_max))
                    except Exception:
                        v = 0.001
                else:
                    if hi == lo:
                        v = 1.0
                    else:
                        v = (val_at_max - lo) / (hi - lo)
                vals.append(float(np.clip(v, 0.001, 1.0)))
            fig = plt.Figure(figsize=(4, 2.5))
            # leave extra bottom room for rotated x-labels in small overview
            try:
                fig.subplots_adjust(bottom=0.28)
            except Exception:
                pass
            ax = fig.add_subplot(1, 1, 1)
            # thinner bars and increased spacing for clarity of x-axis labels
            bar_width = 0.4
            spacing = 1.6
            x = np.arange(len(display_names)) * spacing
            bars = ax.bar(x, vals, width=bar_width, color='#9ecae1')
            ax.set_xticks(x)
            ax.set_xticklabels(display_names, rotation=25, ha='right', fontsize=10)
            ax.set_ylim(0, 1)
            ax.set_title('Candidate ranges (value functions at max)')
            for j in range(len(display_names)):
                low = min(mins[j], maxs[j])
                high = max(mins[j], maxs[j])
                ax.text(x[j], 0.02, f"{low:.2f}", ha='center', va='bottom', fontsize=9, color='black')
                ax.text(x[j], 0.98, f"{high:.2f}", ha='center', va='top', fontsize=9, color='black')

            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(0, 12))
        except Exception:
            pass

        def on_continue():
            b = best_var.get()
            w = worst_var.get()
            if not b or not w:
                messagebox.showerror('Error', 'Please select both best and worst for between-groups elicitation')
                return
            dlg.destroy()
            # prepare context criteria list
            context_criteria = [self.criteria_by_name[n] for n in candidates]
            group_label = f'Between-groups-{"B" if phase=="B" else "W"}'
            scope_label = f'between-groups-{"B" if phase=="B" else "W"}'
            # Start comparisons using the same flow as intra-group contexts so
            # the updated comparison UI and CSV behavior are applied.
            try:
                self.start_comparisons_for_context(b, w, context_criteria, group_label, scope_label)
            except Exception:
                # If something goes wrong, show an error so the user is not stuck.
                messagebox.showerror('Error', 'Failed to start between-groups comparisons')

        ttk.Button(sel_frame, text='Continue', command=on_continue).grid(row=3, column=0, columnspan=2, pady=6)

    def start_comparisons_for_context(self, best, worst, context_criteria, context_group_name, scope_label):
        # Set context and prepare comparisons over the provided context_criteria
        self._context_criteria = context_criteria
        self._context_group_name = context_group_name
        self._context_scope = scope_label
        names = [c['name'] for c in context_criteria]
        # store as a synthetic 'group' selection (use default confidence 2)
        self.group_selected[context_group_name] = {'best': best, 'worst': worst, 'confidence': 2}

        # Build comparisons (Best vs Worst first, then other bests, then other worsts)
        self.comparisons = []
        if worst in names and best in names:
            self.comparisons.append(('best', best, worst))
        for name in names:
            if name != best and name != worst:
                self.comparisons.append(('best', best, name))
        for name in names:
            if name != worst and name != best:
                self.comparisons.append(('worst', worst, name))

        self.current_comparison = 0
        # Try to restore BW threshold for this synthetic context as well
        try:
            self._restore_bw_threshold_from_csv(context_group_name, best, worst)
        except Exception:
            pass
        self.show_next_comparison()

    def _restore_bw_threshold_from_csv(self, group_name, best, worst):
        """If a Best-vs-Worst comparison exists in CSV for the given group,
        compute its VF value and store as threshold for consistency guidance."""
        fn = getattr(self, '_results_fn', None)
        if fn is None:
            ui_dir = os.path.dirname(os.path.abspath(__file__))
            fn = os.path.join(ui_dir, 'BWT_results.csv')
        if not os.path.exists(fn):
            return
        # find BW row: Type='best', Reference=best, Other=worst, Group=group_name
        try:
            with open(fn, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('Group') != group_name:
                        continue
                    if row.get('Type') == 'best' and row.get('Reference') == best and row.get('Other') == worst:
                        try:
                            x_val = float(row.get('Value', 'nan'))
                        except Exception:
                            continue
                        # compute VF on the Best criterion
                        crit = self.criteria_by_name.get(best)
                        if not crit:
                            continue
                        vf = crit.get('value_function')
                        y = None
                        try:
                            if callable(vf):
                                y = float(vf(x_val))
                            else:
                                # fallback linear between min..max of the Best criterion
                                lo = float(crit.get('min', 0.0))
                                hi = float(crit.get('max', lo + 1.0))
                                if hi == lo:
                                    y = 1.0
                                else:
                                    y = (x_val - lo) / (hi - lo)
                        except Exception:
                            pass
                        if y is not None:
                            self._bw_vf_thresholds[group_name] = float(np.clip(y, 0.0, 1.0))
                        break
        except Exception:
            pass

    def _compute_worst_ordinal_constraint(self, group_name, worst_other_name):
        """For a worst-comparison (worst vs worst_other_name), check if any previously
        completed worst comparison can provide an ordinal constraint.
        
        Returns (lower_bound, upper_bound) where:
        - lower_bound: max VF from more-important criteria (require current >= this)
        - upper_bound: min VF from less-important criteria (require current <= this)
        """
        fn = getattr(self, '_results_fn', None)
        if fn is None:
            ui_dir = os.path.dirname(os.path.abspath(__file__))
            fn = os.path.join(ui_dir, 'BWT_results.csv')
        if not os.path.exists(fn):
            return None, None

        # First, build best-phase ranking: criterion -> vf_value (lower = more important)
        best_ranks = {}  # criterion_name -> vf_value from best comparisons
        try:
            with open(fn, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('Group') != group_name or row.get('Type') != 'best':
                        continue
                    ref = row.get('Reference')
                    other = row.get('Other')
                    try:
                        x_val = float(row.get('Value', 'nan'))
                    except Exception:
                        continue
                    # Compute VF on the reference (best) criterion
                    crit = self.criteria_by_name.get(ref)
                    if not crit:
                        continue
                    vf = crit.get('value_function')
                    try:
                        if callable(vf):
                            y = float(vf(x_val))
                        else:
                            lo = float(crit.get('min', 0.0))
                            hi = float(crit.get('max', lo + 1.0))
                            y = 1.0 if hi == lo else (x_val - lo) / (hi - lo)
                    except Exception:
                        continue
                    y = float(np.clip(y, 0.0, 1.0))
                    # Store: other's rank is y (lower y = higher importance)
                    best_ranks[other] = y
        except Exception:
            return None, None

        # Get the rank of the current worst_other from best phase
        if worst_other_name not in best_ranks:
            return None, None
        other_rank = best_ranks[worst_other_name]

        # Now check already-completed worst comparisons
        lower_bound = None  # max VF from more-important criteria
        upper_bound = None  # min VF from less-important criteria
        try:
            with open(fn, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('Group') != group_name or row.get('Type') != 'worst':
                        continue
                    ref = row.get('Reference')  # This is the Worst criterion (always same)
                    other = row.get('Other')    # The "other" criterion in this comparison
                    if other == worst_other_name:
                        # Skip self-comparisons
                        continue
                    try:
                        x_val = float(row.get('Value', 'nan'))
                    except Exception:
                        continue
                    
                    # Check if 'other' exists in best rankings
                    if other not in best_ranks:
                        continue
                    other_rank_val = best_ranks[other]
                    
                    # Compute VF of the 'other' criterion at the value assigned to it
                    crit = self.criteria_by_name.get(other)
                    if not crit:
                        continue
                    vf = crit.get('value_function')
                    try:
                        if callable(vf):
                            y = float(vf(x_val))
                        else:
                            lo = float(crit.get('min', 0.0))
                            hi = float(crit.get('max', lo + 1.0))
                            y = 1.0 if hi == lo else (x_val - lo) / (hi - lo)
                    except Exception:
                        continue
                    y = float(np.clip(y, 0.0, 1.0))
                    
                    # Check ordering relationship
                    if other_rank_val < other_rank:
                        # 'other' IS more important, creates a lower bound
                        if lower_bound is None or y > lower_bound:
                            lower_bound = y
                    elif other_rank_val > other_rank:
                        # 'other' IS less important, creates an upper bound
                        if upper_bound is None or y < upper_bound:
                            upper_bound = y
        except Exception:
            pass

        return lower_bound, upper_bound

    def _compute_worst_consistency_bounds(self, group_name, worst_other_name):
        """DEPRECATED: Use _compute_worst_ordinal_constraint instead.
        Kept for backward compatibility."""
        return None, None

    def show_next_comparison(self):
        # Minimal elicitation UI: simple labels and a Tk slider. Each comparison
        # is immediately appended to `BWT_results.csv` when the user presses Next.
        for widget in self.root.winfo_children():
            widget.destroy()
        if self.current_comparison >= len(self.comparisons):
            self.finalize_results()
            return

        comparison_type, ref_criterion, other_criterion = self.comparisons[self.current_comparison]

        # Determine the active context: prefer an explicit context (between-groups or current group)
        if hasattr(self, '_context_criteria') and self._context_criteria is not None:
            group_criteria = self._context_criteria
            group_name = getattr(self, '_context_group_name', 'Ungrouped')
        else:
            group_name = self.groups[self.current_group_idx] if self.groups else 'Ungrouped'
            group_criteria = self.group_map.get(group_name, [])

        # Prepare base and display names (display includes units when available)
        base_names = [c.get('name') for c in group_criteria]
        display_names = [f"{c.get('name')} \n ({c.get('unit')})" if c.get('unit') else c.get('name') for c in group_criteria]
        n = len(base_names)
        ref_idx = next((i for i, c in enumerate(group_criteria) if c.get('name') == ref_criterion), 0)
        other_idx = next((i for i, c in enumerate(group_criteria) if c.get('name') == other_criterion), 0)

        # Base values: start all at min for this group
        mins = []
        maxs = []
        for c in group_criteria:
            try:
                lo = float(c.get('min', 0.0))
            except Exception:
                lo = 0.0
            try:
                hi = float(c.get('max', lo + 1.0))
            except Exception:
                hi = lo + 1.0
            if hi == lo:
                hi = lo + 1.0
            mins.append(lo)
            maxs.append(hi)

        # Build baseline values: the 'minimum' data point per criterion
        # interpreted according to polarity: if type is positive -> data min; if negative -> data max
        base_vals = []
        best_vals = []
        for i, c in enumerate(group_criteria):
            ctype = str(c.get('type', '')).strip().lower()
            if ctype in ('negative', 'neg', '-'):
                # For negative criteria, the 'minimum' (worst) in value terms is the CSV max
                base_vals.append(maxs[i])
                # The 'best' data point is the CSV min
                best_vals.append(mins[i])
            else:
                # For positive criteria, the 'minimum' (worst) is the CSV min
                base_vals.append(mins[i])
                # The 'best' data point is the CSV max
                best_vals.append(maxs[i])

        left_vals = base_vals.copy()
        right_vals = base_vals.copy()

        # Precompute x positions for VF levels 0.0,0.1,...,1.0 for each criterion
        # Store as list under key '_precomputed_x_for_y' in each criterion dict
        try:
            for i, c in enumerate(group_criteria):
                lo = mins[i]
                hi = maxs[i]
                low = float(min(lo, hi))
                high = float(max(lo, hi))
                vf = c.get('value_function')
                pre = []
                for k in range(11):
                    y = float(k) / 10.0
                    xval = low
                    try:
                        if callable(vf):
                            # sample densely and pick closest x (choose smallest x on ties)
                            xs = np.linspace(low, high, 401)
                            ys = np.empty_like(xs)
                            for j, x in enumerate(xs):
                                try:
                                    ys[j] = float(vf(x))
                                except Exception:
                                    ys[j] = np.nan
                            valid = ~np.isnan(ys)
                            if np.any(valid):
                                diffs = np.abs(ys[valid] - y)
                                vidx = np.nonzero(valid)[0]
                                minpos = int(np.argmin(diffs))
                                # pick smallest x among equal diffs
                                eq = np.isclose(diffs, diffs[minpos])
                                if np.any(eq):
                                    chosen = vidx[eq].min()
                                else:
                                    chosen = vidx[minpos]
                                xval = float(xs[chosen])
                            else:
                                # no valid evaluations: fallback linear
                                xval = float(low + (high - low) * np.clip(y, 0.0, 1.0))
                        else:
                            xval = float(low + (high - low) * np.clip(y, 0.0, 1.0))
                    except Exception:
                        xval = float(low)
                    # ensure within bounds
                    xval = float(np.clip(xval, low, high))
                    pre.append(xval)
                c['_precomputed_x_for_y'] = pre
        except Exception:
            # non-fatal: continue without precomputation
            pass

        # Configure left plot according to comparison type
        if comparison_type == 'best':
            # For best elicitation: left shows the OTHER criterion at its best data point,
            # and the user adjusts the REFERENCE (best) on the right.
            left_vals[other_idx] = best_vals[other_idx]
            slider_target = ref_idx
        else:
            # For worst comparisons: left shows the REFERENCE (worst) at its best data point,
            # and the user adjusts the OTHER on the right.
            left_vals[ref_idx] = best_vals[ref_idx]
            slider_target = other_idx

        # Right initially all mins (slider will modify slider_target)

        # Create UI frame and titles
        frm = ttk.Frame(self.root, padding=8)
        frm.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frm, text=f"Comparison {self.current_comparison + 1} of {len(self.comparisons)}", font=(None, 12, 'bold')).pack(pady=(0, 6))

        # per-comparison confidence combobox will be placed together with the
        # slider/radio controls at the bottom (created later) so it appears
        # to the right of the slider as requested.

        # Matplotlib figure with two side-by-side bar plots
        # Increase figure size so plots have more vertical space and are not cut off
        fig = plt.Figure(figsize=(10, 6))
        # leave extra room at the bottom so rotated x-labels aren't clipped
        try:
            fig.subplots_adjust(bottom=0.22)
        except Exception:
            pass
        # three subplots: left (other/reference), middle (adjustable), right (value function)
        ax_left = fig.add_subplot(1, 3, 1)
        ax_right = fig.add_subplot(1, 3, 2)
        ax_vf = fig.add_subplot(1, 3, 3)

        def vf_values(vals):
            """
            Convert raw data values `vals` (x in original units) to value-function outputs
            in [0.001, 1.0]. Uses attached `value_function` if available, otherwise falls back
            to linear mapping between min->0.001 and max->1.0.
            """
            out = []
            for i, xval in enumerate(vals):
                c = group_criteria[i]
                lo = mins[i]
                hi = maxs[i]
                vf = c.get('value_function')
                if callable(vf):
                    try:
                        v = float(vf(xval))
                    except Exception:
                        v = 0.001
                else:
                    # fallback linear mapping
                    try:
                        if hi == lo:
                            v = 1.0
                        else:
                            v = (float(xval) - lo) / (hi - lo)
                    except Exception:
                        v = 0.001
                out.append(float(np.clip(v, 0.001, 1.0)))
            return out

        # For elicitation display: left shows the "maxed" criterion at vf=1
        # (which is the OTHER for 'best' comparisons, and the REFERENCE for 'worst').
        # Right starts with all criteria at vf=0 and the slider will change the target bar.
        n = len(base_names)
        if comparison_type == 'best':
            left_vf = [1.0 if i == other_idx else 0.0 for i in range(n)]
        else:
            left_vf = [1.0 if i == ref_idx else 0.0 for i in range(n)]
        right_vf = [0.0 for _ in range(n)]

        # thinner bars and more spacing so x-axis labels don't overlap
        bar_width = 0.35
        spacing = 1.6
        x = np.arange(n) * spacing
        bars_left = ax_left.bar(x, left_vf, width=bar_width, color='#9ecae1')
        ax_left.set_xticks(x)
        ax_left.set_xticklabels(display_names, rotation=25, ha='right', fontsize=10)
        ax_left.set_ylim(0, 1)
        # Clarify which criterion is maxed on the left
        if comparison_type == 'best':
            ax_left.set_title(f"Other left: {display_names[other_idx]} maxed")
        else:
            ax_left.set_title(f"Reference left: {display_names[ref_idx]} maxed")

        bars_right = ax_right.bar(x, right_vf, width=bar_width, color='#9ecae1')
        ax_right.set_xticks(x)
        ax_right.set_xticklabels(display_names, rotation=25, ha='right', fontsize=10)
        ax_right.set_ylim(0, 1)
        # Clarify which criterion the user adjusts on the right
        ax_right.set_title(f"Adjustable right: {display_names[slider_target]}")

        # --- Plot the value function for the slider target on the right subplot ---
        try:
            idx_vf = slider_target
            c_vf = group_criteria[idx_vf]
            lo_vf = min(mins[idx_vf], maxs[idx_vf])
            hi_vf = max(mins[idx_vf], maxs[idx_vf])
            xs_vf = np.linspace(lo_vf, hi_vf, 400)
            vf_fn = c_vf.get('value_function')
            if callable(vf_fn):
                ys_vf = [float(vf_fn(x)) if (vf_fn is not None) else 0.001 for x in xs_vf]
            else:
                # linear fallback
                if hi_vf == lo_vf:
                    ys_vf = [1.0 for _ in xs_vf]
                else:
                    ys_vf = [(x - lo_vf) / (hi_vf - lo_vf) for x in xs_vf]
            ys_vf = [float(np.clip(y, 0.001, 1.0)) for y in ys_vf]
            ax_vf.plot(xs_vf, ys_vf, color='#1f77b4')
            ax_vf.set_xlim(lo_vf, hi_vf)
            ax_vf.set_ylim(0.0, 1.0)
            ax_vf.set_title(f"Value function: {display_names[slider_target]}")
            ax_vf.set_xlabel('Data')
            ax_vf.set_ylabel('Value')
            # initial marker for the current slider position; will be updated by slider
            try:
                start_x = lo_vf
                if callable(vf_fn):
                    init_y = float(vf_fn(float(start_x)))
                else:
                    init_y = 0.001 if hi_vf == lo_vf else ((float(start_x) - lo_vf) / (hi_vf - lo_vf))
            except Exception:
                init_y = 0.0
            init_y = float(np.clip(init_y, 0.0, 1.0))
            vf_marker, = ax_vf.plot([start_x], [init_y], marker='o', color='C3', markersize=6)
        except Exception:
            ax_vf = None
            vf_marker = None

        # (old min/max labels removed) — we now draw value-based labels centered on bars below

        canvas = FigureCanvasTkAgg(fig, master=frm)
        canvas.draw()
        # Allow the canvas to expand so the plots get full available space
        # add more bottom padding so the slider/radio controls don't overlap the labels
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(0, 12))

        # --- Draw value-function labels at increments 0.0,0.1,...,1.0 for each bar ---
        def invert_vf_for_criterion(c, y_target, lo, hi):
            """Return an x such that vf(x)=y_target (approx)."""
            vf = c.get('value_function')
            # If VF exposes node arrays, invert analytically on segments
            if callable(vf) and hasattr(vf, '_xs') and hasattr(vf, '_ys'):
                xs = np.array(vf._xs, dtype=float)
                ys = np.array(vf._ys, dtype=float)
                # search segments where y_target lies between ys[i] and ys[i+1]
                for i in range(len(xs) - 1):
                    y0, y1 = ys[i], ys[i+1]
                    x0, x1 = xs[i], xs[i+1]
                    if (y_target >= min(y0, y1) - 1e-12) and (y_target <= max(y0, y1) + 1e-12):
                        if abs(y1 - y0) < 1e-12:
                            return float(x0)
                        t = (y_target - y0) / (y1 - y0)
                        return float(x0 + t * (x1 - x0))
                # If not in any segment, clamp to the criterion domain extremes
                # rather than extrapolating numerically. This ensures labels
                # correspond to the actual data range (low/high) shown on plots.
                    # Use VF node endpoints to decide which data-end corresponds
                    # to the extreme VF values, taking the node monotonicity into account.
                    try:
                        x_start = float(xs[0])
                        x_end = float(xs[-1])
                        y_start = float(ys[0])
                        y_end = float(ys[-1])
                        # If VF increases with x (y_start < y_end), small y -> x_start
                        if y_start <= y_end:
                            if y_target <= float(ys.min()):
                                return float(x_start)
                            if y_target >= float(ys.max()):
                                return float(x_end)
                        else:
                            # VF decreases with x: small y -> x_end
                            if y_target <= float(ys.min()):
                                return float(x_end)
                            if y_target >= float(ys.max()):
                                return float(x_start)
                    except Exception:
                        # fallback to domain extremes if anything goes wrong
                        low_dom = float(min(lo, hi))
                        high_dom = float(max(lo, hi))
                        if y_target <= float(ys.min()):
                            return float(low_dom)
                        if y_target >= float(ys.max()):
                            return float(high_dom)
            # fallback: sample within [lo,hi] and find closest
            try:
                samples = np.linspace(lo, hi, 201)
                vals = [float(vf(x)) if callable(vf) else (0.001 if hi==lo else (x-lo)/(hi-lo)) for x in samples]
                idx = (np.abs(np.array(vals) - y_target)).argmin()
                return float(samples[idx])
            except Exception:
                return float(lo)

        def compute_x_for_vf(c, y_target, lo, hi, samples=1001):
            """Robustly compute an x in [lo,hi] whose VF(x) is closest to y_target.
            Returns the smallest x if multiple matches. Falls back to linear inverse if VF sampling fails."""
            vf = c.get('value_function')
            # simple linear fallback/analytic inverse when no callable VF
            if not callable(vf):
                try:
                    return float(lo + (hi - lo) * float(np.clip(y_target, 0.0, 1.0)))
                except Exception:
                    return float(lo)

            # sample densely and evaluate VF safely (sample in ascending order)
            try:
                sample_lo = min(lo, hi)
                sample_hi = max(lo, hi)
                xs = np.linspace(sample_lo, sample_hi, samples)
                ys = np.empty_like(xs)
                ys.fill(np.nan)
                for i, x in enumerate(xs):
                    try:
                        y = float(vf(x))
                        ys[i] = y
                    except Exception:
                        ys[i] = np.nan
                # mask invalid values
                valid = ~np.isnan(ys)
                if not np.any(valid):
                    # all evaluations failed; fallback to linear inverse
                    return float(lo + (hi - lo) * float(np.clip(y_target, 0.0, 1.0)))
                # compute absolute differences only on valid entries
                diffs = np.abs(ys[valid] - float(y_target))
                min_idx = np.argmin(diffs)
                # map back to original xs index; choose smallest x in case multiple
                valid_idxs = np.nonzero(valid)[0]
                chosen_idx = valid_idxs[min_idx]
                # if multiple entries have same diff, pick smallest index
                equal_idxs = valid_idxs[np.isclose(diffs, diffs[min_idx])]
                if equal_idxs.size:
                    chosen_idx = int(equal_idxs.min())
                return float(xs[chosen_idx])
            except Exception:
                # final fallback
                try:
                    return float(lo + (hi - lo) * float(np.clip(y_target, 0.0, 1.0)))
                except Exception:
                    return float(lo)

        # Add labels for each axis: left and right
        for ax in (ax_left, ax_right):
            for j in range(n):
                # For each bar, place small labels at y=0.0..1.0 (step 0.1)
                for ytick in np.linspace(0.0, 1.0, 11):
                    low = min(mins[j], maxs[j])
                    high = max(mins[j], maxs[j])
                    xval = invert_vf_for_criterion(group_criteria[j], ytick, low, high)
                    # format label with 2 decimals
                    lbl = f"{xval:.2f}"
                    # Make first/last (0.0 and 1.0) a bit larger and nudge inward
                    if abs(ytick - 0.0) < 1e-8:
                        fontsize = 9
                        y_pos = min(ytick + 0.035, 0.05)
                    elif abs(ytick - 1.0) < 1e-8:
                        fontsize = 9
                        y_pos = max(ytick - 0.035, 0.95)
                    else:
                        fontsize = 8
                        y_pos = ytick
                    # center label horizontally with respect to the bar; ensure it's inside plot
                    ax.text(x[j], y_pos, lbl, fontsize=fontsize, va='center', ha='center', color='black', clip_on=True)

        # Always check CSV for existing data for this comparison
        # This ensures that navigating back and forth preserves previously entered values
        restored_value = None
        restored_conf = None
        fn = getattr(self, '_results_fn', None)
        if fn is None:
            ui_dir = os.path.dirname(os.path.abspath(__file__))
            fn = os.path.join(ui_dir, 'BWT_results.csv')
        
        # Try to find existing data for this comparison in the CSV
        try:
            if os.path.exists(fn):
                with open(fn, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if (row.get('Type') == comparison_type and row.get('Reference') == ref_criterion and 
                            row.get('Other') == other_criterion and row.get('Group') == group_name):
                            try:
                                restored_value = float(row.get('Value', 0.0))
                            except Exception:
                                pass
                            try:
                                restored_conf = int(row.get('Confidence', 2))
                            except Exception:
                                pass
                            break
        except Exception:
            pass
        
        # Determine initial slider value: use CSV data if available, otherwise start at vf(x)=0
        if restored_value is not None:
            # Use the value from CSV
            initial_x_zero = float(restored_value)
        else:
            # Determine initial slider x such that vf(x)=0 (choose smallest if multiple)
            try:
                pre = group_criteria[slider_target].get('_precomputed_x_for_y')
                if pre and len(pre) > 0 and pre[0] is not None:
                    initial_x_zero = float(pre[0])
                else:
                    low = min(mins[slider_target], maxs[slider_target])
                    high = max(mins[slider_target], maxs[slider_target])
                    initial_x_zero = float(compute_x_for_vf(group_criteria[slider_target], 0.0, low, high))
            except Exception:
                initial_x_zero = float(mins[slider_target])
        
        # Store restored confidence for later use
        self._current_restored_conf = restored_conf

        # Slider to adjust the target criterion on the right plot
        resolution = max((maxs[slider_target] - mins[slider_target]) / 200.0, 1e-6)
        # Create a controls frame divided into three equal columns:
        # left (info), center (slider & radios), right (confidence)
        controls_frame = ttk.Frame(frm)
        controls_frame.pack(fill=tk.X, pady=(6, 4))
        # use grid to make three equal columns
        left_col = ttk.Frame(controls_frame)
        center_col = ttk.Frame(controls_frame)
        right_col = ttk.Frame(controls_frame)
        left_col.grid(row=0, column=0, sticky='nsew', padx=(8, 8))
        center_col.grid(row=0, column=1, sticky='n', padx=(8, 8))
        right_col.grid(row=0, column=2, sticky='nsew', padx=(8, 8))
        controls_frame.grid_columnconfigure(0, weight=1)
        controls_frame.grid_columnconfigure(1, weight=1)
        controls_frame.grid_columnconfigure(2, weight=1)

        # Move the info texts into the left column (center them vertically)
        try:
            info_frame = ttk.Frame(left_col)
            info_frame.pack(expand=True)
            ttk.Label(info_frame, text=f"Type: {comparison_type}").pack(anchor='w', pady=(6, 2))
            ttk.Label(info_frame, text=f"Reference: {ref_criterion}").pack(anchor='w', pady=2)
            ttk.Label(info_frame, text=f"Other: {other_criterion}").pack(anchor='w', pady=2)
        except Exception:
            pass

        # center column gets the slider and radios; keep slider reasonably sized
        slider_length = 400
        slider = tk.Scale(center_col, from_=mins[slider_target], to=maxs[slider_target], orient=tk.HORIZONTAL, resolution=resolution, length=slider_length)
        # Add a small canvas above the slider to show the consistency threshold marker
        threshold_canvas = tk.Canvas(center_col, width=slider_length, height=14, highlightthickness=0)
        threshold_canvas.pack(pady=(4, 0))
        slider.pack(pady=(2, 8))

        # Fetch the VF threshold for consistency guidance if available
        y_thresh = self._bw_vf_thresholds.get(group_name, None)
        
        # For worst comparisons, also check ordinal constraint from already-completed worst comparisons
        y_min_ordinal, y_max_ordinal = None, None
        if comparison_type == 'worst':
            y_min_ordinal, y_max_ordinal = self._compute_worst_ordinal_constraint(group_name, other_criterion)

        def on_slide(val):
            try:
                v = float(val)
            except Exception:
                return
            # Compute VF for the slider target only; others remain at 0 for display
            idx = slider_target
            c = group_criteria[idx]
            vf = c.get('value_function')
            if callable(vf):
                try:
                    h = float(vf(v))
                except Exception:
                    h = 0.001
            else:
                lo = mins[idx]
                hi = maxs[idx]
                try:
                    if hi == lo:
                        h = 1.0
                    else:
                        h = (v - lo) / (hi - lo)
                except Exception:
                    h = 0.001
            h = float(np.clip(h, 0.001, 1.0))
            for i, rect in enumerate(bars_right):
                rect.set_height(h if i == idx else 0.0)
                # Reset non-target bar color; target colored below based on consistency
                if i != idx:
                    try:
                        rect.set_facecolor('#9ecae1')
                    except Exception:
                        pass
            # Check consistency against threshold and update slider accent color
            violation = False
            if y_thresh is not None:
                if comparison_type == 'best':
                    violation = (h + 1e-9) < y_thresh  # require h >= y_thresh (equal allowed)
                else:
                    violation = (h - 1e-9) > y_thresh  # require h <= y_thresh (equal allowed)
            # Also check ordinal constraints for worst phase (from completed worst comparisons)
            if comparison_type == 'worst':
                if y_min_ordinal is not None:
                    violation = violation or ((h + 1e-9) < y_min_ordinal)  # require h >= lower bound
                if y_max_ordinal is not None:
                    violation = violation or ((h - 1e-9) > y_max_ordinal)  # require h <= upper bound
            # Update slider accent color (may vary by Tk theme)
            try:
                if y_thresh is not None or y_min_ordinal is not None or y_max_ordinal is not None:
                    slider.configure(activebackground=('#d62728' if violation else '#2b78c8'))
            except Exception:
                pass
            # keep radio buttons in sync (if present)
            try:
                # indicate we're updating radios programmatically so the trace ignores it
                if 'rb_state' in locals():
                    rb_state['updating'] = True
                rb_var.set(int(round(h * 10)))
            except Exception:
                pass
            finally:
                try:
                    if 'rb_state' in locals():
                        rb_state['updating'] = False
                except Exception:
                    pass
            # update value-function marker if present
            try:
                if vf_marker is not None:
                    vf_marker.set_data([v], [h])
            except Exception:
                pass
            canvas.draw_idle()

        slider.configure(command=on_slide)

        # Initialize slider position to x where vf(x)=0 and update the right bars
        try:
            # Clip initial_x_zero into allowable data range
            initial_x_zero = max(min(initial_x_zero, maxs[slider_target]), mins[slider_target])
            slider.set(initial_x_zero)
            # Trigger the slide handler to update bar heights accordingly
            on_slide(initial_x_zero)
        except Exception:
            # Fallback: set to min
            slider.set(mins[slider_target])

        # --- Radio buttons: choose VF levels 0.0,0.1,...,1.0 and jump slider to x with vf(x)=y ---
        try:
            # Determine initial VF value at current slider pos for radio init
            try:
                vf_fn = group_criteria[slider_target].get('value_function')
                if callable(vf_fn):
                    current_y = float(vf_fn(float(slider.get())))
                else:
                    lo = mins[slider_target]; hi = maxs[slider_target]
                    current_y = 0.001 if hi == lo else ((float(slider.get()) - lo) / (hi - lo))
            except Exception:
                current_y = 0.0
            current_y = float(np.clip(current_y, 0.0, 1.0))
            rb_init = int(round(current_y * 10))
            rb_var = tk.IntVar(value=rb_init)
            # guard to avoid recursive updates between slider->radio and radio->slider
            rb_state = {'updating': False}

            radiobtn_frame = ttk.Frame(center_col)
            radiobtn_frame.pack(pady=(4, 6))
            # create radio buttons horizontally
            for i in range(11):
                rb = ttk.Radiobutton(radiobtn_frame, text=f"{i/10:.1f}", variable=rb_var, value=i)
                rb.pack(side=tk.LEFT, padx=2)

            # trace callback: when the IntVar changes (user clicks a radio), move the slider
            def on_rb_change(*args):
                # if we're programmatically updating the radio from the slider, ignore
                try:
                    if rb_state.get('updating'):
                        return
                except Exception:
                    pass
                try:
                    ix = int(rb_var.get())
                    target_y = ix / 10.0
                    # prefer precomputed mapping if available
                    pre = group_criteria[slider_target].get('_precomputed_x_for_y')
                    try:
                        if pre and 0 <= ix < len(pre) and pre[ix] is not None:
                            x = float(pre[ix])
                        else:
                            x = compute_x_for_vf(group_criteria[slider_target], target_y, mins[slider_target], maxs[slider_target])
                    except Exception:
                        x = compute_x_for_vf(group_criteria[slider_target], target_y, mins[slider_target], maxs[slider_target])
                    # clip into allowed data range (support lo>hi)
                    low = float(min(mins[slider_target], maxs[slider_target]))
                    high = float(max(mins[slider_target], maxs[slider_target]))
                    x = float(np.clip(x, low, high))
                    slider.set(x)
                    on_slide(x)
                except Exception:
                    pass

            try:
                # trace_add is available on modern Tkinter; fallback to trace for older versions
                if hasattr(rb_var, 'trace_add'):
                    rb_var.trace_add('write', on_rb_change)
                else:
                    rb_var.trace('w', lambda *a: on_rb_change())
            except Exception:
                pass
        except Exception:
            # don't fail the UI if radios can't be created
            pass

        # --- Consistency threshold markers ---
        try:
            # Fetch the B-W threshold
            y_thresh = self._bw_vf_thresholds.get(group_name, None)
            # Fetch the ordinal constraint (worst phase only)
            y_ordinal_lower, y_ordinal_upper = (None, None)
            if comparison_type == 'worst':
                y_ordinal_lower, y_ordinal_upper = self._compute_worst_ordinal_constraint(group_name, other_criterion)
            # Use whichever ordinal bound exists (prefer lower if both exist)
            y_ordinal = y_ordinal_lower if y_ordinal_lower is not None else y_ordinal_upper
            
            # Draw both markers if available
            threshold_canvas.delete('all')
            low = float(min(mins[slider_target], maxs[slider_target]))
            high = float(max(mins[slider_target], maxs[slider_target]))
            
            # Helper to convert VF value to pixel position
            def y_to_px(y_vf):
                try:
                    pre = group_criteria[slider_target].get('_precomputed_x_for_y')
                    ix = int(round(float(np.clip(y_vf, 0.0, 1.0)) * 10))
                    if pre and 0 <= ix < len(pre) and pre[ix] is not None:
                        x_val = float(pre[ix])
                    else:
                        x_val = compute_x_for_vf(group_criteria[slider_target], float(y_vf), mins[slider_target], maxs[slider_target])
                except Exception:
                    x_val = compute_x_for_vf(group_criteria[slider_target], float(y_vf), mins[slider_target], maxs[slider_target])
                if high == low:
                    return slider_length / 2.0
                else:
                    frac = float(np.clip((x_val - low) / (high - low), 0.0, 1.0))
                    return 2 + frac * (slider_length - 4)
            
            # Draw B-W threshold marker (if available)
            if y_thresh is not None:
                px_bw = y_to_px(y_thresh)
                threshold_canvas.create_polygon(px_bw-5, 2, px_bw+5, 2, px_bw, 12, fill='red', outline='black')
                threshold_canvas.create_text(px_bw, 1, text='A', font=('Helvetica', 8, 'bold'), fill='white', anchor='s')
                # Also mark on VF plot
                try:
                    if ax_vf is not None:
                        ax_vf.plot([compute_x_for_vf(group_criteria[slider_target], y_thresh, mins[slider_target], maxs[slider_target])], [y_thresh], marker='o', color='red', markersize=5)
                except Exception:
                    pass
            
            # Draw ordinal constraint marker (worst phase only, if available)
            if comparison_type == 'worst' and y_ordinal is not None:
                px_ord = y_to_px(y_ordinal)
                threshold_canvas.create_polygon(px_ord-5, 2, px_ord+5, 2, px_ord, 12, fill='orange', outline='black')
                threshold_canvas.create_text(px_ord, 1, text='B', font=('Helvetica', 8, 'bold'), fill='white', anchor='s')
                # Also mark on VF plot
                try:
                    if ax_vf is not None:
                        ax_vf.plot([compute_x_for_vf(group_criteria[slider_target], y_ordinal, mins[slider_target], maxs[slider_target])], [y_ordinal], marker='s', color='orange', markersize=5)
                except Exception:
                    pass
            
            # Redraw canvas if either marker was added
            try:
                canvas.draw_idle()
            except Exception:
                pass
        except Exception:
            # non-fatal if marker cannot be drawn
            pass

        # Create the per-comparison confidence selector on the right of the slider
        try:
            conf_options = [
                "0: Not confident at all (pure guess)",
                "1: Low confidence",
                "2: Moderately confident",
                "3: High confidence",
                "4: Extremely confident (certain)",
            ]
            # Check if we have a restored confidence value from CSV
            restored_conf = getattr(self, '_current_restored_conf', None)
            if restored_conf is not None:
                default_conf = int(restored_conf)
                self._current_restored_conf = None
            else:
                try:
                    default_conf = int(self.group_selected.get(group_name, {}).get('confidence', 2))
                except Exception:
                    default_conf = 2
            self._current_conf_var = tk.StringVar(value=conf_options[default_conf])
            # add a top offset so the confidence selector sits aligned with the slider
            ttk.Label(right_col, text='Confidence:').pack(anchor='w', pady=(18, 0))
            conf_cb = ttk.Combobox(right_col, textvariable=self._current_conf_var, values=conf_options, width=40)
            conf_cb.pack(pady=(8, 0))
        except Exception:
            self._current_conf_var = None

        # Create navigation buttons: Back and Next (triangular) inline in the right column
        nav_frame = ttk.Frame(right_col)
        nav_frame.pack(side=tk.RIGHT, padx=(8, 4), pady=(4, 4))
        
        # Back button (only show if not on first comparison)
        if self.current_comparison > 0:
            try:
                back_h = 56
                back_w = 48
                back_canvas = tk.Canvas(nav_frame, width=back_w, height=back_h, highlightthickness=0)
                # draw a left-pointing triangle
                back_canvas.create_polygon(back_w-4, 4, back_w-4, back_h-4, 6, back_h/2, fill='#c8782b', outline='black')
                back_canvas.configure(cursor='hand2')
                back_canvas.pack(side=tk.LEFT, padx=(4, 4))
                def on_back_click(event=None):
                    try:
                        self.go_back()
                    except Exception:
                        pass
                back_canvas.bind('<Button-1>', on_back_click)
            except Exception:
                # fallback to a regular button if canvas creation fails
                back_btn = ttk.Button(nav_frame, text='Back', command=self.go_back)
                back_btn.pack(side=tk.LEFT, padx=(4, 4))
        
        # Next button (triangular)
        try:
            tri_h = 56
            tri_w = 48
            tri_canvas = tk.Canvas(nav_frame, width=tri_w, height=tri_h, highlightthickness=0)
            # draw a right-pointing triangle
            tri_canvas.create_polygon(4, 4, 4, tri_h-4, tri_w-6, tri_h/2, fill='#2b78c8', outline='black')
            tri_canvas.configure(cursor='hand2')
            tri_canvas.pack(side=tk.LEFT, padx=(4, 4))
            def on_tri_click(event=None):
                try:
                    self.save_and_next()
                except Exception:
                    pass
            tri_canvas.bind('<Button-1>', on_tri_click)
        except Exception:
            # fallback to a regular button if canvas creation fails
            next_btn = ttk.Button(nav_frame, text='Next', command=self.save_and_next)
            next_btn.pack(side=tk.LEFT, padx=(4, 4))

        # Save state
        self._current_comp = (comparison_type, ref_criterion, other_criterion)
        self._slider = slider
        self._canvas = canvas
        self._bars_right = bars_right
        self._right_vals = right_vals
        self._mins = mins
        self._maxs = maxs
        self._group_criteria = group_criteria
        self._slider_target = slider_target



    def save_current_comparison(self):
        """Save the current comparison value to CSV (either append or update)"""
        comp_type, ref, other = self._current_comp
        try:
            val = float(self._slider.get())
        except Exception:
            val = 0.0

        # Get the results file
        fn = getattr(self, '_results_fn', None)
        if fn is None:
            # Fallback: use the canonical file in the module dir
            ui_dir = os.path.dirname(os.path.abspath(__file__))
            fn = os.path.join(ui_dir, 'BWT_results.csv')
            if not os.path.exists(fn):
                with open(fn, 'w', newline='') as f:
                    csv.writer(f).writerow(['Type', 'Reference', 'Other', 'Value', 'Group', 'Confidence', 'a'])

        # write row with Group (use current context)
        group_name = getattr(self, '_context_group_name', self.groups[self.current_group_idx] if self.groups else 'Ungrouped')
        # determine confidence to write: prefer the per-screen selector, fall back to group default
        try:
            conf_val = None
            if hasattr(self, '_current_conf_var') and self._current_conf_var is not None:
                conf_val = int(self._current_conf_var.get().split(':', 1)[0])
            if conf_val is None:
                conf_val = int(self.group_selected.get(group_name, {}).get('confidence', 2))
        except Exception:
            conf_val = ''

        # Compute a = 1/vf(value)
        a_val = ''
        try:
            # Get the criterion that was adjusted (slider_target)
            slider_target = getattr(self, '_slider_target', None)
            group_criteria = getattr(self, '_group_criteria', None)
            if slider_target is not None and group_criteria is not None:
                criterion = group_criteria[slider_target]
                vf = criterion.get('value_function')
                if callable(vf):
                    vf_val = float(vf(val))
                    # Avoid division by zero
                    if vf_val != 0:
                        a_val = 1.0 / vf_val
        except Exception:
            # If anything fails, leave a_val empty
            pass

        # Read the entire CSV
        rows = []
        header = ['Type', 'Reference', 'Other', 'Value', 'Group', 'Confidence', 'a']
        try:
            if os.path.exists(fn):
                with open(fn, 'r', newline='') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if not rows or rows[0] != header:
                        rows = [header]
            else:
                rows = [header]
        except Exception:
            rows = [header]

        # Find if this comparison already exists
        existing_row_idx = None
        for i, row in enumerate(rows):
            if i == 0:  # skip header
                continue
            if (len(row) >= 5 and row[0] == comp_type and row[1] == ref and 
                row[2] == other and row[4] == group_name):
                existing_row_idx = i
                break

        # Update or add the row
        new_row = [comp_type, ref, other, val, group_name, conf_val, a_val]
        if existing_row_idx is not None:
            rows[existing_row_idx] = new_row
        else:
            rows.append(new_row)

        # Write the complete CSV
        with open(fn, 'w', newline='') as f:
            w = csv.writer(f)
            for row in rows:
                w.writerow(row)

        # keep an in-memory record (optional) keyed by criterion name
        if comp_type == 'best':
            self.best_comparison_results[other] = val
        else:
            self.worst_comparison_results[other] = val

        # If this saved comparison is the Best-vs-Worst for the current context,
        # compute and store the VF threshold so following screens can show guidance.
        try:
            group_name = getattr(self, '_context_group_name', self.groups[self.current_group_idx] if self.groups else 'Ungrouped')
            sel = self.group_selected.get(group_name, {})
            worst_name = sel.get('worst')
            best_name = sel.get('best')
            slider_target = getattr(self, '_slider_target', None)
            group_criteria = getattr(self, '_group_criteria', None)
            if comp_type == 'best' and best_name and worst_name and ref == best_name and other == worst_name and slider_target is not None and group_criteria is not None:
                criterion = group_criteria[slider_target]
                vf = criterion.get('value_function')
                # compute VF value (y in [0,1]) for current saved raw x 'val'
                try:
                    if callable(vf):
                        y = float(vf(val))
                    else:
                        lo = float(criterion.get('min', 0.0))
                        hi = float(criterion.get('max', lo + 1.0))
                        if hi == lo:
                            y = 1.0
                        else:
                            y = (val - lo) / (hi - lo)
                except Exception:
                    y = None
                if y is not None:
                    self._bw_vf_thresholds[group_name] = float(np.clip(y, 0.0, 1.0))
        except Exception:
            pass

    def save_and_next(self):
        # Save the current comparison and move to the next one
        # Only save if we're not on a summary screen
        if self.current_comparison < len(self.comparisons):
            self.save_current_comparison()
            self.current_comparison += 1
        else:
            # We're on a summary screen, proceed with next action
            self.current_comparison += 1
        
        # Check if we've exhausted all comparisons
        if self.current_comparison >= len(self.comparisons):
            # Show summary/confirmation screen
            self.show_summary_screen()
        else:
            # Show next comparison
            self.show_next_comparison()
    
    def show_summary_screen(self):
        # Show a summary screen when group comparisons are complete
        for widget in self.root.winfo_children():
            widget.destroy()
        
        frm = ttk.Frame(self.root, padding=8)
        frm.pack(fill=tk.BOTH, expand=True)
        
        # Use the context group name from comparisons, not the current_group_idx
        # This ensures we're showing data for the group we actually just finished
        group_name = getattr(self, '_context_group_name', 
                            self.groups[self.current_group_idx] if self.groups else 'Ungrouped')
        ttk.Label(frm, text=f"Group '{group_name}' Elicitation Complete", font=(None, 14, 'bold')).pack(pady=(20, 10))
        ttk.Label(frm, text="All comparisons for this group have been completed.", font=(None, 11)).pack(pady=5)
        ttk.Label(frm, text="You can modify comparisons by clicking Back.", font=(None, 10), foreground='gray').pack(pady=5)
        
        # Toggle button to show/hide consistency plot (hidden by default)
        plot_frame = ttk.Frame(frm)
        plot_frame.pack(pady=(10, 10))
        
        # State holder for the plot
        plot_state = {'visible': False, 'widget': None}
        
        def toggle_plot():
            if plot_state['visible']:
                # Hide the plot
                if plot_state['widget'] is not None:
                    plot_state['widget'].pack_forget()
                plot_state['visible'] = False
                toggle_btn.configure(text='Show Consistency Check')
            else:
                # Show the plot
                if plot_state['widget'] is None:
                    # Create the plot widget
                    try:
                        plot_state['widget'] = ttk.Frame(frm)
                        self.create_consistency_plot(plot_state['widget'], group_name)
                    except Exception as e:
                        ttk.Label(plot_state['widget'], text=f"Could not generate plot: {str(e)}", foreground='red').pack(pady=10)
                else:
                    # Just show the existing widget
                    plot_state['widget'].pack(pady=(10, 20), fill=tk.BOTH, expand=True)
                plot_state['visible'] = True
                toggle_btn.configure(text='Hide Consistency Check')
        
        toggle_btn = ttk.Button(plot_frame, text='Show Consistency Check', command=toggle_plot)
        toggle_btn.pack(pady=(10, 0))
        
        # Navigation buttons
        btn_frame = ttk.Frame(frm)
        btn_frame.pack(pady=(30, 10))
        
        ttk.Button(btn_frame, text='Back to Comparisons', command=self.go_back_to_comparisons).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text='Continue', command=self.proceed_from_summary).pack(side=tk.LEFT, padx=5)
    
    def create_consistency_plot(self, parent_frame, group_name):
        """Create a horizontal bar plot showing 'a' values for consistency checking"""
        fn = getattr(self, '_results_fn', None)
        if fn is None:
            ui_dir = os.path.dirname(os.path.abspath(__file__))
            fn = os.path.join(ui_dir, 'BWT_results.csv')
        
        if not os.path.exists(fn):
            return
        
        # Get best and worst criteria for this group
        group_selected = self.group_selected.get(group_name, {})
        best_criterion = group_selected.get('best', None)
        worst_criterion = group_selected.get('worst', None)
        
        # Read CSV and filter data for current group
        comparisons = []
        a_values = []
        labels = []
        
        try:
            with open(fn, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Only include rows from current group
                    if row.get('Group') != group_name:
                        continue
                    
                    comp_type = row.get('Type', '')
                    ref = row.get('Reference', '')
                    other = row.get('Other', '')
                    
                    try:
                        a_val = float(row.get('a', 0))
                        # Invert a_val for worst comparisons
                        if comp_type == 'worst' and a_val != 0:
                            a_val = 1.0 / a_val
                    except Exception:
                        a_val = 0
                    
                    comparisons.append((comp_type, ref, other))
                    a_values.append(a_val)
                    
                    # Create label with best/worst highlighted
                    # Format: "best/worst: Reference ← → Other"
                    ref_label = f"**{ref}**" if ref in [best_criterion, worst_criterion] else ref
                    other_label = f"**{other}**" if other in [best_criterion, worst_criterion] else other
                    label = f"{comp_type}: {ref_label} ← → {other_label}"
                    labels.append(label)
        except Exception:
            return
        
        if not a_values:
            return
        
        # Create matplotlib figure: slightly narrower width, more left padding
        fig = plt.Figure(figsize=(7, max(3, len(a_values) * 0.5)))
        ax = fig.add_subplot(111)
        # Ensure plenty of space for long y-axis labels
        try:
            fig.subplots_adjust(left=0.35, right=0.98, top=0.9, bottom=0.12)
        except Exception:
            pass
        
        # Create horizontal bar chart with colors for best/worst
        colors = []
        for comp_type, ref, other in comparisons:
            if ref in [best_criterion, worst_criterion]:
                colors.append('#2b78c8' if comp_type == 'best' else '#c8782b')  # Blue for best, orange for worst
            else:
                colors.append('#cccccc')
        
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, a_values, color=colors, alpha=0.7, edgecolor='black')
        
        # Set labels with best/worst in bold using mathtext ($\\bf{...}$)
        def bold_name(name):
            if name in [best_criterion, worst_criterion]:
                # escape backslashes and replace spaces for mathtext
                safe = str(name).replace('\\', '\\\\').replace(' ', '\\ ')
                return f'$\\bf{{{safe}}}$'
            return name
        y_labels = []
        for comp_type, ref, other in comparisons:
            ref_fmt = bold_name(ref)
            other_fmt = bold_name(other)
            label = f"{comp_type}: {ref_fmt} ← → {other_fmt}"
            y_labels.append(label)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=9)
        # Align y tick labels to the right so they are fully visible
        try:
            for tick in ax.get_yticklabels():
                tick.set_horizontalalignment('right')
        except Exception:
            pass
        ax.set_xlabel('a values', fontsize=10)
        ax.set_title(f'Consistency Check - Group: {group_name}\n(Best/Worst highlighted)', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Embed the plot in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(10, 20))
    
    def go_back_to_comparisons(self):
        # Go back to the last comparison
        self.current_comparison = len(self.comparisons) - 1
        self.show_next_comparison()
    
    def proceed_from_summary(self):
        # Proceed to next group or finalize
        if self.current_group_idx < len(self.groups) - 1:
            # Ask if user wants to continue to next group
            group_name = self.groups[self.current_group_idx]
            next_group = self.groups[self.current_group_idx + 1]
            proceed = messagebox.askyesno('Continue to Next Group', 
                f'Completed group: {group_name}\n\nContinue with group: {next_group}?')
            if proceed:
                self.current_group_idx += 1
                # Show selection UI for next group so user may choose best/worst
                self.setup_selection_ui()
                return
            else:
                # User wants to stay, go back to comparisons
                self.current_comparison = len(self.comparisons) - 1
                self.show_next_comparison()
                return
        else:
            # We've finished all comparisons in the current context.
            scope = getattr(self, '_context_scope', 'intra-group')
            # If we finished intra-group phase for all groups, ask about between-groups Best
            if scope.startswith('intra-group'):
                proceed_B = messagebox.askyesno('Between-groups Best', 'All intra-group elicitation completed. Proceed with between-groups BEST comparisons?')
                if proceed_B:
                    self.prepare_intergroup_phase('B')
                    return
                # If user declines B, ask about between-groups W
                proceed_W = messagebox.askyesno('Between-groups Worst', 'Proceed with between-groups WORST comparisons?')
                if proceed_W:
                    self.prepare_intergroup_phase('W')
                    return
                self.finalize_results()
                return
            elif scope.startswith('between-groups-B'):
                # After between-groups B, ask about between-groups W
                proceed_W = messagebox.askyesno('Between-groups Worst', 'Between-groups BEST completed. Proceed with between-groups WORST comparisons?')
                if proceed_W:
                    self.prepare_intergroup_phase('W')
                    return
                self.finalize_results()
                return
            else:
                # between-groups-W or unknown scope: finalize
                self.finalize_results()

    def go_back(self):
        # Save the current comparison before going back
        # The value restoration is handled by show_next_comparison which always checks CSV
        if self.current_comparison <= 0:
            return
        
        # Save current comparison first
        self.save_current_comparison()
        
        # Decrement comparison index
        self.current_comparison -= 1
        
        # Clear the revisit flag since we're navigating normally
        self._is_revisit = False
        
        # Show the comparison (which will automatically restore values from CSV)
        self.show_next_comparison()
    
    def save_results_to_file(self):
        fn = getattr(self, '_results_fn', None) or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'BWT_results.csv')
        messagebox.showinfo('Info', f'Results are saved automatically during elicitation to {os.path.basename(fn)}')

    def finalize_results(self):
        # Finalize: inform the user and close the window. Individual comparisons
        # were appended to `BWT_results.csv` as they were entered.
        fn = getattr(self, '_results_fn', None) or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'BWT_results.csv')
        messagebox.showinfo('Completed', f'Elicitation completed. Results appended to {os.path.basename(fn)}')
        self.root.quit()
        self.root.destroy()

    def cancel(self):
        fn = getattr(self, '_results_fn', None) or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'BWT_results.csv')
        if messagebox.askyesno('Cancel', f'Abort elicitation? Already-saved comparisons will stay in {os.path.basename(fn)}'):
            self.root.quit()
            self.root.destroy()

