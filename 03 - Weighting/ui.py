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
        ttk.Button(selection_frame, text="Continue", command=self.start_comparisons).grid(row=3, columnspan=2)

        # Show a single overview plot with all criteria at their max and their ranges
        try:
            # Overview plot for the current group
            group_name = self.groups[self.current_group_idx] if self.groups else 'Ungrouped'
            group_criteria = self.group_map.get(group_name, [])
            names = [c.get('name') for c in group_criteria]
            display_names = [f"{c.get('name')} ({c.get('unit')})" if c.get('unit') else c.get('name') for c in group_criteria]
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

            # Values: show all criteria at their max (for overview)
            vals = maxs.copy()

            fig = plt.Figure(figsize=(8, 3))
            ax = fig.add_subplot(1, 1, 1)

            # Normalize to 0..1 for display
            normalized = []
            for i, v in enumerate(vals):
                lo = mins[i]
                hi = maxs[i]
                try:
                    normalized.append((v - lo) / (hi - lo))
                except Exception:
                    normalized.append(0.0)

            bars = ax.bar(range(n), normalized, tick_label=display_names)
            ax.set_ylim(0, 1)
            ax.set_title(f'Overview — {group_name} (all criteria at max)')
            for j in range(n):
                ax.text(j, 0.02, f"{mins[j]:.2f}", ha='center', va='bottom', fontsize=8)
                ax.text(j, 0.98, f"{maxs[j]:.2f}", ha='center', va='top', fontsize=8)

            # Embed the overview plot inside the selection frame to the right
            selection_frame.grid_columnconfigure(2, weight=1)
            plot_frame = ttk.Frame(selection_frame)
            plot_frame.grid(row=0, column=2, rowspan=4, sticky='nsew', padx=(8, 0))
            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
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
        self.group_selected[group_name] = {'best': best, 'worst': worst}

        # set current context for plotting/saving
        self._context_criteria = group_criteria
        self._context_group_name = group_name
        self._context_scope = 'intra-group'

        # Generate best comparisons (best vs each other criterion in group)
        self.comparisons = []
        for name in group_names:
            if name != best:
                self.comparisons.append(("best", best, name))
        # Generate worst comparisons (worst vs each other criterion in group)
        for name in group_names:
            # skip the worst itself and also skip the best to avoid duplicating
            # the inverse comparison (best vs worst) which was already added
            if name != worst and name != best:
                self.comparisons.append(("worst", worst, name))
        self.current_comparison = 0
        # Use a single persistent results file for all groups: 'wbt_results.csv'
        ui_dir = os.path.dirname(os.path.abspath(__file__))
        self._results_fn = os.path.join(ui_dir, 'wbt_results.csv')
        # Create the file with header if it doesn't exist
        if not os.path.exists(self._results_fn):
            with open(self._results_fn, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['Type', 'Reference', 'Other', 'Value', 'Group'])

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
            display_names = [f"{c.get('name')} ({c.get('unit')})" if c.get('unit') else c.get('name') for c in crit_list]
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

            fig = plt.Figure(figsize=(4, 2.5))
            ax = fig.add_subplot(1, 1, 1)
            # Represent range as normalized bar from 0 to 1
            normalized = [1.0] * len(display_names)
            bars = ax.bar(range(len(display_names)), normalized, tick_label=display_names)
            ax.set_ylim(0, 1)
            ax.set_title('Candidate ranges (max shown)')
            for j in range(len(display_names)):
                ax.text(j, 0.02, f"{mins[j]:.2f}", ha='center', va='bottom', fontsize=7)
                ax.text(j, 0.98, f"{maxs[j]:.2f}", ha='center', va='top', fontsize=7)

            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
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
            self.start_comparisons_for_context(b, w, context_criteria, group_label, scope_label)

        ttk.Button(sel_frame, text='Continue', command=on_continue).grid(row=3, column=0, columnspan=2, pady=6)

    def start_comparisons_for_context(self, best, worst, context_criteria, context_group_name, scope_label):
        # Set context and prepare comparisons over the provided context_criteria
        self._context_criteria = context_criteria
        self._context_group_name = context_group_name
        self._context_scope = scope_label
        names = [c['name'] for c in context_criteria]
        # store as a synthetic 'group' selection
        self.group_selected[context_group_name] = {'best': best, 'worst': worst}

        # Build comparisons (best vs others, worst vs others) within the context
        self.comparisons = []
        for name in names:
            if name != best:
                self.comparisons.append(('best', best, name))
        for name in names:
            if name != worst and name != best:
                self.comparisons.append(('worst', worst, name))

        self.current_comparison = 0
        self.show_next_comparison()

    def show_next_comparison(self):
        # Minimal elicitation UI: simple labels and a Tk slider. Each comparison
        # is immediately appended to `wbt_results.csv` when the user presses Next.
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
        display_names = [f"{c.get('name')} ({c.get('unit')})" if c.get('unit') else c.get('name') for c in group_criteria]
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

        left_vals = mins.copy()
        right_vals = mins.copy()
        # Configure left plot according to comparison type
        if comparison_type == 'best':
            # For best elicitation: left shows the OTHER criterion maxed,
            # and the user adjusts the REFERENCE (best) on the right.
            left_vals[other_idx] = maxs[other_idx]
            slider_target = ref_idx
        else:
            # For worst comparisons: left shows the REFERENCE (worst) maxed,
            # and the user adjusts the OTHER on the right.
            left_vals[ref_idx] = maxs[ref_idx]
            slider_target = other_idx

        # Right initially all mins (slider will modify slider_target)

        # Create UI frame and titles
        frm = ttk.Frame(self.root, padding=8)
        frm.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frm, text=f"Comparison {self.current_comparison + 1} of {len(self.comparisons)}", font=(None, 12, 'bold')).pack(pady=(0, 6))
        ttk.Label(frm, text=f"Type: {comparison_type}").pack()
        ttk.Label(frm, text=f"Reference: {ref_criterion}").pack()
        ttk.Label(frm, text=f"Other: {other_criterion}").pack(pady=(0, 6))

        # Matplotlib figure with two side-by-side bar plots
        fig = plt.Figure(figsize=(8, 3.5))
        ax_left = fig.add_subplot(1, 2, 1)
        ax_right = fig.add_subplot(1, 2, 2)

        def normalize(vals):
            out = []
            for i, v in enumerate(vals):
                lo = mins[i]
                hi = maxs[i]
                try:
                    out.append((v - lo) / (hi - lo))
                except Exception:
                    out.append(0.0)
            return out

        nl = normalize(left_vals)
        nr = normalize(right_vals)

        n = len(base_names)
        bars_left = ax_left.bar(range(n), nl, tick_label=display_names)
        ax_left.set_ylim(0, 1)
        # Clarify which criterion is maxed on the left
        if comparison_type == 'best':
            ax_left.set_title(f"Other left: {display_names[other_idx]} maxed")
        else:
            ax_left.set_title(f"Reference left: {display_names[ref_idx]} maxed")

        bars_right = ax_right.bar(range(n), nr, tick_label=display_names)
        ax_right.set_ylim(0, 1)
        # Clarify which criterion the user adjusts on the right
        ax_right.set_title(f"Adjustable right: {display_names[slider_target]}")

        # Add min/max text inside the plot (near bottom/top) so it doesn't overlap
        # with outer UI elements.
        for ax, vals in ((ax_left, left_vals), (ax_right, right_vals)):
            for j in range(n):
                ax.text(j, 0.02, f"{mins[j]:.2f}", ha='center', va='bottom', fontsize=8)
                ax.text(j, 0.98, f"{maxs[j]:.2f}", ha='center', va='top', fontsize=8)

        canvas = FigureCanvasTkAgg(fig, master=frm)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Slider to adjust the target criterion on the right plot
        resolution = max((maxs[slider_target] - mins[slider_target]) / 200.0, 1e-6)
        slider = tk.Scale(frm, from_=mins[slider_target], to=maxs[slider_target], orient=tk.HORIZONTAL, resolution=resolution, length=500)
        slider.set(mins[slider_target])
        slider.pack(pady=(6, 8))

        def on_slide(val):
            try:
                v = float(val)
            except Exception:
                return
            right_vals[slider_target] = v
            nr2 = normalize(right_vals)
            for rect, h in zip(bars_right, nr2):
                rect.set_height(h)
            canvas.draw_idle()

        slider.configure(command=on_slide)

        btn_frm = ttk.Frame(frm)
        btn_frm.pack(pady=(6, 2))
        ttk.Button(btn_frm, text='Next', command=self.save_and_next).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frm, text='Cancel', command=self.cancel).pack(side=tk.LEFT, padx=6)

        # Save state
        self._current_comp = (comparison_type, ref_criterion, other_criterion)
        self._slider = slider
        self._canvas = canvas
        self._bars_right = bars_right
        self._right_vals = right_vals
        self._mins = mins
        self._maxs = maxs



    def save_and_next(self):
        # Read slider value and append to CSV immediately.
        comp_type, ref, other = self._current_comp
        try:
            val = float(self._slider.get())
        except Exception:
            val = 0.0

        # Append to the pre-created results file
        fn = getattr(self, '_results_fn', None)
        if fn is None:
            # Fallback: use the canonical file in the module dir
            ui_dir = os.path.dirname(os.path.abspath(__file__))
            fn = os.path.join(ui_dir, 'wbt_results.csv')
            if not os.path.exists(fn):
                with open(fn, 'w', newline='') as f:
                    csv.writer(f).writerow(['Type', 'Reference', 'Other', 'Value', 'Group'])

        # write row with Group (use current context)
        group_name = getattr(self, '_context_group_name', self.groups[self.current_group_idx] if self.groups else 'Ungrouped')
        with open(fn, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([comp_type, ref, other, val, group_name])

        # keep an in-memory record (optional) keyed by criterion name
        if comp_type == 'best':
            self.best_comparison_results[other] = val
        else:
            self.worst_comparison_results[other] = val

        self.current_comparison += 1
        # If we've exhausted comparisons for this group, move to next group or finish
        if self.current_comparison >= len(self.comparisons):
            # Advance to next group if available
            if self.current_group_idx < len(self.groups) - 1:
                messagebox.showinfo('Group completed', f'Completed group: {self.groups[self.current_group_idx]}\nMoving to next group')
                self.current_group_idx += 1
                # Show selection UI for next group so user may choose best/worst
                self.setup_selection_ui()
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
                    return

        self.show_next_comparison()

    def save_results_to_file(self):
        fn = getattr(self, '_results_fn', None) or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wbt_results.csv')
        messagebox.showinfo('Info', f'Results are saved automatically during elicitation to {os.path.basename(fn)}')

    def finalize_results(self):
        # Finalize: inform the user and close the window. Individual comparisons
        # were appended to `wbt_results.csv` as they were entered.
        fn = getattr(self, '_results_fn', None) or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wbt_results.csv')
        messagebox.showinfo('Completed', f'Elicitation completed. Results appended to {os.path.basename(fn)}')
        self.root.quit()
        self.root.destroy()

    def cancel(self):
        fn = getattr(self, '_results_fn', None) or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wbt_results.csv')
        if messagebox.askyesno('Cancel', f'Abort elicitation? Already-saved comparisons will stay in {os.path.basename(fn)}'):
            self.root.quit()
            self.root.destroy()

