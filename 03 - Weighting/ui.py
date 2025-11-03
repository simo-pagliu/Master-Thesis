import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider
from tkinter import ttk, messagebox

class WBT_ui:
    def __init__(self, root, criteria):
        self.root = root
        self.root.title("Criteria Plotter")
        self.criteria = criteria
        self.best_criterion = tk.StringVar()
        self.worst_criterion = tk.StringVar()
        self.current_comparison = 0
        self.comparisons = []
        self.best_comparison_results = None
        self.worst_comparison_results = None

        # Set style for larger fonts
        plt.rcParams.update({'font.size': 12})

        # Initial selection UI
        self.setup_selection_ui()

    def setup_selection_ui(self):
        # Clear previous widgets
        for widget in self.root.winfo_children():
            widget.destroy()

        # Selection frame
        selection_frame = ttk.Frame(self.root, padding="10")
        selection_frame.pack(fill=tk.X)

        ttk.Label(selection_frame, text="Best Criterion:").grid(row=0, column=0)
        self.best_dropdown = ttk.Combobox(selection_frame, textvariable=self.best_criterion, values=[c["name"] for c in self.criteria])
        self.best_dropdown.grid(row=0, column=1)

        ttk.Label(selection_frame, text="Worst Criterion:").grid(row=1, column=0)
        self.worst_dropdown = ttk.Combobox(selection_frame, textvariable=self.worst_criterion, values=[c["name"] for c in self.criteria])
        self.worst_dropdown.grid(row=1, column=1)

        ttk.Button(selection_frame, text="Continue", command=self.start_comparisons).grid(row=2, columnspan=2)

        # Plot all criteria
        self.plot_initial_comparisons()

    def plot_initial_comparisons(self):
        fig, axes = plt.subplots(nrows=1, ncols=len(self.criteria), figsize=(5 * len(self.criteria), 5))
        if len(self.criteria) == 1:
            axes = [axes]

        for i, criterion in enumerate(self.criteria):
            values = [c["min"] for c in self.criteria]
            values[i] = criterion["max"]
            normalized = [(v - c["min"]) / (c["max"] - c["min"]) for v, c in zip(values, self.criteria)]
            bars = axes[i].bar(range(len(self.criteria)), normalized, tick_label=[c["name"] for c in self.criteria])
            axes[i].set_ylim(0, 1)
            axes[i].set_title(f"{criterion['name']} at max", pad=20)

            # Add fixed min/max text for each criterion
            for j, c in enumerate(self.criteria):
                axes[i].text(j, 0.02, f"{c['min']:.1f}", ha='center', va='bottom', fontsize=10)
                axes[i].text(j, 0.98, f"{c['max']:.1f}", ha='center', va='top', fontsize=10)

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.root)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def start_comparisons(self):
        best = self.best_criterion.get()
        worst = self.worst_criterion.get()
        if not best or not worst:
            messagebox.showerror("Error", "Please select both best and worst criteria.")
            return

        self.best = best
        self.worst = worst

        # Initialize result arrays with np.nan
        self.best_comparison_results = [np.nan] * len(self.criteria)
        self.worst_comparison_results = [np.nan] * len(self.criteria)

        # Generate best comparisons (best vs each other criterion)
        self.comparisons = []
        for criterion in self.criteria:
            if criterion["name"] != best:
                self.comparisons.append(("best", best, criterion["name"]))

        # Generate worst comparisons (worst vs each other criterion)
        for criterion in self.criteria:
            if criterion["name"] != worst:
                self.comparisons.append(("worst", worst, criterion["name"]))

        self.current_comparison = 0
        self.show_next_comparison()

    def show_next_comparison(self):
        # Clear previous widgets
        for widget in self.root.winfo_children():
            widget.destroy()

        if self.current_comparison >= len(self.comparisons):
            self.finalize_results()
            return

        comparison_type, ref_criterion, other_criterion = self.comparisons[self.current_comparison]

        # Create figure with adjusted gridspec for more space
        fig = plt.figure(figsize=(15, 9))
        gs = fig.add_gridspec(4, 2, height_ratios=[1, 4, 0.8, 0.5], hspace=0.1, width_ratios=[1, 1])

        ax_left = fig.add_subplot(gs[1, 0])
        ax_right = fig.add_subplot(gs[1, 1])

        # Get indices for best and worst criteria
        ref_idx = next(i for i, c in enumerate(self.criteria) if c["name"] == ref_criterion)
        other_idx = next(i for i, c in enumerate(self.criteria) if c["name"] == other_criterion)
        best_idx = next(i for i, c in enumerate(self.criteria) if c["name"] == self.best)
        worst_idx = next(i for i, c in enumerate(self.criteria) if c["name"] == self.worst)

        # Reorder criteria to put best on left and worst on right
        ordered_criteria = self.criteria.copy()
        if ref_criterion == self.best:
            # Put best on left, worst on right
            ordered_criteria = [self.criteria[best_idx]] + \
                              [c for i, c in enumerate(self.criteria) if i != best_idx and i != worst_idx] + \
                              [self.criteria[worst_idx]]
        elif ref_criterion == self.worst:
            # Put best on left, worst on right
            ordered_criteria = [self.criteria[best_idx]] + \
                              [c for i, c in enumerate(self.criteria) if i != best_idx and i != worst_idx] + \
                              [self.criteria[worst_idx]]

        # Update indices based on ordered criteria
        ordered_indices = {c["name"]: i for i, c in enumerate(ordered_criteria)}
        ref_idx = ordered_indices[ref_criterion]
        other_idx = ordered_indices[other_criterion]

        # Left plot: Other criterion at max
        values_left = [c["min"] for c in ordered_criteria]
        values_left[other_idx] = ordered_criteria[other_idx]["max"]

        # Right plot: All at min, adjust ref_criterion
        values_right = [c["min"] for c in ordered_criteria]

        # Normalize values
        normalized_left = [(v - c["min"]) / (c["max"] - c["min"]) for v, c in zip(values_left, ordered_criteria)]
        normalized_right = [(v - c["min"]) / (c["max"] - c["min"]) for v, c in zip(values_right, ordered_criteria)]

        # Plot left (other criterion at max)
        bars_left = ax_left.bar(range(len(ordered_criteria)), normalized_left, tick_label=[c["name"] for c in ordered_criteria])
        ax_left.set_ylim(0, 1)
        ax_left.set_title(f"Having {ref_criterion} at {ordered_criteria[ref_idx]['min']:.1f} and {other_criterion} at {values_left[other_idx]:.1f}",
                         fontsize=12, pad=20)

        # Plot right (adjust ref_criterion)
        bars_right = ax_right.bar(range(len(ordered_criteria)), normalized_right, tick_label=[c["name"] for c in ordered_criteria])
        ax_right.set_ylim(0, 1)
        ax_right.set_title(f"Having {other_criterion} at {ordered_criteria[other_idx]['min']:.1f} and {ref_criterion} at {values_right[ref_idx]:.1f}",
                          fontsize=12, pad=20)

        # Add "is indifferent to" text in its own row with more space
        fig.text(0.5, 0.52, "is indifferent to", ha='center', va='center', fontsize=14, fontweight='bold')

        # Add fixed min/max text for both plots
        for ax, values in [(ax_left, values_left), (ax_right, values_right)]:
            for j, c in enumerate(ordered_criteria):
                ax.text(j, 0.02, f"{c['min']:.1f}", ha='center', va='bottom', fontsize=10)
                ax.text(j, 0.98, f"{c['max']:.1f}", ha='center', va='top', fontsize=10)

        # Create slider axes with no visible elements
        slider_ax = plt.axes([0.3, 0.05, 0.4, 0.02])
        slider_ax.set_xticks([])
        slider_ax.set_yticks([])
        slider_ax.patch.set_alpha(0.0)

        # Create slider with no labels
        slider = Slider(
            ax=slider_ax,
            label='',
            valmin=ordered_criteria[ref_idx]["min"],
            valmax=ordered_criteria[ref_idx]["max"],
            valinit=ordered_criteria[ref_idx]["min"],
        )

        # Remove slider labels and ticks
        slider.label.set_visible(False)
        slider.valtext.set_visible(False)
        slider.poly.set_visible(False)  # Remove the slider line

        def update(val):
            values_right[ref_idx] = slider.val
            normalized_right = [(v - c["min"]) / (c["max"] - c["min"]) for v, c in zip(values_right, ordered_criteria)]
            for i, bar in enumerate(bars_right):
                bar.set_height(normalized_right[i])
            ax_right.set_title(f"Having {other_criterion} at {ordered_criteria[other_idx]['min']:.1f} and {ref_criterion} at {slider.val:.1f}",
                              fontsize=12, pad=20)
            fig.canvas.draw_idle()

        slider.on_changed(update)

        # Adjust the space between subplots
        plt.subplots_adjust(wspace=0.4)

        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.root)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Next button
        next_button = ttk.Button(self.root, text="Next", command=self.save_and_next)
        next_button.pack(pady=10)

        # Store references
        self.current_fig = fig
        self.current_slider = slider
        self.current_ref_idx = ref_idx
        self.current_values_right = values_right
        self.current_ordered_criteria = ordered_criteria
        self.current_ref_criterion = ref_criterion
        self.current_other_criterion = other_criterion
        self.current_other_idx = other_idx

    def save_and_next(self):
        # Save adjusted value
        ref_criterion = self.current_ref_criterion
        other_criterion = self.current_other_criterion
        adjusted_value = self.current_slider.val

        # Normalize the adjusted value to [0, 1] range
        ref_idx = next(i for i, c in enumerate(self.criteria) if c["name"] == ref_criterion)
        normalized_value = (adjusted_value - self.criteria[ref_idx]["min"]) / (self.criteria[ref_idx]["max"] - self.criteria[ref_idx]["min"])

        # Store in the appropriate results array
        if self.comparisons[self.current_comparison][0] == "best":
            # Find index of other_criterion in the original criteria list
            other_idx = next(i for i, c in enumerate(self.criteria) if c["name"] == other_criterion)
            # Store the normalized value in the correct position
            self.best_comparison_results[other_idx] = normalized_value
        else:  # worst
            # Find index of other_criterion in the original criteria list
            other_idx = next(i for i, c in enumerate(self.criteria) if c["name"] == other_criterion)
            # Store the normalized value in the correct position
            self.worst_comparison_results[other_idx] = normalized_value

        self.current_comparison += 1
        self.show_next_comparison()

    def finalize_results(self):
        # Convert numpy floats to regular floats and round to 2 decimal places
        best_results = [round(float(x), 2) if not np.isnan(x) else np.nan for x in self.best_comparison_results]
        worst_results = [round(float(x), 2) if not np.isnan(x) else np.nan for x in self.worst_comparison_results]

        # Store results in class variables
        self.best_comparison_results = best_results
        self.worst_comparison_results = worst_results

        # Close the window
        self.root.quit()
        self.root.destroy()
