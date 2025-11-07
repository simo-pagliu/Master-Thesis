import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider
from tkinter import ttk, messagebox
import csv

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
        self.reordered_criteria_names = None
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
                self.comparisons.append(("worst", worst,  criterion["name"]))
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
        best_idx = next(i for i, c in enumerate(self.criteria) if c["name"] == self.best)
        worst_idx = next(i for i, c in enumerate(self.criteria) if c["name"] == self.worst)

        # Reorder criteria to put best on left and worst on right
        ordered_criteria = self.criteria.copy()
        if ref_criterion == self.best or ref_criterion == self.worst:
            ordered_criteria = [self.criteria[best_idx]] + \
                            [c for i, c in enumerate(self.criteria) if i != best_idx and i != worst_idx] + \
                            [self.criteria[worst_idx]]

        # Update indices based on ordered criteria
        ordered_indices = {c["name"]: i for i, c in enumerate(ordered_criteria)}
        ref_idx = ordered_indices[ref_criterion]
        other_idx = ordered_indices[other_criterion]

        # Left plot: Fixed criterion at max (for best) or max (for worst)
        values_left = [c["min"] for c in ordered_criteria]

        # Right plot: All at min, adjust the user-controlled criterion
        values_right = [c["min"] for c in ordered_criteria]

        if comparison_type == "best":
            # For best comparisons: Fix "other" at max on the left, adjust "best" on the right
            values_left[other_idx] = ordered_criteria[other_idx]["max"]
            slider_min = ordered_criteria[ref_idx]["min"]
            slider_max = ordered_criteria[ref_idx]["max"]
            slider_init = ordered_criteria[ref_idx]["min"]
            left_title = f"Having {other_criterion} at {values_left[other_idx]:.1f} and {ref_criterion} at {ordered_criteria[ref_idx]['min']:.1f}"
            right_title = f"Having {other_criterion} at {values_left[other_idx]:.1f} and {ref_criterion} at {slider_init:.1f}"
        else:  # worst
            # For worst comparisons: Fix "worst" at max on the left, adjust "other" on the right
            values_left[ref_idx] = ordered_criteria[ref_idx]["max"]
            slider_min = ordered_criteria[other_idx]["min"]
            slider_max = ordered_criteria[other_idx]["max"]
            slider_init = ordered_criteria[other_idx]["min"]
            left_title = f"Having {ref_criterion} at {values_left[ref_idx]:.1f} and {other_criterion} at {ordered_criteria[other_idx]['min']:.1f}"
            right_title = f"Having {ref_criterion} at {values_left[ref_idx]:.1f} and {other_criterion} at {slider_init:.1f}"

        # Normalize values
        normalized_left = [(v - c["min"]) / (c["max"] - c["min"]) for v, c in zip(values_left, ordered_criteria)]
        normalized_right = [(v - c["min"]) / (c["max"] - c["min"]) for v, c in zip(values_right, ordered_criteria)]

        # Plot left (fixed criterion)
        bars_left = ax_left.bar(range(len(ordered_criteria)), normalized_left, tick_label=[c["name"] for c in ordered_criteria])
        ax_left.set_ylim(0, 1)
        ax_left.set_title(left_title, fontsize=12, pad=20)

        # Plot right (adjustable criterion)
        bars_right = ax_right.bar(range(len(ordered_criteria)), normalized_right, tick_label=[c["name"] for c in ordered_criteria])
        ax_right.set_ylim(0, 1)
        ax_right.set_title(right_title, fontsize=12, pad=20)

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
        slider = Slider(ax=slider_ax, label='', valmin=slider_min, valmax=slider_max, valinit=slider_init)

        def update(val):
            if comparison_type == "best":
                values_right[ref_idx] = slider.val
                ax_right.set_title(f"Having {other_criterion} at {values_left[other_idx]:.1f} and {ref_criterion} at {slider.val:.1f}", fontsize=12, pad=20)
            else:  # worst
                values_right[other_idx] = slider.val
                ax_right.set_title(f"Having {ref_criterion} at {values_left[ref_idx]:.1f} and {other_criterion} at {slider.val:.1f}", fontsize=12, pad=20)
            normalized_right = [(v - c["min"]) / (c["max"] - c["min"]) for v, c in zip(values_right, ordered_criteria)]
            for i, bar in enumerate(bars_right):
                bar.set_height(normalized_right[i])
            fig.canvas.draw_idle()

        slider.label.set_visible(False)
        slider.valtext.set_visible(False)
        slider.poly.set_visible(False)

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
        # Find index of other_criterion in the original criteria list
        other_idx = next(i for i, c in enumerate(self.criteria) if c["name"] == other_criterion)
        # Store in the appropriate results array
        if self.comparisons[self.current_comparison][0] == "best":
            self.best_comparison_results[other_idx] = adjusted_value
        else:  # worst
            self.worst_comparison_results[other_idx] = adjusted_value
        self.current_comparison += 1
        self.show_next_comparison()

    def save_results_to_file(self):
        filename = "wbt_results.csv"
        with open(filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Criterion", "Best Comparison", "Worst Comparison"])
            for i, c in enumerate(self.reordered_criteria_names):
                best_val = self.best_comparison_results[i]
                worst_val = self.worst_comparison_results[i]
                writer.writerow([c, best_val, worst_val])

    def finalize_results(self):
        # Convert numpy floats to regular floats and round to 2 decimal places
        best_results = [round(float(x), 2) if not np.isnan(x) else np.nan for x in self.best_comparison_results]
        worst_results = [round(float(x), 2) if not np.isnan(x) else np.nan for x in self.worst_comparison_results]

        # Find the indices of the NaN values
        idx_nan_best = next((i for i, x in enumerate(best_results) if np.isnan(x)), None)
        idx_nan_worst = next((i for i, x in enumerate(worst_results) if np.isnan(x)), None)

        # Initialize reordered arrays
        reordered_best = [np.nan] * len(best_results)
        reordered_worst = [np.nan] * len(worst_results)
        reordered_criteria_names = [np.nan] * len(worst_results)

        # Place NaN in best_results at index 0
        reordered_best[0] = best_results[idx_nan_best]
        reordered_worst[0] = worst_results[idx_nan_best]
        reordered_criteria_names[0] = self.criteria[idx_nan_best]["name"]

        # Place NaN in worst_results at index -1
        reordered_best[-1] = best_results[idx_nan_worst]
        reordered_worst[-1] = worst_results[idx_nan_worst]
        reordered_criteria_names[-1] = self.criteria[idx_nan_worst]["name"]

        # Move all other values in pair (excluding the NaN positions)
        other_indices = [i for i in range(len(best_results)) if i != idx_nan_best and i != idx_nan_worst]
        for i, idx in enumerate(other_indices):
            reordered_criteria_names[i + 1] = self.criteria[idx]["name"]

        # Fill the middle positions (1 to -2) in both arrays
        for i, idx in enumerate(other_indices):
            reordered_best[i + 1] = best_results[idx]
            reordered_worst[i + 1] = worst_results[idx]

        # Update class variables with reordered results
        self.best_comparison_results = reordered_best
        self.worst_comparison_results = reordered_worst
        self.reordered_criteria_names = reordered_criteria_names

        # Print results
        print("Reordered Best Results:", self.best_comparison_results)
        print("Reordered Worst Results:", self.worst_comparison_results)

        # Prompt to save and exit
        messagebox.showinfo("Completed", "Elicitation completed. Results have been printed to the console.")

        # Optionally, save results to a file
        save_results = messagebox.askyesno("Save Results", "Do you want to save the results to a file?")
        if save_results:
            self.save_results_to_file()

        # Close the window
        self.root.quit()
        self.root.destroy()

