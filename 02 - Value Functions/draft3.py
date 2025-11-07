import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ElicitationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Attribute Elicitation")

        # Variables
        self.attributes = []
        self.current_attribute = None
        self.current_index = 0
        self.points = []
        self.indicator_name = ""
        self.indicator_range = [0, 1]
        self.is_gaussian = None
        self.direction = None
        self.peak_value = None
        self.value_R = None
        self.value_L = None
        self.x_05_R = None
        self.x_05_L = None
        self.low_value = None
        self.x_05 = None
        self.x_025 = None
        self.x_075 = None
        self.y_peak = None
        self.y_boundary = None
        self.step = 0
        self.reviewing = False
        self.skip_025_075 = False

        # UI Layout
        self.left_frame = ttk.Frame(root, width=300, padding="10")
        self.left_frame.grid(row=0, column=0, sticky="nsew")

        self.right_frame = ttk.Frame(root, padding="10")
        self.right_frame.grid(row=0, column=1, sticky="nsew")

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().pack()

        # Start by loading the CSV
        self.load_csv()

    def load_csv(self):
        csv_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not csv_path:
            messagebox.showerror("Error", "No file selected.")
            return

        try:
            self.df = pd.read_csv(csv_path)
            required_columns = ["name", "group", "min", "max", "unit"]
            self.df.columns = self.df.columns.str.strip()
            if not all(col in self.df.columns for col in required_columns):
                messagebox.showerror("Error", f"CSV must contain columns: {', '.join(required_columns)}")
                return

            # Ensure min/max are numeric to avoid type errors later when plotting
            self.df['min'] = pd.to_numeric(self.df['min'], errors='coerce')
            self.df['max'] = pd.to_numeric(self.df['max'], errors='coerce')
            if self.df['min'].isna().any() or self.df['max'].isna().any():
                messagebox.showerror("Error", "CSV 'min' and 'max' columns must contain numeric values.")
                return

            self.attributes = self.df.to_dict("records")
            self.current_index = 0
            self.show_next_attribute()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV: {e}")

    def show_next_attribute(self):
        print(f"show_next_attribute: current_index={self.current_index}, total={len(self.attributes)}")
        if self.current_index >= len(self.attributes):
            print("show_next_attribute: reached end of attributes, saving results")
            self.save_results()
            return

        try:
            self.reviewing = False
            self.current_attribute = self.attributes[self.current_index]
            self.indicator_name = self.current_attribute["name"]
            self.indicator_range = [self.current_attribute["min"], self.current_attribute["max"]]
            self.points = []
            self.step = 0
            self.skip_025_075 = False

            # Clear frames
            for widget in self.left_frame.winfo_children():
                widget.destroy()

            # Show attribute name and "Next" button
            ttk.Label(self.left_frame, text=f"Next attribute: {self.indicator_name}").pack(pady=5)
            ttk.Button(self.left_frame, text="Next", command=self.start_elicitation).pack(pady=5)

            # Initialize plot
            self.ax.clear()
            self.ax.set_xlim(self.indicator_range[0], self.indicator_range[1])
            self.ax.set_ylim(-0.1, 1.1)
            self.ax.set_title(f"Value Function for {self.indicator_name}")
            self.ax.set_xlabel(self.indicator_name)
            self.ax.set_ylabel("Value")
            self.ax.grid(True)
            self.ax.set_xticks([self.indicator_range[0], (self.indicator_range[0] + self.indicator_range[1]) / 2, self.indicator_range[1]])
            self.ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            self.canvas.draw()
        except Exception as e:
            import traceback
            traceback.print_exc()
            # If something goes wrong while setting up the next attribute, show an error and allow user to continue
            for widget in self.left_frame.winfo_children():
                widget.destroy()
            ttk.Label(self.left_frame, text=f"Error loading attribute at index {self.current_index}: {e}").pack(pady=5)
            ttk.Button(self.left_frame, text="Skip to next", command=lambda: [setattr(self, 'current_index', self.current_index + 1), self.show_next_attribute()]).pack(pady=5)
            ttk.Button(self.left_frame, text="Quit", command=self.root.quit).pack(pady=5)

    def start_elicitation(self):
        self.step = 0
        self.show_step()

    def show_step(self):
        # Clear the left frame
        for widget in self.left_frame.winfo_children():
            widget.destroy()

        if self.step == 0:
            self.ask_gaussian()
        elif self.step == 1:
            if self.is_gaussian:
                self.ask_direction()
            else:
                self.ask_peak_and_low()
        elif self.step == 2:
            if self.is_gaussian:
                self.ask_peak_boundary()
            else:
                self.ask_indifference_05()
        elif self.step == 3:
            if self.is_gaussian:
                self.ask_indifference_gaussian()
            else:
                self.ask_skip_025_075()
        elif self.step == 4:
            if not self.is_gaussian and not self.skip_025_075:
                self.ask_indifference_025_075()
            else:
                self.show_points_before_fit()
        elif self.step == 5:
            self.review_points()

    def ask_gaussian(self):
        ttk.Label(self.left_frame, text="Is the most or least important value inside the data range and not at the extremes?").pack(pady=5)
        ttk.Button(self.left_frame, text="Yes", command=lambda: self.set_gaussian(True)).pack(pady=5)
        ttk.Button(self.left_frame, text="No", command=lambda: self.set_gaussian(False)).pack(pady=5)

    def ask_direction(self):
        ttk.Label(self.left_frame, text="Is the value function concave or convex?").pack(pady=5)
        ttk.Button(self.left_frame, text="Concave (highest point in the middle)", command=lambda: self.set_direction("concave")).pack(pady=5)
        ttk.Button(self.left_frame, text="Convex (lowest point in the middle)", command=lambda: self.set_direction("convex")).pack(pady=5)

    def ask_peak_boundary(self):
        ttk.Label(self.left_frame, text=f"Most important value in the range from {self.indicator_range[0]} to {self.indicator_range[1]}):").pack(pady=5)
        self.peak_entry = ttk.Entry(self.left_frame)
        self.peak_entry.pack(pady=5)
        ttk.Label(self.left_frame, text="After which value does the importance stagnate?").pack(pady=5)
        self.value_R_entry = ttk.Entry(self.left_frame)
        self.value_R_entry.pack(pady=5)
        ttk.Label(self.left_frame, text="Below which value does the importance stagnate?").pack(pady=5)
        self.value_L_entry = ttk.Entry(self.left_frame)
        self.value_L_entry.pack(pady=5)
        ttk.Button(self.left_frame, text="Next", command=self.save_peak_boundary).pack(pady=5)
        ttk.Button(self.left_frame, text="Back", command=self.prev_step).pack(pady=5)

    def ask_indifference_gaussian(self):
        ttk.Label(self.left_frame, text=f"Indifference point (right, between {self.value_R} and {self.peak_value}):").pack(pady=5)
        self.x_05_R_entry = ttk.Entry(self.left_frame)
        self.x_05_R_entry.pack(pady=5)
        ttk.Label(self.left_frame, text=f"Indifference point (left, between {self.value_L} and {self.peak_value}):").pack(pady=5)
        self.x_05_L_entry = ttk.Entry(self.left_frame)
        self.x_05_L_entry.pack(pady=5)
        ttk.Button(self.left_frame, text="Next", command=self.save_indifference_gaussian).pack(pady=5)
        ttk.Button(self.left_frame, text="Back", command=self.prev_step).pack(pady=5)

    def ask_peak_and_low(self):
        ttk.Label(self.left_frame, text="Peak value:").pack(pady=5)
        self.peak_entry = ttk.Entry(self.left_frame)
        self.peak_entry.pack(pady=5)
        ttk.Label(self.left_frame, text="Lowest value:").pack(pady=5)
        self.low_entry = ttk.Entry(self.left_frame)
        self.low_entry.pack(pady=5)
        ttk.Button(self.left_frame, text="Next", command=self.save_peak_low).pack(pady=5)
        ttk.Button(self.left_frame, text="Back", command=self.prev_step).pack(pady=5)

    def ask_indifference_05(self):
        ttk.Label(self.left_frame, text=f"Which value of {self.indicator_name} makes it indifferent to increase from {self.low_value} to X and from X to {self.peak_value}?").pack(pady=5)
        self.x_05_entry = ttk.Entry(self.left_frame)
        self.x_05_entry.pack(pady=5)
        ttk.Button(self.left_frame, text="Next", command=self.save_indifference_05).pack(pady=5)
        ttk.Button(self.left_frame, text="Back", command=self.prev_step).pack(pady=5)

    def ask_skip_025_075(self):
        ttk.Label(self.left_frame, text="Do you want to add 0.25 and 0.75 indifference points?").pack(pady=5)
        ttk.Button(self.left_frame, text="Yes", command=lambda: self.set_skip_025_075(False)).pack(pady=5)
        ttk.Button(self.left_frame, text="No", command=lambda: self.set_skip_025_075(True)).pack(pady=5)
        ttk.Button(self.left_frame, text="Back", command=self.prev_step).pack(pady=5)

    def ask_indifference_025_075(self):
        ttk.Label(self.left_frame, text=f"Which value of {self.indicator_name} makes it indifferent to increase from {self.low_value} to X and from X to {self.x_05}?").pack(pady=5)
        self.x_025_entry = ttk.Entry(self.left_frame)
        self.x_025_entry.pack(pady=5)
        ttk.Label(self.left_frame, text=f"Which value of {self.indicator_name} makes it indifferent to increase from {self.x_05} to X and from X to {self.peak_value}?").pack(pady=5)
        self.x_075_entry = ttk.Entry(self.left_frame)
        self.x_075_entry.pack(pady=5)
        ttk.Button(self.left_frame, text="Next", command=self.save_indifference_025_075).pack(pady=5)
        ttk.Button(self.left_frame, text="Back", command=self.prev_step).pack(pady=5)

    def set_gaussian(self, is_gaussian):
        self.is_gaussian = is_gaussian
        self.step += 1
        self.show_step()

    def set_direction(self, direction):
        self.direction = direction
        self.y_peak = 1.0 if direction == "concave" else 0.0
        self.y_boundary = 0.0 if direction == "concave" else 1.0
        self.step += 1
        self.show_step()

    def set_skip_025_075(self, skip):
        self.skip_025_075 = skip
        if not skip:
            self.step += 1
            self.show_step()
        else:
            self.step += 1
            self.show_points_before_fit()

    def save_peak_boundary(self):
        try:
            self.peak_value = float(self.peak_entry.get())
            self.value_R = float(self.value_R_entry.get())
            self.value_L = float(self.value_L_entry.get())
            self.points = [
                (self.indicator_range[0], self.y_boundary),
                (self.value_L, self.y_boundary),
                (self.peak_value, self.y_peak),
                (self.value_R, self.y_boundary),
                (self.indicator_range[1], self.y_boundary)
            ]
            self.step += 1
            self.plot_points()
            self.show_step()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers.")

    def save_indifference_gaussian(self):
        try:
            self.x_05_R = float(self.x_05_R_entry.get())
            self.x_05_L = float(self.x_05_L_entry.get())
            self.points = [
                (self.indicator_range[0], self.y_boundary),
                (self.value_L, self.y_boundary),
                (self.x_05_L, 0.5),
                (self.peak_value, self.y_peak),
                (self.x_05_R, 0.5),
                (self.value_R, self.y_boundary),
                (self.indicator_range[1], self.y_boundary)
            ]
            self.step += 1
            self.show_points_before_fit()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers.")

    def save_peak_low(self):
        try:
            self.peak_value = float(self.peak_entry.get())
            self.low_value = float(self.low_entry.get())
            self.points = [
                (self.indicator_range[0], 0.0),
                (self.low_value, 0.0),
                (self.peak_value, 1.0),
                (self.indicator_range[1], 1.0)
            ]
            self.step += 1
            self.plot_points()
            self.show_step()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers.")

    def save_indifference_05(self):
        try:
            self.x_05 = float(self.x_05_entry.get())
            self.points = [
                (self.indicator_range[0], 0.0),
                (self.low_value, 0.0),
                (self.x_05, 0.5),
                (self.peak_value, 1.0),
                (self.indicator_range[1], 1.0)
            ]
            # move to the next step and let show_step() render the next screen
            # also plot the newly added 0.5 point so the user sees it immediately
            self.step += 1
            self.plot_points()
            self.show_step()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers.")

    def save_indifference_025_075(self):
        try:
            self.x_025 = float(self.x_025_entry.get())
            self.x_075 = float(self.x_075_entry.get())
            self.points = [
                (self.indicator_range[0], 0.0),
                (self.low_value, 0.0),
                (self.x_025, 0.25),
                (self.x_05, 0.5),
                (self.x_075, 0.75),
                (self.peak_value, 1.0),
                (self.indicator_range[1], 1.0)
            ]
            self.step += 1
            self.show_points_before_fit()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers.")

    def prev_step(self):
        if self.step > 0:
            self.step -= 1
            self.show_step()

    def plot_points(self):
        x_vals = [p[0] for p in self.points]
        y_vals = [p[1] for p in self.points]
        self.ax.clear()
        self.ax.plot(x_vals, y_vals, 'o-', label="Points")
        self.ax.set_title(f"Value Function for {self.indicator_name}")
        self.ax.set_xlabel(self.indicator_name)
        self.ax.set_ylabel("Value")
        self.ax.set_xlim(self.indicator_range[0], self.indicator_range[1])
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.grid(True)
        self.ax.set_xticks(x_vals)
        self.ax.set_xticklabels([f"{x:.1f}" for x in x_vals])
        self.ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        self.ax.legend()
        self.canvas.draw()

    def increment_step_and_show(self):
        self.step += 1
        self.show_step()

    def show_points_before_fit(self):
        for widget in self.left_frame.winfo_children():
            widget.destroy()

        for i, (x, y) in enumerate(self.points):
            ttk.Label(self.left_frame, text=f"Point {i+1}: ({x:.2f}, {y:.2f})").pack(pady=2)
        ttk.Button(self.left_frame, text="Next", command=self.increment_step_and_show).pack(pady=5)

        ttk.Button(self.left_frame, text="Back", command=self.prev_step).pack(pady=5)

        self.plot_points()

    def review_points(self):
        self.reviewing = True
        for widget in self.left_frame.winfo_children():
            widget.destroy()

        ttk.Label(self.left_frame, text=f"Review points for {self.indicator_name}:").pack(pady=5)
        for i, (x, y) in enumerate(self.points):
            ttk.Label(self.left_frame, text=f"Point {i+1}: ({x:.2f}, {y:.2f})").pack(pady=2)
        ttk.Button(self.left_frame, text="Back", command=self.back_to_edit).pack(pady=5)

        if self.current_index < len(self.attributes) - 1:
            ttk.Button(self.left_frame, text="Next", command=self.confirm_and_next).pack(pady=5)
        else:
            ttk.Button(self.left_frame, text="Save", command=self.save_results).pack(pady=5)

    def back_to_edit(self):
        self.reviewing = False
        self.step = 0
        self.show_step()

    def confirm_and_next(self):
        print(f"confirm_and_next: saving points for index {self.current_index}")
        self.attributes[self.current_index]["points"] = self.points
        self.current_index += 1
        print(f"confirm_and_next: incremented current_index to {self.current_index}")
        self.show_next_attribute()

    def save_results(self):
        print(f"save_results: current_index={self.current_index}, total={len(self.attributes)}")
        # only save points for the current_index if it's within range
        if 0 <= self.current_index < len(self.attributes):
            self.attributes[self.current_index]["points"] = self.points
        else:
            print("save_results: current_index out of range, skipping per-attribute save")

        for attr in self.attributes:
            self.df.loc[self.df["name"] == attr["name"], "points"] = str(attr["points"])

        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if save_path:
            self.df.to_csv(save_path, index=False)
            messagebox.showinfo("Success", "Results saved successfully!")
            self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = ElicitationApp(root)
    root.mainloop()
