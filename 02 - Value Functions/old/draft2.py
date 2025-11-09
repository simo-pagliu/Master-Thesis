import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
from scipy.interpolate import make_interp_spline, interp1d
from scipy.optimize import curve_fit

class ElicitationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Value Function Elicitation")

        # Variables
        self.indicator_name = "attribute"
        self.indicator_range = [0, 57]
        self.points = []
        self.function_str = ""
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
        self.function_type_choice = None
        self.step = 0
        self.y_peak = None
        self.y_boundary = None

        # Parameters for interactive tuning
        self.params = {}
        self.sliders = {}

        # UI Layout
        self.left_frame = ttk.Frame(root, width=300, padding="10")
        self.left_frame.grid(row=0, column=0, sticky="nsew")

        self.right_frame = ttk.Frame(root, padding="10")
        self.right_frame.grid(row=0, column=1, sticky="nsew")

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().pack()

        # Initialize plot
        self.ax.set_xlim(self.indicator_range[0], self.indicator_range[1])
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.set_title(f"Value Function for {self.indicator_name}")
        self.ax.set_xlabel(self.indicator_name)
        self.ax.set_ylabel("Value")
        self.ax.grid(True)
        self.ax.set_xticks([self.indicator_range[0], (self.indicator_range[0] + self.indicator_range[1]) / 2, self.indicator_range[1]])
        self.ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        self.canvas.draw()

        # Start the process
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
                self.ask_indifference_025_075()
        elif self.step == 4:
            self.ask_fit_type()
        elif self.step == 5:
            self.tune_parameters()
        elif self.step == 6:
            self.finish()

    def ask_gaussian(self):
        ttk.Label(self.left_frame, text="Does the value present a maximum or minimum within its range?").pack(pady=5)
        ttk.Button(self.left_frame, text="Yes", command=lambda: self.set_gaussian(True)).pack(pady=5)
        ttk.Button(self.left_frame, text="No", command=lambda: self.set_gaussian(False)).pack(pady=5)

    def ask_direction(self):
        ttk.Label(self.left_frame, text="Is the value function concave or convex?").pack(pady=5)
        ttk.Button(self.left_frame, text="Concave", command=lambda: self.set_direction("concave")).pack(pady=5)
        ttk.Button(self.left_frame, text="Convex", command=lambda: self.set_direction("convex")).pack(pady=5)

    def ask_peak_boundary(self):
        ttk.Label(self.left_frame, text=f"Peak value (range: {self.indicator_range[0]} to {self.indicator_range[1]}):").pack(pady=5)
        self.peak_entry = ttk.Entry(self.left_frame)
        self.peak_entry.pack(pady=5)

        ttk.Label(self.left_frame, text="Right boundary value:").pack(pady=5)
        self.value_R_entry = ttk.Entry(self.left_frame)
        self.value_R_entry.pack(pady=5)

        ttk.Label(self.left_frame, text="Left boundary value:").pack(pady=5)
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

    def ask_indifference_025_075(self):
        ttk.Label(self.left_frame, text=f"Which value of {self.indicator_name} makes it indifferent to increase from {self.low_value} to X and from X to {self.x_05}?").pack(pady=5)
        self.x_025_entry = ttk.Entry(self.left_frame)
        self.x_025_entry.pack(pady=5)

        ttk.Label(self.left_frame, text=f"Which value of {self.indicator_name} makes it indifferent to increase from {self.x_05} to X and from X to {self.peak_value}?").pack(pady=5)
        self.x_075_entry = ttk.Entry(self.left_frame)
        self.x_075_entry.pack(pady=5)

        ttk.Button(self.left_frame, text="Next", command=self.save_indifference_025_075).pack(pady=5)
        ttk.Button(self.left_frame, text="Back", command=self.prev_step).pack(pady=5)

    def ask_fit_type(self):
        ttk.Label(self.left_frame, text="Choose a function to fit:").pack(pady=5)
        ttk.Button(self.left_frame, text="Linear", command=lambda: self.set_fit_type("linear")).pack(pady=5)
        ttk.Button(self.left_frame, text="Piecewise Linear", command=lambda: self.set_fit_type("piecewise_linear")).pack(pady=5)
        ttk.Button(self.left_frame, text="Polynomial", command=lambda: self.set_fit_type("polynomial")).pack(pady=5)
        ttk.Button(self.left_frame, text="Piecewise Polynomial", command=lambda: self.set_fit_type("piecewise_polynomial")).pack(pady=5)
        ttk.Button(self.left_frame, text="Spline", command=lambda: self.set_fit_type("spline")).pack(pady=5)
        ttk.Button(self.left_frame, text="Exponential", command=lambda: self.set_fit_type("exponential")).pack(pady=5)
        if not self.is_gaussian:
            ttk.Button(self.left_frame, text="Sigmoid", command=lambda: self.set_fit_type("sigmoid")).pack(pady=5)
        if self.is_gaussian:
            ttk.Button(self.left_frame, text="Gaussian", command=lambda: self.set_fit_type("gaussian")).pack(pady=5)
        ttk.Button(self.left_frame, text="Back", command=self.prev_step).pack(pady=5)

    def tune_parameters(self):
        ttk.Label(self.left_frame, text="Tune the function parameters:").pack(pady=5)

        if self.function_type_choice == "linear":
            ttk.Label(self.left_frame, text="No parameters to tune for linear interpolation.").pack(pady=5)
        elif self.function_type_choice == "piecewise_linear":
            ttk.Label(self.left_frame, text="No parameters to tune for piecewise linear interpolation.").pack(pady=5)
        elif self.function_type_choice == "polynomial":
            ttk.Label(self.left_frame, text="Polynomial degree (1-10):").pack(pady=5)
            self.sliders["degree"] = tk.Scale(self.left_frame, from_=1, to=10, orient="horizontal", command=self.update_plot)
            self.sliders["degree"].set(3)
            self.sliders["degree"].pack(pady=5)
        elif self.function_type_choice == "piecewise_polynomial":
            ttk.Label(self.left_frame, text="Polynomial degree for each segment (1-5):").pack(pady=5)
            self.sliders["degree"] = tk.Scale(self.left_frame, from_=1, to=5, orient="horizontal", command=self.update_plot)
            self.sliders["degree"].set(2)
            self.sliders["degree"].pack(pady=5)
        elif self.function_type_choice == "spline":
            ttk.Label(self.left_frame, text="Spline degree (1-5):").pack(pady=5)
            self.sliders["degree"] = tk.Scale(self.left_frame, from_=1, to=5, orient="horizontal", command=self.update_plot)
            self.sliders["degree"].set(3)
            self.sliders["degree"].pack(pady=5)

            ttk.Label(self.left_frame, text="Smoothing factor (0-1):").pack(pady=5)
            self.sliders["smoothing"] = tk.Scale(self.left_frame, from_=0, to=100, orient="horizontal", command=self.update_plot, resolution=1)
            self.sliders["smoothing"].set(0)
            self.sliders["smoothing"].pack(pady=5)
        elif self.function_type_choice == "exponential":
            ttk.Label(self.left_frame, text="Base (1.1-10):").pack(pady=5)
            self.sliders["base"] = tk.Scale(self.left_frame, from_=11, to=100, orient="horizontal", command=self.update_plot, resolution=1)
            self.sliders["base"].set(20)
            self.sliders["base"].pack(pady=5)

            ttk.Label(self.left_frame, text="Exponent scale (0.1-5):").pack(pady=5)
            self.sliders["scale"] = tk.Scale(self.left_frame, from_=1, to=50, orient="horizontal", command=self.update_plot, resolution=1)
            self.sliders["scale"].set(10)
            self.sliders["scale"].pack(pady=5)
        elif self.function_type_choice == "sigmoid":
            # Use x0 (midpoint) and k (steepness) sliders.
            # k is specified on a log scale by the user (slider range -3..3).
            # effective k used in the formula is sign(raw_k) * 10**abs(raw_k) / attribute_range
            ttk.Label(self.left_frame, text="x0 (midpoint):").pack(pady=5)
            self.sliders["x0"] = tk.Scale(
                self.left_frame,
                from_=self.low_value,
                to=self.peak_value,
                orient="horizontal",
                command=self.update_plot,
                resolution=0.1
            )
            self.sliders["x0"].set((self.low_value + self.peak_value) / 2)
            self.sliders["x0"].pack(pady=5)

            ttk.Label(self.left_frame, text="k (steepness, log scale):").pack(pady=5)
            # Slider value is raw_k in [-3,3]. Effective k = sign(raw_k)*10**abs(raw_k)/(peak-low)
            self.sliders["k"] = tk.Scale(
                self.left_frame,
                from_=-3.0,
                to=3.0,
                orient="horizontal",
                command=self.update_plot,
                resolution=0.01
            )
            self.sliders["k"].set(-1.0)
            self.sliders["k"].pack(pady=5)
        elif self.function_type_choice == "gaussian":
            ttk.Label(self.left_frame, text="Amplitude:").pack(pady=5)
            self.sliders["amplitude"] = tk.Scale(
                self.left_frame,
                from_=0.1,
                to=self.y_peak,
                orient="horizontal",
                command=self.update_plot,
                resolution=0.01
            )
            self.sliders["amplitude"].set(self.y_peak - self.y_boundary)
            self.sliders["amplitude"].pack(pady=5)

            ttk.Label(self.left_frame, text="Sigma:").pack(pady=5)
            self.sliders["sigma"] = tk.Scale(
                self.left_frame,
                from_=0.1,
                to=(self.peak_value - self.value_L) / 2,
                orient="horizontal",
                command=self.update_plot,
                resolution=0.1
            )
            self.sliders["sigma"].set((self.peak_value - self.value_L) / 3)
            self.sliders["sigma"].pack(pady=5)

        ttk.Button(self.left_frame, text="Next", command=self.next_step).pack(pady=5)
        ttk.Button(self.left_frame, text="Back", command=self.prev_step).pack(pady=5)

        # Initialize parameters
        self.update_plot()

    def finish(self):
        for widget in self.left_frame.winfo_children():
            widget.destroy()

        ttk.Label(self.left_frame, text="Elicitation complete!").pack(pady=5)
        ttk.Button(self.left_frame, text="Save Results", command=self.save_results).pack(pady=5)
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

    def set_fit_type(self, fit_type):
        self.function_type_choice = fit_type
        self.step += 1
        self.show_step()

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
            self.plot_points()
            self.show_step()
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
            self.plot_points()
            self.show_step()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers.")

    def next_step(self):
        self.step += 1
        self.show_step()

    def prev_step(self):
        if self.step > 0:
            self.step -= 1
            self.show_step()

    def plot_points(self):
        self.ax.clear()
        x_vals = [p[0] for p in self.points]
        y_vals = [p[1] for p in self.points]
        self.ax.plot(x_vals, y_vals, 'o', label="Points")
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

    def update_plot(self, _=None):
        x_vals = [p[0] for p in self.points]
        y_vals = [p[1] for p in self.points]
        x_test = np.linspace(self.indicator_range[0], self.indicator_range[1], 100)
        y_test = np.zeros_like(x_test)

        if self.function_type_choice == "linear":
            y_test = np.interp(x_test, x_vals, y_vals)
            self.function_str = f"lambda x: np.interp(x, {x_vals}, {y_vals})"
        elif self.function_type_choice == "piecewise_linear":
            y_test = np.interp(x_test, x_vals, y_vals)
            self.function_str = f"lambda x: np.interp(x, {x_vals}, {y_vals})"
        elif self.function_type_choice == "polynomial":
            degree = self.sliders["degree"].get()
            if self.is_gaussian:
                x_fit = [p[0] for p in self.points[1:6]]
                y_fit = [p[1] for p in self.points[1:6]]
            else:
                x_fit = [p[0] for p in self.points[1:6]]
                y_fit = [p[1] for p in self.points[1:6]]
            coefficients = np.polyfit(x_fit, y_fit, degree)
            y_test = np.polyval(coefficients, x_test)
            if self.is_gaussian:
                y_test[x_test <= self.value_L] = self.y_boundary
                y_test[x_test >= self.value_R] = self.y_boundary
                y_test = np.clip(y_test, self.y_boundary, self.y_peak)
                self.function_str = (
                    f"lambda x: {self.y_boundary} if x <= {self.value_L} else "
                    f"{self.y_boundary} if x >= {self.value_R} else "
                    f"np.clip({np.poly1d(coefficients)}(x), {self.y_boundary}, {self.y_peak})"
                )
            else:
                y_test[x_test <= self.low_value] = 0.0
                y_test[x_test >= self.peak_value] = 1.0
                y_test = np.clip(y_test, 0, 1)
                self.function_str = (
                    f"lambda x: 0.0 if x <= {self.low_value} else "
                    f"1.0 if x >= {self.peak_value} else "
                    f"np.clip({np.poly1d(coefficients)}(x), 0, 1)"
                )
        elif self.function_type_choice == "piecewise_polynomial":
            degree = self.sliders["degree"].get()
            y_test = np.interp(x_test, x_vals, y_vals)  # Default to linear if degree=1
            if degree > 1:
                y_test = np.zeros_like(x_test)
                for i in range(len(x_vals) - 1):
                    mask = (x_test >= x_vals[i]) & (x_test <= x_vals[i+1])
                    x_segment = x_test[mask]
                    if len(x_segment) > 0:
                        coeffs = np.polyfit([x_vals[i], x_vals[i+1]], [y_vals[i], y_vals[i+1]], degree-1)
                        y_test[mask] = np.polyval(coeffs, x_segment)
            if self.is_gaussian:
                y_test[x_test <= self.value_L] = self.y_boundary
                y_test[x_test >= self.value_R] = self.y_boundary
                y_test = np.clip(y_test, self.y_boundary, self.y_peak)
                self.function_str = (
                    f"lambda x: {self.y_boundary} if x <= {self.value_L} else "
                    f"{self.y_boundary} if x >= {self.value_R} else "
                    f"np.clip(piecewise_poly(x, {x_vals}, {y_vals}, degree={degree}), {self.y_boundary}, {self.y_peak})"
                )
            else:
                y_test[x_test <= self.low_value] = 0.0
                y_test[x_test >= self.peak_value] = 1.0
                y_test = np.clip(y_test, 0, 1)
                self.function_str = (
                    f"lambda x: 0.0 if x <= {self.low_value} else "
                    f"1.0 if x >= {self.peak_value} else "
                    f"np.clip(piecewise_poly(x, {x_vals}, {y_vals}, degree={degree}), 0, 1)"
                )
        elif self.function_type_choice == "spline":
            degree = self.sliders["degree"].get()
            smoothing = self.sliders["smoothing"].get() / 100.0
            x_fit = np.array(x_vals)
            y_fit = np.array(y_vals)
            spline = make_interp_spline(x_fit, y_fit, k=degree)
            y_test = spline(x_test)
            if self.is_gaussian:
                y_test[x_test <= self.value_L] = self.y_boundary
                y_test[x_test >= self.value_R] = self.y_boundary
                y_test = np.clip(y_test, self.y_boundary, self.y_peak)
                self.function_str = (
                    f"lambda x: {self.y_boundary} if x <= {self.value_L} else "
                    f"{self.y_boundary} if x >= {self.value_R} else "
                    f"np.clip(make_interp_spline({x_vals}, {y_vals}, k={degree})(x), {self.y_boundary}, {self.y_peak})"
                )
            else:
                y_test[x_test <= self.low_value] = 0.0
                y_test[x_test >= self.peak_value] = 1.0
                y_test = np.clip(y_test, 0, 1)
                self.function_str = (
                    f"lambda x: 0.0 if x <= {self.low_value} else "
                    f"1.0 if x >= {self.peak_value} else "
                    f"np.clip(make_interp_spline({x_vals}, {y_vals}, k={degree})(x), 0, 1)"
                )
        elif self.function_type_choice == "exponential":
            base = self.sliders["base"].get() / 10.0
            scale = self.sliders["scale"].get() / 10.0
            y_test = 1 - np.exp(-scale * (x_test - self.low_value) / (self.peak_value - self.low_value))
            y_test[x_test <= self.low_value] = 0.0
            y_test[x_test >= self.peak_value] = 1.0
            y_test = np.clip(y_test, 0, 1)
            self.function_str = (
                f"lambda x: 0.0 if x <= {self.low_value} else "
                f"1.0 if x >= {self.peak_value} else "
                f"np.clip(1 - np.exp(-{scale} * (x - {self.low_value}) / ({self.peak_value} - {self.low_value})), 0, 1)"
            )
        elif self.function_type_choice == "sigmoid":
            # read the user sliders: x0 and raw_k (log-scale)
            x0 = self.sliders["x0"].get()
            raw_k = self.sliders["k"].get()
            # interpret k on a log scale relative to the attribute range
            attr_range = max(1e-6, (self.peak_value - self.low_value))
            k = np.sign(raw_k) * (10 ** (abs(raw_k))) / attr_range
            y_test = 1 / (1 + np.exp(-k * (x_test - x0)))
            # enforce piecewise constant tails and clipping
            y_test[x_test <= self.low_value] = 0.0
            y_test[x_test >= self.peak_value] = 1.0
            self.function_str = (
                f"lambda x: 0.0 if x <= {self.low_value} else "
                f"1.0 if x >= {self.peak_value} else "
                f"1 / (1 + np.exp(-{k} * (x - {x0})))"
            )
        elif self.function_type_choice == "gaussian":
            amplitude = self.sliders["amplitude"].get()
            sigma = self.sliders["sigma"].get()
            y_test = self.y_boundary + amplitude * np.exp(-((x_test - self.peak_value) ** 2) / (2 * sigma ** 2))
            y_test[x_test <= self.value_L] = self.y_boundary
            y_test[x_test >= self.value_R] = self.y_boundary
            y_test = np.clip(y_test, self.y_boundary, self.y_peak)
            self.function_str = (
                f"lambda x: {self.y_boundary} if x <= {self.value_L} else "
                f"{self.y_boundary} if x >= {self.value_R} else "
                f"np.clip({self.y_boundary} + {amplitude} * np.exp(-((x - {self.peak_value}) ** 2) / (2 * {sigma} ** 2)), {self.y_boundary}, {self.y_peak})"
            )

        self.ax.clear()
        self.ax.plot(x_vals, y_vals, 'o', label="Points")
        self.ax.plot(x_test, y_test, '-', label=f"{self.function_type_choice.capitalize()} Fit")
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

    def save_results(self):
        # Save to CSV
        data = {
            "Attribute": [self.indicator_name],
            "Range": [self.indicator_range],
            "Points": [self.points],
            "Function": [self.function_str]
        }
        df = pd.DataFrame(data)
        csv_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if csv_path:
            df.to_csv(csv_path, index=False)

        # Save to text file
        txt_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if txt_path:
            with open(txt_path, "w") as f:
                f.write(f"Attribute: {self.indicator_name}\n")
                f.write(f"Range: {self.indicator_range}\n")
                f.write(f"Points: {self.points}\n")
                f.write(f"Function: {self.function_str}\n")

        messagebox.showinfo("Success", "Results saved successfully!")

if __name__ == "__main__":
    root = tk.Tk()
    app = ElicitationApp(root)
    root.mainloop()
