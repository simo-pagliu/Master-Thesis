"""plotter.py

Contains a small Plotter class that encapsulates matplotlib plotting
so the UI can delegate plotting responsibilities.

The Plotter API is intentionally simple:
  plot(ax, points, x_fit, y_fit, xmin, xmax, **opts)

This keeps plotting logic separate from PyQt UI code and makes it
easier to test or reuse the plotting behavior elsewhere.
"""
from typing import List, Optional, Sequence, Tuple
import matplotlib.axes


class Plotter:
    def __init__(self):
        pass

    def plot(
        self,
        ax: matplotlib.axes.Axes,
        points: Sequence[Tuple[float, float]],
        x_fit: Optional[Sequence[float]],
        y_fit: Optional[Sequence[float]],
        xmin: Optional[float],
        xmax: Optional[float],
        lower_threshold: Optional[float] = None,
        upper_threshold: Optional[float] = None,
        left_tail_value: Optional[float] = None,
        right_tail_value: Optional[float] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
    ) -> matplotlib.axes.Axes:
        """Draw points and optional fitted curve onto the given Axes.

        Parameters
        - ax: Axes to draw on (will be cleared)
        - points: sequence of (x, y) tuples
        - x_fit, y_fit: optional fit arrays to plot as a line
        - xmin, xmax: optional axis limits (if None, autoscale)

        Returns the Axes for chaining.
        """
        ax.clear()

        # Plot points
        if points:
            x_points = [p[0] for p in points]
            y_points = [p[1] for p in points]
            ax.scatter(x_points, y_points, color="red", label="Points")

        # Plot fit
        if x_fit is not None and y_fit is not None:
            ax.plot(x_fit, y_fit, label="Fit")
        else:
            # fallback: if there is no fit but we have points, draw piecewise linear interpolation
            if points:
                # sort points by x
                pts = sorted(points, key=lambda t: t[0])
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, color='blue', linestyle='-', label='Linear')

        # Draw tail constants if thresholds are set and axis range provided
        try:
            if xmin is not None and xmax is not None:
                xmin_f = float(xmin)
                xmax_f = float(xmax)
                if xmin_f == xmax_f:
                    xmin_f -= 0.5
                    xmax_f += 0.5
                # left tail
                if lower_threshold is not None and left_tail_value is not None:
                    lt = float(min(lower_threshold, upper_threshold)) if upper_threshold is not None else float(lower_threshold)
                    if lt > xmin_f:
                        ax.hlines(left_tail_value, xmin_f, lt, colors='gray', linestyles='dashed', label='_nolegend_')
                # right tail
                if upper_threshold is not None and right_tail_value is not None:
                    rt = float(max(upper_threshold, lower_threshold)) if lower_threshold is not None else float(upper_threshold)
                    if rt < xmax_f:
                        ax.hlines(right_tail_value, rt, xmax_f, colors='gray', linestyles='dashed', label='_nolegend_')
                ax.set_xlim(xmin_f, xmax_f)
        except Exception:
            # ignore invalid limits
            pass

        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        ax.legend()
        ax.grid(True)
        return ax
