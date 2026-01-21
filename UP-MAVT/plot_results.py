#################################################################################
# Plot saved MC results using the shared plotting helpers
#################################################################################

import os
import re

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('TkAgg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt

from plotting import create_heatmap, create_histograms, update_plots, finalize_layout


# ---------------------------------------------------------------------------- #
# USER INPUT
RESULTS_FILE = "./results/results.csv"  # will auto-pick highest suffix if exists
PLOT_BINS = 50
# ---------------------------------------------------------------------------- #


def _clean_column_name(name: str) -> str:
    s = str(name)
    if s.strip() in {"Run", "Elicitation"} or s.strip().startswith("Alternative_"):
        return re.sub(r"\s+", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _latest_results(path: str) -> str:
    base, ext = os.path.splitext(path)
    if os.path.exists(path):
        return path
    idx = 1
    candidate = f"{base}_{idx}{ext}"
    last_existing = None
    while os.path.exists(candidate):
        last_existing = candidate
        idx += 1
        candidate = f"{base}_{idx}{ext}"
    if last_existing:
        return last_existing
    raise FileNotFoundError(f"No results file found starting at {path}")


def _rank_probability_matrix(values: np.ndarray) -> np.ndarray:
    n_runs, n_alts = values.shape
    order = np.argsort(-values, axis=1)
    counts = np.zeros((n_alts, n_alts), dtype=float)
    for run_idx in range(n_runs):
        for rank_idx, alt_idx in enumerate(order[run_idx]):
            counts[rank_idx, alt_idx] += 1.0
    return counts / float(n_runs)


def _group_runs(values: np.ndarray, elicitation_col: np.ndarray | None, n_alts: int):
    if elicitation_col is None:
        return [values.tolist()], 1
    elic_ids = sorted(pd.unique(elicitation_col))
    grouped = []
    for el_id in elic_ids:
        grouped.append(values[elicitation_col == el_id].tolist())
    return grouped, len(grouped)


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    results_path = _latest_results(RESULTS_FILE)
    df = pd.read_csv(results_path)
    df.columns = [_clean_column_name(c) for c in df.columns]

    excluded = {"Run", "Elicitation"}
    alt_cols = [c for c in df.columns if c not in excluded]
    for c in alt_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    alt_cols = [c for c in alt_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not alt_cols:
        raise ValueError(f"No numeric alternative columns found in {results_path}. Columns: {list(df.columns)}")

    values = df[alt_cols].to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError("Found NaNs in alternative values after parsing; check the CSV formatting.")

    n_runs, n_alts = values.shape
    rank_probs = _rank_probability_matrix(values).T  # match live plot shape (alt x rank)^T in helper
    distributions = values.T

    # Heatmap (non-strict)
    fig_hm, ax_hm = create_heatmap(alt_cols, n_alts, strict=False)
    update_plots(rank_probs, distributions, n_runs - 1, n_runs, n_alts, alt_cols, strict=False, fig=fig_hm, ax=ax_hm, plot_bins=PLOT_BINS)

    # Histograms (reuse strict branch to get per-elicitation overlays if present)
    elic_col = df["Elicitation"].to_numpy() if "Elicitation" in df.columns else None
    lists_of_full_sets, n_elicitations = _group_runs(values, elic_col, n_alts)
    fig_hist, axes = create_histograms(n_alts, n_elicitations, strict=True, plot_bins=PLOT_BINS)
    update_plots(
        rank_probs,
        distributions,
        n_runs - 1,
        n_runs,
        n_alts,
        alt_cols,
        strict=True,
        n_elicitations=n_elicitations,
        lists_of_full_sets=lists_of_full_sets,
        fig_hist=fig_hist,
        axes=axes,
        plot_bins=PLOT_BINS,
    )

    finalize_layout(fig_hm, fig_hist)
    plt.show()


if __name__ == "__main__":
    main()
