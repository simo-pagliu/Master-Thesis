#################################################################################
# Simple plotting script for UP-MAVT Monte Carlo results
#
# Reads the saved MC output CSV and creates a couple of quick summary plots.
#################################################################################

import os
import re

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('TkAgg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------- #
# USER INPUT
# Path is relative to this script's folder unless you use an absolute path.
RESULTS_FILE = "./results/results_1.csv"
PLOT_BINS = 50
OVERLAY_BY_ELICITATION = True
PLOT_HEATMAP = True

# Paper-friendly layout controls for the histogram grid
HIST_NCOLS = 2  # number of columns in the histogram grid
HIST_FIGSIZE = (11.69, 8.27)  # A4 landscape in inches
# ---------------------------------------------------------------------------- #


def _clean_column_name(name: str) -> str:
    # Some CSV headers may contain stray whitespace/newlines.
    # Example observed: "Alter\nnative_5".
    # For technical columns we remove all whitespace; for named alternatives
    # we preserve spaces and only normalize whitespace.
    s = str(name)
    if s.strip() in {"Run", "Elicitation"} or s.strip().startswith("Alternative_"):
        return re.sub(r"\s+", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _plot_histograms(df: pd.DataFrame, alt_cols: list[str]) -> None:
    n_alts = len(alt_cols)
    ncols = max(1, min(HIST_NCOLS, n_alts))
    nrows = int(np.ceil(n_alts / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=HIST_FIGSIZE, sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)

    legend_handles = None
    legend_labels = None

    if OVERLAY_BY_ELICITATION and "Elicitation" in df.columns:
        elicitation_ids = sorted(pd.unique(df["Elicitation"]))
        cmap = plt.get_cmap("tab10")
        colors = list(cmap(np.linspace(0, 1, max(1, len(elicitation_ids)))))

        for alt_idx, col in enumerate(alt_cols):
            ax = axes[alt_idx]
            any_data = False
            for e_idx, el_id in enumerate(elicitation_ids):
                subset = df.loc[df["Elicitation"] == el_id, col].to_numpy(dtype=float)
                if subset.size == 0:
                    continue
                any_data = True
                weights = np.ones_like(subset) / subset.size
                ax.hist(
                    subset,
                    bins=PLOT_BINS,
                    weights=weights,
                    alpha=0.5,
                    color=colors[e_idx],
                    label=f"E{int(el_id)}" if float(el_id).is_integer() else f"E{el_id}",
                )

            if not any_data:
                ax.hist([], bins=PLOT_BINS, alpha=0.7)
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

            ax.set_title(col)
            ax.set_xlim(0, 1)
    else:
        for alt_idx, col in enumerate(alt_cols):
            ax = axes[alt_idx]
            data = df[col].to_numpy(dtype=float)
            weights = np.ones_like(data) / data.size
            ax.hist(data, bins=PLOT_BINS, weights=weights, alpha=0.7)
            ax.set_title(col)
            ax.set_xlim(0, 1)

    # Turn off unused axes
    for i in range(n_alts, len(axes)):
        axes[i].axis("off")

    # Shared labels (paper-friendly)
    # fig.suptitle("Alternative value histograms (MC)")
    fig.supxlabel("value")
    fig.supylabel("Probability")

    if legend_handles and legend_labels:
        # Put the legend at the top so it doesn't overlap the shared x-label.
        # Also keep it compact/readable for papers.
        fig.legend(
            legend_handles,
            legend_labels,
            title="Elicitation",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=min(len(legend_labels), 4),
            frameon=True,
            fontsize=9,
            title_fontsize=10,
            columnspacing=1.0,
            handlelength=1.4,
            borderaxespad=0.2,
        )
        # Reserve space at the top for the legend and at the bottom for xlabel.
        fig.tight_layout(rect=(0, 0.08, 1, 0.90))
    else:
        # Leave room at the bottom for xlabel
        fig.tight_layout(rect=(0, 0.08, 1, 0.95))


def _rank_probability_matrix(values: np.ndarray) -> np.ndarray:
    """Compute P(alt is rank r) for r=1..N, higher value is better."""
    n_runs, n_alts = values.shape
    order = np.argsort(-values, axis=1)  # descending

    counts = np.zeros((n_alts, n_alts), dtype=float)  # (rank, alt)
    for run_idx in range(n_runs):
        for rank_idx, alt_idx in enumerate(order[run_idx]):
            counts[rank_idx, alt_idx] += 1.0

    return counts / float(n_runs)


def _plot_heatmap(values: np.ndarray, alt_cols: list[str]) -> None:
    rank_probs = _rank_probability_matrix(values)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        rank_probs,
        annot=True,
        fmt=".2f",
        xticklabels=alt_cols,
        yticklabels=[f"{i+1}th" for i in range(len(alt_cols))],
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        cbar=False,
        ax=ax,
    )
    ax.set_title("Ranking probabilities (MC)")
    ax.set_xlabel("Alternative")
    ax.set_ylabel("Rank")
    fig.tight_layout()


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    df = pd.read_csv(RESULTS_FILE)
    df.columns = [_clean_column_name(c) for c in df.columns]

    # Identify alternative columns (default output format: Alternative_1..Alternative_N)
    excluded = {"Run", "Elicitation"}
    alt_cols = [c for c in df.columns if c not in excluded]

    # Keep only numeric alternative columns
    for c in alt_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    alt_cols = [c for c in alt_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not alt_cols:
        raise ValueError(f"No numeric alternative columns found in {RESULTS_FILE}. Columns: {list(df.columns)}")

    values = df[alt_cols].to_numpy(dtype=float)
    if np.isnan(values).any():
        # Fail-fast but with a clear reason
        raise ValueError("Found NaNs in alternative values after parsing; check the CSV formatting.")

    _plot_histograms(df, alt_cols)
    if PLOT_HEATMAP:
        _plot_heatmap(values, alt_cols)

    plt.show()


if __name__ == "__main__":
    main()
