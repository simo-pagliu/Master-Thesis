import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def create_heatmap(alternative_names, n_alternatives, strict):
    """Initialize heatmap figure/axes for aggregated rankings."""
    if strict:
        return None, None
    fig, ax = plt.subplots(figsize=(10, 8))
    initial_data = np.zeros((n_alternatives, n_alternatives))
    sns.heatmap(
        initial_data.T,
        cmap=sns.color_palette("Blues", as_cmap=True),
        ax=ax,
        xticklabels=alternative_names,
        yticklabels=[f"{j+1}th" for j in range(n_alternatives)],
        vmin=0,
        vmax=1,
        square=True,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
    try:
        fig.tight_layout()
        fig.subplots_adjust(top=0.95, bottom=0.08)
    except Exception:
        pass
    return fig, ax


def create_histograms(n_alternatives, n_elicitations, strict, plot_bins):
    """Initialize histogram figure/axes for strict mode distributions."""
    if not strict:
        return None, None
    fig_hist, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.reshape(-1)
    try:
        fig_hist.tight_layout()
        fig_hist.subplots_adjust(
            hspace=0.40,
            wspace=0.30,
            top=0.95,
            bottom=0.05,
            left=0.08,
            right=0.95,
        )
    except Exception:
        pass
    return fig_hist, axes


def update_plots(
    rank_probs,
    distributions,
    run_index,
    n_runs,
    n_alternatives,
    alternative_names,
    strict=False,
    n_elicitations=1,
    lists_of_full_sets=None,
    rank_counts_per_el=None,
    fig=None,
    ax=None,
    fig_hist=None,
    axes=None,
    plot_bins=50,
):
    """Update live plots for heatmap and histograms."""
    if not strict and ax is not None:
        ax.clear()
        sns.heatmap(
            rank_probs.T,
            annot=True,
            fmt=".2f",
            xticklabels=alternative_names,
            yticklabels=[f"{j+1}th" for j in range(n_alternatives)],
            cmap=sns.color_palette("Blues", as_cmap=True),
            ax=ax,
            vmin=0,
            vmax=1,
            cbar=False,
            square=True,
        )
        ax.set_title(f"Ranking Probabilities (Run {run_index+1}/{n_runs})")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")

    if strict and axes is not None:
        for alt_idx in range(n_alternatives):
            axes[alt_idx].clear()
            if lists_of_full_sets is not None:
                cmap = plt.get_cmap("tab10")
                color_list = list(cmap(np.linspace(0, 1, max(1, n_elicitations))))
                any_data = False
                for e in range(n_elicitations):
                    runs_e = len(lists_of_full_sets[e])
                    if runs_e == 0:
                        continue
                    data_e = np.array(lists_of_full_sets[e]).T[alt_idx]
                    if data_e.size > 0:
                        any_data = True
                        weights = np.ones_like(data_e) / data_e.size
                        axes[alt_idx].hist(
                            data_e,
                            bins=plot_bins,
                            weights=weights,
                            alpha=0.5,
                            color=color_list[e],
                            label=f"E{e+1}",
                        )
                if not any_data:
                    axes[alt_idx].hist([], bins=plot_bins, alpha=0.7)
                else:
                    axes[alt_idx].legend(title="Elicitation")
                axes[alt_idx].set_xlim(0, 1)
            axes[alt_idx].set_title(f"Distribution of Values for {alternative_names[alt_idx]}")
            axes[alt_idx].set_xlabel("Value")
            axes[alt_idx].set_ylabel("Probability")
            axes[alt_idx].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
            axes[alt_idx].set_ylim(0, max(axes[alt_idx].get_ylim()[1], 0.1))

    if fig is not None:
        fig.canvas.draw()
        fig.canvas.flush_events()
    if fig_hist is not None:
        fig_hist.canvas.draw()
        fig_hist.canvas.flush_events()


def finalize_layout(fig, fig_hist):
    """Apply final layout tweaks to plots."""
    try:
        if fig is not None:
            fig.tight_layout()
            fig.subplots_adjust(top=0.95, bottom=0.08)
        if fig_hist is not None:
            fig_hist.tight_layout()
            fig_hist.subplots_adjust(
                hspace=0.40,
                wspace=0.30,
                top=0.95,
                bottom=0.05,
                left=0.08,
                right=0.95,
            )
    except Exception:
        pass
