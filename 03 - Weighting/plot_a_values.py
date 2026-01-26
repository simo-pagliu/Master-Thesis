import csv
import os
import sys
from typing import List, Dict, Any

import matplotlib.pyplot as plt

# Allow importing the shared auxiliary utilities from UP-MAVT
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UP_MAVT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "UP-MAVT")
sys.path.insert(0, UP_MAVT_DIR)

from auxiliary import load_value_functions_with_confidence  # type: ignore


def recompute_a_values(rows: List[Dict[str, Any]], vf_map: Dict[str, Any]) -> List[float]:
    """Recompute `a` as 1 / vf(Value) for each comparison, in-place on rows."""
    recomputed = []
    for row in rows:
        comp_type = (row.get("Type") or "").strip().lower()
        ref = (row.get("Reference") or "").strip()
        other = (row.get("Other") or "").strip()
        crit_for_value = ref if comp_type == "best" else other

        vf = vf_map.get(crit_for_value)
        value_raw = row.get("Value")
        a_val = None
        if vf is not None and value_raw is not None:
            try:
                vf_val = float(vf(float(value_raw)))
                if vf_val <= 0.0:
                    vf_val = 0.001
                a_val = 1.0 / vf_val
            except Exception:
                a_val = None

        if a_val is None:
            try:
                a_val = float(row.get("a", 0.0))
            except Exception:
                continue

        row["a"] = str(float(a_val))
        recomputed.append(float(a_val))
    return recomputed


def plot_a_values(a_values: List[float], labels: List[str], groups: List[str], title: str) -> None:
    fig, ax = plt.subplots(figsize=(max(10, len(a_values) * 0.4), 6))
    x_pos = range(len(a_values))

    palette = {
        "Technical": "#2b78c8",
        "Economic": "#c87c2b",
        "Social": "#2bc889",
        "Feasibility": "#9b59b6",
        "Between-groups-B": "#f39c12",
        "Between-groups-W": "#e74c3c",
    }
    colors = [palette.get(g, "#7f8c8d") for g in groups]

    ax.bar(x_pos, a_values, color=colors, alpha=0.8, edgecolor="black")
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("a value", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Legend keyed by group
    handles = []
    seen = {}
    for g in groups:
        if g not in seen and g:
            handle = plt.Rectangle((0, 0), 1, 1, color=palette.get(g, "#7f8c8d"), edgecolor="black")
            handles.append((g, handle))
            seen[g] = True
    if handles:
        ax.legend([h for _, h in handles], [g for g, _ in handles], title="Group")

    plt.tight_layout()
    plt.show()


def main() -> None:
    bwt_file = os.path.join(SCRIPT_DIR, "BWT_results.csv")
    vf_file = os.path.join(SCRIPT_DIR, "value_functions.csv")

    vf_map, _ = load_value_functions_with_confidence(vf_file)

    with open(bwt_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames or "a" not in fieldnames:
        raise ValueError("Unexpected BWT file schema (missing 'a' column)")

    a_values = recompute_a_values(rows, vf_map)

    # Persist updated a values back to the CSV
    with open(bwt_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Build labels and groups for the plot
    labels = []
    groups = []
    for row in rows:
        typ = (row.get("Type") or "").strip()
        ref = (row.get("Reference") or "").strip()
        other = (row.get("Other") or "").strip()
        groups.append((row.get("Group") or "").strip())
        labels.append(f"{typ[:1].upper()}: {ref[:14]} vs {other[:14]}")

    plot_title = "Declared a values (recomputed)"
    plot_a_values(a_values, labels, groups, plot_title)


if __name__ == "__main__":
    main()
