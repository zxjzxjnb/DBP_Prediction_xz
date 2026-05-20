"""Plot compact cross-dataset SHAP attribution summaries.

The figures aggregate relative mean |SHAP| share across models, targets, and
subsample seeds, then collapse TOC and UV254 into an organic-precursor group.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
INPUT = PROJECT / "results" / "shap_attribution" / "all_shap_results.csv"
OUTDIR = PROJECT / "results" / "shap_attribution" / "summary"

CONDITION_ORDER = ["A'", "D", "B", "E", "C"]
CONDITION_LABELS = {
    "A'": "A'\nDWTP-B\n5F",
    "D": "D\nD1-5F\nN175",
    "B": "B\nD1-5F\nN488",
    "E": "E\nD1+Cl$_2$\nN175",
    "C": "C\nD1+Cl$_2$\nN488",
}

GROUP_ORDER = ["Temperature", "Cl2 dose", "Bromide", "Organic precursors", "pH"]
GROUP_LABELS = {
    "Temperature": "Temp",
    "Cl2 dose": "Cl$_2$ dose",
    "Bromide": "Br$^-$",
    "Organic precursors": "TOC + UV254",
    "pH": "pH",
}
GROUP_COLORS = {
    "Temperature": "#D6604D",
    "Cl2 dose": "#2166AC",
    "Bromide": "#018571",
    "Organic precursors": "#7F8C2A",
    "pH": "#6B7280",
}


def load_grouped_share() -> pd.DataFrame:
    df = pd.read_csv(INPUT)
    feature_group = {
        "Temperature": "Temperature",
        "Cl\u2082 dose": "Cl2 dose",
        "Bromide": "Bromide",
        "TOC": "Organic precursors",
        "UV254": "Organic precursors",
        "pH": "pH",
    }
    df = df[df["feature_label"].isin(feature_group)].copy()
    df["feature_group"] = df["feature_label"].map(feature_group)

    unit_keys = ["condition", "seed", "model", "target_key"]
    grouped = (
        df.groupby(unit_keys + ["feature_group"], as_index=False)["mean_abs_shap"]
        .sum()
        .rename(columns={"mean_abs_shap": "group_abs_shap"})
    )
    totals = (
        df.groupby(unit_keys, as_index=False)["mean_abs_shap"]
        .sum()
        .rename(columns={"mean_abs_shap": "total_abs_shap"})
    )
    grouped = grouped.merge(totals, on=unit_keys)
    grouped["relative_share"] = grouped["group_abs_shap"] / grouped["total_abs_shap"] * 100

    summary = (
        grouped.groupby(["condition", "feature_group"], as_index=False)["relative_share"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary = summary.rename(columns={"mean": "share_mean", "std": "share_std"})
    return summary


def summary_matrix(summary: pd.DataFrame) -> pd.DataFrame:
    matrix = summary.pivot(index="feature_group", columns="condition", values="share_mean")
    return matrix.reindex(GROUP_ORDER)[CONDITION_ORDER]


def set_common_axis_style(ax: plt.Axes) -> None:
    ax.set_xlim(-0.5, len(CONDITION_ORDER) - 0.5)
    ax.set_ylim(len(GROUP_ORDER) - 0.5, -0.5)
    ax.set_xticks(range(len(CONDITION_ORDER)))
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER])
    ax.set_yticks(range(len(GROUP_ORDER)))
    ax.set_yticklabels([GROUP_LABELS[g] for g in GROUP_ORDER])
    ax.tick_params(axis="both", length=0, colors="#1F2933")
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_dot_matrix(matrix: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4.05, 2.45), dpi=300, facecolor="white")
    ax.set_facecolor("white")
    set_common_axis_style(ax)
    ax.grid(color="#E1E5EA", linewidth=0.8, zorder=0)

    for y, group in enumerate(GROUP_ORDER):
        for x, condition in enumerate(CONDITION_ORDER):
            value = matrix.loc[group, condition]
            if pd.isna(value):
                ax.text(x, y, "-", ha="center", va="center", color="#9AA3AD", fontsize=8)
                continue
            size = 28 + value * 13.5
            color = GROUP_COLORS[group]
            ax.scatter(x, y, s=size, color=color, alpha=0.88, edgecolor="white", linewidth=1.0, zorder=3)
            ax.text(
                x,
                y,
                f"{value:.0f}",
                ha="center",
                va="center",
                color="white" if value >= 14 else "#1F2933",
                fontsize=7.0,
                fontweight="bold" if value >= 14 else "normal",
                zorder=4,
            )

    ax.text(
        1.0,
        1.025,
        "Relative |SHAP| share (%)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        color="#68737D",
    )

    fig.savefig(OUTDIR / "cross_dataset_attribution_fingerprint_dot_matrix.png", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUTDIR / "cross_dataset_attribution_fingerprint_dot_matrix.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def blend_with_white(hex_color: str, strength: float) -> tuple[float, float, float]:
    rgb = mcolors.to_rgb(hex_color)
    strength = max(0.0, min(1.0, strength))
    return tuple((1 - strength) * 1.0 + strength * channel for channel in rgb)


def plot_tile_heatmap(matrix: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4.05, 2.45), dpi=300, facecolor="white")
    ax.set_facecolor("white")
    set_common_axis_style(ax)

    max_value = 60.0
    for y, group in enumerate(GROUP_ORDER):
        for x, condition in enumerate(CONDITION_ORDER):
            value = matrix.loc[group, condition]
            if pd.isna(value):
                face = "#F5F7FA"
                text = "-"
                text_color = "#9AA3AD"
                weight = "normal"
            else:
                strength = 0.16 + 0.78 * min(value / max_value, 1)
                face = blend_with_white(GROUP_COLORS[group], strength)
                text = f"{value:.0f}"
                text_color = "white" if value >= 28 else "#1F2933"
                weight = "bold" if value >= 28 else "normal"
            rect = plt.Rectangle((x - 0.47, y - 0.42), 0.94, 0.84, facecolor=face, edgecolor="white", linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x, y, text, ha="center", va="center", fontsize=7.4, color=text_color, fontweight=weight)

    ax.text(
        1.0,
        1.025,
        "Relative |SHAP| share (%)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        color="#68737D",
    )

    fig.savefig(OUTDIR / "cross_dataset_attribution_fingerprint_tile_heatmap.png", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUTDIR / "cross_dataset_attribution_fingerprint_tile_heatmap.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary = load_grouped_share()
    summary.to_csv(OUTDIR / "cross_dataset_attribution_fingerprint_summary.csv", index=False)
    matrix = summary_matrix(summary)
    matrix.to_csv(OUTDIR / "cross_dataset_attribution_fingerprint_matrix.csv")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 10,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    plot_dot_matrix(matrix)
    plot_tile_heatmap(matrix)


if __name__ == "__main__":
    main()
