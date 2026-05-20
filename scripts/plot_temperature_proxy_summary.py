"""Create a compact poster figure for the SHAP temperature-proxy argument.

The figure summarizes the controlled comparisons behind the conclusion that
temperature attribution in DWTP-B behaves like a source-specific proxy for
chlorination practice rather than a stable chemical driver.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch


PROJECT = Path(__file__).resolve().parents[1]
INPUT = PROJECT / "results" / "shap_attribution" / "all_shap_results.csv"
OUTDIR = PROJECT / "results" / "shap_attribution" / "summary"


COLORS = {
    "temp": "#D6604D",
    "temp_light": "#F5C5B8",
    "cl2": "#2166AC",
    "cl2_light": "#BFD7EA",
    "ink": "#1F2933",
    "muted": "#68737D",
    "grid": "#D9DEE3",
    "panel": "#F7F8FA",
    "sample": "#6B7280",
    "source": "#B45309",
    "proxy": "#1D4ED8",
}


def fmt_rank(value: float) -> str:
    if abs(value - round(value)) < 0.05:
        return f"#{round(value):.0f}"
    return f"#{value:.1f}"


def condition_mean(df: pd.DataFrame, condition: str, feature: str) -> float:
    subset = df[(df["condition"] == condition) & (df["feature_label"] == feature)]
    return float(subset["rank"].mean())


def pooled_mean(df: pd.DataFrame, conditions: list[str], feature: str) -> float:
    subset = df[df["condition"].isin(conditions) & (df["feature_label"] == feature)]
    return float(subset["rank"].mean())


def draw_card(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    color: str,
    title: str,
    body: str,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.035",
        linewidth=1.1,
        edgecolor=color,
        facecolor="white",
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.04,
        y + height * 0.66,
        title,
        transform=ax.transAxes,
        color=color,
        fontsize=7.9,
        fontweight="bold",
        va="center",
    )
    ax.text(
        x + 0.04,
        y + height * 0.30,
        body,
        transform=ax.transAxes,
        color=COLORS["ink"],
        fontsize=6.95,
        va="center",
    )


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)

    temp_ap = condition_mean(df, "A'", "Temperature")
    temp_d = condition_mean(df, "D", "Temperature")
    temp_b = condition_mean(df, "B", "Temperature")
    temp_plus_cl2 = pooled_mean(df, ["C", "E"], "Temperature")
    cl2_plus_cl2 = pooled_mean(df, ["C", "E"], "Cl₂ dose")

    summary_rows = pd.DataFrame(
        [
            {"signal": "Temperature", "condition_group": "A' DWTP-B, N=175, 5 features", "mean_rank": temp_ap},
            {"signal": "Temperature", "condition_group": "D Dataset1, N=175, 5 features", "mean_rank": temp_d},
            {"signal": "Temperature", "condition_group": "B Dataset1, N=488, 5 features", "mean_rank": temp_b},
            {"signal": "Temperature", "condition_group": "C/E Dataset1, +Cl2", "mean_rank": temp_plus_cl2},
            {"signal": "Cl2 dose", "condition_group": "C/E Dataset1, +Cl2", "mean_rank": cl2_plus_cl2},
        ]
    )
    summary_rows.to_csv(OUTDIR / "temperature_proxy_summary_ranks.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 12,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(4.15, 6.15), dpi=300, facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.16, 0.98], hspace=0.42)

    ax = fig.add_subplot(gs[0])
    x = [0, 1, 2, 3]
    temp_y = [temp_ap, temp_d, temp_b, temp_plus_cl2]
    labels = [
        "A'\nDWTP-B\nN=175\n5 feat",
        "D\nDataset1\nN=175\n5 feat",
        "B\nDataset1\nN=488\n5 feat",
        "C/E\nDataset1\n+ Cl2",
    ]

    ax.set_facecolor("white")
    ax.plot(
        x,
        temp_y,
        color=COLORS["temp"],
        linewidth=2.6,
        marker="o",
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=2.1,
        zorder=5,
    )
    ax.scatter(
        [3],
        [cl2_plus_cl2],
        s=82,
        marker="D",
        facecolor="white",
        edgecolor=COLORS["cl2"],
        linewidth=2.1,
        zorder=6,
    )

    for xi, yi in zip(x, temp_y):
        ax.text(
            xi,
            yi + 0.35,
            fmt_rank(yi),
            ha="center",
            va="bottom",
            color=COLORS["temp"],
            fontsize=8.4,
            fontweight="bold",
        )
    ax.text(
        3,
        cl2_plus_cl2 - 0.35,
        f"Cl2 {fmt_rank(cl2_plus_cl2)}",
        ha="center",
        va="top",
        color=COLORS["cl2"],
        fontsize=8.2,
        fontweight="bold",
    )

    ax.annotate(
        "same N/features\nsource effect",
        xy=(1, temp_d),
        xytext=(0.5, 5.55),
        arrowprops=dict(arrowstyle="->", color=COLORS["source"], lw=1.2, shrinkA=0, shrinkB=5),
        color=COLORS["source"],
        ha="center",
        va="center",
        fontsize=7.6,
        fontweight="bold",
    )
    ax.annotate(
        "same source/features\nN not cause",
        xy=(2, temp_b),
        xytext=(1.5, 1.55),
        arrowprops=dict(arrowstyle="->", color=COLORS["sample"], lw=1.2, shrinkA=0, shrinkB=5),
        color=COLORS["sample"],
        ha="center",
        va="center",
        fontsize=7.6,
        fontweight="bold",
    )
    ax.annotate(
        "+ direct\nchlorination signal",
        xy=(3, cl2_plus_cl2),
        xytext=(2.48, 2.55),
        arrowprops=dict(arrowstyle="->", color=COLORS["cl2"], lw=1.2, shrinkA=0, shrinkB=5),
        color=COLORS["cl2"],
        ha="center",
        va="center",
        fontsize=7.6,
        fontweight="bold",
    )

    ax.set_ylim(6.25, 0.55)
    ax.set_xlim(-0.35, 3.35)
    ax.set_yticks([1, 2, 3, 4, 5, 6])
    ax.set_ylabel("Mean SHAP rank\n(#1 = dominant)", color=COLORS["ink"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.tick_params(axis="both", colors=COLORS["muted"], length=0)
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    ax.text(
        0.0,
        1.13,
        "Temperature proxy diagnosis",
        transform=ax.transAxes,
        color=COLORS["ink"],
        fontsize=12.4,
        fontweight="bold",
        va="bottom",
    )
    ax.text(
        0.0,
        1.045,
        "Mean SHAP rank across 2 models x 3 targets; D/E average five N=175 subsamples",
        transform=ax.transAxes,
        color=COLORS["muted"],
        fontsize=7.15,
        va="bottom",
    )

    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    ax2.text(
        0.0,
        1.0,
        "Evidence checks",
        transform=ax2.transAxes,
        color=COLORS["ink"],
        fontsize=10.2,
        fontweight="bold",
        va="top",
    )

    draw_card(
        ax2,
        (0.00, 0.66),
        1.0,
        0.22,
        COLORS["sample"],
        "1  Sample-size control: D vs B",
        f"D {fmt_rank(temp_d)} ~= B {fmt_rank(temp_b)} -> smaller N does not restore dominance",
    )
    draw_card(
        ax2,
        (0.00, 0.38),
        1.0,
        0.22,
        COLORS["source"],
        "2  Data-source control: D vs A'",
        f"A' {fmt_rank(temp_ap)} -> D {fmt_rank(temp_d)} at same N/features -> source structure",
    )
    draw_card(
        ax2,
        (0.00, 0.10),
        1.0,
        0.22,
        COLORS["proxy"],
        "3  Proxy check: add Cl2 dose",
        f"With Cl2: Cl2 {fmt_rank(cl2_plus_cl2)}, Temp {fmt_rank(temp_plus_cl2)} -> proxy resolved",
    )

    fig.text(
        0.5,
        0.032,
        "Conclusion: multi-system data dissolves the temperature-confounding signal.",
        ha="center",
        va="bottom",
        fontsize=8.4,
        fontweight="bold",
        color=COLORS["ink"],
    )

    png = OUTDIR / "temperature_proxy_summary_rank_trajectory.png"
    pdf = OUTDIR / "temperature_proxy_summary_rank_trajectory.pdf"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.16)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


if __name__ == "__main__":
    main()
