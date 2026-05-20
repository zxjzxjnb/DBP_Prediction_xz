"""Create a minimal poster-ready SHAP-rank summary figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
INPUT = PROJECT / "results" / "shap_attribution" / "all_shap_results.csv"
OUTDIR = PROJECT / "results" / "shap_attribution" / "summary"


def fmt_rank(value: float) -> str:
    if abs(value - round(value)) < 0.05:
        return f"#{round(value):.0f}"
    return f"#{value:.1f}"


def mean_rank(df: pd.DataFrame, condition: str, feature: str) -> float:
    subset = df[(df["condition"] == condition) & (df["feature_label"] == feature)]
    return float(subset["rank"].mean())


def pooled_mean_rank(df: pd.DataFrame, conditions: list[str], feature: str) -> float:
    subset = df[df["condition"].isin(conditions) & (df["feature_label"] == feature)]
    return float(subset["rank"].mean())


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)

    temp_ranks = [
        mean_rank(df, "A'", "Temperature"),
        mean_rank(df, "D", "Temperature"),
        mean_rank(df, "B", "Temperature"),
        pooled_mean_rank(df, ["C", "E"], "Temperature"),
    ]
    cl2_rank = pooled_mean_rank(df, ["C", "E"], "Cl₂ dose")

    pd.DataFrame(
        [
            {"signal": "Temperature", "condition": "A'", "mean_rank": temp_ranks[0]},
            {"signal": "Temperature", "condition": "D", "mean_rank": temp_ranks[1]},
            {"signal": "Temperature", "condition": "B", "mean_rank": temp_ranks[2]},
            {"signal": "Temperature", "condition": "C/E", "mean_rank": temp_ranks[3]},
            {"signal": "Cl2 dose", "condition": "C/E", "mean_rank": cl2_rank},
        ]
    ).to_csv(OUTDIR / "temperature_proxy_minimal_ranks.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    x = [0, 1, 2, 3]
    labels = ["A'\nDWTP-B", "D\nDataset1", "B\nDataset1", "C/E\n+Cl$_2$"]

    fig, ax = plt.subplots(figsize=(3.15, 2.35), dpi=300, facecolor="white")
    red = "#D6604D"
    blue = "#2166AC"
    ink = "#1F2933"
    muted = "#68737D"
    grid = "#D9DEE3"

    ax.plot(
        x,
        temp_ranks,
        color=red,
        linewidth=2.2,
        marker="o",
        markersize=6.2,
        markerfacecolor="white",
        markeredgewidth=1.8,
        zorder=3,
    )
    ax.scatter(
        [3],
        [cl2_rank],
        s=48,
        marker="D",
        facecolor="white",
        edgecolor=blue,
        linewidth=1.8,
        zorder=4,
    )

    for xi, yi in zip(x, temp_ranks):
        ax.text(
            xi,
            yi + 0.18,
            fmt_rank(yi),
            ha="center",
            va="top",
            fontsize=7.2,
            color=red,
            fontweight="bold",
        )
    ax.text(
        2.84,
        cl2_rank,
        fmt_rank(cl2_rank),
        ha="right",
        va="center",
        fontsize=7.2,
        color=blue,
        fontweight="bold",
    )

    ax.text(3.18, temp_ranks[-1], "Temp", ha="left", va="center", fontsize=7.2, color=red, fontweight="bold")
    ax.text(3.18, cl2_rank, "Cl$_2$", ha="left", va="center", fontsize=7.2, color=blue, fontweight="bold")

    ax.set_xlim(-0.35, 3.60)
    ax.set_ylim(5.35, 0.7)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylabel("Mean SHAP rank", color=ink)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", color=grid, linewidth=0.75)
    ax.tick_params(axis="both", colors=muted, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    png = OUTDIR / "temperature_proxy_minimal_rank_shift.png"
    pdf = OUTDIR / "temperature_proxy_minimal_rank_shift.pdf"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    main()
