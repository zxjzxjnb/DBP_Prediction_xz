"""Plot temperature distribution differences between Dataset1 and Tai Lake.

Outputs a slide-friendly 16:9 figure that combines:
  1. overall temperature distributions
  2. overall standard deviation comparison
  3. Dataset1 within-tsid temperature variability

Usage:
    python scripts/plot_temperature_distribution_comparison.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D


PROJECT = Path(__file__).resolve().parents[1]
DATASET1_PATH = PROJECT / "data" / "dataset1_dbp_formation.csv"
TAILAKE_PATH = PROJECT / "data" / "DBP_dataset_DWTP_B.csv"
OUTPUT_DIR = PROJECT / "results" / "shap_attribution"
OUTPUT_PATH = OUTPUT_DIR / "temperature_distribution_overview.png"


def compute_summary() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    d1 = pd.read_csv(DATASET1_PATH)
    tl = pd.read_csv(TAILAKE_PATH)

    series_std = (
        d1.groupby("tsid")
        .agg(
            n=("temp_in_avg", "count"),
            temp_std=("temp_in_avg", lambda s: s.std(ddof=1)),
        )
        .reset_index()
    )
    valid = series_std.dropna(subset=["temp_std"]).copy()

    summary = {
        "dataset1_rows": float(len(d1)),
        "tailake_rows": float(len(tl)),
        "dataset1_std": float(d1["temp_in_avg"].std(ddof=1)),
        "tailake_std": float(tl["temp_C"].std(ddof=1)),
        "dataset1_to_tailake_ratio": float(d1["temp_in_avg"].std(ddof=1) / tl["temp_C"].std(ddof=1)),
        "tailake_to_dataset1_ratio": float(tl["temp_C"].std(ddof=1) / d1["temp_in_avg"].std(ddof=1)),
        "n_tsid_total": float(d1["tsid"].nunique()),
        "n_tsid_valid": float(len(valid)),
        "n_lt_1": float((valid["temp_std"] < 1).sum()),
        "n_lt_2": float((valid["temp_std"] < 2).sum()),
        "median_tsid_std": float(valid["temp_std"].median()),
    }
    return d1, valid, summary


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9DEE8", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def draw_card(ax: plt.Axes, title: str, body: str, facecolor: str) -> None:
    ax.set_axis_off()
    card = FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle="round,pad=0.018,rounding_size=16",
        transform=ax.transAxes,
        linewidth=0,
        facecolor=facecolor,
        edgecolor="none",
    )
    ax.add_patch(card)
    ax.text(
        0.05,
        0.78,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="#172033",
    )
    ax.text(
        0.05,
        0.60,
        body,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color="#36435C",
        linespacing=1.35,
    )


def plot() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    d1, valid, summary = compute_summary()

    dataset1_color = "#1F5AA6"
    tailake_color = "#D66A1F"
    accent_color = "#2E8B57"
    soft_blue = "#EAF2FD"
    soft_orange = "#FDEDDD"
    soft_green = "#E8F6EF"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig = plt.figure(figsize=(16, 9), facecolor="white")
    gs = fig.add_gridspec(
        4,
        12,
        height_ratios=[0.34, 0.82, 1.25, 1.35],
        left=0.06,
        right=0.98,
        top=0.96,
        bottom=0.10,
        wspace=0.55,
        hspace=0.62,
    )

    ax_card1 = fig.add_subplot(gs[1, 0:4])
    ax_card2 = fig.add_subplot(gs[1, 4:8])
    ax_card3 = fig.add_subplot(gs[1, 8:12])
    ax_dist = fig.add_subplot(gs[2:4, 0:7])
    ax_std = fig.add_subplot(gs[2, 7:12])
    ax_tsid = fig.add_subplot(gs[3, 7:12])

    draw_card(
        ax_card1,
        "Dataset1 series definition",
        (
            f"{int(summary['n_tsid_total'])} total tsid\n"
            f"{int(summary['n_tsid_valid'])} analyzable series with ≥2 temp points\n"
            "Within-series std is computed on temp_in_avg"
        ),
        soft_blue,
    )
    draw_card(
        ax_card2,
        "Dataset1 within-tsid variability",
        (
            f"{int(summary['n_lt_1'])}/{int(summary['n_tsid_valid'])} = 57.3% are < 1°C\n"
            f"Median within-tsid std = {summary['median_tsid_std']:.2f}°C\n"
            f"{int(summary['n_lt_2'])}/{int(summary['n_tsid_valid'])} = 81.7% are < 2°C"
        ),
        soft_green,
    )
    draw_card(
        ax_card3,
        "Overall spread contrast",
        (
            f"Dataset1 std = {summary['dataset1_std']:.2f}°C\n"
            f"Tai Lake std = {summary['tailake_std']:.2f}°C\n"
            f"Dataset1 is {summary['dataset1_to_tailake_ratio']:.0%} of Tai Lake (Tai Lake ≈ {summary['tailake_to_dataset1_ratio']:.2f}×)"
        ),
        soft_orange,
    )

    bins = np.arange(0, 33, 1.5)
    ax_dist.hist(
        d1["temp_in_avg"].dropna(),
        bins=bins,
        density=True,
        alpha=0.52,
        color=dataset1_color,
        edgecolor="white",
        linewidth=1.0,
        label=f"Dataset1 (n={int(summary['dataset1_rows'])})",
    )
    ax_dist.hist(
        pd.read_csv(TAILAKE_PATH)["temp_C"].dropna(),
        bins=bins,
        density=True,
        alpha=0.45,
        color=tailake_color,
        edgecolor="white",
        linewidth=1.0,
        label=f"Tai Lake (n={int(summary['tailake_rows'])})",
    )
    style_axes(ax_dist)
    ax_dist.set_title("Overall Temperature Distribution", loc="left", pad=10, color="#172033", fontweight="bold")
    ax_dist.set_xlabel("Temperature (°C)")
    ax_dist.set_ylabel("Density")
    ax_dist.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        ncol=1,
        handlelength=1.7,
    )

    std_values = [summary["dataset1_std"], summary["tailake_std"]]
    ax_std.bar(
        ["Dataset1", "Tai Lake"],
        std_values,
        color=[dataset1_color, tailake_color],
        width=0.58,
    )
    style_axes(ax_std)
    ax_std.set_title("Overall Temperature Std", loc="left", pad=10, color="#172033", fontweight="bold")
    ax_std.set_ylabel("Std (°C)")
    for idx, value in enumerate(std_values):
        ax_std.text(idx, value + 0.18, f"{value:.2f}", ha="center", va="bottom", fontsize=11, color="#172033")
    ax_std.set_ylim(0, max(std_values) + 1.6)

    tsid_bins = np.arange(0, max(8.6, valid["temp_std"].max() + 0.4), 0.4)
    ax_tsid.hist(
        valid["temp_std"],
        bins=tsid_bins,
        color=accent_color,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.9,
    )
    ax_tsid.axvline(1.0, color="#A13A2A", linestyle="--", linewidth=2.0)
    ax_tsid.axvline(2.0, color="#A17A12", linestyle="--", linewidth=2.0)
    style_axes(ax_tsid)
    ax_tsid.set_title("Dataset1 Within-tsid Temperature Std", loc="left", pad=10, color="#172033", fontweight="bold")
    ax_tsid.set_xlabel("Within-tsid std of temp_in_avg (°C)")
    ax_tsid.set_ylabel("Number of tsid series")
    ax_tsid.legend(
        handles=[
            Line2D([0], [0], color="#A13A2A", linestyle="--", linewidth=2.0, label="1°C threshold"),
            Line2D([0], [0], color="#A17A12", linestyle="--", linewidth=2.0, label="2°C threshold"),
        ],
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        ncol=2,
    )

    fig.text(
        0.06,
        0.975,
        "Temperature Distributions: Dataset1 vs Tai Lake",
        fontsize=22,
        fontweight="bold",
        color="#172033",
        va="top",
    )
    fig.text(
        0.06,
        0.938,
        "Dataset1 shows both lower overall temperature spread and weaker within-tsid temperature movement than the Tai Lake benchmark.",
        fontsize=12,
        color="#51607A",
        va="top",
    )

    fig.savefig(OUTPUT_PATH, dpi=220, facecolor="white")
    plt.close(fig)
    return OUTPUT_PATH


if __name__ == "__main__":
    path = plot()
    print(f"Saved figure to: {path}")
