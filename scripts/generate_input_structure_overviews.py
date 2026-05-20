"""Generate slide-friendly overview figures for each modeled input.

The layout follows the style of temperature_distribution_overview.png:
  1. three summary cards
  2. overall distribution panel
  3. spread comparison panel
  4. Dataset1 within-tsid variability panel when available

Outputs:
  results/input_structure/overview_pngs/*.png

Usage:
    python scripts/generate_input_structure_overviews.py
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
from matplotlib.lines import Line2D

from analyze_input_structure import (
    DATASET1_COLOR,
    TAILAKE_COLOR,
    ACCENT_COLOR,
    FEATURE_SPECS,
    PROJECT,
    add_card,
    histogram_bins,
    safe_float,
    slugify,
    style_axes,
)


DATASET1_PATH = PROJECT / "data" / "dataset1_dbp_formation.csv"
TAILAKE_PATH = PROJECT / "data" / "DBP_dataset_DWTP_B.csv"
OUTPUT_DIR = PROJECT / "results" / "input_structure" / "overview_pngs"

SOFT_BLUE = "#EAF2FD"
SOFT_GREEN = "#E8F6EF"
SOFT_ORANGE = "#FDEDDD"
TEXT_COLOR = "#172033"
MUTED_TEXT = "#51607A"


def fmt(value: float | int, digits: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    return f"{float(value):.{digits}f}"


def display_unit(spec_name: str, base_unit: str) -> str:
    if spec_name == "Bromide":
        return "mg/L"
    return base_unit


def transform_series(spec_name: str, dataset_name: str, series: pd.Series) -> pd.Series:
    """Return a visualization-friendly series.

    Bromide is the only special case. Dataset1 bromide is likely stored on a
    microgram-per-liter scale while the Tai Lake table explicitly uses mg/L.
    For visual comparability, the overview figures render Dataset1 bromide in
    mg/L-equivalent by dividing by 1000. This is an inference and is called out
    in the figure subtitle.
    """
    if spec_name == "Bromide" and dataset_name == "dataset1":
        return series / 1000.0
    return series


def summary_stats(series: pd.Series) -> dict[str, float | int]:
    clean = series.dropna()
    q1 = safe_float(clean.quantile(0.25)) if not clean.empty else float("nan")
    q3 = safe_float(clean.quantile(0.75)) if not clean.empty else float("nan")
    p05 = safe_float(clean.quantile(0.05)) if not clean.empty else float("nan")
    p95 = safe_float(clean.quantile(0.95)) if not clean.empty else float("nan")
    return {
        "n_total": int(len(series)),
        "n_non_missing": int(clean.size),
        "n_missing": int(series.isna().sum()),
        "missing_rate": safe_float(series.isna().mean()),
        "mean": safe_float(clean.mean()) if not clean.empty else float("nan"),
        "median": safe_float(clean.median()) if not clean.empty else float("nan"),
        "std": safe_float(clean.std(ddof=1)) if len(clean) > 1 else float("nan"),
        "min": safe_float(clean.min()) if not clean.empty else float("nan"),
        "q25": q1,
        "q75": q3,
        "iqr": safe_float(q3 - q1),
        "p05": p05,
        "p95": p95,
        "p95_p05": safe_float(p95 - p05),
    }


def dataset1_within_tsid_values(df: pd.DataFrame, spec_name: str, column: str) -> pd.Series:
    values = transform_series(spec_name, "dataset1", df[column])
    grouped = pd.DataFrame({"tsid": df["tsid"], "value": values}).groupby("tsid")["value"].std(ddof=1)
    return grouped.dropna()


def axis_label(feature_name: str, unit: str) -> str:
    return f"{feature_name} ({unit})" if unit else feature_name


def feature_title(spec) -> str:
    if spec.dataset1_col and spec.tailake_col:
        return f"{spec.name} Distributions: Dataset1 vs Tai Lake"
    if spec.dataset1_col:
        return f"{spec.name} Distribution: Dataset1"
    return f"{spec.name} Distribution: Tai Lake"


def feature_subtitle(spec, d1_stats: dict | None, tl_stats: dict | None, within_stats: dict | None) -> str:
    if spec.name == "Bromide":
        return (
            "Dataset1 bromide is plotted in mg/L-equivalent for shape comparison, "
            "assuming the stored values are on a ug/L scale."
        )
    if spec.dataset1_col and spec.tailake_col:
        ratio = safe_float(d1_stats["std"] / tl_stats["std"]) if tl_stats and tl_stats["std"] else float("nan")
        if pd.notna(ratio) and ratio < 1:
            return (
                f"Dataset1 has a narrower overall {spec.name} spread than Tai Lake, "
                "while its within-tsid movement remains smaller than dataset-level spread."
            )
        return (
            f"Dataset1 has a wider overall {spec.name} spread than Tai Lake, "
            "but most within-tsid variation stays below the overall dataset spread."
        )
    if spec.dataset1_col:
        return (
            f"{spec.name} is a Dataset1-only modeled input. The overview highlights both "
            "overall spread and how much movement occurs within individual tsid series."
        )
    return (
        f"{spec.name} is a Tai Lake-only modeled input from the 9-feature backbone. "
        "No Dataset1 within-tsid analogue is available for this variable."
    )


def coverage_card_lines(spec, d1_stats: dict | None, tl_stats: dict | None) -> tuple[str, str]:
    if spec.dataset1_col and spec.tailake_col:
        title = "Coverage and center"
        body = (
            f"Dataset1: n={d1_stats['n_non_missing']}, miss={100 * d1_stats['missing_rate']:.1f}%\n"
            f"Tai Lake: n={tl_stats['n_non_missing']}, miss={100 * tl_stats['missing_rate']:.1f}%\n"
            f"Medians: {fmt(d1_stats['median'])} vs {fmt(tl_stats['median'])}"
        )
        return title, body
    if spec.dataset1_col:
        title = "Dataset1 coverage"
        body = (
            f"Rows: {d1_stats['n_non_missing']} / {d1_stats['n_total']}\n"
            f"Missing rate: {100 * d1_stats['missing_rate']:.1f}%\n"
            f"Mean / median: {fmt(d1_stats['mean'])} / {fmt(d1_stats['median'])}"
        )
        return title, body
    title = "Tai Lake coverage"
    body = (
        f"Rows: {tl_stats['n_non_missing']} / {tl_stats['n_total']}\n"
        f"Missing rate: {100 * tl_stats['missing_rate']:.1f}%\n"
        f"Mean / median: {fmt(tl_stats['mean'])} / {fmt(tl_stats['median'])}"
    )
    return title, body


def within_card_lines(spec, within_values: pd.Series, within_stats: dict | None) -> tuple[str, str]:
    if spec.dataset1_col:
        title = "Dataset1 within-tsid variability"
        below_half = int((within_values < 0.5 * within_stats["overall_std"]).sum()) if not within_values.empty else 0
        body = (
            f"{int(within_stats['n_valid'])}/{int(within_stats['n_total'])} tsid have valid std\n"
            f"Median within-tsid std = {fmt(within_stats['median'])}\n"
            f"{below_half}/{int(within_stats['n_valid'])} are < 0.5x overall std"
        )
        return title, body
    title = "Tai Lake dataset note"
    body = (
        "Single DWTP table (B)\n"
        "No tsid-style repeated-series index\n"
        "Within-series variability panel is not applicable"
    )
    return title, body


def spread_card_lines(spec, d1_stats: dict | None, tl_stats: dict | None, within_stats: dict | None) -> tuple[str, str]:
    if spec.dataset1_col and spec.tailake_col:
        ratio = safe_float(d1_stats["std"] / tl_stats["std"]) if tl_stats["std"] else float("nan")
        title = "Overall spread contrast"
        if pd.notna(ratio) and ratio < 1:
            reverse = safe_float(tl_stats["std"] / d1_stats["std"]) if d1_stats["std"] else float("nan")
            body = (
                f"Dataset1 std = {fmt(d1_stats['std'])}\n"
                f"Tai Lake std = {fmt(tl_stats['std'])}\n"
                f"Dataset1 is {ratio:.0%} of Tai Lake (Tai Lake ≈ {fmt(reverse)}x)"
            )
        else:
            body = (
                f"Dataset1 std = {fmt(d1_stats['std'])}\n"
                f"Tai Lake std = {fmt(tl_stats['std'])}\n"
                f"Dataset1 ≈ {fmt(ratio)}x Tai Lake"
            )
        return title, body
    if spec.dataset1_col:
        title = "Dataset1 spread profile"
        body = (
            f"Std = {fmt(d1_stats['std'])}\n"
            f"IQR = {fmt(d1_stats['iqr'])}\n"
            f"P95-P05 = {fmt(d1_stats['p95_p05'])}"
        )
        return title, body
    title = "Tai Lake spread profile"
    body = (
        f"Std = {fmt(tl_stats['std'])}\n"
        f"IQR = {fmt(tl_stats['iqr'])}\n"
        f"P95-P05 = {fmt(tl_stats['p95_p05'])}"
    )
    return title, body


def add_hist_panel(ax: plt.Axes, spec, d1_values: pd.Series, tl_values: pd.Series, unit: str) -> None:
    bins = histogram_bins([s for s in [d1_values, tl_values] if not s.empty], log_scale=spec.log_scale)

    if not d1_values.empty:
        ax.hist(
            d1_values,
            bins=bins,
            density=True,
            alpha=0.52,
            color=DATASET1_COLOR,
            edgecolor="white",
            linewidth=1.0,
            label=f"Dataset1 (n={len(d1_values)})",
        )
    if not tl_values.empty:
        ax.hist(
            tl_values,
            bins=bins,
            density=True,
            alpha=0.45,
            color=TAILAKE_COLOR,
            edgecolor="white",
            linewidth=1.0,
            label=f"Tai Lake (n={len(tl_values)})",
        )
    if spec.log_scale:
        ax.set_xscale("log")

    style_axes(ax)
    ax.set_title(f"Overall {spec.name} Distribution", loc="left", pad=10, color=TEXT_COLOR, fontweight="bold")
    ax.set_xlabel(axis_label(spec.name, unit))
    ax.set_ylabel("Density")
    if not d1_values.empty or not tl_values.empty:
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.99), ncol=1, handlelength=1.7)


def add_shared_std_panel(ax: plt.Axes, spec, d1_stats: dict, tl_stats: dict, unit: str) -> None:
    values = [d1_stats["std"], tl_stats["std"]]
    ax.bar(["Dataset1", "Tai Lake"], values, color=[DATASET1_COLOR, TAILAKE_COLOR], width=0.58)
    style_axes(ax)
    ax.set_title(f"Overall {spec.name} Std", loc="left", pad=10, color=TEXT_COLOR, fontweight="bold")
    ax.set_ylabel(f"Std ({unit})" if unit else "Std")
    max_val = max(values)
    for idx, value in enumerate(values):
        ax.text(idx, value + max_val * 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=11, color=TEXT_COLOR)
    ax.set_ylim(0, max_val * 1.18 if max_val > 0 else 1)


def add_dataset1_only_std_panel(ax: plt.Axes, spec, d1_stats: dict, within_stats: dict, unit: str) -> None:
    labels = ["Overall std", "Within median", "Within mean"]
    values = [d1_stats["std"], within_stats["median"], within_stats["mean"]]
    ax.bar(labels, values, color=[DATASET1_COLOR, ACCENT_COLOR, "#7AC4A1"], width=0.58)
    style_axes(ax)
    ax.set_title(f"{spec.name} Spread Summary", loc="left", pad=10, color=TEXT_COLOR, fontweight="bold")
    ax.set_ylabel(f"Value ({unit})" if unit else "Value")
    max_val = max(values)
    for idx, value in enumerate(values):
        ax.text(idx, value + max_val * 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=11, color=TEXT_COLOR)
    ax.set_ylim(0, max_val * 1.18 if max_val > 0 else 1)


def add_tailake_only_spread_panel(ax: plt.Axes, spec, tl_stats: dict, unit: str) -> None:
    labels = ["Std", "IQR", "P95-P05"]
    values = [tl_stats["std"], tl_stats["iqr"], tl_stats["p95_p05"]]
    ax.bar(labels, values, color=[TAILAKE_COLOR, "#E58D49", "#F0B07D"], width=0.58)
    style_axes(ax)
    ax.set_title(f"{spec.name} Spread Summary", loc="left", pad=10, color=TEXT_COLOR, fontweight="bold")
    ax.set_ylabel(f"Value ({unit})" if unit else "Value")
    max_val = max(values)
    for idx, value in enumerate(values):
        ax.text(idx, value + max_val * 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=11, color=TEXT_COLOR)
    ax.set_ylim(0, max_val * 1.18 if max_val > 0 else 1)


def add_within_panel(ax: plt.Axes, spec, within_values: pd.Series, within_stats: dict, unit: str) -> None:
    bins = histogram_bins([within_values], log_scale=False)
    ax.hist(
        within_values,
        bins=bins,
        color=ACCENT_COLOR,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.9,
    )
    half_threshold = 0.5 * within_stats["overall_std"]
    full_threshold = within_stats["overall_std"]
    ax.axvline(half_threshold, color="#A13A2A", linestyle="--", linewidth=2.0)
    ax.axvline(full_threshold, color="#A17A12", linestyle="--", linewidth=2.0)
    style_axes(ax)
    ax.set_title(
        f"Dataset1 Within-tsid {spec.name} Std",
        loc="left",
        pad=10,
        color=TEXT_COLOR,
        fontweight="bold",
    )
    ax.set_xlabel(f"Within-tsid std of {axis_label(spec.name, unit)}")
    ax.set_ylabel("Number of tsid series")
    ax.legend(
        handles=[
            Line2D([0], [0], color="#A13A2A", linestyle="--", linewidth=2.0, label="0.5x overall std"),
            Line2D([0], [0], color="#A17A12", linestyle="--", linewidth=2.0, label="1.0x overall std"),
        ],
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        ncol=2,
    )


def add_tailake_quantile_panel(ax: plt.Axes, spec, tl_stats: dict, unit: str) -> None:
    style_axes(ax)
    ax.set_title(f"Tai Lake {spec.name} Quantile Bands", loc="left", pad=10, color=TEXT_COLOR, fontweight="bold")
    ax.hlines(1, tl_stats["p05"], tl_stats["p95"], color=TAILAKE_COLOR, linewidth=5, alpha=0.35)
    ax.hlines(1, tl_stats["q25"], tl_stats["q75"], color=TAILAKE_COLOR, linewidth=11, alpha=0.65)
    ax.plot(tl_stats["median"], 1, marker="o", color=TEXT_COLOR, markersize=7)
    ax.set_yticks([])
    ax.set_xlabel(axis_label(spec.name, unit))
    ax.text(tl_stats["p05"], 1.10, "P05", ha="center", va="bottom", fontsize=10, color=MUTED_TEXT)
    ax.text(tl_stats["q25"], 0.82, "Q1", ha="center", va="top", fontsize=10, color=MUTED_TEXT)
    ax.text(tl_stats["median"], 1.15, "Median", ha="center", va="bottom", fontsize=10, color=MUTED_TEXT)
    ax.text(tl_stats["q75"], 0.82, "Q3", ha="center", va="top", fontsize=10, color=MUTED_TEXT)
    ax.text(tl_stats["p95"], 1.10, "P95", ha="center", va="bottom", fontsize=10, color=MUTED_TEXT)
    ax.set_ylim(0.55, 1.40)


def generate_feature_figure(spec, dataset1: pd.DataFrame, tailake: pd.DataFrame) -> Path:
    d1_values = (
        transform_series(spec.name, "dataset1", dataset1[spec.dataset1_col]).dropna()
        if spec.dataset1_col
        else pd.Series(dtype=float)
    )
    tl_values = (
        transform_series(spec.name, "tailake", tailake[spec.tailake_col]).dropna()
        if spec.tailake_col
        else pd.Series(dtype=float)
    )
    within_values = (
        dataset1_within_tsid_values(dataset1, spec.name, spec.dataset1_col)
        if spec.dataset1_col
        else pd.Series(dtype=float)
    )

    d1_stats = summary_stats(transform_series(spec.name, "dataset1", dataset1[spec.dataset1_col])) if spec.dataset1_col else None
    tl_stats = summary_stats(transform_series(spec.name, "tailake", tailake[spec.tailake_col])) if spec.tailake_col else None
    within_stats = None
    if spec.dataset1_col:
        overall_std = d1_stats["std"]
        within_stats = {
            "n_total": float(dataset1["tsid"].nunique()),
            "n_valid": float(len(within_values)),
            "mean": safe_float(within_values.mean()) if not within_values.empty else float("nan"),
            "median": safe_float(within_values.median()) if not within_values.empty else float("nan"),
            "overall_std": safe_float(overall_std),
        }

    unit = display_unit(spec.name, spec.unit)

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
    ax_aux = fig.add_subplot(gs[3, 7:12])

    card1_title, card1_body = coverage_card_lines(spec, d1_stats, tl_stats)
    card2_title, card2_body = within_card_lines(spec, within_values, within_stats)
    card3_title, card3_body = spread_card_lines(spec, d1_stats, tl_stats, within_stats)

    add_card(ax_card1, [card1_title, card1_body], facecolor=SOFT_BLUE)
    add_card(ax_card2, [card2_title, card2_body], facecolor=SOFT_GREEN)
    add_card(ax_card3, [card3_title, card3_body], facecolor=SOFT_ORANGE)

    add_hist_panel(ax_dist, spec, d1_values, tl_values, unit)

    if spec.dataset1_col and spec.tailake_col:
        add_shared_std_panel(ax_std, spec, d1_stats, tl_stats, unit)
    elif spec.dataset1_col:
        add_dataset1_only_std_panel(ax_std, spec, d1_stats, within_stats, unit)
    else:
        add_tailake_only_spread_panel(ax_std, spec, tl_stats, unit)

    if spec.dataset1_col:
        add_within_panel(ax_aux, spec, within_values, within_stats, unit)
    else:
        add_tailake_quantile_panel(ax_aux, spec, tl_stats, unit)

    fig.text(0.06, 0.975, feature_title(spec), fontsize=22, fontweight="bold", color=TEXT_COLOR, va="top")
    fig.text(0.06, 0.938, feature_subtitle(spec, d1_stats, tl_stats, within_stats), fontsize=12, color=MUTED_TEXT, va="top")

    output_path = OUTPUT_DIR / f"{slugify(spec.name)}_overview.png"
    fig.savefig(output_path, dpi=220, facecolor="white")
    plt.close(fig)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset1 = pd.read_csv(DATASET1_PATH)
    tailake = pd.read_csv(TAILAKE_PATH)

    saved: list[Path] = []
    for spec in FEATURE_SPECS:
        saved.append(generate_feature_figure(spec, dataset1, tailake))

    print(f"Saved {len(saved)} overview figures to: {OUTPUT_DIR}")
    for path in saved:
        print(path.name)


if __name__ == "__main__":
    main()
