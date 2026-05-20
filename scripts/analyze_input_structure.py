"""Analyze modeled input structure across Dataset1 and Tai Lake.

This script focuses on the inputs that actually define the two modeling
backbones used in this repository:

- Tai Lake / small dataset: 9 original water-quality inputs from README
- Dataset1: 7-feature backbone used in the formal reports

Outputs:
  - results/input_structure/input_structure_summary.csv
  - results/input_structure/dataset1_within_tsid_summary.csv
  - results/input_structure/shared_input_comparison.csv
  - results/input_structure/input_structure_summary.md
  - results/input_structure/plots/<feature>.png

Usage:
    python scripts/analyze_input_structure.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


PROJECT = Path(__file__).resolve().parents[1]
DATASET1_PATH = PROJECT / "data" / "dataset1_dbp_formation.csv"
TAILAKE_PATH = PROJECT / "data" / "DBP_dataset_DWTP_B.csv"
OUTPUT_DIR = PROJECT / "results" / "input_structure"
PLOTS_DIR = OUTPUT_DIR / "plots"

SUMMARY_CSV = OUTPUT_DIR / "input_structure_summary.csv"
WITHIN_TSID_CSV = OUTPUT_DIR / "dataset1_within_tsid_summary.csv"
SHARED_CSV = OUTPUT_DIR / "shared_input_comparison.csv"
SUMMARY_MD = OUTPUT_DIR / "input_structure_summary.md"

DATASET1_COLOR = "#1F5AA6"
TAILAKE_COLOR = "#D66A1F"
ACCENT_COLOR = "#2E8B57"
TEXT_COLOR = "#172033"
MUTED_TEXT = "#4D5B73"
GRID_COLOR = "#D9DEE8"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dataset1_col: str | None = None
    tailake_col: str | None = None
    unit: str = ""
    log_scale: bool = False

    @property
    def role(self) -> str:
        if self.dataset1_col and self.tailake_col:
            return "shared"
        if self.dataset1_col:
            return "dataset1_only"
        return "tailake_only"


FEATURE_SPECS = [
    FeatureSpec("pH", "ph_in_avg", "pH"),
    FeatureSpec("UV254", "uv_in_avg", "UV254_A_cm", unit="A/cm"),
    FeatureSpec("Temperature", "temp_in_avg", "temp_C", unit="°C"),
    FeatureSpec("TOC", "toc_in_avg", "TOC_mg_L", unit="mg/L"),
    FeatureSpec("Bromide", "br_in_avg", "Br_mg_L", unit="mg/L", log_scale=True),
    FeatureSpec("Cl2 dose", "cl2d_in_avg", unit="mg/L", log_scale=True),
    FeatureSpec("Contact time", "time_sds_avg", unit="min", log_scale=True),
    FeatureSpec("COD", tailake_col="COD_mg_L", unit="mg/L"),
    FeatureSpec("NH4-N", tailake_col="NH4_N_mg_L", unit="mg/L", log_scale=True),
    FeatureSpec("NO2-N", tailake_col="NO2_N_mg_L", unit="mg/L", log_scale=True),
    FeatureSpec("NO3-N", tailake_col="NO3_N_mg_L", unit="mg/L", log_scale=True),
]


def safe_float(value: float) -> float:
    if pd.isna(value) or np.isinf(value):
        return float("nan")
    return float(value)


def slugify(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("₂", "2")
    )


def fmt(value: float | int, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    return f"{float(value):.{digits}f}"


def axis_label(spec: FeatureSpec) -> str:
    return f"{spec.name} ({spec.unit})" if spec.unit else spec.name


def add_card(ax: plt.Axes, lines: list[str], facecolor: str = "#F5F8FC") -> None:
    ax.set_axis_off()
    patch = FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle="round,pad=0.02,rounding_size=16",
        transform=ax.transAxes,
        linewidth=0,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        0.05,
        0.93,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color=TEXT_COLOR,
        linespacing=1.45,
    )


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = [[str(value) for value in row] for row in df.to_numpy().tolist()]
    widths = [
        max(len(str(header)), *(len(row[idx]) for row in rows)) if rows else len(str(header))
        for idx, header in enumerate(headers)
    ]

    def _format_row(values: list[str]) -> str:
        cells = [value.ljust(width) for value, width in zip(values, widths, strict=False)]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [_format_row([str(header) for header in headers]), separator]
    lines.extend(_format_row(row) for row in rows)
    return "\n".join(lines)


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def histogram_bins(series_list: list[pd.Series], log_scale: bool) -> np.ndarray:
    arrays = [np.asarray(s.dropna(), dtype=float) for s in series_list if len(s.dropna()) > 0]
    if not arrays:
        return np.linspace(0, 1, 11)

    values = np.concatenate(arrays)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.linspace(0, 1, 11)

    vmin = float(values.min())
    vmax = float(values.max())
    if np.isclose(vmin, vmax):
        return np.linspace(vmin - 0.5, vmax + 0.5, 10)

    if log_scale:
        positive = values[values > 0]
        if positive.size >= 2:
            return np.geomspace(float(positive.min()), float(positive.max()), 20)

    bins = np.histogram_bin_edges(values, bins="fd")
    if bins.size < 8:
        bins = np.histogram_bin_edges(values, bins=12)
    if bins.size > 35:
        bins = np.histogram_bin_edges(values, bins=35)
    return bins


def series_summary(
    values: pd.Series,
    dataset_name: str,
    feature_name: str,
    source_column: str,
) -> dict[str, float | int | str]:
    clean = values.dropna()
    q1 = safe_float(clean.quantile(0.25)) if not clean.empty else float("nan")
    q3 = safe_float(clean.quantile(0.75)) if not clean.empty else float("nan")
    mean = safe_float(clean.mean()) if not clean.empty else float("nan")
    std = safe_float(clean.std(ddof=1)) if len(clean) > 1 else float("nan")

    return {
        "dataset": dataset_name,
        "feature": feature_name,
        "source_column": source_column,
        "n_total": int(len(values)),
        "n_non_missing": int(clean.size),
        "n_missing": int(values.isna().sum()),
        "missing_rate": safe_float(values.isna().mean()),
        "mean": mean,
        "median": safe_float(clean.median()) if not clean.empty else float("nan"),
        "std": std,
        "cv": safe_float(std / mean) if mean not in (0, np.nan) else float("nan"),
        "min": safe_float(clean.min()) if not clean.empty else float("nan"),
        "q25": q1,
        "q75": q3,
        "iqr": safe_float(q3 - q1),
        "max": safe_float(clean.max()) if not clean.empty else float("nan"),
        "p05": safe_float(clean.quantile(0.05)) if not clean.empty else float("nan"),
        "p95": safe_float(clean.quantile(0.95)) if not clean.empty else float("nan"),
        "skew": safe_float(clean.skew()) if len(clean) > 2 else float("nan"),
        "zero_fraction": safe_float((clean == 0).mean()) if not clean.empty else float("nan"),
    }


def dataset1_within_tsid_summary(df: pd.DataFrame, feature_name: str, column: str) -> tuple[dict[str, float | int | str], pd.Series]:
    grouped_std = df.groupby("tsid")[column].std(ddof=1)
    valid = grouped_std.dropna()
    overall_std = df[column].dropna().std(ddof=1)
    q1 = safe_float(valid.quantile(0.25)) if not valid.empty else float("nan")
    q3 = safe_float(valid.quantile(0.75)) if not valid.empty else float("nan")

    summary = {
        "feature": feature_name,
        "source_column": column,
        "n_tsid_total": int(df["tsid"].nunique()),
        "n_tsid_valid": int(valid.size),
        "within_std_mean": safe_float(valid.mean()) if not valid.empty else float("nan"),
        "within_std_median": safe_float(valid.median()) if not valid.empty else float("nan"),
        "within_std_q25": q1,
        "within_std_q75": q3,
        "within_std_iqr": safe_float(q3 - q1),
        "within_std_max": safe_float(valid.max()) if not valid.empty else float("nan"),
        "within_std_to_overall_ratio": safe_float(valid.median() / overall_std)
        if not valid.empty and pd.notna(overall_std) and overall_std != 0
        else float("nan"),
        "share_below_half_overall_std": safe_float((valid < 0.5 * overall_std).mean())
        if not valid.empty and pd.notna(overall_std)
        else float("nan"),
    }
    return summary, valid


def build_summaries(dataset1: pd.DataFrame, tailake: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
    summary_rows: list[dict[str, float | int | str]] = []
    within_rows: list[dict[str, float | int | str]] = []
    within_lookup: dict[str, pd.Series] = {}
    shared_rows: list[dict[str, float | int | str]] = []

    for spec in FEATURE_SPECS:
        d1_summary: dict[str, float | int | str] | None = None
        tl_summary: dict[str, float | int | str] | None = None

        if spec.dataset1_col:
            d1_summary = series_summary(dataset1[spec.dataset1_col], "dataset1", spec.name, spec.dataset1_col)
            summary_rows.append(d1_summary)
            within_stats, within_values = dataset1_within_tsid_summary(dataset1, spec.name, spec.dataset1_col)
            within_rows.append(within_stats)
            within_lookup[spec.name] = within_values

        if spec.tailake_col:
            tl_summary = series_summary(tailake[spec.tailake_col], "tailake", spec.name, spec.tailake_col)
            summary_rows.append(tl_summary)

        if d1_summary and tl_summary:
            shared_rows.append(
                {
                    "feature": spec.name,
                    "dataset1_column": spec.dataset1_col,
                    "tailake_column": spec.tailake_col,
                    "dataset1_mean": d1_summary["mean"],
                    "tailake_mean": tl_summary["mean"],
                    "dataset1_median": d1_summary["median"],
                    "tailake_median": tl_summary["median"],
                    "dataset1_std": d1_summary["std"],
                    "tailake_std": tl_summary["std"],
                    "std_ratio_dataset1_to_tailake": safe_float(d1_summary["std"] / tl_summary["std"])
                    if pd.notna(d1_summary["std"]) and pd.notna(tl_summary["std"]) and tl_summary["std"] != 0
                    else float("nan"),
                    "dataset1_iqr": d1_summary["iqr"],
                    "tailake_iqr": tl_summary["iqr"],
                    "iqr_ratio_dataset1_to_tailake": safe_float(d1_summary["iqr"] / tl_summary["iqr"])
                    if pd.notna(d1_summary["iqr"]) and pd.notna(tl_summary["iqr"]) and tl_summary["iqr"] != 0
                    else float("nan"),
                }
            )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(within_rows),
        pd.DataFrame(shared_rows),
        within_lookup,
    )


def summary_lines_for_dataset(label: str, row: pd.Series | None) -> list[str]:
    if row is None:
        return [f"{label}: not available in this dataset."]
    return [
        f"{label}: n={fmt(row['n_non_missing'], 0)} / {fmt(row['n_total'], 0)}",
        f"Missing={fmt(100 * row['missing_rate'], 1)}%",
        f"Mean={fmt(row['mean'])}, median={fmt(row['median'])}",
        f"Std={fmt(row['std'])}, IQR={fmt(row['iqr'])}",
        f"Range={fmt(row['min'])} to {fmt(row['max'])}",
    ]


def plot_feature(
    spec: FeatureSpec,
    dataset1: pd.DataFrame,
    tailake: pd.DataFrame,
    summary_df: pd.DataFrame,
    within_lookup: dict[str, pd.Series],
) -> Path:
    d1_row = None
    tl_row = None
    if spec.dataset1_col:
        d1_row = summary_df[(summary_df["dataset"] == "dataset1") & (summary_df["feature"] == spec.name)].iloc[0]
    if spec.tailake_col:
        tl_row = summary_df[(summary_df["dataset"] == "tailake") & (summary_df["feature"] == spec.name)].iloc[0]

    d1_values = dataset1[spec.dataset1_col].dropna() if spec.dataset1_col else pd.Series(dtype=float)
    tl_values = tailake[spec.tailake_col].dropna() if spec.tailake_col else pd.Series(dtype=float)
    within_values = within_lookup.get(spec.name, pd.Series(dtype=float))

    fig = plt.figure(figsize=(13, 8), facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.28)
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_box = fig.add_subplot(gs[0, 1])
    ax_aux = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[1, 1])

    fig.suptitle(f"{spec.name} Input Structure", x=0.06, y=0.98, ha="left", fontsize=18, fontweight="bold", color=TEXT_COLOR)
    subtitle = {
        "shared": "Cross-dataset comparison for the shared modeling backbone.",
        "dataset1_only": "Dataset1-only model input from the 7-feature formal backbone.",
        "tailake_only": "Tai Lake-only model input from the 9-feature small dataset.",
    }[spec.role]
    fig.text(0.06, 0.94, subtitle, ha="left", va="top", fontsize=11, color=MUTED_TEXT)

    histogram_series = [s for s in [d1_values, tl_values] if not s.empty]
    bins = histogram_bins(histogram_series, log_scale=spec.log_scale)

    if not d1_values.empty:
        ax_hist.hist(
            d1_values,
            bins=bins,
            density=True,
            alpha=0.48,
            color=DATASET1_COLOR,
            edgecolor="white",
            linewidth=0.8,
            label=f"Dataset1 (n={len(d1_values)})",
        )
    if not tl_values.empty:
        ax_hist.hist(
            tl_values,
            bins=bins,
            density=True,
            alpha=0.42,
            color=TAILAKE_COLOR,
            edgecolor="white",
            linewidth=0.8,
            label=f"Tai Lake (n={len(tl_values)})",
        )
    if spec.log_scale:
        ax_hist.set_xscale("log")
    style_axes(ax_hist)
    ax_hist.set_title("Overall Distribution", loc="left", pad=8, color=TEXT_COLOR, fontweight="bold")
    ax_hist.set_xlabel(axis_label(spec))
    ax_hist.set_ylabel("Density")
    if not d1_values.empty or not tl_values.empty:
        ax_hist.legend(frameon=False, loc="upper right")

    box_data: list[np.ndarray] = []
    box_labels: list[str] = []
    box_colors: list[str] = []
    if not d1_values.empty:
        box_data.append(d1_values.to_numpy(dtype=float))
        box_labels.append("Dataset1")
        box_colors.append(DATASET1_COLOR)
    if not tl_values.empty:
        box_data.append(tl_values.to_numpy(dtype=float))
        box_labels.append("Tai Lake")
        box_colors.append(TAILAKE_COLOR)
    if box_data:
        bplot = ax_box.boxplot(box_data, vert=False, tick_labels=box_labels, patch_artist=True)
        for patch, color in zip(bplot["boxes"], box_colors, strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
            patch.set_edgecolor(color)
        for median in bplot["medians"]:
            median.set_color(TEXT_COLOR)
            median.set_linewidth(1.6)
        if spec.log_scale:
            ax_box.set_xscale("log")
        style_axes(ax_box)
        ax_box.set_title("Location And Spread", loc="left", pad=8, color=TEXT_COLOR, fontweight="bold")
        ax_box.set_xlabel(axis_label(spec))

    if spec.dataset1_col and not within_values.empty:
        aux_bins = histogram_bins([within_values], log_scale=False)
        ax_aux.hist(
            within_values,
            bins=aux_bins,
            color=ACCENT_COLOR,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.8,
        )
        ax_aux.axvline(within_values.median(), color=TEXT_COLOR, linestyle="--", linewidth=1.4)
        style_axes(ax_aux)
        ax_aux.set_title("Dataset1 Within-tsid Std", loc="left", pad=8, color=TEXT_COLOR, fontweight="bold")
        ax_aux.set_xlabel(f"Within-tsid std of {axis_label(spec)}")
        ax_aux.set_ylabel("tsid count")
    else:
        add_card(
            ax_aux,
            [
                "No Dataset1 within-tsid panel for this input.",
                "This feature belongs only to the Tai Lake dataset.",
            ],
        )

    lines = [f"Role: {spec.role.replace('_', ' ')}"]
    if spec.unit:
        lines.append(f"Unit: {spec.unit}")
    lines.append("")
    lines.extend(summary_lines_for_dataset("Dataset1", d1_row))
    lines.append("")
    lines.extend(summary_lines_for_dataset("Tai Lake", tl_row))

    if d1_row is not None and tl_row is not None:
        std_ratio = safe_float(d1_row["std"] / tl_row["std"]) if tl_row["std"] not in (0, np.nan) else float("nan")
        iqr_ratio = safe_float(d1_row["iqr"] / tl_row["iqr"]) if tl_row["iqr"] not in (0, np.nan) else float("nan")
        lines.extend(
            [
                "",
                f"Std ratio D1/TL = {fmt(std_ratio)}",
                f"IQR ratio D1/TL = {fmt(iqr_ratio)}",
            ]
        )
    if spec.dataset1_col and spec.name in within_lookup and not within_values.empty and d1_row is not None:
        ratio = safe_float(within_values.median() / d1_row["std"]) if d1_row["std"] not in (0, np.nan) else float("nan")
        lines.extend(
            [
                "",
                f"Within-tsid median std = {fmt(within_values.median())}",
                f"Within/overall std ratio = {fmt(ratio)}",
            ]
        )
    add_card(ax_text, lines)

    plot_path = PLOTS_DIR / f"{slugify(spec.name)}.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def key_takeaways(summary_df: pd.DataFrame, within_df: pd.DataFrame, shared_df: pd.DataFrame) -> list[str]:
    shared_sorted = shared_df.sort_values("std_ratio_dataset1_to_tailake")
    narrower = shared_sorted.iloc[0]
    wider = shared_sorted.iloc[-1]

    within_top = within_df.sort_values("within_std_to_overall_ratio")
    most_stable = within_top.iloc[0]
    least_stable = within_top.iloc[-1]

    d1_only = summary_df[(summary_df["dataset"] == "dataset1") & (summary_df["feature"].isin(["Cl2 dose", "Contact time"]))]
    tl_only = summary_df[(summary_df["dataset"] == "tailake") & (summary_df["feature"].isin(["COD", "NH4-N", "NO2-N", "NO3-N"]))]

    takeaways = [
        (
            f"Among the 5 shared inputs, Dataset1 is narrowest relative to Tai Lake for "
            f"{narrower['feature']} (std ratio D1/TL = {fmt(narrower['std_ratio_dataset1_to_tailake'])}) "
            f"and widest for {wider['feature']} (std ratio D1/TL = {fmt(wider['std_ratio_dataset1_to_tailake'])})."
        ),
        (
            f"Dataset1 within-tsid variability is smallest for {most_stable['feature']} "
            f"(median within/overall std ratio = {fmt(most_stable['within_std_to_overall_ratio'])}) "
            f"and largest for {least_stable['feature']} "
            f"(ratio = {fmt(least_stable['within_std_to_overall_ratio'])})."
        ),
        (
            f"Dataset1-only inputs remain strongly right-skewed: Cl2 dose std = "
            f"{fmt(d1_only.loc[d1_only['feature'] == 'Cl2 dose', 'std'].iloc[0])} mg/L and "
            f"contact time std = {fmt(d1_only.loc[d1_only['feature'] == 'Contact time', 'std'].iloc[0])} min."
        ),
        (
            f"Tai Lake-only chemistry inputs are fully observed in this table; the widest spread among them is "
            f"{tl_only.sort_values('std', ascending=False).iloc[0]['feature']} "
            f"(std = {fmt(tl_only.sort_values('std', ascending=False).iloc[0]['std'])})."
        ),
    ]
    return takeaways


def write_markdown(
    summary_df: pd.DataFrame,
    within_df: pd.DataFrame,
    shared_df: pd.DataFrame,
    plot_paths: dict[str, Path],
) -> None:
    shared_table = shared_df.copy()
    for col in [
        "dataset1_mean",
        "tailake_mean",
        "dataset1_median",
        "tailake_median",
        "dataset1_std",
        "tailake_std",
        "std_ratio_dataset1_to_tailake",
        "dataset1_iqr",
        "tailake_iqr",
        "iqr_ratio_dataset1_to_tailake",
    ]:
        shared_table[col] = shared_table[col].map(fmt)

    within_table = within_df.copy()
    for col in [
        "within_std_mean",
        "within_std_median",
        "within_std_q25",
        "within_std_q75",
        "within_std_iqr",
        "within_std_max",
        "within_std_to_overall_ratio",
        "share_below_half_overall_std",
    ]:
        within_table[col] = within_table[col].map(fmt)

    md_lines = [
        "# Input Structure Summary",
        "",
        "Scope:",
        "- Tai Lake / small dataset: 9 README inputs (`pH`, `COD`, `NH4-N`, `NO2-N`, `NO3-N`, `Bromide`, `TOC`, `UV254`, `Temperature`).",
        "- Dataset1: 7-feature formal backbone (`pH`, `UV254`, `Temperature`, `TOC`, `Bromide`, `Cl2 dose`, `Contact time`).",
        "- Bromide raw values sit on very different absolute scales across the two tables; treat the raw D1/TL bromide ratios as a structural flag first, and verify unit alignment before making a literal magnitude claim.",
        "",
        "## Key Takeaways",
    ]
    md_lines.extend([f"- {line}" for line in key_takeaways(summary_df, within_df, shared_df)])
    md_lines.extend(
        [
            "",
            "## Shared Inputs",
            "",
            dataframe_to_markdown(shared_table),
            "",
            "## Dataset1 Within-tsid Structure",
            "",
            dataframe_to_markdown(within_table),
            "",
            "## Plot Inventory",
        ]
    )
    md_lines.extend([f"- `{feature}`: `{path.name}`" for feature, path in plot_paths.items()])
    SUMMARY_MD.write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset1 = pd.read_csv(DATASET1_PATH)
    tailake = pd.read_csv(TAILAKE_PATH)

    summary_df, within_df, shared_df, within_lookup = build_summaries(dataset1, tailake)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    within_df.to_csv(WITHIN_TSID_CSV, index=False)
    shared_df.to_csv(SHARED_CSV, index=False)

    plot_paths: dict[str, Path] = {}
    for spec in FEATURE_SPECS:
        plot_paths[spec.name] = plot_feature(spec, dataset1, tailake, summary_df, within_lookup)

    write_markdown(summary_df, within_df, shared_df, plot_paths)

    print(f"Saved summary CSV to: {SUMMARY_CSV}")
    print(f"Saved within-tsid CSV to: {WITHIN_TSID_CSV}")
    print(f"Saved shared comparison CSV to: {SHARED_CSV}")
    print(f"Saved markdown summary to: {SUMMARY_MD}")
    print(f"Saved {len(plot_paths)} feature plots to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
