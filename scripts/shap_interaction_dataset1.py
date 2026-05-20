"""SHAP interaction analysis for Dataset1 7-feature tree models.

Targets covered:
  - THM4  -> Random Forest from the 7-feature best script
  - BDCM  -> Random Forest from the 7-feature best script
  - DBCM  -> best available tree model (RF or XGB) from the DBCM formal run

Outputs:
  - Per-target 7x7 mean |SHAP interaction| heatmap
  - Per-target Cl2 dose SHAP dependence plot colored by Bromide
  - Per-target Bromide SHAP dependence plot colored by TOC
  - Per-target Bromide x Cl2 dose scatter colored by pure Cl2-Bromide interaction
  - Per-target and combined CSV summaries of top interacting pairs

Usage:
    python scripts/shap_interaction_dataset1.py
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="shap")
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but StandardScaler was fitted with feature names",
)

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.colors import TwoSlopeNorm


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT / "results" / "shap_interaction_dataset1"

TREE_MODEL_LABELS = {
    "rf": "Random Forest",
    "xgb": "XGBoost",
}
MODEL_FAMILIES = {
    "rf": "random_forest",
    "xgb": "xgboost",
}
TARGET_LABELS = {
    "thm4_in_avg": "THM4",
    "dbcm_in_avg": "DBCM",
    "bdcm_in_avg": "BDCM",
}
FEATURE_LABELS = {
    "ph_in_avg": "pH",
    "uv_in_avg": "UV254",
    "temp_in_avg": "Temperature",
    "toc_in_avg": "TOC",
    "br_in_avg": "Bromide",
    "cl2d_in_avg": "Cl2 dose",
    "time_sds_avg": "Contact time",
}

CL2_COL = "cl2d_in_avg"
BR_COL = "br_in_avg"
TOC_COL = "toc_in_avg"


@dataclass(frozen=True)
class InteractionSpec:
    target: str
    label: str
    run_dir: Path
    model_name: str | None


DEFAULT_SPECS = [
    InteractionSpec(
        target="thm4_in_avg",
        label="THM4",
        run_dir=PROJECT / "checkpoints" / "formal_dataset1_7feat_cl2d_contact_time_thm4_avg" / "20260331T214748Z",
        model_name="rf",
    ),
    InteractionSpec(
        target="bdcm_in_avg",
        label="BDCM",
        run_dir=PROJECT / "checkpoints" / "formal_dataset1_7feat_cl2d_contact_time_bdcm_avg" / "20260331T214748Z",
        model_name="rf",
    ),
    InteractionSpec(
        target="dbcm_in_avg",
        label="DBCM",
        run_dir=PROJECT / "checkpoints" / "formal_dataset1_7feat_cl2d_contact_time_dbcm_avg" / "20260331T225500Z",
        model_name=None,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=0,
        help="Optional cap on the number of test rows per target. 0 means use all rows.",
    )
    parser.add_argument(
        "--top-pairs",
        type=int,
        default=21,
        help="Number of ranked off-diagonal feature pairs to include in each per-target CSV.",
    )
    return parser.parse_args()


def feature_label(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


def checkpoint_path_for(run_dir: Path, model_name: str) -> Path:
    return run_dir / f"{model_name}_tuned_checkpoint.joblib"


def load_checkpoint(path: Path) -> dict[str, Any]:
    return joblib.load(path)


def load_dataset_for_run(run_dir: Path) -> pd.DataFrame:
    snapshot = json.loads((run_dir / "dataset_snapshot.json").read_text())
    return pd.read_csv(Path(snapshot["path"]))


def filter_test_rows(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    max_test_samples: int,
) -> pd.DataFrame:
    required = feature_cols + [target_col]
    subset = df.loc[df["split"] == "test", required].dropna(axis=0, how="any")
    if max_test_samples and len(subset) > max_test_samples:
        subset = subset.iloc[:max_test_samples].copy()
    return subset


def choose_best_available_tree_model(run_dir: Path, target_col: str) -> str:
    candidates = [
        model_name
        for model_name in ("rf", "xgb")
        if checkpoint_path_for(run_dir, model_name).exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No RF/XGB tree checkpoints found in {run_dir}")

    comparison_path = run_dir / "metrics" / "model_comparison.json"
    if not comparison_path.exists():
        return candidates[0]

    comparison = json.loads(comparison_path.read_text())
    scored: list[tuple[float, str]] = []
    for model_name in candidates:
        metrics = comparison.get("models", {}).get(model_name, {})
        target_metrics = metrics.get("target_metrics", {}).get(target_col, {})
        rmse = target_metrics.get("rmse", metrics.get("macro_test_metrics", {}).get("rmse"))
        if rmse is not None:
            scored.append((float(rmse), model_name))

    if not scored:
        return candidates[0]
    return min(scored, key=lambda item: item[0])[1]


def normalize_interaction_values(values: Any) -> np.ndarray:
    if isinstance(values, list):
        if len(values) != 1:
            raise ValueError(f"Expected one SHAP interaction output, got {len(values)}")
        values = values[0]

    arr = np.asarray(values)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    elif arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    if arr.ndim != 3:
        raise ValueError(f"Expected interaction shape (n, f, f), got {arr.shape}")
    if arr.shape[1] != arr.shape[2]:
        raise ValueError(f"Expected square interaction matrices, got {arr.shape}")
    return arr


def compute_tree_interactions(members: list[dict[str, Any]], x_raw: np.ndarray) -> np.ndarray:
    interactions_all = []
    for fold_idx, member in enumerate(members, start=1):
        x_scaled = member["scaler_x"].transform(x_raw)
        estimator = member["model_state"]["estimator"]
        explainer = shap.TreeExplainer(estimator)
        interactions = normalize_interaction_values(
            explainer.shap_interaction_values(x_scaled)
        )
        interactions_all.append(interactions)
        print(f"    Fold {fold_idx}/{len(members)} complete")
    return np.mean(np.stack(interactions_all, axis=0), axis=0)


def shap_values_from_interactions(interactions: np.ndarray) -> np.ndarray:
    return interactions.sum(axis=2)


def pure_pair_interaction(interactions: np.ndarray, idx_a: int, idx_b: int) -> np.ndarray:
    return 0.5 * (interactions[:, idx_a, idx_b] + interactions[:, idx_b, idx_a])


def symmetric_mean_abs(interactions: np.ndarray) -> np.ndarray:
    mean_abs = np.abs(interactions).mean(axis=0)
    return 0.5 * (mean_abs + mean_abs.T)


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def plot_heatmap(
    mean_abs_matrix: np.ndarray,
    feature_cols: list[str],
    title: str,
    save_path: Path,
) -> None:
    labels = [feature_label(col) for col in feature_cols]
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    im = ax.imshow(mean_abs_matrix, cmap="viridis")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean |SHAP interaction value|")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    threshold = float(np.nanmax(mean_abs_matrix)) * 0.55 if mean_abs_matrix.size else 0.0
    for i in range(mean_abs_matrix.shape[0]):
        for j in range(mean_abs_matrix.shape[1]):
            value = mean_abs_matrix[i, j]
            text_color = "white" if value > threshold else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")


def plot_summary_beeswarm(
    shap_values: np.ndarray,
    x_raw_df: pd.DataFrame,
    feature_cols: list[str],
    title: str,
    save_path: Path,
) -> None:
    x_display = x_raw_df.rename(columns={col: feature_label(col) for col in feature_cols})
    plt.figure(figsize=(8, max(4, 0.55 * x_display.shape[1] + 1)))
    shap.summary_plot(shap_values, x_display, show=False, plot_size=None)
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")


def plot_shap_bar(
    shap_values: np.ndarray,
    feature_cols: list[str],
    title: str,
    save_path: Path,
) -> pd.Series:
    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols)
    ordered = mean_abs.sort_values(ascending=True)
    plt.figure(figsize=(8, max(3, 0.5 * len(feature_cols) + 1)))
    plt.barh(
        [feature_label(col) for col in ordered.index],
        ordered.values,
        color="#4E79A7",
    )
    plt.xlabel("Mean |SHAP value|")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")
    return mean_abs.sort_values(ascending=False)


def plot_dependence(
    x_raw_df: pd.DataFrame,
    shap_values: np.ndarray,
    feature_cols: list[str],
    feature_col: str,
    color_col: str,
    title: str,
    save_path: Path,
) -> None:
    feature_idx = feature_cols.index(feature_col)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    scatter = ax.scatter(
        x_raw_df[feature_col],
        shap_values[:, feature_idx],
        c=x_raw_df[color_col],
        cmap="viridis",
        s=36,
        alpha=0.82,
        edgecolors="none",
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(feature_label(color_col))
    ax.axhline(0, color="#6B7280", linewidth=0.8, alpha=0.65)
    ax.set_xlabel(feature_label(feature_col))
    ax.set_ylabel(f"{feature_label(feature_col)} SHAP value")
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")


def plot_cl2_br_interaction_scatter(
    x_raw_df: pd.DataFrame,
    interactions: np.ndarray,
    feature_cols: list[str],
    title: str,
    save_path: Path,
) -> None:
    cl2_idx = feature_cols.index(CL2_COL)
    br_idx = feature_cols.index(BR_COL)
    pure_interaction = pure_pair_interaction(interactions, cl2_idx, br_idx)
    abs_limit = float(np.nanmax(np.abs(pure_interaction))) if pure_interaction.size else 0.0
    norm = (
        None
        if abs_limit == 0.0
        else TwoSlopeNorm(vmin=-abs_limit, vcenter=0.0, vmax=abs_limit)
    )

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    scatter = ax.scatter(
        x_raw_df[BR_COL],
        x_raw_df[CL2_COL],
        c=pure_interaction,
        cmap="RdBu_r",
        norm=norm,
        s=44,
        alpha=0.86,
        edgecolors="#FFFFFF",
        linewidths=0.25,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Pure SHAP interaction: Cl2 dose x Bromide")
    ax.set_xlabel(feature_label(BR_COL))
    ax.set_ylabel(feature_label(CL2_COL))
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")


def build_pair_summary(
    interactions: np.ndarray,
    feature_cols: list[str],
    spec: InteractionSpec,
    model_name: str,
    run_dir: Path,
    top_pairs: int,
    test_rows: int,
) -> pd.DataFrame:
    mean_abs_matrix = symmetric_mean_abs(interactions)
    rows: list[dict[str, Any]] = []
    for idx_a, feature_a in enumerate(feature_cols):
        for idx_b in range(idx_a + 1, len(feature_cols)):
            feature_b = feature_cols[idx_b]
            signed_values = pure_pair_interaction(interactions, idx_a, idx_b)
            rows.append(
                {
                    "target": spec.target,
                    "target_label": spec.label,
                    "model": model_name,
                    "model_label": TREE_MODEL_LABELS[model_name],
                    "pair_type": "pairwise_interaction",
                    "feature_a": feature_a,
                    "feature_a_label": feature_label(feature_a),
                    "feature_b": feature_b,
                    "feature_b_label": feature_label(feature_b),
                    "mean_abs_interaction": float(mean_abs_matrix[idx_a, idx_b]),
                    "mean_signed_interaction": float(np.mean(signed_values)),
                    "test_rows": int(test_rows),
                    "run_dir": str(run_dir),
                }
            )

    summary_df = pd.DataFrame(rows).sort_values(
        ["mean_abs_interaction", "feature_a", "feature_b"],
        ascending=[False, True, True],
    )
    summary_df.insert(0, "rank", np.arange(1, len(summary_df) + 1))
    return summary_df.head(top_pairs).copy()


def analyze_target(
    spec: InteractionSpec,
    max_test_samples: int,
    top_pairs: int,
) -> pd.DataFrame:
    model_name = spec.model_name or choose_best_available_tree_model(spec.run_dir, spec.target)
    print(f"\n{'=' * 80}")
    print(f"SHAP interaction analysis: {spec.label}")
    print(f"Run dir: {spec.run_dir}")
    print(f"Model: {TREE_MODEL_LABELS[model_name]}")
    print(f"{'=' * 80}")

    checkpoint = load_checkpoint(checkpoint_path_for(spec.run_dir, model_name))
    model_family = checkpoint["model_family"]
    expected_family = MODEL_FAMILIES[model_name]
    if model_family != expected_family:
        raise ValueError(
            f"Expected {model_name} checkpoint family {expected_family}, got {model_family}"
        )

    feature_cols = checkpoint["feature_cols"]
    for required_col in (CL2_COL, BR_COL, TOC_COL):
        if required_col not in feature_cols:
            raise ValueError(f"{required_col} not found in checkpoint features: {feature_cols}")

    df = load_dataset_for_run(spec.run_dir)
    test_subset = filter_test_rows(df, feature_cols, spec.target, max_test_samples)
    x_raw_df = test_subset[feature_cols].copy()
    x_raw = x_raw_df.to_numpy(dtype=np.float64)
    members = checkpoint["target_payloads"][spec.target]["members"]

    interactions = compute_tree_interactions(members, x_raw)
    if interactions.shape[1] != len(feature_cols):
        raise ValueError(
            f"Feature count mismatch: interactions={interactions.shape}, features={len(feature_cols)}"
        )

    target_dir = OUTPUT_DIR / spec.target
    target_dir.mkdir(parents=True, exist_ok=True)
    np.save(target_dir / "shap_interaction_values.npy", interactions)
    np.save(target_dir / "shap_values_from_interactions.npy", shap_values_from_interactions(interactions))

    model_label = TREE_MODEL_LABELS[model_name]
    mean_abs_matrix = symmetric_mean_abs(interactions)
    plot_heatmap(
        mean_abs_matrix,
        feature_cols,
        f"{spec.label} - {model_label} - mean |SHAP interaction|",
        target_dir / "interaction_heatmap.png",
    )

    shap_values = shap_values_from_interactions(interactions)
    plot_summary_beeswarm(
        shap_values,
        x_raw_df,
        feature_cols,
        f"{spec.label} - {model_label} - SHAP beeswarm",
        target_dir / "shap_summary_beeswarm.png",
    )
    plot_shap_bar(
        shap_values,
        feature_cols,
        f"{spec.label} - {model_label} - mean |SHAP|",
        target_dir / "shap_bar_importance.png",
    )
    plot_dependence(
        x_raw_df,
        shap_values,
        feature_cols,
        feature_col=CL2_COL,
        color_col=BR_COL,
        title=f"{spec.label} - Cl2 dose SHAP colored by Bromide",
        save_path=target_dir / "dependence_cl2d_colored_by_bromide.png",
    )
    plot_dependence(
        x_raw_df,
        shap_values,
        feature_cols,
        feature_col=BR_COL,
        color_col=TOC_COL,
        title=f"{spec.label} - Bromide SHAP colored by TOC",
        save_path=target_dir / "dependence_bromide_colored_by_toc.png",
    )
    plot_cl2_br_interaction_scatter(
        x_raw_df,
        interactions,
        feature_cols,
        title=f"{spec.label} - pure Cl2 dose x Bromide interaction",
        save_path=target_dir / "scatter_bromide_cl2d_pure_interaction.png",
    )

    pair_summary = build_pair_summary(
        interactions=interactions,
        feature_cols=feature_cols,
        spec=spec,
        model_name=model_name,
        run_dir=spec.run_dir,
        top_pairs=top_pairs,
        test_rows=len(test_subset),
    )
    pair_summary.to_csv(target_dir / "top_interacting_pairs.csv", index=False)
    return pair_summary


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries = [
        analyze_target(
            spec=spec,
            max_test_samples=args.max_test_samples,
            top_pairs=args.top_pairs,
        )
        for spec in DEFAULT_SPECS
    ]
    combined = pd.concat(summaries, axis=0, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "top_interacting_pairs_summary.csv", index=False)

    print(f"\n{'=' * 80}")
    print(f"SHAP interaction outputs saved to: {OUTPUT_DIR}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
