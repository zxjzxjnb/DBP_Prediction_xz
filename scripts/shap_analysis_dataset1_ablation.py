"""SHAP analysis for the dataset1 formal ablation experiments.

This script compares the baseline 5-feature formal run against the
6-feature formal run that adds ``cl2d_in_avg``.

Outputs:
  - Per experiment / model / target SHAP plots
  - Cross-model importance plots within each experiment
  - Cross-experiment importance plots for each model/target pair
  - A tidy CSV/JSON summary of mean absolute SHAP values

Usage:
    python scripts/shap_analysis_dataset1_ablation.py
"""

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
import torch


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT / "results" / "shap_analysis_dataset1_ablation"

MODEL_ORDER = ["rf", "xgb", "mlp", "kan"]
MODEL_LABELS = {
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "mlp": "MLP",
    "kan": "KAN",
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
}


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    label: str
    run_dir: Path


DEFAULT_EXPERIMENTS = [
    ExperimentSpec(
        key="formal_5feat",
        label="Formal 5-feature",
        run_dir=PROJECT / "checkpoints" / "formal_dataset1_5feat_avg" / "20260330T235734Z",
    ),
    ExperimentSpec(
        key="formal_6feat_cl2d",
        label="Formal 6-feature + Cl2",
        run_dir=PROJECT / "checkpoints" / "formal_dataset1_6feat_cl2d_avg" / "20260331T130339Z",
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
        "--nn-nsamples",
        type=int,
        default=200,
        help="KernelExplainer nsamples value for MLP/KAN.",
    )
    parser.add_argument(
        "--nn-background-k",
        type=int,
        default=30,
        help="Number of kmeans centroids used as KernelExplainer background.",
    )
    return parser.parse_args()


def feature_label(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


def load_checkpoint(path: Path) -> dict[str, Any]:
    if path.suffix == ".joblib":
        return joblib.load(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def load_dataset_for_run(run_dir: Path) -> pd.DataFrame:
    snapshot_path = run_dir / "dataset_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    dataset_path = Path(snapshot["path"])
    return pd.read_csv(dataset_path)


def get_checkpoint_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "rf": run_dir / "rf_tuned_checkpoint.joblib",
        "xgb": run_dir / "xgb_tuned_checkpoint.joblib",
        "mlp": run_dir / "mlp_tuned_checkpoint.pt",
        "kan": run_dir / "kan_tuned_checkpoint.pt",
    }


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


def build_predict_fn(member: dict[str, Any], model_family: str):
    scaler_x = member["scaler_x"]
    model_state = member["model_state"]

    if model_family in ("random_forest", "xgboost"):
        estimator = model_state["estimator"]

        def predict_fn(x_scaled: np.ndarray) -> np.ndarray:
            return estimator.predict(x_scaled).reshape(-1, 1)

        def prepare_inputs(x_raw: np.ndarray) -> np.ndarray:
            return scaler_x.transform(x_raw)

        return prepare_inputs, predict_fn

    in_dim = member["in_dim"]
    out_dim = member["out_dim"]
    model_params = member["model_params"]
    seed = member["seed"]

    if model_family == "mlp":
        from dbp_prediction.models.mlp import MLP

        model = MLP(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=list(model_params.get("hidden_dims", [32, 16])),
            dropout=float(model_params.get("dropout", 0.0)),
            activation=str(model_params.get("activation", "ReLU")),
        )
    elif model_family == "kan":
        from dbp_prediction.models.kan import build_kan

        model = build_kan(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=list(model_params.get("hidden_dims", [32, 16])),
            grid=int(model_params.get("grid", 3)),
            k=int(model_params.get("k", 5)),
            base_fun=str(model_params.get("base_fun", "silu")),
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    model.load_state_dict(model_state)
    model.eval()

    def predict_fn(x_scaled: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            y_pred = model(torch.from_numpy(x_scaled.astype(np.float32))).numpy()
        return y_pred.reshape(-1, 1)

    def prepare_inputs(x_raw: np.ndarray) -> np.ndarray:
        return scaler_x.transform(x_raw)

    return prepare_inputs, predict_fn


def compute_shap_tree(members: list[dict[str, Any]], x_raw: np.ndarray) -> np.ndarray:
    shap_values_all = []
    for member in members:
        prepare_inputs, _ = build_predict_fn(member, "random_forest")
        estimator = member["model_state"]["estimator"]
        x_scaled = prepare_inputs(x_raw)
        explainer = shap.TreeExplainer(estimator)
        shap_values_all.append(np.asarray(explainer.shap_values(x_scaled)))
    return np.mean(np.stack(shap_values_all, axis=0), axis=0)


def compute_shap_xgb(members: list[dict[str, Any]], x_raw: np.ndarray) -> np.ndarray:
    shap_values_all = []
    for member in members:
        prepare_inputs, _ = build_predict_fn(member, "xgboost")
        estimator = member["model_state"]["estimator"]
        x_scaled = prepare_inputs(x_raw)
        explainer = shap.TreeExplainer(estimator)
        shap_values_all.append(np.asarray(explainer.shap_values(x_scaled)))
    return np.mean(np.stack(shap_values_all, axis=0), axis=0)


def compute_shap_nn(
    members: list[dict[str, Any]],
    x_raw: np.ndarray,
    model_family: str,
    background_k: int,
    nsamples: int,
) -> np.ndarray:
    shap_values_all = []
    for fold_idx, member in enumerate(members, start=1):
        prepare_inputs, predict_fn = build_predict_fn(member, model_family)
        x_scaled = prepare_inputs(x_raw)
        n_background = min(background_k, len(x_scaled))
        background = shap.kmeans(x_scaled, n_background)
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = np.asarray(
            explainer.shap_values(x_scaled, nsamples=nsamples, silent=True)
        )
        shap_values_all.append(shap_values)
        print(f"      Fold {fold_idx}/{len(members)} complete")
    return np.mean(np.stack(shap_values_all, axis=0), axis=0)


def compute_shap_values(
    checkpoint: dict[str, Any],
    target_col: str,
    x_raw: np.ndarray,
    background_k: int,
    nsamples: int,
) -> np.ndarray:
    model_family = checkpoint["model_family"]
    members = checkpoint["target_payloads"][target_col]["members"]
    if model_family == "random_forest":
        return compute_shap_tree(members, x_raw)
    if model_family == "xgboost":
        return compute_shap_xgb(members, x_raw)
    return compute_shap_nn(members, x_raw, model_family, background_k, nsamples)


def ensure_2d_shap(shap_values: np.ndarray) -> np.ndarray:
    arr = np.asarray(shap_values)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D SHAP values, got shape {arr.shape}")
    return arr


def plot_summary_beeswarm(
    shap_values: np.ndarray,
    x_display: pd.DataFrame,
    title: str,
    save_path: Path,
) -> None:
    plt.figure(figsize=(8, max(4, 0.55 * x_display.shape[1] + 1)))
    shap.summary_plot(shap_values, x_display, show=False, plot_size=None)
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")


def plot_bar(
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


def plot_dependence_top3(
    shap_values: np.ndarray,
    x_raw_df: pd.DataFrame,
    feature_cols: list[str],
    title_prefix: str,
    save_dir: Path,
) -> None:
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs)[-3:][::-1]
    raw_display = x_raw_df.rename(columns={col: feature_label(col) for col in feature_cols})
    feature_names = list(raw_display.columns)
    for rank, idx in enumerate(top_indices, start=1):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        shap.dependence_plot(
            idx,
            shap_values,
            raw_display,
            feature_names=feature_names,
            interaction_index="auto",
            show=False,
            ax=ax,
        )
        ax.set_title(
            f"{title_prefix} - {feature_names[idx]}",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(save_dir / f"dependence_top{rank}_{feature_cols[idx]}.png")
        plt.close("all")


def plot_cross_model_importance(
    shap_by_model: dict[str, dict[str, np.ndarray]],
    feature_cols: list[str],
    target_col: str,
    title: str,
    save_path: Path,
) -> None:
    x = np.arange(len(feature_cols))
    width = 0.2
    plt.figure(figsize=(10, 5))
    for idx, model_name in enumerate(MODEL_ORDER):
        target_data = shap_by_model.get(model_name, {})
        if target_col not in target_data:
            continue
        mean_abs = np.abs(target_data[target_col]).mean(axis=0)
        plt.bar(
            x + idx * width,
            mean_abs,
            width,
            label=MODEL_LABELS[model_name],
        )
    offset = width * (len(MODEL_ORDER) - 1) / 2
    plt.xticks(x + offset, [feature_label(col) for col in feature_cols], rotation=30, ha="right")
    plt.ylabel("Mean |SHAP value|")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")


def plot_cross_experiment_importance(
    summary_df: pd.DataFrame,
    model_name: str,
    target_col: str,
    save_path: Path,
) -> None:
    subset = summary_df[
        (summary_df["model"] == model_name) & (summary_df["target"] == target_col)
    ].copy()
    if subset.empty:
        return

    feature_order = (
        subset.groupby("feature")["mean_abs_shap"]
        .max()
        .sort_values(ascending=False)
        .index.tolist()
    )
    label_map = (
        subset[["experiment", "experiment_label"]]
        .drop_duplicates()
        .set_index("experiment")["experiment_label"]
        .to_dict()
    )
    wide = (
        subset.pivot(index="feature", columns="experiment", values="mean_abs_shap")
        .reindex(feature_order)
        .fillna(0.0)
    )
    plt.figure(figsize=(9, max(3.5, 0.5 * len(wide) + 1)))
    y = np.arange(len(wide))
    width = 0.35
    experiments = list(wide.columns)
    for idx, experiment_key in enumerate(experiments):
        plt.barh(
            y + idx * width,
            wide[experiment_key].values,
            width,
            label=label_map.get(experiment_key, experiment_key),
        )
    plt.yticks(
        y + width * (len(experiments) - 1) / 2,
        [feature_label(col) for col in wide.index],
    )
    plt.xlabel("Mean |SHAP value|")
    plt.title(
        f"{MODEL_LABELS[model_name]} - {TARGET_LABELS[target_col]} - Cross Experiment",
        fontsize=13,
        fontweight="bold",
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")


def analyze_experiment(
    spec: ExperimentSpec,
    max_test_samples: int,
    nn_background_k: int,
    nn_nsamples: int,
) -> list[dict[str, Any]]:
    print(f"\n{'=' * 80}")
    print(f"SHAP analysis: {spec.label}")
    print(f"Run dir: {spec.run_dir}")
    print(f"{'=' * 80}")

    df = load_dataset_for_run(spec.run_dir)
    checkpoint_paths = get_checkpoint_paths(spec.run_dir)
    exp_dir = OUTPUT_DIR / spec.key
    exp_dir.mkdir(parents=True, exist_ok=True)

    shap_by_model: dict[str, dict[str, np.ndarray]] = {}
    summary_rows: list[dict[str, Any]] = []

    for model_name in MODEL_ORDER:
        checkpoint_path = checkpoint_paths[model_name]
        checkpoint = load_checkpoint(checkpoint_path)
        feature_cols = checkpoint["feature_cols"]
        target_cols = checkpoint["target_cols"]
        model_family = checkpoint["model_family"]
        model_dir = exp_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        shap_by_model[model_name] = {}

        print(f"\n--- {MODEL_LABELS[model_name]} ({model_family}) ---")
        print(f"Features: {feature_cols}")

        for target_col in target_cols:
            print(f"  Target: {target_col}")
            test_subset = filter_test_rows(df, feature_cols, target_col, max_test_samples)
            x_raw_df = test_subset[feature_cols].copy()
            x_raw = x_raw_df.to_numpy(dtype=np.float64)
            shap_values = ensure_2d_shap(
                compute_shap_values(
                    checkpoint=checkpoint,
                    target_col=target_col,
                    x_raw=x_raw,
                    background_k=nn_background_k,
                    nsamples=nn_nsamples,
                )
            )
            shap_by_model[model_name][target_col] = shap_values

            target_dir = model_dir / target_col
            target_dir.mkdir(parents=True, exist_ok=True)
            np.save(target_dir / "shap_values.npy", shap_values)

            x_display = x_raw_df.rename(columns={col: feature_label(col) for col in feature_cols})
            plot_summary_beeswarm(
                shap_values,
                x_display,
                f"{spec.label} - {MODEL_LABELS[model_name]} - {TARGET_LABELS[target_col]}",
                target_dir / "summary_beeswarm.png",
            )
            mean_abs = plot_bar(
                shap_values,
                feature_cols,
                f"{spec.label} - {MODEL_LABELS[model_name]} - {TARGET_LABELS[target_col]}",
                target_dir / "bar_importance.png",
            )
            plot_dependence_top3(
                shap_values,
                x_raw_df,
                feature_cols,
                f"{spec.label} - {MODEL_LABELS[model_name]} - {TARGET_LABELS[target_col]}",
                target_dir,
            )

            for rank, (feature_name, value) in enumerate(mean_abs.items(), start=1):
                summary_rows.append(
                    {
                        "experiment": spec.key,
                        "experiment_label": spec.label,
                        "model": model_name,
                        "model_label": MODEL_LABELS[model_name],
                        "target": target_col,
                        "target_label": TARGET_LABELS.get(target_col, target_col),
                        "feature": feature_name,
                        "feature_label": feature_label(feature_name),
                        "mean_abs_shap": float(value),
                        "rank": rank,
                        "test_rows": int(len(test_subset)),
                    }
                )

        for target_col in target_cols:
            plot_cross_model_importance(
                shap_by_model,
                feature_cols,
                target_col,
                f"{spec.label} - Cross-model - {TARGET_LABELS[target_col]}",
                exp_dir / f"cross_model_{target_col}.png",
            )

    return summary_rows


def write_summary_outputs(summary_rows: list[dict[str, Any]]) -> pd.DataFrame:
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["experiment", "model", "target", "rank"]
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_DIR / "mean_abs_shap_summary.csv", index=False)
    (OUTPUT_DIR / "mean_abs_shap_summary.json").write_text(
        json.dumps(summary_rows, indent=2)
    )
    return summary_df


def write_top_feature_report(summary_df: pd.DataFrame) -> None:
    lines = ["# Dataset1 Formal Ablation SHAP Summary", ""]
    for experiment in summary_df["experiment"].drop_duplicates():
        subset_exp = summary_df[summary_df["experiment"] == experiment]
        lines.append(f"## {subset_exp['experiment_label'].iloc[0]}")
        lines.append("")
        for model_name in MODEL_ORDER:
            subset_model = subset_exp[subset_exp["model"] == model_name]
            if subset_model.empty:
                continue
            lines.append(f"### {MODEL_LABELS[model_name]}")
            for target_col in subset_model["target"].drop_duplicates():
                top3 = subset_model[subset_model["target"] == target_col].head(3)
                joined = ", ".join(
                    f"{row.feature_label} ({row.mean_abs_shap:.4f})"
                    for row in top3.itertuples()
                )
                lines.append(f"- {TARGET_LABELS[target_col]}: {joined}")
            lines.append("")
    (OUTPUT_DIR / "top_features.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for spec in DEFAULT_EXPERIMENTS:
        summary_rows.extend(
            analyze_experiment(
                spec=spec,
                max_test_samples=args.max_test_samples,
                nn_background_k=args.nn_background_k,
                nn_nsamples=args.nn_nsamples,
            )
        )

    summary_df = write_summary_outputs(summary_rows)
    write_top_feature_report(summary_df)

    for model_name in MODEL_ORDER:
        for target_col in TARGET_LABELS:
            plot_cross_experiment_importance(
                summary_df,
                model_name=model_name,
                target_col=target_col,
                save_path=OUTPUT_DIR / f"cross_experiment_{model_name}_{target_col}.png",
            )

    print(f"\n{'=' * 80}")
    print(f"SHAP outputs saved to: {OUTPUT_DIR}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
