"""SHAP analysis for the best 7-feature formal model of each dataset1 target.

Targets covered:
  - THM4  -> Random Forest
  - BDCM  -> Random Forest
  - DBCM  -> MLP

Outputs:
  - Per-target SHAP beeswarm plot
  - Per-target mean |SHAP| bar chart
  - Per-target top-3 dependence plots
  - A combined CSV / JSON summary of mean absolute SHAP values
  - A grouped cross-target importance figure

Usage:
    python scripts/shap_analysis_dataset1_7feat_best.py
    python scripts/shap_analysis_dataset1_7feat_best.py \
      --thm4-run-dir checkpoints/formal_dataset1_7feat_cl2d_contact_time_thm4_avg/<run_id> \
      --bdcm-run-dir checkpoints/formal_dataset1_7feat_cl2d_contact_time_bdcm_avg/<run_id> \
      --dbcm-run-dir checkpoints/formal_dataset1_7feat_cl2d_contact_time_dbcm_avg/<run_id>
"""

# ruff: noqa: E402

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
OUTPUT_DIR = PROJECT / "results" / "shap_analysis_dataset1_7feat_best"

MODEL_LABELS = {
    "rf": "Random Forest",
    "mlp": "MLP",
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


@dataclass(frozen=True)
class BestSpec:
    target: str
    label: str
    run_dir: Path
    model_name: str


DEFAULT_RUN_PARENTS = {
    "thm4": PROJECT / "checkpoints" / "formal_dataset1_7feat_cl2d_contact_time_thm4_avg",
    "bdcm": PROJECT / "checkpoints" / "formal_dataset1_7feat_cl2d_contact_time_bdcm_avg",
    "dbcm": PROJECT / "checkpoints" / "formal_dataset1_7feat_cl2d_contact_time_dbcm_avg",
}


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
        help="KernelExplainer nsamples value for neural models.",
    )
    parser.add_argument(
        "--nn-background-k",
        type=int,
        default=30,
        help="Number of kmeans centroids used as KernelExplainer background.",
    )
    parser.add_argument(
        "--thm4-run-dir",
        type=str,
        default=None,
        help="Run directory for the THM4 formal model. Defaults to the latest run in the THM4 checkpoint directory.",
    )
    parser.add_argument(
        "--bdcm-run-dir",
        type=str,
        default=None,
        help="Run directory for the BDCM formal model. Defaults to the latest run in the BDCM checkpoint directory.",
    )
    parser.add_argument(
        "--dbcm-run-dir",
        type=str,
        default=None,
        help="Run directory for the DBCM formal model. Defaults to the latest run in the DBCM checkpoint directory.",
    )
    return parser.parse_args()


def latest_run_dir(parent: Path) -> Path:
    if not parent.exists():
        raise FileNotFoundError(
            f"No checkpoint directory found at {parent}. Run the formal experiment first "
            "or pass an explicit --*-run-dir value."
        )
    candidates = sorted(path for path in parent.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(
            f"No run directories found under {parent}. Run the formal experiment first "
            "or pass an explicit --*-run-dir value."
        )
    return candidates[-1]


def resolve_run_dir(raw_path: str | None, default_parent: Path) -> Path:
    if raw_path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = PROJECT / path
        return path.resolve()
    return latest_run_dir(default_parent)


def build_specs(args: argparse.Namespace) -> list[BestSpec]:
    return [
        BestSpec(
            target="thm4_in_avg",
            label="THM4",
            run_dir=resolve_run_dir(args.thm4_run_dir, DEFAULT_RUN_PARENTS["thm4"]),
            model_name="rf",
        ),
        BestSpec(
            target="bdcm_in_avg",
            label="BDCM",
            run_dir=resolve_run_dir(args.bdcm_run_dir, DEFAULT_RUN_PARENTS["bdcm"]),
            model_name="rf",
        ),
        BestSpec(
            target="dbcm_in_avg",
            label="DBCM",
            run_dir=resolve_run_dir(args.dbcm_run_dir, DEFAULT_RUN_PARENTS["dbcm"]),
            model_name="mlp",
        ),
    ]


def feature_label(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


def load_checkpoint(path: Path) -> dict[str, Any]:
    if path.suffix == ".joblib":
        return joblib.load(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def checkpoint_path_for(run_dir: Path, model_name: str) -> Path:
    suffix = ".joblib" if model_name == "rf" else ".pt"
    return run_dir / f"{model_name}_tuned_checkpoint{suffix}"


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


def build_predict_fn(member: dict[str, Any], model_family: str):
    scaler_x = member["scaler_x"]
    model_state = member["model_state"]

    if model_family == "random_forest":
        estimator = model_state["estimator"]

        def prepare_inputs(x_raw: np.ndarray) -> np.ndarray:
            return scaler_x.transform(x_raw)

        def predict_fn(x_scaled: np.ndarray) -> np.ndarray:
            return estimator.predict(x_scaled).reshape(-1, 1)

        return prepare_inputs, predict_fn

    in_dim = member["in_dim"]
    out_dim = member["out_dim"]
    model_params = member["model_params"]

    from dbp_prediction.models.mlp import MLP

    model = MLP(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dims=list(model_params.get("hidden_dims", [32, 16])),
        dropout=float(model_params.get("dropout", 0.0)),
        activation=str(model_params.get("activation", "ReLU")),
    )
    model.load_state_dict(model_state)
    model.eval()

    def prepare_inputs(x_raw: np.ndarray) -> np.ndarray:
        return scaler_x.transform(x_raw)

    def predict_fn(x_scaled: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            y_pred = model(torch.from_numpy(x_scaled.astype(np.float32))).numpy()
        return y_pred.reshape(-1, 1)

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


def compute_shap_nn(
    members: list[dict[str, Any]],
    x_raw: np.ndarray,
    background_k: int,
    nsamples: int,
) -> np.ndarray:
    shap_values_all = []
    for fold_idx, member in enumerate(members, start=1):
        prepare_inputs, predict_fn = build_predict_fn(member, "mlp")
        x_scaled = prepare_inputs(x_raw)
        n_background = min(background_k, len(x_scaled))
        background = shap.kmeans(x_scaled, n_background)
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = np.asarray(explainer.shap_values(x_scaled, nsamples=nsamples, silent=True))
        shap_values_all.append(shap_values)
        print(f"    Fold {fold_idx}/{len(members)} complete")
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
    if model_family == "mlp":
        return compute_shap_nn(members, x_raw, background_k, nsamples)
    raise ValueError(f"Unsupported model family for this script: {model_family}")


def ensure_2d_shap(shap_values: np.ndarray) -> np.ndarray:
    arr = np.asarray(shap_values)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D SHAP values, got shape {arr.shape}")
    return arr


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


def plot_cross_target_importance(
    summary_df: pd.DataFrame,
    feature_cols: list[str],
    specs: list[BestSpec],
    save_path: Path,
) -> None:
    targets = [spec.target for spec in specs]
    width = 0.25
    x = np.arange(len(feature_cols))
    colors = ["#4E79A7", "#59A14F", "#E15759"]
    plt.figure(figsize=(10, 5))
    for idx, target in enumerate(targets):
        subset = summary_df[summary_df["target"] == target].copy()
        subset = subset.set_index("feature").reindex(feature_cols).fillna(0.0)
        plt.bar(
            x + idx * width,
            subset["mean_abs_shap"].values,
            width,
            label=TARGET_LABELS[target],
            color=colors[idx],
        )
    plt.xticks(
        x + width,
        [feature_label(col) for col in feature_cols],
        rotation=30,
        ha="right",
    )
    plt.ylabel("Mean |SHAP value|")
    plt.title(
        "Best 7-feature model per target - cross-target importance", fontsize=13, fontweight="bold"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")


def analyze_target(
    spec: BestSpec,
    max_test_samples: int,
    nn_background_k: int,
    nn_nsamples: int,
) -> list[dict[str, Any]]:
    print(f"\n{'=' * 80}")
    print(f"SHAP analysis: {spec.label}")
    print(f"Run dir: {spec.run_dir}")
    print(f"Model: {MODEL_LABELS[spec.model_name]}")
    print(f"{'=' * 80}")

    checkpoint = load_checkpoint(checkpoint_path_for(spec.run_dir, spec.model_name))
    feature_cols = checkpoint["feature_cols"]
    df = load_dataset_for_run(spec.run_dir)
    test_subset = filter_test_rows(df, feature_cols, spec.target, max_test_samples)
    x_raw_df = test_subset[feature_cols].copy()
    x_raw = x_raw_df.to_numpy(dtype=np.float64)

    shap_values = ensure_2d_shap(
        compute_shap_values(
            checkpoint=checkpoint,
            target_col=spec.target,
            x_raw=x_raw,
            background_k=nn_background_k,
            nsamples=nn_nsamples,
        )
    )

    target_dir = OUTPUT_DIR / spec.target
    target_dir.mkdir(parents=True, exist_ok=True)
    np.save(target_dir / "shap_values.npy", shap_values)

    x_display = x_raw_df.rename(columns={col: feature_label(col) for col in feature_cols})
    plot_summary_beeswarm(
        shap_values,
        x_display,
        f"{spec.label} - {MODEL_LABELS[spec.model_name]} - 7feat",
        target_dir / "summary_beeswarm.png",
    )
    mean_abs = plot_bar(
        shap_values,
        feature_cols,
        f"{spec.label} - {MODEL_LABELS[spec.model_name]} - 7feat",
        target_dir / "bar_importance.png",
    )
    plot_dependence_top3(
        shap_values,
        x_raw_df,
        feature_cols,
        f"{spec.label} - {MODEL_LABELS[spec.model_name]}",
        target_dir,
    )

    rows: list[dict[str, Any]] = []
    for rank, (feature_name, value) in enumerate(mean_abs.items(), start=1):
        rows.append(
            {
                "target": spec.target,
                "target_label": spec.label,
                "model": spec.model_name,
                "model_label": MODEL_LABELS[spec.model_name],
                "feature": feature_name,
                "feature_label": feature_label(feature_name),
                "mean_abs_shap": float(value),
                "rank": rank,
                "test_rows": int(len(test_subset)),
                "run_dir": str(spec.run_dir),
            }
        )
    return rows


def write_summary_outputs(summary_rows: list[dict[str, Any]]) -> pd.DataFrame:
    summary_df = pd.DataFrame(summary_rows).sort_values(["target", "rank"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUTPUT_DIR / "mean_abs_shap_summary.csv", index=False)
    (OUTPUT_DIR / "mean_abs_shap_summary.json").write_text(json.dumps(summary_rows, indent=2))
    return summary_df


def write_top_feature_report(summary_df: pd.DataFrame, specs: list[BestSpec]) -> None:
    lines = ["# Dataset1 7-feature SHAP Summary", ""]
    for spec in specs:
        subset = summary_df[summary_df["target"] == spec.target]
        lines.append(f"## {spec.label}")
        lines.append("")
        lines.append(f"- Model: {MODEL_LABELS[spec.model_name]}")
        lines.append(f"- Run dir: `{spec.run_dir}`")
        top3 = subset.head(3)
        joined = ", ".join(
            f"{row.feature_label} ({row.mean_abs_shap:.4f})" for row in top3.itertuples()
        )
        lines.append(f"- Top 3: {joined}")
        lines.append("")
    (OUTPUT_DIR / "top_features.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    specs = build_specs(args)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    shared_feature_cols: list[str] | None = None

    for spec in specs:
        summary_rows.extend(
            analyze_target(
                spec=spec,
                max_test_samples=args.max_test_samples,
                nn_background_k=args.nn_background_k,
                nn_nsamples=args.nn_nsamples,
            )
        )
        if shared_feature_cols is None:
            ckpt = load_checkpoint(checkpoint_path_for(spec.run_dir, spec.model_name))
            shared_feature_cols = ckpt["feature_cols"]

    summary_df = write_summary_outputs(summary_rows)
    write_top_feature_report(summary_df, specs)
    if shared_feature_cols is not None:
        plot_cross_target_importance(
            summary_df=summary_df,
            feature_cols=shared_feature_cols,
            specs=specs,
            save_path=OUTPUT_DIR / "cross_target_importance.png",
        )

    print(f"\n{'=' * 80}")
    print(f"SHAP outputs saved to: {OUTPUT_DIR}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
