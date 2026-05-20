"""Stratified Dataset1 importance analysis by within-tsid temperature variability.

This script labels each Dataset1 ``tsid`` by the within-series standard deviation
of ``temp_in_avg`` and then computes subgroup-specific feature importance on the
existing full-data Dataset1 formal tree-model checkpoints.

Default strata:
  - low:  tsid temperature std < 1.0 C
  - high: tsid temperature std >= 1.0 C

Default runs:
  - B: formal Dataset1 5-feature checkpoint
  - C: formal Dataset1 6-feature + Cl2 checkpoint

Importance methods:
  - SHAP (TreeExplainer, exact for tree models)
  - Permutation importance (delta RMSE in raw target units)

Outputs:
  - results/shap_attribution/temp_variability_stratified/importance_by_group.csv
  - results/shap_attribution/temp_variability_stratified/temperature_rank_summary.csv
  - results/shap_attribution/temp_variability_stratified/group_sample_sizes.csv
  - results/shap_attribution/temp_variability_stratified/run_metadata.json

Usage:
    python experiments/shap_attribution/stratified_temp_variability_importance.py
    python experiments/shap_attribution/stratified_temp_variability_importance.py --methods shap
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
    message="X does not have valid feature names",
)

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_squared_error


PROJECT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT / "results" / "shap_attribution" / "temp_variability_stratified"

FEATURE_LABELS = {
    "ph_in_avg": "pH",
    "uv_in_avg": "UV254",
    "temp_in_avg": "Temperature",
    "toc_in_avg": "TOC",
    "br_in_avg": "Bromide",
    "cl2d_in_avg": "Cl2 dose",
}
TARGET_LABELS = {
    "thm4_in_avg": "THM4",
    "dbcm_in_avg": "DBCM",
    "bdcm_in_avg": "BDCM",
}
MODEL_LABELS = {
    "rf": "Random Forest",
    "xgb": "XGBoost",
}


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    parent_dir: Path


DEFAULT_RUN_SPECS = (
    RunSpec(
        key="B",
        label="Dataset1 formal 5-feature",
        parent_dir=PROJECT / "checkpoints" / "formal_dataset1_5feat_avg",
    ),
    RunSpec(
        key="C",
        label="Dataset1 formal 6-feature + Cl2",
        parent_dir=PROJECT / "checkpoints" / "formal_dataset1_6feat_cl2d_avg",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Low/high tsid split threshold on within-tsid temp std in degrees C.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("shap", "permutation"),
        default=("shap", "permutation"),
        help="Importance methods to compute.",
    )
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=20,
        help="Number of shuffles per feature for permutation importance.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for permutation importance shuffling.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=0,
        help="Optional cap per subgroup after filtering. 0 uses all rows.",
    )
    return parser.parse_args()


def latest_run_dir(parent_dir: Path) -> Path:
    if not parent_dir.exists():
        raise FileNotFoundError(f"Checkpoint parent directory not found: {parent_dir}")
    run_dirs = sorted(path for path in parent_dir.iterdir() if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under: {parent_dir}")
    return run_dirs[-1]


def load_run_dataset(run_dir: Path) -> pd.DataFrame:
    snapshot_path = run_dir / "dataset_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    dataset_path = Path(snapshot["path"])
    if not dataset_path.is_absolute():
        dataset_path = (run_dir / dataset_path).resolve()
    return pd.read_csv(dataset_path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    return joblib.load(path)


def feature_label(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


def target_label(name: str) -> str:
    return TARGET_LABELS.get(name, name)


def annotate_temp_variability_groups(
    df: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tsid_stats = (
        df.groupby("tsid", dropna=False)
        .agg(
            tsid_row_count=("temp_in_avg", "count"),
            tsid_temp_std=("temp_in_avg", lambda s: s.std(ddof=1)),
        )
        .reset_index()
    )
    tsid_stats["temp_var_group"] = np.where(
        tsid_stats["tsid_temp_std"].isna(),
        "unknown",
        np.where(tsid_stats["tsid_temp_std"] < threshold, "low", "high"),
    )
    annotated = df.merge(tsid_stats, on="tsid", how="left")
    return annotated, tsid_stats


def filter_test_rows(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_name: str,
    max_test_samples: int,
) -> pd.DataFrame:
    required = feature_cols + [target_col, "tsid", "temp_var_group", "tsid_temp_std"]
    subset = df.loc[
        (df["split"] == "test") & (df["temp_var_group"] == group_name),
        required,
    ].dropna(axis=0, how="any")
    if max_test_samples and len(subset) > max_test_samples:
        subset = subset.iloc[:max_test_samples].copy()
    return subset.reset_index(drop=True)


def extract_members(checkpoint: dict[str, Any], target_col: str) -> list[dict[str, Any]]:
    members_raw = checkpoint["target_payloads"][target_col]["members"]
    members: list[dict[str, Any]] = []
    for member in members_raw:
        members.append(
            {
                "scaler_x": member["scaler_x"],
                "scaler_y": member.get("scaler_y"),
                "estimator": member["model_state"]["estimator"],
            }
        )
    return members


def predict_member_raw(member: dict[str, Any], x_raw: np.ndarray) -> np.ndarray:
    x_scaled = member["scaler_x"].transform(x_raw)
    pred_scaled = np.asarray(member["estimator"].predict(x_scaled)).reshape(-1, 1)
    scaler_y = member.get("scaler_y")
    if scaler_y is not None:
        pred_raw = scaler_y.inverse_transform(pred_scaled).ravel()
    else:
        pred_raw = pred_scaled.ravel()
    return pred_raw


def predict_ensemble_raw(members: list[dict[str, Any]], x_raw: np.ndarray) -> np.ndarray:
    preds = [predict_member_raw(member, x_raw) for member in members]
    return np.mean(np.stack(preds, axis=0), axis=0)


def compute_shap_ensemble(members: list[dict[str, Any]], x_raw: np.ndarray) -> np.ndarray:
    all_sv = []
    for member in members:
        x_scaled = member["scaler_x"].transform(x_raw)
        explainer = shap.TreeExplainer(member["estimator"])
        sv = np.asarray(explainer.shap_values(x_scaled))
        if sv.ndim == 3 and sv.shape[2] == 1:
            sv = sv[:, :, 0]
        scaler_y = member.get("scaler_y")
        if scaler_y is not None:
            sv = sv * float(scaler_y.scale_[0])
        all_sv.append(sv)
    return np.mean(np.stack(all_sv, axis=0), axis=0)


def compute_permutation_importance(
    members: list[dict[str, Any]],
    x_raw: np.ndarray,
    y_true: np.ndarray,
    feature_cols: list[str],
    n_repeats: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    baseline_pred = predict_ensemble_raw(members, x_raw)
    baseline_rmse = float(np.sqrt(mean_squared_error(y_true, baseline_pred)))

    rows: list[dict[str, Any]] = []
    for feature_idx, feature_name in enumerate(feature_cols):
        deltas = []
        for _ in range(n_repeats):
            x_perm = np.array(x_raw, copy=True)
            shuffled = np.array(x_perm[:, feature_idx], copy=True)
            rng.shuffle(shuffled)
            x_perm[:, feature_idx] = shuffled
            pred_perm = predict_ensemble_raw(members, x_perm)
            perm_rmse = float(np.sqrt(mean_squared_error(y_true, pred_perm)))
            deltas.append(perm_rmse - baseline_rmse)
        rows.append(
            {
                "feature": feature_name,
                "feature_label": feature_label(feature_name),
                "importance_mean": float(np.mean(deltas)),
                "importance_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                "baseline_rmse": baseline_rmse,
            }
        )
    return pd.DataFrame(rows)


def shap_to_importance_df(shap_values: np.ndarray, feature_cols: list[str]) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    return pd.DataFrame(
        {
            "feature": feature_cols,
            "feature_label": [feature_label(column) for column in feature_cols],
            "importance_mean": mean_abs,
            "importance_std": 0.0,
        }
    )


def rank_importance(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.sort_values("importance_mean", ascending=False).reset_index(drop=True).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def group_sample_summary(
    subset: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> dict[str, Any]:
    unique_tsids = subset["tsid"].nunique()
    return {
        "n_rows": int(len(subset)),
        "n_tsids": int(unique_tsids),
        "target": target_col,
        "feature_count": int(len(feature_cols)),
        "temp_mean": float(subset["temp_in_avg"].mean()),
        "temp_std": float(subset["temp_in_avg"].std(ddof=1)) if len(subset) > 1 else float("nan"),
        "tsid_temp_std_mean": float(subset["tsid_temp_std"].mean()),
        "tsid_temp_std_median": float(subset["tsid_temp_std"].median()),
    }


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    run_metadata: list[dict[str, Any]] = []

    for run_spec in DEFAULT_RUN_SPECS:
        run_dir = latest_run_dir(run_spec.parent_dir)
        dataset = load_run_dataset(run_dir)
        dataset, tsid_stats = annotate_temp_variability_groups(dataset, threshold=args.threshold)

        run_metadata.append(
            {
                "run_key": run_spec.key,
                "run_label": run_spec.label,
                "run_dir": str(run_dir),
                "threshold_c": args.threshold,
                "n_rows_total": int(len(dataset)),
                "n_tsids_total": int(dataset["tsid"].nunique()),
                "n_tsids_low": int((tsid_stats["temp_var_group"] == "low").sum()),
                "n_tsids_high": int((tsid_stats["temp_var_group"] == "high").sum()),
                "n_tsids_unknown": int((tsid_stats["temp_var_group"] == "unknown").sum()),
            }
        )

        for model_name in ("rf", "xgb"):
            ckpt_path = run_dir / f"{model_name}_tuned_checkpoint.joblib"
            checkpoint = load_checkpoint(ckpt_path)
            feature_cols = list(checkpoint["feature_cols"])

            for target_col in checkpoint["target_cols"]:
                members = extract_members(checkpoint, target_col)

                for group_name in ("low", "high"):
                    subset = filter_test_rows(
                        dataset,
                        feature_cols=feature_cols,
                        target_col=target_col,
                        group_name=group_name,
                        max_test_samples=args.max_test_samples,
                    )
                    if subset.empty:
                        continue

                    group_summary = group_sample_summary(
                        subset=subset,
                        feature_cols=feature_cols,
                        target_col=target_col,
                    )
                    group_rows.append(
                        {
                            "run": run_spec.key,
                            "run_label": run_spec.label,
                            "model": model_name,
                            "model_label": MODEL_LABELS[model_name],
                            "target": target_col,
                            "target_label": target_label(target_col),
                            "group": group_name,
                            **group_summary,
                        }
                    )

                    x_raw = subset[feature_cols].to_numpy(dtype=np.float64)
                    y_true = subset[target_col].to_numpy(dtype=np.float64)

                    for method in args.methods:
                        if method == "shap":
                            shap_values = compute_shap_ensemble(members, x_raw)
                            ranked = rank_importance(shap_to_importance_df(shap_values, feature_cols))
                        elif method == "permutation":
                            ranked = rank_importance(
                                compute_permutation_importance(
                                    members=members,
                                    x_raw=x_raw,
                                    y_true=y_true,
                                    feature_cols=feature_cols,
                                    n_repeats=args.permutation_repeats,
                                    seed=args.seed,
                                )
                            )
                        else:
                            raise ValueError(f"Unsupported method: {method}")

                        for _, row in ranked.iterrows():
                            all_rows.append(
                                {
                                    "run": run_spec.key,
                                    "run_label": run_spec.label,
                                    "model": model_name,
                                    "model_label": MODEL_LABELS[model_name],
                                    "target": target_col,
                                    "target_label": target_label(target_col),
                                    "group": group_name,
                                    "method": method,
                                    "feature": row["feature"],
                                    "feature_label": row["feature_label"],
                                    "importance_mean": float(row["importance_mean"]),
                                    "importance_std": float(row["importance_std"]),
                                    "rank": int(row["rank"]),
                                    "n_rows": int(len(subset)),
                                    "n_tsids": int(subset["tsid"].nunique()),
                                }
                            )

    importance_df = pd.DataFrame(all_rows)
    group_df = pd.DataFrame(group_rows)

    importance_df.to_csv(OUTPUT_DIR / "importance_by_group.csv", index=False)
    group_df.to_csv(OUTPUT_DIR / "group_sample_sizes.csv", index=False)
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2))

    temperature_rank_df = (
        importance_df[importance_df["feature"] == "temp_in_avg"]
        .sort_values(["method", "run", "model", "target", "group"])
        .reset_index(drop=True)
    )
    temperature_rank_df.to_csv(OUTPUT_DIR / "temperature_rank_summary.csv", index=False)

    print(f"Saved subgroup importance rows: {len(importance_df)}")
    print(f"Saved group summaries: {len(group_df)}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
