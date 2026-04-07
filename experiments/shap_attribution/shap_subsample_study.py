"""SHAP Attribution Study — Subsample experiments (conditions D & E).

For each condition (5-common, 6-feat) × K random seeds:
  1. Subsample Dataset1 to N_TL rows (matching Tai Lake size).
  2. Split 141 train / 34 test (matching Tai Lake split).
  3. Train RF + XGBoost with 5-fold CV using fixed best hyperparameters
     from the full-data experiments B (5-feat) and C (6-feat).
  4. Compute SHAP via TreeExplainer (fast & exact).
  5. Save per-seed results and aggregated summary.

Also loads results from conditions A' (Tai Lake 5-common), B, and C
for the final cross-condition comparison table.

Usage:
    python experiments/shap_attribution/shap_subsample_study.py
    python experiments/shap_attribution/shap_subsample_study.py --seeds 10
    python experiments/shap_attribution/shap_subsample_study.py --skip-training  # reuse saved models
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass, field
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parents[2]
DATA_D1 = PROJECT / "data" / "dataset1_dbp_formation_with_split.csv"
DATA_TL = PROJECT / "data" / "DBP_dataset_DWTP_B.csv"
OUTPUT_DIR = PROJECT / "results" / "shap_attribution"

# Existing full-data experiment directories (B and C)
B_RUN_DIR = (
    PROJECT / "checkpoints" / "formal_dataset1_5feat_avg" / "20260330T235734Z"
)
C_RUN_DIR = (
    PROJECT / "checkpoints" / "formal_dataset1_6feat_cl2d_avg" / "20260331T130339Z"
)

# ---------------------------------------------------------------------------
# Feature / target definitions
# ---------------------------------------------------------------------------
FEATURES_5COMMON_D1 = ["ph_in_avg", "uv_in_avg", "temp_in_avg", "toc_in_avg", "br_in_avg"]
FEATURES_6FEAT_D1 = FEATURES_5COMMON_D1 + ["cl2d_in_avg"]
TARGETS_D1 = ["thm4_in_avg", "dbcm_in_avg", "bdcm_in_avg"]

FEATURES_5COMMON_TL = ["pH", "UV254_A_cm", "temp_C", "TOC_mg_L", "Br_mg_L"]
TARGETS_TL = ["T_THMs_ug_L", "DBCM_ug_L", "BDCM_ug_L"]

# Unified display labels (keyed by Dataset1 column names)
FEATURE_LABELS = {
    "ph_in_avg": "pH",
    "uv_in_avg": "UV254",
    "temp_in_avg": "Temperature",
    "toc_in_avg": "TOC",
    "br_in_avg": "Bromide",
    "cl2d_in_avg": "Cl₂ dose",
    # Tai Lake columns
    "pH": "pH",
    "UV254_A_cm": "UV254",
    "temp_C": "Temperature",
    "TOC_mg_L": "TOC",
    "Br_mg_L": "Bromide",
}

TARGET_LABELS = {
    "thm4_in_avg": "THM4",
    "dbcm_in_avg": "DBCM",
    "bdcm_in_avg": "BDCM",
    "T_THMs_ug_L": "T-THMs",
    "DBCM_ug_L": "DBCM",
    "BDCM_ug_L": "BDCM",
}

# Canonical feature order for cross-condition comparison
CANONICAL_FEATURES = ["pH", "UV254", "Temperature", "TOC", "Bromide"]

MODEL_LABELS = {"rf": "Random Forest", "xgb": "XGBoost"}

# Tai Lake sample counts (the target for subsampling)
N_TL = 175
N_TRAIN_TL = 141
N_TEST_TL = 34


# ---------------------------------------------------------------------------
# Best hyperparameters from B (5-feat) and C (6-feat) full-data runs
# ---------------------------------------------------------------------------
def _load_best_hyperparams(run_dir: Path) -> dict[str, dict[str, dict]]:
    """Load per-target best model_params from trial_history.json.

    Returns: {model: {target: {param: value}}}
    """
    path = run_dir / "trial_history.json"
    raw = json.loads(path.read_text())
    result: dict[str, dict[str, dict]] = {}
    for model_name in ("rf", "xgb"):
        if model_name not in raw:
            continue
        result[model_name] = {}
        for target, trials in raw[model_name]["targets"].items():
            # First entry is the best trial
            result[model_name][target] = trials[0]["model_params"]
    return result


# ---------------------------------------------------------------------------
# Condition spec
# ---------------------------------------------------------------------------
@dataclass
class ConditionSpec:
    key: str  # "D" or "E"
    label: str
    feature_cols: list[str]
    hyperparams: dict[str, dict[str, dict]]  # from B or C
    n_features: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_features = len(self.feature_cols)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def _build_estimator(model_name: str, params: dict) -> Any:
    """Instantiate an RF or XGBoost model with the given hyperparameters."""
    clean = {k: v for k, v in params.items() if v is not None}
    if model_name == "rf":
        return RandomForestRegressor(random_state=42, **clean)
    # XGBoost
    clean.pop("tree_method", None)
    clean.pop("early_stopping_rounds", None)
    return XGBRegressor(
        random_state=42,
        tree_method="hist",
        verbosity=0,
        **clean,
    )


def train_cv_ensemble(
    model_name: str,
    params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Train a 5-fold CV ensemble and return list of member dicts."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    members = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        scaler_x = StandardScaler().fit(X_tr)
        X_tr_s = scaler_x.transform(X_tr)

        estimator = _build_estimator(model_name, params)
        estimator.fit(X_tr_s, y_tr.ravel())

        members.append({
            "scaler_x": scaler_x,
            "estimator": estimator,
        })
    return members


# ---------------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------------
def compute_shap_ensemble(
    members: list[dict],
    X_test: np.ndarray,
) -> np.ndarray:
    """Compute SHAP values averaged across CV ensemble members."""
    all_sv = []
    for member in members:
        X_scaled = member["scaler_x"].transform(X_test)
        explainer = shap.TreeExplainer(member["estimator"])
        sv = np.asarray(explainer.shap_values(X_scaled))
        if sv.ndim == 3 and sv.shape[2] == 1:
            sv = sv[:, :, 0]
        all_sv.append(sv)
    return np.mean(np.stack(all_sv, axis=0), axis=0)


def shap_to_ranking(shap_values: np.ndarray, feature_cols: list[str]) -> pd.DataFrame:
    """Convert SHAP matrix → DataFrame with mean|SHAP| and rank."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    labels = [FEATURE_LABELS.get(c, c) for c in feature_cols]
    df = pd.DataFrame({
        "feature": feature_cols,
        "feature_label": labels,
        "mean_abs_shap": mean_abs,
    })
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


# ---------------------------------------------------------------------------
# Subsample + train + SHAP for one seed
# ---------------------------------------------------------------------------
def run_one_seed(
    condition: ConditionSpec,
    df_pool: pd.DataFrame,
    seed: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Run training + SHAP for one random subsample.

    Returns list of summary rows.
    """
    rng = np.random.RandomState(seed)

    # 1. Subsample N_TL rows
    idx = rng.choice(len(df_pool), size=N_TL, replace=False)
    df_sub = df_pool.iloc[idx].reset_index(drop=True)

    # 2. Split into train/test (141/34)
    perm = rng.permutation(N_TL)
    train_idx = perm[:N_TRAIN_TL]
    test_idx = perm[N_TRAIN_TL:]

    X_all = df_sub[condition.feature_cols].to_numpy(dtype=np.float64)
    X_train = X_all[train_idx]
    X_test = X_all[test_idx]

    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []

    for target_col in TARGETS_D1:
        y_all = df_sub[target_col].to_numpy(dtype=np.float64).reshape(-1, 1)
        y_train = y_all[train_idx]

        for model_name in ("rf", "xgb"):
            params = condition.hyperparams[model_name][target_col]

            # 3. Train 5-fold CV ensemble
            members = train_cv_ensemble(
                model_name=model_name,
                params=params,
                X_train=X_train,
                y_train=y_train,
                n_folds=5,
                seed=seed,
            )

            # 4. SHAP
            sv = compute_shap_ensemble(members, X_test)
            ranking = shap_to_ranking(sv, condition.feature_cols)

            # Save per-seed artifacts
            target_dir = seed_dir / model_name / target_col
            target_dir.mkdir(parents=True, exist_ok=True)
            np.save(target_dir / "shap_values.npy", sv)
            ranking.to_csv(target_dir / "ranking.csv", index=False)

            # Collect summary
            for _, row in ranking.iterrows():
                summary_rows.append({
                    "condition": condition.key,
                    "condition_label": condition.label,
                    "seed": seed,
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "target": target_col,
                    "target_label": TARGET_LABELS[target_col],
                    "feature": row["feature"],
                    "feature_label": row["feature_label"],
                    "mean_abs_shap": float(row["mean_abs_shap"]),
                    "rank": int(row["rank"]),
                    "n_train": N_TRAIN_TL,
                    "n_test": N_TEST_TL,
                })

    return summary_rows


# ---------------------------------------------------------------------------
# Load existing results from B/C checkpoints (conditions B and C)
# ---------------------------------------------------------------------------
def load_existing_shap_results(run_dir: Path, condition_key: str,
                               condition_label: str,
                               feature_cols: list[str],
                               targets: list[str]) -> list[dict[str, Any]]:
    """Load SHAP summary from an existing full-data experiment.

    Reads the pre-computed SHAP results from the shap_analysis_dataset1_ablation
    output, or recomputes from checkpoints if needed.
    """
    # Try the pre-computed summary CSV first
    summary_csv = PROJECT / "results" / "shap_analysis_dataset1_ablation" / "mean_abs_shap_summary.csv"
    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
        # Map experiment keys to our condition keys
        exp_key_map = {
            "B": "formal_5feat",
            "C": "formal_6feat_cl2d",
        }
        exp_key = exp_key_map.get(condition_key)
        if exp_key and exp_key in df["experiment"].values:
            subset = df[
                (df["experiment"] == exp_key)
                & (df["model"].isin(["rf", "xgb"]))
            ].copy()
            rows = []
            for _, r in subset.iterrows():
                rows.append({
                    "condition": condition_key,
                    "condition_label": condition_label,
                    "seed": -1,  # not applicable
                    "model": r["model"],
                    "model_label": r["model_label"],
                    "target": r["target"],
                    "target_label": r["target_label"],
                    "feature": r["feature"],
                    "feature_label": r["feature_label"],
                    "mean_abs_shap": float(r["mean_abs_shap"]),
                    "rank": int(r["rank"]),
                    "n_train": 382,
                    "n_test": 106,
                })
            return rows
    return []


# ---------------------------------------------------------------------------
# Load A' results (Tai Lake 5-common)
# ---------------------------------------------------------------------------
def load_aprime_results(run_dir: Path | None) -> list[dict[str, Any]]:
    """Load SHAP results from the A' experiment checkpoint.

    If run_dir is None or doesn't exist, return empty list (A' not yet run).
    """
    if run_dir is None or not run_dir.exists():
        return []

    rows: list[dict[str, Any]] = []
    for model_name in ("rf", "xgb"):
        ckpt_path = run_dir / f"{model_name}_tuned_checkpoint.joblib"
        if not ckpt_path.exists():
            continue

        ckpt = joblib.load(ckpt_path)
        feature_cols = ckpt["feature_cols"]
        df = pd.read_csv(DATA_TL)

        for target_col in ckpt["target_cols"]:
            # Get test data
            test_df = df.loc[df["split"] == "test", feature_cols + [target_col]].dropna()
            X_test = test_df[feature_cols].to_numpy(dtype=np.float64)

            # Compute SHAP from ensemble members
            members_raw = ckpt["target_payloads"][target_col]["members"]
            # Adapt to our expected format
            members = []
            for m in members_raw:
                members.append({
                    "scaler_x": m["scaler_x"],
                    "estimator": m["model_state"]["estimator"],
                })
            sv = compute_shap_ensemble(members, X_test)
            ranking = shap_to_ranking(sv, feature_cols)

            target_label_map = dict(zip(TARGETS_TL, ["THM4", "DBCM", "BDCM"]))
            for _, r in ranking.iterrows():
                rows.append({
                    "condition": "A'",
                    "condition_label": "A' Tai Lake 5-common",
                    "seed": -1,
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "target": target_col,
                    "target_label": target_label_map.get(target_col, target_col),
                    "feature": r["feature"],
                    "feature_label": r["feature_label"],
                    "mean_abs_shap": float(r["mean_abs_shap"]),
                    "rank": int(r["rank"]),
                    "n_train": N_TRAIN_TL,
                    "n_test": N_TEST_TL,
                })
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_subsample_results(all_rows: list[dict]) -> pd.DataFrame:
    """Compute mean ± std of SHAP values and ranks across seeds."""
    df = pd.DataFrame(all_rows)
    subsample = df[df["seed"] >= 0].copy()
    if subsample.empty:
        return pd.DataFrame()

    agg = (
        subsample.groupby(["condition", "condition_label", "model", "model_label",
                           "target", "target_label", "feature", "feature_label"])
        .agg(
            shap_mean=("mean_abs_shap", "mean"),
            shap_std=("mean_abs_shap", "std"),
            rank_mean=("rank", "mean"),
            rank_std=("rank", "std"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    return agg


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_cross_condition_bars(
    all_rows: list[dict],
    output_dir: Path,
) -> None:
    """Grouped bar chart: conditions side-by-side for each model × target."""
    df = pd.DataFrame(all_rows)
    if df.empty:
        return

    # For subsample conditions (D, E), average across seeds
    fixed = df[df["seed"] < 0].copy()
    sub = df[df["seed"] >= 0].copy()

    if not sub.empty:
        sub_agg = (
            sub.groupby(["condition", "condition_label", "model", "model_label",
                         "target", "target_label", "feature", "feature_label"])
            .agg(mean_abs_shap=("mean_abs_shap", "mean"),
                 shap_std=("mean_abs_shap", "std"))
            .reset_index()
        )
        sub_agg["seed"] = -1
    else:
        sub_agg = pd.DataFrame()

    if not fixed.empty:
        fixed["shap_std"] = 0.0

    plot_df = pd.concat([fixed, sub_agg], ignore_index=True) if not sub_agg.empty else fixed.copy()
    if plot_df.empty:
        return

    # Use canonical feature labels for x-axis
    plot_df["feature_label"] = plot_df["feature_label"].fillna(
        plot_df["feature"].map(FEATURE_LABELS)
    )

    condition_order = ["A'", "B", "D", "C", "E"]
    conditions_present = [c for c in condition_order if c in plot_df["condition"].values]

    colors = {
        "A'": "#E15759",  # red
        "B": "#4E79A7",   # blue
        "D": "#76B7B2",   # teal
        "C": "#F28E2B",   # orange
        "E": "#59A14F",   # green
    }

    for model_name in plot_df["model"].unique():
        for target in plot_df["target"].unique():
            subset = plot_df[
                (plot_df["model"] == model_name) & (plot_df["target"] == target)
            ]
            if subset.empty:
                continue

            # Get all feature labels in this subset
            all_feats = sorted(
                subset["feature_label"].unique(),
                key=lambda f: CANONICAL_FEATURES.index(f) if f in CANONICAL_FEATURES else 99,
            )

            fig, ax = plt.subplots(figsize=(10, 5))
            n_cond = len(conditions_present)
            width = 0.8 / max(n_cond, 1)
            x = np.arange(len(all_feats))

            for i, cond in enumerate(conditions_present):
                cond_data = subset[subset["condition"] == cond]
                if cond_data.empty:
                    continue
                vals = []
                errs = []
                for feat in all_feats:
                    row = cond_data[cond_data["feature_label"] == feat]
                    if row.empty:
                        vals.append(0)
                        errs.append(0)
                    else:
                        vals.append(row["mean_abs_shap"].iloc[0])
                        errs.append(row.get("shap_std", pd.Series([0])).iloc[0])
                lbl_map = {
                    "A'": "A' (TL,175,5)",
                    "B": "B (D1,488,5)",
                    "C": "C (D1,488,6)",
                    "D": "D (D1,175,5)",
                    "E": "E (D1,175,6)",
                }
                ax.bar(
                    x + i * width,
                    vals,
                    width,
                    yerr=errs if any(e > 0 for e in errs) else None,
                    capsize=3,
                    label=lbl_map.get(cond, cond),
                    color=colors.get(cond, "#999999"),
                    alpha=0.85,
                )

            ax.set_xticks(x + width * (n_cond - 1) / 2)
            ax.set_xticklabels(all_feats, rotation=20, ha="right")
            ax.set_ylabel("Mean |SHAP value|")
            target_lbl = TARGET_LABELS.get(target, target)
            model_lbl = MODEL_LABELS.get(model_name, model_name)
            ax.set_title(
                f"Cross-condition SHAP — {model_lbl} × {target_lbl}",
                fontsize=13, fontweight="bold",
            )
            ax.legend(fontsize=9)
            plt.tight_layout()
            plt.savefig(output_dir / f"cross_condition_{model_name}_{target}.png", dpi=150)
            plt.close("all")


def plot_ranking_heatmap(
    all_rows: list[dict],
    output_dir: Path,
) -> None:
    """Heatmap of feature rankings across conditions for each model × target."""
    df = pd.DataFrame(all_rows)
    if df.empty:
        return

    # Average ranks for subsample conditions
    fixed = df[df["seed"] < 0].copy()
    sub = df[df["seed"] >= 0].copy()
    if not sub.empty:
        sub_agg = (
            sub.groupby(["condition", "model", "target", "feature_label"])
            ["rank"].mean().reset_index()
        )
    else:
        sub_agg = pd.DataFrame()
    plot_df = pd.concat([fixed[["condition", "model", "target", "feature_label", "rank"]],
                         sub_agg], ignore_index=True) if not sub_agg.empty else fixed.copy()

    condition_order = ["A'", "B", "D", "C", "E"]

    for model_name in plot_df["model"].unique():
        for target in plot_df["target"].unique():
            subset = plot_df[
                (plot_df["model"] == model_name) & (plot_df["target"] == target)
            ]
            if subset.empty:
                continue

            conditions_present = [c for c in condition_order if c in subset["condition"].values]
            features = sorted(
                subset["feature_label"].unique(),
                key=lambda f: CANONICAL_FEATURES.index(f) if f in CANONICAL_FEATURES else 99,
            )

            matrix = np.full((len(features), len(conditions_present)), np.nan)
            for i, feat in enumerate(features):
                for j, cond in enumerate(conditions_present):
                    row = subset[(subset["feature_label"] == feat) & (subset["condition"] == cond)]
                    if not row.empty:
                        matrix[i, j] = row["rank"].iloc[0]

            fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * len(features) + 1)))
            im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=1, vmax=len(features))
            ax.set_xticks(range(len(conditions_present)))
            ax.set_xticklabels(conditions_present)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)

            # Annotate cells
            for i in range(len(features)):
                for j in range(len(conditions_present)):
                    if not np.isnan(matrix[i, j]):
                        val = matrix[i, j]
                        text = f"{val:.1f}" if val != int(val) else f"{int(val)}"
                        ax.text(j, i, text, ha="center", va="center",
                                fontsize=11, fontweight="bold",
                                color="white" if val <= 2 else "black")

            target_lbl = TARGET_LABELS.get(target, target)
            model_lbl = MODEL_LABELS.get(model_name, model_name)
            ax.set_title(
                f"Feature Ranking — {model_lbl} × {target_lbl}",
                fontsize=13, fontweight="bold",
            )
            plt.colorbar(im, ax=ax, label="Rank (1=most important)", shrink=0.8)
            plt.tight_layout()
            plt.savefig(output_dir / f"ranking_heatmap_{model_name}_{target}.png", dpi=150)
            plt.close("all")


# ---------------------------------------------------------------------------
# Master comparison table
# ---------------------------------------------------------------------------
def write_master_table(
    all_rows: list[dict],
    agg_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write the master comparison markdown table."""
    df_fixed = pd.DataFrame([r for r in all_rows if r["seed"] < 0])
    lines = ["# SHAP Attribution Study — Master Comparison", ""]

    condition_order = ["A'", "B", "D", "C", "E"]

    for model_name in ("rf", "xgb"):
        model_lbl = MODEL_LABELS[model_name]
        lines.append(f"## {model_lbl}")
        lines.append("")

        # Get all targets present for this model
        targets_present = sorted(
            set(r["target"] for r in all_rows if r["model"] == model_name)
        )

        for target in targets_present:
            target_lbl = TARGET_LABELS.get(target, target)
            lines.append(f"### {target_lbl}")
            lines.append("")

            # Header
            conds_in_data = [c for c in condition_order
                             if any(r["condition"] == c and r["model"] == model_name
                                    and r["target"] == target for r in all_rows)]
            header = "| Feature |"
            sep = "|---------|"
            for cond in conds_in_data:
                header += f" {cond} |"
                sep += "------|"
            lines.append(header)
            lines.append(sep)

            # Build lookup: (condition, feature_label) → display string
            lookup: dict[tuple[str, str], str] = {}

            # Fixed conditions
            if not df_fixed.empty:
                for _, r in df_fixed[
                    (df_fixed["model"] == model_name) & (df_fixed["target"] == target)
                ].iterrows():
                    key = (r["condition"], r["feature_label"])
                    lookup[key] = f"#{int(r['rank'])} {r['mean_abs_shap']:.4f}"

            # Subsample conditions (from agg)
            if not agg_df.empty:
                for _, r in agg_df[
                    (agg_df["model"] == model_name) & (agg_df["target"] == target)
                ].iterrows():
                    key = (r["condition"], r["feature_label"])
                    lookup[key] = f"#{r['rank_mean']:.1f} {r['shap_mean']:.4f}±{r['shap_std']:.4f}"

            all_feats = sorted(
                set(r["feature_label"] for r in all_rows
                    if r["model"] == model_name and r["target"] == target),
                key=lambda f: CANONICAL_FEATURES.index(f) if f in CANONICAL_FEATURES else 99,
            )

            for feat in all_feats:
                row_str = f"| {feat} |"
                for cond in conds_in_data:
                    val = lookup.get((cond, feat), "—")
                    row_str += f" {val} |"
                lines.append(row_str)

            lines.append("")

    (output_dir / "master_comparison.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of random subsampling seeds (default: 5)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip D/E training, only load existing results")
    parser.add_argument("--aprime-run-dir", type=str, default=None,
                        help="Path to A' run directory (auto-detected if omitted)")
    return parser.parse_args()


def auto_detect_aprime_dir() -> Path | None:
    """Find the most recent A' checkpoint directory."""
    parent = PROJECT / "checkpoints" / "shap_attribution" / "tailake_5common_formal"
    if not parent.exists():
        return None
    runs = sorted(parent.iterdir())
    return runs[-1] if runs else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # A': Tai Lake 5-common (if available)
    # ------------------------------------------------------------------
    aprime_dir = Path(args.aprime_run_dir) if args.aprime_run_dir else auto_detect_aprime_dir()
    if aprime_dir and aprime_dir.exists():
        print(f"Loading A' results from {aprime_dir}")
        aprime_rows = load_aprime_results(aprime_dir)
        all_rows.extend(aprime_rows)
        print(f"  → {len(aprime_rows)} rows")
    else:
        print("A' not yet available (run tailake_5common_formal.yaml first)")

    # ------------------------------------------------------------------
    # B & C: existing full-data results
    # ------------------------------------------------------------------
    print("\nLoading B (Dataset1 5-feat full) results...")
    b_rows = load_existing_shap_results(
        B_RUN_DIR, "B", "B Dataset1 5-common (N=488)",
        FEATURES_5COMMON_D1, TARGETS_D1,
    )
    all_rows.extend(b_rows)
    print(f"  → {len(b_rows)} rows")

    print("Loading C (Dataset1 6-feat full) results...")
    c_rows = load_existing_shap_results(
        C_RUN_DIR, "C", "C Dataset1 6-feat (N=488)",
        FEATURES_6FEAT_D1, TARGETS_D1,
    )
    all_rows.extend(c_rows)
    print(f"  → {len(c_rows)} rows")

    # ------------------------------------------------------------------
    # D & E: subsample experiments
    # ------------------------------------------------------------------
    if not args.skip_training:
        print(f"\nLoading Dataset1 for subsampling...")
        df_d1 = pd.read_csv(DATA_D1)

        # Pool: rows with complete data for 6-feat + targets
        pool_cols = FEATURES_6FEAT_D1 + TARGETS_D1
        df_pool = df_d1.dropna(subset=pool_cols).reset_index(drop=True)
        print(f"  Pool size: {len(df_pool)} (from {len(df_d1)} total)")

        # Load hyperparams
        print("Loading best hyperparams from B and C...")
        hp_b = _load_best_hyperparams(B_RUN_DIR)
        hp_c = _load_best_hyperparams(C_RUN_DIR)

        conditions = [
            ConditionSpec(
                key="D",
                label="D Dataset1 5-common (N=175)",
                feature_cols=FEATURES_5COMMON_D1,
                hyperparams=hp_b,
            ),
            ConditionSpec(
                key="E",
                label="E Dataset1 6-feat (N=175)",
                feature_cols=FEATURES_6FEAT_D1,
                hyperparams=hp_c,
            ),
        ]

        seeds = list(range(1, args.seeds + 1))

        for cond in conditions:
            print(f"\n{'=' * 60}")
            print(f"Condition {cond.key}: {cond.label}")
            print(f"Features: {cond.feature_cols}")
            print(f"{'=' * 60}")

            cond_dir = OUTPUT_DIR / cond.key
            for seed in seeds:
                print(f"\n  Seed {seed}/{len(seeds)}...")
                rows = run_one_seed(cond, df_pool, seed, cond_dir)
                all_rows.extend(rows)
                print(f"    → {len(rows)} rows")

    # ------------------------------------------------------------------
    # Save raw results
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Saving results...")

    raw_df = pd.DataFrame(all_rows)
    raw_df.to_csv(OUTPUT_DIR / "all_shap_results.csv", index=False)

    # Aggregate subsample results
    agg_df = aggregate_subsample_results(all_rows)
    if not agg_df.empty:
        agg_df.to_csv(OUTPUT_DIR / "subsample_aggregated.csv", index=False)

    # Master comparison table
    write_master_table(all_rows, agg_df, OUTPUT_DIR)

    # Plots
    print("Generating plots...")
    plot_cross_condition_bars(all_rows, OUTPUT_DIR)
    plot_ranking_heatmap(all_rows, OUTPUT_DIR)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
