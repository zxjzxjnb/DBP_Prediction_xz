"""SHAP interpretability analysis for DBP prediction models.

Loads tuned model checkpoints and computes SHAP values for each target.
Generates summary plots, bar plots, and dependence plots.

Usage:
    python scripts/shap_analysis.py
"""

from __future__ import annotations

import os
import sys
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="shap")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT, "data", "DBP_dataset_DWTP_B.csv")
OUTPUT_DIR = os.path.join(PROJECT, "results", "shap_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHECKPOINTS = {
    "5feat": {
        "rf": os.path.join(PROJECT, "checkpoints/ablation_5feat_formal/20260315T231614Z/rf_tuned_checkpoint.joblib"),
        "xgb": os.path.join(PROJECT, "checkpoints/ablation_5feat_formal/20260315T231614Z/xgb_tuned_checkpoint.joblib"),
        "mlp": os.path.join(PROJECT, "checkpoints/ablation_5feat_formal/20260315T231614Z/mlp_tuned_checkpoint.pt"),
        "kan": os.path.join(PROJECT, "checkpoints/ablation_5feat_formal/20260315T231614Z/kan_tuned_checkpoint.pt"),
    },
    "6feat": {
        "rf": os.path.join(PROJECT, "checkpoints/ablation_6feat_formal/20260316T004125Z/rf_tuned_checkpoint.joblib"),
        "xgb": os.path.join(PROJECT, "checkpoints/ablation_6feat_formal/20260316T004125Z/xgb_tuned_checkpoint.joblib"),
        "mlp": os.path.join(PROJECT, "checkpoints/ablation_6feat_formal/20260316T004125Z/mlp_tuned_checkpoint.pt"),
        "kan": os.path.join(PROJECT, "checkpoints/ablation_6feat_formal/20260316T004125Z/kan_tuned_checkpoint.pt"),
    },
    "9feat": {
        "rf": os.path.join(PROJECT, "checkpoints/ablation_9feat_formal/20260316T013327Z/rf_tuned_checkpoint.joblib"),
        "xgb": os.path.join(PROJECT, "checkpoints/ablation_9feat_formal/20260316T013327Z/xgb_tuned_checkpoint.joblib"),
        "mlp": os.path.join(PROJECT, "checkpoints/ablation_9feat_formal/20260316T013327Z/mlp_tuned_checkpoint.pt"),
        "kan": os.path.join(PROJECT, "checkpoints/ablation_9feat_formal/20260316T013327Z/kan_tuned_checkpoint.pt"),
    },
}

TARGETS = ["T_THMs_ug_L", "DBCM_ug_L", "BDCM_ug_L"]
TARGET_LABELS = {"T_THMs_ug_L": "T-THMs", "DBCM_ug_L": "DBCM", "BDCM_ug_L": "BDCM"}

FEATURE_LABELS = {
    "pH": "pH",
    "UV254_A_cm": "UV254",
    "temp_C": "Temperature",
    "TOC_mg_L": "TOC",
    "COD_mg_L": "COD",
    "Br_mg_L": "Br⁻",
    "NH4_N_mg_L": "NH₄-N",
    "NO2_N_mg_L": "NO₂-N",
    "NO3_N_mg_L": "NO₃-N",
}

MODEL_LABELS = {"rf": "Random Forest", "xgb": "XGBoost", "mlp": "MLP", "kan": "KAN"}

# Best model per target (from ablation study)
BEST_CONFIGS = {
    "T_THMs_ug_L": ("5feat", "rf"),
    "DBCM_ug_L": ("5feat", "mlp"),
    "BDCM_ug_L": ("6feat", "mlp"),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
    return train_df, test_df


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def load_checkpoint(path: str) -> dict:
    if path.endswith(".joblib"):
        return joblib.load(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def get_member_predict_fn(member: dict, model_family: str):
    """Build a predict function from a single fold member."""
    scaler_x = member["scaler_x"]
    model_state = member["model_state"]

    if model_family in ("random_forest", "xgboost"):
        estimator = model_state["estimator"]

        def predict_fn(X: np.ndarray) -> np.ndarray:
            X_scaled = scaler_x.transform(X)
            return estimator.predict(X_scaled).reshape(-1, 1)

        return predict_fn

    in_dim = member["in_dim"]
    out_dim = member["out_dim"]
    model_params = member["model_params"]
    seed = member["seed"]

    if model_family == "mlp":
        sys.path.insert(0, PROJECT)
        from dbp_prediction.models.mlp import MLP

        model = MLP(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=list(model_params.get("hidden_dims", [32, 16])),
            dropout=float(model_params.get("dropout", 0.2)),
            activation=str(model_params.get("activation", "ReLU")),
        )
        model.load_state_dict(model_state)
        model.eval()

        def predict_fn(X: np.ndarray) -> np.ndarray:
            X_scaled = scaler_x.transform(X)
            with torch.no_grad():
                out = model(torch.from_numpy(X_scaled).float())
            return out.numpy()

        return predict_fn

    if model_family == "kan":
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
        model.load_state_dict(model_state)
        model.eval()

        def predict_fn(X: np.ndarray) -> np.ndarray:
            X_scaled = scaler_x.transform(X)
            with torch.no_grad():
                out = model(torch.from_numpy(X_scaled).float())
            return out.numpy()

        return predict_fn

    raise ValueError(f"Unknown model family: {model_family}")


def get_ensemble_predict_fn(members: list[dict], model_family: str):
    """Average predictions across CV fold members."""
    fns = [get_member_predict_fn(m, model_family) for m in members]

    def predict_fn(X: np.ndarray) -> np.ndarray:
        preds = [fn(X) for fn in fns]
        return np.mean(preds, axis=0).ravel()

    return predict_fn


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------
def compute_shap_tree(members: list[dict], X_test: np.ndarray, model_family: str):
    """Compute SHAP for tree-based ensemble using TreeExplainer."""
    shap_values_all = []
    for member in members:
        scaler_x = member["scaler_x"]
        estimator = member["model_state"]["estimator"]
        X_scaled = scaler_x.transform(X_test)
        explainer = shap.TreeExplainer(estimator)
        sv = explainer.shap_values(X_scaled)
        shap_values_all.append(sv)
    return np.mean(shap_values_all, axis=0)


def compute_shap_nn(members: list[dict], X_test: np.ndarray, model_family: str):
    """Compute SHAP for neural networks using KernelExplainer with background summary."""
    shap_values_all = []
    for i, member in enumerate(members):
        scaler_x = member["scaler_x"]
        model_state = member["model_state"]
        in_dim = member["in_dim"]
        out_dim = member["out_dim"]
        model_params = member["model_params"]
        seed = member["seed"]

        X_scaled = scaler_x.transform(X_test)

        if model_family == "mlp":
            from dbp_prediction.models.mlp import MLP
            model = MLP(
                in_dim=in_dim, out_dim=out_dim,
                hidden_dims=list(model_params.get("hidden_dims", [32, 16])),
                dropout=float(model_params.get("dropout", 0.2)),
                activation=str(model_params.get("activation", "ReLU")),
            )
        else:
            from dbp_prediction.models.kan import build_kan
            model = build_kan(
                in_dim=in_dim, out_dim=out_dim,
                hidden_dims=list(model_params.get("hidden_dims", [32, 16])),
                grid=int(model_params.get("grid", 3)),
                k=int(model_params.get("k", 5)),
                base_fun=str(model_params.get("base_fun", "silu")),
                seed=seed,
            )
        model.load_state_dict(model_state)
        model.eval()

        def predict_fn(X_arr):
            with torch.no_grad():
                return model(torch.from_numpy(X_arr.astype(np.float32)).float()).numpy().ravel()

        background = shap.kmeans(X_scaled, 30)
        explainer = shap.KernelExplainer(predict_fn, background)
        sv = explainer.shap_values(X_scaled, nsamples=200, silent=True)
        shap_values_all.append(sv)
        print(f"    Fold {i + 1}/{len(members)} done")

    return np.mean(shap_values_all, axis=0)


def compute_shap_values(checkpoint: dict, target: str, X_test: np.ndarray, model_family: str):
    """Dispatch to the appropriate SHAP computation method."""
    payload = checkpoint["target_payloads"][target]
    members = payload["members"]

    if model_family in ("random_forest", "xgboost"):
        return compute_shap_tree(members, X_test, model_family)
    else:
        return compute_shap_nn(members, X_test, model_family)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def rename_features(feature_names):
    return [FEATURE_LABELS.get(f, f) for f in feature_names]


def plot_summary_beeswarm(shap_vals, X_display, feature_names, title, save_path):
    """SHAP beeswarm summary plot."""
    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.5)))
    shap.summary_plot(
        shap_vals, X_display,
        feature_names=rename_features(feature_names),
        show=False, plot_size=None,
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")
    print(f"  Saved: {save_path}")


def plot_bar(shap_vals, feature_names, title, save_path):
    """SHAP mean absolute value bar chart."""
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    labels = rename_features(feature_names)

    fig, ax = plt.subplots(figsize=(8, max(3, len(feature_names) * 0.45)))
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(order)))
    ax.barh(range(len(order)), mean_abs[order][::-1], color=colors[::-1])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels[i] for i in order[::-1]])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")
    print(f"  Saved: {save_path}")


def plot_dependence_top3(shap_vals, X_test_raw, feature_names, title_prefix, save_dir):
    """Dependence plots for the top 3 most important features."""
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top3_idx = np.argsort(mean_abs)[-3:][::-1]
    labels = rename_features(feature_names)

    for rank, idx in enumerate(top3_idx):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        shap.dependence_plot(
            idx, shap_vals, X_test_raw,
            feature_names=labels,
            interaction_index="auto",
            show=False, ax=ax,
        )
        ax.set_title(f"{title_prefix} — {labels[idx]}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fname = os.path.join(save_dir, f"dep_top{rank + 1}_{feature_names[idx]}.png")
        plt.savefig(fname)
        plt.close("all")
        print(f"  Saved: {fname}")


def plot_multi_target_importance(all_shap_data: dict, feature_names: list, model_name: str, save_path: str):
    """Grouped bar chart: feature importance across all 3 targets."""
    labels = rename_features(feature_names)
    n_features = len(feature_names)
    n_targets = len(TARGETS)
    x = np.arange(n_features)
    width = 0.25
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, target in enumerate(TARGETS):
        if target in all_shap_data:
            vals = np.abs(all_shap_data[target]).mean(axis=0)
            ax.bar(x + i * width, vals, width, label=TARGET_LABELS[target], color=colors[i], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title(f"Feature Importance — {MODEL_LABELS.get(model_name, model_name)}", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")
    print(f"  Saved: {save_path}")


def plot_cross_model_importance(all_model_shap: dict, target: str, feature_names: list, save_path: str):
    """Compare feature importance across 4 models for one target."""
    labels = rename_features(feature_names)
    n_features = len(feature_names)
    models_with_data = [m for m in ["rf", "xgb", "mlp", "kan"] if m in all_model_shap and target in all_model_shap[m]]

    if not models_with_data:
        return

    x = np.arange(n_features)
    width = 0.2
    colors = {"rf": "#2196F3", "xgb": "#FF5722", "mlp": "#9C27B0", "kan": "#4CAF50"}

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model_name in enumerate(models_with_data):
        vals = np.abs(all_model_shap[model_name][target]).mean(axis=0)
        ax.bar(x + i * width, vals, width, label=MODEL_LABELS[model_name], color=colors[model_name], alpha=0.85)

    ax.set_xticks(x + width * (len(models_with_data) - 1) / 2)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title(f"Cross-Model Feature Importance — {TARGET_LABELS[target]}", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def analyze_experiment(exp_name: str, model_names: list[str] | None = None):
    """Run full SHAP analysis for one experiment configuration."""
    print(f"\n{'=' * 70}")
    print(f"SHAP Analysis: {exp_name}")
    print(f"{'=' * 70}")

    train_df, test_df = load_data()
    ckpts = CHECKPOINTS[exp_name]

    if model_names is None:
        model_names = list(ckpts.keys())

    first_ckpt = load_checkpoint(list(ckpts.values())[0])
    feature_cols = first_ckpt["feature_cols"]
    print(f"Features: {feature_cols}")

    X_test_raw = test_df[feature_cols].values
    exp_dir = os.path.join(OUTPUT_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    all_model_shap = {}

    for model_name in model_names:
        ckpt_path = ckpts[model_name]
        ckpt = load_checkpoint(ckpt_path)
        model_family = ckpt["model_family"]
        print(f"\n--- {MODEL_LABELS.get(model_name, model_name)} ({model_family}) ---")

        model_dir = os.path.join(exp_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        all_model_shap[model_name] = {}

        for target in TARGETS:
            print(f"\n  Target: {TARGET_LABELS[target]}")
            shap_vals = compute_shap_values(ckpt, target, X_test_raw, model_family)
            all_model_shap[model_name][target] = shap_vals

            target_dir = os.path.join(model_dir, target)
            os.makedirs(target_dir, exist_ok=True)

            X_display = pd.DataFrame(X_test_raw, columns=rename_features(feature_cols))

            plot_summary_beeswarm(
                shap_vals, X_display, feature_cols,
                f"{MODEL_LABELS[model_name]} — {TARGET_LABELS[target]}",
                os.path.join(target_dir, "summary_beeswarm.png"),
            )
            plot_bar(
                shap_vals, feature_cols,
                f"{MODEL_LABELS[model_name]} — {TARGET_LABELS[target]}",
                os.path.join(target_dir, "bar_importance.png"),
            )
            plot_dependence_top3(
                shap_vals, X_test_raw, feature_cols,
                f"{MODEL_LABELS[model_name]} — {TARGET_LABELS[target]}",
                target_dir,
            )

        plot_multi_target_importance(
            all_model_shap[model_name], feature_cols, model_name,
            os.path.join(model_dir, "multi_target_importance.png"),
        )

    for target in TARGETS:
        plot_cross_model_importance(
            all_model_shap, target, feature_cols,
            os.path.join(exp_dir, f"cross_model_{target}.png"),
        )

    return all_model_shap, feature_cols


def analyze_best_per_target():
    """SHAP analysis for the best model per target from optimal feature sets."""
    print(f"\n{'=' * 70}")
    print("SHAP Analysis: Best Model per Target")
    print(f"{'=' * 70}")

    train_df, test_df = load_data()
    best_dir = os.path.join(OUTPUT_DIR, "best_per_target")
    os.makedirs(best_dir, exist_ok=True)

    for target, (exp_name, model_name) in BEST_CONFIGS.items():
        ckpt_path = CHECKPOINTS[exp_name][model_name]
        ckpt = load_checkpoint(ckpt_path)
        model_family = ckpt["model_family"]
        feature_cols = ckpt["feature_cols"]

        print(f"\n--- {TARGET_LABELS[target]}: {MODEL_LABELS[model_name]} ({exp_name}, {len(feature_cols)} features) ---")
        print(f"  Features: {feature_cols}")

        X_test_raw = test_df[feature_cols].values
        shap_vals = compute_shap_values(ckpt, target, X_test_raw, model_family)

        target_dir = os.path.join(best_dir, target)
        os.makedirs(target_dir, exist_ok=True)

        X_display = pd.DataFrame(X_test_raw, columns=rename_features(feature_cols))

        plot_summary_beeswarm(
            shap_vals, X_display, feature_cols,
            f"Best: {MODEL_LABELS[model_name]} ({exp_name}) — {TARGET_LABELS[target]}",
            os.path.join(target_dir, "summary_beeswarm.png"),
        )
        plot_bar(
            shap_vals, feature_cols,
            f"Best: {MODEL_LABELS[model_name]} ({exp_name}) — {TARGET_LABELS[target]}",
            os.path.join(target_dir, "bar_importance.png"),
        )
        plot_dependence_top3(
            shap_vals, X_test_raw, feature_cols,
            f"Best: {MODEL_LABELS[model_name]} — {TARGET_LABELS[target]}",
            target_dir,
        )


def main():
    # Part 1: 9-feature experiment — all 4 models, comprehensive view
    analyze_experiment("9feat", model_names=["rf", "xgb", "mlp", "kan"])

    # Part 2: Best model per target from ablation study
    analyze_best_per_target()

    print(f"\n{'=' * 70}")
    print(f"All SHAP results saved to: {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
