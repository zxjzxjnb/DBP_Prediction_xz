"""Cross-ablation SHAP analysis: track feature importance evolution across rounds.

For each ablation round (3→4→5→6→9 features), compute SHAP values and
visualize how the same feature's contribution changes as new features
are added. Uses RF and XGBoost (TreeExplainer — fast & exact).

Usage:
    python scripts/shap_cross_ablation.py
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
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import shap
import torch

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT, "data", "DBP_dataset_DWTP_B.csv")
OUTPUT_DIR = os.path.join(PROJECT, "results", "shap_analysis", "cross_ablation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ABLATION_ROUNDS = [
    {
        "label": "3 feat",
        "features": ["pH", "UV254_A_cm", "temp_C"],
        "added": None,
        "ckpts": {
            "rf": os.path.join(PROJECT, "checkpoints/ablation_3feat_formal/20260315T214708Z/rf_tuned_checkpoint.joblib"),
            "xgb": os.path.join(PROJECT, "checkpoints/ablation_3feat_formal/20260315T214708Z/xgb_tuned_checkpoint.joblib"),
            "mlp": os.path.join(PROJECT, "checkpoints/ablation_3feat_formal/20260315T214708Z/mlp_tuned_checkpoint.pt"),
            "kan": os.path.join(PROJECT, "checkpoints/ablation_3feat_formal/20260315T214708Z/kan_tuned_checkpoint.pt"),
        },
    },
    {
        "label": "4 feat\n(+TOC)",
        "features": ["pH", "UV254_A_cm", "temp_C", "TOC_mg_L"],
        "added": "TOC_mg_L",
        "ckpts": {
            "rf": os.path.join(PROJECT, "checkpoints/ablation_4feat_formal/20260315T222224Z/rf_tuned_checkpoint.joblib"),
            "xgb": os.path.join(PROJECT, "checkpoints/ablation_4feat_formal/20260315T222224Z/xgb_tuned_checkpoint.joblib"),
            "mlp": os.path.join(PROJECT, "checkpoints/ablation_4feat_formal/20260315T222224Z/mlp_tuned_checkpoint.pt"),
            "kan": os.path.join(PROJECT, "checkpoints/ablation_4feat_formal/20260315T222224Z/kan_tuned_checkpoint.pt"),
        },
    },
    {
        "label": "5 feat\n(+COD)",
        "features": ["pH", "UV254_A_cm", "temp_C", "TOC_mg_L", "COD_mg_L"],
        "added": "COD_mg_L",
        "ckpts": {
            "rf": os.path.join(PROJECT, "checkpoints/ablation_5feat_formal/20260315T231614Z/rf_tuned_checkpoint.joblib"),
            "xgb": os.path.join(PROJECT, "checkpoints/ablation_5feat_formal/20260315T231614Z/xgb_tuned_checkpoint.joblib"),
            "mlp": os.path.join(PROJECT, "checkpoints/ablation_5feat_formal/20260315T231614Z/mlp_tuned_checkpoint.pt"),
            "kan": os.path.join(PROJECT, "checkpoints/ablation_5feat_formal/20260315T231614Z/kan_tuned_checkpoint.pt"),
        },
    },
    {
        "label": "6 feat\n(+Br)",
        "features": ["pH", "UV254_A_cm", "temp_C", "TOC_mg_L", "COD_mg_L", "Br_mg_L"],
        "added": "Br_mg_L",
        "ckpts": {
            "rf": os.path.join(PROJECT, "checkpoints/ablation_6feat_formal/20260316T004125Z/rf_tuned_checkpoint.joblib"),
            "xgb": os.path.join(PROJECT, "checkpoints/ablation_6feat_formal/20260316T004125Z/xgb_tuned_checkpoint.joblib"),
            "mlp": os.path.join(PROJECT, "checkpoints/ablation_6feat_formal/20260316T004125Z/mlp_tuned_checkpoint.pt"),
            "kan": os.path.join(PROJECT, "checkpoints/ablation_6feat_formal/20260316T004125Z/kan_tuned_checkpoint.pt"),
        },
    },
    {
        "label": "9 feat\n(+N)",
        "features": ["pH", "UV254_A_cm", "temp_C", "TOC_mg_L", "COD_mg_L",
                      "Br_mg_L", "NH4_N_mg_L", "NO2_N_mg_L", "NO3_N_mg_L"],
        "added": "NH4_N/NO2_N/NO3_N",
        "ckpts": {
            "rf": os.path.join(PROJECT, "checkpoints/ablation_9feat_formal/20260316T013327Z/rf_tuned_checkpoint.joblib"),
            "xgb": os.path.join(PROJECT, "checkpoints/ablation_9feat_formal/20260316T013327Z/xgb_tuned_checkpoint.joblib"),
            "mlp": os.path.join(PROJECT, "checkpoints/ablation_9feat_formal/20260316T013327Z/mlp_tuned_checkpoint.pt"),
            "kan": os.path.join(PROJECT, "checkpoints/ablation_9feat_formal/20260316T013327Z/kan_tuned_checkpoint.pt"),
        },
    },
]

TARGETS = ["T_THMs_ug_L", "DBCM_ug_L", "BDCM_ug_L"]
TARGET_LABELS = {"T_THMs_ug_L": "T-THMs", "DBCM_ug_L": "DBCM", "BDCM_ug_L": "BDCM"}

FEATURE_LABELS = {
    "pH": "pH", "UV254_A_cm": "UV254", "temp_C": "Temperature",
    "TOC_mg_L": "TOC", "COD_mg_L": "COD", "Br_mg_L": "Br⁻",
    "NH4_N_mg_L": "NH₄-N", "NO2_N_mg_L": "NO₂-N", "NO3_N_mg_L": "NO₃-N",
}

FEATURE_COLORS = {
    "pH": "#E91E63", "UV254_A_cm": "#9C27B0", "temp_C": "#FF5722",
    "TOC_mg_L": "#2196F3", "COD_mg_L": "#00BCD4", "Br_mg_L": "#4CAF50",
    "NH4_N_mg_L": "#FFC107", "NO2_N_mg_L": "#795548", "NO3_N_mg_L": "#607D8B",
}

MODEL_LABELS = {"rf": "Random Forest", "xgb": "XGBoost", "mlp": "MLP", "kan": "KAN"}

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df[df["split"] == "test"].copy()


def load_checkpoint(path: str) -> dict:
    if path.endswith(".joblib"):
        return joblib.load(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def compute_shap_tree(members, X_test: np.ndarray):
    sv_all = []
    for member in members:
        scaler_x = member["scaler_x"]
        estimator = member["model_state"]["estimator"]
        X_scaled = scaler_x.transform(X_test)
        explainer = shap.TreeExplainer(estimator)
        sv_all.append(explainer.shap_values(X_scaled))
    return np.mean(sv_all, axis=0)


def compute_shap_nn(members, X_test: np.ndarray, model_family: str):
    sv_all = []
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
                return model(torch.from_numpy(X_arr.astype(np.float32))).numpy().ravel()

        background = shap.kmeans(X_scaled, 30)
        explainer = shap.KernelExplainer(predict_fn, background)
        sv_all.append(explainer.shap_values(X_scaled, nsamples=200, silent=True))
        print(f"      Fold {i + 1}/{len(members)} done")

    return np.mean(sv_all, axis=0)


def compute_all_shap(test_df: pd.DataFrame, model_name: str):
    """Compute SHAP for one model type across all ablation rounds and targets."""
    results = {}
    is_tree = model_name in ("rf", "xgb")

    for rnd in ABLATION_ROUNDS:
        label = rnd["label"]
        features = rnd["features"]
        ckpt_path = rnd["ckpts"][model_name]
        ckpt = load_checkpoint(ckpt_path)
        model_family = ckpt["model_family"]
        X_test = test_df[features].values
        results[label] = {"features": features}

        for target in TARGETS:
            print(f"    {label.replace(chr(10), ' ')} | {TARGET_LABELS[target]}...")
            members = ckpt["target_payloads"][target]["members"]
            if is_tree:
                sv = compute_shap_tree(members, X_test)
            else:
                sv = compute_shap_nn(members, X_test, model_family)
            mean_abs = np.abs(sv).mean(axis=0)
            results[label][target] = {
                "shap_values": sv,
                "mean_abs": dict(zip(features, mean_abs.tolist())),
            }
    return results


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def plot_evolution_lines(results: dict, target: str, model_name: str, save_path: str):
    """Line chart: Mean |SHAP| of each feature across ablation rounds."""
    round_labels = [r["label"] for r in ABLATION_ROUNDS]
    all_features = ABLATION_ROUNDS[-1]["features"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for feat in all_features:
        values = []
        x_positions = []
        for i, rl in enumerate(round_labels):
            importance = results[rl].get(target, {}).get("mean_abs", {})
            if feat in importance:
                values.append(importance[feat])
                x_positions.append(i)

        if values:
            ax.plot(x_positions, values,
                    marker="o", linewidth=2.2, markersize=7,
                    label=FEATURE_LABELS.get(feat, feat),
                    color=FEATURE_COLORS.get(feat, "#999999"))

    ax.set_xticks(range(len(round_labels)))
    ax.set_xticklabels(round_labels, fontsize=10)
    ax.set_ylabel("Mean |SHAP value|", fontsize=12)
    ax.set_xlabel("Ablation Round", fontsize=12)
    ax.set_title(f"Feature Importance Evolution — {MODEL_LABELS[model_name]} — {TARGET_LABELS[target]}",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")
    print(f"  Saved: {save_path}")


def plot_stacked_bars(results: dict, target: str, model_name: str, save_path: str):
    """Stacked bar chart: relative feature importance composition per round."""
    round_labels = [r["label"] for r in ABLATION_ROUNDS]

    round_data = []
    for rl in round_labels:
        importance = results[rl].get(target, {}).get("mean_abs", {})
        round_data.append(importance)

    all_features_ordered = ABLATION_ROUNDS[-1]["features"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(round_labels))
    bottom = np.zeros(len(round_labels))

    for feat in all_features_ordered:
        values = []
        for rd in round_data:
            values.append(rd.get(feat, 0))
        values = np.array(values)

        ax.bar(x, values, bottom=bottom, width=0.6,
               label=FEATURE_LABELS.get(feat, feat),
               color=FEATURE_COLORS.get(feat, "#999999"))
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(round_labels, fontsize=10)
    ax.set_ylabel("Cumulative Mean |SHAP value|", fontsize=12)
    ax.set_xlabel("Ablation Round", fontsize=12)
    ax.set_title(f"Feature Importance Composition — {MODEL_LABELS[model_name]} — {TARGET_LABELS[target]}",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")
    print(f"  Saved: {save_path}")


def plot_relative_share(results: dict, target: str, model_name: str, save_path: str):
    """100% stacked bar: relative share of each feature per round."""
    round_labels = [r["label"] for r in ABLATION_ROUNDS]
    all_features_ordered = ABLATION_ROUNDS[-1]["features"]

    round_data = []
    for rl in round_labels:
        importance = results[rl].get(target, {}).get("mean_abs", {})
        round_data.append(importance)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(round_labels))
    bottom = np.zeros(len(round_labels))

    for feat in all_features_ordered:
        raw = np.array([rd.get(feat, 0) for rd in round_data])
        totals = np.array([sum(rd.values()) for rd in round_data])
        totals = np.where(totals == 0, 1, totals)
        pct = raw / totals * 100

        ax.bar(x, pct, bottom=bottom, width=0.6,
               label=FEATURE_LABELS.get(feat, feat),
               color=FEATURE_COLORS.get(feat, "#999999"))
        bottom += pct

    ax.set_xticks(x)
    ax.set_xticklabels(round_labels, fontsize=10)
    ax.set_ylabel("Relative Importance (%)", fontsize=12)
    ax.set_xlabel("Ablation Round", fontsize=12)
    ax.set_title(f"Relative Feature Share — {MODEL_LABELS[model_name]} — {TARGET_LABELS[target]}",
                 fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")
    print(f"  Saved: {save_path}")


def plot_beeswarm_grid(results: dict, test_df: pd.DataFrame, target: str,
                       model_name: str, save_dir: str):
    """Side-by-side beeswarm plots across ablation rounds for one target."""
    round_labels = [r["label"] for r in ABLATION_ROUNDS]
    n_rounds = len(round_labels)

    fig, axes = plt.subplots(1, n_rounds, figsize=(5 * n_rounds, 5))
    if n_rounds == 1:
        axes = [axes]

    for i, (rnd, ax) in enumerate(zip(ABLATION_ROUNDS, axes)):
        rl = rnd["label"]
        features = rnd["features"]
        sv = results[rl][target]["shap_values"]
        X_raw = test_df[features].values
        labels = [FEATURE_LABELS.get(f, f) for f in features]

        plt.sca(ax)
        shap.summary_plot(sv, X_raw, feature_names=labels,
                          show=False, plot_size=None, max_display=9)
        ax.set_title(rl.replace("\n", " "), fontsize=12, fontweight="bold")
        if i > 0:
            ax.set_ylabel("")

    fig.suptitle(f"SHAP Beeswarm Across Rounds — {MODEL_LABELS[model_name]} — {TARGET_LABELS[target]}",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"beeswarm_grid_{target}.png"))
    plt.close("all")
    print(f"  Saved: beeswarm_grid_{target}.png")


def plot_delta_heatmap(results: dict, model_name: str, save_path: str):
    """Heatmap: change in Mean |SHAP| when adding each new feature group."""
    core_features = ["pH", "UV254_A_cm", "temp_C"]
    round_labels = [r["label"] for r in ABLATION_ROUNDS]
    transition_labels = []
    for i in range(1, len(ABLATION_ROUNDS)):
        added = ABLATION_ROUNDS[i]["added"]
        transition_labels.append(f"{round_labels[i-1].split(chr(10))[0]} → {round_labels[i].split(chr(10))[0]}\n({added})")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for t_idx, target in enumerate(TARGETS):
        data = []
        for feat in core_features:
            row = []
            for i in range(1, len(ABLATION_ROUNDS)):
                prev_rl = round_labels[i - 1]
                curr_rl = round_labels[i]
                prev_val = results[prev_rl].get(target, {}).get("mean_abs", {}).get(feat, 0)
                curr_val = results[curr_rl].get(target, {}).get("mean_abs", {}).get(feat, 0)
                if prev_val > 0:
                    delta_pct = (curr_val - prev_val) / prev_val * 100
                else:
                    delta_pct = 0
                row.append(delta_pct)
            data.append(row)

        data = np.array(data)
        feat_labels = [FEATURE_LABELS[f] for f in core_features]

        ax = axes[t_idx]
        vmax = max(abs(data.min()), abs(data.max()), 10)
        im = ax.imshow(data, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

        for y in range(len(core_features)):
            for x in range(len(transition_labels)):
                val = data[y, x]
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(x, y, f"{val:+.1f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)

        ax.set_xticks(range(len(transition_labels)))
        ax.set_xticklabels(transition_labels, fontsize=8.5)
        ax.set_yticks(range(len(feat_labels)))
        ax.set_yticklabels(feat_labels, fontsize=10)
        ax.set_title(TARGET_LABELS[target], fontsize=13, fontweight="bold")

    fig.suptitle(f"Core Feature Importance Change (%) — {MODEL_LABELS[model_name]}",
                 fontsize=15, fontweight="bold")
    fig.colorbar(im, ax=axes, label="Change in Mean |SHAP| (%)", shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")
    print(f"  Saved: {save_path}")


def plot_r2_vs_total_shap(results: dict, model_name: str, save_path: str):
    """Scatter: total SHAP magnitude vs model R² across rounds."""
    from pathlib import Path
    import json

    metric_files = {
        "3 feat": "checkpoints/ablation_3feat_formal/20260315T214708Z/metrics/model_comparison.json",
        "4 feat\n(+TOC)": "checkpoints/ablation_4feat_formal/20260315T222224Z/metrics/model_comparison.json",
        "5 feat\n(+COD)": "checkpoints/ablation_5feat_formal/20260315T231614Z/metrics/model_comparison.json",
        "6 feat\n(+Br)": "checkpoints/ablation_6feat_formal/20260316T004125Z/metrics/model_comparison.json",
        "9 feat\n(+N)": "checkpoints/ablation_9feat_formal/20260316T013327Z/metrics/model_comparison.json",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors_target = {"T_THMs_ug_L": "#2196F3", "DBCM_ug_L": "#FF9800", "BDCM_ug_L": "#4CAF50"}

    for t_idx, target in enumerate(TARGETS):
        ax = axes[t_idx]
        total_shaps = []
        r2_values = []
        labels = []

        for rl, mf in metric_files.items():
            mpath = os.path.join(PROJECT, mf)
            if not os.path.exists(mpath):
                continue
            with open(mpath) as f:
                metrics = json.load(f)

            r2 = metrics["models"].get(model_name, {}).get("target_metrics", {}).get(target, {}).get("r2", None)
            total_shap = sum(results[rl].get(target, {}).get("mean_abs", {}).values())

            if r2 is not None:
                total_shaps.append(total_shap)
                r2_values.append(r2)
                labels.append(rl.split("\n")[0])

        ax.scatter(total_shaps, r2_values, s=100, c=colors_target[target], edgecolors="black", zorder=5)
        for j, lbl in enumerate(labels):
            ax.annotate(lbl, (total_shaps[j], r2_values[j]),
                        textcoords="offset points", xytext=(8, 5), fontsize=9)
        ax.set_xlabel("Total Mean |SHAP|", fontsize=11)
        ax.set_ylabel("Test R²", fontsize=11)
        ax.set_title(TARGET_LABELS[target], fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)

    fig.suptitle(f"Total SHAP Magnitude vs Model Performance — {MODEL_LABELS[model_name]}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close("all")
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    test_df = load_data()

    for model_name in ["rf", "xgb", "mlp", "kan"]:
        print(f"\n{'=' * 70}")
        print(f"Cross-Ablation SHAP: {MODEL_LABELS[model_name]}")
        print(f"{'=' * 70}")

        results = compute_all_shap(test_df, model_name)

        model_dir = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)

        for target in TARGETS:
            plot_evolution_lines(results, target, model_name,
                                os.path.join(model_dir, f"evolution_{target}.png"))
            plot_stacked_bars(results, target, model_name,
                              os.path.join(model_dir, f"stacked_{target}.png"))
            plot_relative_share(results, target, model_name,
                                os.path.join(model_dir, f"relative_{target}.png"))
            plot_beeswarm_grid(results, test_df, target, model_name, model_dir)

        plot_delta_heatmap(results, model_name,
                           os.path.join(model_dir, "delta_heatmap.png"))
        plot_r2_vs_total_shap(results, model_name,
                              os.path.join(model_dir, "r2_vs_shap.png"))

    print(f"\n{'=' * 70}")
    print(f"All cross-ablation SHAP results saved to: {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
