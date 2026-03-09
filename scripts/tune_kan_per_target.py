"""
Optuna tuning for per-target KAN models on DBP_dataset_DWTP_B.csv.

This script complements scripts/tune_kan.py:
  - tune_kan.py keeps one multi-output KAN for all 3 targets.
  - tune_kan_per_target.py tunes one KAN ensemble per target.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Keep KAN/matplotlib cache in writable paths.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

from kan import KAN

optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURE_COLS = [
    "pH", "COD_mg_L", "NH4_N_mg_L", "NO2_N_mg_L", "NO3_N_mg_L",
    "Br_mg_L", "TOC_mg_L", "UV254_A_cm", "temp_C",
]
TARGET_COLS = ["T_THMs_ug_L", "DBCM_ug_L", "BDCM_ug_L"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "DBP_dataset_DWTP_B.csv"

HIDDEN_DIMS_MAP = {
    "8": (8,),
    "16": (16,),
    "32": (32,),
    "16-8": (16, 8),
    "24-12": (24, 12),
    "32-16": (32, 16),
}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_kan(in_dim: int, out_dim: int, params: Dict, seed: int) -> nn.Module:
    width = [in_dim] + list(params["hidden_dims"]) + [out_dim]
    return KAN(
        width=width,
        grid=params["grid"],
        k=params["k"],
        base_fun="silu",
        symbolic_enabled=False,
        save_act=False,
        auto_save=False,
        seed=seed,
        device="cpu",
    )


def make_optimizer(model: nn.Module, params: Dict) -> torch.optim.Optimizer:
    if params["optimizer"] == "AdamW":
        return torch.optim.AdamW(
            model.parameters(),
            lr=params["lr"],
            weight_decay=params["weight_decay"],
        )
    return torch.optim.Adam(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )


def train_one_fold(
    model: nn.Module,
    X_tr: torch.Tensor,
    Y_tr: torch.Tensor,
    X_va: torch.Tensor,
    Y_va: torch.Tensor,
    params: Dict,
    max_epochs: int,
    patience: int,
) -> Tuple[nn.Module, float, int, np.ndarray]:
    optimizer = make_optimizer(model, params)
    loss_fn = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X_tr, Y_tr)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params["batch_size"],
        shuffle=True,
    )

    best_val = float("inf")
    best_epoch = 0
    wait = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_mse = loss_fn(model(X_va), Y_va).item()

        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_va).detach().cpu().numpy()

    return model, best_val, best_epoch, pred_scaled


def sample_params(trial: optuna.Trial) -> Dict:
    hidden_dims_key = trial.suggest_categorical(
        "hidden_dims_key",
        list(HIDDEN_DIMS_MAP.keys()),
    )
    return {
        "hidden_dims_key": hidden_dims_key,
        "hidden_dims": HIDDEN_DIMS_MAP[hidden_dims_key],
        "grid": trial.suggest_categorical("grid", [3, 5, 8]),
        "k": trial.suggest_categorical("k", [3, 5]),
        "lr": trial.suggest_float("lr", 2e-4, 8e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "AdamW"]),
    }


def normalize_best_params(params: Dict) -> Dict:
    normalized = params.copy()
    key = normalized.get("hidden_dims_key", normalized.get("hidden_dims"))
    normalized["hidden_dims"] = HIDDEN_DIMS_MAP[key]
    return normalized


def fit_and_eval_fold(
    X_all: np.ndarray,
    y_all: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    params: Dict,
    seed: int,
    max_epochs: int,
    patience: int,
    keep_member: bool,
) -> Dict:
    set_seed(seed)

    scaler_x = StandardScaler().fit(X_all[train_idx])
    scaler_y = StandardScaler().fit(y_all[train_idx])

    X_tr = torch.tensor(scaler_x.transform(X_all[train_idx]), dtype=torch.float32)
    Y_tr = torch.tensor(scaler_y.transform(y_all[train_idx]), dtype=torch.float32)
    X_va = torch.tensor(scaler_x.transform(X_all[val_idx]), dtype=torch.float32)
    Y_va = torch.tensor(scaler_y.transform(y_all[val_idx]), dtype=torch.float32)

    model = build_kan(
        in_dim=X_all.shape[1],
        out_dim=1,
        params=params,
        seed=seed,
    )
    model, val_mse_scaled, best_epoch, pred_scaled = train_one_fold(
        model=model,
        X_tr=X_tr,
        Y_tr=Y_tr,
        X_va=X_va,
        Y_va=Y_va,
        params=params,
        max_epochs=max_epochs,
        patience=patience,
    )

    pred_raw = scaler_y.inverse_transform(pred_scaled)[:, 0]
    y_true = y_all[val_idx, 0]

    result = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred_raw))),
        "mae": float(mean_absolute_error(y_true, pred_raw)),
        "r2": float(r2_score(y_true, pred_raw)),
        "val_mse_scaled": float(val_mse_scaled),
        "best_epoch": int(best_epoch),
    }
    if keep_member:
        result["member"] = {
            "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "scaler_x": scaler_x,
            "scaler_y": scaler_y,
            "best_epoch": int(best_epoch),
        }
    return result


def make_objective(
    X_train_all: np.ndarray,
    y_train_target: np.ndarray,
    seed: int,
    folds: int,
    max_epochs: int,
    patience: int,
    stability_penalty: float,
):
    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial)
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        fold_rmses: List[float] = []

        for fold_id, (tr_idx, va_idx) in enumerate(kf.split(X_train_all), start=1):
            fold_result = fit_and_eval_fold(
                X_all=X_train_all,
                y_all=y_train_target,
                train_idx=tr_idx,
                val_idx=va_idx,
                params=params,
                seed=seed + fold_id,
                max_epochs=max_epochs,
                patience=patience,
                keep_member=False,
            )
            fold_rmses.append(fold_result["rmse"])

            running_rmse = float(np.mean(fold_rmses))
            trial.report(running_rmse, step=fold_id)
            if trial.should_prune():
                raise optuna.TrialPruned()

        rmse_mean = float(np.mean(fold_rmses))
        rmse_std = float(np.std(fold_rmses))
        return rmse_mean + stability_penalty * rmse_std

    return objective


def train_cv_ensemble(
    X_train_all: np.ndarray,
    y_train_target: np.ndarray,
    params: Dict,
    seed: int,
    folds: int,
    max_epochs: int,
    patience: int,
) -> Tuple[List[Dict], Dict]:
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    members: List[Dict] = []
    fold_rmses: List[float] = []
    fold_maes: List[float] = []
    fold_r2s: List[float] = []
    fold_epochs: List[int] = []

    for fold_id, (tr_idx, va_idx) in enumerate(kf.split(X_train_all), start=1):
        fold_result = fit_and_eval_fold(
            X_all=X_train_all,
            y_all=y_train_target,
            train_idx=tr_idx,
            val_idx=va_idx,
            params=params,
            seed=seed + fold_id,
            max_epochs=max_epochs,
            patience=patience,
            keep_member=True,
        )
        members.append(fold_result["member"])
        fold_rmses.append(fold_result["rmse"])
        fold_maes.append(fold_result["mae"])
        fold_r2s.append(fold_result["r2"])
        fold_epochs.append(fold_result["best_epoch"])
        print(
            f"    Fold {fold_id}/{folds} | RMSE={fold_result['rmse']:.3f} "
            f"MAE={fold_result['mae']:.3f} R²={fold_result['r2']:.4f} "
            f"best_epoch={fold_result['best_epoch']}"
        )

    summary = {
        "cv_rmse_mean": float(np.mean(fold_rmses)),
        "cv_rmse_std": float(np.std(fold_rmses)),
        "cv_mae_mean": float(np.mean(fold_maes)),
        "cv_r2_mean": float(np.mean(fold_r2s)),
        "cv_best_epochs": fold_epochs,
    }
    return members, summary


def predict_with_ensemble(members: List[Dict], params: Dict, X_test_raw: np.ndarray) -> np.ndarray:
    preds = []
    for member in members:
        model = build_kan(
            in_dim=X_test_raw.shape[1],
            out_dim=1,
            params=params,
            seed=1,
        )
        model.load_state_dict(member["model_state"])
        model.eval()
        X_te = torch.tensor(member["scaler_x"].transform(X_test_raw), dtype=torch.float32)
        with torch.no_grad():
            pred_scaled = model(X_te).detach().cpu().numpy()
        pred_raw = member["scaler_y"].inverse_transform(pred_scaled)[:, 0]
        preds.append(pred_raw)
    return np.mean(np.vstack(preds), axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune per-target KAN models for DBP prediction")
    parser.add_argument("--trials", type=int, default=int(os.getenv("KAN_TUNE_TRIALS", "60")))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=1400)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--targets",
        type=str,
        default=",".join(TARGET_COLS),
        help="Comma-separated target names to tune (default: all)",
    )
    parser.add_argument(
        "--stability-penalty",
        type=float,
        default=0.10,
        help="Objective = mean_rmse + penalty * std_rmse",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "kan_tuned_per_target_checkpoint.pt"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    df = pd.read_csv(DATA_PATH)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    X_train_all = train_df[FEATURE_COLS].values.astype(np.float32)
    X_test_raw = test_df[FEATURE_COLS].values.astype(np.float32)

    selected_targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    unknown_targets = sorted(set(selected_targets) - set(TARGET_COLS))
    if unknown_targets:
        raise ValueError(f"Unknown targets: {unknown_targets}. Allowed: {TARGET_COLS}")

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Features: {len(FEATURE_COLS)}, Targets: {len(TARGET_COLS)}")
    print(f"Trials per target: {args.trials}, CV folds: {args.folds}")
    print(f"Max epochs: {args.max_epochs}, patience: {args.patience}\n")

    target_payloads: Dict[str, Dict] = {}

    for target_name in selected_targets:
        target_idx = TARGET_COLS.index(target_name)
        print("=" * 72)
        print(f"Tuning target: {target_name}")
        print("=" * 72)

        y_train_target = train_df[[target_name]].values.astype(np.float32)
        y_test_target = test_df[target_name].values

        objective = make_objective(
            X_train_all=X_train_all,
            y_train_target=y_train_target,
            seed=args.seed + target_idx * 1000,
            folds=args.folds,
            max_epochs=args.max_epochs,
            patience=args.patience,
            stability_penalty=args.stability_penalty,
        )

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=args.seed + target_idx),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
        )
        study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

        best = study.best_trial
        best_params = normalize_best_params(best.params.copy())

        print("\nBest trial summary")
        print(f"  objective (RMSE + {args.stability_penalty:g}*std): {best.value:.4f}")
        for k, v in best_params.items():
            print(f"  {k}: {v}")

        print("\nTraining CV ensemble with best params ...")
        members, cv_summary = train_cv_ensemble(
            X_train_all=X_train_all,
            y_train_target=y_train_target,
            params=best_params,
            seed=args.seed + target_idx * 1000,
            folds=args.folds,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )

        y_pred_test = predict_with_ensemble(members, best_params, X_test_raw)

        rmse_test = float(np.sqrt(mean_squared_error(y_test_target, y_pred_test)))
        mae_test = float(mean_absolute_error(y_test_target, y_pred_test))
        r2_test = float(r2_score(y_test_target, y_pred_test))

        print("\nTest metrics (ensemble)")
        print(f"  RMSE={rmse_test:.3f}  MAE={mae_test:.3f}  R²={r2_test:.4f}\n")

        target_payloads[target_name] = {
            "best_params": best_params,
            "best_objective": float(best.value),
            "cv_summary": cv_summary,
            "members": members,
            "test_metrics": {
                "rmse": rmse_test,
                "mae": mae_test,
                "r2": r2_test,
            },
        }

    print("=" * 72)
    print("Final test summary")
    print("=" * 72)
    for target_name in selected_targets:
        metrics = target_payloads[target_name]["test_metrics"]
        print(
            f"  {target_name:15s} RMSE={metrics['rmse']:7.3f} "
            f"MAE={metrics['mae']:7.3f}  R²={metrics['r2']:.4f}"
        )

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_family": "kan",
            "paradigm": "per_target",
            "feature_cols": FEATURE_COLS,
            "target_cols": selected_targets,
            "target_payloads": target_payloads,
            "seed": args.seed,
            "folds": args.folds,
            "trials": args.trials,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "stability_penalty": args.stability_penalty,
        },
        out_path,
    )
    print(f"\nSaved tuned per-target KAN checkpoint to {out_path}")


if __name__ == "__main__":
    main()
