"""
Optuna tuning for KAN on DBP_dataset_DWTP_B.csv.

Fairness constraints (aligned with train_mlp.py/train_kan.py):
  - Only training split is used for hyperparameter search.
  - Feature/target scalers are fit on each fold's training subset only.
  - Test split is used once at the very end for final reporting.
"""

from __future__ import annotations

import argparse
import os
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
        return torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    return torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])


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
        ["8", "16", "32", "16-8", "24-12", "32-16"],
    )
    hidden_dims_map = {
        "8": (8,),
        "16": (16,),
        "32": (32,),
        "16-8": (16, 8),
        "24-12": (24, 12),
        "32-16": (32, 16),
    }
    hidden_dims = hidden_dims_map[hidden_dims_key]

    return {
        "hidden_dims_key": hidden_dims_key,
        "hidden_dims": hidden_dims,
        "grid": trial.suggest_categorical("grid", [3, 5, 8]),
        "k": trial.suggest_categorical("k", [3, 5]),
        "lr": trial.suggest_float("lr", 2e-4, 8e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "AdamW"]),
    }


def fold_metrics_raw(y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> Dict:
    rmse_per_target = []
    mae_per_target = []
    r2_per_target = []
    for i in range(y_true_raw.shape[1]):
        rmse_per_target.append(float(np.sqrt(mean_squared_error(y_true_raw[:, i], y_pred_raw[:, i]))))
        mae_per_target.append(float(mean_absolute_error(y_true_raw[:, i], y_pred_raw[:, i])))
        r2_per_target.append(float(r2_score(y_true_raw[:, i], y_pred_raw[:, i])))

    return {
        "rmse_per_target": rmse_per_target,
        "mae_per_target": mae_per_target,
        "r2_per_target": r2_per_target,
        "rmse_macro": float(np.mean(rmse_per_target)),
        "mae_macro": float(np.mean(mae_per_target)),
        "r2_macro": float(np.mean(r2_per_target)),
    }


def fit_and_eval_fold(
    X_all: np.ndarray,
    Y_all: np.ndarray,
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
    scaler_y = StandardScaler().fit(Y_all[train_idx])

    X_tr = torch.tensor(scaler_x.transform(X_all[train_idx]), dtype=torch.float32)
    Y_tr = torch.tensor(scaler_y.transform(Y_all[train_idx]), dtype=torch.float32)
    X_va = torch.tensor(scaler_x.transform(X_all[val_idx]), dtype=torch.float32)
    Y_va = torch.tensor(scaler_y.transform(Y_all[val_idx]), dtype=torch.float32)

    model = build_kan(in_dim=X_all.shape[1], out_dim=Y_all.shape[1], params=params, seed=seed)
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

    y_pred_raw = scaler_y.inverse_transform(pred_scaled)
    y_true_raw = Y_all[val_idx]
    metrics = fold_metrics_raw(y_true_raw, y_pred_raw)

    result = {
        "val_mse_scaled": float(val_mse_scaled),
        "best_epoch": int(best_epoch),
        **metrics,
    }
    if keep_member:
        result["member"] = {
            "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "scaler_x": scaler_x,
            "scaler_y": scaler_y,
        }
    return result


def make_objective(
    X_train_all: np.ndarray,
    Y_train_all: np.ndarray,
    seed: int,
    folds: int,
    max_epochs: int,
    patience: int,
    stability_penalty: float,
):
    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial)
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        fold_scores: List[float] = []

        for fold_id, (tr_idx, va_idx) in enumerate(kf.split(X_train_all), start=1):
            fold_result = fit_and_eval_fold(
                X_all=X_train_all,
                Y_all=Y_train_all,
                train_idx=tr_idx,
                val_idx=va_idx,
                params=params,
                seed=seed + fold_id,
                max_epochs=max_epochs,
                patience=patience,
                keep_member=False,
            )
            fold_scores.append(fold_result["rmse_macro"])

            running = float(np.mean(fold_scores))
            trial.report(running, step=fold_id)
            if trial.should_prune():
                raise optuna.TrialPruned()

        score_mean = float(np.mean(fold_scores))
        score_std = float(np.std(fold_scores))
        return score_mean + stability_penalty * score_std

    return objective


def train_cv_ensemble(
    X_train_all: np.ndarray,
    Y_train_all: np.ndarray,
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
            Y_all=Y_train_all,
            train_idx=tr_idx,
            val_idx=va_idx,
            params=params,
            seed=seed + fold_id,
            max_epochs=max_epochs,
            patience=patience,
            keep_member=True,
        )
        members.append(fold_result["member"])
        fold_rmses.append(fold_result["rmse_macro"])
        fold_maes.append(fold_result["mae_macro"])
        fold_r2s.append(fold_result["r2_macro"])
        fold_epochs.append(fold_result["best_epoch"])
        print(
            f"    Fold {fold_id}/{folds} | RMSE(macro)={fold_result['rmse_macro']:.3f} "
            f"MAE(macro)={fold_result['mae_macro']:.3f} "
            f"R²(macro)={fold_result['r2_macro']:.4f} "
            f"best_epoch={fold_result['best_epoch']}"
        )

    summary = {
        "cv_rmse_macro_mean": float(np.mean(fold_rmses)),
        "cv_rmse_macro_std": float(np.std(fold_rmses)),
        "cv_mae_macro_mean": float(np.mean(fold_maes)),
        "cv_r2_macro_mean": float(np.mean(fold_r2s)),
        "cv_best_epochs": fold_epochs,
    }
    return members, summary


def predict_with_ensemble(members: List[Dict], params: Dict, X_test_raw: np.ndarray, out_dim: int) -> np.ndarray:
    preds = []
    for member in members:
        model = build_kan(
            in_dim=X_test_raw.shape[1],
            out_dim=out_dim,
            params=params,
            seed=1,
        )
        model.load_state_dict(member["model_state"])
        model.eval()
        X_te = torch.tensor(member["scaler_x"].transform(X_test_raw), dtype=torch.float32)
        with torch.no_grad():
            pred_scaled = model(X_te).detach().cpu().numpy()
        pred_raw = member["scaler_y"].inverse_transform(pred_scaled)
        preds.append(pred_raw)
    return np.mean(np.stack(preds, axis=0), axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune KAN baseline for DBP prediction")
    parser.add_argument("--trials", type=int, default=int(os.getenv("KAN_TUNE_TRIALS", "60")))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=1400)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stability-penalty", type=float, default=0.10)
    parser.add_argument("--out", type=str, default="kan_tuned_checkpoint.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    df = pd.read_csv("DBP_dataset_DWTP_B.csv")
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    X_train_all = train_df[FEATURE_COLS].values.astype(np.float32)
    Y_train_all = train_df[TARGET_COLS].values.astype(np.float32)
    X_test_raw = test_df[FEATURE_COLS].values.astype(np.float32)
    Y_test_raw = test_df[TARGET_COLS].values

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Features: {len(FEATURE_COLS)}, Targets: {len(TARGET_COLS)}")
    print(f"Trials: {args.trials}, CV folds: {args.folds}")
    print(f"Max epochs: {args.max_epochs}, patience: {args.patience}\n")

    objective = make_objective(
        X_train_all=X_train_all,
        Y_train_all=Y_train_all,
        seed=args.seed,
        folds=args.folds,
        max_epochs=args.max_epochs,
        patience=args.patience,
        stability_penalty=args.stability_penalty,
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best = study.best_trial
    best_params = best.params.copy()
    hidden_dims_map = {
        "8": (8,),
        "16": (16,),
        "32": (32,),
        "16-8": (16, 8),
        "24-12": (24, 12),
        "32-16": (32, 16),
    }
    key = best_params.get("hidden_dims_key", best_params.get("hidden_dims"))
    best_params["hidden_dims"] = hidden_dims_map[key]

    print("\nBest trial summary")
    print(f"  objective (RMSE_macro + penalty*std): {best.value:.4f}")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    print("\nTraining CV ensemble with best params ...")
    members, cv_summary = train_cv_ensemble(
        X_train_all=X_train_all,
        Y_train_all=Y_train_all,
        params=best_params,
        seed=args.seed,
        folds=args.folds,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )

    Y_pred_test = predict_with_ensemble(
        members=members,
        params=best_params,
        X_test_raw=X_test_raw,
        out_dim=Y_train_all.shape[1],
    )

    print("\n" + "=" * 60)
    print("Final evaluation on test set (original scale)")
    print("=" * 60)
    test_metrics = {}
    for i, target in enumerate(TARGET_COLS):
        y_true = Y_test_raw[:, i]
        y_pred = Y_pred_test[:, i]
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        print(f"  {target:15s}  RMSE={rmse:7.3f}  MAE={mae:7.3f}  R²={r2:.4f}")
        test_metrics[target] = {"rmse": rmse, "mae": mae, "r2": r2}

    torch.save(
        {
            "feature_cols": FEATURE_COLS,
            "target_cols": TARGET_COLS,
            "best_params": best_params,
            "best_objective": float(best.value),
            "cv_summary": cv_summary,
            "members": members,
            "test_metrics": test_metrics,
            "seed": args.seed,
            "trials": args.trials,
            "folds": args.folds,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "stability_penalty": args.stability_penalty,
        },
        args.out,
    )
    print(f"\nSaved tuned KAN checkpoint to {args.out}")


if __name__ == "__main__":
    main()
