"""
Robust Optuna tuning for the MLP baseline on DBP_dataset_DWTP_B.csv.

Improvements over the previous script:
  - Cleans duplicated code paths and keeps one deterministic pipeline.
  - Tunes each target independently (T_THMs, DBCM, BDCM).
  - Uses 5-fold CV objective on original-scale RMSE (with light stability penalty).
  - Trains a CV ensemble with best params for stronger generalization on small data.
  - Keeps test set strictly for final one-shot evaluation.
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

optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURE_COLS = [
    "pH", "COD_mg_L", "NH4_N_mg_L", "NO2_N_mg_L", "NO3_N_mg_L",
    "Br_mg_L", "TOC_mg_L", "UV254_A_cm", "temp_C",
]
TARGET_COLS = ["T_THMs_ug_L", "DBCM_ug_L", "BDCM_ug_L"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "DBP_dataset_DWTP_B.csv"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_model(
    in_dim: int,
    out_dim: int,
    n_layers: int,
    hidden_dim: int,
    dropout: float,
    activation_name: str,
) -> nn.Module:
    if n_layers == 0:
        return nn.Linear(in_dim, out_dim)

    act_map = {
        "ReLU": nn.ReLU,
        "LeakyReLU": nn.LeakyReLU,
        "SiLU": nn.SiLU,
        "Tanh": nn.Tanh,
    }
    act_fn = act_map[activation_name]

    layers: List[nn.Module] = []
    prev = in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(prev, hidden_dim), act_fn(), nn.Dropout(dropout)]
        prev = hidden_dim
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


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


def make_train_loss(params: Dict) -> nn.Module:
    if params["loss"] == "Huber":
        return nn.SmoothL1Loss(beta=params["huber_delta"])
    return nn.MSELoss()


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
    train_loss_fn = make_train_loss(params)
    val_loss_fn = nn.MSELoss()  # use scaled MSE for early stopping stability

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
            loss = train_loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_va)
            val_mse = val_loss_fn(val_pred, Y_va).item()

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
        pred_best = model(X_va).detach().cpu().numpy()

    return model, best_val, best_epoch, pred_best


def sample_params(trial: optuna.Trial) -> Dict:
    n_layers = trial.suggest_int("n_layers", 0, 3)

    if n_layers == 0:
        hidden_dim = 0
        dropout = 0.0
        activation = "ReLU"
    else:
        hidden_dim = trial.suggest_categorical("hidden_dim", [8, 16, 24, 32, 48, 64])
        dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.05)
        activation = trial.suggest_categorical("activation", ["ReLU", "LeakyReLU", "SiLU", "Tanh"])

    params = {
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "activation": activation,
        "lr": trial.suggest_float("lr", 3e-4, 2e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 2e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "AdamW"]),
        "loss": trial.suggest_categorical("loss", ["MSE", "Huber"]),
    }

    if params["loss"] == "Huber":
        params["huber_delta"] = trial.suggest_categorical("huber_delta", [0.5, 1.0, 2.0, 4.0])
    else:
        params["huber_delta"] = 1.0

    return params


def normalize_best_params(params: Dict) -> Dict:
    """Backfill optional keys for configurations that skip hidden layers."""
    normalized = params.copy()
    n_layers = int(normalized.get("n_layers", 0))

    if n_layers == 0:
        normalized.setdefault("hidden_dim", 0)
        normalized.setdefault("dropout", 0.0)
        normalized.setdefault("activation", "ReLU")

    normalized.setdefault("huber_delta", 1.0)
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

    model = build_model(
        in_dim=X_all.shape[1],
        out_dim=1,
        n_layers=params["n_layers"],
        hidden_dim=params["hidden_dim"],
        dropout=params["dropout"],
        activation_name=params["activation"],
    )

    model, val_mse_scaled, best_epoch, pred_scaled = train_one_fold(
        model,
        X_tr,
        Y_tr,
        X_va,
        Y_va,
        params,
        max_epochs=max_epochs,
        patience=patience,
    )

    pred_raw = scaler_y.inverse_transform(pred_scaled)[:, 0]
    y_true = y_all[val_idx, 0]

    rmse = float(np.sqrt(mean_squared_error(y_true, pred_raw)))
    mae = float(mean_absolute_error(y_true, pred_raw))
    r2 = float(r2_score(y_true, pred_raw))

    result = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
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

        # Light penalty discourages highly unstable folds.
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
        model = build_model(
            in_dim=X_test_raw.shape[1],
            out_dim=1,
            n_layers=params["n_layers"],
            hidden_dim=params["hidden_dim"],
            dropout=params["dropout"],
            activation_name=params["activation"],
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
    parser = argparse.ArgumentParser(description="Tune MLP baseline for DBP prediction")
    parser.add_argument("--trials", type=int, default=int(os.getenv("MLP_TUNE_TRIALS", "120")))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=2000)
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
        default=str(PROJECT_ROOT / "checkpoints" / "mlp_tuned_checkpoint_best.pt"),
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
    test_preds_all = {}

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
            pruner=optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=2),
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
        test_preds_all[target_name] = y_pred_test

    print("=" * 72)
    print("Final test summary")
    print("=" * 72)
    for target_name in selected_targets:
        m = target_payloads[target_name]["test_metrics"]
        print(f"  {target_name:15s} RMSE={m['rmse']:7.3f}  MAE={m['mae']:7.3f}  R²={m['r2']:.4f}")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
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
    print(f"\nSaved tuned ensemble checkpoint to {out_path}")


if __name__ == "__main__":
    main()
