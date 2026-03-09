"""Optuna tuning for per-target KAN models on DBP prediction.

Usage::

    python -m dbp_prediction.cli.tune_kan_per_target --trials 30
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
from sklearn.model_selection import KFold

from dbp_prediction.config import CHECKPOINT_DIR, FEATURE_COLS, TARGET_COLS
from dbp_prediction.data import get_train_test_split, load_dataset
from dbp_prediction.metrics import compute_metrics
from dbp_prediction.models.kan import build_kan_from_params
from dbp_prediction.training import (
    fit_and_eval_fold,
    predict_with_ensemble,
    set_seed,
    train_cv_ensemble,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

HIDDEN_DIMS_MAP = {
    "8": (8,),
    "16": (16,),
    "32": (32,),
    "16-8": (16, 8),
    "24-12": (24, 12),
    "32-16": (32, 16),
}


def sample_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample KAN hyperparameters for per-target tuning."""
    hidden_dims_key = trial.suggest_categorical(
        "hidden_dims_key", list(HIDDEN_DIMS_MAP.keys()),
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


def _make_kan_builder(in_dim: int, params: dict[str, Any], seed: int):
    def builder():
        return build_kan_from_params(in_dim, 1, params, seed=seed)
    return builder


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
        builder = _make_kan_builder(X_train_all.shape[1], params, seed)

        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        fold_rmses: list[float] = []

        for fold_id, (tr_idx, va_idx) in enumerate(kf.split(X_train_all), start=1):
            fold_result = fit_and_eval_fold(
                model_builder=builder,
                X_all=X_train_all,
                Y_all=y_train_target,
                train_idx=tr_idx,
                val_idx=va_idx,
                params=params,
                seed=seed + fold_id,
                max_epochs=max_epochs,
                patience=patience,
            )
            fold_rmses.append(fold_result["rmse"])

            trial.report(float(np.mean(fold_rmses)), step=fold_id)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_rmses)) + stability_penalty * float(np.std(fold_rmses))

    return objective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune per-target KAN models for DBP prediction")
    parser.add_argument("--trials", type=int, default=int(os.getenv("KAN_TUNE_TRIALS", "60")))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=1400)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--targets", type=str, default=",".join(TARGET_COLS))
    parser.add_argument("--stability-penalty", type=float, default=0.10)
    parser.add_argument("--out", type=str,
                        default=str(CHECKPOINT_DIR / "kan_tuned_per_target_checkpoint.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    df = load_dataset()
    train_df, test_df = get_train_test_split(df)

    X_train_all = train_df[FEATURE_COLS].values.astype(np.float32)
    X_test_raw = test_df[FEATURE_COLS].values.astype(np.float32)
    selected_targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Trials per target: {args.trials}, CV folds: {args.folds}\n")

    target_payloads: dict[str, dict] = {}

    for target_name in selected_targets:
        target_idx = TARGET_COLS.index(target_name)
        print("=" * 72)
        print(f"Tuning target: {target_name}")
        print("=" * 72)

        y_train = train_df[[target_name]].values.astype(np.float32)
        y_test = test_df[target_name].values

        objective = make_objective(
            X_train_all, y_train,
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

        best_params = study.best_trial.params.copy()
        key = best_params.get("hidden_dims_key")
        best_params["hidden_dims"] = HIDDEN_DIMS_MAP[key]

        print(f"\nBest objective: {study.best_trial.value:.4f}")
        for k, v in best_params.items():
            print(f"  {k}: {v}")

        print("\nTraining CV ensemble with best params ...")
        builder = _make_kan_builder(X_train_all.shape[1], best_params, args.seed + target_idx)
        members, cv_summary = train_cv_ensemble(
            model_builder=builder,
            X_train_all=X_train_all,
            Y_train_all=y_train,
            params=best_params,
            seed=args.seed + target_idx * 1000,
            folds=args.folds,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )

        y_pred_test = predict_with_ensemble(builder, members, X_test_raw)
        if y_pred_test.ndim > 1:
            y_pred_test = y_pred_test[:, 0]

        test_metrics = compute_metrics(y_test, y_pred_test)
        print(f"\nTest: RMSE={test_metrics['rmse']:.3f} "
              f"MAE={test_metrics['mae']:.3f} R²={test_metrics['r2']:.4f}\n")

        target_payloads[target_name] = {
            "best_params": best_params,
            "best_objective": float(study.best_trial.value),
            "cv_summary": cv_summary,
            "members": members,
            "test_metrics": test_metrics,
        }

    print("=" * 72)
    print("Final test summary")
    print("=" * 72)
    for name in selected_targets:
        m = target_payloads[name]["test_metrics"]
        print(f"  {name:15s} RMSE={m['rmse']:7.3f}  MAE={m['mae']:7.3f}  R²={m['r2']:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
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
    }, out_path)
    print(f"\nSaved tuned per-target KAN checkpoint to {out_path}")


if __name__ == "__main__":
    main()
