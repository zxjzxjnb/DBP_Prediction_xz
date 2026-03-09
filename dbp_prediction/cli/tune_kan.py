"""Optuna tuning for a multi-output KAN on DBP prediction.

Usage::

    python -m dbp_prediction.cli.tune_kan --trials 30
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
from dbp_prediction.metrics import compute_per_target_metrics, print_metrics_table
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
    """Sample KAN hyperparameters from the Optuna search space."""
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


def _make_kan_builder(in_dim: int, out_dim: int, params: dict[str, Any], seed: int):
    """Return a zero-arg callable that builds a fresh KAN from params."""
    def builder():
        return build_kan_from_params(in_dim, out_dim, params, seed=seed)
    return builder


def make_objective(
    X_train_all: np.ndarray,
    Y_train_all: np.ndarray,
    seed: int,
    folds: int,
    max_epochs: int,
    patience: int,
    stability_penalty: float,
):
    """Create an Optuna objective for multi-output KAN tuning."""

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial)
        builder = _make_kan_builder(
            X_train_all.shape[1], Y_train_all.shape[1], params, seed,
        )

        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        fold_scores: list[float] = []

        for fold_id, (tr_idx, va_idx) in enumerate(kf.split(X_train_all), start=1):
            fold_result = fit_and_eval_fold(
                model_builder=builder,
                X_all=X_train_all,
                Y_all=Y_train_all,
                train_idx=tr_idx,
                val_idx=va_idx,
                params=params,
                seed=seed + fold_id,
                max_epochs=max_epochs,
                patience=patience,
            )
            fold_scores.append(fold_result["rmse"])

            trial.report(float(np.mean(fold_scores)), step=fold_id)
            if trial.should_prune():
                raise optuna.TrialPruned()

        score_mean = float(np.mean(fold_scores))
        score_std = float(np.std(fold_scores))
        return score_mean + stability_penalty * score_std

    return objective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune a multi-output KAN for DBP prediction")
    parser.add_argument("--trials", type=int, default=int(os.getenv("KAN_TUNE_TRIALS", "60")))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=1400)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stability-penalty", type=float, default=0.10)
    parser.add_argument("--out", type=str,
                        default=str(CHECKPOINT_DIR / "kan_tuned_checkpoint.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    df = load_dataset()
    train_df, test_df = get_train_test_split(df)

    X_train_all = train_df[FEATURE_COLS].values.astype(np.float32)
    Y_train_all = train_df[TARGET_COLS].values.astype(np.float32)
    X_test_raw = test_df[FEATURE_COLS].values.astype(np.float32)
    Y_test_raw = test_df[TARGET_COLS].values

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Trials: {args.trials}, CV folds: {args.folds}")
    print(f"Max epochs: {args.max_epochs}, patience: {args.patience}\n")

    objective = make_objective(
        X_train_all, Y_train_all,
        seed=args.seed, folds=args.folds,
        max_epochs=args.max_epochs, patience=args.patience,
        stability_penalty=args.stability_penalty,
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best_params = study.best_trial.params.copy()
    key = best_params.get("hidden_dims_key", best_params.get("hidden_dims"))
    best_params["hidden_dims"] = HIDDEN_DIMS_MAP[key]

    print(f"\nBest objective: {study.best_trial.value:.4f}")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    print("\nTraining CV ensemble with best params ...")
    builder = _make_kan_builder(
        X_train_all.shape[1], Y_train_all.shape[1], best_params, args.seed,
    )
    members, cv_summary = train_cv_ensemble(
        model_builder=builder,
        X_train_all=X_train_all,
        Y_train_all=Y_train_all,
        params=best_params,
        seed=args.seed, folds=args.folds,
        max_epochs=args.max_epochs, patience=args.patience,
    )

    Y_pred_test = predict_with_ensemble(builder, members, X_test_raw)

    test_metrics_dict = compute_per_target_metrics(Y_test_raw, Y_pred_test, TARGET_COLS)
    print_metrics_table(test_metrics_dict)

    # Flatten for checkpoint format compatibility
    test_metrics = {
        target: test_metrics_dict[target] for target in TARGET_COLS
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_family": "kan",
        "paradigm": "multi_output",
        "feature_cols": FEATURE_COLS,
        "target_cols": TARGET_COLS,
        "best_params": best_params,
        "best_objective": float(study.best_trial.value),
        "cv_summary": cv_summary,
        "members": members,
        "test_metrics": test_metrics,
        "seed": args.seed,
        "trials": args.trials,
        "folds": args.folds,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "stability_penalty": args.stability_penalty,
    }, out_path)
    print(f"\nSaved tuned KAN checkpoint to {out_path}")


if __name__ == "__main__":
    main()
