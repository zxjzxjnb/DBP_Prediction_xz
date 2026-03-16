"""Optuna tuning for per-target KAN models on DBP prediction.

Usage::

    python -m dbp_prediction.cli.tune_kan_per_target --trials 30
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dbp_prediction.config import CHECKPOINT_DIR, FEATURE_COLS, TARGET_COLS
from dbp_prediction.engine import PerTargetTuningRequest, run_per_target_tuning_job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune per-target KAN models for DBP prediction")
    parser.add_argument("--trials", type=int, default=int(os.getenv("KAN_TUNE_TRIALS", "60")))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=1400)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--targets", type=str, default=",".join(TARGET_COLS))
    parser.add_argument("--stability-penalty", type=float, default=0.10)
    parser.add_argument(
        "--out",
        type=str,
        default=str(CHECKPOINT_DIR / "kan_tuned_per_target_checkpoint.pt"),
    )
    return parser.parse_args()


def _parse_targets(raw_targets: str) -> list[str]:
    selected = [target.strip() for target in raw_targets.split(",") if target.strip()]
    unknown = sorted(set(selected) - set(TARGET_COLS))
    if unknown:
        raise ValueError(f"Unknown targets: {unknown}. Allowed: {TARGET_COLS}")
    return selected


def main() -> None:
    args = parse_args()
    run_per_target_tuning_job(
        PerTargetTuningRequest(
            model_name="kan",
            feature_cols=list(FEATURE_COLS),
            allowed_targets=list(TARGET_COLS),
            selected_targets=_parse_targets(args.targets),
            base_model_params={},
            training_params={
                "seed": args.seed,
                "max_epochs": args.max_epochs,
                "patience": args.patience,
                "max_grad_norm": 5.0,
                "optimizer": "Adam",
                "loss": "MSE",
                "huber_delta": 1.0,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "batch_size": 16,
                "val_fraction": 0.15,
            },
            tuning_params={
                "trials": args.trials,
                "folds": args.folds,
                "stability_penalty": args.stability_penalty,
            },
            output_path=Path(args.out),
            show_progress_bar=True,
        )
    )


if __name__ == "__main__":
    main()
