"""Train a baseline KAN model for DBP prediction.

Usage::

    python -m dbp_prediction.cli.train_kan
    python -m dbp_prediction.cli.train_kan --seed 2024 --targets T_THMs_ug_L,DBCM_ug_L
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from dbp_prediction.cli.config_bridge import bind_legacy_model_config
from dbp_prediction.config import CHECKPOINT_DIR, FEATURE_COLS, TARGET_COLS
from dbp_prediction.engine import LegacyTrainingRequest, run_legacy_training_job
from dbp_prediction.settings import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_VAL_FRACTION,
    DEFAULT_WEIGHT_DECAY,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUT_PATH = CHECKPOINT_DIR / "kan_checkpoint.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train per-target baseline KAN models for DBP prediction")
    parser.add_argument("--config", type=str, help="Path to an experiment YAML/JSON config")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--targets", type=str,
                        help="Comma-separated targets to train independently")
    parser.add_argument("--hidden-dims", type=str,
                        help="Comma-separated hidden layer widths")
    parser.add_argument("--grid", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--base-fun", type=str)
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--loss", type=str)
    parser.add_argument("--huber-delta", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--val-fraction", type=float)
    parser.add_argument("--out", type=str)
    return parser.parse_args()


def _get_arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    value = getattr(args, name, None)
    return default if value is None else value


def _parse_targets(raw_targets: str, allowed_targets: list[str]) -> list[str]:
    selected = [t.strip() for t in raw_targets.split(",") if t.strip()]
    unknown = sorted(set(selected) - set(allowed_targets))
    if unknown:
        raise ValueError(f"Unknown targets: {unknown}. Allowed: {allowed_targets}")
    return selected


def _parse_hidden_dims(raw_hidden_dims: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(raw_hidden_dims, str):
        values = [item.strip() for item in raw_hidden_dims.split(",") if item.strip()]
        if not values:
            raise ValueError("At least one hidden layer width is required")
        return [int(value) for value in values]
    return [int(value) for value in raw_hidden_dims]


def main() -> None:
    args = parse_args()
    binding = None
    if getattr(args, "config", None):
        binding = bind_legacy_model_config(args.config, "kan", DEFAULT_OUT_PATH)

    training_cfg = binding.experiment.training if binding else None
    model_params = binding.model.params if binding else {}
    dataset = binding.dataset if binding else None

    seed = int(_get_arg(args, "seed", training_cfg.seed if training_cfg else DEFAULT_SEED))
    feature_cols = list(dataset.features) if dataset else list(FEATURE_COLS)
    allowed_targets = list(dataset.targets) if dataset else list(TARGET_COLS)
    selected_targets = _parse_targets(
        _get_arg(args, "targets", ",".join(binding.selected_targets if binding else TARGET_COLS)),
        allowed_targets,
    )
    hidden_dims = _parse_hidden_dims(
        _get_arg(args, "hidden_dims", model_params.get("hidden_dims", "32,16"))
    )
    grid = int(_get_arg(args, "grid", model_params.get("grid", 8)))
    k = int(_get_arg(args, "k", model_params.get("k", 3)))
    base_fun = str(_get_arg(args, "base_fun", model_params.get("base_fun", "silu")))
    optimizer_name = str(_get_arg(args, "optimizer", training_cfg.optimizer if training_cfg else "Adam"))
    loss_name = str(_get_arg(args, "loss", training_cfg.loss if training_cfg else "MSE"))
    huber_delta = float(_get_arg(args, "huber_delta", training_cfg.huber_delta if training_cfg else 1.0))
    max_grad_norm = float(
        _get_arg(args, "max_grad_norm", training_cfg.max_grad_norm if training_cfg else 5.0)
    )
    lr = float(_get_arg(args, "lr", training_cfg.lr if training_cfg else DEFAULT_LR))
    weight_decay = float(
        _get_arg(args, "weight_decay", training_cfg.weight_decay if training_cfg else DEFAULT_WEIGHT_DECAY)
    )
    batch_size = int(
        _get_arg(args, "batch_size", training_cfg.batch_size if training_cfg else DEFAULT_BATCH_SIZE)
    )
    max_epochs = int(
        _get_arg(args, "max_epochs", training_cfg.max_epochs if training_cfg else DEFAULT_MAX_EPOCHS)
    )
    patience = int(
        _get_arg(args, "patience", training_cfg.patience if training_cfg else DEFAULT_PATIENCE)
    )
    val_fraction = float(
        _get_arg(args, "val_fraction", training_cfg.val_fraction if training_cfg else DEFAULT_VAL_FRACTION)
    )
    out_path = Path(_get_arg(args, "out", str(binding.output_path) if binding else str(DEFAULT_OUT_PATH)))
    save_models = binding.save_models if binding else True
    run_legacy_training_job(
        LegacyTrainingRequest(
            model_name="kan",
            feature_cols=feature_cols,
            allowed_targets=allowed_targets,
            selected_targets=selected_targets,
            model_params={
                "hidden_dims": hidden_dims,
                "grid": grid,
                "k": k,
                "base_fun": base_fun,
            },
            training_params={
                "seed": seed,
                "optimizer": optimizer_name,
                "loss": loss_name,
                "huber_delta": huber_delta,
                "max_grad_norm": max_grad_norm,
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                "patience": patience,
                "val_fraction": val_fraction,
            },
            output_path=out_path,
            save_models=save_models,
            dataset=dataset,
            config_source=str(binding.experiment.source_path) if binding else None,
            feature_steps=(
                [
                    {"name": step.name, "params": dict(step.params)}
                    for step in binding.experiment.features.steps
                ]
                if binding
                else []
            ),
        )
    )


if __name__ == "__main__":
    main()
