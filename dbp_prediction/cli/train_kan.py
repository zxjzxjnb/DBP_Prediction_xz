"""Train a baseline KAN model for DBP prediction.

Usage::

    python -m dbp_prediction.cli.train_kan
    python -m dbp_prediction.cli.train_kan --seed 2024 --targets T_THMs_ug_L,DBCM_ug_L
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from dbp_prediction.config import CHECKPOINT_DIR, FEATURE_COLS, TARGET_COLS
from dbp_prediction.data import prepare_data
from dbp_prediction.metrics import compute_metrics
from dbp_prediction.models.kan import build_kan
from dbp_prediction.training import set_seed, train_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train per-target baseline KAN models for DBP prediction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--targets", type=str, default=",".join(TARGET_COLS),
                        help="Comma-separated targets to train independently")
    parser.add_argument("--hidden-dims", type=str, default="32,16",
                        help="Comma-separated hidden layer widths")
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--out", type=str,
                        default=str(CHECKPOINT_DIR / "kan_checkpoint.pt"))
    return parser.parse_args()


def _parse_targets(raw_targets: str) -> list[str]:
    selected = [t.strip() for t in raw_targets.split(",") if t.strip()]
    unknown = sorted(set(selected) - set(TARGET_COLS))
    if unknown:
        raise ValueError(f"Unknown targets: {unknown}. Allowed: {TARGET_COLS}")
    return selected


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    selected_targets = _parse_targets(args.targets)
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]
    target_payloads: dict[str, dict] = {}

    print(f"Targets to train independently: {selected_targets}")
    print(f"Features: {len(FEATURE_COLS)}")

    for target_name in selected_targets:
        print("\n" + "=" * 72)
        print(f"Training target: {target_name}")
        print("=" * 72)

        data = prepare_data(
            val_fraction=args.val_fraction,
            seed=args.seed,
            target_cols=[target_name],
        )
        print(f"Train samples: {len(data['train_sub_df'])}, "
              f"Validation: {len(data['val_df'])}, "
              f"Test: {len(data['test_df'])}")

        model = build_kan(
            in_dim=len(FEATURE_COLS),
            out_dim=1,
            hidden_dims=hidden_dims,
            grid=args.grid,
            k=args.k,
            seed=args.seed,
        )
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        model, best_val, best_epoch, _ = train_model(
            model=model,
            X_train=data["X_train"],
            Y_train=data["Y_train"],
            X_val=data["X_val"],
            Y_val=data["Y_val"],
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )

        with torch.no_grad():
            pred_scaled = model(data["X_test"]).numpy()
        pred_raw = data["scaler_y"].inverse_transform(pred_scaled).ravel()
        test_metrics = compute_metrics(data["Y_test_raw"].ravel(), pred_raw)
        print(f"Test: RMSE={test_metrics['rmse']:.3f} "
              f"MAE={test_metrics['mae']:.3f} R²={test_metrics['r2']:.4f}")

        target_payloads[target_name] = {
            "model_state": model.state_dict(),
            "scaler_x": data["scaler_x"],
            "scaler_y": data["scaler_y"],
            "best_val": float(best_val),
            "best_epoch": int(best_epoch),
            "test_metrics": test_metrics,
            "hyperparams": {
                "hidden_dims": hidden_dims,
                "grid": args.grid,
                "k": args.k,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "max_epochs": args.max_epochs,
                "patience": args.patience,
                "val_fraction": args.val_fraction,
            },
        }

    print("\n" + "=" * 72)
    print("Final test summary")
    print("=" * 72)
    for target_name in selected_targets:
        m = target_payloads[target_name]["test_metrics"]
        print(f"  {target_name:15s} RMSE={m['rmse']:7.3f}  "
              f"MAE={m['mae']:7.3f}  R²={m['r2']:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_family": "kan",
        "paradigm": "per_target_baseline",
        "feature_cols": FEATURE_COLS,
        "target_cols": selected_targets,
        "target_payloads": target_payloads,
        "seed": args.seed,
    }, out_path)
    print(f"\nModel saved to {out_path}")


if __name__ == "__main__":
    main()
