"""Train a baseline MLP model for DBP prediction.

Usage::

    python -m dbp_prediction.cli.train_mlp
    python -m dbp_prediction.cli.train_mlp --seed 2024 --hidden-dims 64,32
"""

from __future__ import annotations

import argparse
import logging

import torch

from dbp_prediction.config import CHECKPOINT_DIR, FEATURE_COLS, TARGET_COLS
from dbp_prediction.data import prepare_data
from dbp_prediction.metrics import compute_per_target_metrics, print_metrics_table
from dbp_prediction.models.mlp import MLP
from dbp_prediction.training import set_seed, train_model

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline MLP for DBP prediction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dims", type=str, default="32,16",
                        help="Comma-separated hidden layer widths")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--out", type=str,
                        default=str(CHECKPOINT_DIR / "mlp_checkpoint.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]

    # ── Load data ────────────────────────────────────────────────────────
    data = prepare_data(val_fraction=args.val_fraction, seed=args.seed)

    print(f"Train samples: {len(data['train_sub_df'])}, "
          f"Validation: {len(data['val_df'])}, "
          f"Test: {len(data['test_df'])}")
    print(f"Features: {len(FEATURE_COLS)}, Targets: {len(TARGET_COLS)}")

    # ── Build model ──────────────────────────────────────────────────────
    model = MLP(
        in_dim=len(FEATURE_COLS),
        out_dim=len(TARGET_COLS),
        hidden_dims=hidden_dims,
        dropout=args.dropout,
    )
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(model)

    # ── Train ────────────────────────────────────────────────────────────
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

    # ── Evaluate ─────────────────────────────────────────────────────────
    with torch.no_grad():
        pred_scaled = model(data["X_test"]).numpy()
    pred_raw = data["scaler_y"].inverse_transform(pred_scaled)

    per_target = compute_per_target_metrics(
        data["Y_test_raw"], pred_raw, TARGET_COLS,
    )
    print_metrics_table(per_target)

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = args.out
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "scaler_x": data["scaler_x"],
        "scaler_y": data["scaler_y"],
        "feature_cols": FEATURE_COLS,
        "target_cols": TARGET_COLS,
        "hyperparams": {
            "hidden_dims": hidden_dims,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
    }, out_path)
    print(f"\nModel saved to {out_path}")


if __name__ == "__main__":
    main()
