"""Compare legacy multi-output and current per-target KAN checkpoints.

Usage::

    python -m dbp_prediction.cli.compare_kan_paradigms
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dbp_prediction.config import CHECKPOINT_DIR, resolve_artifact_path

DEFAULT_MULTI_OUTPUT_CANDIDATES = [
    CHECKPOINT_DIR / "kan_tuned_checkpoint.pt",
    CHECKPOINT_DIR / "kan_tuned_multi_output.pt",
    CHECKPOINT_DIR / "kan_tuned_checkpoint_best.pt",
]
DEFAULT_PER_TARGET_CANDIDATES = [
    CHECKPOINT_DIR / "kan_tuned_per_target_checkpoint.pt",
    CHECKPOINT_DIR / "kan_tuned_per_target.pt",
    CHECKPOINT_DIR / "kan_tuned_per_target_60trials.pt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Historical comparison: legacy multi-output vs per-target KAN",
    )
    parser.add_argument(
        "--multi-output", type=str,
        default=None,
        help="Multi-output KAN checkpoint. Auto-detects standard filenames if omitted.",
    )
    parser.add_argument(
        "--per-target", type=str,
        default=None,
        help="Per-target KAN checkpoint. Auto-detects standard filenames if omitted.",
    )
    return parser.parse_args()


def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(p, map_location="cpu")


def _extract_multi(ckpt: dict) -> dict[str, dict[str, float]]:
    return ckpt["test_metrics"]


def _extract_per_target(ckpt: dict) -> dict[str, dict[str, float]]:
    return {t: p["test_metrics"] for t, p in ckpt["target_payloads"].items()}


def main() -> None:
    args = parse_args()
    multi_path = resolve_artifact_path(
        args.multi_output,
        DEFAULT_MULTI_OUTPUT_CANDIDATES,
        "multi-output KAN checkpoint",
    )
    per_target_path = resolve_artifact_path(
        args.per_target,
        DEFAULT_PER_TARGET_CANDIDATES,
        "per-target KAN checkpoint",
    )

    multi = _extract_multi(_load(str(multi_path)))
    per_target = _extract_per_target(_load(str(per_target_path)))

    targets = sorted(set(multi) & set(per_target))

    print("KAN paradigm comparison on test metrics")
    print(f"  Multi-output: {multi_path}")
    print(f"  Per-target:   {per_target_path}\n")
    print("Delta = per-target − multi-output.  Lower RMSE/MAE better, higher R² better.\n")

    header = (
        f"{'Target':15s} {'MO_RMSE':>9s} {'PT_RMSE':>9s} {'dRMSE':>9s} "
        f"{'MO_MAE':>9s} {'PT_MAE':>9s} {'dMAE':>9s} "
        f"{'MO_R2':>9s} {'PT_R2':>9s} {'dR2':>9s}"
    )
    print(header)
    print("-" * len(header))

    for t in targets:
        mo, pt = multi[t], per_target[t]
        print(
            f"{t:15s} "
            f"{mo['rmse']:9.3f} {pt['rmse']:9.3f} {pt['rmse'] - mo['rmse']:9.3f} "
            f"{mo['mae']:9.3f} {pt['mae']:9.3f} {pt['mae'] - mo['mae']:9.3f} "
            f"{mo['r2']:9.4f} {pt['r2']:9.4f} {pt['r2'] - mo['r2']:9.4f}"
        )

    mo_macro = {m: float(np.mean([multi[t][m] for t in targets])) for m in ("rmse", "mae", "r2")}
    pt_macro = {m: float(np.mean([per_target[t][m] for t in targets])) for m in ("rmse", "mae", "r2")}
    print("\nMacro averages")
    print(f"  Multi-output  RMSE={mo_macro['rmse']:.3f} MAE={mo_macro['mae']:.3f} R²={mo_macro['r2']:.4f}")
    print(f"  Per-target    RMSE={pt_macro['rmse']:.3f} MAE={pt_macro['mae']:.3f} R²={pt_macro['r2']:.4f}")


if __name__ == "__main__":
    main()
