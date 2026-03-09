"""
Compare test-set metrics between multi-output and per-target KAN checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multi-output vs per-target KAN checkpoints")
    parser.add_argument(
        "--multi-output",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "kan_tuned_checkpoint.pt"),
        help="Checkpoint produced by scripts/tune_kan.py",
    )
    parser.add_argument(
        "--per-target",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "kan_tuned_per_target_checkpoint.pt"),
        help="Checkpoint produced by scripts/tune_kan_per_target.py",
    )
    return parser.parse_args()


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_checkpoint(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_multi_output_metrics(ckpt: Dict) -> Dict[str, Dict[str, float]]:
    if ckpt.get("paradigm") not in (None, "multi_output"):
        raise ValueError(f"Expected a multi-output checkpoint, got paradigm={ckpt.get('paradigm')!r}.")
    metrics = ckpt.get("test_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Multi-output checkpoint is missing 'test_metrics'.")
    return metrics


def extract_per_target_metrics(ckpt: Dict) -> Dict[str, Dict[str, float]]:
    if ckpt.get("paradigm") not in (None, "per_target"):
        raise ValueError(f"Expected a per-target checkpoint, got paradigm={ckpt.get('paradigm')!r}.")
    payloads = ckpt.get("target_payloads")
    if not isinstance(payloads, dict):
        raise ValueError("Per-target checkpoint is missing 'target_payloads'.")
    return {target: payload["test_metrics"] for target, payload in payloads.items()}


def metric_means(metrics_by_target: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    return {
        metric_name: float(np.mean([metrics[metric_name] for metrics in metrics_by_target.values()]))
        for metric_name in ("rmse", "mae", "r2")
    }


def iter_targets(
    multi_metrics: Dict[str, Dict[str, float]],
    per_target_metrics: Dict[str, Dict[str, float]],
) -> Iterable[Tuple[str, Dict[str, float], Dict[str, float]]]:
    targets = [target for target in multi_metrics if target in per_target_metrics]
    missing_multi = sorted(set(per_target_metrics) - set(multi_metrics))
    missing_per_target = sorted(set(multi_metrics) - set(per_target_metrics))
    if missing_multi or missing_per_target:
        raise ValueError(
            "Target mismatch between checkpoints. "
            f"Missing in multi-output: {missing_multi}; "
            f"missing in per-target: {missing_per_target}"
        )
    for target in targets:
        yield target, multi_metrics[target], per_target_metrics[target]


def print_table(
    multi_metrics: Dict[str, Dict[str, float]],
    per_target_metrics: Dict[str, Dict[str, float]],
) -> None:
    header = (
        f"{'Target':15s} {'MO_RMSE':>9s} {'PT_RMSE':>9s} {'dRMSE':>9s} "
        f"{'MO_MAE':>9s} {'PT_MAE':>9s} {'dMAE':>9s} "
        f"{'MO_R2':>9s} {'PT_R2':>9s} {'dR2':>9s}"
    )
    print(header)
    print("-" * len(header))
    for target, multi_row, per_target_row in iter_targets(multi_metrics, per_target_metrics):
        print(
            f"{target:15s} "
            f"{multi_row['rmse']:9.3f} {per_target_row['rmse']:9.3f} {per_target_row['rmse'] - multi_row['rmse']:9.3f} "
            f"{multi_row['mae']:9.3f} {per_target_row['mae']:9.3f} {per_target_row['mae'] - multi_row['mae']:9.3f} "
            f"{multi_row['r2']:9.4f} {per_target_row['r2']:9.4f} {per_target_row['r2'] - multi_row['r2']:9.4f}"
        )


def main() -> None:
    args = parse_args()
    multi_path = resolve_path(args.multi_output)
    per_target_path = resolve_path(args.per_target)

    multi_ckpt = load_checkpoint(multi_path)
    per_target_ckpt = load_checkpoint(per_target_path)

    multi_metrics = extract_multi_output_metrics(multi_ckpt)
    per_target_metrics = extract_per_target_metrics(per_target_ckpt)

    print("KAN paradigm comparison on test metrics")
    print(f"  Multi-output checkpoint: {multi_path}")
    print(f"  Per-target checkpoint:   {per_target_path}\n")

    print("Delta columns are computed as per-target minus multi-output.")
    print("Lower is better for RMSE/MAE; higher is better for R2.\n")
    print_table(multi_metrics, per_target_metrics)

    multi_macro = metric_means(multi_metrics)
    per_target_macro = metric_means(per_target_metrics)
    print("\nMacro averages")
    print(
        f"  Multi-output  RMSE={multi_macro['rmse']:.3f} "
        f"MAE={multi_macro['mae']:.3f} R2={multi_macro['r2']:.4f}"
    )
    print(
        f"  Per-target    RMSE={per_target_macro['rmse']:.3f} "
        f"MAE={per_target_macro['mae']:.3f} R2={per_target_macro['r2']:.4f}"
    )


if __name__ == "__main__":
    main()
