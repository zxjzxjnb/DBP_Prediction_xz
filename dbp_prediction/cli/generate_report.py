"""Generate a report-ready markdown comparison table from two checkpoints.

Usage::

    python -m dbp_prediction.cli.generate_report
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dbp_prediction.config import CHECKPOINT_DIR, RESULTS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate report-ready markdown comparison table",
    )
    parser.add_argument("--baseline-ckpt", type=str,
                        default=str(CHECKPOINT_DIR / "mlp_tuned_checkpoint_best.pt"))
    parser.add_argument("--baseline-label", type=str, default="Baseline MLP")
    parser.add_argument("--candidate-ckpt", type=str,
                        default=str(CHECKPOINT_DIR / "kan_tuned_per_target_checkpoint.pt"))
    parser.add_argument("--candidate-label", type=str, default="KAN (Per-target)")
    parser.add_argument("--title", type=str,
                        default="Protocol-Matched Comparison (Test Set)")
    parser.add_argument("--out", type=str,
                        default=str(RESULTS_DIR / "protocol_matched_comparison_table.md"))
    return parser.parse_args()


def _load(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _extract(ckpt: dict, label: str = "checkpoint") -> dict[str, dict[str, float]]:
    """Extract per-target test metrics from either checkpoint format.

    Supports:
    - Per-target checkpoints (``target_payloads`` dict)
    - Multi-output checkpoints (``test_metrics`` dict keyed by target name)

    Raises ``ValueError`` with a clear message if neither format is found.
    """
    # Per-target format (tune_mlp, tune_kan_per_target)
    payloads = ckpt.get("target_payloads")
    if isinstance(payloads, dict) and payloads:
        return {t: p["test_metrics"] for t, p in payloads.items()}

    # Multi-output format (tune_kan)
    test_metrics = ckpt.get("test_metrics")
    if isinstance(test_metrics, dict) and test_metrics:
        # Verify it's keyed by target name (not a flat metrics dict)
        first_val = next(iter(test_metrics.values()))
        if isinstance(first_val, dict) and "rmse" in first_val:
            return test_metrics

    raise ValueError(
        f"Cannot extract per-target metrics from {label}. "
        f"Expected 'target_payloads' or 'test_metrics' dict keyed by target name. "
        f"Found keys: {sorted(ckpt.keys())}"
    )


def _better(metric: str, b: float, c: float, bl: str, cl: str) -> str:
    if abs(b - c) < 5e-4:
        return "Tie"
    if metric in ("rmse", "mae"):
        return bl if b < c else cl
    return bl if b > c else cl


def render(title: str, bl: str, cl: str, b_ckpt: dict, c_ckpt: dict,
           b_metrics: dict, c_metrics: dict) -> str:
    targets = [t for t in b_metrics if t in c_metrics]
    lines: list[str] = [f"# {title}", ""]
    lines.append(f"- Baseline: `{bl}`, Candidate: `{cl}`")
    lines.append(f"- Protocol: per-target tuning, {b_ckpt.get('folds')} folds, "
                 f"seed={b_ckpt.get('seed')}")
    lines.append("")

    hdr = (f"| Target | {bl} RMSE | {cl} RMSE | Better | "
           f"{bl} MAE | {cl} MAE | Better | "
           f"{bl} R² | {cl} R² | Better |")
    lines.append(hdr)
    lines.append("| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |")

    for t in targets:
        bm, cm = b_metrics[t], c_metrics[t]
        lines.append(
            f"| {t} | {bm['rmse']:.3f} | {cm['rmse']:.3f} | "
            f"{_better('rmse', bm['rmse'], cm['rmse'], bl, cl)} | "
            f"{bm['mae']:.3f} | {cm['mae']:.3f} | "
            f"{_better('mae', bm['mae'], cm['mae'], bl, cl)} | "
            f"{bm['r2']:.4f} | {cm['r2']:.4f} | "
            f"{_better('r2', bm['r2'], cm['r2'], bl, cl)} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    b_ckpt = _load(args.baseline_ckpt)
    c_ckpt = _load(args.candidate_ckpt)
    md = render(args.title, args.baseline_label, args.candidate_label,
                b_ckpt, c_ckpt,
                _extract(b_ckpt, f"baseline ({args.baseline_ckpt})"),
                _extract(c_ckpt, f"candidate ({args.candidate_ckpt})"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"Wrote report to {out}")


if __name__ == "__main__":
    main()
