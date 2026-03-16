"""Generate a report-ready markdown comparison table from two checkpoints.

Usage::

    python -m dbp_prediction.cli.generate_report
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dbp_prediction.config import CHECKPOINT_DIR, RESULTS_DIR, resolve_artifact_path

DEFAULT_BASELINE_CANDIDATES = [
    CHECKPOINT_DIR / "mlp_tuned_checkpoint_best.pt",
    CHECKPOINT_DIR / "mlp_tuned_checkpoint_best_40.pt",
    CHECKPOINT_DIR / "mlp_tuned_checkpoint_best_30.pt",
]
DEFAULT_CANDIDATE_CANDIDATES = [
    CHECKPOINT_DIR / "kan_tuned_per_target_checkpoint.pt",
    CHECKPOINT_DIR / "kan_tuned_per_target.pt",
    CHECKPOINT_DIR / "kan_tuned_per_target_60trials.pt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate report-ready markdown comparison table",
    )
    parser.add_argument("--baseline-ckpt", type=str,
                        default=None,
                        help="Baseline checkpoint. Auto-detects standard filenames if omitted.")
    parser.add_argument("--baseline-label", type=str, default=None,
                        help="Display label for the baseline model. Auto-inferred if omitted.")
    parser.add_argument("--candidate-ckpt", type=str,
                        default=None,
                        help="Candidate checkpoint. Auto-detects standard filenames if omitted.")
    parser.add_argument("--candidate-label", type=str, default=None,
                        help="Display label for the candidate model. Auto-inferred if omitted.")
    parser.add_argument("--title", type=str,
                        default="Protocol-Matched Comparison (Test Set)")
    parser.add_argument("--out", type=str,
                        default=str(RESULTS_DIR / "protocol_matched_comparison_table.md"))
    return parser.parse_args()


def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    if p.suffix == ".joblib":
        import joblib
        return joblib.load(p)
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(p, map_location="cpu")


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


def _infer_label(ckpt: dict, fallback: str) -> str:
    model_family = ckpt.get("model_family")
    paradigm = ckpt.get("paradigm")
    has_target_payloads = isinstance(ckpt.get("target_payloads"), dict)

    if model_family == "kan":
        if paradigm in {"per_target", "per_target_baseline"}:
            return "KAN (Per-target)"
        if paradigm == "multi_output":
            return "KAN (Multi-output)"
        if has_target_payloads:
            return "KAN (Per-target)"
        return "KAN"

    if model_family == "mlp":
        if paradigm in {"per_target", "per_target_baseline"} or has_target_payloads:
            return "MLP (Per-target)"
        return "MLP"

    return fallback


def _describe_protocol(ckpt: dict) -> str:
    parts: list[str] = []

    paradigm = ckpt.get("paradigm")
    payloads = ckpt.get("target_payloads")
    if paradigm == "per_target_baseline":
        parts.append("per-target baseline training")
    elif paradigm == "per_target":
        parts.append("per-target tuning")
    elif paradigm == "multi_output":
        parts.append("multi-output tuning")
    elif isinstance(payloads, dict):
        if ckpt.get("trials") is not None or ckpt.get("folds") is not None:
            parts.append("per-target tuning")
        else:
            parts.append("per-target checkpoint")

    folds = ckpt.get("folds")
    if folds is not None:
        parts.append(f"{folds} folds")

    seed = ckpt.get("seed")
    if seed is not None:
        parts.append(f"seed={seed}")

    trials = ckpt.get("trials")
    if trials is not None:
        parts.append(f"trials={trials}")

    return ", ".join(parts) if parts else "protocol unavailable"


def _better(metric: str, b: float, c: float, bl: str, cl: str) -> str:
    if abs(b - c) < 5e-4:
        return "Tie"
    if metric in ("rmse", "mae"):
        return bl if b < c else cl
    return bl if b > c else cl


def render(title: str, bl: str, cl: str, b_ckpt: dict, c_ckpt: dict,
           b_metrics: dict, c_metrics: dict) -> str:
    targets = [t for t in b_metrics if t in c_metrics]
    if not targets:
        raise ValueError("No overlapping targets found between checkpoints.")

    lines: list[str] = [f"# {title}", ""]
    lines.append(f"- Baseline: `{bl}`, Candidate: `{cl}`")
    baseline_protocol = _describe_protocol(b_ckpt)
    candidate_protocol = _describe_protocol(c_ckpt)
    if baseline_protocol == candidate_protocol:
        lines.append(f"- Protocol: {baseline_protocol}")
    else:
        lines.append(f"- Baseline protocol: {baseline_protocol}")
        lines.append(f"- Candidate protocol: {candidate_protocol}")
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
    baseline_path = resolve_artifact_path(
        args.baseline_ckpt,
        DEFAULT_BASELINE_CANDIDATES,
        "baseline checkpoint",
    )
    candidate_path = resolve_artifact_path(
        args.candidate_ckpt,
        DEFAULT_CANDIDATE_CANDIDATES,
        "candidate checkpoint",
    )

    b_ckpt = _load(str(baseline_path))
    c_ckpt = _load(str(candidate_path))
    baseline_label = args.baseline_label or _infer_label(b_ckpt, "Baseline MLP")
    candidate_label = args.candidate_label or _infer_label(c_ckpt, "Candidate model")

    md = render(args.title, baseline_label, candidate_label,
                b_ckpt, c_ckpt,
                _extract(b_ckpt, f"baseline ({baseline_path})"),
                _extract(c_ckpt, f"candidate ({candidate_path})"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"Wrote report to {out}")


if __name__ == "__main__":
    main()
