from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch


MetricRow = Dict[str, float]
MetricsByTarget = Dict[str, MetricRow]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate a report-ready markdown table from two per-target checkpoints."
    )
    parser.add_argument(
        "--baseline-ckpt",
        type=str,
        default=str(project_root / "checkpoints" / "mlp_tuned_checkpoint_best.pt"),
        help="Per-target baseline checkpoint.",
    )
    parser.add_argument(
        "--baseline-label",
        type=str,
        default="Baseline MLP",
        help="Display label for the baseline model.",
    )
    parser.add_argument(
        "--candidate-ckpt",
        type=str,
        default=str(project_root / "checkpoints" / "kan_tuned_per_target_60trials.pt"),
        help="Per-target candidate checkpoint.",
    )
    parser.add_argument(
        "--candidate-label",
        type=str,
        default="KAN (Per-target)",
        help="Display label for the candidate model.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Protocol-Matched Comparison (Test Set)",
        help="Report title.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(project_root / "results" / "protocol_matched_comparison_table.md"),
        help="Markdown output path.",
    )
    return parser.parse_args()


def load_checkpoint(path: str) -> Dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def extract_metrics(ckpt: Dict) -> MetricsByTarget:
    payloads = ckpt.get("target_payloads")
    if not isinstance(payloads, dict) or not payloads:
        raise ValueError("Checkpoint is missing 'target_payloads'.")
    metrics: MetricsByTarget = {}
    for target, payload in payloads.items():
        target_metrics = payload.get("test_metrics")
        if not isinstance(target_metrics, dict):
            raise ValueError(f"Checkpoint target '{target}' is missing 'test_metrics'.")
        metrics[target] = {
            "rmse": float(target_metrics["rmse"]),
            "mae": float(target_metrics["mae"]),
            "r2": float(target_metrics["r2"]),
        }
    return metrics


def compute_macro(metrics_by_target: MetricsByTarget) -> MetricRow:
    rows = list(metrics_by_target.values())
    return {
        "rmse": sum(row["rmse"] for row in rows) / len(rows),
        "mae": sum(row["mae"] for row in rows) / len(rows),
        "r2": sum(row["r2"] for row in rows) / len(rows),
    }


def better_label(
    metric_name: str,
    baseline_value: float,
    candidate_value: float,
    baseline_label: str,
    candidate_label: str,
) -> str:
    if metric_name in {"rmse", "mae"}:
        if abs(baseline_value - candidate_value) < 5e-4:
            return "Tie"
        return baseline_label if baseline_value < candidate_value else candidate_label
    if abs(baseline_value - candidate_value) < 5e-4:
        return "Tie"
    return baseline_label if baseline_value > candidate_value else candidate_label


def row_for_target(
    target: str,
    baseline_metrics: MetricRow,
    candidate_metrics: MetricRow,
    baseline_label: str,
    candidate_label: str,
) -> str:
    return (
        f"| {target} | "
        f"{baseline_metrics['rmse']:.3f} | {candidate_metrics['rmse']:.3f} | "
        f"{better_label('rmse', baseline_metrics['rmse'], candidate_metrics['rmse'], baseline_label, candidate_label)} | "
        f"{baseline_metrics['mae']:.3f} | {candidate_metrics['mae']:.3f} | "
        f"{better_label('mae', baseline_metrics['mae'], candidate_metrics['mae'], baseline_label, candidate_label)} | "
        f"{baseline_metrics['r2']:.4f} | {candidate_metrics['r2']:.4f} | "
        f"{better_label('r2', baseline_metrics['r2'], candidate_metrics['r2'], baseline_label, candidate_label)} |"
    )


def render_markdown(
    title: str,
    baseline_label: str,
    candidate_label: str,
    baseline_ckpt: Dict,
    candidate_ckpt: Dict,
    baseline_metrics: MetricsByTarget,
    candidate_metrics: MetricsByTarget,
) -> str:
    common_targets = [target for target in baseline_metrics if target in candidate_metrics]
    if not common_targets:
        raise ValueError("No overlapping targets found between checkpoints.")

    baseline_macro = compute_macro({target: baseline_metrics[target] for target in common_targets})
    candidate_macro = compute_macro({target: candidate_metrics[target] for target in common_targets})

    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Baseline: `{baseline_label}`")
    lines.append(f"- Candidate: `{candidate_label}`")
    lines.append(
        f"- Shared protocol: per-target tuning, {baseline_ckpt.get('folds')} folds, "
        f"seed={baseline_ckpt.get('seed')}, stability penalty={baseline_ckpt.get('stability_penalty')}"
    )
    lines.append(
        f"- Search budget: baseline trials={baseline_ckpt.get('trials')}, "
        f"candidate trials={candidate_ckpt.get('trials')}"
    )
    lines.append("")
    lines.append("## Report-Ready Table")
    lines.append("")
    lines.append(
        f"| Target | {baseline_label} RMSE | {candidate_label} RMSE | Better RMSE | "
        f"{baseline_label} MAE | {candidate_label} MAE | Better MAE | "
        f"{baseline_label} R² | {candidate_label} R² | Better R² |"
    )
    lines.append("| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |")

    for target in common_targets:
        lines.append(
            row_for_target(
                target,
                baseline_metrics[target],
                candidate_metrics[target],
                baseline_label,
                candidate_label,
            )
        )
    lines.append(
        row_for_target(
            "Macro Average",
            baseline_macro,
            candidate_macro,
            baseline_label,
            candidate_label,
        )
    )
    lines.append("")
    lines.append("## Suggested Caption")
    lines.append("")
    lines.append(
        f"Comparison between {baseline_label} and {candidate_label} under a protocol-matched "
        f"per-target tuning setup on the held-out test set."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    baseline_ckpt = load_checkpoint(args.baseline_ckpt)
    candidate_ckpt = load_checkpoint(args.candidate_ckpt)
    baseline_metrics = extract_metrics(baseline_ckpt)
    candidate_metrics = extract_metrics(candidate_ckpt)
    markdown = render_markdown(
        title=args.title,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        baseline_ckpt=baseline_ckpt,
        candidate_ckpt=candidate_ckpt,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote markdown report to {out_path}")


if __name__ == "__main__":
    main()
