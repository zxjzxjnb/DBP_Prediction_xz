"""
Run a multi-seed comparison between multi-output and per-target KAN paradigms.

For each seed this script:
  1. runs scripts/tune_kan.py
  2. runs scripts/tune_kan_per_target.py
  3. loads both checkpoints
  4. writes seed-level and aggregate summaries

Use --skip-existing to resume long sweeps without recomputing finished seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def log(message: str = "") -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep KAN paradigms across multiple random seeds")
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,2024,3407,7777,10086",
        help="Comma-separated random seeds",
    )
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=1400)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--stability-penalty", type=float, default=0.10)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "kan_seed_sweep"),
        help="Directory to store per-seed checkpoints",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "kan_seed_sweep"),
        help="Directory to store CSV/JSON/Markdown summaries",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose checkpoint already exists",
    )
    return parser.parse_args()


def parse_seeds(raw: str) -> List[int]:
    seeds: List[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        seeds.append(int(chunk))
    if not seeds:
        raise ValueError("At least one seed must be provided.")
    return seeds


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_checkpoint(path: Path) -> Dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_multi_metrics(ckpt: Dict) -> Dict[str, Dict[str, float]]:
    if ckpt.get("paradigm") not in (None, "multi_output"):
        raise ValueError(f"Expected a multi-output checkpoint, got {ckpt.get('paradigm')!r}.")
    metrics = ckpt.get("test_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Multi-output checkpoint missing 'test_metrics'.")
    return metrics


def extract_per_target_metrics(ckpt: Dict) -> Dict[str, Dict[str, float]]:
    if ckpt.get("paradigm") not in (None, "per_target"):
        raise ValueError(f"Expected a per-target checkpoint, got {ckpt.get('paradigm')!r}.")
    payloads = ckpt.get("target_payloads")
    if not isinstance(payloads, dict):
        raise ValueError("Per-target checkpoint missing 'target_payloads'.")
    return {target: payload["test_metrics"] for target, payload in payloads.items()}


def metric_mean(metrics_by_target: Dict[str, Dict[str, float]], metric: str) -> float:
    return float(np.mean([row[metric] for row in metrics_by_target.values()]))


def build_command(script_name: str, seed: int, out_path: Path, args: argparse.Namespace) -> List[str]:
    return [
        sys.executable,
        str(SCRIPTS_DIR / script_name),
        "--trials",
        str(args.trials),
        "--folds",
        str(args.folds),
        "--max-epochs",
        str(args.max_epochs),
        "--patience",
        str(args.patience),
        "--seed",
        str(seed),
        "--stability-penalty",
        str(args.stability_penalty),
        "--out",
        str(out_path),
    ]


def run_command(command: List[str]) -> None:
    log(f"$ {shlex.join(command)}")
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def ensure_checkpoint(
    script_name: str,
    seed: int,
    out_path: Path,
    args: argparse.Namespace,
) -> None:
    if args.skip_existing and out_path.exists():
        log(f"Reusing existing checkpoint: {out_path}")
        return
    run_command(build_command(script_name, seed, out_path, args))


def compare_targets(
    multi_metrics: Dict[str, Dict[str, float]],
    per_target_metrics: Dict[str, Dict[str, float]],
) -> List[str]:
    targets = sorted(multi_metrics)
    if sorted(per_target_metrics) != targets:
        raise ValueError(
            "Target mismatch between checkpoints. "
            f"multi={sorted(multi_metrics)} per_target={sorted(per_target_metrics)}"
        )
    return targets


def collect_seed_rows(seed: int, multi_metrics: Dict[str, Dict[str, float]], per_target_metrics: Dict[str, Dict[str, float]]) -> List[Dict]:
    rows: List[Dict] = []
    targets = compare_targets(multi_metrics, per_target_metrics)

    for target in targets:
        mo = multi_metrics[target]
        pt = per_target_metrics[target]
        rows.append(
            {
                "seed": seed,
                "target": target,
                "mo_rmse": float(mo["rmse"]),
                "pt_rmse": float(pt["rmse"]),
                "d_rmse": float(pt["rmse"] - mo["rmse"]),
                "mo_mae": float(mo["mae"]),
                "pt_mae": float(pt["mae"]),
                "d_mae": float(pt["mae"] - mo["mae"]),
                "mo_r2": float(mo["r2"]),
                "pt_r2": float(pt["r2"]),
                "d_r2": float(pt["r2"] - mo["r2"]),
            }
        )

    rows.append(
        {
            "seed": seed,
            "target": "MACRO",
            "mo_rmse": metric_mean(multi_metrics, "rmse"),
            "pt_rmse": metric_mean(per_target_metrics, "rmse"),
            "d_rmse": metric_mean(per_target_metrics, "rmse") - metric_mean(multi_metrics, "rmse"),
            "mo_mae": metric_mean(multi_metrics, "mae"),
            "pt_mae": metric_mean(per_target_metrics, "mae"),
            "d_mae": metric_mean(per_target_metrics, "mae") - metric_mean(multi_metrics, "mae"),
            "mo_r2": metric_mean(multi_metrics, "r2"),
            "pt_r2": metric_mean(per_target_metrics, "r2"),
            "d_r2": metric_mean(per_target_metrics, "r2") - metric_mean(multi_metrics, "r2"),
        }
    )
    return rows


def aggregate_rows(seed_rows: List[Dict]) -> List[Dict]:
    by_target: Dict[str, List[Dict]] = {}
    for row in seed_rows:
        by_target.setdefault(row["target"], []).append(row)

    aggregate = []
    for target in sorted(by_target, key=lambda name: (name != "MACRO", name)):
        rows = by_target[target]
        aggregate.append(
            {
                "target": target,
                "num_seeds": len(rows),
                "mo_rmse_mean": float(np.mean([row["mo_rmse"] for row in rows])),
                "mo_rmse_std": float(np.std([row["mo_rmse"] for row in rows])),
                "pt_rmse_mean": float(np.mean([row["pt_rmse"] for row in rows])),
                "pt_rmse_std": float(np.std([row["pt_rmse"] for row in rows])),
                "d_rmse_mean": float(np.mean([row["d_rmse"] for row in rows])),
                "d_rmse_std": float(np.std([row["d_rmse"] for row in rows])),
                "mo_mae_mean": float(np.mean([row["mo_mae"] for row in rows])),
                "mo_mae_std": float(np.std([row["mo_mae"] for row in rows])),
                "pt_mae_mean": float(np.mean([row["pt_mae"] for row in rows])),
                "pt_mae_std": float(np.std([row["pt_mae"] for row in rows])),
                "d_mae_mean": float(np.mean([row["d_mae"] for row in rows])),
                "d_mae_std": float(np.std([row["d_mae"] for row in rows])),
                "mo_r2_mean": float(np.mean([row["mo_r2"] for row in rows])),
                "mo_r2_std": float(np.std([row["mo_r2"] for row in rows])),
                "pt_r2_mean": float(np.mean([row["pt_r2"] for row in rows])),
                "pt_r2_std": float(np.std([row["pt_r2"] for row in rows])),
                "d_r2_mean": float(np.mean([row["d_r2"] for row in rows])),
                "d_r2_std": float(np.std([row["d_r2"] for row in rows])),
            }
        )
    return aggregate


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def build_markdown(seed_rows: List[Dict], aggregate_rows_: List[Dict], meta: Dict) -> str:
    lines = [
        "# KAN Seed Sweep Summary",
        "",
        f"- Generated at: {meta['generated_at_utc']}",
        f"- Seeds: {', '.join(str(seed) for seed in meta['seeds'])}",
        f"- Trials: {meta['trials']}",
        f"- Folds: {meta['folds']}",
        f"- Max epochs: {meta['max_epochs']}",
        f"- Patience: {meta['patience']}",
        f"- Stability penalty: {meta['stability_penalty']}",
        "",
        "## Aggregate Summary",
        "",
        "| Target | MO_RMSE(mean+/-std) | PT_RMSE(mean+/-std) | dRMSE(mean+/-std) | MO_MAE(mean+/-std) | PT_MAE(mean+/-std) | dMAE(mean+/-std) | MO_R2(mean+/-std) | PT_R2(mean+/-std) | dR2(mean+/-std) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in aggregate_rows_:
        lines.append(
            "| {target} | {mo_rmse} | {pt_rmse} | {d_rmse} | {mo_mae} | {pt_mae} | {d_mae} | {mo_r2} | {pt_r2} | {d_r2} |".format(
                target=row["target"],
                mo_rmse=f"{format_float(row['mo_rmse_mean'])}+/-{format_float(row['mo_rmse_std'])}",
                pt_rmse=f"{format_float(row['pt_rmse_mean'])}+/-{format_float(row['pt_rmse_std'])}",
                d_rmse=f"{format_float(row['d_rmse_mean'])}+/-{format_float(row['d_rmse_std'])}",
                mo_mae=f"{format_float(row['mo_mae_mean'])}+/-{format_float(row['mo_mae_std'])}",
                pt_mae=f"{format_float(row['pt_mae_mean'])}+/-{format_float(row['pt_mae_std'])}",
                d_mae=f"{format_float(row['d_mae_mean'])}+/-{format_float(row['d_mae_std'])}",
                mo_r2=f"{format_float(row['mo_r2_mean'], 4)}+/-{format_float(row['mo_r2_std'], 4)}",
                pt_r2=f"{format_float(row['pt_r2_mean'], 4)}+/-{format_float(row['pt_r2_std'], 4)}",
                d_r2=f"{format_float(row['d_r2_mean'], 4)}+/-{format_float(row['d_r2_std'], 4)}",
            )
        )

    lines.extend(
        [
            "",
            "## Seed-Level Summary",
            "",
            "| Seed | Target | MO_RMSE | PT_RMSE | dRMSE | MO_MAE | PT_MAE | dMAE | MO_R2 | PT_R2 | dR2 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in sorted(seed_rows, key=lambda item: (item["seed"], item["target"] != "MACRO", item["target"])):
        lines.append(
            "| {seed} | {target} | {mo_rmse} | {pt_rmse} | {d_rmse} | {mo_mae} | {pt_mae} | {d_mae} | {mo_r2} | {pt_r2} | {d_r2} |".format(
                seed=row["seed"],
                target=row["target"],
                mo_rmse=format_float(row["mo_rmse"]),
                pt_rmse=format_float(row["pt_rmse"]),
                d_rmse=format_float(row["d_rmse"]),
                mo_mae=format_float(row["mo_mae"]),
                pt_mae=format_float(row["pt_mae"]),
                d_mae=format_float(row["d_mae"]),
                mo_r2=format_float(row["mo_r2"], 4),
                pt_r2=format_float(row["pt_r2"], 4),
                d_r2=format_float(row["d_r2"], 4),
            )
        )
    return "\n".join(lines) + "\n"


def print_aggregate_summary(rows: List[Dict]) -> None:
    header = (
        f"{'Target':15s} {'MO_RMSE':>17s} {'PT_RMSE':>17s} {'dRMSE':>17s} "
        f"{'MO_MAE':>17s} {'PT_MAE':>17s} {'dMAE':>17s} "
        f"{'MO_R2':>17s} {'PT_R2':>17s} {'dR2':>17s}"
    )
    log("\nAggregate summary across seeds")
    log(header)
    log("-" * len(header))
    for row in rows:
        log(
            f"{row['target']:15s} "
            f"{row['mo_rmse_mean']:8.3f}+/-{row['mo_rmse_std']:<7.3f} "
            f"{row['pt_rmse_mean']:8.3f}+/-{row['pt_rmse_std']:<7.3f} "
            f"{row['d_rmse_mean']:8.3f}+/-{row['d_rmse_std']:<7.3f} "
            f"{row['mo_mae_mean']:8.3f}+/-{row['mo_mae_std']:<7.3f} "
            f"{row['pt_mae_mean']:8.3f}+/-{row['pt_mae_std']:<7.3f} "
            f"{row['d_mae_mean']:8.3f}+/-{row['d_mae_std']:<7.3f} "
            f"{row['mo_r2_mean']:8.4f}+/-{row['mo_r2_std']:<7.4f} "
            f"{row['pt_r2_mean']:8.4f}+/-{row['pt_r2_std']:<7.4f} "
            f"{row['d_r2_mean']:8.4f}+/-{row['d_r2_std']:<7.4f}"
        )


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    checkpoint_dir = resolve_path(args.checkpoint_dir)
    report_dir = resolve_path(args.report_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    seed_rows: List[Dict] = []
    manifest_runs = []

    for seed in seeds:
        log("\n" + "=" * 80)
        log(f"Seed {seed}")
        log("=" * 80)

        multi_path = checkpoint_dir / f"kan_multi_output_seed{seed}.pt"
        per_target_path = checkpoint_dir / f"kan_per_target_seed{seed}.pt"

        ensure_checkpoint("tune_kan.py", seed, multi_path, args)
        ensure_checkpoint("tune_kan_per_target.py", seed, per_target_path, args)

        multi_ckpt = load_checkpoint(multi_path)
        per_target_ckpt = load_checkpoint(per_target_path)
        multi_metrics = extract_multi_metrics(multi_ckpt)
        per_target_metrics = extract_per_target_metrics(per_target_ckpt)

        seed_rows.extend(collect_seed_rows(seed, multi_metrics, per_target_metrics))
        manifest_runs.append(
            {
                "seed": seed,
                "multi_output_checkpoint": str(multi_path),
                "per_target_checkpoint": str(per_target_path),
            }
        )

    aggregate = aggregate_rows(seed_rows)
    generated_at = datetime.now(timezone.utc).isoformat()
    meta = {
        "generated_at_utc": generated_at,
        "seeds": seeds,
        "trials": args.trials,
        "folds": args.folds,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "stability_penalty": args.stability_penalty,
        "checkpoint_dir": str(checkpoint_dir),
        "report_dir": str(report_dir),
        "runs": manifest_runs,
    }

    seed_csv = report_dir / "seed_level_summary.csv"
    aggregate_csv = report_dir / "aggregate_summary.csv"
    manifest_json = report_dir / "manifest.json"
    report_md = report_dir / "report.md"

    write_csv(
        seed_csv,
        seed_rows,
        [
            "seed",
            "target",
            "mo_rmse",
            "pt_rmse",
            "d_rmse",
            "mo_mae",
            "pt_mae",
            "d_mae",
            "mo_r2",
            "pt_r2",
            "d_r2",
        ],
    )
    write_csv(
        aggregate_csv,
        aggregate,
        [
            "target",
            "num_seeds",
            "mo_rmse_mean",
            "mo_rmse_std",
            "pt_rmse_mean",
            "pt_rmse_std",
            "d_rmse_mean",
            "d_rmse_std",
            "mo_mae_mean",
            "mo_mae_std",
            "pt_mae_mean",
            "pt_mae_std",
            "d_mae_mean",
            "d_mae_std",
            "mo_r2_mean",
            "mo_r2_std",
            "pt_r2_mean",
            "pt_r2_std",
            "d_r2_mean",
            "d_r2_std",
        ],
    )
    manifest_json.write_text(json.dumps(meta, indent=2) + "\n")
    report_md.write_text(build_markdown(seed_rows, aggregate, meta))

    print_aggregate_summary(aggregate)
    log(f"\nWrote seed summary: {seed_csv}")
    log(f"Wrote aggregate summary: {aggregate_csv}")
    log(f"Wrote manifest: {manifest_json}")
    log(f"Wrote markdown report: {report_md}")


if __name__ == "__main__":
    main()
