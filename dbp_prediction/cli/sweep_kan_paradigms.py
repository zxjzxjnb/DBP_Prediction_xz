"""Multi-seed historical sweep comparing KAN paradigms.

Orchestrates legacy ``tune_kan`` and current ``tune_kan_per_target`` across
multiple random seeds and generates CSV/JSON/Markdown reports.

Usage::

    python -m dbp_prediction.cli.sweep_kan_paradigms --seeds 42,2024,3407 --trials 30
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
from typing import Any

import numpy as np
import torch

from dbp_prediction.config import CHECKPOINT_DIR, PROJECT_ROOT, RESULTS_DIR


def log(msg: str = "") -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Historical multi-seed sweep: multi-output vs per-target KAN",
    )
    parser.add_argument("--seeds", type=str, default="42,2024,3407,7777,10086")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=1400)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--stability-penalty", type=float, default=0.10)
    parser.add_argument("--checkpoint-dir", type=str,
                        default=str(CHECKPOINT_DIR / "kan_seed_sweep"))
    parser.add_argument("--report-dir", type=str,
                        default=str(RESULTS_DIR / "kan_seed_sweep"))
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _run(cmd: list[str]) -> None:
    log(f"$ {shlex.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def _build_cmd(module: str, seed: int, out: Path, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable, "-m", module,
        "--trials", str(args.trials),
        "--folds", str(args.folds),
        "--max-epochs", str(args.max_epochs),
        "--patience", str(args.patience),
        "--seed", str(seed),
        "--stability-penalty", str(args.stability_penalty),
        "--out", str(out),
    ]


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _aggregate_rows(seed_rows: list[dict]) -> list[dict]:
    """Group seed rows by target and compute mean +/- std."""
    by_target: dict[str, list[dict]] = {}
    for row in seed_rows:
        by_target.setdefault(row["target"], []).append(row)

    result = []
    for target in sorted(by_target, key=lambda t: (t != "MACRO", t)):
        rows = by_target[target]
        entry: dict[str, Any] = {"target": target, "num_seeds": len(rows)}
        for metric in ("mo_rmse", "pt_rmse", "d_rmse", "mo_mae", "pt_mae",
                        "d_mae", "mo_r2", "pt_r2", "d_r2"):
            vals = [r[metric] for r in rows]
            entry[f"{metric}_mean"] = float(np.mean(vals))
            entry[f"{metric}_std"] = float(np.std(vals))
        result.append(entry)
    return result


def _build_markdown(seed_rows: list[dict], agg: list[dict], meta: dict) -> str:
    """Generate a Markdown report with aggregate and seed-level tables."""
    lines = [
        "# KAN Seed Sweep Summary",
        "",
        f"- Generated: {meta['generated_at']}",
        f"- Seeds: {', '.join(str(s) for s in meta['seeds'])}",
        f"- Trials: {meta['trials']}, Folds: {meta['folds']}",
        f"- Max epochs: {meta['max_epochs']}, Patience: {meta['patience']}",
        "",
        "## Aggregate Summary",
        "",
        "| Target | MO_RMSE | PT_RMSE | dRMSE | MO_R2 | PT_R2 | dR2 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in agg:
        lines.append(
            f"| {row['target']} "
            f"| {_fmt(row['mo_rmse_mean'])}+/-{_fmt(row['mo_rmse_std'])} "
            f"| {_fmt(row['pt_rmse_mean'])}+/-{_fmt(row['pt_rmse_std'])} "
            f"| {_fmt(row['d_rmse_mean'])}+/-{_fmt(row['d_rmse_std'])} "
            f"| {_fmt(row['mo_r2_mean'], 4)}+/-{_fmt(row['mo_r2_std'], 4)} "
            f"| {_fmt(row['pt_r2_mean'], 4)}+/-{_fmt(row['pt_r2_std'], 4)} "
            f"| {_fmt(row['d_r2_mean'], 4)}+/-{_fmt(row['d_r2_std'], 4)} |"
        )

    lines.extend([
        "",
        "## Seed-Level Detail",
        "",
        "| Seed | Target | MO_RMSE | PT_RMSE | dRMSE | MO_R2 | PT_R2 | dR2 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for row in sorted(seed_rows, key=lambda r: (r["seed"], r["target"] != "MACRO", r["target"])):
        lines.append(
            f"| {row['seed']} | {row['target']} "
            f"| {_fmt(row['mo_rmse'])} | {_fmt(row['pt_rmse'])} | {_fmt(row['d_rmse'])} "
            f"| {_fmt(row['mo_r2'], 4)} | {_fmt(row['pt_r2'], 4)} | {_fmt(row['d_r2'], 4)} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    ckpt_dir = Path(args.checkpoint_dir)
    report_dir = Path(args.report_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    seed_rows: list[dict[str, Any]] = []

    for seed in seeds:
        log(f"\n{'=' * 80}\nSeed {seed}\n{'=' * 80}")

        multi_path = ckpt_dir / f"kan_multi_output_seed{seed}.pt"
        pt_path = ckpt_dir / f"kan_per_target_seed{seed}.pt"

        for module, path in [
            ("dbp_prediction.cli.tune_kan", multi_path),
            ("dbp_prediction.cli.tune_kan_per_target", pt_path),
        ]:
            if args.skip_existing and path.exists():
                log(f"Reusing: {path}")
            else:
                _run(_build_cmd(module, seed, path, args))

        mo_metrics = _load(multi_path)["test_metrics"]
        pt_payloads = _load(pt_path)["target_payloads"]
        pt_metrics = {t: p["test_metrics"] for t, p in pt_payloads.items()}

        targets = sorted(mo_metrics)
        for t in targets:
            mo, pt = mo_metrics[t], pt_metrics[t]
            seed_rows.append({
                "seed": seed, "target": t,
                "mo_rmse": mo["rmse"], "pt_rmse": pt["rmse"],
                "d_rmse": pt["rmse"] - mo["rmse"],
                "mo_mae": mo["mae"], "pt_mae": pt["mae"],
                "d_mae": pt["mae"] - mo["mae"],
                "mo_r2": mo["r2"], "pt_r2": pt["r2"],
                "d_r2": pt["r2"] - mo["r2"],
            })

        # MACRO row per seed
        macro_mo = {m: float(np.mean([mo_metrics[t][m] for t in targets]))
                    for m in ("rmse", "mae", "r2")}
        macro_pt = {m: float(np.mean([pt_metrics[t][m] for t in targets]))
                    for m in ("rmse", "mae", "r2")}
        seed_rows.append({
            "seed": seed, "target": "MACRO",
            "mo_rmse": macro_mo["rmse"], "pt_rmse": macro_pt["rmse"],
            "d_rmse": macro_pt["rmse"] - macro_mo["rmse"],
            "mo_mae": macro_mo["mae"], "pt_mae": macro_pt["mae"],
            "d_mae": macro_pt["mae"] - macro_mo["mae"],
            "mo_r2": macro_mo["r2"], "pt_r2": macro_pt["r2"],
            "d_r2": macro_pt["r2"] - macro_mo["r2"],
        })

    # Aggregate across seeds
    agg = _aggregate_rows(seed_rows)

    meta: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "trials": args.trials,
        "folds": args.folds,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "stability_penalty": args.stability_penalty,
    }

    # Write seed-level CSV
    seed_fields = ["seed", "target", "mo_rmse", "pt_rmse", "d_rmse",
                   "mo_mae", "pt_mae", "d_mae", "mo_r2", "pt_r2", "d_r2"]
    _write_csv(report_dir / "seed_level_summary.csv", seed_rows, seed_fields)

    # Write aggregate CSV
    agg_fields = ["target", "num_seeds"]
    for m in ("mo_rmse", "pt_rmse", "d_rmse", "mo_mae", "pt_mae",
              "d_mae", "mo_r2", "pt_r2", "d_r2"):
        agg_fields.extend([f"{m}_mean", f"{m}_std"])
    _write_csv(report_dir / "aggregate_summary.csv", agg, agg_fields)

    # Write JSON manifest
    (report_dir / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")

    # Write Markdown report
    (report_dir / "report.md").write_text(_build_markdown(seed_rows, agg, meta))

    # Console summary
    log("\nAggregate summary across seeds")
    for row in agg:
        log(
            f"  {row['target']:15s} "
            f"MO_RMSE={row['mo_rmse_mean']:.3f}+/-{row['mo_rmse_std']:.3f} "
            f"PT_RMSE={row['pt_rmse_mean']:.3f}+/-{row['pt_rmse_std']:.3f} "
            f"MO_R2={row['mo_r2_mean']:.4f}+/-{row['mo_r2_std']:.4f} "
            f"PT_R2={row['pt_r2_mean']:.4f}+/-{row['pt_r2_std']:.4f}"
        )

    log(f"\nWrote seed CSV:      {report_dir / 'seed_level_summary.csv'}")
    log(f"Wrote aggregate CSV: {report_dir / 'aggregate_summary.csv'}")
    log(f"Wrote manifest:      {report_dir / 'manifest.json'}")
    log(f"Wrote Markdown:      {report_dir / 'report.md'}")


if __name__ == "__main__":
    main()
