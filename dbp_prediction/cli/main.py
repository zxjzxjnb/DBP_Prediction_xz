"""Unified CLI entrypoint for config-driven experiment execution."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dbp_prediction.engine import ExperimentRunner


def _setup_logging(verbosity: int = 1) -> None:
    level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(
        verbosity, logging.DEBUG
    )
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("optuna").setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dbp", description="DBP experiment utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Execute a config-driven experiment run and write artifacts",
    )
    run_parser.add_argument("config", type=str, help="Path to experiment YAML/JSON config")
    run_parser.add_argument("--output-dir", type=str, help="Override the run output directory")
    run_parser.add_argument("--run-id", type=str, help="Optional explicit run identifier")
    run_parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare the run directory and plan without executing models",
    )
    run_parser.add_argument(
        "--skip-data-inspection",
        action="store_true",
        help="Do not read the dataset during preparation",
    )
    run_parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Print the generated plan summary after artifacts are written",
    )

    run_parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=1,
        help="Increase logging verbosity (-v for INFO, -vv for DEBUG)",
    )
    run_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress logging (WARNING level only)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    verbosity = 0 if getattr(args, "quiet", False) else getattr(args, "verbose", 1)
    _setup_logging(verbosity)

    if args.command == "run":
        runner = ExperimentRunner.from_path(args.config)
        common_kwargs = {
            "output_dir": Path(args.output_dir).expanduser() if args.output_dir else None,
            "run_id": args.run_id,
            "inspect_data": not args.skip_data_inspection,
        }

        if args.prepare_only:
            prepared = runner.prepare(**common_kwargs)
            print(f"Prepared run {prepared.run_id}")
            print(f"Output dir: {prepared.output_dir}")
            print(f"Plan: {prepared.plan_path}")
            print(f"Config snapshot: {prepared.config_snapshot_path}")
        else:
            executed = runner.run(**common_kwargs)
            print(f"Executed run {executed.prepared.run_id}")
            print(f"Output dir: {executed.prepared.output_dir}")
            print(f"Plan: {executed.plan_path}")
            print(f"Comparison: {executed.comparison_path}")
            print(f"Manifest: {executed.manifest_path}")
        if args.print_plan:
            print()
            plan = prepared.plan if args.prepare_only else executed.prepared.plan
            print(runner.render_summary(plan).rstrip())


if __name__ == "__main__":
    main()
