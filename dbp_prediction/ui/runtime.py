"""Runtime helpers for executing experiments from the UI."""

from __future__ import annotations

import io
import logging
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dbp_prediction.engine import ExperimentRunner
from dbp_prediction.schemas import ExperimentConfig


@dataclass
class ExecutionResult:
    """A prepared or executed run returned to the UI layer."""

    mode: str
    run_id: str
    output_dir: Path
    summary_path: Path
    plan_path: Path
    config_snapshot_path: Path
    manifest_path: Path | None
    comparison_path: Path | None
    logs: str


def execute_payload(
    payload: dict[str, Any],
    *,
    run_id: str | None = None,
    prepare_only: bool = False,
    inspect_data: bool = True,
) -> ExecutionResult:
    """Run an experiment config payload through the shared runner."""
    config = ExperimentConfig.from_dict(payload)
    runner = ExperimentRunner(config=config)

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.addHandler(handler)
    root_logger.setLevel(min(previous_level, logging.INFO) if previous_level else logging.INFO)
    logging.getLogger("optuna").setLevel(logging.WARNING)

    try:
        with redirect_stdout(stream):
            if prepare_only:
                prepared = runner.prepare(run_id=run_id, inspect_data=inspect_data)
                result = ExecutionResult(
                    mode="prepare",
                    run_id=prepared.run_id,
                    output_dir=prepared.output_dir,
                    summary_path=prepared.summary_path,
                    plan_path=prepared.plan_path,
                    config_snapshot_path=prepared.config_snapshot_path,
                    manifest_path=None,
                    comparison_path=None,
                    logs=stream.getvalue(),
                )
            else:
                executed = runner.run(run_id=run_id, inspect_data=inspect_data)
                result = ExecutionResult(
                    mode=executed.mode,
                    run_id=executed.prepared.run_id,
                    output_dir=executed.prepared.output_dir,
                    summary_path=executed.summary_path,
                    plan_path=executed.plan_path,
                    config_snapshot_path=executed.prepared.config_snapshot_path,
                    manifest_path=executed.manifest_path,
                    comparison_path=executed.comparison_path,
                    logs=stream.getvalue(),
                )
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(previous_level)

    return result
