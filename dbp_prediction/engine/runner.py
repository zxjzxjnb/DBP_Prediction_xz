"""Experiment runner for config-driven orchestration."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dbp_prediction.artifacts import ArtifactStore
from dbp_prediction.datasets import export_predictions, load_dataset
from dbp_prediction.engine._data_helpers import dataset_payload
from dbp_prediction.engine.evaluator import build_model_evaluation, summarize_model_comparison
from dbp_prediction.engine.legacy_trainer import (
    LegacyTrainingRequest,
    LegacyTrainingResult,
    run_legacy_training_job,
)
from dbp_prediction.features import FeaturePipeline
from dbp_prediction.models import get_model_adapter
from dbp_prediction.schemas import ExperimentConfig, load_experiment_config
from dbp_prediction.settings import RESULTS_DIR

logger = logging.getLogger(__name__)

DEFAULT_RUNS_DIR = RESULTS_DIR / "experiments"
LEGACY_TRAIN_COMMANDS = {
    "mlp": "python -m dbp_prediction.cli.train_mlp",
    "kan": "python -m dbp_prediction.cli.train_kan",
}


@dataclass
class PreparedRun:
    """Artifacts created during experiment preparation."""

    run_id: str
    output_dir: Path
    config_snapshot_path: Path
    plan_path: Path
    dataset_snapshot_path: Path
    summary_path: Path
    plan: dict[str, Any]
    dataset_snapshot: dict[str, Any]


@dataclass
class ExecutedRun:
    """Artifacts created during a full experiment execution."""

    prepared: PreparedRun
    mode: str
    model_results: dict[str, Any]
    metrics_paths: dict[str, Path]
    comparison_path: Path
    prediction_paths: dict[str, Path]
    model_paths: dict[str, Path]
    manifest_path: Path
    plan_path: Path
    summary_path: Path
    comparison: dict[str, Any]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ExperimentRunner:
    """Prepare and describe an experiment run from a validated config."""

    def __init__(
        self,
        config: ExperimentConfig,
        config_path: Path | None = None,
    ) -> None:
        self.config = config
        self.config_path = (
            Path(config_path or config.source_path).expanduser().resolve()
            if (config_path or config.source_path)
            else None
        )

    @classmethod
    def from_path(cls, config_path: str | Path) -> "ExperimentRunner":
        config = load_experiment_config(config_path)
        return cls(config=config, config_path=config.source_path)

    def make_run_id(self, now: datetime | None = None) -> str:
        timestamp = (now or _utc_now()).astimezone(timezone.utc)
        return timestamp.strftime("%Y%m%dT%H%M%SZ")

    def resolve_output_dir(
        self,
        run_id: str,
        output_dir: str | Path | None = None,
    ) -> Path:
        if output_dir is not None:
            return Path(output_dir).expanduser()

        if self.config.outputs.dir is not None:
            return self.config.outputs.dir / run_id

        config_stem = self.config_path.stem if self.config_path else "adhoc_experiment"
        return DEFAULT_RUNS_DIR / config_stem / run_id

    def inspect_dataset(self) -> dict[str, Any]:
        from dbp_prediction.datasets import get_train_test_split

        dataset = self.config.dataset
        df = load_dataset(
            path=dataset.path,
            feature_cols=dataset.features,
            target_cols=dataset.targets,
            split_col=dataset.split.column,
            file_format=dataset.format,
            read_options=dataset.reader_options,
        )

        split_counts = {}
        train_rows = None
        test_rows = None
        if dataset.split.strategy in {"predefined", "column"}:
            train_df, test_df = get_train_test_split(
                df,
                split_col=dataset.split.column,
                train_label=dataset.split.train_label,
                test_label=dataset.split.test_label,
            )
            train_rows = len(train_df)
            test_rows = len(test_df)
            split_counts = dict(Counter(str(value) for value in df[dataset.split.column].tolist()))

        return {
            "path": str(dataset.path),
            "format": dataset.format,
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "feature_count": int(len(dataset.features)),
            "target_count": int(len(dataset.targets)),
            "columns": list(df.columns),
            "selected_targets": self.config.selected_targets(),
            "split": {
                "strategy": dataset.split.strategy,
                "column": dataset.split.column,
                "train_label": dataset.split.train_label,
                "test_label": dataset.split.test_label,
                "counts": split_counts,
                "train_rows": train_rows,
                "test_rows": test_rows,
            },
        }

    def build_legacy_command_hints(self) -> list[dict[str, Any]]:
        hints: list[dict[str, Any]] = []
        config_ref = str(self.config_path) if self.config_path else "<config-path>"
        enabled_family_counts = Counter(
            model.name
            for model in self.config.models
            if model.enabled
        )
        legacy_compatible = (
            self.config.task.strategy == "per_target"
            and not self.config.tuning.enabled
            and not self.config.outputs.save_predictions
            and self.config.dataset.split.strategy in {"predefined", "column"}
        )

        for model in self.config.models:
            hint = {
                "name": model.name,
                "alias": model.alias,
                "enabled": model.enabled,
                "supported": False,
                "reason": None,
                "command": None,
            }

            if not model.enabled:
                hint["reason"] = "model is disabled"
            elif model.name not in LEGACY_TRAIN_COMMANDS:
                hint["reason"] = "no legacy CLI is available for this model family yet"
            elif enabled_family_counts[model.name] > 1:
                hint["reason"] = "multiple enabled configs from this model family are ambiguous for legacy CLIs"
            elif not legacy_compatible:
                hint["reason"] = "experiment uses config features not supported by legacy CLIs"
            else:
                hint["supported"] = True
                hint["command"] = f"{LEGACY_TRAIN_COMMANDS[model.name]} --config {config_ref}"

            hints.append(hint)
        return hints

    def build_plan(
        self,
        run_id: str,
        output_dir: Path,
        dataset_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "status": "prepared",
            "phase": "phase7_unified_cli",
            "run_id": run_id,
            "prepared_at_utc": _utc_now().isoformat(),
            "config_source": str(self.config_path) if self.config_path else None,
            "output_dir": str(output_dir),
            "task": {
                "strategy": self.config.task.strategy,
                "selected_targets": self.config.selected_targets(),
            },
            "dataset": dataset_snapshot,
            "features": {
                "step_count": len(self.config.features.steps),
                "steps": [
                    {"name": step.name, "params": step.params}
                    for step in self.config.features.steps
                ],
            },
            "models": [
                {
                    "name": model.name,
                    "alias": model.alias,
                    "enabled": model.enabled,
                    "params": model.params,
                }
                for model in self.config.models
            ],
            "training": {
                "seed": self.config.training.seed,
                "max_epochs": self.config.training.max_epochs,
                "patience": self.config.training.patience,
                "batch_size": self.config.training.batch_size,
                "lr": self.config.training.lr,
                "weight_decay": self.config.training.weight_decay,
                "val_fraction": self.config.training.val_fraction,
                "optimizer": self.config.training.optimizer,
                "loss": self.config.training.loss,
            },
            "tuning": {
                "enabled": self.config.tuning.enabled,
                "trials": self.config.tuning.trials,
                "folds": self.config.tuning.folds,
                "stability_penalty": self.config.tuning.stability_penalty,
            },
            "outputs": {
                "dir": str(output_dir),
                "save_models": self.config.outputs.save_models,
                "save_predictions": self.config.outputs.save_predictions,
            },
            "legacy_command_hints": self.build_legacy_command_hints(),
        }

    def render_summary(
        self,
        plan: dict[str, Any],
    ) -> str:
        dataset = plan["dataset"]
        dataset_rows = dataset.get("row_count", dataset.get("status", "unknown"))
        execution = plan.get("execution", {})
        is_completed = plan.get("status") == "completed"
        lines = [
            "# Experiment Run" if is_completed else "# Experiment Preparation",
            "",
            f"- Status: `{plan['status']}`",
            f"- Run ID: `{plan['run_id']}`",
            f"- Output dir: `{plan['output_dir']}`",
            f"- Task: `{plan['task']['strategy']}`",
            f"- Models: {', '.join(model['name'] for model in plan['models'] if model['enabled'])}",
            f"- Targets: {', '.join(plan['task']['selected_targets'])}",
            f"- Feature steps: {plan['features']['step_count']}",
            f"- Dataset rows: {dataset_rows}",
        ]

        if is_completed:
            lines.append(f"- Mode: `{execution.get('mode', 'unknown')}`")
            best_model = execution.get("comparison", {}).get("best_by_macro_rmse")
            if best_model:
                lines.append(f"- Best by macro RMSE: `{best_model}`")

        split = dataset.get("split")
        if (
            isinstance(split, dict)
            and split.get("train_rows") is not None
            and split.get("test_rows") is not None
        ):
            lines.append(
                f"- Split: {split['train_rows']} train / {split['test_rows']} test "
                f"via `{split['column']}`"
            )

        supported = [
            hint["command"]
            for hint in plan["legacy_command_hints"]
            if hint["supported"] and hint["command"] is not None
        ]
        if supported:
            lines.extend(
                [
                    "",
                    "## Legacy Commands",
                    "",
                    *[f"- `{command}`" for command in supported],
                ]
            )

        return "\n".join(lines) + "\n"

    def _enabled_models(self) -> list[Any]:
        return [model for model in self.config.models if model.enabled]

    def _feature_step_specs(self) -> list[dict[str, Any]]:
        return [
            {"name": step.name, "params": dict(step.params)}
            for step in self.config.features.steps
        ]

    def _shared_training_params(self) -> dict[str, Any]:
        return {
            "seed": self.config.training.seed,
            "optimizer": self.config.training.optimizer,
            "loss": self.config.training.loss,
            "huber_delta": self.config.training.huber_delta,
            "max_grad_norm": self.config.training.max_grad_norm,
            "lr": self.config.training.lr,
            "weight_decay": self.config.training.weight_decay,
            "batch_size": self.config.training.batch_size,
            "max_epochs": self.config.training.max_epochs,
            "patience": self.config.training.patience,
            "val_fraction": self.config.training.val_fraction,
        }

    def _validate_execution_support(self) -> None:
        if self.config.task.strategy != "per_target":
            raise ValueError(
                "dbp run currently supports task.strategy='per_target' only. "
                f"Received '{self.config.task.strategy}'."
            )

    def _run_training_suite(self, output_dir: Path) -> dict[str, LegacyTrainingResult]:
        self._validate_execution_support()

        enabled_models = self._enabled_models()
        label_counts = Counter(model.alias or model.name for model in enabled_models)
        results: dict[str, LegacyTrainingResult] = {}
        feature_steps = self._feature_step_specs()
        training_params = self._shared_training_params()
        selected_targets = self.config.selected_targets()

        for model in enabled_models:
            label = model.alias or model.name
            if label_counts[label] > 1:
                raise ValueError(
                    f"Duplicate enabled model label '{label}' requires unique aliases for dbp run output naming"
                )

            adapter = get_model_adapter(model.name)
            ext = adapter.checkpoint_extension
            results[label] = run_legacy_training_job(
                LegacyTrainingRequest(
                    model_name=model.name,
                    feature_cols=list(self.config.dataset.features),
                    allowed_targets=list(self.config.dataset.targets),
                    selected_targets=selected_targets,
                    model_params=dict(model.params),
                    training_params=dict(training_params),
                    output_path=output_dir / f"{label}_checkpoint{ext}",
                    save_models=self.config.outputs.save_models,
                    dataset=self.config.dataset,
                    config_source=str(self.config_path) if self.config_path else None,
                    feature_steps=feature_steps,
                )
            )

        return results

    def _build_model_evaluations(
        self,
        model_results: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        return {
            label: build_model_evaluation(
                label=label,
                model_family=result.model_name,
                target_payloads=result.target_payloads,
                paradigm=str(result.checkpoint_payload.get("paradigm", "per_target")),
            )
            for label, result in model_results.items()
        }

    def _write_metrics_artifacts(
        self,
        store: ArtifactStore,
        model_evaluations: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, Path], Path, dict[str, Any]]:
        metrics_paths: dict[str, Path] = {}
        for label, evaluation in model_evaluations.items():
            metrics_paths[label] = store.write_json(
                Path("metrics") / f"{label}.json",
                evaluation,
            )

        comparison = summarize_model_comparison(model_evaluations)
        comparison_path = store.write_json(
            Path("metrics") / "model_comparison.json",
            comparison,
        )
        return metrics_paths, comparison_path, comparison

    def _write_prediction_artifacts(
        self,
        output_dir: Path,
        model_results: dict[str, Any],
    ) -> dict[str, Path]:
        if not self.config.outputs.save_predictions:
            return {}

        prediction_paths: dict[str, Path] = {}
        for label, result in model_results.items():
            prediction_paths[label] = export_predictions(
                output_dir / "predictions" / f"{label}.csv",
                result.test_outputs,
            )
        return prediction_paths

    def _build_manifest(
        self,
        *,
        prepared: PreparedRun,
        mode: str,
        metrics_paths: dict[str, Path],
        comparison_path: Path,
        prediction_paths: dict[str, Path],
        model_paths: dict[str, Path],
    ) -> dict[str, Any]:
        return {
            "run_id": prepared.run_id,
            "status": "completed",
            "mode": mode,
            "config_source": str(self.config_path) if self.config_path else None,
            "artifacts": {
                "config_snapshot": str(prepared.config_snapshot_path),
                "dataset_snapshot": str(prepared.dataset_snapshot_path),
                "plan": str(prepared.plan_path),
                "summary": str(prepared.summary_path),
                "metrics": {label: str(path) for label, path in metrics_paths.items()},
                "comparison": str(comparison_path),
                "predictions": {label: str(path) for label, path in prediction_paths.items()},
                "models": {label: str(path) for label, path in model_paths.items()},
            },
        }

    def _finalize_plan(
        self,
        prepared: PreparedRun,
        *,
        mode: str,
        comparison: dict[str, Any],
        metrics_paths: dict[str, Path],
        prediction_paths: dict[str, Path],
        model_paths: dict[str, Path],
    ) -> dict[str, Any]:
        plan = dict(prepared.plan)
        plan["status"] = "completed"
        plan["executed_at_utc"] = _utc_now().isoformat()
        plan["execution"] = {
            "mode": mode,
            "metrics": {label: str(path) for label, path in metrics_paths.items()},
            "predictions": {label: str(path) for label, path in prediction_paths.items()},
            "models": {label: str(path) for label, path in model_paths.items()},
            "comparison": comparison,
        }
        return plan

    def run(
        self,
        output_dir: str | Path | None = None,
        run_id: str | None = None,
        inspect_data: bool = True,
    ) -> ExecutedRun:
        prepared = self.prepare(output_dir=output_dir, run_id=run_id, inspect_data=inspect_data)
        store = ArtifactStore(prepared.output_dir)

        if self.config.tuning.enabled:
            from dbp_prediction.engine.tuner import run_tuning_suite

            tuning_result = run_tuning_suite(
                self.config,
                config_path=self.config_path,
                output_dir=prepared.output_dir,
            )
            mode = "tuning"
            model_results: dict[str, Any] = tuning_result.model_results
        else:
            mode = "training"
            model_results = self._run_training_suite(prepared.output_dir)

        model_evaluations = self._build_model_evaluations(model_results)
        metrics_paths, comparison_path, comparison = self._write_metrics_artifacts(
            store,
            model_evaluations,
        )
        prediction_paths = self._write_prediction_artifacts(prepared.output_dir, model_results)
        model_paths = {
            label: result.output_path
            for label, result in model_results.items()
            if result.saved
        }

        plan = self._finalize_plan(
            prepared,
            mode=mode,
            comparison=comparison,
            metrics_paths=metrics_paths,
            prediction_paths=prediction_paths,
            model_paths=model_paths,
        )
        plan_path = store.write_json("run_plan.json", plan)
        summary_path = store.write_text("README.md", self.render_summary(plan))
        prepared.plan = plan
        prepared.plan_path = plan_path
        prepared.summary_path = summary_path

        manifest = self._build_manifest(
            prepared=prepared,
            mode=mode,
            metrics_paths=metrics_paths,
            comparison_path=comparison_path,
            prediction_paths=prediction_paths,
            model_paths=model_paths,
        )
        manifest_path = store.write_json("artifact_manifest.json", manifest)

        return ExecutedRun(
            prepared=prepared,
            mode=mode,
            model_results=model_results,
            metrics_paths=metrics_paths,
            comparison_path=comparison_path,
            prediction_paths=prediction_paths,
            model_paths=model_paths,
            manifest_path=manifest_path,
            plan_path=plan_path,
            summary_path=summary_path,
            comparison=comparison,
        )

    def prepare(
        self,
        output_dir: str | Path | None = None,
        run_id: str | None = None,
        inspect_data: bool = True,
    ) -> PreparedRun:
        resolved_run_id = run_id or self.make_run_id()
        resolved_output_dir = self.resolve_output_dir(resolved_run_id, output_dir=output_dir)
        store = ArtifactStore(resolved_output_dir)
        store.ensure_dir()

        dataset_snapshot = self.inspect_dataset() if inspect_data else {
            "status": "skipped",
            "reason": "dataset inspection disabled",
        }
        plan = self.build_plan(resolved_run_id, resolved_output_dir, dataset_snapshot)

        if self.config_path and self.config_path.exists():
            copied_name = "config_source" + self.config_path.suffix.lower()
            store.copy_file(self.config_path, copied_name)

        config_snapshot_path = store.write_json("resolved_config.json", self.config)
        plan_path = store.write_json("run_plan.json", plan)
        dataset_snapshot_path = store.write_json("dataset_snapshot.json", dataset_snapshot)
        summary_path = store.write_text("README.md", self.render_summary(plan))

        return PreparedRun(
            run_id=resolved_run_id,
            output_dir=resolved_output_dir,
            config_snapshot_path=config_snapshot_path,
            plan_path=plan_path,
            dataset_snapshot_path=dataset_snapshot_path,
            summary_path=summary_path,
            plan=plan,
            dataset_snapshot=dataset_snapshot,
        )
