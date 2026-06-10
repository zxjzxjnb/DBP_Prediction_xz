"""Helpers for using experiment configs with legacy model-specific CLIs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dbp_prediction.schemas import (
    DatasetSchema,
    ExperimentConfig,
    ModelConfig,
    load_experiment_config,
)

LEGACY_SUPPORTED_SPLITS = {"predefined", "column"}


@dataclass
class LegacyModelBinding:
    """Normalized experiment config values consumable by current train CLIs."""

    experiment: ExperimentConfig
    dataset: DatasetSchema
    model: ModelConfig
    selected_targets: list[str]
    output_path: Path
    save_models: bool


def bind_legacy_model_config(
    config_path: str | Path,
    model_name: str,
    default_output_path: Path,
) -> LegacyModelBinding:
    """Load an experiment config and extract the subset the legacy CLIs can honor."""
    experiment = load_experiment_config(config_path)

    if experiment.task.strategy != "per_target":
        raise ValueError(
            "Current train_mlp/train_kan CLIs only support task.strategy='per_target'. "
            f"Received '{experiment.task.strategy}'."
        )

    if experiment.dataset.split.strategy not in LEGACY_SUPPORTED_SPLITS:
        raise ValueError(
            "Current train_mlp/train_kan CLIs only support column-based dataset splits. "
            f"Received '{experiment.dataset.split.strategy}'."
        )

    if experiment.tuning.enabled:
        raise ValueError(
            "Current train_mlp/train_kan CLIs do not support 'tuning.enabled=true'. "
            "Use the dedicated tuning commands for now."
        )

    if experiment.outputs.save_predictions:
        raise ValueError(
            "Current train_mlp/train_kan CLIs do not support 'outputs.save_predictions=true' yet."
        )

    matching_models = [
        candidate
        for candidate in experiment.models
        if candidate.name == model_name.strip().lower() and candidate.enabled
    ]
    if not matching_models:
        raise ValueError(f"Experiment config does not define an enabled model named '{model_name}'")
    if len(matching_models) > 1:
        raise ValueError(
            "Current train_mlp/train_kan CLIs require exactly one enabled model of each family. "
            f"Found {len(matching_models)} entries for '{model_name}'."
        )

    model = matching_models[0]
    output_path = experiment.outputs.resolve_path(default_output_path)

    return LegacyModelBinding(
        experiment=experiment,
        dataset=experiment.dataset,
        model=model,
        selected_targets=experiment.selected_targets(),
        output_path=output_path,
        save_models=experiment.outputs.save_models,
    )
