"""Experiment configuration schema and file loading helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dbp_prediction.features import create_transform
from dbp_prediction.schemas.dataset import DatasetSchema
from dbp_prediction.settings import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_FOLDS,
    DEFAULT_LR,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_STABILITY_PENALTY,
    DEFAULT_TRIALS,
    DEFAULT_VAL_FRACTION,
    DEFAULT_WEIGHT_DECAY,
)

VALID_TASK_STRATEGIES = {"per_target", "multi_output"}


def _ensure_mapping(raw: Any, field_name: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"'{field_name}' must be a mapping")
    return dict(raw)


def _ensure_string(raw: Any, field_name: str) -> str:
    if raw is None:
        raise ValueError(f"'{field_name}' cannot be null")
    value = str(raw).strip()
    if not value:
        raise ValueError(f"'{field_name}' cannot be empty")
    return value


def _ensure_targets(raw: Any, field_name: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"'{field_name}' must be a list of strings")
    if any(value is None for value in raw):
        raise ValueError(f"'{field_name}' cannot contain null values")
    cleaned = [str(value).strip() for value in raw]
    if any(not value for value in cleaned):
        raise ValueError(f"'{field_name}' cannot contain empty values")
    return cleaned


def _rebase_relative_path(raw_path: Any, base_dir: Path) -> Any:
    if raw_path is None:
        return None

    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return str(candidate)

    return str((base_dir / candidate).resolve())


def _rebase_config_paths(raw: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    payload = dict(raw)

    dataset_raw = payload.get("dataset")
    if isinstance(dataset_raw, dict):
        dataset_copy = dict(dataset_raw)
        if "path" in dataset_copy:
            dataset_copy["path"] = _rebase_relative_path(dataset_copy["path"], base_dir)
        payload["dataset"] = dataset_copy

    outputs_key = "outputs" if isinstance(payload.get("outputs"), dict) else "output"
    outputs_raw = payload.get(outputs_key)
    if isinstance(outputs_raw, dict):
        outputs_copy = dict(outputs_raw)
        if "dir" in outputs_copy:
            outputs_copy["dir"] = _rebase_relative_path(outputs_copy["dir"], base_dir)
        payload[outputs_key] = outputs_copy

    return payload


@dataclass
class TaskConfig:
    """Task-level strategy, such as per-target or multi-output training."""

    strategy: str = "per_target"
    targets: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.strategy = self.strategy.strip().lower()
        if self.strategy not in VALID_TASK_STRATEGIES:
            raise ValueError(
                f"Unsupported task strategy '{self.strategy}'. "
                f"Supported: {sorted(VALID_TASK_STRATEGIES)}"
            )
        self.targets = _ensure_targets(self.targets, "task.targets")

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "TaskConfig":
        raw = _ensure_mapping(raw, "task")
        return cls(
            strategy=str(raw.get("strategy", "per_target")),
            targets=raw.get("targets"),
        )


@dataclass
class FeatureStepConfig:
    """One feature-engineering step in an experiment config."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = _ensure_string(self.name, "features.steps[].name")
        if not isinstance(self.params, dict):
            raise ValueError("'features.steps[].params' must be a mapping")
        create_transform(self.name, self.params)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "FeatureStepConfig":
        raw = _ensure_mapping(raw, "features.steps[]")
        return cls(
            name=raw.get("name"),
            params=dict(raw.get("params", {})),
        )


@dataclass
class FeaturesConfig:
    """Feature-engineering pipeline configuration."""

    steps: list[FeatureStepConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "FeaturesConfig":
        raw = _ensure_mapping(raw, "features")
        steps = raw.get("steps", [])
        if not isinstance(steps, list):
            raise ValueError("'features.steps' must be a list")
        return cls(steps=[FeatureStepConfig.from_dict(step) for step in steps])


@dataclass
class ModelConfig:
    """A model entry in an experiment config."""

    name: str
    alias: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def __post_init__(self) -> None:
        self.name = _ensure_string(self.name, "models[].name").lower()
        if self.alias is not None:
            self.alias = _ensure_string(self.alias, "models[].alias")
        if not isinstance(self.params, dict):
            raise ValueError("'models[].params' must be a mapping")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelConfig":
        raw = _ensure_mapping(raw, "models[]")
        return cls(
            name=raw.get("name"),
            alias=raw.get("alias"),
            params=dict(raw.get("params", {})),
            enabled=bool(raw.get("enabled", True)),
        )


@dataclass
class TrainingConfig:
    """Training defaults shared across models in an experiment."""

    seed: int = DEFAULT_SEED
    max_epochs: int = DEFAULT_MAX_EPOCHS
    patience: int = DEFAULT_PATIENCE
    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    val_fraction: float = DEFAULT_VAL_FRACTION
    optimizer: str = "Adam"
    loss: str = "MSE"
    huber_delta: float = 1.0
    max_grad_norm: float = 5.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "TrainingConfig":
        raw = _ensure_mapping(raw, "training")
        return cls(
            seed=int(raw.get("seed", DEFAULT_SEED)),
            max_epochs=int(raw.get("max_epochs", DEFAULT_MAX_EPOCHS)),
            patience=int(raw.get("patience", DEFAULT_PATIENCE)),
            batch_size=int(raw.get("batch_size", DEFAULT_BATCH_SIZE)),
            lr=float(raw.get("lr", DEFAULT_LR)),
            weight_decay=float(raw.get("weight_decay", DEFAULT_WEIGHT_DECAY)),
            val_fraction=float(raw.get("val_fraction", DEFAULT_VAL_FRACTION)),
            optimizer=str(raw.get("optimizer", "Adam")),
            loss=str(raw.get("loss", "MSE")),
            huber_delta=float(raw.get("huber_delta", 1.0)),
            max_grad_norm=float(raw.get("max_grad_norm", 5.0)),
        )


@dataclass
class TuningConfig:
    """Hyperparameter tuning options for an experiment."""

    enabled: bool = False
    trials: int = DEFAULT_TRIALS
    folds: int = DEFAULT_FOLDS
    stability_penalty: float = DEFAULT_STABILITY_PENALTY

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "TuningConfig":
        raw = _ensure_mapping(raw, "tuning")
        return cls(
            enabled=bool(raw.get("enabled", False)),
            trials=int(raw.get("trials", DEFAULT_TRIALS)),
            folds=int(raw.get("folds", DEFAULT_FOLDS)),
            stability_penalty=float(raw.get("stability_penalty", DEFAULT_STABILITY_PENALTY)),
        )


@dataclass
class OutputConfig:
    """Output controls for metrics, predictions, and model checkpoints."""

    dir: Path | None = None
    save_models: bool = True
    save_predictions: bool = False

    def __post_init__(self) -> None:
        if self.dir is not None:
            self.dir = Path(self.dir).expanduser()

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "OutputConfig":
        raw = _ensure_mapping(raw, "outputs")
        directory = raw.get("dir")
        return cls(
            dir=Path(directory) if directory else None,
            save_models=bool(raw.get("save_models", True)),
            save_predictions=bool(raw.get("save_predictions", False)),
        )

    def resolve_path(self, default_path: Path) -> Path:
        if self.dir is None:
            return default_path
        return self.dir / default_path.name


@dataclass
class ExperimentConfig:
    """Full, validated configuration for one experiment run."""

    dataset: DatasetSchema
    task: TaskConfig = field(default_factory=TaskConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    models: list[ModelConfig] = field(default_factory=list)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.models:
            raise ValueError("Experiment config must declare at least one model")

        selected_targets = self.selected_targets()
        unknown_targets = sorted(set(selected_targets) - set(self.dataset.targets))
        if unknown_targets:
            raise ValueError(
                "Task targets must be declared in dataset.targets. "
                f"Unknown: {unknown_targets}"
            )

        aliases = [model.alias for model in self.models if model.alias is not None]
        duplicate_aliases = {
            alias for alias in aliases if aliases.count(alias) > 1
        }
        if duplicate_aliases:
            raise ValueError(
                "Model aliases must be unique within an experiment config. "
                f"Duplicates: {sorted(duplicate_aliases)}"
            )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        if not isinstance(raw, dict):
            raise ValueError("Experiment config must be a mapping")

        models_raw = raw.get("models", raw.get("model"))
        if isinstance(models_raw, dict):
            models_raw = [models_raw]
        if not isinstance(models_raw, list) or not models_raw:
            raise ValueError("'models' must be a non-empty list")

        features_raw = raw.get("features")
        if features_raw is None:
            features_raw = raw.get("feature_engineering")

        outputs_raw = raw.get("outputs")
        if outputs_raw is None:
            outputs_raw = raw.get("output")

        return cls(
            dataset=DatasetSchema.from_dict(raw.get("dataset", {})),
            task=TaskConfig.from_dict(raw.get("task")),
            features=FeaturesConfig.from_dict(features_raw),
            models=[ModelConfig.from_dict(model_raw) for model_raw in models_raw],
            training=TrainingConfig.from_dict(raw.get("training")),
            tuning=TuningConfig.from_dict(raw.get("tuning")),
            outputs=OutputConfig.from_dict(outputs_raw),
        )

    def selected_targets(self) -> list[str]:
        return list(self.task.targets or self.dataset.targets)

    def get_model(self, name: str) -> ModelConfig | None:
        lookup = name.strip().lower()
        for model in self.models:
            if model.name == lookup and model.enabled:
                return model
        return None

    def require_model(self, name: str) -> ModelConfig:
        model = self.get_model(name)
        if model is None:
            raise ValueError(f"Experiment config does not define an enabled model named '{name}'")
        return model


def _read_config_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()

    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Loading YAML experiment configs requires PyYAML. "
                "Install the project dependencies or run 'pip install PyYAML'."
            ) from exc

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if payload is None:
            raise ValueError(f"Experiment config file is empty: {path}")
        if not isinstance(payload, dict):
            raise ValueError("Experiment config root must be a mapping")
        return payload

    raise ValueError(
        f"Unsupported experiment config format '{path.suffix}'. "
        "Use .yaml, .yml, or .json."
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment config from disk and validate it."""
    config_path = Path(path).expanduser().resolve()
    raw = _read_config_mapping(config_path)
    raw = _rebase_config_paths(raw, config_path.parent)
    config = ExperimentConfig.from_dict(raw)
    config.source_path = config_path
    return config
