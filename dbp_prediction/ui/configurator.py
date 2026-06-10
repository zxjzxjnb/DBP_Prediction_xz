"""Reusable helpers for the research UI configurator."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from dbp_prediction.config import FEATURE_COLS, SPLIT_COL, TARGET_COLS
from dbp_prediction.datasets.loaders import READERS, resolve_file_format
from dbp_prediction.features import TRANSFORM_REGISTRY
from dbp_prediction.models import MODEL_REGISTRY, get_model_adapter
from dbp_prediction.schemas import ExperimentConfig
from dbp_prediction.settings import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATA_PATH,
    DEFAULT_FOLDS,
    DEFAULT_LR,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_STABILITY_PENALTY,
    DEFAULT_TRIALS,
    DEFAULT_VAL_FRACTION,
    DEFAULT_WEIGHT_DECAY,
    RESULTS_DIR,
)

MODEL_ORDER = ("mlp", "kan", "random_forest", "xgboost")
MODEL_LABELS = {
    "mlp": "MLP",
    "kan": "KAN",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
PARAM_LABELS = {
    "hidden_dims": "Hidden Layers",
    "dropout": "Dropout",
    "activation": "Activation",
    "grid": "Grid",
    "k": "Spline Order",
    "base_fun": "Base Function",
    "n_estimators": "Estimators",
    "max_depth": "Max Depth",
    "min_samples_split": "Min Samples Split",
    "min_samples_leaf": "Min Samples Leaf",
    "max_features": "Max Features",
    "bootstrap": "Bootstrap",
    "max_samples": "Max Samples",
    "learning_rate": "Learning Rate",
    "min_child_weight": "Min Child Weight",
    "subsample": "Subsample",
    "colsample_bytree": "Column Sample by Tree",
    "gamma": "Gamma",
    "reg_alpha": "L1 Regularization",
    "reg_lambda": "L2 Regularization",
    "early_stopping_rounds": "Early Stopping Rounds",
    "optimizer": "Optimizer",
    "loss": "Loss",
    "huber_delta": "Huber Delta",
    "weight_decay": "Weight Decay",
    "batch_size": "Batch Size",
    "lr": "Learning Rate",
    "n_startup_trials": "Startup Trials",
    "n_warmup_steps": "Warmup Steps",
}
TUNING_PRESETS = {
    "Quick": {"trials": 15, "folds": 3, "stability_penalty": 0.05},
    "Standard": {
        "trials": DEFAULT_TRIALS,
        "folds": DEFAULT_FOLDS,
        "stability_penalty": DEFAULT_STABILITY_PENALTY,
    },
    "Deep": {"trials": 120, "folds": 5, "stability_penalty": 0.15},
}


@dataclass(frozen=True)
class FieldSpec:
    """Simple description for rendering one UI input."""

    key: str
    label: str
    kind: str
    default: Any
    help_text: str | None = None
    choices: tuple[Any, ...] = ()
    min_value: int | float | None = None
    max_value: int | float | None = None
    step: int | float | None = None


MODEL_FIELD_SPECS: dict[str, tuple[FieldSpec, ...]] = {
    "mlp": (
        FieldSpec(
            key="hidden_dims",
            label="Hidden Layers",
            kind="int_list",
            default=[32, 16],
            help_text="Comma-separated widths, for example 64,32.",
        ),
        FieldSpec(
            key="dropout",
            label="Dropout",
            kind="float",
            default=0.2,
            min_value=0.0,
            max_value=0.9,
            step=0.05,
        ),
        FieldSpec(
            key="activation",
            label="Activation",
            kind="select",
            default="ReLU",
            choices=("ReLU", "LeakyReLU", "SiLU", "Tanh"),
        ),
    ),
    "kan": (
        FieldSpec(
            key="hidden_dims",
            label="Hidden Layers",
            kind="int_list",
            default=[32, 16],
            help_text="Comma-separated widths, for example 32,16.",
        ),
        FieldSpec(
            key="grid",
            label="Grid",
            kind="int",
            default=8,
            min_value=1,
            max_value=64,
            step=1,
        ),
        FieldSpec(
            key="k",
            label="Spline Order",
            kind="int",
            default=3,
            min_value=1,
            max_value=10,
            step=1,
        ),
        FieldSpec(
            key="base_fun",
            label="Base Function",
            kind="text",
            default="silu",
            help_text="Passed to pykan as the base activation name.",
        ),
    ),
    "random_forest": (
        FieldSpec(
            key="n_estimators",
            label="Estimators",
            kind="int",
            default=300,
            min_value=10,
            max_value=1000,
            step=10,
        ),
        FieldSpec(
            key="max_depth",
            label="Max Depth",
            kind="optional_int",
            default=None,
            help_text="Leave blank for no maximum depth.",
        ),
        FieldSpec(
            key="min_samples_split",
            label="Min Samples Split",
            kind="int",
            default=2,
            min_value=2,
            max_value=50,
            step=1,
        ),
        FieldSpec(
            key="min_samples_leaf",
            label="Min Samples Leaf",
            kind="int",
            default=1,
            min_value=1,
            max_value=25,
            step=1,
        ),
        FieldSpec(
            key="max_features",
            label="Max Features",
            kind="select",
            default="sqrt",
            choices=("sqrt", "log2", 1.0, 0.7),
        ),
        FieldSpec(
            key="bootstrap",
            label="Bootstrap",
            kind="bool",
            default=True,
        ),
        FieldSpec(
            key="max_samples",
            label="Max Samples",
            kind="optional_float",
            default=None,
            help_text="Only used when bootstrap is enabled.",
        ),
    ),
    "xgboost": (
        FieldSpec(
            key="n_estimators",
            label="Estimators",
            kind="int",
            default=300,
            min_value=10,
            max_value=2000,
            step=10,
        ),
        FieldSpec(
            key="max_depth",
            label="Max Depth",
            kind="int",
            default=4,
            min_value=1,
            max_value=16,
            step=1,
        ),
        FieldSpec(
            key="learning_rate",
            label="Learning Rate",
            kind="float",
            default=0.05,
            min_value=1e-5,
            max_value=1.0,
            step=0.001,
        ),
        FieldSpec(
            key="min_child_weight",
            label="Min Child Weight",
            kind="float",
            default=1.0,
            min_value=0.0,
            max_value=50.0,
            step=0.5,
        ),
        FieldSpec(
            key="subsample",
            label="Subsample",
            kind="float",
            default=0.8,
            min_value=0.1,
            max_value=1.0,
            step=0.05,
        ),
        FieldSpec(
            key="colsample_bytree",
            label="Column Sample by Tree",
            kind="float",
            default=0.8,
            min_value=0.1,
            max_value=1.0,
            step=0.05,
        ),
        FieldSpec(
            key="gamma",
            label="Gamma",
            kind="float",
            default=0.0,
            min_value=0.0,
            max_value=10.0,
            step=0.1,
        ),
        FieldSpec(
            key="reg_alpha",
            label="L1 Regularization",
            kind="float",
            default=0.0,
            min_value=0.0,
            max_value=10.0,
            step=0.1,
        ),
        FieldSpec(
            key="reg_lambda",
            label="L2 Regularization",
            kind="float",
            default=1.0,
            min_value=0.0,
            max_value=200.0,
            step=0.1,
        ),
        FieldSpec(
            key="early_stopping_rounds",
            label="Early Stopping Rounds",
            kind="int",
            default=20,
            min_value=1,
            max_value=200,
            step=1,
        ),
        FieldSpec(
            key="objective",
            label="Objective",
            kind="text",
            default="reg:squarederror",
        ),
        FieldSpec(
            key="eval_metric",
            label="Eval Metric",
            kind="text",
            default="rmse",
        ),
        FieldSpec(
            key="tree_method",
            label="Tree Method",
            kind="text",
            default="hist",
        ),
    ),
}

TRAINING_FIELD_SPECS = (
    FieldSpec(key="seed", label="Random Seed", kind="int", default=DEFAULT_SEED),
    FieldSpec(
        key="max_epochs",
        label="Max Epochs",
        kind="int",
        default=DEFAULT_MAX_EPOCHS,
        min_value=1,
        max_value=10000,
        step=1,
    ),
    FieldSpec(
        key="patience",
        label="Patience",
        kind="int",
        default=DEFAULT_PATIENCE,
        min_value=1,
        max_value=5000,
        step=1,
    ),
    FieldSpec(
        key="batch_size",
        label="Batch Size",
        kind="int",
        default=DEFAULT_BATCH_SIZE,
        min_value=1,
        max_value=4096,
        step=1,
    ),
    FieldSpec(
        key="lr",
        label="Learning Rate",
        kind="float",
        default=DEFAULT_LR,
        min_value=1e-6,
        max_value=1.0,
        step=1e-4,
    ),
    FieldSpec(
        key="weight_decay",
        label="Weight Decay",
        kind="float",
        default=DEFAULT_WEIGHT_DECAY,
        min_value=0.0,
        max_value=1.0,
        step=1e-4,
    ),
    FieldSpec(
        key="val_fraction",
        label="Validation Fraction",
        kind="float",
        default=DEFAULT_VAL_FRACTION,
        min_value=0.01,
        max_value=0.5,
        step=0.01,
    ),
    FieldSpec(
        key="optimizer",
        label="Optimizer",
        kind="select",
        default="Adam",
        choices=("Adam", "AdamW"),
    ),
    FieldSpec(
        key="loss",
        label="Loss",
        kind="select",
        default="MSE",
        choices=("MSE", "Huber"),
    ),
    FieldSpec(
        key="huber_delta",
        label="Huber Delta",
        kind="float",
        default=1.0,
        min_value=0.01,
        max_value=20.0,
        step=0.1,
    ),
    FieldSpec(
        key="max_grad_norm",
        label="Max Gradient Norm",
        kind="float",
        default=5.0,
        min_value=0.0,
        max_value=100.0,
        step=0.5,
    ),
)

DEFAULT_OUTPUT_DIR = RESULTS_DIR / "ui_runs"
DEFAULT_FEATURE_STEPS_TEXT = "[]"


def ordered_model_names() -> list[str]:
    """Return stable model ordering for the UI."""
    known = [name for name in MODEL_ORDER if name in MODEL_REGISTRY]
    extras = sorted(set(MODEL_REGISTRY) - set(known))
    return known + extras


def get_model_label(name: str) -> str:
    """Return a user-facing English label for a model family."""
    return MODEL_LABELS.get(name, name.replace("_", " ").title())


def get_model_field_specs(model_name: str) -> tuple[FieldSpec, ...]:
    """Return base-parameter field definitions for one model family."""
    return MODEL_FIELD_SPECS.get(model_name, ())


def get_training_field_specs() -> tuple[FieldSpec, ...]:
    """Return shared training field definitions."""
    return TRAINING_FIELD_SPECS


def default_training_params() -> dict[str, Any]:
    """Return default shared training values."""
    return {field.key: deepcopy(field.default) for field in TRAINING_FIELD_SPECS}


def default_model_params(model_name: str) -> dict[str, Any]:
    """Return default base parameters for one model family."""
    return {field.key: deepcopy(field.default) for field in get_model_field_specs(model_name)}


def get_model_search_space(model_name: str) -> dict[str, Any]:
    """Return the adapter-owned default search space."""
    return deepcopy(get_model_adapter(model_name).search_space())


def default_search_form_state(model_name: str) -> dict[str, Any]:
    """Build editable UI state from an adapter default search space."""
    search_space = get_model_search_space(model_name)
    state: dict[str, Any] = {"model": {}, "training": {}, "study": {}}

    for group_name in ("model", "training"):
        for param_name, spec in search_space.get(group_name, {}).items():
            state[group_name][param_name] = {
                "enabled": True,
                "type": spec.get("type"),
                "choices": list(spec.get("choices", [])),
                "low": spec.get("low"),
                "high": spec.get("high"),
                "step": spec.get("step"),
                "log": bool(spec.get("log", False)),
                "when": deepcopy(spec.get("when")),
                "default": deepcopy(spec.get("default")),
            }

    for key, value in search_space.get("study", {}).items():
        state["study"][key] = deepcopy(value)

    return state


def default_feature_columns(columns: list[str]) -> list[str]:
    """Pick sensible default feature columns from a previewed dataset."""
    preferred = [column for column in FEATURE_COLS if column in columns]
    if preferred:
        return preferred
    excluded = set(default_target_columns(columns)) | {default_split_column(columns)}
    return [column for column in columns if column not in excluded]


def default_target_columns(columns: list[str]) -> list[str]:
    """Pick sensible default target columns from a previewed dataset."""
    preferred = [column for column in TARGET_COLS if column in columns]
    if preferred:
        return preferred
    return columns[-1:] if columns else []


def default_split_column(columns: list[str]) -> str:
    """Pick a split column for the dataset form."""
    if SPLIT_COL in columns:
        return SPLIT_COL
    return columns[-1] if columns else SPLIT_COL


def list_transform_names() -> list[str]:
    """Return known feature transform names."""
    return sorted(TRANSFORM_REGISTRY)


def parse_list_of_ints(raw_value: str) -> list[int]:
    """Parse comma-separated integer lists from a text input."""
    values = [item.strip() for item in raw_value.split(",")]
    cleaned = [item for item in values if item]
    if not cleaned:
        raise ValueError("Please provide at least one integer value.")
    return [int(item) for item in cleaned]


def parse_optional_number(raw_value: str, *, as_type: type[int] | type[float]) -> int | float | None:
    """Parse optional numeric text input."""
    value = raw_value.strip()
    if not value:
        return None
    return as_type(value)


def parse_feature_steps(text: str) -> list[dict[str, Any]]:
    """Parse YAML or JSON feature-step definitions from a text area."""
    raw = yaml.safe_load(text or "[]")
    if raw in (None, ""):
        return []
    if not isinstance(raw, list):
        raise ValueError("Feature steps must be a list of objects.")
    return [dict(item) for item in raw]


def render_config_preview(payload: dict[str, Any]) -> str:
    """Render a YAML preview for download or inspection."""
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)


def preview_dataset(
    path: str | Path,
    *,
    file_format: str | None = None,
    read_options: dict[str, Any] | None = None,
    max_rows: int = 100,
) -> pd.DataFrame:
    """Read a small preview dataframe from disk."""
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset file not found: {resolved}")

    options = dict(read_options or {})
    resolved_format = resolve_file_format(resolved, file_format=file_format)
    reader = READERS.get(resolved_format)
    if reader is None:
        raise ValueError(
            f"Unsupported file format '{resolved_format}'. Supported: {sorted(READERS)}"
        )

    if resolved_format in {"csv", "excel"} and "nrows" not in options:
        options["nrows"] = max_rows

    frame = reader(resolved, **options)
    return frame.head(max_rows).copy()


def persist_uploaded_dataset(
    filename: str,
    payload: bytes,
    *,
    target_dir: str | Path | None = None,
) -> Path:
    """Persist uploaded dataset bytes to a stable workspace path."""
    directory = Path(target_dir or (RESULTS_DIR / "ui_uploads")).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target_path = directory / Path(filename).name
    target_path.write_bytes(payload)
    return target_path


def estimate_training_load(
    *,
    selected_models: list[dict[str, Any]],
    target_count: int,
    tuning_enabled: bool,
    global_trials: int,
    global_folds: int,
) -> dict[str, Any]:
    """Estimate the experiment workload for the current configuration."""
    enabled_models = [model for model in selected_models if model.get("enabled")]
    model_count = len(enabled_models)
    target_count = max(target_count, 0)

    if tuning_enabled:
        total_fit_calls = 0
        for model in enabled_models:
            overrides = model.get("tuning") or {}
            trials = int(overrides.get("trials", global_trials))
            folds = int(overrides.get("folds", global_folds))
            total_fit_calls += trials * folds * target_count
        summary = (
            f"{model_count} model(s) x {target_count} target(s) x tuning trials/folds"
            f" = {total_fit_calls} fit calls"
        )
    else:
        total_fit_calls = model_count * target_count
        summary = f"{model_count} model(s) x {target_count} target(s) = {total_fit_calls} fit calls"

    return {
        "enabled_model_count": model_count,
        "target_count": target_count,
        "tuning_enabled": tuning_enabled,
        "total_fit_calls": total_fit_calls,
        "summary": summary,
    }


def build_search_space_override(
    model_name: str,
    form_state: dict[str, Any],
) -> dict[str, Any]:
    """Convert editable search-space state into minimal override payloads."""
    default_space = get_model_search_space(model_name)
    overrides: dict[str, Any] = {}

    for group_name in ("model", "training"):
        group_overrides: dict[str, Any] = {}
        default_group = default_space.get(group_name, {})
        form_group = form_state.get(group_name, {})
        for param_name, param_state in form_group.items():
            default_spec = default_group.get(param_name, {})
            if not param_state.get("enabled", True):
                group_overrides[param_name] = None
                continue

            spec_override: dict[str, Any] = {}
            if param_state.get("type") == "categorical":
                current_choices = list(param_state.get("choices", []))
                if current_choices != list(default_spec.get("choices", [])):
                    spec_override["choices"] = current_choices
            elif param_state.get("type") in {"int", "float"}:
                for key in ("low", "high", "step", "log"):
                    if key in default_spec and param_state.get(key) != default_spec.get(key):
                        spec_override[key] = param_state.get(key)

            if spec_override:
                group_overrides[param_name] = spec_override

        if group_overrides:
            overrides[group_name] = group_overrides

    default_study = default_space.get("study", {})
    study_state = form_state.get("study", {})
    study_override = {
        key: value
        for key, value in study_state.items()
        if value != default_study.get(key)
    }
    if study_override:
        overrides["study"] = study_override

    return overrides


def build_model_entry(
    *,
    model_name: str,
    enabled: bool,
    alias: str,
    params: dict[str, Any],
    tuning_enabled: bool,
    use_global_tuning: bool,
    model_tuning: dict[str, Any],
    use_default_search_space: bool,
    search_form_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one model entry for an experiment config payload."""
    payload: dict[str, Any] = {
        "name": model_name,
        "enabled": bool(enabled),
        "params": deepcopy(params),
    }

    if alias.strip():
        payload["alias"] = alias.strip()

    if tuning_enabled and not use_global_tuning:
        tuning_override = {
            key: value
            for key, value in {
                "trials": model_tuning.get("trials"),
                "folds": model_tuning.get("folds"),
                "stability_penalty": model_tuning.get("stability_penalty"),
            }.items()
            if value is not None
        }
        if tuning_override:
            payload["tuning"] = tuning_override

    if tuning_enabled and not use_default_search_space and search_form_state is not None:
        search_override = build_search_space_override(model_name, search_form_state)
        if search_override:
            payload["search_space"] = search_override

    return payload


def build_experiment_payload(
    *,
    dataset_path: str,
    dataset_format: str,
    feature_columns: list[str],
    target_columns: list[str],
    split_column: str,
    train_label: str,
    test_label: str,
    feature_steps: list[dict[str, Any]],
    selected_targets: list[str],
    model_entries: list[dict[str, Any]],
    training_params: dict[str, Any],
    tuning_enabled: bool,
    tuning_params: dict[str, Any],
    output_dir: str,
    save_models: bool,
    save_predictions: bool,
) -> dict[str, Any]:
    """Build and validate one experiment config payload."""
    if not feature_columns:
        raise ValueError("Select at least one feature column.")
    if not target_columns:
        raise ValueError("Select at least one target column.")
    if not selected_targets:
        raise ValueError("Select at least one prediction target.")
    if not any(model.get("enabled") for model in model_entries):
        raise ValueError("Enable at least one model before running the experiment.")

    payload: dict[str, Any] = {
        "dataset": {
            "path": str(Path(dataset_path).expanduser()),
            "format": dataset_format,
            "features": list(feature_columns),
            "targets": list(target_columns),
            "split": {
                "strategy": "predefined",
                "column": split_column,
                "train_label": train_label,
                "test_label": test_label,
            },
        },
        "task": {
            "strategy": "per_target",
            "targets": list(selected_targets),
        },
        "features": {
            "steps": list(feature_steps),
        },
        "models": list(model_entries),
        "training": deepcopy(training_params),
        "tuning": {
            "enabled": bool(tuning_enabled),
            "trials": int(tuning_params["trials"]),
            "folds": int(tuning_params["folds"]),
            "stability_penalty": float(tuning_params["stability_penalty"]),
            "scout": bool(tuning_params.get("scout", False)),
        },
        "outputs": {
            "dir": str(Path(output_dir).expanduser()),
            "save_models": bool(save_models),
            "save_predictions": bool(save_predictions),
        },
    }

    ExperimentConfig.from_dict(payload)
    return payload


def default_dataset_path() -> str:
    """Return the default dataset path for the UI."""
    return str(DEFAULT_DATA_PATH)
