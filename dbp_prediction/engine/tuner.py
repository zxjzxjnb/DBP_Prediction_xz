"""Generic Optuna tuning utilities for per-target experiments."""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold

from dbp_prediction.artifacts.store import to_jsonable
from dbp_prediction.engine._data_helpers import (
    dataset_payload,
    inverse_predictions,
    load_train_test_frames,
    prepare_pipeline_data,
)
from dbp_prediction.engine.evaluator import build_model_evaluation, summarize_model_comparison
from dbp_prediction.metrics import compute_metrics
from dbp_prediction.models import TrainedModelArtifact, get_model_adapter
from dbp_prediction.schemas import DatasetSchema, ExperimentConfig, load_experiment_config
from dbp_prediction.settings import RESULTS_DIR
from dbp_prediction.training import set_seed

logger = logging.getLogger(__name__)

DEFAULT_TUNING_DIR = RESULTS_DIR / "tuning"

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class PerTargetTuningRequest:
    """Normalized inputs for running one shared per-target tuning job."""

    model_name: str
    feature_cols: list[str]
    allowed_targets: list[str]
    selected_targets: list[str]
    base_model_params: dict[str, Any]
    training_params: dict[str, Any]
    tuning_params: dict[str, Any]
    output_path: Path
    save_models: bool = True
    dataset: DatasetSchema | None = None
    config_source: str | None = None
    feature_steps: list[dict[str, Any]] | None = None
    show_progress_bar: bool = True
    search_space_overrides: dict[str, Any] | None = None


@dataclass
class PerTargetTuningResult:
    """Outputs from a unified per-target tuning run."""

    model_name: str
    output_path: Path
    target_payloads: dict[str, dict[str, Any]]
    trial_histories: dict[str, list[dict[str, Any]]]
    stability_penalty_sensitivity: dict[str, dict[str, Any]]
    test_outputs: dict[str, dict[str, list[float]]]
    checkpoint_payload: dict[str, Any]
    saved: bool


@dataclass
class TuningSuiteResult:
    """Combined outputs from tuning all enabled models in an experiment."""

    output_dir: Path
    model_results: dict[str, PerTargetTuningResult]
    comparison: dict[str, Any]


# Shared helpers imported from engine._data_helpers:
# dataset_payload, inverse_predictions, load_train_test_frames, prepare_pipeline_data





def _prepare_fold_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_name: str,
    feature_steps: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    data = prepare_pipeline_data(
        feature_cols=feature_cols,
        target_name=target_name,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_steps=feature_steps,
    )
    data["train_df"] = data.pop("train_frame")
    data["val_df"] = data.pop("val_frame")
    data["test_df"] = data.pop("test_frame")
    return data


# _inverse_predictions is now imported as inverse_predictions from _data_helpers


def _serialize_member(artifact: TrainedModelArtifact, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_state": artifact.model_state,
        "in_dim": artifact.in_dim,
        "out_dim": artifact.out_dim,
        "model_params": artifact.model_params,
        "training_params": artifact.training_params,
        "seed": artifact.seed,
        "best_val": artifact.best_val,
        "best_step": artifact.best_step,
        "parameter_count": artifact.parameter_count,
        "scaler_x": data["scaler_x"],
        "scaler_y": data["scaler_y"],
        "feature_pipeline": data.get("feature_pipeline"),
        "processed_feature_cols": data.get("feature_cols_processed", []),
    }


def _merge_params(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged.update(overrides)
    return merged


def _merge_search_space(
    base: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Deep-merge YAML search-space overrides onto adapter-defined defaults.

    Semantics per parameter group (``model``, ``training``):
    * Override individual spec fields (e.g. change ``low`` / ``high`` while
      keeping the ``type``).
    * Add new parameters not present in the adapter defaults.
    * Remove a parameter by setting it to ``null`` in the YAML.

    The ``study`` group is a flat dict and is merged with simple update.
    """
    merged: dict[str, Any] = {}

    for group in ("model", "training"):
        base_group = dict(base.get(group, {}))
        override_group = overrides.get(group)
        if isinstance(override_group, dict):
            for param_name, param_override in override_group.items():
                if param_override is None:
                    base_group.pop(param_name, None)
                elif param_name in base_group:
                    base_group[param_name] = {**base_group[param_name], **param_override}
                else:
                    base_group[param_name] = dict(param_override)
        merged[group] = base_group

    base_study = dict(base.get("study", {}))
    override_study = overrides.get("study")
    if isinstance(override_study, dict):
        base_study.update(override_study)
    if base_study:
        merged["study"] = base_study

    return merged


def _condition_matches(spec: dict[str, Any], sampled: dict[str, Any]) -> bool:
    conditions = spec.get("when")
    if not conditions:
        return True

    for key, expected in dict(conditions).items():
        if key not in sampled:
            return False
        value = sampled[key]
        if isinstance(expected, dict):
            if "in" in expected and value not in expected["in"]:
                return False
            if "not_in" in expected and value in expected["not_in"]:
                return False
            continue
        if isinstance(expected, (list, tuple, set)):
            if value not in expected:
                return False
            continue
        if value != expected:
            return False
    return True


def _transform_sampled_value(value: Any, spec: dict[str, Any]) -> Any:
    value_map = spec.get("value_map")
    if not isinstance(value_map, dict):
        return value
    return value_map.get(value, value)


def _suggest_value(
    trial: optuna.Trial,
    name: str,
    spec: dict[str, Any],
) -> tuple[Any, Any]:
    kind = str(spec.get("type", "categorical")).strip().lower()
    if kind == "categorical":
        raw_value = trial.suggest_categorical(name, list(spec["choices"]))
    elif kind == "float":
        raw_value = trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
            step=spec.get("step"),
        )
    elif kind == "int":
        raw_value = trial.suggest_int(
            name,
            int(spec["low"]),
            int(spec["high"]),
            log=bool(spec.get("log", False)),
            step=int(spec.get("step", 1)),
        )
    else:
        raise ValueError(f"Unsupported search-space parameter type '{kind}' for '{name}'")
    return raw_value, _transform_sampled_value(raw_value, spec)


def _sample_search_group(
    trial: optuna.Trial,
    group_specs: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_values: dict[str, Any] = {}
    actual_values: dict[str, Any] = {}

    for name, spec in group_specs.items():
        if not _condition_matches(spec, actual_values):
            if "default" in spec:
                actual_values[name] = spec["default"]
            continue
        raw_value, actual_value = _suggest_value(trial, name, spec)
        raw_values[name] = raw_value
        actual_values[name] = actual_value

    return raw_values, actual_values


def _materialize_search_group(
    raw_values: dict[str, Any],
    group_specs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    actual_values: dict[str, Any] = {}
    for name, spec in group_specs.items():
        if not _condition_matches(spec, actual_values):
            if "default" in spec:
                actual_values[name] = spec["default"]
            continue
        if name not in raw_values:
            continue
        actual_values[name] = _transform_sampled_value(raw_values[name], spec)
    return actual_values


def _run_cv_for_target(
    request: PerTargetTuningRequest,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_name: str,
    target_seed: int,
    model_params: dict[str, Any],
    training_params: dict[str, Any],
    *,
    trial: optuna.Trial | None = None,
    record_members: bool = False,
) -> dict[str, Any]:
    adapter = get_model_adapter(request.model_name)
    folds = int(request.tuning_params["folds"])
    kf = KFold(n_splits=folds, shuffle=True, random_state=target_seed)

    fold_metrics: list[dict[str, float]] = []
    fold_best_steps: list[int] = []
    processed_feature_cols: list[str] | None = None
    members: list[dict[str, Any]] = []
    test_predictions: list[np.ndarray] = []
    test_truth: np.ndarray | None = None

    for fold_id, (train_idx, val_idx) in enumerate(kf.split(train_df), start=1):
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)
        data = _prepare_fold_data(
            fold_train_df,
            fold_val_df,
            test_df,
            request.feature_cols,
            target_name,
            request.feature_steps,
        )

        set_seed(target_seed + fold_id)
        artifact = adapter.fit(
            X_train=data["X_train"],
            Y_train=data["Y_train"],
            X_val=data["X_val"],
            Y_val=data["Y_val"],
            in_dim=len(data["feature_cols_processed"]),
            out_dim=1,
            model_params=model_params,
            training_params=training_params,
            seed=target_seed + fold_id,
        )
        fold_best_steps.append(artifact.best_step)
        if processed_feature_cols is None:
            processed_feature_cols = list(data["feature_cols_processed"])

        val_pred_scaled = adapter.predict(artifact, data["X_val"])
        val_pred_raw = inverse_predictions(val_pred_scaled, data, target_name)
        metrics = compute_metrics(data["Y_val_raw"].ravel(), val_pred_raw)
        fold_metrics.append(metrics)

        if trial is not None:
            trial.report(float(np.mean([fold["rmse"] for fold in fold_metrics])), step=fold_id)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if record_members:
            test_pred_scaled = adapter.predict(artifact, data["X_test"])
            test_pred_raw = inverse_predictions(test_pred_scaled, data, target_name)
            test_predictions.append(test_pred_raw)
            if test_truth is None:
                test_truth = data["Y_test_raw"].ravel()
            members.append(_serialize_member(artifact, data))

    cv_summary = {
        "cv_rmse_mean": float(np.mean([fold["rmse"] for fold in fold_metrics])),
        "cv_rmse_std": float(np.std([fold["rmse"] for fold in fold_metrics])),
        "cv_mae_mean": float(np.mean([fold["mae"] for fold in fold_metrics])),
        "cv_mae_std": float(np.std([fold["mae"] for fold in fold_metrics])),
        "cv_r2_mean": float(np.mean([fold["r2"] for fold in fold_metrics])),
        "cv_r2_std": float(np.std([fold["r2"] for fold in fold_metrics])),
        "cv_best_steps": fold_best_steps,
        "fold_metrics": fold_metrics,
    }

    result = {
        "cv_summary": cv_summary,
        "processed_feature_cols": processed_feature_cols or list(request.feature_cols),
    }

    if record_members:
        ensemble_pred = np.mean(np.stack(test_predictions, axis=0), axis=0)
        if test_truth is None:
            raise ValueError("Expected test targets for ensemble evaluation, but none were captured.")
        test_metrics = compute_metrics(
            test_truth,
            ensemble_pred.ravel(),
        )
        result["members"] = members
        result["test_metrics"] = test_metrics
        result["test_output"] = {
            "y_true": test_truth.tolist(),
            "y_pred": ensemble_pred.ravel().tolist(),
        }

    return result


def _best_trial_params(
    study: optuna.Study,
    search_space: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_specs = dict(search_space.get("model", {}))
    training_specs = dict(search_space.get("training", {}))
    return (
        _materialize_search_group(study.best_trial.params, model_specs),
        _materialize_search_group(study.best_trial.params, training_specs),
    )


def _create_study(search_space: dict[str, Any], seed: int) -> optuna.Study:
    study_cfg = dict(search_space.get("study", {}))
    return optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=int(study_cfg.get("n_startup_trials", 10)),
            n_warmup_steps=int(study_cfg.get("n_warmup_steps", 2)),
        ),
    )


def _analyze_study(
    study: optuna.Study,
    *,
    top_n: int = 5,
) -> dict[str, Any]:
    """Extract parameter importances and top trial distributions from a study.

    Provides actionable insight for narrowing search spaces in subsequent runs.
    """
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    analysis: dict[str, Any] = {
        "total_trials": len(study.trials),
        "completed_trials": len(completed),
        "pruned_trials": len(study.trials) - len(completed),
    }

    if len(completed) >= 2:
        try:
            importances = optuna.importance.get_param_importances(study)
            analysis["parameter_importances"] = {
                name: round(value, 4) for name, value in importances.items()
            }
        except Exception:
            analysis["parameter_importances"] = None
    else:
        analysis["parameter_importances"] = None

    sorted_trials = sorted(completed, key=lambda t: t.value)[:top_n]
    analysis["top_trials"] = [
        {
            "rank": rank,
            "objective": round(trial.value, 4),
            "params": dict(trial.params),
        }
        for rank, trial in enumerate(sorted_trials, 1)
    ]

    if sorted_trials:
        param_values: dict[str, list[Any]] = {}
        for trial in sorted_trials:
            for name, value in trial.params.items():
                param_values.setdefault(name, []).append(value)

        promising: dict[str, Any] = {}
        for name, values in param_values.items():
            if all(isinstance(v, (int, float)) for v in values):
                promising[name] = {
                    "min": round(min(values), 6),
                    "max": round(max(values), 6),
                    "median": round(float(np.median(values)), 6),
                }
            else:
                counts: dict[str, int] = {}
                for v in values:
                    key = str(v)
                    counts[key] = counts.get(key, 0) + 1
                promising[name] = {"distribution": counts}
        analysis["promising_ranges"] = promising

    return analysis


def _log_scout_analysis(analysis: dict[str, Any], target_name: str) -> None:
    """Log a human-readable summary of parameter landscape analysis."""
    logger.info("-" * 60)
    logger.info("Parameter Analysis: %s", target_name)
    logger.info(
        "  Trials: %d completed, %d pruned",
        analysis["completed_trials"],
        analysis["pruned_trials"],
    )

    importances = analysis.get("parameter_importances")
    if importances:
        logger.info("  Parameter Importances:")
        for name, value in importances.items():
            bar = "\u2588" * max(1, int(value * 30))
            logger.info("    %-25s %s  %.4f", name, bar, value)

    ranges = analysis.get("promising_ranges", {})
    if ranges:
        n_top = len(analysis.get("top_trials", []))
        logger.info("  Promising Ranges (top %d trials):", n_top)
        for name, info in ranges.items():
            if "min" in info:
                logger.info(
                    "    %-25s [%s .. %s]  median=%s",
                    name, info["min"], info["max"], info["median"],
                )
            else:
                dist = info.get("distribution", {})
                parts = ", ".join(f"{k}({v})" for k, v in dist.items())
                logger.info("    %-25s %s", name, parts)
    logger.info("-" * 60)


def _last_intermediate_value(trial: optuna.trial.FrozenTrial) -> float | None:
    if not trial.intermediate_values:
        return None
    last_step = max(trial.intermediate_values)
    return float(trial.intermediate_values[last_step])


def _build_trial_history(
    study: optuna.Study,
    *,
    stability_penalty: float,
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []

    for trial in sorted(study.trials, key=lambda item: item.number):
        attrs = dict(trial.user_attrs)
        history.append(
            {
                "trial_number": int(trial.number),
                "state": trial.state.name,
                "objective": float(trial.value) if trial.value is not None else None,
                "reported_step_count": int(max(trial.intermediate_values)) if trial.intermediate_values else None,
                "reported_cv_rmse_mean": _last_intermediate_value(trial),
                "stability_penalty": float(stability_penalty),
                "penalty_component": attrs.get("penalty_component"),
                "cv_rmse_mean": attrs.get("cv_rmse_mean"),
                "cv_rmse_std": attrs.get("cv_rmse_std"),
                "cv_mae_mean": attrs.get("cv_mae_mean"),
                "cv_mae_std": attrs.get("cv_mae_std"),
                "cv_r2_mean": attrs.get("cv_r2_mean"),
                "cv_r2_std": attrs.get("cv_r2_std"),
                "cv_best_steps": attrs.get("cv_best_steps"),
                "fold_metrics": attrs.get("fold_metrics"),
                "raw_params": dict(trial.params),
                "tuned_model_params": attrs.get("tuned_model_params", {}),
                "tuned_training_params": attrs.get("tuned_training_params", {}),
                "model_params": attrs.get("model_params", {}),
                "training_params": attrs.get("training_params", {}),
            }
        )

    return history


def _estimate_stability_penalty_sensitivity(
    trial_history: list[dict[str, Any]],
    *,
    current_penalty: float,
    top_n: int = 8,
) -> dict[str, Any]:
    completed = [
        record for record in trial_history
        if record["state"] == "COMPLETE"
        and record["cv_rmse_mean"] is not None
        and record["cv_rmse_std"] is not None
    ]
    ranked = sorted(
        completed,
        key=lambda record: (float(record["cv_rmse_mean"]), float(record["cv_rmse_std"])),
    )[:top_n]

    switch_points: list[float] = []
    for idx, left in enumerate(ranked):
        left_mean = float(left["cv_rmse_mean"])
        left_std = float(left["cv_rmse_std"])
        for right in ranked[idx + 1:]:
            right_mean = float(right["cv_rmse_mean"])
            right_std = float(right["cv_rmse_std"])

            # Only trade-off pairs can flip ranking as lambda changes.
            if left_mean < right_mean and left_std > right_std:
                switch = (right_mean - left_mean) / (left_std - right_std)
            elif right_mean < left_mean and right_std > left_std:
                switch = (left_mean - right_mean) / (right_std - left_std)
            else:
                continue

            if np.isfinite(switch) and switch >= 0:
                switch_points.append(round(float(switch), 6))

    switch_points = sorted(set(switch_points))
    sensitivity: dict[str, Any] = {
        "current_penalty": round(float(current_penalty), 6),
        "completed_trials": len(completed),
        "top_n_considered": len(ranked),
        "switch_points": switch_points,
    }

    if not switch_points:
        sensitivity["status"] = "insensitive"
        sensitivity["summary"] = (
            "No trade-off switch points found among the top completed trials. "
            "Changing lambda is unlikely to alter the top-ranked configuration much."
        )
        return sensitivity

    quartiles = {
        "min": round(float(np.min(switch_points)), 6),
        "q25": round(float(np.quantile(switch_points, 0.25)), 6),
        "median": round(float(np.quantile(switch_points, 0.5)), 6),
        "q75": round(float(np.quantile(switch_points, 0.75)), 6),
        "max": round(float(np.max(switch_points)), 6),
    }
    sensitivity["status"] = "ok"
    sensitivity["informative_range"] = quartiles

    if current_penalty < quartiles["q25"]:
        position = "below"
    elif current_penalty > quartiles["q75"]:
        position = "above"
    else:
        position = "within"
    sensitivity["current_penalty_position"] = position

    return sensitivity


def _serialize_csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(to_jsonable(value), sort_keys=True)
    return value


def _flatten_prefixed(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in values.items():
        flat[f"{prefix}__{key}"] = _serialize_csv_cell(value)
    return flat


def _trial_history_rows(
    *,
    model_label: str,
    model_name: str,
    target_name: str,
    trial_history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in trial_history:
        row = {
            "model_label": model_label,
            "model_name": model_name,
            "target_name": target_name,
            "trial_number": record["trial_number"],
            "state": record["state"],
            "objective": record["objective"],
            "reported_step_count": record["reported_step_count"],
            "reported_cv_rmse_mean": record["reported_cv_rmse_mean"],
            "cv_rmse_mean": record["cv_rmse_mean"],
            "cv_rmse_std": record["cv_rmse_std"],
            "penalty_component": record["penalty_component"],
            "stability_penalty": record["stability_penalty"],
            "cv_mae_mean": record["cv_mae_mean"],
            "cv_mae_std": record["cv_mae_std"],
            "cv_r2_mean": record["cv_r2_mean"],
            "cv_r2_std": record["cv_r2_std"],
            "cv_best_steps_json": _serialize_csv_cell(record["cv_best_steps"]),
            "fold_metrics_json": _serialize_csv_cell(record["fold_metrics"]),
            "raw_params_json": _serialize_csv_cell(record["raw_params"]),
            "tuned_model_params_json": _serialize_csv_cell(record["tuned_model_params"]),
            "tuned_training_params_json": _serialize_csv_cell(record["tuned_training_params"]),
            "model_params_json": _serialize_csv_cell(record["model_params"]),
            "training_params_json": _serialize_csv_cell(record["training_params"]),
        }
        row.update(_flatten_prefixed("raw_param", dict(record["raw_params"])))
        row.update(_flatten_prefixed("tuned_model", dict(record["tuned_model_params"])))
        row.update(_flatten_prefixed("tuned_training", dict(record["tuned_training_params"])))
        row.update(_flatten_prefixed("model", dict(record["model_params"])))
        row.update(_flatten_prefixed("training", dict(record["training_params"])))
        rows.append(row)

    return rows


def _write_json_artifact(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _write_trial_history_artifacts(
    root_dir: Path,
    *,
    file_prefix: str,
    model_label: str,
    model_name: str,
    trial_histories: dict[str, list[dict[str, Any]]],
    stability_penalty_sensitivity: dict[str, dict[str, Any]],
) -> tuple[Path, Path, Path]:
    payload = {
        "model_label": model_label,
        "model_name": model_name,
        "targets": trial_histories,
        "stability_penalty_sensitivity": stability_penalty_sensitivity,
    }
    json_path = _write_json_artifact(root_dir / f"{file_prefix}_trial_history.json", payload)
    sensitivity_path = _write_json_artifact(
        root_dir / f"{file_prefix}_stability_penalty_sensitivity.json",
        stability_penalty_sensitivity,
    )

    rows: list[dict[str, Any]] = []
    for target_name, history in trial_histories.items():
        rows.extend(
            _trial_history_rows(
                model_label=model_label,
                model_name=model_name,
                target_name=target_name,
                trial_history=history,
            )
        )
    csv_path = root_dir / f"{file_prefix}_trial_history.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(
        by=["target_name", "trial_number"],
        kind="stable",
    ).to_csv(csv_path, index=False)

    return json_path, csv_path, sensitivity_path


def _macro_test_metrics(target_payloads: dict[str, dict[str, Any]]) -> dict[str, float]:
    return build_model_evaluation(
        label="__macro__",
        model_family="__macro__",
        target_payloads=target_payloads,
        paradigm="per_target",
    )["macro_test_metrics"]


def run_per_target_tuning_job(request: PerTargetTuningRequest) -> PerTargetTuningResult:
    """Run a shared Optuna tuning flow for one model family."""
    adapter = get_model_adapter(request.model_name)
    search_space = adapter.search_space()
    if not search_space:
        raise ValueError(f"Model adapter '{request.model_name}' does not declare a tuning search space")
    if request.search_space_overrides:
        search_space = _merge_search_space(search_space, request.search_space_overrides)

    seed = int(request.training_params["seed"])
    set_seed(seed)

    train_df, test_df = load_train_test_frames(
        request.dataset, request.feature_cols, request.selected_targets
    )

    logger.info("Train: %d, Test: %d", len(train_df), len(test_df))
    logger.info(
        "Trials per target: %s, CV folds: %s",
        request.tuning_params["trials"],
        request.tuning_params["folds"],
    )
    logger.info(
        "Max epochs: %s, patience: %s",
        request.training_params["max_epochs"],
        request.training_params["patience"],
    )

    target_payloads: dict[str, dict[str, Any]] = {}
    trial_histories: dict[str, list[dict[str, Any]]] = {}
    stability_penalty_sensitivity: dict[str, dict[str, Any]] = {}
    test_outputs: dict[str, dict[str, list[float]]] = {}

    for target_idx, target_name in enumerate(request.selected_targets):
        target_seed = seed + target_idx * 1000
        logger.info("=" * 72)
        logger.info("Tuning target: %s", target_name)
        logger.info("=" * 72)

        model_specs = dict(search_space.get("model", {}))
        training_specs = dict(search_space.get("training", {}))
        stability_penalty = float(request.tuning_params["stability_penalty"])

        best_so_far: dict[str, Any] = {}

        def objective(trial: optuna.Trial) -> float:
            _, tuned_model_params = _sample_search_group(trial, model_specs)
            _, tuned_training_params = _sample_search_group(trial, training_specs)

            merged_model = _merge_params(request.base_model_params, tuned_model_params)
            merged_training = _merge_params(request.training_params, tuned_training_params)
            trial.set_user_attr("tuned_model_params", dict(tuned_model_params))
            trial.set_user_attr("tuned_training_params", dict(tuned_training_params))
            trial.set_user_attr("model_params", dict(merged_model))
            trial.set_user_attr("training_params", dict(merged_training))

            cv_run = _run_cv_for_target(
                request,
                train_df,
                test_df,
                target_name,
                target_seed,
                model_params=merged_model,
                training_params=merged_training,
                trial=trial,
                record_members=False,
            )
            cv_summary = cv_run["cv_summary"]
            penalty_component = stability_penalty * float(cv_summary["cv_rmse_std"])
            obj_value = float(cv_summary["cv_rmse_mean"]) + penalty_component
            trial.set_user_attr("cv_rmse_mean", float(cv_summary["cv_rmse_mean"]))
            trial.set_user_attr("cv_rmse_std", float(cv_summary["cv_rmse_std"]))
            trial.set_user_attr("cv_mae_mean", float(cv_summary["cv_mae_mean"]))
            trial.set_user_attr("cv_mae_std", float(cv_summary["cv_mae_std"]))
            trial.set_user_attr("cv_r2_mean", float(cv_summary["cv_r2_mean"]))
            trial.set_user_attr("cv_r2_std", float(cv_summary["cv_r2_std"]))
            trial.set_user_attr("cv_best_steps", list(cv_summary["cv_best_steps"]))
            trial.set_user_attr("fold_metrics", list(cv_summary["fold_metrics"]))
            trial.set_user_attr("penalty_component", float(penalty_component))

            if not best_so_far or obj_value < best_so_far["value"]:
                best_so_far["value"] = obj_value
                best_so_far["model_params"] = merged_model
                best_so_far["training_params"] = merged_training
                best_so_far["tuned_model_params"] = tuned_model_params
                best_so_far["tuned_training_params"] = tuned_training_params

            return obj_value

        study = _create_study(search_space, seed + target_idx)
        study.optimize(
            objective,
            n_trials=int(request.tuning_params["trials"]),
            show_progress_bar=request.show_progress_bar,
        )

        target_trial_history = _build_trial_history(study, stability_penalty=stability_penalty)
        target_penalty_sensitivity = _estimate_stability_penalty_sensitivity(
            target_trial_history,
            current_penalty=stability_penalty,
        )
        scout_analysis = _analyze_study(study)
        _log_scout_analysis(scout_analysis, target_name)

        best_model_params = best_so_far["model_params"]
        best_training_params = best_so_far["training_params"]
        best_params = dict(best_so_far["tuned_model_params"])
        best_params.update(best_so_far["tuned_training_params"])

        logger.info("Best trial objective: %.4f", study.best_trial.value)
        for key, value in best_params.items():
            logger.info("  %s: %s", key, value)

        logger.info("Re-running best configuration for final evaluation...")
        final_cv_run = _run_cv_for_target(
            request,
            train_df,
            test_df,
            target_name,
            target_seed,
            model_params=best_model_params,
            training_params=best_training_params,
            record_members=True,
        )

        test_metrics = final_cv_run["test_metrics"]
        logger.info(
            "Test: RMSE=%.3f MAE=%.3f R²=%.4f",
            test_metrics["rmse"],
            test_metrics["mae"],
            test_metrics["r2"],
        )

        target_payloads[target_name] = {
            "best_params": best_params,
            "best_model_params": best_model_params,
            "best_training_params": best_training_params,
            "best_objective": float(study.best_trial.value),
            "cv_summary": final_cv_run["cv_summary"],
            "members": final_cv_run["members"],
            "test_metrics": test_metrics,
            "processed_feature_cols": final_cv_run["processed_feature_cols"],
            "scout_analysis": scout_analysis,
            "trial_history": target_trial_history,
            "stability_penalty_sensitivity": target_penalty_sensitivity,
        }
        trial_histories[target_name] = target_trial_history
        stability_penalty_sensitivity[target_name] = target_penalty_sensitivity
        test_outputs[target_name] = dict(final_cv_run["test_output"])

    logger.info("=" * 72)
    logger.info("Final test summary")
    logger.info("=" * 72)
    for target_name in request.selected_targets:
        metrics = target_payloads[target_name]["test_metrics"]
        logger.info(
            "  %15s RMSE=%7.3f  MAE=%7.3f  R²=%.4f",
            target_name,
            metrics["rmse"],
            metrics["mae"],
            metrics["r2"],
        )

    checkpoint_payload = {
        "model_family": request.model_name,
        "paradigm": "per_target",
        "feature_cols": target_payloads[request.selected_targets[0]].get(
            "processed_feature_cols",
            request.feature_cols,
        ),
        "raw_feature_cols": request.feature_cols,
        "target_cols": request.selected_targets,
        "target_payloads": target_payloads,
        "dataset_schema": dataset_payload(request.dataset, request.feature_cols, request.allowed_targets),
        "config_source": request.config_source,
        "feature_pipeline_steps": list(request.feature_steps or []),
        "macro_test_metrics": _macro_test_metrics(target_payloads),
        "seed": seed,
        "trials": int(request.tuning_params["trials"]),
        "folds": int(request.tuning_params["folds"]),
        "max_epochs": int(request.training_params["max_epochs"]),
        "patience": int(request.training_params["patience"]),
        "stability_penalty": float(request.tuning_params["stability_penalty"]),
    }

    saved = False
    if request.save_models:
        adapter.save_checkpoint(checkpoint_payload, request.output_path)
        saved = True
        logger.info("Saved tuned checkpoint to %s", request.output_path)
    else:
        logger.info("Skipping checkpoint save because outputs.save_models is false")

    trial_history_json, trial_history_csv, sensitivity_json = _write_trial_history_artifacts(
        request.output_path.parent,
        file_prefix=request.output_path.stem,
        model_label=request.output_path.stem,
        model_name=request.model_name,
        trial_histories=trial_histories,
        stability_penalty_sensitivity=stability_penalty_sensitivity,
    )
    logger.info("Trial history saved to %s", trial_history_json)
    logger.info("Trial history CSV saved to %s", trial_history_csv)
    logger.info("Penalty sensitivity saved to %s", sensitivity_json)

    return PerTargetTuningResult(
        model_name=request.model_name,
        output_path=request.output_path,
        target_payloads=target_payloads,
        trial_histories=trial_histories,
        stability_penalty_sensitivity=stability_penalty_sensitivity,
        test_outputs=test_outputs,
        checkpoint_payload=checkpoint_payload,
        saved=saved,
    )


def summarize_tuning_comparison(model_results: dict[str, PerTargetTuningResult]) -> dict[str, Any]:
    """Build a comparison summary across tuned models."""
    return summarize_model_comparison(
        {
            label: build_model_evaluation(
                label=label,
                model_family=result.model_name,
                target_payloads=result.target_payloads,
                paradigm="per_target",
            )
            for label, result in model_results.items()
        }
    )


def run_tuning_suite(
    config: ExperimentConfig,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> TuningSuiteResult:
    """Tune all enabled models in an experiment through the shared per-target engine."""
    if config.task.strategy != "per_target":
        raise ValueError("Phase 6 tuning suite currently supports task.strategy='per_target' only")
    if not config.tuning.enabled:
        raise ValueError("Experiment tuning suite requires tuning.enabled=true")

    resolved_config_path = (
        Path(config_path or config.source_path).expanduser().resolve()
        if (config_path or config.source_path)
        else None
    )
    resolved_output_dir = (
        Path(output_dir).expanduser()
        if output_dir is not None
        else (
            config.outputs.dir
            if config.outputs.dir is not None
            else DEFAULT_TUNING_DIR / (resolved_config_path.stem if resolved_config_path else "adhoc")
        )
    )
    resolved_output_dir = resolved_output_dir.expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    enabled_models = [model for model in config.models if model.enabled]
    model_results: dict[str, PerTargetTuningResult] = {}
    name_counts = Counter(model.name for model in enabled_models)

    for model in enabled_models:
        label = model.alias or model.name
        if name_counts[model.name] > 1 and model.alias is None:
            raise ValueError(
                f"Duplicate enabled model family '{model.name}' requires aliases for output naming"
            )
        adapter = get_model_adapter(model.name)
        ext = adapter.checkpoint_extension
        filename = f"{label}_tuned_checkpoint{ext}"
        request = PerTargetTuningRequest(
            model_name=model.name,
            feature_cols=list(config.dataset.features),
            allowed_targets=list(config.dataset.targets),
            selected_targets=config.selected_targets(),
            base_model_params=dict(model.params),
            training_params={
                "seed": config.training.seed,
                "optimizer": config.training.optimizer,
                "loss": config.training.loss,
                "huber_delta": config.training.huber_delta,
                "max_grad_norm": config.training.max_grad_norm,
                "lr": config.training.lr,
                "weight_decay": config.training.weight_decay,
                "batch_size": config.training.batch_size,
                "max_epochs": config.training.max_epochs,
                "patience": config.training.patience,
                "val_fraction": config.training.val_fraction,
            },
            tuning_params={
                "trials": model.tuning.trials if model.tuning and model.tuning.trials is not None else config.tuning.trials,
                "folds": model.tuning.folds if model.tuning and model.tuning.folds is not None else config.tuning.folds,
                "stability_penalty": model.tuning.stability_penalty if model.tuning and model.tuning.stability_penalty is not None else config.tuning.stability_penalty,
                "scout": config.tuning.scout,
            },
            output_path=resolved_output_dir / filename,
            save_models=config.outputs.save_models,
            dataset=config.dataset,
            config_source=str(resolved_config_path) if resolved_config_path else None,
            feature_steps=[
                {"name": step.name, "params": dict(step.params)}
                for step in config.features.steps
            ],
            show_progress_bar=False,
            search_space_overrides=dict(model.search_space) if model.search_space else None,
        )
        model_results[label] = run_per_target_tuning_job(request)

    all_analyses: dict[str, dict[str, Any]] = {}
    for label, result in model_results.items():
        model_analyses = {}
        for target_name, payload in result.target_payloads.items():
            analysis = payload.get("scout_analysis")
            if analysis:
                model_analyses[target_name] = analysis
        if model_analyses:
            all_analyses[label] = model_analyses
    if all_analyses:
        analysis_path = resolved_output_dir / "scout_analysis.json"
        _write_json_artifact(analysis_path, all_analyses)
        logger.info("Scout analysis saved to %s", analysis_path)

    aggregated_trial_history = {
        label: {
            "model_name": result.model_name,
            "targets": result.trial_histories,
        }
        for label, result in model_results.items()
    }
    if aggregated_trial_history:
        _write_json_artifact(
            resolved_output_dir / "trial_history.json",
            aggregated_trial_history,
        )
        aggregated_rows: list[dict[str, Any]] = []
        for label, result in model_results.items():
            for target_name, history in result.trial_histories.items():
                aggregated_rows.extend(
                    _trial_history_rows(
                        model_label=label,
                        model_name=result.model_name,
                        target_name=target_name,
                        trial_history=history,
                    )
                )
        pd.DataFrame(aggregated_rows).sort_values(
            by=["model_label", "target_name", "trial_number"],
            kind="stable",
        ).to_csv(resolved_output_dir / "trial_history.csv", index=False)
        _write_json_artifact(
            resolved_output_dir / "stability_penalty_sensitivity.json",
            {
                label: result.stability_penalty_sensitivity
                for label, result in model_results.items()
            },
        )

    return TuningSuiteResult(
        output_dir=resolved_output_dir,
        model_results=model_results,
        comparison=summarize_tuning_comparison(model_results),
    )


def run_tuning_suite_from_path(config_path: str | Path) -> TuningSuiteResult:
    """Load an experiment config from disk and tune all enabled models."""
    config = load_experiment_config(config_path)
    return run_tuning_suite(config, config_path=config.source_path)
