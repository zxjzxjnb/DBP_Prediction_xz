"""Generic Optuna tuning utilities for per-target experiments."""

from __future__ import annotations

import logging

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.model_selection import KFold

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


@dataclass
class PerTargetTuningResult:
    """Outputs from a unified per-target tuning run."""

    model_name: str
    output_path: Path
    target_payloads: dict[str, dict[str, Any]]
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
    data["Y_val_raw"] = val_df[[target_name]].to_numpy()
    data["Y_test_raw"] = test_df[[target_name]].to_numpy()
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
        "best_epoch": artifact.best_epoch,
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
    fold_best_epochs: list[int] = []
    processed_feature_cols: list[str] | None = None
    members: list[dict[str, Any]] = []
    test_predictions: list[np.ndarray] = []

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
        fold_best_epochs.append(artifact.best_epoch)
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
            members.append(_serialize_member(artifact, data))

    cv_summary = {
        "cv_rmse_mean": float(np.mean([fold["rmse"] for fold in fold_metrics])),
        "cv_rmse_std": float(np.std([fold["rmse"] for fold in fold_metrics])),
        "cv_mae_mean": float(np.mean([fold["mae"] for fold in fold_metrics])),
        "cv_mae_std": float(np.std([fold["mae"] for fold in fold_metrics])),
        "cv_r2_mean": float(np.mean([fold["r2"] for fold in fold_metrics])),
        "cv_r2_std": float(np.std([fold["r2"] for fold in fold_metrics])),
        "cv_best_epochs": fold_best_epochs,
        "fold_metrics": fold_metrics,
    }

    result = {
        "cv_summary": cv_summary,
        "processed_feature_cols": processed_feature_cols or list(request.feature_cols),
    }

    if record_members:
        ensemble_pred = np.mean(np.stack(test_predictions, axis=0), axis=0)
        test_metrics = compute_metrics(
            test_df[target_name].to_numpy(),
            ensemble_pred.ravel(),
        )
        result["members"] = members
        result["test_metrics"] = test_metrics
        result["test_output"] = {
            "y_true": test_df[target_name].to_numpy().ravel().tolist(),
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
    test_outputs: dict[str, dict[str, list[float]]] = {}

    for target_idx, target_name in enumerate(request.selected_targets):
        target_seed = seed + target_idx * 1000
        logger.info("=" * 72)
        logger.info("Tuning target: %s", target_name)
        logger.info("=" * 72)

        model_specs = dict(search_space.get("model", {}))
        training_specs = dict(search_space.get("training", {}))

        def objective(trial: optuna.Trial) -> float:
            _, tuned_model_params = _sample_search_group(trial, model_specs)
            _, tuned_training_params = _sample_search_group(trial, training_specs)

            cv_run = _run_cv_for_target(
                request,
                train_df,
                test_df,
                target_name,
                target_seed,
                model_params=_merge_params(request.base_model_params, tuned_model_params),
                training_params=_merge_params(request.training_params, tuned_training_params),
                trial=trial,
                record_members=False,
            )
            cv_summary = cv_run["cv_summary"]
            return float(cv_summary["cv_rmse_mean"]) + float(request.tuning_params["stability_penalty"]) * float(
                cv_summary["cv_rmse_std"]
            )

        study = _create_study(search_space, seed + target_idx)
        study.optimize(
            objective,
            n_trials=int(request.tuning_params["trials"]),
            show_progress_bar=request.show_progress_bar,
        )

        tuned_model_params, tuned_training_params = _best_trial_params(study, search_space)
        best_model_params = _merge_params(request.base_model_params, tuned_model_params)
        best_training_params = _merge_params(request.training_params, tuned_training_params)
        best_params = dict(tuned_model_params)
        best_params.update(tuned_training_params)

        logger.info("Best trial objective: %.4f", study.best_trial.value)
        for key, value in best_params.items():
            logger.info("  %s: %s", key, value)

        logger.info("Training CV ensemble with best params ...")
        cv_run = _run_cv_for_target(
            request,
            train_df,
            test_df,
            target_name,
            target_seed,
            model_params=best_model_params,
            training_params=best_training_params,
            record_members=True,
        )

        test_metrics = cv_run["test_metrics"]
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
            "cv_summary": cv_run["cv_summary"],
            "members": cv_run["members"],
            "test_metrics": test_metrics,
            "processed_feature_cols": cv_run["processed_feature_cols"],
        }
        test_outputs[target_name] = dict(cv_run["test_output"])

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
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_payload, request.output_path)
        saved = True
        logger.info("Saved tuned checkpoint to %s", request.output_path)
    else:
        logger.info("Skipping checkpoint save because outputs.save_models is false")

    return PerTargetTuningResult(
        model_name=request.model_name,
        output_path=request.output_path,
        target_payloads=target_payloads,
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
        filename = f"{label}_tuned_checkpoint.pt"
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
                "trials": config.tuning.trials,
                "folds": config.tuning.folds,
                "stability_penalty": config.tuning.stability_penalty,
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
        )
        model_results[label] = run_per_target_tuning_job(request)

    return TuningSuiteResult(
        output_dir=resolved_output_dir,
        model_results=model_results,
        comparison=summarize_tuning_comparison(model_results),
    )


def run_tuning_suite_from_path(config_path: str | Path) -> TuningSuiteResult:
    """Load an experiment config from disk and tune all enabled models."""
    config = load_experiment_config(config_path)
    return run_tuning_suite(config, config_path=config.source_path)
