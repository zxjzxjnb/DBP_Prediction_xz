"""Legacy per-target training flow using the model adapter registry."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dbp_prediction.datasets import get_train_val_split, load_dataset, prepare_data
from dbp_prediction.engine._data_helpers import (
    dataset_payload,
    inverse_predictions,
    prepare_pipeline_data,
)
from dbp_prediction.metrics import compute_metrics
from dbp_prediction.models import TrainedModelArtifact, get_model_adapter
from dbp_prediction.schemas import DatasetSchema
from dbp_prediction.training import set_seed

logger = logging.getLogger(__name__)


@dataclass
class LegacyTrainingRequest:
    """Normalized inputs for running one legacy per-target training job."""

    model_name: str
    feature_cols: list[str]
    allowed_targets: list[str]
    selected_targets: list[str]
    model_params: dict[str, Any]
    training_params: dict[str, Any]
    output_path: Path
    save_models: bool = True
    dataset: DatasetSchema | None = None
    config_source: str | None = None
    feature_steps: list[dict[str, Any]] | None = None


@dataclass
class LegacyTrainingResult:
    """Outputs from a unified per-target training run."""

    model_name: str
    output_path: Path
    target_payloads: dict[str, dict[str, Any]]
    test_outputs: dict[str, dict[str, list[float]]]
    checkpoint_payload: dict[str, Any]
    saved: bool


def _target_hyperparams(
    model_params: dict[str, Any],
    training_params: dict[str, Any],
) -> dict[str, Any]:
    keys = [
        "optimizer",
        "loss",
        "huber_delta",
        "max_grad_norm",
        "lr",
        "weight_decay",
        "batch_size",
        "max_epochs",
        "patience",
        "val_fraction",
    ]
    hyperparams = dict(model_params)
    for key in keys:
        if key in training_params:
            hyperparams[key] = training_params[key]
    return hyperparams


def _prepare_target_training_data(
    request: LegacyTrainingRequest,
    target_name: str,
) -> dict[str, Any]:
    """Prepare training data from either a feature pipeline or the legacy path."""
    if request.feature_steps:
        dataset = request.dataset
        if dataset is None:
            raise ValueError("Feature pipeline training requires an explicit dataset schema")

        df = load_dataset(
            path=dataset.path,
            feature_cols=request.feature_cols,
            target_cols=[target_name],
            split_col=dataset.split.column,
            file_format=dataset.format,
            read_options=dataset.reader_options,
        )
        from dbp_prediction.datasets import get_train_test_split

        train_df, test_df = get_train_test_split(
            df,
            split_col=dataset.split.column,
            train_label=dataset.split.train_label,
            test_label=dataset.split.test_label,
        )
        train_sub_df, val_df = get_train_val_split(
            train_df,
            val_fraction=float(request.training_params["val_fraction"]),
            seed=int(request.training_params["seed"]),
        )

        pipeline_data = prepare_pipeline_data(
            feature_cols=request.feature_cols,
            target_name=target_name,
            train_df=train_sub_df,
            val_df=val_df,
            test_df=test_df,
            feature_steps=request.feature_steps,
        )
        pipeline_data["train_df"] = train_df
        pipeline_data["train_sub_df"] = pipeline_data.pop("train_frame")
        pipeline_data["val_df"] = pipeline_data.pop("val_frame")
        pipeline_data["test_df"] = pipeline_data.pop("test_frame")
        return pipeline_data

    dataset = request.dataset
    return prepare_data(
        data_path=dataset.path if dataset else None,
        val_fraction=float(request.training_params["val_fraction"]),
        seed=int(request.training_params["seed"]),
        feature_cols=request.feature_cols,
        target_cols=[target_name],
        split_col=dataset.split.column if dataset else "split",
        train_label=dataset.split.train_label if dataset else "train",
        test_label=dataset.split.test_label if dataset else "test",
        file_format=dataset.format if dataset else None,
        read_options=dataset.reader_options if dataset else None,
    ) | {
        "feature_pipeline": None,
        "feature_cols_processed": list(request.feature_cols),
    }


def _serialize_target_payload(
    artifact: TrainedModelArtifact,
    data: dict[str, Any],
    test_metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "model_state": artifact.model_state,
        "scaler_x": data["scaler_x"],
        "scaler_y": data["scaler_y"],
        "best_val": artifact.best_val,
        "best_step": artifact.best_step,
        "test_metrics": test_metrics,
        "hyperparams": _target_hyperparams(artifact.model_params, artifact.training_params),
        "feature_pipeline": data.get("feature_pipeline"),
        "processed_feature_cols": data.get("feature_cols_processed", []),
    }


def run_legacy_training_job(request: LegacyTrainingRequest) -> LegacyTrainingResult:
    """Run a legacy per-target training flow through the shared model registry."""
    adapter = get_model_adapter(request.model_name)
    seed = int(request.training_params["seed"])
    set_seed(seed)

    target_payloads: dict[str, dict[str, Any]] = {}
    test_outputs: dict[str, dict[str, list[float]]] = {}
    logger.info("Targets to train independently: %s", request.selected_targets)
    logger.info("Raw features: %d", len(request.feature_cols))

    for target_name in request.selected_targets:
        logger.info("\n" + "=" * 72)
        logger.info("Training target: %s", target_name)
        logger.info("=" * 72)

        data = _prepare_target_training_data(request, target_name)
        logger.info(
            "Train samples: %d, Validation: %d, Test: %d",
            len(data["train_sub_df"]),
            len(data["val_df"]),
            len(data["test_df"]),
        )
        logger.info("Processed features: %d", len(data["feature_cols_processed"]))

        artifact = adapter.fit(
            X_train=data["X_train"],
            Y_train=data["Y_train"],
            X_val=data["X_val"],
            Y_val=data["Y_val"],
            in_dim=len(data["feature_cols_processed"]),
            out_dim=1,
            model_params=request.model_params,
            training_params=request.training_params,
            seed=seed,
        )
        logger.info("Model parameters: %s", f"{artifact.parameter_count:,}")

        pred_scaled = adapter.predict(artifact, data["X_test"])
        pred_raw = inverse_predictions(pred_scaled, data, target_name)
        test_metrics = compute_metrics(data["Y_test_raw"].ravel(), pred_raw)
        logger.info(
            "Test: RMSE=%.3f MAE=%.3f R²=%.4f",
            test_metrics["rmse"],
            test_metrics["mae"],
            test_metrics["r2"],
        )

        target_payloads[target_name] = _serialize_target_payload(artifact, data, test_metrics)
        test_outputs[target_name] = {
            "y_true": data["Y_test_raw"].ravel().tolist(),
            "y_pred": pred_raw.ravel().tolist(),
        }

    logger.info("\n" + "=" * 72)
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
        "paradigm": "per_target_baseline",
        "feature_cols": target_payloads[request.selected_targets[0]].get(
            "processed_feature_cols", request.feature_cols
        ),
        "raw_feature_cols": request.feature_cols,
        "target_cols": request.selected_targets,
        "target_payloads": target_payloads,
        "dataset_schema": dataset_payload(
            request.dataset, request.feature_cols, request.allowed_targets
        ),
        "config_source": request.config_source,
        "seed": seed,
        "feature_pipeline_steps": list(request.feature_steps or []),
    }
    saved = False
    if request.save_models:
        adapter.save_checkpoint(checkpoint_payload, request.output_path)
        saved = True
        logger.info("Model saved to %s", request.output_path)
    else:
        logger.info("Skipping checkpoint save because outputs.save_models is false")

    return LegacyTrainingResult(
        model_name=request.model_name,
        output_path=request.output_path,
        target_payloads=target_payloads,
        test_outputs=test_outputs,
        checkpoint_payload=checkpoint_payload,
        saved=saved,
    )
