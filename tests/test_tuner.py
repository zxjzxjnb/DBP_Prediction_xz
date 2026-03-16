"""Tests for the shared Phase 6 tuning engine."""

from __future__ import annotations

from pathlib import Path

from dbp_prediction.engine import (
    PerTargetTuningRequest,
    run_per_target_tuning_job,
    run_tuning_suite,
)
from dbp_prediction.schemas import ExperimentConfig


class TestSharedTuner:
    """Tests for generic Optuna tuning across models."""

    def test_run_per_target_tuning_job_supports_feature_pipeline(
        self,
        sample_csv: Path,
        tmp_path: Path,
    ) -> None:
        request = PerTargetTuningRequest(
            model_name="mlp",
            feature_cols=[
                "pH",
                "COD_mg_L",
                "NH4_N_mg_L",
                "NO2_N_mg_L",
                "NO3_N_mg_L",
                "Br_mg_L",
                "TOC_mg_L",
                "UV254_A_cm",
                "temp_C",
            ],
            allowed_targets=["T_THMs_ug_L", "DBCM_ug_L", "BDCM_ug_L"],
            selected_targets=["T_THMs_ug_L"],
            base_model_params={},
            training_params={
                "seed": 42,
                "optimizer": "Adam",
                "loss": "MSE",
                "huber_delta": 1.0,
                "max_grad_norm": 5.0,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "batch_size": 8,
                "max_epochs": 1,
                "patience": 1,
                "val_fraction": 0.15,
            },
            tuning_params={
                "trials": 1,
                "folds": 2,
                "stability_penalty": 0.1,
            },
            output_path=tmp_path / "mlp_tuned.pt",
            dataset=ExperimentConfig.from_dict(
                {
                    "dataset": {
                        "path": str(sample_csv),
                        "features": [
                            "pH",
                            "COD_mg_L",
                            "NH4_N_mg_L",
                            "NO2_N_mg_L",
                            "NO3_N_mg_L",
                            "Br_mg_L",
                            "TOC_mg_L",
                            "UV254_A_cm",
                            "temp_C",
                        ],
                        "targets": ["T_THMs_ug_L", "DBCM_ug_L", "BDCM_ug_L"],
                    },
                    "models": [{"name": "mlp"}],
                }
            ).dataset,
            feature_steps=[
                {"name": "select_columns", "params": {"columns": ["pH", "COD_mg_L"]}},
                {"name": "polynomial", "params": {"columns": ["COD_mg_L"], "degree": 2}},
                {"name": "target_transform", "params": {"method": "log1p"}},
            ],
            show_progress_bar=False,
        )

        result = run_per_target_tuning_job(request)

        assert result.saved is True
        assert result.output_path.exists()
        assert result.checkpoint_payload["model_family"] == "mlp"
        assert result.checkpoint_payload["paradigm"] == "per_target"
        assert result.checkpoint_payload["feature_pipeline_steps"] == request.feature_steps
        assert "macro_test_metrics" in result.checkpoint_payload
        payload = result.target_payloads["T_THMs_ug_L"]
        assert payload["processed_feature_cols"] == ["pH", "COD_mg_L", "COD_mg_L__pow_2"]
        assert payload["members"]

    def test_run_per_target_tuning_job_supports_random_forest(
        self,
        sample_csv: Path,
        tmp_path: Path,
    ) -> None:
        request = PerTargetTuningRequest(
            model_name="random_forest",
            feature_cols=[
                "pH",
                "COD_mg_L",
                "NH4_N_mg_L",
                "NO2_N_mg_L",
                "NO3_N_mg_L",
                "Br_mg_L",
                "TOC_mg_L",
                "UV254_A_cm",
                "temp_C",
            ],
            allowed_targets=["T_THMs_ug_L", "DBCM_ug_L", "BDCM_ug_L"],
            selected_targets=["T_THMs_ug_L"],
            base_model_params={"n_estimators": 50, "max_depth": 5},
            training_params={
                "seed": 42,
                "optimizer": "Adam",
                "loss": "MSE",
                "huber_delta": 1.0,
                "max_grad_norm": 5.0,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "batch_size": 8,
                "max_epochs": 1,
                "patience": 1,
                "val_fraction": 0.15,
            },
            tuning_params={
                "trials": 1,
                "folds": 2,
                "stability_penalty": 0.1,
            },
            output_path=tmp_path / "rf_tuned.pt",
            dataset=ExperimentConfig.from_dict(
                {
                    "dataset": {
                        "path": str(sample_csv),
                        "features": [
                            "pH",
                            "COD_mg_L",
                            "NH4_N_mg_L",
                            "NO2_N_mg_L",
                            "NO3_N_mg_L",
                            "Br_mg_L",
                            "TOC_mg_L",
                            "UV254_A_cm",
                            "temp_C",
                        ],
                        "targets": ["T_THMs_ug_L", "DBCM_ug_L", "BDCM_ug_L"],
                    },
                    "models": [{"name": "random_forest"}],
                }
            ).dataset,
            feature_steps=[
                {"name": "select_columns", "params": {"columns": ["pH", "UV254_A_cm", "temp_C"]}},
            ],
            show_progress_bar=False,
        )

        result = run_per_target_tuning_job(request)

        assert result.saved is True
        assert result.output_path.exists()
        assert result.checkpoint_payload["model_family"] == "random_forest"
        assert "macro_test_metrics" in result.checkpoint_payload
        assert result.target_payloads["T_THMs_ug_L"]["members"]

    def test_run_tuning_suite_compares_multiple_enabled_models(
        self,
        sample_csv: Path,
        tmp_path: Path,
    ) -> None:
        config = ExperimentConfig.from_dict(
            {
                "dataset": {
                    "path": str(sample_csv),
                    "features": [
                        "pH",
                        "COD_mg_L",
                        "NH4_N_mg_L",
                        "NO2_N_mg_L",
                        "NO3_N_mg_L",
                        "Br_mg_L",
                        "TOC_mg_L",
                        "UV254_A_cm",
                        "temp_C",
                    ],
                    "targets": ["T_THMs_ug_L", "DBCM_ug_L", "BDCM_ug_L"],
                },
                "task": {
                    "strategy": "per_target",
                    "targets": ["T_THMs_ug_L"],
                },
                "models": [
                    {"name": "mlp", "alias": "baseline"},
                    {"name": "kan", "alias": "candidate"},
                ],
                "training": {
                    "seed": 42,
                    "max_epochs": 1,
                    "patience": 1,
                    "batch_size": 8,
                    "lr": 1e-3,
                    "weight_decay": 1e-4,
                    "optimizer": "Adam",
                    "loss": "MSE",
                },
                "tuning": {
                    "enabled": True,
                    "trials": 1,
                    "folds": 2,
                    "stability_penalty": 0.1,
                },
                "outputs": {
                    "dir": str(tmp_path / "tuning_suite"),
                    "save_models": False,
                },
            }
        )

        result = run_tuning_suite(config)

        assert result.output_dir == (tmp_path / "tuning_suite").resolve()
        assert set(result.model_results) == {"baseline", "candidate"}
        assert result.model_results["baseline"].saved is False
        assert result.model_results["candidate"].saved is False
        assert result.comparison["best_by_macro_rmse"] in {"baseline", "candidate"}
        assert "baseline" in result.comparison["models"]
        assert "candidate" in result.comparison["models"]
