"""Tests for the Phase 2 experiment runner skeleton."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dbp_prediction.engine import ExperimentRunner
from dbp_prediction.engine.runner import LegacyTrainingRequest, run_legacy_training_job
from dbp_prediction.schemas import DatasetSchema


class TestExperimentRunner:
    """Tests for run preparation and artifact writing."""

    def test_prepare_writes_run_artifacts(self, sample_csv: Path, tmp_path: Path) -> None:
        config_path = tmp_path / "experiment.json"
        config_path.write_text(
            json.dumps(
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
                    "models": [
                        {"name": "mlp", "alias": "baseline"},
                        {"name": "kan", "alias": "candidate"},
                    ],
                }
            ),
            encoding="utf-8",
        )

        runner = ExperimentRunner.from_path(config_path)
        prepared = runner.prepare(
            output_dir=tmp_path / "prepared_run",
            run_id="phase2-test",
        )

        assert prepared.output_dir == tmp_path / "prepared_run"
        assert prepared.config_snapshot_path.exists()
        assert prepared.plan_path.exists()
        assert prepared.dataset_snapshot_path.exists()
        assert prepared.summary_path.exists()
        assert (prepared.output_dir / "config_source.json").exists()

        plan = json.loads(prepared.plan_path.read_text(encoding="utf-8"))
        snapshot = json.loads(prepared.dataset_snapshot_path.read_text(encoding="utf-8"))

        assert plan["run_id"] == "phase2-test"
        assert plan["task"]["strategy"] == "per_target"
        assert len(plan["legacy_command_hints"]) == 2
        assert snapshot["row_count"] == 25
        assert snapshot["split"]["train_rows"] == 20
        assert snapshot["split"]["test_rows"] == 5

    def test_prepare_can_skip_dataset_inspection(self, sample_csv: Path, tmp_path: Path) -> None:
        config_path = tmp_path / "experiment.json"
        config_path.write_text(
            json.dumps(
                {
                    "dataset": {
                        "path": str(sample_csv),
                        "features": ["pH"],
                        "targets": ["T_THMs_ug_L"],
                    },
                    "models": [{"name": "mlp"}],
                }
            ),
            encoding="utf-8",
        )

        runner = ExperimentRunner.from_path(config_path)
        prepared = runner.prepare(
            output_dir=tmp_path / "prepared_run",
            run_id="phase2-skip",
            inspect_data=False,
        )

        assert prepared.dataset_snapshot["status"] == "skipped"

    def test_legacy_hints_flag_unsupported_configs(self, sample_csv: Path, tmp_path: Path) -> None:
        config_path = tmp_path / "experiment.json"
        config_path.write_text(
            json.dumps(
                {
                    "dataset": {
                        "path": str(sample_csv),
                        "features": ["pH"],
                        "targets": ["T_THMs_ug_L"],
                    },
                    "models": [{"name": "mlp"}],
                    "outputs": {"save_predictions": True},
                }
            ),
            encoding="utf-8",
        )

        runner = ExperimentRunner.from_path(config_path)
        hints = runner.build_legacy_command_hints()

        assert hints[0]["supported"] is False
        assert "not supported" in hints[0]["reason"]

    def test_prepare_uses_run_scoped_output_dir_when_config_declares_outputs_dir(
        self,
        sample_csv: Path,
        tmp_path: Path,
    ) -> None:
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        config_path = config_dir / "experiment.json"
        config_path.write_text(
            json.dumps(
                {
                    "dataset": {
                        "path": str(sample_csv),
                        "features": ["pH"],
                        "targets": ["T_THMs_ug_L"],
                    },
                    "models": [{"name": "mlp"}],
                    "outputs": {"dir": "../prepared_runs"},
                }
            ),
            encoding="utf-8",
        )

        runner = ExperimentRunner.from_path(config_path)
        prepared_a = runner.prepare(run_id="run-a", inspect_data=False)
        prepared_b = runner.prepare(run_id="run-b", inspect_data=False)

        assert prepared_a.output_dir == (tmp_path / "prepared_runs" / "run-a").resolve()
        assert prepared_b.output_dir == (tmp_path / "prepared_runs" / "run-b").resolve()
        assert prepared_a.output_dir != prepared_b.output_dir
        assert prepared_a.plan_path.exists()
        assert prepared_b.plan_path.exists()

    def test_legacy_hints_reject_duplicate_enabled_model_families(
        self,
        sample_csv: Path,
        tmp_path: Path,
    ) -> None:
        config_path = tmp_path / "experiment.json"
        config_path.write_text(
            json.dumps(
                {
                    "dataset": {
                        "path": str(sample_csv),
                        "features": ["pH"],
                        "targets": ["T_THMs_ug_L"],
                    },
                    "models": [
                        {"name": "mlp", "alias": "baseline"},
                        {"name": "mlp", "alias": "wide"},
                    ],
                }
            ),
            encoding="utf-8",
        )

        runner = ExperimentRunner.from_path(config_path)
        hints = runner.build_legacy_command_hints()

        assert all(hint["supported"] is False for hint in hints)
        assert all("ambiguous" in hint["reason"] for hint in hints)

    def test_run_legacy_training_job_uses_registered_adapter(
        self,
        sample_csv: Path,
        tmp_path: Path,
    ) -> None:
        dataset = DatasetSchema.from_dict(
            {
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
            }
        )
        request = LegacyTrainingRequest(
            model_name="mlp",
            feature_cols=list(dataset.features),
            allowed_targets=list(dataset.targets),
            selected_targets=["T_THMs_ug_L"],
            model_params={"hidden_dims": [8], "dropout": 0.0, "activation": "ReLU"},
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
                "val_fraction": 0.2,
            },
            output_path=tmp_path / "mlp_phase4.pt",
            dataset=dataset,
        )

        result = run_legacy_training_job(request)

        assert result.saved is True
        assert result.output_path.exists()
        assert result.checkpoint_payload["model_family"] == "mlp"
        assert "T_THMs_ug_L" in result.target_payloads

    def test_run_legacy_training_job_supports_feature_pipeline(
        self,
        sample_csv: Path,
        tmp_path: Path,
    ) -> None:
        dataset = DatasetSchema.from_dict(
            {
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
            }
        )
        request = LegacyTrainingRequest(
            model_name="mlp",
            feature_cols=list(dataset.features),
            allowed_targets=list(dataset.targets),
            selected_targets=["T_THMs_ug_L"],
            model_params={"hidden_dims": [8], "dropout": 0.0, "activation": "ReLU"},
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
                "val_fraction": 0.2,
            },
            output_path=tmp_path / "mlp_phase5.pt",
            dataset=dataset,
            feature_steps=[
                {"name": "select_columns", "params": {"columns": ["pH", "COD_mg_L"]}},
                {"name": "polynomial", "params": {"columns": ["COD_mg_L"], "degree": 2}},
                {"name": "target_transform", "params": {"method": "log1p"}},
            ],
        )

        result = run_legacy_training_job(request)

        assert result.saved is True
        assert result.checkpoint_payload["raw_feature_cols"] == list(dataset.features)
        assert result.checkpoint_payload["feature_pipeline_steps"] == request.feature_steps
        assert result.checkpoint_payload["feature_cols"] == ["pH", "COD_mg_L", "COD_mg_L__pow_2"]
        payload = result.target_payloads["T_THMs_ug_L"]
        assert payload["feature_pipeline"] is not None
        assert payload["processed_feature_cols"] == ["pH", "COD_mg_L", "COD_mg_L__pow_2"]

    def test_run_legacy_training_job_supports_drop_missing_pipeline_step(
        self,
        sample_dataframe: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        df = sample_dataframe.copy()
        df.loc[0, "pH"] = np.nan
        df.loc[df.index[-1], "T_THMs_ug_L"] = np.nan
        csv_path = tmp_path / "missing_rows.csv"
        df.to_csv(csv_path, index=False)

        dataset = DatasetSchema.from_dict(
            {
                "path": str(csv_path),
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
            }
        )
        request = LegacyTrainingRequest(
            model_name="mlp",
            feature_cols=list(dataset.features),
            allowed_targets=list(dataset.targets),
            selected_targets=["T_THMs_ug_L"],
            model_params={"hidden_dims": [8], "dropout": 0.0, "activation": "ReLU"},
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
                "val_fraction": 0.2,
            },
            output_path=tmp_path / "mlp_drop_missing.pt",
            dataset=dataset,
            feature_steps=[{"name": "drop_missing"}],
        )

        result = run_legacy_training_job(request)

        assert result.saved is True
        output = result.test_outputs["T_THMs_ug_L"]
        assert len(output["y_true"]) == 4
        assert len(output["y_pred"]) == 4

    def test_run_legacy_training_job_supports_random_forest(
        self,
        sample_csv: Path,
        tmp_path: Path,
    ) -> None:
        dataset = DatasetSchema.from_dict(
            {
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
            }
        )
        request = LegacyTrainingRequest(
            model_name="random_forest",
            feature_cols=list(dataset.features),
            allowed_targets=list(dataset.targets),
            selected_targets=["T_THMs_ug_L"],
            model_params={
                "n_estimators": 30,
                "max_depth": 5,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "bootstrap": True,
            },
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
                "val_fraction": 0.2,
            },
            output_path=tmp_path / "rf_phase.pt",
            dataset=dataset,
        )

        result = run_legacy_training_job(request)

        assert result.saved is True
        assert result.output_path.exists()
        assert result.checkpoint_payload["model_family"] == "random_forest"
        assert "T_THMs_ug_L" in result.target_payloads

    def test_run_executes_training_and_writes_phase7_artifacts(
        self,
        sample_csv: Path,
        tmp_path: Path,
    ) -> None:
        config_path = tmp_path / "experiment.json"
        config_path.write_text(
            json.dumps(
                {
                    "dataset": {
                        "path": str(sample_csv),
                        "features": ["pH", "COD_mg_L", "NH4_N_mg_L"],
                        "targets": ["T_THMs_ug_L"],
                    },
                    "models": [{"name": "mlp"}],
                    "training": {
                        "max_epochs": 1,
                        "patience": 1,
                        "batch_size": 8,
                    },
                    "outputs": {
                        "save_predictions": True,
                    },
                }
            ),
            encoding="utf-8",
        )

        runner = ExperimentRunner.from_path(config_path)
        executed = runner.run(
            output_dir=tmp_path / "executed_run",
            run_id="phase7-train",
            inspect_data=False,
        )

        comparison = json.loads(executed.comparison_path.read_text(encoding="utf-8"))
        manifest = json.loads(executed.manifest_path.read_text(encoding="utf-8"))
        plan = json.loads(executed.plan_path.read_text(encoding="utf-8"))

        assert executed.mode == "training"
        assert executed.metrics_paths["mlp"].exists()
        assert executed.prediction_paths["mlp"].exists()
        assert executed.model_paths["mlp"].exists()
        assert comparison["best_by_macro_rmse"] == "mlp"
        assert plan["status"] == "completed"
        assert plan["execution"]["mode"] == "training"
        assert "mlp" in manifest["artifacts"]["predictions"]

    def test_run_executes_tuning_and_writes_phase7_artifacts(
        self,
        sample_csv: Path,
        tmp_path: Path,
    ) -> None:
        config_path = tmp_path / "experiment.json"
        config_path.write_text(
            json.dumps(
                {
                    "dataset": {
                        "path": str(sample_csv),
                        "features": ["pH", "COD_mg_L", "NH4_N_mg_L"],
                        "targets": ["T_THMs_ug_L"],
                    },
                    "task": {
                        "strategy": "per_target",
                        "targets": ["T_THMs_ug_L"],
                    },
                    "models": [{"name": "mlp"}],
                    "training": {
                        "max_epochs": 1,
                        "patience": 1,
                        "batch_size": 8,
                    },
                    "tuning": {
                        "enabled": True,
                        "trials": 1,
                        "folds": 2,
                    },
                    "outputs": {
                        "save_predictions": True,
                    },
                }
            ),
            encoding="utf-8",
        )

        runner = ExperimentRunner.from_path(config_path)
        executed = runner.run(
            output_dir=tmp_path / "tuned_run",
            run_id="phase7-tune",
            inspect_data=False,
        )

        comparison = json.loads(executed.comparison_path.read_text(encoding="utf-8"))
        plan = json.loads(executed.plan_path.read_text(encoding="utf-8"))

        assert executed.mode == "tuning"
        assert executed.metrics_paths["mlp"].exists()
        assert executed.prediction_paths["mlp"].exists()
        assert executed.model_paths["mlp"].exists()
        assert comparison["best_by_macro_rmse"] == "mlp"
        assert plan["execution"]["mode"] == "tuning"
        assert (executed.prepared.output_dir / "trial_history.json").exists()
        assert (executed.prepared.output_dir / "trial_history.csv").exists()
        assert (executed.prepared.output_dir / "stability_penalty_sensitivity.json").exists()

    def test_run_rejects_multi_output_until_runner_support_lands(
        self,
        sample_csv: Path,
        tmp_path: Path,
    ) -> None:
        config_path = tmp_path / "experiment.json"
        config_path.write_text(
            json.dumps(
                {
                    "dataset": {
                        "path": str(sample_csv),
                        "features": ["pH"],
                        "targets": ["T_THMs_ug_L"],
                    },
                    "task": {"strategy": "multi_output"},
                    "models": [{"name": "mlp"}],
                }
            ),
            encoding="utf-8",
        )

        runner = ExperimentRunner.from_path(config_path)

        with pytest.raises(ValueError, match="task.strategy='per_target'"):
            runner.run(output_dir=tmp_path / "unsupported", inspect_data=False)
