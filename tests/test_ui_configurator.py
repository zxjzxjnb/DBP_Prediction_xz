"""Tests for the reusable UI configurator helpers."""

from __future__ import annotations

from pathlib import Path

from dbp_prediction.ui.configurator import (
    build_experiment_payload,
    build_search_space_override,
    default_search_form_state,
    estimate_training_load,
)


class TestUiConfigurator:
    """Tests for the non-Streamlit UI helper layer."""

    def test_build_search_space_override_tracks_changes_and_disabled_params(self) -> None:
        form_state = default_search_form_state("random_forest")
        form_state["model"]["n_estimators"]["low"] = 150
        form_state["model"]["n_estimators"]["high"] = 350
        form_state["model"]["max_samples"]["enabled"] = False
        form_state["study"]["n_startup_trials"] = 18

        override = build_search_space_override("random_forest", form_state)

        assert override == {
            "model": {
                "n_estimators": {"low": 150, "high": 350},
                "max_samples": None,
            },
            "study": {"n_startup_trials": 18},
        }

    def test_build_experiment_payload_accepts_per_model_tuning_overrides(
        self,
        sample_csv: Path,
        tmp_path: Path,
    ) -> None:
        payload = build_experiment_payload(
            dataset_path=str(sample_csv),
            dataset_format="csv",
            feature_columns=[
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
            target_columns=["T_THMs_ug_L", "DBCM_ug_L", "BDCM_ug_L"],
            split_column="split",
            train_label="train",
            test_label="test",
            feature_steps=[],
            selected_targets=["T_THMs_ug_L", "DBCM_ug_L"],
            model_entries=[
                {
                    "name": "mlp",
                    "enabled": True,
                    "params": {"hidden_dims": [32, 16], "dropout": 0.2, "activation": "ReLU"},
                    "tuning": {"trials": 25, "folds": 4, "stability_penalty": 0.2},
                    "search_space": {"model": {"dropout": {"low": 0.1, "high": 0.4}}},
                },
                {
                    "name": "kan",
                    "enabled": False,
                    "params": {"hidden_dims": [32, 16], "grid": 8, "k": 3, "base_fun": "silu"},
                },
            ],
            training_params={
                "seed": 42,
                "max_epochs": 10,
                "patience": 2,
                "batch_size": 8,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "val_fraction": 0.15,
                "optimizer": "Adam",
                "loss": "MSE",
                "huber_delta": 1.0,
                "max_grad_norm": 5.0,
            },
            tuning_enabled=True,
            tuning_params={
                "trials": 60,
                "folds": 5,
                "stability_penalty": 0.1,
                "scout": True,
            },
            output_dir=str(tmp_path / "ui_runs"),
            save_models=True,
            save_predictions=False,
        )

        assert payload["task"]["strategy"] == "per_target"
        assert payload["task"]["targets"] == ["T_THMs_ug_L", "DBCM_ug_L"]
        assert payload["tuning"]["enabled"] is True
        assert payload["tuning"]["scout"] is True
        assert payload["models"][0]["tuning"]["trials"] == 25
        assert payload["models"][0]["search_space"]["model"]["dropout"]["high"] == 0.4
        assert payload["outputs"]["dir"] == str(tmp_path / "ui_runs")

    def test_estimate_training_load_handles_tuning_and_plain_training(self) -> None:
        tuned = estimate_training_load(
            selected_models=[
                {"name": "mlp", "enabled": True},
                {"name": "kan", "enabled": True, "tuning": {"trials": 20, "folds": 4}},
            ],
            target_count=3,
            tuning_enabled=True,
            global_trials=10,
            global_folds=5,
        )
        plain = estimate_training_load(
            selected_models=[
                {"name": "mlp", "enabled": True},
                {"name": "kan", "enabled": False},
                {"name": "random_forest", "enabled": True},
            ],
            target_count=3,
            tuning_enabled=False,
            global_trials=10,
            global_folds=5,
        )

        assert tuned["total_fit_calls"] == (10 * 5 * 3) + (20 * 4 * 3)
        assert plain["total_fit_calls"] == 6
