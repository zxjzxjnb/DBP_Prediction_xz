"""Tests for experiment configuration schemas and file loading."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from dbp_prediction.schemas import ExperimentConfig, load_experiment_config


class TestExperimentConfig:
    """Tests for ExperimentConfig validation and aliases."""

    def test_from_dict_supports_alias_fields(self) -> None:
        payload = {
            "dataset": {
                "path": "data/custom.csv",
                "feature_cols": ["f1", "f2"],
                "target_cols": ["y1", "y2"],
                "split": {
                    "method": "column",
                    "column": "partition",
                    "train_label": "fit",
                    "test_label": "holdout",
                },
            },
            "task": {
                "strategy": "per_target",
                "targets": ["y2"],
            },
            "feature_engineering": {
                "steps": [
                    {"name": "log1p", "params": {"columns": ["f2"]}},
                ],
            },
            "models": [
                {"name": "mlp", "params": {"hidden_dims": [32, 16]}},
                {"name": "kan", "params": {"grid": 8, "k": 3}},
            ],
            "training": {
                "seed": 2024,
                "optimizer": "AdamW",
            },
            "output": {
                "dir": "results/phase1",
                "save_predictions": True,
            },
        }

        config = ExperimentConfig.from_dict(payload)

        assert config.dataset.path == Path("data/custom.csv")
        assert config.dataset.features == ["f1", "f2"]
        assert config.dataset.targets == ["y1", "y2"]
        assert config.dataset.split.column == "partition"
        assert config.selected_targets() == ["y2"]
        assert config.features.steps[0].name == "log1p"
        assert config.training.optimizer == "AdamW"
        assert config.outputs.dir == Path("results/phase1")
        assert config.outputs.save_predictions is True

    def test_require_model_raises_for_missing_model(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "dataset": {
                    "path": "data/custom.csv",
                    "features": ["f1"],
                    "targets": ["y1"],
                },
                "models": [{"name": "mlp"}],
            }
        )

        with pytest.raises(ValueError, match="does not define an enabled model"):
            config.require_model("kan")

    def test_allows_duplicate_model_names_when_aliases_differ(self) -> None:
        config = ExperimentConfig.from_dict(
            {
                "dataset": {
                    "path": "data/custom.csv",
                    "features": ["f1"],
                    "targets": ["y1"],
                },
                "models": [
                    {"name": "mlp", "alias": "baseline"},
                    {"name": "mlp", "alias": "wide"},
                ],
            }
        )

        assert [model.alias for model in config.models] == ["baseline", "wide"]

    def test_rejects_duplicate_model_aliases(self) -> None:
        with pytest.raises(ValueError, match="aliases must be unique"):
            ExperimentConfig.from_dict(
                {
                    "dataset": {
                        "path": "data/custom.csv",
                        "features": ["f1"],
                        "targets": ["y1"],
                    },
                    "models": [
                        {"name": "mlp", "alias": "candidate"},
                        {"name": "kan", "alias": "candidate"},
                    ],
                }
            )

    def test_rejects_split_strategies_that_are_not_implemented_yet(self) -> None:
        with pytest.raises(ValueError, match="not implemented yet"):
            ExperimentConfig.from_dict(
                {
                    "dataset": {
                        "path": "data/custom.csv",
                        "features": ["f1"],
                        "targets": ["y1"],
                        "split": {"strategy": "random"},
                    },
                    "models": [{"name": "mlp"}],
                }
            )

    def test_rejects_unknown_feature_step_names(self) -> None:
        with pytest.raises(ValueError, match="Unknown feature transform"):
            ExperimentConfig.from_dict(
                {
                    "dataset": {
                        "path": "data/custom.csv",
                        "features": ["f1"],
                        "targets": ["y1"],
                    },
                    "features": {
                        "steps": [{"name": "mystery_step"}],
                    },
                    "models": [{"name": "mlp"}],
                }
            )


class TestExperimentConfigLoading:
    """Tests for loading experiment configs from disk."""

    def test_load_experiment_config_supports_json(self, tmp_path: Path) -> None:
        config_path = tmp_path / "experiment.json"
        config_path.write_text(
            json.dumps(
                {
                    "dataset": {
                        "path": "data/custom.csv",
                        "features": ["f1", "f2"],
                        "targets": ["y1"],
                    },
                    "models": [{"name": "mlp", "params": {"hidden_dims": [64, 32]}}],
                }
            ),
            encoding="utf-8",
        )

        config = load_experiment_config(config_path)

        assert config.source_path == config_path
        assert config.require_model("mlp").params["hidden_dims"] == [64, 32]

    def test_load_experiment_config_rebases_relative_paths_to_config_dir(
        self,
        tmp_path: Path,
    ) -> None:
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        config_path = config_dir / "experiment.json"
        config_path.write_text(
            json.dumps(
                {
                    "dataset": {
                        "path": "../data/custom.csv",
                        "features": ["f1"],
                        "targets": ["y1"],
                    },
                    "models": [{"name": "mlp"}],
                    "outputs": {"dir": "../results/demo"},
                }
            ),
            encoding="utf-8",
        )

        config = load_experiment_config(config_path)

        assert config.dataset.path == (tmp_path / "data" / "custom.csv").resolve()
        assert config.outputs.dir == (tmp_path / "results" / "demo").resolve()

    def test_yaml_loading_reports_missing_dependency_cleanly(self, tmp_path: Path) -> None:
        config_path = tmp_path / "experiment.yaml"
        config_path.write_text(
            "dataset:\n  path: data/custom.csv\n  features: [f1]\n  targets: [y1]\nmodels:\n  - name: mlp\n",
            encoding="utf-8",
        )

        if importlib.util.find_spec("yaml") is not None:
            config = load_experiment_config(config_path)
            assert config.dataset.features == ["f1"]
        else:
            with pytest.raises(ModuleNotFoundError, match="PyYAML"):
                load_experiment_config(config_path)
