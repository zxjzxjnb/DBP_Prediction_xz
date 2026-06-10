"""Tree-based regression model adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from dbp_prediction.models.base import (
    ModelAdapter,
    ModelInput,
    TrainedModelArtifact,
    register_model_adapter,
)


def _as_numpy(values: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert tensors or arrays to CPU numpy arrays."""
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _as_2d(values: np.ndarray) -> np.ndarray:
    """Ensure regression predictions keep a 2D shape."""
    array = np.asarray(values)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def _flatten_target(values: np.ndarray, out_dim: int) -> np.ndarray:
    """Flatten single-target labels for sklearn-style estimators."""
    array = _as_2d(values)
    if out_dim == 1:
        return array.ravel()
    return array


def _get_xgb_regressor_class() -> type:
    """Lazily import XGBRegressor so the package still imports without xgboost."""
    try:
        from xgboost import XGBRegressor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "XGBoost support requires the 'xgboost' package. "
            "Install project dependencies and retry."
        ) from exc
    return XGBRegressor


class TreeRegressorAdapter(ModelAdapter):
    """Shared adapter behavior for non-Torch tabular regressors."""

    step_label: str = "iteration"

    @property
    def estimator_label(self) -> str:
        return self.name

    def _build_estimator(
        self,
        *,
        in_dim: int,
        out_dim: int,
        model_params: dict[str, Any],
        training_params: dict[str, Any],
        seed: int,
    ):
        raise NotImplementedError

    def _fit_estimator(
        self,
        estimator,
        *,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        out_dim: int,
    ):
        del X_val, Y_val
        estimator.fit(X_train, _flatten_target(Y_train, out_dim))
        return estimator

    def _best_iteration(self, estimator) -> int:
        n_estimators = getattr(estimator, "n_estimators", 0)
        return int(n_estimators) if n_estimators is not None else 0

    def _parameter_count(self, estimator) -> int:
        return self._best_iteration(estimator)

    def fit(
        self,
        *,
        X_train: ModelInput,
        Y_train: ModelInput,
        X_val: ModelInput,
        Y_val: ModelInput,
        in_dim: int,
        out_dim: int,
        model_params: dict[str, Any],
        training_params: dict[str, Any],
        seed: int,
    ) -> TrainedModelArtifact:
        X_train_np = _as_numpy(X_train)
        Y_train_np = _as_numpy(Y_train)
        X_val_np = _as_numpy(X_val)
        Y_val_np = _as_numpy(Y_val)

        estimator = self._build_estimator(
            in_dim=in_dim,
            out_dim=out_dim,
            model_params=model_params,
            training_params=training_params,
            seed=seed,
        )
        estimator = self._fit_estimator(
            estimator,
            X_train=X_train_np,
            Y_train=Y_train_np,
            X_val=X_val_np,
            Y_val=Y_val_np,
            out_dim=out_dim,
        )

        val_pred = _as_2d(estimator.predict(X_val_np))
        val_true = _as_2d(Y_val_np)
        best_val = float(mean_squared_error(val_true, val_pred))

        return TrainedModelArtifact(
            family=self.name,
            in_dim=in_dim,
            out_dim=out_dim,
            model_params=dict(model_params),
            training_params=dict(training_params),
            seed=seed,
            model_state={"estimator": estimator},
            best_val=best_val,
            best_step=self._best_iteration(estimator),
            parameter_count=self._parameter_count(estimator),
        )

    def predict(
        self,
        artifact: TrainedModelArtifact,
        X: ModelInput,
    ) -> np.ndarray:
        estimator = artifact.model_state["estimator"]
        return _as_2d(estimator.predict(_as_numpy(X)))

    @property
    def checkpoint_extension(self) -> str:
        return ".joblib"

    def save_checkpoint(self, payload: dict[str, Any], path: str | Path) -> Path:
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, resolved)
        return resolved

    def save(
        self,
        artifact: TrainedModelArtifact,
        path: str | Path,
    ) -> Path:
        resolved_path = Path(path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "family": artifact.family,
                "in_dim": artifact.in_dim,
                "out_dim": artifact.out_dim,
                "model_params": artifact.model_params,
                "training_params": artifact.training_params,
                "seed": artifact.seed,
                "model_state": artifact.model_state,
                "best_val": artifact.best_val,
                "best_step": artifact.best_step,
                "parameter_count": artifact.parameter_count,
            },
            resolved_path,
        )
        return resolved_path

    def load(self, path: str | Path) -> TrainedModelArtifact:
        resolved = Path(path)
        if resolved.suffix == ".joblib":
            payload = joblib.load(resolved)
        else:
            payload = torch.load(resolved, map_location="cpu", weights_only=False)
        best_step = int(payload.get("best_step", payload.get("best_epoch", 0)))
        return TrainedModelArtifact(
            family=str(payload["family"]),
            in_dim=int(payload["in_dim"]),
            out_dim=int(payload["out_dim"]),
            model_params=dict(payload["model_params"]),
            training_params=dict(payload["training_params"]),
            seed=int(payload["seed"]),
            model_state=dict(payload["model_state"]),
            best_val=float(payload["best_val"]),
            best_step=best_step,
            parameter_count=int(payload["parameter_count"]),
        )


@register_model_adapter("random_forest")
class RandomForestAdapter(TreeRegressorAdapter):
    """ModelAdapter implementation for Random Forest regression."""

    def search_space(self) -> dict[str, Any]:
        return {
            "model": {
                "n_estimators": {"type": "int", "low": 100, "high": 500, "step": 50},
                "max_depth": {
                    "type": "categorical",
                    "choices": [None, 3, 5, 8, 12, 16],
                },
                "min_samples_split": {"type": "int", "low": 2, "high": 10},
                "min_samples_leaf": {"type": "int", "low": 1, "high": 5},
                "max_features": {
                    "type": "categorical",
                    "choices": ["sqrt", "log2", 1.0, 0.7],
                },
                "bootstrap": {
                    "type": "categorical",
                    "choices": [True, False],
                },
                "max_samples": {
                    "type": "float",
                    "low": 0.6,
                    "high": 1.0,
                    "step": 0.1,
                    "when": {"bootstrap": True},
                    "default": None,
                },
            },
            "training": {},
            "study": {
                "n_startup_trials": 10,
                "n_warmup_steps": 1,
            },
        }

    def _build_estimator(
        self,
        *,
        in_dim: int,
        out_dim: int,
        model_params: dict[str, Any],
        training_params: dict[str, Any],
        seed: int,
    ) -> RandomForestRegressor:
        del in_dim, out_dim, training_params
        bootstrap = bool(model_params.get("bootstrap", True))
        return RandomForestRegressor(
            n_estimators=int(model_params.get("n_estimators", 300)),
            max_depth=model_params.get("max_depth"),
            min_samples_split=int(model_params.get("min_samples_split", 2)),
            min_samples_leaf=int(model_params.get("min_samples_leaf", 1)),
            max_features=model_params.get("max_features", "sqrt"),
            bootstrap=bootstrap,
            max_samples=model_params.get("max_samples") if bootstrap else None,
            random_state=seed,
            n_jobs=1,
        )

    def _parameter_count(self, estimator: RandomForestRegressor) -> int:
        return int(sum(tree.tree_.node_count for tree in estimator.estimators_))


@register_model_adapter("xgboost")
class XGBoostAdapter(TreeRegressorAdapter):
    """ModelAdapter implementation for XGBoost regression."""

    def search_space(self) -> dict[str, Any]:
        return {
            "model": {
                "n_estimators": {"type": "int", "low": 100, "high": 500, "step": 50},
                "max_depth": {"type": "int", "low": 2, "high": 8},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
                "min_child_weight": {"type": "float", "low": 1.0, "high": 10.0},
                "subsample": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
                "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0, "step": 0.1},
                "gamma": {"type": "float", "low": 1e-8, "high": 5.0, "log": True},
                "reg_alpha": {"type": "float", "low": 1e-8, "high": 1.0, "log": True},
                "reg_lambda": {"type": "float", "low": 1e-3, "high": 100.0, "log": True},
                "early_stopping_rounds": {
                    "type": "categorical",
                    "choices": [10, 20, 30, 50],
                },
            },
            "training": {},
            "study": {
                "n_startup_trials": 12,
                "n_warmup_steps": 1,
            },
        }

    def _build_estimator(
        self,
        *,
        in_dim: int,
        out_dim: int,
        model_params: dict[str, Any],
        training_params: dict[str, Any],
        seed: int,
    ):
        del in_dim, out_dim, training_params
        XGBRegressor = _get_xgb_regressor_class()
        return XGBRegressor(
            objective=str(model_params.get("objective", "reg:squarederror")),
            eval_metric=str(model_params.get("eval_metric", "rmse")),
            n_estimators=int(model_params.get("n_estimators", 300)),
            max_depth=int(model_params.get("max_depth", 4)),
            learning_rate=float(model_params.get("learning_rate", 0.05)),
            min_child_weight=float(model_params.get("min_child_weight", 1.0)),
            subsample=float(model_params.get("subsample", 0.8)),
            colsample_bytree=float(model_params.get("colsample_bytree", 0.8)),
            gamma=float(model_params.get("gamma", 0.0)),
            reg_alpha=float(model_params.get("reg_alpha", 0.0)),
            reg_lambda=float(model_params.get("reg_lambda", 1.0)),
            early_stopping_rounds=int(model_params.get("early_stopping_rounds", 20)),
            random_state=seed,
            n_jobs=1,
            tree_method=str(model_params.get("tree_method", "hist")),
            verbosity=0,
        )

    def _fit_estimator(
        self,
        estimator,
        *,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        out_dim: int,
    ):
        estimator.fit(
            X_train,
            _flatten_target(Y_train, out_dim),
            eval_set=[(X_val, _flatten_target(Y_val, out_dim))],
            verbose=False,
        )
        return estimator

    def _best_iteration(self, estimator) -> int:
        best_iteration = getattr(estimator, "best_iteration", None)
        if best_iteration is not None:
            return int(best_iteration) + 1
        booster = estimator.get_booster()
        return int(booster.num_boosted_rounds())

    def _parameter_count(self, estimator) -> int:
        return int(estimator.get_booster().num_boosted_rounds())
