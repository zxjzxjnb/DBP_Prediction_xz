"""Evaluation aggregation helpers for experiment runs."""

from __future__ import annotations

from typing import Any

from dbp_prediction.metrics import macro_average


def extract_target_metrics(target_payloads: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Collect per-target metrics from a run payload."""
    return {
        target_name: dict(payload["test_metrics"])
        for target_name, payload in target_payloads.items()
    }


def build_model_evaluation(
    *,
    label: str,
    model_family: str,
    target_payloads: dict[str, dict[str, Any]],
    paradigm: str,
) -> dict[str, Any]:
    """Build one model's evaluation summary."""
    target_metrics = extract_target_metrics(target_payloads)
    return {
        "label": label,
        "model_family": model_family,
        "paradigm": paradigm,
        "macro_test_metrics": macro_average(target_metrics),
        "target_metrics": target_metrics,
    }


def summarize_model_comparison(
    evaluations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare multiple evaluated models by macro RMSE."""
    summary: dict[str, Any] = {
        "models": {},
        "best_by_macro_rmse": None,
    }

    macro_rows: list[tuple[str, float]] = []
    for label, evaluation in evaluations.items():
        summary["models"][label] = dict(evaluation)
        macro_rows.append((label, float(evaluation["macro_test_metrics"]["rmse"])))

    if macro_rows:
        summary["best_by_macro_rmse"] = min(macro_rows, key=lambda item: item[1])[0]

    return summary
