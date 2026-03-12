"""Helpers for exporting processed experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def build_predictions_frame(
    predictions: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Build a wide predictions table from per-target outputs."""
    if not predictions:
        raise ValueError("Predictions payload cannot be empty")

    lengths = {
        target_name: (
            len(payload.get("y_true", [])),
            len(payload.get("y_pred", [])),
        )
        for target_name, payload in predictions.items()
    }
    invalid = {
        target_name: pair
        for target_name, pair in lengths.items()
        if pair[0] == 0 or pair[0] != pair[1]
    }
    if invalid:
        raise ValueError(f"Predictions payload contains invalid target lengths: {invalid}")

    row_count = next(iter(lengths.values()))[0]
    if any(pair[0] != row_count for pair in lengths.values()):
        raise ValueError("All target predictions must have the same row count")

    frame = pd.DataFrame({"row_id": list(range(row_count))})
    for target_name, payload in predictions.items():
        frame[f"{target_name}__actual"] = list(payload["y_true"])
        frame[f"{target_name}__prediction"] = list(payload["y_pred"])
    return frame


def export_predictions(
    path: str | Path,
    predictions: dict[str, dict[str, Any]],
) -> Path:
    """Write per-target predictions to CSV."""
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    build_predictions_frame(predictions).to_csv(resolved_path, index=False)
    return resolved_path
