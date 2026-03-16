"""Dataset loading and input validation helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from dbp_prediction.config import DEFAULT_DATA_PATH, FEATURE_COLS, SPLIT_COL, TARGET_COLS

logger = logging.getLogger(__name__)

READERS = {
    "csv": pd.read_csv,
    "excel": pd.read_excel,
    "parquet": pd.read_parquet,
}
OPTIONAL_FORMAT_DEPENDENCIES = {
    "excel": "openpyxl",
    "parquet": "pyarrow or fastparquet",
}


def resolve_file_format(path: Path, file_format: str | None = None) -> str:
    """Normalize an explicit or inferred tabular data format name."""
    resolved_format = (file_format or path.suffix.lstrip(".") or "csv").lower()
    if resolved_format in {"xlsx", "xls"}:
        return "excel"
    return resolved_format


def load_dataset(
    path: Path | None = None,
    feature_cols: list[str] | None = None,
    target_cols: list[str] | None = None,
    split_col: str = SPLIT_COL,
    file_format: str | None = None,
    read_options: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load a tabular dataset and validate expected columns."""
    path = path or DEFAULT_DATA_PATH
    feature_cols = feature_cols or FEATURE_COLS
    target_cols = target_cols or TARGET_COLS
    read_options = read_options or {}

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    resolved_format = resolve_file_format(path, file_format=file_format)
    reader = READERS.get(resolved_format)
    if reader is None:
        raise ValueError(
            f"Unsupported file format '{resolved_format}'. "
            f"Supported: {sorted(READERS)}"
        )

    try:
        df = reader(path, **read_options)
    except ImportError as exc:
        dependency = OPTIONAL_FORMAT_DEPENDENCIES.get(resolved_format)
        if dependency is None:
            raise
        raise ModuleNotFoundError(
            f"Reading '{resolved_format}' datasets requires {dependency}. "
            "Install the project dependencies for that format and retry."
        ) from exc
    required = set(feature_cols) | set(target_cols) | {split_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {sorted(missing)}")

    logger.info("Loaded %d rows from %s", len(df), path)
    return df
