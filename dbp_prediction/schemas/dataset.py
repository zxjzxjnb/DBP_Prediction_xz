"""Dataset and split configuration schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dbp_prediction.config import SPLIT_COL, TEST_LABEL, TRAIN_LABEL
from dbp_prediction.settings import DEFAULT_DATA_PATH

VALID_DATA_FORMATS = {"csv", "excel", "parquet"}
VALID_SPLIT_STRATEGIES = {"predefined", "column", "random", "stratified", "kfold", "time"}
IMPLEMENTED_SPLIT_STRATEGIES = {"predefined", "column"}


def _normalise_str_list(values: Any, field_name: str) -> list[str]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"'{field_name}' must be a non-empty list of strings")

    if any(value is None for value in values):
        raise ValueError(f"'{field_name}' cannot contain null values")

    cleaned = [str(value).strip() for value in values]
    if any(not value for value in cleaned):
        raise ValueError(f"'{field_name}' cannot contain empty values")
    return cleaned


def _infer_data_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".xls", ".xlsx"}:
        return "excel"
    if suffix == ".parquet":
        return "parquet"
    return "csv"


@dataclass
class SplitConfig:
    """How a dataset should be partitioned into train/test subsets."""

    strategy: str = "predefined"
    column: str = SPLIT_COL
    train_label: str = TRAIN_LABEL
    test_label: str = TEST_LABEL

    def __post_init__(self) -> None:
        self.strategy = self.strategy.strip().lower()
        if self.strategy not in VALID_SPLIT_STRATEGIES:
            raise ValueError(
                f"Unsupported split strategy '{self.strategy}'. "
                f"Supported: {sorted(VALID_SPLIT_STRATEGIES)}"
            )
        if self.strategy not in IMPLEMENTED_SPLIT_STRATEGIES:
            raise ValueError(
                f"Split strategy '{self.strategy}' is not implemented yet. "
                f"Currently supported: {sorted(IMPLEMENTED_SPLIT_STRATEGIES)}"
            )

        if self.strategy in {"predefined", "column"} and not self.column:
            raise ValueError("Split strategy 'predefined' requires a split column")

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "SplitConfig":
        raw = raw or {}
        if not isinstance(raw, dict):
            raise ValueError("'dataset.split' must be a mapping")

        return cls(
            strategy=str(raw.get("strategy", raw.get("method", "predefined"))),
            column="" if raw.get("column", SPLIT_COL) is None else str(raw.get("column", SPLIT_COL)),
            train_label=(
                TRAIN_LABEL
                if raw.get("train_label", TRAIN_LABEL) is None
                else str(raw.get("train_label", TRAIN_LABEL))
            ),
            test_label=(
                TEST_LABEL
                if raw.get("test_label", TEST_LABEL) is None
                else str(raw.get("test_label", TEST_LABEL))
            ),
        )


@dataclass
class DatasetSchema:
    """Contract describing how a raw dataset should be read."""

    path: Path = DEFAULT_DATA_PATH
    format: str = "csv"
    features: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    split: SplitConfig = field(default_factory=SplitConfig)
    reader_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = Path(self.path).expanduser()
        self.format = self.format.strip().lower()
        if self.format not in VALID_DATA_FORMATS:
            raise ValueError(
                f"Unsupported dataset format '{self.format}'. "
                f"Supported: {sorted(VALID_DATA_FORMATS)}"
            )

        self.features = _normalise_str_list(self.features, "dataset.features")
        self.targets = _normalise_str_list(self.targets, "dataset.targets")

        overlap = sorted(set(self.features) & set(self.targets))
        if overlap:
            raise ValueError(
                "Feature and target columns must be disjoint. "
                f"Overlap: {overlap}"
            )

        if not isinstance(self.reader_options, dict):
            raise ValueError("'dataset.reader_options' must be a mapping")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DatasetSchema":
        if not isinstance(raw, dict):
            raise ValueError("'dataset' must be a mapping")

        path = Path(raw.get("path", DEFAULT_DATA_PATH))
        data_format = str(raw.get("format", _infer_data_format(path)))
        features = raw.get("features", raw.get("feature_cols"))
        targets = raw.get("targets", raw.get("target_cols"))

        return cls(
            path=path,
            format=data_format,
            features=features or [],
            targets=targets or [],
            split=SplitConfig.from_dict(raw.get("split")),
            reader_options=dict(raw.get("reader_options", raw.get("read_options", {})) or {}),
        )
