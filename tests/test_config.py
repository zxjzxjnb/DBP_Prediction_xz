"""Tests for configuration helpers and packaged assets."""

from __future__ import annotations

from pathlib import Path

import pytest

from dbp_prediction.config import PACKAGED_DATA_PATH, first_existing_path, resolve_artifact_path


class TestPathHelpers:
    """Tests for ordered path resolution helpers."""

    def test_first_existing_path_returns_first_match(self, tmp_path: Path) -> None:
        first = tmp_path / "first.txt"
        second = tmp_path / "second.txt"
        second.write_text("ok", encoding="utf-8")
        first.write_text("preferred", encoding="utf-8")

        resolved = first_existing_path([first, second])

        assert resolved == first

    def test_first_existing_path_returns_none_when_missing(self, tmp_path: Path) -> None:
        resolved = first_existing_path([tmp_path / "a.txt", tmp_path / "b.txt"])
        assert resolved is None

    def test_resolve_artifact_path_uses_existing_fallback(self, tmp_path: Path) -> None:
        preferred = tmp_path / "preferred.pt"
        fallback = tmp_path / "fallback.pt"
        fallback.write_text("checkpoint", encoding="utf-8")

        resolved = resolve_artifact_path(None, [preferred, fallback], "checkpoint")

        assert resolved == fallback

    def test_resolve_artifact_path_raises_when_all_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No checkpoint found"):
            resolve_artifact_path(None, [tmp_path / "a.pt", tmp_path / "b.pt"], "checkpoint")


class TestPackagedData:
    """Tests that keep the packaged dataset copy in sync."""

    def test_packaged_dataset_exists(self) -> None:
        assert PACKAGED_DATA_PATH.exists()

    def test_packaged_dataset_matches_repo_copy(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        source = repo_root / "data" / "DBP_dataset_DWTP_B.csv"
        assert PACKAGED_DATA_PATH.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
