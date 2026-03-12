"""Tests for artifact serialization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dbp_prediction.artifacts import ArtifactStore, to_jsonable


@dataclass
class _Example:
    path: Path
    values: list[int]


class TestArtifactHelpers:
    """Tests for artifact JSON serialization and writes."""

    def test_to_jsonable_serializes_dataclasses_and_paths(self) -> None:
        payload = _Example(path=Path("results/demo"), values=[1, 2, 3])

        assert to_jsonable(payload) == {
            "path": "results/demo",
            "values": [1, 2, 3],
        }

    def test_artifact_store_writes_json(self, tmp_path: Path) -> None:
        store = ArtifactStore(tmp_path / "artifacts")
        output = store.write_json("nested/payload.json", {"path": Path("demo")})

        assert output.exists()
        assert output.read_text(encoding="utf-8").strip().startswith("{")
