"""Artifact storage helpers for experiment runs."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Recursively convert Python values into JSON-serializable structures."""
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


class ArtifactStore:
    """Small helper around a run output directory."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = Path(root_dir)

    def ensure_dir(self) -> Path:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        return self.root_dir

    def write_json(self, relative_path: str | Path, payload: Any) -> Path:
        path = self.root_dir / Path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(to_jsonable(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    def write_text(self, relative_path: str | Path, content: str) -> Path:
        path = self.root_dir / Path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def copy_file(self, source_path: str | Path, relative_path: str | Path | None = None) -> Path:
        source = Path(source_path)
        target = self.root_dir / (Path(relative_path) if relative_path else source.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return target
