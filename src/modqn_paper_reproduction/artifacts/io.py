"""Serialization helpers for the Phase 04B training artifact seam."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch

from .models import (
    CheckpointPayloadV1,
    RunMetadataV1,
    TrainingLogRow,
)


def _write_json(path: Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2))
    return target


def write_run_metadata(path: Path, metadata: RunMetadataV1) -> Path:
    return _write_json(path, metadata.to_dict())


def read_run_metadata(path: Path) -> RunMetadataV1:
    return RunMetadataV1.from_dict(json.loads(Path(path).read_text()))


def write_training_log(path: Path, rows: Iterable[TrainingLogRow]) -> Path:
    payload = [row.to_dict() for row in rows]
    return _write_json(path, payload)


def read_training_log(path: Path) -> list[TrainingLogRow]:
    payload = json.loads(Path(path).read_text())
    return [TrainingLogRow.from_dict(row) for row in payload]


def write_checkpoint(path: Path, payload: CheckpointPayloadV1) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload.to_dict(), checkpoint_path)
    return checkpoint_path


def read_checkpoint(
    path: Path,
    *,
    map_location: Any = "cpu",
) -> CheckpointPayloadV1:
    payload = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=False,
    )
    return CheckpointPayloadV1.from_dict(payload)
