"""Shared helpers for analysis and plotting modules."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_MPL_CACHE_DIR = Path("/tmp/modqn-mpl-cache")
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..artifacts import RunArtifactPaths, read_training_log


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_training_log_dicts(run_dir: str | Path) -> list[dict[str, Any]]:
    paths = RunArtifactPaths(Path(run_dir))
    return [row.to_dict() for row in read_training_log(paths.training_log_json)]


def safe_abs_scale(value: float, floor: float = 1e-12) -> float:
    """Avoid zero or near-zero normalization scales."""
    return max(abs(float(value)), floor)


__all__ = ["load_training_log_dicts", "plt", "safe_abs_scale", "write_json"]
