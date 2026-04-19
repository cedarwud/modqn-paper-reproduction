"""Pure loaders and assertion helpers for Phase 04 Slice A golden tests.

Deliberately free of pytest state: every helper takes a ``Path`` or
parsed structure and returns data or raises. Fixtures live in the
calling test module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    """Read and parse a JSON file."""
    return json.loads(Path(path).read_text())


def load_timeline_rows(bundle_dir: Path) -> list[dict[str, Any]]:
    """Return every non-empty row in ``timeline/step-trace.jsonl``."""
    rows: list[dict[str, Any]] = []
    timeline_path = Path(bundle_dir) / "timeline" / "step-trace.jsonl"
    with timeline_path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def assert_key_set(actual: dict, expected: set[str], *, context: str) -> None:
    """Assert that ``actual`` has exactly the keys in ``expected``.

    Emits a per-difference message so the failure identifies which
    side of the contract drifted.
    """
    actual_keys = set(actual.keys())
    missing = expected - actual_keys
    extra = actual_keys - expected
    assert not missing and not extra, (
        f"{context} key-set mismatch: "
        f"missing={sorted(missing)} extra={sorted(extra)}"
    )


def assert_key_superset(
    actual: dict, required: set[str], *, context: str
) -> None:
    """Assert ``actual`` contains every key in ``required`` (extras allowed)."""
    missing = required - set(actual.keys())
    assert not missing, (
        f"{context} is missing required keys: {sorted(missing)}"
    )


def assert_type(value: Any, expected_type, *, context: str) -> None:
    """Assert ``value`` is an instance of ``expected_type``."""
    assert isinstance(value, expected_type), (
        f"{context} should be {expected_type}, got {type(value).__name__}"
    )


def assert_optional_type(
    value: Any, expected_type, *, context: str
) -> None:
    """Assert ``value`` is ``None`` or an instance of ``expected_type``."""
    if value is None:
        return
    assert isinstance(value, expected_type), (
        f"{context} should be None or {expected_type}, "
        f"got {type(value).__name__}"
    )


def iter_directory_files(root: Path) -> list[Path]:
    """Return every regular file under ``root`` in sorted order."""
    return sorted(p for p in Path(root).rglob("*") if p.is_file())
