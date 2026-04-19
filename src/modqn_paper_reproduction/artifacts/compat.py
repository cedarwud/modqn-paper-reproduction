"""Artifact compatibility helpers shared by export surfaces."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from ..config_loader import load_training_yaml, require_training_config
from .models import RunMetadataV1


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_existing_path(
    raw_path: str | Path,
    *,
    artifact_dir: Path,
    default_subdir: str | None = None,
) -> Path:
    raw = Path(raw_path)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                raw,
                _repo_root() / raw,
                artifact_dir / raw,
            ]
        )
        if default_subdir is not None:
            candidates.append(artifact_dir / default_subdir / raw.name)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not resolve artifact path {raw_path!r} from {artifact_dir}."
    )


def resolve_training_config_snapshot(
    metadata: RunMetadataV1,
    *,
    artifact_dir: Path,
) -> dict[str, Any]:
    snapshot = metadata.resolved_config_snapshot
    if snapshot:
        require_training_config(
            snapshot,
            config_path="<run_metadata.resolved_config_snapshot>",
        )
        return copy.deepcopy(snapshot)

    config_path = metadata.config_path
    if not config_path:
        raise FileNotFoundError(
            "Run artifact is missing both resolved_config_snapshot and config_path."
        )
    resolved_path = _resolve_existing_path(config_path, artifact_dir=artifact_dir)
    return load_training_yaml(resolved_path)


def select_replay_checkpoint(
    metadata: RunMetadataV1,
    *,
    artifact_dir: Path,
) -> tuple[Path, str]:
    secondary = metadata.checkpoint_files.secondary_best_eval
    if secondary:
        return (
            _resolve_existing_path(
                secondary,
                artifact_dir=artifact_dir,
                default_subdir="checkpoints",
            ),
            "best-weighted-reward-on-eval",
        )

    primary = metadata.checkpoint_files.primary_final
    if primary:
        return (
            _resolve_existing_path(
                primary,
                artifact_dir=artifact_dir,
                default_subdir="checkpoints",
            ),
            "final-episode-policy",
        )

    raise FileNotFoundError(
        "Run artifact does not expose a replayable checkpoint file in checkpoint_files."
    )


def _select_timeline_seed(
    metadata: RunMetadataV1,
    checkpoint_payload: dict[str, Any],
    *,
    cfg: dict[str, Any],
) -> tuple[int, str]:
    """Pick a deterministic replay seed and disclose where it came from."""
    del cfg
    evaluation_summary = checkpoint_payload.get("evaluation_summary", {})
    if isinstance(evaluation_summary, dict):
        eval_seeds = evaluation_summary.get("eval_seeds", [])
        if eval_seeds:
            return (
                int(eval_seeds[0]),
                "checkpoint.evaluation_summary.eval_seeds[0]",
            )

    best_eval_summary = metadata.best_eval_summary
    if isinstance(best_eval_summary, dict):
        eval_seeds = best_eval_summary.get("eval_seeds", [])
        if eval_seeds:
            return (
                int(eval_seeds[0]),
                "run_metadata.best_eval_summary.eval_seeds[0]",
            )

    seeds = metadata.seeds.to_dict()
    seed_source = "run_metadata.seeds.evaluation_seed_set[0]"
    evaluation_seed_set = seeds.get("evaluation_seed_set", [])
    if evaluation_seed_set:
        return (int(evaluation_seed_set[0]), seed_source)
    return (
        int(seeds.get("train_seed", 42)),
        seed_source.replace("evaluation_seed_set[0]", "train_seed"),
    )
