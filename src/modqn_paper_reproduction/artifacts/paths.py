"""Path helpers for the training artifact contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import CheckpointRuleV1


@dataclass(frozen=True)
class RunArtifactPaths:
    run_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_dir", Path(self.run_dir))

    @property
    def training_log_json(self) -> Path:
        return self.run_dir / "training_log.json"

    @property
    def run_metadata_json(self) -> Path:
        return self.run_dir / "run_metadata.json"

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    def primary_checkpoint(self, rule: CheckpointRuleV1) -> Path:
        return self.checkpoints_dir / f"{rule.primary_report}.pt"

    def secondary_checkpoint(self, rule: CheckpointRuleV1) -> Path:
        return self.checkpoints_dir / f"{rule.secondary_report}.pt"
