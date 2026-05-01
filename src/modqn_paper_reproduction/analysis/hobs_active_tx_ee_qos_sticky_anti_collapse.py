"""QoS-sticky HOBS active-TX EE anti-collapse gate wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .hobs_active_tx_ee_anti_collapse import (
    export_tiny_matched_anti_collapse_pilot,
    interpret_anti_collapse_verdict,
    predeclared_tolerances,
    prove_matched_boundary,
)

CONTROL_CONFIG = Path(
    "configs/hobs-active-tx-ee-qos-sticky-anti-collapse-control.resolved.yaml"
)
CANDIDATE_CONFIG = Path(
    "configs/hobs-active-tx-ee-qos-sticky-anti-collapse-candidate.resolved.yaml"
)
CONTROL_ARTIFACT_DIR = Path(
    "artifacts/hobs-active-tx-ee-qos-sticky-anti-collapse-control"
)
CANDIDATE_ARTIFACT_DIR = Path(
    "artifacts/hobs-active-tx-ee-qos-sticky-anti-collapse-candidate"
)
PAIRED_COMPARISON_DIR = CANDIDATE_ARTIFACT_DIR / "paired-comparison-vs-control"


def export_tiny_matched_qos_sticky_anti_collapse_pilot() -> dict[str, Any]:
    return export_tiny_matched_anti_collapse_pilot(
        control_config_path=CONTROL_CONFIG,
        candidate_config_path=CANDIDATE_CONFIG,
        control_output_dir=CONTROL_ARTIFACT_DIR,
        candidate_output_dir=CANDIDATE_ARTIFACT_DIR,
        paired_output_dir=PAIRED_COMPARISON_DIR,
    )


__all__ = [
    "CANDIDATE_CONFIG",
    "CONTROL_CONFIG",
    "export_tiny_matched_qos_sticky_anti_collapse_pilot",
    "interpret_anti_collapse_verdict",
    "predeclared_tolerances",
    "prove_matched_boundary",
]
