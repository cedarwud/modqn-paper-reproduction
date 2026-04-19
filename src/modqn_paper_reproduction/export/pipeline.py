"""Sweep/export helpers for Phase 01 artifacts."""

from __future__ import annotations

from pathlib import Path

from ..analysis import (
    export_figure_sweep_results,
    export_reward_geometry_analysis,
    export_table_ii_results,
    export_training_log_artifacts,
)
from ..analysis._common import write_json
from ..artifacts import RunArtifactPaths, read_run_metadata, read_training_log
from .replay_bundle import export_replay_bundle, validate_replay_bundle


def export_training_run(input_dir: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Export a completed run artifact into CSV/PNG bundle surfaces."""
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    artifact_paths = RunArtifactPaths(in_dir)
    metadata = read_run_metadata(artifact_paths.run_metadata_json)
    training_log = [row.to_dict() for row in read_training_log(artifact_paths.training_log_json)]

    training_outputs = export_training_log_artifacts(
        out_dir,
        training_log=training_log,
    )
    evaluation_dir = out_dir / "evaluation"
    sweeps_dir = evaluation_dir / "sweeps"
    sweeps_dir.mkdir(parents=True, exist_ok=True)

    assumptions_path = write_json(
        out_dir / "assumptions.json",
        metadata.resolved_assumptions,
    )
    replay_outputs = export_replay_bundle(
        in_dir,
        out_dir,
        metadata=metadata,
    )
    summary_path = write_json(
        evaluation_dir / "summary.json",
        {
            "paper_id": metadata.paper_id,
            "config_path": metadata.config_path,
            "checkpoint_rule": metadata.checkpoint_rule.to_dict(),
            "checkpoint_files": metadata.checkpoint_files.to_dict(),
            "best_eval_summary": metadata.best_eval_summary,
            "training_summary": metadata.training_summary.to_dict(),
            "bundle_schema_version": replay_outputs["bundle_schema_version"],
            "replay_timeline": replay_outputs["replay_summary_model"].to_dict(),
        },
    )
    validate_replay_bundle(out_dir)

    return {
        "manifest": replay_outputs["manifest"],
        "assumptions": assumptions_path,
        "config_resolved_json": replay_outputs["config_resolved"],
        "provenance_map_json": replay_outputs["provenance_map"],
        "timeline_step_trace_jsonl": replay_outputs["timeline_step_trace"],
        "summary_json": summary_path,
        **training_outputs,
    }


__all__ = [
    "export_figure_sweep_results",
    "export_reward_geometry_analysis",
    "export_table_ii_results",
    "export_training_run",
]
