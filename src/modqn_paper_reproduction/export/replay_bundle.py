"""Replay-bundle export helpers for the Phase 03A producer surface."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .. import PACKAGE_VERSION
from ..algorithms.modqn import MODQNTrainer
from ..artifacts import RunMetadataV1
from ..artifacts.compat import (
    _resolve_existing_path,
    _select_timeline_seed,
    resolve_training_config_snapshot,
    select_replay_checkpoint,
)
from ..bundle.fixture_tools import (
    sync_replay_summary_in_evaluation_summary,
    trim_replay_bundle_for_sample,
)
from ..bundle.models import ReplaySummary
from ..bundle.provenance import build_provenance_map
from ..bundle.schema import (
    BEAM_CATALOG_ORDER,
    BUNDLE_SCHEMA_VERSION,
    POLICY_DIAGNOSTICS_TIMELINE_FIELD,
    POLICY_DIAGNOSTICS_TOP_CANDIDATES,
    POLICY_DIAGNOSTICS_VERSION,
    REPLAY_TRUTH_MODE,
    SLOT_INDEX_NOTE,
    SLOT_INDEX_OFFSET,
    TIMELINE_FORMAT_VERSION,
    TIMELINE_ROW_REQUIRED_FIELDS,
)
from ..bundle.serializers import (
    build_optional_policy_diagnostics_manifest,
    build_replay_summary,
    serialize_beam_states,
    serialize_satellite_states,
    serialize_timeline_row,
)
from ..bundle.validator import validate_replay_bundle
from ..config_loader import build_environment, build_trainer_config


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def _timeline_row_required_fields() -> tuple[str, ...]:
    return TIMELINE_ROW_REQUIRED_FIELDS


def export_replay_bundle(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    metadata: RunMetadataV1,
    replay_start_time_s: float = 0.0,
    replay_slot_count: int | None = None,
) -> dict[str, Any]:
    """Export the replay-complete Phase 03A producer bundle surfaces."""
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    if replay_start_time_s < 0.0:
        raise ValueError(
            f"replay_start_time_s must be >= 0.0, got {replay_start_time_s!r}"
        )
    if replay_slot_count is not None and replay_slot_count < 1:
        raise ValueError(
            f"replay_slot_count must be >= 1 when provided, got {replay_slot_count!r}"
        )
    cfg = resolve_training_config_snapshot(metadata, artifact_dir=in_dir)
    trainer_cfg = build_trainer_config(cfg)
    seeds = metadata.seeds.to_dict()

    env = build_environment(copy.deepcopy(cfg))
    trainer = MODQNTrainer(
        env=env,
        config=trainer_cfg,
        train_seed=int(seeds["train_seed"]),
        env_seed=int(seeds["environment_seed"]),
        mobility_seed=int(seeds["mobility_seed"]),
    )

    checkpoint_path, replay_checkpoint_kind = select_replay_checkpoint(
        metadata,
        artifact_dir=in_dir,
    )
    checkpoint_payload = trainer.load_checkpoint(
        checkpoint_path,
        load_optimizers=False,
    )
    replay_seed, replay_seed_source = _select_timeline_seed(
        metadata,
        checkpoint_payload,
        cfg=cfg,
    )
    objective_weights = tuple(
        float(x)
        for x in checkpoint_payload.get(
            "trainer_config",
            {},
        ).get("objective_weights", trainer.config.objective_weights)
    )

    timeline_dir = out_dir / "timeline"
    timeline_path = timeline_dir / "step-trace.jsonl"
    timeline_dir.mkdir(parents=True, exist_ok=True)

    env_seed_seq, mobility_seed_seq = np.random.SeedSequence(replay_seed).spawn(2)
    env_rng = np.random.default_rng(env_seed_seq)
    mobility_rng = np.random.default_rng(mobility_seed_seq)
    states, masks, diagnostics = trainer.env.reset(
        env_rng,
        mobility_rng,
        initial_time_s=replay_start_time_s,
    )
    encoded = trainer.encode_states(states)

    ground_lat_deg = float(trainer.env.config.user_lat_deg)
    ground_lon_deg = float(trainer.env.config.user_lon_deg)
    beams_per_satellite = int(trainer.env.beam_pattern.num_beams)
    row_count = 0
    slot_count = 0
    handover_count = 0
    diagnostics_row_count = 0
    diagnostics_missing_row_count = 0

    with timeline_path.open("w") as handle:
        while True:
            previous_time_s = float(trainer.env.time_s)
            previous_positions = trainer.env.current_user_positions()
            previous_assignments = trainer.env.current_assignments()
            decision_masks = [mask.mask.copy() for mask in masks]

            actions, policy_diagnostics = trainer.select_actions_with_diagnostics(
                encoded,
                masks,
                objective_weights=objective_weights,
                top_k=POLICY_DIAGNOSTICS_TOP_CANDIDATES,
            )
            result = trainer.env.step(actions, env_rng)

            post_positions = trainer.env.current_user_positions()
            post_masks = [mask.mask.copy() for mask in result.action_masks]
            beam_loads = (
                [int(value) for value in result.user_states[0].beam_loads.tolist()]
                if result.user_states
                else []
            )
            beam_throughputs = [
                float(value) for value in result.beam_throughputs.tolist()
            ]
            satellite_states = serialize_satellite_states(trainer.env)
            beam_states = serialize_beam_states(trainer.env)

            slot_count = int(result.step_index)
            for user_index in range(trainer.num_users):
                row = serialize_timeline_row(
                    result=result,
                    user_index=user_index,
                    previous_time_s=previous_time_s,
                    previous_positions=previous_positions,
                    post_positions=post_positions,
                    previous_assignments=previous_assignments,
                    actions=actions,
                    decision_masks=decision_masks,
                    post_masks=post_masks,
                    beam_loads=beam_loads,
                    beam_throughputs=beam_throughputs,
                    satellite_states=satellite_states,
                    beam_states=beam_states,
                    objective_weights=objective_weights,
                    beams_per_satellite=beams_per_satellite,
                    ground_lat_deg=ground_lat_deg,
                    ground_lon_deg=ground_lon_deg,
                    policy_diagnostics=policy_diagnostics[user_index],
                )
                if row["handoverEvent"].get("eventId") is not None:
                    handover_count += 1
                if POLICY_DIAGNOSTICS_TIMELINE_FIELD in row:
                    diagnostics_row_count += 1
                else:
                    diagnostics_missing_row_count += 1
                handle.write(json.dumps(row) + "\n")
                row_count += 1

            stop_after_slot = (
                replay_slot_count is not None
                and result.step_index >= replay_slot_count
            ) or (replay_slot_count is None and result.done)
            if stop_after_slot:
                break

            encoded = trainer.encode_states(result.user_states)
            masks = result.action_masks

    config_resolved_path = _write_json(out_dir / "config-resolved.json", cfg)
    provenance_path = _write_json(
        out_dir / "provenance-map.json",
        build_provenance_map(cfg, metadata),
    )

    replay_summary = build_replay_summary(
        checkpoint_path=str(checkpoint_path),
        checkpoint_kind=replay_checkpoint_kind,
        policy_episode=int(checkpoint_payload.get("episode", -1)),
        timeline_seed=int(replay_seed),
        replay_seed_source=replay_seed_source,
        row_count=row_count,
        slot_count=slot_count,
        handover_event_count=handover_count,
        reward_weights=objective_weights,
        diagnostics=diagnostics,
    )
    optional_policy_diagnostics = build_optional_policy_diagnostics_manifest(
        diagnostics_row_count=diagnostics_row_count,
        diagnostics_missing_row_count=diagnostics_missing_row_count,
    )
    replay_window = None
    if replay_start_time_s > 0.0 or replay_slot_count is not None:
        replay_window = {
            "startTimeSec": float(replay_start_time_s),
            "slotCount": (
                None
                if replay_slot_count is None
                else int(replay_slot_count)
            ),
            "selectionMode": "producer-configured-replay-window",
        }

    manifest = {
        "paperId": metadata.paper_id,
        "runId": in_dir.name,
        "bundleSchemaVersion": BUNDLE_SCHEMA_VERSION,
        "producerVersion": metadata.package_version or PACKAGE_VERSION,
        "exportedAt": datetime.now(timezone.utc).isoformat(),
        "sourceArtifactDir": str(in_dir),
        "inputArtifactDir": str(in_dir),
        "outputDir": str(out_dir),
        "configPath": metadata.config_path,
        "checkpointRule": metadata.checkpoint_rule.to_dict(),
        "replayTruthMode": REPLAY_TRUTH_MODE,
        "timelineFormatVersion": TIMELINE_FORMAT_VERSION,
        "coordinateFrame": {
            "userPosition": (
                "geodetic-deg + local-tangent-east-north-km anchored at "
                f"ground_point=({ground_lat_deg:.6f}, {ground_lon_deg:.6f})"
            ),
            "satellitePosition": "eci-km-no-earth-rotation",
            "beamCenter": (
                "geodetic-deg + local-tangent-east-north-km anchored at "
                f"ground_point=({ground_lat_deg:.6f}, {ground_lon_deg:.6f})"
            ),
            "groundPoint": {
                "latDeg": float(ground_lat_deg),
                "lonDeg": float(ground_lon_deg),
            },
        },
        "scenarioSurface": {
            "groundPoint": {
                "latDeg": float(ground_lat_deg),
                "lonDeg": float(ground_lon_deg),
            },
            "actionMaskEligibilityMode": str(
                trainer.env.config.action_mask_eligibility_mode
            ),
            "userAreaDistribution": str(trainer.env.config.user_scatter_distribution),
            "userScatterRadiusKm": float(trainer.env.config.user_scatter_radius_km),
            "userAreaWidthKm": float(trainer.env.config.user_area_width_km),
            "userAreaHeightKm": float(trainer.env.config.user_area_height_km),
            "mobilityModel": str(trainer.env.config.mobility_model),
            "randomWanderingMaxTurnRad": float(
                trainer.env.config.random_wandering_max_turn_rad
            ),
        },
        "slotIndexSemantics": {
            "firstIndex": SLOT_INDEX_OFFSET,
            "note": SLOT_INDEX_NOTE,
        },
        "beamCatalogOrder": BEAM_CATALOG_ORDER,
        "replaySummary": replay_summary.to_dict(),
        "optionalPolicyDiagnostics": optional_policy_diagnostics,
    }
    if replay_window is not None:
        manifest["replayWindow"] = replay_window
    manifest_path = _write_json(out_dir / "manifest.json", manifest)

    return {
        "config_resolved": config_resolved_path,
        "provenance_map": provenance_path,
        "timeline_step_trace": timeline_path,
        "manifest": manifest_path,
        "replay_summary": replay_summary.to_dict(),
        "replay_summary_model": replay_summary,
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "timeline_required_fields": list(TIMELINE_ROW_REQUIRED_FIELDS),
    }


__all__ = [
    "BEAM_CATALOG_ORDER",
    "BUNDLE_SCHEMA_VERSION",
    "POLICY_DIAGNOSTICS_TIMELINE_FIELD",
    "POLICY_DIAGNOSTICS_TOP_CANDIDATES",
    "POLICY_DIAGNOSTICS_VERSION",
    "ReplaySummary",
    "TIMELINE_FORMAT_VERSION",
    "_resolve_existing_path",
    "_select_timeline_seed",
    "_timeline_row_required_fields",
    "build_provenance_map",
    "export_replay_bundle",
    "resolve_training_config_snapshot",
    "select_replay_checkpoint",
    "sync_replay_summary_in_evaluation_summary",
    "trim_replay_bundle_for_sample",
    "validate_replay_bundle",
]
