"""Replay-bundle export helpers for the Phase 03A producer surface."""

from __future__ import annotations

import copy
import json
import math
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .. import PACKAGE_VERSION
from ..algorithms.modqn import MODQNTrainer, scalarize_objectives
from ..artifacts import RunMetadataV1
from ..config_loader import (
    build_environment,
    build_trainer_config,
    load_training_yaml,
    require_training_config,
)
from ..env.step import local_tangent_offset_km


BUNDLE_SCHEMA_VERSION = "phase-03a-replay-bundle-v1"
TIMELINE_FORMAT_VERSION = "step-trace.jsonl/v1"
BEAM_CATALOG_ORDER = "satellite-major-beam-minor"
POLICY_DIAGNOSTICS_VERSION = "phase-03b-policy-diagnostics-v1"
POLICY_DIAGNOSTICS_TIMELINE_FIELD = "policyDiagnostics"
POLICY_DIAGNOSTICS_TOP_CANDIDATES = 3

_PAPER_BACKED_BASELINE_FIELDS: dict[str, str] = {
    "satellites": "Table I baseline topology count.",
    "beams_per_satellite": "Table I baseline beam count.",
    "users": "Baseline user count used by the paper protocol.",
    "altitude_km": "Table I satellite altitude.",
    "user_speed_kmh": "Figure sweep baseline user speed.",
    "satellite_speed_km_s": "Figure sweep baseline satellite speed.",
    "carrier_frequency_ghz": "Table I carrier frequency.",
    "bandwidth_mhz": "Table I bandwidth.",
    "tx_power_w": "Table I transmit power.",
    "noise_psd_dbm_hz": "Table I thermal-noise PSD.",
    "rician_k_db": "Table I Rician K factor.",
    "atmospheric_attenuation_coefficient_db_per_km": "Table I attenuation coefficient.",
    "slot_duration_s": "Paper slot duration.",
    "episode_duration_s": "Paper episode duration.",
    "episodes": "Paper baseline training horizon.",
    "hidden_layers": "Paper network width/depth.",
    "activation": "Paper activation surface.",
    "optimizer": "Paper optimizer family.",
    "learning_rate": "Paper learning rate.",
    "discount_factor": "Paper discount factor.",
    "batch_size": "Paper batch size.",
    "objective_weights": "Baseline objective weights used for the primary run.",
}

_RECOVERED_FIELDS: dict[str, str] = {
    "configResolved.paper_backed_weight_rows.table_ii": "Recovered Table II weight rows.",
    "configResolved.paper_ranges.user_count": "Recovered user-count figure range.",
    "configResolved.paper_ranges.satellite_count": "Recovered satellite-count figure range.",
    "configResolved.paper_ranges.user_speed_kmh": "Recovered user-speed figure range.",
    "configResolved.paper_ranges.satellite_speed_km_s": "Recovered satellite-speed figure range.",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


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
        require_training_config(snapshot, config_path="<run_metadata.resolved_config_snapshot>")
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
    """Pick a deterministic replay seed and report which source provided it.

    Returns ``(seed, source_label)``. The source label is embedded in
    ``manifest.replaySummary.replaySeedSource`` so the consumer can
    disclose *why* a particular scenario is being replayed instead of
    silently trusting a numeric seed.
    """
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
    return (int(seeds.get("train_seed", 42)), seed_source.replace(
        "evaluation_seed_set[0]", "train_seed"
    ))


def _satellite_id(index: int) -> str:
    return f"sat-{index}"


def _beam_id(sat_index: int, local_beam_index: int) -> str:
    return f"sat-{sat_index}-beam-{local_beam_index}"


def _user_id(index: int) -> str:
    return f"user-{index}"


def _serving_state(
    beam_index: int,
    *,
    beams_per_satellite: int,
    decision_mask: np.ndarray | None = None,
    post_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    sat_index = int(beam_index) // beams_per_satellite
    local_beam_index = int(beam_index) % beams_per_satellite
    payload = {
        "beamId": _beam_id(sat_index, local_beam_index),
        "beamIndex": int(beam_index),
        "satId": _satellite_id(sat_index),
        "satIndex": sat_index,
        "localBeamIndex": local_beam_index,
    }
    if decision_mask is not None and 0 <= beam_index < len(decision_mask):
        payload["validUnderDecisionMask"] = bool(decision_mask[beam_index])
    if post_mask is not None and 0 <= beam_index < len(post_mask):
        payload["validUnderPostStepMask"] = bool(post_mask[beam_index])
    return payload


def _objective_triplet_object(
    values: list[float] | tuple[float, float, float],
) -> dict[str, float]:
    triplet = [float(value) for value in values]
    if len(triplet) != 3:
        raise ValueError(
            "Expected a three-objective payload, "
            f"got {values!r}."
        )
    return {
        "r1Throughput": float(triplet[0]),
        "r2Handover": float(triplet[1]),
        "r3LoadBalance": float(triplet[2]),
    }


def _serialize_policy_diagnostics(
    policy_diagnostics: dict[str, Any],
    *,
    beams_per_satellite: int,
    decision_mask: np.ndarray,
) -> dict[str, Any]:
    top_candidates: list[dict[str, Any]] = []
    for candidate in policy_diagnostics.get("topCandidates", []):
        beam_index = int(candidate["action"])
        candidate_payload = _serving_state(
            beam_index,
            beams_per_satellite=beams_per_satellite,
            decision_mask=decision_mask,
        )
        candidate_valid = bool(candidate.get("validUnderDecisionMask", False))
        if not bool(candidate_payload.get("validUnderDecisionMask", False)) or not candidate_valid:
            raise ValueError(
                "Policy diagnostics top candidates must remain valid under the decision mask."
            )
        candidate_payload["validUnderDecisionMask"] = True
        candidate_payload["objectiveQ"] = _objective_triplet_object(
            candidate["objectiveQ"]
        )
        candidate_payload["scalarizedQ"] = float(candidate["scalarizedQ"])
        top_candidates.append(candidate_payload)

    if not top_candidates:
        raise ValueError("Policy diagnostics require at least one top candidate.")

    selected_action = int(policy_diagnostics["selectedAction"])
    if int(top_candidates[0]["beamIndex"]) != selected_action:
        raise ValueError(
            "Policy diagnostics topCandidates[0] must align with the selected action."
        )

    return {
        "diagnosticsVersion": POLICY_DIAGNOSTICS_VERSION,
        "objectiveWeights": _objective_triplet_object(
            policy_diagnostics["objectiveWeights"]
        ),
        "selectedScalarizedQ": float(policy_diagnostics["selectedScalarizedQ"]),
        "runnerUpScalarizedQ": (
            None
            if policy_diagnostics["runnerUpScalarizedQ"] is None
            else float(policy_diagnostics["runnerUpScalarizedQ"])
        ),
        "scalarizedMarginToRunnerUp": (
            None
            if policy_diagnostics["scalarizedMarginToRunnerUp"] is None
            else float(policy_diagnostics["scalarizedMarginToRunnerUp"])
        ),
        "availableActionCount": int(policy_diagnostics["availableActionCount"]),
        "topCandidates": top_candidates,
    }


def _handover_event(
    prev_beam: int,
    cur_beam: int,
    *,
    beams_per_satellite: int,
    slot_index: int,
    user_index: int,
) -> dict[str, Any]:
    if prev_beam == cur_beam:
        return {"kind": "none", "eventId": None}

    prev_sat = int(prev_beam) // beams_per_satellite
    cur_sat = int(cur_beam) // beams_per_satellite
    if prev_sat == cur_sat:
        kind = "intra-satellite-beam-switch"
    else:
        kind = "inter-satellite-handover"
    return {
        "kind": kind,
        "eventId": f"handover-slot-{slot_index}-user-{user_index}",
    }


def _serialize_user_position(
    lat_deg: float,
    lon_deg: float,
    *,
    ground_lat_deg: float,
    ground_lon_deg: float,
) -> dict[str, Any]:
    east_km, north_km = local_tangent_offset_km(
        ground_lat_deg,
        ground_lon_deg,
        lat_deg,
        lon_deg,
    )
    return {
        "latDeg": float(lat_deg),
        "lonDeg": float(lon_deg),
        "localTangentKm": {
            "east": float(east_km),
            "north": float(north_km),
        },
    }


def _serialize_satellite_states(env) -> list[dict[str, Any]]:
    return [
        {
            "satId": _satellite_id(sat.index),
            "satIndex": int(sat.index),
            "trueAnomalyDeg": float(sat.true_anomaly_deg),
            "positionEciKm": {
                "x": float(sat.x_km),
                "y": float(sat.y_km),
                "z": float(sat.z_km),
            },
            "subSatellitePoint": {
                "latDeg": float(sat.lat_deg),
                "lonDeg": float(sat.lon_deg),
            },
        }
        for sat in env.current_satellites()
    ]


def _serialize_beam_states(env) -> list[dict[str, Any]]:
    beam_states: list[dict[str, Any]] = []
    ground_lat_deg = float(env.config.user_lat_deg)
    ground_lon_deg = float(env.config.user_lon_deg)
    beams_per_satellite = int(env.beam_pattern.num_beams)
    for sat in env.current_satellites():
        for center in env.beam_pattern.beam_centers_ground(sat):
            beam_index = sat.index * beams_per_satellite + center.local_beam_index
            if math.isnan(center.lat_deg) or math.isnan(center.lon_deg):
                local_tangent = None
                center_position = None
            else:
                east_km, north_km = local_tangent_offset_km(
                    ground_lat_deg,
                    ground_lon_deg,
                    center.lat_deg,
                    center.lon_deg,
                )
                local_tangent = {
                    "east": float(east_km),
                    "north": float(north_km),
                }
                center_position = {
                    "latDeg": float(center.lat_deg),
                    "lonDeg": float(center.lon_deg),
                }
            beam_states.append(
                {
                    "beamId": _beam_id(sat.index, center.local_beam_index),
                    "beamIndex": int(beam_index),
                    "satId": _satellite_id(sat.index),
                    "satIndex": int(sat.index),
                    "localBeamIndex": int(center.local_beam_index),
                    "centerPosition": center_position,
                    "centerLocalTangentKm": local_tangent,
                }
            )
    return beam_states


def _build_provenance_entry(
    *,
    primary_classification: str,
    source: str,
    note: str,
    assumption_id: str | None = None,
    source_chain: list[str] | None = None,
) -> dict[str, Any]:
    payload = {
        "primaryClassification": primary_classification,
        "source": source,
        "note": note,
    }
    if assumption_id is not None:
        payload["assumptionId"] = assumption_id
    if source_chain is not None:
        payload["sourceChain"] = list(source_chain)
    return payload


def build_provenance_map(
    cfg: dict[str, Any],
    metadata: RunMetadataV1,
) -> dict[str, Any]:
    paper = cfg.get("paper", {})
    baseline = cfg.get("baseline", {})
    resolved = cfg.get("resolved_assumptions", {})
    fields: dict[str, Any] = {
        "configResolved.paper.paper_id": _build_provenance_entry(
            primary_classification="paper-backed",
            source="paper-source/ref PDF",
            note="Portable paper identifier for PAP-2024-MORL-MULTIBEAM.",
        ),
        "configResolved.paper.title": _build_provenance_entry(
            primary_classification="paper-backed",
            source="paper-source/ref PDF",
            note="Portable paper title snapshot.",
        ),
        "manifest.coordinateFrame": _build_provenance_entry(
            primary_classification="platform-visualization-only",
            source="Phase 03A replay bundle contract",
            note="Consumer-facing rendering coordinate metadata; not a training authority input.",
        ),
        "timeline.stepTrace.userPosition.localTangentKm": _build_provenance_entry(
            primary_classification="platform-visualization-only",
            source="Phase 03A replay bundle contract",
            note="Derived export convenience for frontend rendering; geodetic user positions remain the underlying truth.",
        ),
        "timeline.stepTrace.rewardVector": _build_provenance_entry(
            primary_classification="artifact-derived",
            source="checkpoint replay over the resolved config",
            note="Per-slot replay reward components derived from the exported checkpoint and environment semantics.",
            source_chain=["paper-backed", "reproduction-assumption"],
        ),
        "timeline.stepTrace.selectedServing": _build_provenance_entry(
            primary_classification="artifact-derived",
            source="checkpoint replay over the resolved config",
            note="Serving-beam truth selected by the exported checkpoint during deterministic replay.",
            source_chain=["paper-backed", "reproduction-assumption"],
        ),
        "timeline.stepTrace.satelliteStates": _build_provenance_entry(
            primary_classification="artifact-derived",
            source="checkpoint replay over the resolved config",
            note="Per-slot satellite geometry derived from the replay timeline.",
            source_chain=["reproduction-assumption"],
        ),
        "timeline.stepTrace.beamStates": _build_provenance_entry(
            primary_classification="artifact-derived",
            source="checkpoint replay over the resolved config",
            note="Per-slot beam geometry derived from the replay timeline.",
            source_chain=["reproduction-assumption"],
        ),
        "timeline.stepTrace.policyDiagnostics": _build_provenance_entry(
            primary_classification="artifact-derived",
            source="checkpoint replay over the selected checkpoint",
            note="Optional producer-owned greedy policy diagnostics derived from the same masked decision surface as the exported selected serving action.",
            source_chain=["artifact-derived"],
        ),
        "evaluation.summary.training_summary": _build_provenance_entry(
            primary_classification="artifact-derived",
            source="training_log.json and run_metadata.json",
            note="Aggregated training-run summary exported from the completed artifact.",
            source_chain=["paper-backed", "reproduction-assumption"],
        ),
        "evaluation.summary.best_eval_summary": _build_provenance_entry(
            primary_classification="artifact-derived",
            source="run_metadata.json",
            note="Evaluation summary captured during checkpoint selection.",
            source_chain=["paper-backed", "reproduction-assumption"],
        ),
    }

    if isinstance(paper, dict):
        if "paper_id" in paper:
            fields["configResolved.paper.paper_id"] = _build_provenance_entry(
                primary_classification="paper-backed",
                source="paper-source/ref PDF",
                note=f"Portable paper identifier: {paper['paper_id']}.",
            )
        if "title" in paper:
            fields["configResolved.paper.title"] = _build_provenance_entry(
                primary_classification="paper-backed",
                source="paper-source/ref PDF",
                note="Portable paper title snapshot.",
            )

    if isinstance(baseline, dict):
        for key, note in _PAPER_BACKED_BASELINE_FIELDS.items():
            if key in baseline:
                fields[f"configResolved.baseline.{key}"] = _build_provenance_entry(
                    primary_classification="paper-backed",
                    source="paper-source/ref PDF or paper-backed baseline envelope",
                    note=note,
                )

    for field_path, note in _RECOVERED_FIELDS.items():
        fields[field_path] = _build_provenance_entry(
            primary_classification="recovered-from-paper",
            source="paper-source/txt_layout and paper-source/catalog",
            note=note,
        )

    if isinstance(resolved, dict):
        for name, block in sorted(resolved.items()):
            if not isinstance(block, dict):
                continue
            fields[f"configResolved.resolved_assumptions.{name}"] = _build_provenance_entry(
                primary_classification="reproduction-assumption",
                source="resolved-run config",
                note="Explicit reproduction assumption promoted into the executable config surface.",
                assumption_id=str(block.get("assumption_id", "")) or None,
            )

    fields["manifest.replayTruthMode"] = _build_provenance_entry(
        primary_classification="artifact-derived",
        source="exporter checkpoint selection",
        note="Replay truth source chosen from the exported run artifact.",
        source_chain=["artifact-derived"],
    )
    fields["manifest.optionalPolicyDiagnostics"] = _build_provenance_entry(
        primary_classification="artifact-derived",
        source="Phase 03B additive producer diagnostics export",
        note="Manifest disclosure for the optional policyDiagnostics row surface.",
        source_chain=["artifact-derived"],
    )

    return {
        "bundleSchemaVersion": BUNDLE_SCHEMA_VERSION,
        "classificationLegend": {
            "paper-backed": "Directly supported by the paper or its authority snapshot.",
            "recovered-from-paper": "Recovered from paper tables/figures/text but normalized into executable data.",
            "reproduction-assumption": "Explicit assumption required to make the repo executable and reproducible.",
            "platform-visualization-only": "Consumer/rendering convenience that must not be treated as training authority.",
            "artifact-derived": "Computed from the exported checkpoint and run artifact rather than copied from source authority.",
        },
        "fieldCount": len(fields),
        "producerVersion": str(metadata.package_version or PACKAGE_VERSION),
        "fields": fields,
    }


def _timeline_row_required_fields() -> tuple[str, ...]:
    return (
        "slotIndex",
        "timeSec",
        "userId",
        "userPosition",
        "previousServing",
        "selectedServing",
        "handoverEvent",
        "visibilityMask",
        "actionValidityMask",
        "beamLoads",
        "rewardVector",
        "scalarReward",
        "satelliteStates",
        "beamStates",
        "kpiOverlay",
    )


def export_replay_bundle(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    metadata: RunMetadataV1,
) -> dict[str, Any]:
    """Export the replay-complete Phase 03A producer bundle surfaces."""
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
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
        metadata, checkpoint_payload, cfg=cfg
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
    states, masks, diagnostics = trainer.env.reset(env_rng, mobility_rng)
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
            beam_loads = [
                int(value)
                for value in result.user_states[0].beam_loads.tolist()
            ] if result.user_states else []
            beam_throughputs = [
                float(value) for value in result.beam_throughputs.tolist()
            ]
            satellite_states = _serialize_satellite_states(trainer.env)
            beam_states = _serialize_beam_states(trainer.env)

            slot_count = int(result.step_index)
            for uid in range(trainer.num_users):
                reward = result.rewards[uid]
                reward_vector = {
                    "r1Throughput": float(reward.r1_throughput),
                    "r2Handover": float(reward.r2_handover),
                    "r3LoadBalance": float(reward.r3_load_balance),
                }
                scalar_reward = scalarize_objectives(
                    (
                        reward.r1_throughput,
                        reward.r2_handover,
                        reward.r3_load_balance,
                    ),
                    objective_weights,
                )
                previous_beam = int(previous_assignments[uid])
                selected_beam = int(actions[uid])
                handover_event = _handover_event(
                    previous_beam,
                    selected_beam,
                    beams_per_satellite=beams_per_satellite,
                    slot_index=result.step_index,
                    user_index=uid,
                )
                if handover_event["eventId"] is not None:
                    handover_count += 1

                row = {
                    "slotIndex": int(result.step_index),
                    "timeSec": float(result.time_s),
                    "decisionTimeSec": previous_time_s,
                    "userId": _user_id(uid),
                    "userIndex": uid,
                    "userPosition": _serialize_user_position(
                        post_positions[uid][0],
                        post_positions[uid][1],
                        ground_lat_deg=ground_lat_deg,
                        ground_lon_deg=ground_lon_deg,
                    ),
                    "decisionUserPosition": _serialize_user_position(
                        previous_positions[uid][0],
                        previous_positions[uid][1],
                        ground_lat_deg=ground_lat_deg,
                        ground_lon_deg=ground_lon_deg,
                    ),
                    "previousServing": _serving_state(
                        previous_beam,
                        beams_per_satellite=beams_per_satellite,
                        decision_mask=decision_masks[uid],
                        post_mask=post_masks[uid],
                    ),
                    "selectedServing": _serving_state(
                        selected_beam,
                        beams_per_satellite=beams_per_satellite,
                        decision_mask=decision_masks[uid],
                        post_mask=post_masks[uid],
                    ),
                    "handoverEvent": handover_event,
                    "beamCatalogOrder": BEAM_CATALOG_ORDER,
                    "visibilityMask": [bool(value) for value in post_masks[uid].tolist()],
                    "actionValidityMask": [bool(value) for value in post_masks[uid].tolist()],
                    "decisionVisibilityMask": [
                        bool(value) for value in decision_masks[uid].tolist()
                    ],
                    "decisionActionValidityMask": [
                        bool(value) for value in decision_masks[uid].tolist()
                    ],
                    "beamLoads": beam_loads,
                    "beamThroughputs": beam_throughputs,
                    "rewardVector": reward_vector,
                    "scalarReward": float(scalar_reward),
                    "satelliteStates": satellite_states,
                    "beamStates": beam_states,
                    "kpiOverlay": {
                        "userThroughputBps": float(reward.r1_throughput),
                        "selectedBeamLoad": (
                            int(beam_loads[selected_beam])
                            if 0 <= selected_beam < len(beam_loads)
                            else 0
                        ),
                        "selectedBeamThroughputBps": (
                            float(beam_throughputs[selected_beam])
                            if 0 <= selected_beam < len(beam_throughputs)
                            else 0.0
                        ),
                        "handoverOccurred": handover_event["eventId"] is not None,
                    },
                }
                diagnostics_payload = policy_diagnostics[uid]
                if diagnostics_payload is not None:
                    row[POLICY_DIAGNOSTICS_TIMELINE_FIELD] = _serialize_policy_diagnostics(
                        diagnostics_payload,
                        beams_per_satellite=beams_per_satellite,
                        decision_mask=decision_masks[uid],
                    )
                    diagnostics_row_count += 1
                else:
                    diagnostics_missing_row_count += 1
                handle.write(json.dumps(row) + "\n")
                row_count += 1

            if result.done:
                break

            encoded = trainer.encode_states(result.user_states)
            masks = result.action_masks

    config_resolved_path = _write_json(out_dir / "config-resolved.json", cfg)
    provenance_path = _write_json(
        out_dir / "provenance-map.json",
        build_provenance_map(cfg, metadata),
    )

    replay_summary = {
        "checkpointPath": str(checkpoint_path),
        "checkpointKind": replay_checkpoint_kind,
        "policyEpisode": int(checkpoint_payload.get("episode", -1)),
        "timelineSeed": int(replay_seed),
        "replaySeedSource": replay_seed_source,
        "slotIndexOffset": 1,
        "rowCount": int(row_count),
        "slotCount": int(slot_count),
        "handoverEventCount": int(handover_count),
        "rewardWeights": [float(value) for value in objective_weights],
        "diagnostics": asdict(diagnostics),
    }
    optional_policy_diagnostics = {
        "present": bool(diagnostics_row_count > 0),
        "timelineField": POLICY_DIAGNOSTICS_TIMELINE_FIELD,
        "diagnosticsVersion": POLICY_DIAGNOSTICS_VERSION,
        "requiredByBundleSchema": False,
        "producerOwned": True,
        "selectedActionSource": "selectedServing.beamIndex",
        "topCandidateLimit": POLICY_DIAGNOSTICS_TOP_CANDIDATES,
        "rowsWithDiagnostics": int(diagnostics_row_count),
        "rowsWithoutDiagnostics": int(diagnostics_missing_row_count),
        "note": (
            "Rows include policyDiagnostics only when the exporter can derive "
            "stable greedy masked candidate scores from the replay checkpoint "
            "without changing Phase 03A required field semantics."
        ),
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
        "replayTruthMode": "selected-checkpoint-greedy-replay",
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
            "firstIndex": 1,
            "note": (
                "Slot indices start at 1. Index N corresponds to the "
                "post-step state at time_s = N * slot_duration_s after "
                "the environment reset. Slot 0 is the reset state and "
                "has no exported row."
            ),
        },
        "beamCatalogOrder": BEAM_CATALOG_ORDER,
        "replaySummary": replay_summary,
        "optionalPolicyDiagnostics": optional_policy_diagnostics,
    }
    manifest_path = _write_json(out_dir / "manifest.json", manifest)

    return {
        "config_resolved": config_resolved_path,
        "provenance_map": provenance_path,
        "timeline_step_trace": timeline_path,
        "manifest": manifest_path,
        "replay_summary": replay_summary,
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "timeline_required_fields": list(_timeline_row_required_fields()),
    }


_REQUIRED_NON_EMPTY_MANIFEST_FIELDS = (
    "paperId",
    "runId",
    "bundleSchemaVersion",
    "producerVersion",
    "timelineFormatVersion",
    "replayTruthMode",
)


def _validate_objective_triplet_mapping(
    payload: Any,
    *,
    field_name: str,
) -> None:
    expected_keys = {"r1Throughput", "r2Handover", "r3LoadBalance"}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError(
            f"{field_name} must be an object with keys {sorted(expected_keys)}."
        )
    for key, value in payload.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{field_name}.{key} must be numeric.")


def _validate_policy_diagnostics_row(
    row: dict[str, Any],
    *,
    line_number: int,
) -> bool:
    payload = row.get(POLICY_DIAGNOSTICS_TIMELINE_FIELD)
    if payload is None:
        return False

    required_fields = (
        "diagnosticsVersion",
        "objectiveWeights",
        "selectedScalarizedQ",
        "runnerUpScalarizedQ",
        "scalarizedMarginToRunnerUp",
        "availableActionCount",
        "topCandidates",
    )
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise ValueError(
            f"Replay timeline row {line_number} policyDiagnostics is missing required fields: {missing}"
        )

    if not payload["diagnosticsVersion"]:
        raise ValueError(
            f"Replay timeline row {line_number} policyDiagnostics.diagnosticsVersion must be non-empty."
        )

    _validate_objective_triplet_mapping(
        payload["objectiveWeights"],
        field_name=(
            f"Replay timeline row {line_number} policyDiagnostics.objectiveWeights"
        ),
    )
    if not isinstance(payload["selectedScalarizedQ"], (int, float)) or isinstance(
        payload["selectedScalarizedQ"], bool
    ):
        raise ValueError(
            f"Replay timeline row {line_number} policyDiagnostics.selectedScalarizedQ must be numeric."
        )
    if payload["runnerUpScalarizedQ"] is not None and (
        not isinstance(payload["runnerUpScalarizedQ"], (int, float))
        or isinstance(payload["runnerUpScalarizedQ"], bool)
    ):
        raise ValueError(
            f"Replay timeline row {line_number} policyDiagnostics.runnerUpScalarizedQ must be numeric or null."
        )
    if payload["scalarizedMarginToRunnerUp"] is not None and (
        not isinstance(payload["scalarizedMarginToRunnerUp"], (int, float))
        or isinstance(payload["scalarizedMarginToRunnerUp"], bool)
    ):
        raise ValueError(
            f"Replay timeline row {line_number} policyDiagnostics.scalarizedMarginToRunnerUp must be numeric or null."
        )
    if not isinstance(payload["availableActionCount"], int) or isinstance(
        payload["availableActionCount"], bool
    ):
        raise ValueError(
            f"Replay timeline row {line_number} policyDiagnostics.availableActionCount must be an integer."
        )

    top_candidates = payload["topCandidates"]
    if not isinstance(top_candidates, list) or not top_candidates:
        raise ValueError(
            f"Replay timeline row {line_number} policyDiagnostics.topCandidates must be a non-empty list."
        )

    selected_serving = row.get("selectedServing") or {}
    selected_beam_index = int(selected_serving.get("beamIndex", -1))
    decision_mask = row.get("decisionActionValidityMask")
    if not isinstance(decision_mask, list):
        decision_mask = row.get("actionValidityMask")
    available_action_count = None
    if isinstance(decision_mask, list):
        available_action_count = sum(bool(value) for value in decision_mask)
        if available_action_count != int(payload["availableActionCount"]):
            raise ValueError(
                f"Replay timeline row {line_number} policyDiagnostics.availableActionCount does not match the decision mask."
            )

    previous_scalarized = None
    previous_beam_index = None
    for candidate_index, candidate in enumerate(top_candidates, start=1):
        required_candidate_fields = (
            "beamId",
            "beamIndex",
            "satId",
            "satIndex",
            "localBeamIndex",
            "validUnderDecisionMask",
            "objectiveQ",
            "scalarizedQ",
        )
        missing_candidate = [
            field for field in required_candidate_fields if field not in candidate
        ]
        if missing_candidate:
            raise ValueError(
                "Replay timeline row "
                f"{line_number} policyDiagnostics.topCandidates[{candidate_index}] "
                f"is missing required fields: {missing_candidate}"
            )
        if not bool(candidate["validUnderDecisionMask"]):
            raise ValueError(
                "Replay timeline row "
                f"{line_number} policyDiagnostics.topCandidates[{candidate_index}] "
                "must remain valid under the decision mask."
            )
        beam_index = int(candidate["beamIndex"])
        if isinstance(decision_mask, list):
            if beam_index < 0 or beam_index >= len(decision_mask) or not bool(
                decision_mask[beam_index]
            ):
                raise ValueError(
                    "Replay timeline row "
                    f"{line_number} policyDiagnostics.topCandidates[{candidate_index}] "
                    "does not respect decisionActionValidityMask."
                )
        _validate_objective_triplet_mapping(
            candidate["objectiveQ"],
            field_name=(
                "Replay timeline row "
                f"{line_number} policyDiagnostics.topCandidates[{candidate_index}].objectiveQ"
            ),
        )
        scalarized_q = candidate["scalarizedQ"]
        if not isinstance(scalarized_q, (int, float)) or isinstance(
            scalarized_q, bool
        ):
            raise ValueError(
                "Replay timeline row "
                f"{line_number} policyDiagnostics.topCandidates[{candidate_index}].scalarizedQ "
                "must be numeric."
            )
        scalarized_q = float(scalarized_q)
        if previous_scalarized is not None:
            if scalarized_q > previous_scalarized + 1e-9:
                raise ValueError(
                    f"Replay timeline row {line_number} policyDiagnostics.topCandidates must be sorted by descending scalarizedQ."
                )
            if (
                abs(scalarized_q - previous_scalarized) <= 1e-9
                and previous_beam_index is not None
                and beam_index < previous_beam_index
            ):
                raise ValueError(
                    f"Replay timeline row {line_number} policyDiagnostics.topCandidates must use ascending beamIndex tie-breaks."
                )
        previous_scalarized = scalarized_q
        previous_beam_index = beam_index

    selected_candidate = top_candidates[0]
    if int(selected_candidate["beamIndex"]) != selected_beam_index:
        raise ValueError(
            f"Replay timeline row {line_number} policyDiagnostics must align topCandidates[0] with selectedServing."
        )
    if abs(float(selected_candidate["scalarizedQ"]) - float(payload["selectedScalarizedQ"])) > 1e-9:
        raise ValueError(
            f"Replay timeline row {line_number} policyDiagnostics.selectedScalarizedQ must match topCandidates[0].scalarizedQ."
        )

    has_runner_up = len(top_candidates) > 1
    if has_runner_up:
        runner_up_scalarized = payload["runnerUpScalarizedQ"]
        margin = payload["scalarizedMarginToRunnerUp"]
        if runner_up_scalarized is None or margin is None:
            raise ValueError(
                f"Replay timeline row {line_number} policyDiagnostics must include runner-up score and margin when more than one action is available."
            )
        if abs(float(top_candidates[1]["scalarizedQ"]) - float(runner_up_scalarized)) > 1e-9:
            raise ValueError(
                f"Replay timeline row {line_number} policyDiagnostics.runnerUpScalarizedQ must match topCandidates[1].scalarizedQ."
            )
        expected_margin = float(payload["selectedScalarizedQ"]) - float(runner_up_scalarized)
        if abs(float(margin) - expected_margin) > 1e-9:
            raise ValueError(
                f"Replay timeline row {line_number} policyDiagnostics.scalarizedMarginToRunnerUp is inconsistent with selected/runner-up scores."
            )
    else:
        if payload["runnerUpScalarizedQ"] is not None or payload["scalarizedMarginToRunnerUp"] is not None:
            raise ValueError(
                f"Replay timeline row {line_number} policyDiagnostics must keep runner-up score and margin null when only one candidate is available."
            )

    return True


def _validate_optional_policy_diagnostics_manifest(
    manifest: dict[str, Any],
    *,
    timeline_row_count: int,
    rows_with_diagnostics: int,
) -> None:
    payload = manifest.get("optionalPolicyDiagnostics")
    if payload is None:
        if rows_with_diagnostics > 0:
            raise ValueError(
                "Replay bundle timeline exports policyDiagnostics rows but manifest.optionalPolicyDiagnostics is missing."
            )
        return

    required_fields = (
        "present",
        "timelineField",
        "diagnosticsVersion",
        "requiredByBundleSchema",
        "producerOwned",
        "selectedActionSource",
        "topCandidateLimit",
        "rowsWithDiagnostics",
        "rowsWithoutDiagnostics",
        "note",
    )
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise ValueError(
            "Replay bundle manifest.optionalPolicyDiagnostics is missing required fields: "
            f"{missing}"
        )
    if payload["timelineField"] != POLICY_DIAGNOSTICS_TIMELINE_FIELD:
        raise ValueError(
            "Replay bundle manifest.optionalPolicyDiagnostics.timelineField must be "
            f"{POLICY_DIAGNOSTICS_TIMELINE_FIELD!r}."
        )
    if not payload["diagnosticsVersion"]:
        raise ValueError(
            "Replay bundle manifest.optionalPolicyDiagnostics.diagnosticsVersion must be non-empty."
        )
    if int(payload["rowsWithDiagnostics"]) != rows_with_diagnostics:
        raise ValueError(
            "Replay bundle manifest.optionalPolicyDiagnostics.rowsWithDiagnostics does not match the timeline."
        )
    rows_without_diagnostics = timeline_row_count - rows_with_diagnostics
    if int(payload["rowsWithoutDiagnostics"]) != rows_without_diagnostics:
        raise ValueError(
            "Replay bundle manifest.optionalPolicyDiagnostics.rowsWithoutDiagnostics does not match the timeline."
        )
    if bool(payload["present"]) != bool(rows_with_diagnostics > 0):
        raise ValueError(
            "Replay bundle manifest.optionalPolicyDiagnostics.present does not match timeline coverage."
        )


def validate_replay_bundle(output_dir: str | Path) -> None:
    """Validate the minimum replay-complete Phase 03A producer bundle surface.

    Enforces:

    1. all required top-level paths exist (``manifest.json``,
       ``config-resolved.json``, ``provenance-map.json``,
       ``assumptions.json``, ``training/{episode_metrics,loss_curves}.csv``,
       ``evaluation/summary.json``, ``evaluation/sweeps/``,
       ``timeline/step-trace.jsonl``)
    2. all required ``manifest.json`` fields are present
       (Phase 03A §7.2) and the fixed identity fields are non-empty
    3. every ``timeline/step-trace.jsonl`` row (not just the first) is
       well-formed JSON and carries every required field in
       ``_timeline_row_required_fields()`` (Phase 03A §7.3)

    The ``evaluation/sweeps/`` directory is required to exist, but its
    contents are intentionally not constrained by Slice A so that small
    sample bundles can ship without fabricated sweep rows. Consumers that
    need sweep content must declare that separately.
    """
    out_dir = Path(output_dir)
    required_paths = (
        out_dir / "manifest.json",
        out_dir / "config-resolved.json",
        out_dir / "provenance-map.json",
        out_dir / "assumptions.json",
        out_dir / "training" / "episode_metrics.csv",
        out_dir / "training" / "loss_curves.csv",
        out_dir / "evaluation" / "summary.json",
        out_dir / "evaluation" / "sweeps",
        out_dir / "timeline" / "step-trace.jsonl",
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise ValueError(f"Replay bundle is missing required paths: {missing}")

    manifest = json.loads((out_dir / "manifest.json").read_text())
    required_manifest_fields = (
        "paperId",
        "runId",
        "bundleSchemaVersion",
        "producerVersion",
        "exportedAt",
        "sourceArtifactDir",
        "checkpointRule",
        "replayTruthMode",
        "timelineFormatVersion",
        "coordinateFrame",
    )
    missing_manifest_fields = [
        field for field in required_manifest_fields if field not in manifest
    ]
    if missing_manifest_fields:
        raise ValueError(
            "Replay bundle manifest is missing required fields: "
            f"{missing_manifest_fields}"
        )
    empty_identity_fields = [
        field
        for field in _REQUIRED_NON_EMPTY_MANIFEST_FIELDS
        if manifest.get(field) in (None, "")
    ]
    if empty_identity_fields:
        raise ValueError(
            "Replay bundle manifest has empty identity fields: "
            f"{empty_identity_fields}"
        )

    required_row_fields = _timeline_row_required_fields()
    timeline_path = out_dir / "timeline" / "step-trace.jsonl"
    row_count = 0
    rows_with_policy_diagnostics = 0
    with timeline_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Replay timeline row {line_number} is not valid JSON: {exc}"
                ) from exc
            missing_row_fields = [
                field for field in required_row_fields if field not in row
            ]
            if missing_row_fields:
                raise ValueError(
                    f"Replay timeline row {line_number} is missing required fields: "
                    f"{missing_row_fields}"
                )
            if _validate_policy_diagnostics_row(row, line_number=line_number):
                rows_with_policy_diagnostics += 1
            row_count += 1
    if row_count == 0:
        raise ValueError("Replay bundle timeline is empty.")
    _validate_optional_policy_diagnostics_manifest(
        manifest,
        timeline_row_count=row_count,
        rows_with_diagnostics=rows_with_policy_diagnostics,
    )


def trim_replay_bundle_for_sample(
    source_dir: str | Path,
    target_dir: str | Path,
    *,
    max_users: int = 1,
    max_slots: int | None = None,
    sample_note: str | None = None,
) -> dict[str, Any]:
    """Produce a small reproducible sample bundle from a full replay bundle.

    The trimmed copy keeps every Phase 03A required surface
    (``manifest.json``, ``config-resolved.json``, ``assumptions.json``,
    ``provenance-map.json``, ``training/``, ``evaluation/``, ``figures/``,
    ``timeline/step-trace.jsonl``) but reduces ``timeline/step-trace.jsonl``
    to the first ``max_users`` users for the first ``max_slots`` slots.

    The function is intentionally a strict subset operation: it does not
    rewrite reward semantics, action masks, or geometry. It only filters
    rows and updates the manifest's ``replaySummary`` aggregate counts so
    consumers can reason about the trimmed extent without re-deriving it.

    Returns a small report with the trimmed counts and the new paths.
    """
    src = Path(source_dir)
    dst = Path(target_dir)
    if not src.exists():
        raise FileNotFoundError(f"Source bundle does not exist: {src}")

    validate_replay_bundle(src)

    if dst.is_symlink():
        raise ValueError(
            f"Refusing to overwrite symlink target for safety: {dst}. "
            "Point --target-dir at a plain directory path."
        )
    if dst.exists():
        if not dst.is_dir():
            raise ValueError(
                f"Target path exists and is not a directory: {dst}"
            )
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    # Copy every non-timeline surface verbatim.
    for entry in (
        "manifest.json",
        "config-resolved.json",
        "assumptions.json",
        "provenance-map.json",
        "evaluation",
        "training",
        "figures",
    ):
        source_entry = src / entry
        if not source_entry.exists():
            continue
        target_entry = dst / entry
        if source_entry.is_dir():
            shutil.copytree(source_entry, target_entry)
        else:
            shutil.copy2(source_entry, target_entry)

    # Filter the timeline.
    timeline_src = src / "timeline" / "step-trace.jsonl"
    timeline_dst = dst / "timeline" / "step-trace.jsonl"
    timeline_dst.parent.mkdir(parents=True, exist_ok=True)

    kept_rows = 0
    kept_handovers = 0
    kept_policy_diagnostics_rows = 0
    kept_slot_indices: set[int] = set()
    kept_user_indices: set[int] = set()
    # slot index numbering can start at 0 or 1 depending on the env, so we
    # filter by a stable "first N distinct slots seen" set rather than by
    # raw index comparison.
    accepted_slot_set: set[int] = set()

    with timeline_src.open() as handle, timeline_dst.open("w") as out_handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            user_index = int(row.get("userIndex", -1))
            slot_index = int(row.get("slotIndex", -1))
            if user_index < 0 or user_index >= max_users:
                continue
            if max_slots is not None and slot_index not in accepted_slot_set:
                if len(accepted_slot_set) >= max_slots:
                    continue
                accepted_slot_set.add(slot_index)
            out_handle.write(json.dumps(row) + "\n")
            kept_rows += 1
            kept_slot_indices.add(slot_index)
            kept_user_indices.add(user_index)
            if POLICY_DIAGNOSTICS_TIMELINE_FIELD in row:
                kept_policy_diagnostics_rows += 1
            handover_event = row.get("handoverEvent") or {}
            if handover_event.get("eventId"):
                kept_handovers += 1

    if kept_rows == 0:
        raise ValueError(
            "Trimmed replay bundle would be empty. "
            "Check max_users/max_slots against the source timeline."
        )

    manifest_path = dst / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    replay_summary = manifest.get("replaySummary")
    if not isinstance(replay_summary, dict):
        replay_summary = {}
        manifest["replaySummary"] = replay_summary

    full_row_count = int(replay_summary.get("rowCount", kept_rows))
    full_slot_count = int(replay_summary.get("slotCount", len(kept_slot_indices)))
    full_handover_count = int(replay_summary.get("handoverEventCount", kept_handovers))

    sample_note_text = sample_note or (
        "Trimmed for fixture/sample purposes. Timeline reduced to the first "
        f"{len(kept_user_indices)} user(s) and first {len(kept_slot_indices)} "
        "slot(s) of the source bundle. All other surfaces are byte-equal to "
        "the source."
    )

    replay_summary["rowCount"] = kept_rows
    replay_summary["slotCount"] = len(kept_slot_indices)
    replay_summary["handoverEventCount"] = kept_handovers
    replay_summary["sampleSubset"] = {
        "maxUsers": int(max_users),
        "maxSlots": None if max_slots is None else int(max_slots),
        "userIndices": sorted(kept_user_indices),
        "slotIndices": sorted(kept_slot_indices),
        "sourceFullRowCount": full_row_count,
        "sourceFullSlotCount": full_slot_count,
        "sourceFullHandoverEventCount": full_handover_count,
    }
    optional_policy_diagnostics = manifest.get("optionalPolicyDiagnostics")
    if isinstance(optional_policy_diagnostics, dict):
        optional_policy_diagnostics["present"] = bool(kept_policy_diagnostics_rows > 0)
        optional_policy_diagnostics["rowsWithDiagnostics"] = int(
            kept_policy_diagnostics_rows
        )
        optional_policy_diagnostics["rowsWithoutDiagnostics"] = int(
            kept_rows - kept_policy_diagnostics_rows
        )
    manifest["sampleNote"] = sample_note_text
    manifest_path.write_text(json.dumps(manifest, indent=2))

    validate_replay_bundle(dst)

    return {
        "targetDir": dst,
        "rowCount": kept_rows,
        "slotCount": len(kept_slot_indices),
        "userCount": len(kept_user_indices),
        "handoverEventCount": kept_handovers,
        "manifestPath": manifest_path,
        "timelinePath": timeline_dst,
        "sampleNote": sample_note_text,
    }
