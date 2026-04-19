"""Runtime-to-bundle serializers used by the Phase 04C bundle seam."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import numpy as np

from ..env.step import local_tangent_offset_km
from ..runtime.objective_math import scalarize_objectives
from .models import ReplaySummary
from .schema import (
    BEAM_CATALOG_ORDER,
    POLICY_DIAGNOSTICS_NOTE,
    POLICY_DIAGNOSTICS_SELECTED_ACTION_SOURCE,
    POLICY_DIAGNOSTICS_TIMELINE_FIELD,
    POLICY_DIAGNOSTICS_TOP_CANDIDATES,
    POLICY_DIAGNOSTICS_VERSION,
    SLOT_INDEX_OFFSET,
)


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


def serialize_policy_diagnostics(
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
        if (
            not bool(candidate_payload.get("validUnderDecisionMask", False))
            or not candidate_valid
        ):
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


def serialize_satellite_states(env) -> list[dict[str, Any]]:
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


def serialize_beam_states(env) -> list[dict[str, Any]]:
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


def serialize_timeline_row(
    *,
    result,
    user_index: int,
    previous_time_s: float,
    previous_positions: list[tuple[float, float]],
    post_positions: list[tuple[float, float]],
    previous_assignments: list[int],
    actions: list[int] | np.ndarray,
    decision_masks: list[np.ndarray],
    post_masks: list[np.ndarray],
    beam_loads: list[int],
    beam_throughputs: list[float],
    satellite_states: list[dict[str, Any]],
    beam_states: list[dict[str, Any]],
    objective_weights: tuple[float, ...],
    beams_per_satellite: int,
    ground_lat_deg: float,
    ground_lon_deg: float,
    policy_diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    reward = result.rewards[user_index]
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
    previous_beam = int(previous_assignments[user_index])
    selected_beam = int(actions[user_index])
    handover_event = _handover_event(
        previous_beam,
        selected_beam,
        beams_per_satellite=beams_per_satellite,
        slot_index=result.step_index,
        user_index=user_index,
    )
    row = {
        "slotIndex": int(result.step_index),
        "timeSec": float(result.time_s),
        "decisionTimeSec": previous_time_s,
        "userId": _user_id(user_index),
        "userIndex": user_index,
        "userPosition": _serialize_user_position(
            post_positions[user_index][0],
            post_positions[user_index][1],
            ground_lat_deg=ground_lat_deg,
            ground_lon_deg=ground_lon_deg,
        ),
        "decisionUserPosition": _serialize_user_position(
            previous_positions[user_index][0],
            previous_positions[user_index][1],
            ground_lat_deg=ground_lat_deg,
            ground_lon_deg=ground_lon_deg,
        ),
        "previousServing": _serving_state(
            previous_beam,
            beams_per_satellite=beams_per_satellite,
            decision_mask=decision_masks[user_index],
            post_mask=post_masks[user_index],
        ),
        "selectedServing": _serving_state(
            selected_beam,
            beams_per_satellite=beams_per_satellite,
            decision_mask=decision_masks[user_index],
            post_mask=post_masks[user_index],
        ),
        "handoverEvent": handover_event,
        "beamCatalogOrder": BEAM_CATALOG_ORDER,
        "visibilityMask": [bool(value) for value in post_masks[user_index].tolist()],
        "actionValidityMask": [bool(value) for value in post_masks[user_index].tolist()],
        "decisionVisibilityMask": [
            bool(value) for value in decision_masks[user_index].tolist()
        ],
        "decisionActionValidityMask": [
            bool(value) for value in decision_masks[user_index].tolist()
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
    if policy_diagnostics is not None:
        row[POLICY_DIAGNOSTICS_TIMELINE_FIELD] = serialize_policy_diagnostics(
            policy_diagnostics,
            beams_per_satellite=beams_per_satellite,
            decision_mask=decision_masks[user_index],
        )
    return row


def build_replay_summary(
    *,
    checkpoint_path: str,
    checkpoint_kind: str,
    policy_episode: int,
    timeline_seed: int,
    replay_seed_source: str,
    row_count: int,
    slot_count: int,
    handover_event_count: int,
    reward_weights: tuple[float, ...],
    diagnostics,
) -> ReplaySummary:
    return ReplaySummary(
        checkpoint_path=str(checkpoint_path),
        checkpoint_kind=str(checkpoint_kind),
        policy_episode=int(policy_episode),
        timeline_seed=int(timeline_seed),
        replay_seed_source=str(replay_seed_source),
        slot_index_offset=SLOT_INDEX_OFFSET,
        row_count=int(row_count),
        slot_count=int(slot_count),
        handover_event_count=int(handover_event_count),
        reward_weights=tuple(float(value) for value in reward_weights),
        diagnostics=asdict(diagnostics),
    )


def build_optional_policy_diagnostics_manifest(
    *,
    diagnostics_row_count: int,
    diagnostics_missing_row_count: int,
) -> dict[str, Any]:
    return {
        "present": bool(diagnostics_row_count > 0),
        "timelineField": POLICY_DIAGNOSTICS_TIMELINE_FIELD,
        "diagnosticsVersion": POLICY_DIAGNOSTICS_VERSION,
        "requiredByBundleSchema": False,
        "producerOwned": True,
        "selectedActionSource": POLICY_DIAGNOSTICS_SELECTED_ACTION_SOURCE,
        "topCandidateLimit": POLICY_DIAGNOSTICS_TOP_CANDIDATES,
        "rowsWithDiagnostics": int(diagnostics_row_count),
        "rowsWithoutDiagnostics": int(diagnostics_missing_row_count),
        "note": POLICY_DIAGNOSTICS_NOTE,
    }
