"""RA-EE-06 association counterfactual / oracle design gate.

This module evaluates centralized active-set / served-set association
proposals before any learned hierarchical RL is allowed. It reuses the
RA-EE-04/05 safe-greedy power allocator as a post-association optimizer and
keeps the constrained association + power oracle diagnostic-only.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config_loader import build_environment, get_seeds, load_training_yaml
from ..env.step import HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK
from ._common import write_json
from .ra_ee_02_oracle_power_allocation import (
    _AuditSettings,
    _StepSnapshot,
    _build_unit_power_snapshots,
    _evaluate_power_vector,
    _format_vector,
    _power_vector_for_candidate,
    _select_oracle_step,
    _summarize_all,
)
from .ra_ee_05_fixed_association_robustness import (
    CALIBRATION_BUCKET,
    DEFAULT_CALIBRATION_SEEDS,
    DEFAULT_HELD_OUT_SEEDS,
    HELD_OUT_BUCKET,
    _BucketSpec,
    _RAEE05Settings,
    _safe_greedy_power_vector,
)


DEFAULT_CONFIG = "configs/ra-ee-06-association-counterfactual-oracle.resolved.yaml"
DEFAULT_OUTPUT_DIR = "artifacts/ra-ee-06-association-counterfactual-oracle"

RA_EE_06_METHOD_LABEL = "RA-EE hierarchical association + power counterfactual"
RA_EE_06_MATCHED_CONTROL = "matched-fixed-association+safe-greedy-power-allocator"
RA_EE_06_CANDIDATE = "association-proposal+safe-greedy-power-allocator"
RA_EE_06_GREEDY_DIAGNOSTIC = "per-user-greedy-best-beam+safe-greedy-power-allocator"
RA_EE_06_ORACLE = "association-oracle+constrained-power-upper-bound"

FIXED_HOLD_CURRENT = "fixed-hold-current"
ACTIVE_SET_LOAD_SPREAD = "active-set-load-spread"
ACTIVE_SET_QUALITY_SPREAD = "active-set-quality-spread"
ACTIVE_SET_STICKY_SPREAD = "active-set-sticky-spread"
PER_USER_GREEDY_BEST_BEAM = "per-user-greedy-best-beam"

ACTIVE_SET_POLICIES = (
    ACTIVE_SET_LOAD_SPREAD,
    ACTIVE_SET_QUALITY_SPREAD,
    ACTIVE_SET_STICKY_SPREAD,
)
SUPPORTED_ASSOCIATION_POLICIES = (
    FIXED_HOLD_CURRENT,
    *ACTIVE_SET_POLICIES,
    PER_USER_GREEDY_BEST_BEAM,
)

_POLICY_SEED_OFFSETS = {
    FIXED_HOLD_CURRENT: 1009,
    ACTIVE_SET_LOAD_SPREAD: 11003,
    ACTIVE_SET_QUALITY_SPREAD: 12007,
    ACTIVE_SET_STICKY_SPREAD: 13001,
    PER_USER_GREEDY_BEST_BEAM: 14009,
}


@dataclass(frozen=True)
class _RAEE06Settings:
    method_label: str
    implementation_sublabel: str
    audit: _AuditSettings
    bucket_specs: tuple[_BucketSpec, ...]
    matched_control_association_policy: str
    candidate_association_policies: tuple[str, ...]
    diagnostic_association_policies: tuple[str, ...]
    oracle_association_policies: tuple[str, ...]
    min_active_beams: int
    max_active_beams: int
    target_users_per_active_beam: int
    load_cap_overflow_users: int
    candidate_max_demoted_beams: int
    candidate_step_p05_guardrail_margin: float
    max_one_active_beam_ratio_for_acceptance: float


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _categorical_distribution(values: list[str]) -> dict[str, Any]:
    counts = Counter(str(value) for value in values)
    return {
        "count": int(sum(counts.values())),
        "distinct": sorted(counts),
        "distinct_count": int(len(counts)),
        "histogram": {
            key: int(value)
            for key, value in sorted(counts.items(), key=lambda item: item[0])
        },
    }


def _ra_ee_06_value(cfg: dict[str, Any]) -> dict[str, Any]:
    value = (
        cfg.get("resolved_assumptions", {})
        .get("ra_ee_06_association_counterfactual_oracle", {})
        .get("value", {})
    )
    return value if isinstance(value, dict) else {}


def _power_surface_value(cfg: dict[str, Any]) -> dict[str, Any]:
    value = (
        cfg.get("resolved_assumptions", {})
        .get("hobs_power_surface", {})
        .get("value", {})
    )
    return value if isinstance(value, dict) else {}


def _tuple_ints(raw: Any, fallback: tuple[int, ...]) -> tuple[int, ...]:
    values = raw if raw is not None else fallback
    return tuple(int(value) for value in values)


def _tuple_strings(raw: Any, fallback: tuple[str, ...]) -> tuple[str, ...]:
    values = raw if raw is not None else fallback
    return tuple(str(value) for value in values)


def _bucket_specs_from_config(
    gate: dict[str, Any],
    seeds: dict[str, Any],
) -> tuple[_BucketSpec, ...]:
    buckets = gate.get("evaluation_buckets", {})
    if not isinstance(buckets, dict):
        buckets = {}

    calibration = buckets.get(CALIBRATION_BUCKET, {})
    if not isinstance(calibration, dict):
        calibration = {}
    held_out = buckets.get(HELD_OUT_BUCKET, buckets.get("held_out", {}))
    if not isinstance(held_out, dict):
        held_out = {}

    return (
        _BucketSpec(
            name=CALIBRATION_BUCKET,
            trajectory_families=("association-counterfactual",),
            evaluation_seed_set=_tuple_ints(
                calibration.get("evaluation_seed_set"),
                _tuple_ints(seeds.get("evaluation_seed_set"), DEFAULT_CALIBRATION_SEEDS),
            ),
        ),
        _BucketSpec(
            name=HELD_OUT_BUCKET,
            trajectory_families=("association-counterfactual-heldout",),
            evaluation_seed_set=_tuple_ints(
                held_out.get("evaluation_seed_set"),
                DEFAULT_HELD_OUT_SEEDS,
            ),
        ),
    )


def _validate_policies(name: str, policies: tuple[str, ...]) -> None:
    unsupported = sorted(set(policies) - set(SUPPORTED_ASSOCIATION_POLICIES))
    if unsupported:
        raise ValueError(f"Unsupported RA-EE-06 {name}: {unsupported!r}")


def _settings_from_config(cfg: dict[str, Any]) -> _RAEE06Settings:
    gate = _ra_ee_06_value(cfg)
    power = _power_surface_value(cfg)
    seeds = get_seeds(cfg)

    levels_raw = gate.get(
        "codebook_levels_w",
        power.get("power_codebook_levels_w", [0.5, 0.75, 1.0, 1.5, 2.0]),
    )
    levels = tuple(float(level) for level in levels_raw)
    if not levels:
        raise ValueError("RA-EE-06 requires at least one power codebook level.")
    if tuple(sorted(levels)) != levels:
        raise ValueError(f"RA-EE-06 codebook levels must be sorted, got {levels!r}.")

    matched_control = str(
        gate.get("matched_control_association_policy", FIXED_HOLD_CURRENT)
    )
    candidates = _tuple_strings(
        gate.get("candidate_association_policies"),
        ACTIVE_SET_POLICIES,
    )
    diagnostics = _tuple_strings(
        gate.get("diagnostic_association_policies"),
        (PER_USER_GREEDY_BEST_BEAM,),
    )
    oracle = _tuple_strings(gate.get("oracle_association_policies"), candidates)
    _validate_policies("matched control policy", (matched_control,))
    _validate_policies("candidate policies", candidates)
    _validate_policies("diagnostic policies", diagnostics)
    _validate_policies("oracle policies", oracle)
    if matched_control in candidates:
        raise ValueError("RA-EE-06 matched control must not also be a candidate.")

    min_active = int(gate.get("min_active_beams", 2))
    max_active = int(gate.get("max_active_beams", 8))
    if min_active < 1 or max_active < min_active:
        raise ValueError(
            f"Invalid RA-EE-06 active-beam bounds: min={min_active}, max={max_active}."
        )

    audit = _AuditSettings(
        method_label=str(gate.get("method_label", RA_EE_06_METHOD_LABEL)),
        codebook_levels_w=levels,
        fixed_control_power_w=float(gate.get("fixed_control_power_w", 1.0)),
        total_power_budget_w=float(
            gate.get("total_active_power_budget_w", power.get("total_power_budget_w", 8.0))
        ),
        per_beam_max_power_w=float(
            gate.get("per_beam_max_power_w", power.get("max_power_w", 2.0))
        ),
        active_base_power_w=float(
            gate.get("active_base_power_w", power.get("active_base_power_w", 0.25))
        ),
        load_scale_power_w=float(
            gate.get("load_scale_power_w", power.get("load_scale_power_w", 0.35))
        ),
        load_exponent=float(
            gate.get("load_exponent", power.get("load_exponent", 0.5))
        ),
        p05_min_ratio_vs_control=float(
            gate.get("p05_throughput_min_ratio_vs_control", 0.95)
        ),
        served_ratio_min_delta_vs_control=float(
            gate.get("served_ratio_min_delta_vs_control", 0.0)
        ),
        outage_ratio_max_delta_vs_control=float(
            gate.get("outage_ratio_max_delta_vs_control", 0.0)
        ),
        oracle_max_demoted_beams=int(gate.get("oracle_max_demoted_beams", 3)),
    )
    return _RAEE06Settings(
        method_label=str(gate.get("method_label", RA_EE_06_METHOD_LABEL)),
        implementation_sublabel=str(
            gate.get(
                "implementation_sublabel",
                "RA-EE-06 association counterfactual / oracle design gate",
            )
        ),
        audit=audit,
        bucket_specs=_bucket_specs_from_config(gate, seeds),
        matched_control_association_policy=matched_control,
        candidate_association_policies=candidates,
        diagnostic_association_policies=diagnostics,
        oracle_association_policies=oracle,
        min_active_beams=min_active,
        max_active_beams=max_active,
        target_users_per_active_beam=int(gate.get("target_users_per_active_beam", 16)),
        load_cap_overflow_users=int(gate.get("load_cap_overflow_users", 2)),
        candidate_max_demoted_beams=int(
            gate.get("candidate_max_demoted_beams", audit.oracle_max_demoted_beams)
        ),
        candidate_step_p05_guardrail_margin=float(
            gate.get("candidate_step_p05_guardrail_margin", 0.005)
        ),
        max_one_active_beam_ratio_for_acceptance=float(
            gate.get("max_one_active_beam_ratio_for_acceptance", 0.25)
        ),
    )


def _ra_ee_05_settings(settings: _RAEE06Settings) -> _RAEE05Settings:
    return _RAEE05Settings(
        method_label=settings.method_label,
        implementation_sublabel=settings.implementation_sublabel,
        audit=settings.audit,
        bucket_specs=(),
        candidate_max_demoted_beams=settings.candidate_max_demoted_beams,
        candidate_step_p05_guardrail_margin=(
            settings.candidate_step_p05_guardrail_margin
        ),
    )


def _valid_indices(mask_obj: Any) -> np.ndarray:
    return np.flatnonzero(mask_obj.mask)


def _beam_scores(
    *,
    user_states: list[Any],
    masks: list[Any],
    uncovered_users: set[int] | None,
) -> dict[int, dict[str, float]]:
    scores: dict[int, dict[str, float]] = defaultdict(
        lambda: {"coverage": 0.0, "quality": 0.0}
    )
    for uid, (state, mask_obj) in enumerate(zip(user_states, masks)):
        if uncovered_users is not None and uid not in uncovered_users:
            continue
        for beam_idx in _valid_indices(mask_obj).tolist():
            idx = int(beam_idx)
            scores[idx]["coverage"] += 1.0
            scores[idx]["quality"] += float(state.channel_quality[idx])
    return scores


def _target_active_count(
    *,
    num_users: int,
    valid_beam_count: int,
    settings: _RAEE06Settings,
) -> int:
    if valid_beam_count <= 0:
        return 0
    target_from_load = int(math.ceil(num_users / settings.target_users_per_active_beam))
    target = max(settings.min_active_beams, target_from_load)
    target = min(settings.max_active_beams, target, valid_beam_count)
    return max(1, target)


def _select_active_set(
    policy: str,
    *,
    user_states: list[Any],
    masks: list[Any],
    current_assignments: np.ndarray,
    settings: _RAEE06Settings,
) -> set[int]:
    all_valid = sorted(
        {
            int(beam_idx)
            for mask_obj in masks
            for beam_idx in _valid_indices(mask_obj).tolist()
        }
    )
    target = _target_active_count(
        num_users=len(masks),
        valid_beam_count=len(all_valid),
        settings=settings,
    )
    if target == 0:
        return set()

    current_counts = Counter(
        int(beam_idx)
        for beam_idx in current_assignments.tolist()
        if int(beam_idx) in all_valid
    )
    active: set[int] = set()
    if policy == ACTIVE_SET_STICKY_SPREAD:
        for beam_idx, _count in sorted(
            current_counts.items(),
            key=lambda item: (-int(item[1]), int(item[0])),
        ):
            active.add(int(beam_idx))
            if len(active) >= target:
                return active

    uncovered_users = set(range(len(masks)))
    while len(active) < target:
        scores = _beam_scores(
            user_states=user_states,
            masks=masks,
            uncovered_users=uncovered_users,
        )
        choices = [beam_idx for beam_idx in all_valid if beam_idx not in active]
        if not choices:
            break

        def key_load(beam_idx: int) -> tuple[float, float, int]:
            score = scores[beam_idx]
            return (score["coverage"], -float(current_counts[beam_idx]), -beam_idx)

        def key_quality(beam_idx: int) -> tuple[float, float, int]:
            score = scores[beam_idx]
            return (score["quality"], score["coverage"], -beam_idx)

        if policy == ACTIVE_SET_QUALITY_SPREAD:
            selected = max(choices, key=key_quality)
        else:
            selected = max(choices, key=key_load)
        active.add(int(selected))
        for uid, mask_obj in enumerate(masks):
            if uid in uncovered_users and bool(mask_obj.mask[selected]):
                uncovered_users.remove(uid)
        if not uncovered_users and len(active) >= settings.min_active_beams:
            break

    return active


def _choose_best_valid_by_quality(state: Any, valid: np.ndarray) -> int:
    return int(max(valid.tolist(), key=lambda idx: (float(state.channel_quality[int(idx)]), -int(idx))))


def _select_actions_for_association_policy(
    policy: str,
    *,
    user_states: list[Any],
    masks: list[Any],
    current_assignments: np.ndarray,
    settings: _RAEE06Settings,
) -> np.ndarray:
    """Select centralized counterfactual association actions.

    Active-set policies first choose a bounded set of active beams and then
    assign users into that served set with explicit load caps. This is the
    RA-EE-06 anti-collapse contract; the per-user greedy selector is kept only
    as a diagnostic comparator.
    """
    actions = np.zeros(len(masks), dtype=np.int32)
    if policy == FIXED_HOLD_CURRENT:
        for uid, mask_obj in enumerate(masks):
            valid = _valid_indices(mask_obj)
            current = int(current_assignments[uid])
            if 0 <= current < mask_obj.mask.size and bool(mask_obj.mask[current]):
                actions[uid] = current
            elif valid.size:
                actions[uid] = int(valid[0])
        return actions

    if policy == PER_USER_GREEDY_BEST_BEAM:
        for uid, (state, mask_obj) in enumerate(zip(user_states, masks)):
            valid = _valid_indices(mask_obj)
            actions[uid] = _choose_best_valid_by_quality(state, valid) if valid.size else 0
        return actions

    if policy not in ACTIVE_SET_POLICIES:
        raise ValueError(f"Unsupported RA-EE-06 association policy {policy!r}.")

    active_set = _select_active_set(
        policy,
        user_states=user_states,
        masks=masks,
        current_assignments=current_assignments,
        settings=settings,
    )
    assigned_counts = Counter()
    load_cap = int(
        math.ceil(len(masks) / max(len(active_set), 1)) + settings.load_cap_overflow_users
    )

    for uid, (state, mask_obj) in enumerate(zip(user_states, masks)):
        valid = _valid_indices(mask_obj)
        if valid.size == 0:
            actions[uid] = 0
            continue
        candidates = [int(idx) for idx in valid.tolist() if int(idx) in active_set]
        if not candidates:
            selected = _choose_best_valid_by_quality(state, valid)
            if len(active_set) < settings.max_active_beams:
                active_set.add(selected)
                candidates = [selected]
            else:
                candidates = [selected]

        current = int(current_assignments[uid])
        if (
            policy == ACTIVE_SET_STICKY_SPREAD
            and current in candidates
            and assigned_counts[current] < load_cap
        ):
            selected = current
        else:
            selected = min(
                candidates,
                key=lambda idx: (
                    assigned_counts[idx] >= load_cap,
                    assigned_counts[idx],
                    -float(state.channel_quality[idx]),
                    idx,
                ),
            )
        actions[uid] = int(selected)
        assigned_counts[int(selected)] += 1

    return actions


def _policy_label(bucket: str, policy: str) -> str:
    return f"{bucket}:{policy}"


def _bucket_from_label(policy_label: str) -> str:
    return policy_label.split(":", 1)[0]


def _base_policy_from_label(policy_label: str) -> str:
    return policy_label.split(":", 1)[1]


def _rollout_association_trajectories(
    *,
    cfg: dict[str, Any],
    settings: _RAEE06Settings,
    max_steps: int | None,
) -> tuple[dict[str, dict[int, list[np.ndarray]]], dict[str, dict[str, Any]]]:
    trajectories: dict[str, dict[int, list[np.ndarray]]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    base_policies = (
        settings.matched_control_association_policy,
        *settings.candidate_association_policies,
        *settings.diagnostic_association_policies,
        *settings.oracle_association_policies,
    )
    unique_policies = tuple(dict.fromkeys(base_policies))

    for spec in settings.bucket_specs:
        for policy in unique_policies:
            label = _policy_label(spec.name, policy)
            env = build_environment(cfg)
            rows_by_seed: dict[int, list[np.ndarray]] = defaultdict(list)
            for eval_seed in spec.evaluation_seed_set:
                env_seed_seq, mobility_seed_seq = np.random.SeedSequence(eval_seed).spawn(2)
                env_rng = np.random.default_rng(env_seed_seq)
                mobility_rng = np.random.default_rng(mobility_seed_seq)
                user_states, masks, _diag = env.reset(env_rng, mobility_rng)
                steps_seen = 0
                while True:
                    if max_steps is not None and steps_seen >= max_steps:
                        break
                    actions = _select_actions_for_association_policy(
                        policy,
                        user_states=user_states,
                        masks=masks,
                        current_assignments=env.current_assignments(),
                        settings=settings,
                    )
                    result = env.step(actions, env_rng)
                    rows_by_seed[int(eval_seed)].append(actions.copy())
                    steps_seen += 1
                    if result.done:
                        break
                    user_states = result.user_states
                    masks = result.action_masks
            trajectories[label] = rows_by_seed
            metadata[label] = {
                "evaluation_bucket": spec.name,
                "association_policy": policy,
                "association_action_contract": (
                    "fixed-by-trajectory"
                    if policy == settings.matched_control_association_policy
                    else (
                        "per-user-one-hot-greedy-diagnostic"
                        if policy == PER_USER_GREEDY_BEST_BEAM
                        else "centralized-active-set-served-set-proposal"
                    )
                ),
                "candidate_role": (
                    "matched-control"
                    if policy == settings.matched_control_association_policy
                    else (
                        "diagnostic-greedy-comparator"
                        if policy in settings.diagnostic_association_policies
                        else "candidate-active-set-proposal"
                    )
                ),
            }
    return trajectories, metadata


def _snapshot_index(
    snapshots: list[_StepSnapshot],
) -> dict[tuple[str, int, int], _StepSnapshot]:
    return {
        (
            str(snapshot.trajectory_policy),
            int(snapshot.evaluation_seed),
            int(snapshot.step_index),
        ): snapshot
        for snapshot in snapshots
    }


def _power_vector_key(power_vector: np.ndarray, active_mask: np.ndarray) -> str:
    return _format_vector(power_vector[active_mask])


def _safe_greedy_step_row(
    *,
    snapshot: _StepSnapshot,
    settings: _RAEE06Settings,
    power_semantics: str,
    trajectory_policy: str,
    association_policy: str,
    association_role: str,
) -> dict[str, Any]:
    requested, label = _safe_greedy_power_vector(snapshot, _ra_ee_05_settings(settings))
    row = _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics=power_semantics,
        selected_power_profile=label,
        power_vector=requested,
        settings=settings.audit,
    )
    row["trajectory_policy"] = trajectory_policy
    row["source_association_policy"] = association_policy
    row["association_role"] = association_role
    row["association_action_contract"] = (
        "fixed-by-trajectory"
        if association_role == "matched-control"
        else (
            "per-user-one-hot-greedy-diagnostic"
            if association_policy == PER_USER_GREEDY_BEST_BEAM
            else "centralized-active-set-served-set-proposal"
        )
    )
    row["requested_power_vector_w"] = _format_vector(requested)
    row["effective_power_vector_w"] = row["beam_transmit_power_w"]
    row["power_allocator"] = "safe-greedy-power-allocator"
    row["power_repair_used"] = False
    row["learned_association_enabled"] = False
    row["joint_association_power_training_enabled"] = False
    row["catfish_enabled"] = False
    row["rb_bandwidth_allocation_enabled"] = False
    row["active_set_size"] = row["active_beam_count"]
    row["served_set_size"] = row["served_count"]
    return row


def _fixed_1w_step_row(
    *,
    snapshot: _StepSnapshot,
    settings: _RAEE06Settings,
    trajectory_policy: str,
) -> dict[str, Any]:
    powers = _power_vector_for_candidate(snapshot, settings.audit, "fixed-control")
    row = _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics="matched-fixed-association+fixed-1w-control",
        selected_power_profile=f"fixed-{settings.audit.fixed_control_power_w:g}w-control",
        power_vector=powers,
        settings=settings.audit,
    )
    row["trajectory_policy"] = trajectory_policy
    row["source_association_policy"] = settings.matched_control_association_policy
    row["association_role"] = "matched-control-fixed-1w-diagnostic"
    row["association_action_contract"] = "fixed-by-trajectory"
    row["requested_power_vector_w"] = _format_vector(powers)
    row["effective_power_vector_w"] = row["beam_transmit_power_w"]
    row["power_allocator"] = "fixed-control-1w-per-active-beam"
    row["power_repair_used"] = False
    row["learned_association_enabled"] = False
    row["joint_association_power_training_enabled"] = False
    row["catfish_enabled"] = False
    row["rb_bandwidth_allocation_enabled"] = False
    row["active_set_size"] = row["active_beam_count"]
    row["served_set_size"] = row["served_count"]
    return row


def _oracle_step_row(
    *,
    candidate_label: str,
    control_row: dict[str, Any],
    seed: int,
    step_index: int,
    snapshots_by_key: dict[tuple[str, int, int], _StepSnapshot],
    settings: _RAEE06Settings,
) -> dict[str, Any] | None:
    bucket = _bucket_from_label(candidate_label)
    best: dict[str, Any] | None = None
    best_policy: str | None = None
    for policy in settings.oracle_association_policies:
        proposal_label = _policy_label(bucket, policy)
        snapshot = snapshots_by_key.get((proposal_label, seed, step_index))
        if snapshot is None:
            continue
        row = _select_oracle_step(
            snapshot=snapshot,
            control_row=control_row,
            settings=settings.audit,
        )
        if best is None:
            best = row
            best_policy = policy
            continue
        row_ee = -math.inf if row["EE_system_bps_per_w"] is None else float(row["EE_system_bps_per_w"])
        best_ee = -math.inf if best["EE_system_bps_per_w"] is None else float(best["EE_system_bps_per_w"])
        if (
            row_ee > best_ee + 1e-12
            or (
                abs(row_ee - best_ee) <= 1e-12
                and float(row["sum_user_throughput_bps"])
                > float(best["sum_user_throughput_bps"])
            )
        ):
            best = row
            best_policy = policy

    if best is None or best_policy is None:
        return None
    best["trajectory_policy"] = candidate_label
    best["power_semantics"] = RA_EE_06_ORACLE
    best["source_association_policy"] = best_policy
    best["association_role"] = "diagnostic-association-power-oracle"
    best["association_action_contract"] = "finite-active-set-oracle-diagnostic"
    best["requested_power_vector_w"] = best["beam_transmit_power_w"]
    best["effective_power_vector_w"] = best["beam_transmit_power_w"]
    best["power_allocator"] = "constrained-oracle-upper-bound"
    best["power_repair_used"] = False
    best["learned_association_enabled"] = False
    best["joint_association_power_training_enabled"] = False
    best["catfish_enabled"] = False
    best["rb_bandwidth_allocation_enabled"] = False
    best["active_set_size"] = best["active_beam_count"]
    best["served_set_size"] = best["served_count"]
    return best


def _evaluation_rows(
    *,
    snapshots: list[_StepSnapshot],
    settings: _RAEE06Settings,
    include_oracle: bool,
    include_fixed_1w_control: bool,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[float]]]:
    by_key = _snapshot_index(snapshots)
    rows: list[dict[str, Any]] = []
    user_throughputs_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)

    candidate_policies = (
        *settings.candidate_association_policies,
        *settings.diagnostic_association_policies,
    )
    for spec in settings.bucket_specs:
        control_label = _policy_label(
            spec.name,
            settings.matched_control_association_policy,
        )
        for proposal_policy in candidate_policies:
            candidate_label = _policy_label(spec.name, proposal_policy)
            role = (
                "diagnostic-greedy-comparator"
                if proposal_policy in settings.diagnostic_association_policies
                else "candidate-active-set-proposal"
            )
            for seed in spec.evaluation_seed_set:
                step_keys = sorted(
                    step
                    for label, eval_seed, step in by_key
                    if label == candidate_label and eval_seed == int(seed)
                )
                for step_index in step_keys:
                    control_snapshot = by_key.get((control_label, int(seed), step_index))
                    candidate_snapshot = by_key.get((candidate_label, int(seed), step_index))
                    if control_snapshot is None or candidate_snapshot is None:
                        continue
                    trajectory_policy = candidate_label
                    step_rows: list[dict[str, Any]] = []
                    if include_fixed_1w_control:
                        step_rows.append(
                            _fixed_1w_step_row(
                                snapshot=control_snapshot,
                                settings=settings,
                                trajectory_policy=trajectory_policy,
                            )
                        )
                    matched = _safe_greedy_step_row(
                        snapshot=control_snapshot,
                        settings=settings,
                        power_semantics=RA_EE_06_MATCHED_CONTROL,
                        trajectory_policy=trajectory_policy,
                        association_policy=settings.matched_control_association_policy,
                        association_role="matched-control",
                    )
                    candidate_semantics = (
                        RA_EE_06_GREEDY_DIAGNOSTIC
                        if proposal_policy in settings.diagnostic_association_policies
                        else RA_EE_06_CANDIDATE
                    )
                    candidate = _safe_greedy_step_row(
                        snapshot=candidate_snapshot,
                        settings=settings,
                        power_semantics=candidate_semantics,
                        trajectory_policy=trajectory_policy,
                        association_policy=proposal_policy,
                        association_role=role,
                    )
                    step_rows.extend([matched, candidate])
                    if include_oracle and proposal_policy in settings.candidate_association_policies:
                        oracle = _oracle_step_row(
                            candidate_label=candidate_label,
                            control_row=matched,
                            seed=int(seed),
                            step_index=step_index,
                            snapshots_by_key=by_key,
                            settings=settings,
                        )
                        if oracle is not None:
                            step_rows.append(oracle)

                    for row in step_rows:
                        row["evaluation_bucket"] = spec.name
                        row["matched_control_association_policy"] = (
                            settings.matched_control_association_policy
                        )
                        row["candidate_association_policy"] = proposal_policy
                        throughputs = row.pop("_user_throughputs")
                        rows.append(row)
                        user_throughputs_by_key[
                            (str(row["trajectory_policy"]), str(row["power_semantics"]))
                        ].extend(float(value) for value in throughputs.tolist())

    return rows, user_throughputs_by_key


def _augment_summaries(
    summaries: list[dict[str, Any]],
    *,
    step_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    vectors_by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    metadata_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    active_sets_by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    served_sets_by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    source_policy_by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in step_rows:
        key = (str(row["trajectory_policy"]), str(row["power_semantics"]))
        vectors_by_key[key].append(str(row["effective_power_vector_w"]))
        active_sets_by_key[key].append(int(row["active_set_size"]))
        served_sets_by_key[key].append(int(row["served_set_size"]))
        source_policy_by_key[key].append(str(row["source_association_policy"]))
        metadata_by_key.setdefault(
            key,
            {
                "evaluation_bucket": row["evaluation_bucket"],
                "association_role": row["association_role"],
                "association_action_contract": row["association_action_contract"],
                "matched_control_association_policy": row[
                    "matched_control_association_policy"
                ],
                "candidate_association_policy": row["candidate_association_policy"],
                "power_allocator": row["power_allocator"],
            },
        )

    for summary in summaries:
        key = (str(summary["trajectory_policy"]), str(summary["power_semantics"]))
        summary.update(metadata_by_key[key])
        summary["selected_power_vector_distribution"] = _categorical_distribution(
            vectors_by_key[key]
        )
        summary["active_set_size_distribution"] = _categorical_distribution(
            [str(value) for value in active_sets_by_key[key]]
        )
        summary["served_set_size_distribution"] = _categorical_distribution(
            [str(value) for value in served_sets_by_key[key]]
        )
        summary["source_association_policy_distribution"] = _categorical_distribution(
            source_policy_by_key[key]
        )
        summary["active_set_contract_is_not_per_user_greedy"] = (
            summary["association_action_contract"]
            == "centralized-active-set-served-set-proposal"
        )
    return summaries


def _pct_delta(reference: float | None, value: float | None) -> float | None:
    if reference is None or value is None or abs(float(reference)) < 1e-12:
        return None
    return float((float(value) - float(reference)) / abs(float(reference)))


def _guardrail_result(
    *,
    candidate: dict[str, Any],
    control: dict[str, Any],
    settings: _RAEE06Settings,
) -> dict[str, Any]:
    p05_threshold = settings.audit.p05_min_ratio_vs_control * float(
        control["throughput_p05_user_step_bps"]
    )
    p05_pass = (
        candidate["throughput_p05_user_step_bps"] is not None
        and float(candidate["throughput_p05_user_step_bps"]) >= p05_threshold
    )
    served_threshold = (
        float(control["served_ratio"]) + settings.audit.served_ratio_min_delta_vs_control
    )
    outage_threshold = (
        float(control["outage_ratio"]) + settings.audit.outage_ratio_max_delta_vs_control
    )
    served_pass = float(candidate["served_ratio"]) >= served_threshold
    outage_pass = float(candidate["outage_ratio"]) <= outage_threshold
    budget_pass = int(candidate["budget_violations"]["step_count"]) == 0
    per_beam_pass = int(candidate["per_beam_power_violations"]["step_count"]) == 0
    inactive_pass = int(candidate["inactive_beam_nonzero_power_step_count"]) == 0
    ee_delta = (
        None
        if candidate["EE_system_aggregate_bps_per_w"] is None
        or control["EE_system_aggregate_bps_per_w"] is None
        else float(candidate["EE_system_aggregate_bps_per_w"])
        - float(control["EE_system_aggregate_bps_per_w"])
    )
    throughput_delta = _pct_delta(
        control["throughput_mean_user_step_bps"],
        candidate["throughput_mean_user_step_bps"],
    )
    ee_pct_delta = _pct_delta(
        control["EE_system_aggregate_bps_per_w"],
        candidate["EE_system_aggregate_bps_per_w"],
    )
    noncollapsed = (
        float(candidate["one_active_beam_step_ratio"])
        <= settings.max_one_active_beam_ratio_for_acceptance
    )
    return {
        "evaluation_bucket": candidate["evaluation_bucket"],
        "trajectory_policy": candidate["trajectory_policy"],
        "candidate_association_policy": candidate["candidate_association_policy"],
        "power_semantics": candidate["power_semantics"],
        "matched_control_power_semantics": control["power_semantics"],
        "EE_system_delta_vs_matched_control": ee_delta,
        "EE_system_pct_delta_vs_matched_control": ee_pct_delta,
        "throughput_mean_pct_delta_vs_matched_control": throughput_delta,
        "throughput_p05_ratio_vs_matched_control": (
            None
            if control["throughput_p05_user_step_bps"] is None
            or abs(float(control["throughput_p05_user_step_bps"])) < 1e-12
            else float(candidate["throughput_p05_user_step_bps"])
            / abs(float(control["throughput_p05_user_step_bps"]))
        ),
        "p05_threshold_bps": p05_threshold,
        "p05_guardrail_pass": p05_pass,
        "served_ratio_threshold": served_threshold,
        "served_ratio_guardrail_pass": served_pass,
        "outage_ratio_threshold": outage_threshold,
        "outage_guardrail_pass": outage_pass,
        "budget_guardrail_pass": budget_pass,
        "per_beam_power_guardrail_pass": per_beam_pass,
        "inactive_beam_zero_w_guardrail_pass": inactive_pass,
        "QoS_guardrails_pass": bool(p05_pass and served_pass and outage_pass),
        "noncollapsed_active_set_guardrail_pass": noncollapsed,
        "denominator_varies_in_eval": bool(candidate["denominator_varies_in_eval"]),
        "active_set_contract_is_not_per_user_greedy": bool(
            candidate["active_set_contract_is_not_per_user_greedy"]
        ),
        "accepted": bool(
            ee_delta is not None
            and ee_delta > 0.0
            and p05_pass
            and served_pass
            and outage_pass
            and budget_pass
            and per_beam_pass
            and inactive_pass
            and noncollapsed
            and bool(candidate["denominator_varies_in_eval"])
        ),
    }


def _build_guardrail_checks(
    *,
    summaries: list[dict[str, Any]],
    settings: _RAEE06Settings,
) -> list[dict[str, Any]]:
    by_policy: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for summary in summaries:
        by_policy[str(summary["trajectory_policy"])][str(summary["power_semantics"])] = summary

    checks: list[dict[str, Any]] = []
    for _policy, rows in sorted(by_policy.items()):
        control = rows.get(RA_EE_06_MATCHED_CONTROL)
        if control is None:
            continue
        for semantics in (
            RA_EE_06_CANDIDATE,
            RA_EE_06_GREEDY_DIAGNOSTIC,
            RA_EE_06_ORACLE,
        ):
            candidate = rows.get(semantics)
            if candidate is None:
                continue
            checks.append(
                _guardrail_result(
                    candidate=candidate,
                    control=control,
                    settings=settings,
                )
            )
    return checks


def _ranking_checks(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_policy: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summaries:
        by_policy[str(row["trajectory_policy"])][str(row["power_semantics"])] = row

    checks: list[dict[str, Any]] = []
    compared_semantics = (RA_EE_06_MATCHED_CONTROL, RA_EE_06_CANDIDATE)
    for policy, rows in sorted(by_policy.items()):
        compared = [rows[key] for key in compared_semantics if key in rows]
        if len(compared) < 2:
            continue
        throughput_ranking = [
            row["power_semantics"]
            for row in sorted(
                compared,
                key=lambda row: float(row["throughput_mean_user_step_bps"] or -math.inf),
                reverse=True,
            )
        ]
        ee_ranking = [
            row["power_semantics"]
            for row in sorted(
                compared,
                key=lambda row: float(row["EE_system_aggregate_bps_per_w"] or -math.inf),
                reverse=True,
            )
        ]
        checks.append(
            {
                "evaluation_bucket": compared[0]["evaluation_bucket"],
                "trajectory_policy": policy,
                "candidate_association_policy": compared[1][
                    "candidate_association_policy"
                ],
                "compared_power_semantics": list(compared_semantics),
                "throughput_rescore_ranking": throughput_ranking,
                "EE_rescore_ranking": ee_ranking,
                "throughput_rescore_winner": throughput_ranking[0],
                "EE_rescore_winner": ee_ranking[0],
                "association_policy_throughput_rescore_vs_EE_rescore_ranking_changes": (
                    throughput_ranking != ee_ranking
                ),
                "association_policy_throughput_rescore_vs_EE_rescore_top_changes": (
                    throughput_ranking[0] != ee_ranking[0]
                ),
            }
        )
    return checks


def _oracle_gap_diagnostics(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_policy: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summaries:
        by_policy[str(row["trajectory_policy"])][str(row["power_semantics"])] = row

    diagnostics: list[dict[str, Any]] = []
    for policy, rows in sorted(by_policy.items()):
        candidate = rows.get(RA_EE_06_CANDIDATE)
        oracle = rows.get(RA_EE_06_ORACLE)
        if candidate is None or oracle is None:
            continue
        candidate_ee = candidate["EE_system_aggregate_bps_per_w"]
        oracle_ee = oracle["EE_system_aggregate_bps_per_w"]
        gap = None if candidate_ee is None or oracle_ee is None else float(oracle_ee) - float(candidate_ee)
        diagnostics.append(
            {
                "evaluation_bucket": candidate["evaluation_bucket"],
                "trajectory_policy": policy,
                "candidate_association_policy": candidate[
                    "candidate_association_policy"
                ],
                "candidate_power_semantics": RA_EE_06_CANDIDATE,
                "oracle_power_semantics": RA_EE_06_ORACLE,
                "oracle_is_diagnostic_only": True,
                "candidate_EE_system_aggregate_bps_per_w": candidate_ee,
                "oracle_EE_system_aggregate_bps_per_w": oracle_ee,
                "oracle_EE_gap_vs_candidate_bps_per_w": gap,
                "oracle_EE_pct_gap_vs_candidate": (
                    None
                    if gap is None or abs(float(candidate_ee)) < 1e-12
                    else gap / abs(float(candidate_ee))
                ),
            }
        )
    return diagnostics


def _bucket_results(
    *,
    settings: _RAEE06Settings,
    summaries: list[dict[str, Any]],
    guardrail_checks: list[dict[str, Any]],
    ranking_checks: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    candidate_by_policy = {
        str(row["trajectory_policy"]): row
        for row in summaries
        if row["power_semantics"] == RA_EE_06_CANDIDATE
    }
    guardrail_by_policy = {
        str(row["trajectory_policy"]): row
        for row in guardrail_checks
        if row["power_semantics"] == RA_EE_06_CANDIDATE
    }
    ranking_by_policy = {str(row["trajectory_policy"]): row for row in ranking_checks}

    results: dict[str, dict[str, Any]] = {}
    for spec in settings.bucket_specs:
        labels = [
            _policy_label(spec.name, policy)
            for policy in settings.candidate_association_policies
        ]
        present = [label for label in labels if label in candidate_by_policy]
        noncollapsed = [
            label
            for label in present
            if float(candidate_by_policy[label]["one_active_beam_step_ratio"])
            <= settings.max_one_active_beam_ratio_for_acceptance
        ]
        positive = [
            label
            for label in noncollapsed
            if float(
                guardrail_by_policy.get(label, {}).get(
                    "EE_system_delta_vs_matched_control",
                    -math.inf,
                )
                or -math.inf
            )
            > 0.0
        ]
        accepted = [
            label
            for label in positive
            if bool(guardrail_by_policy.get(label, {}).get("accepted"))
        ]
        no_power_violations = all(
            bool(guardrail_by_policy.get(label, {}).get("budget_guardrail_pass"))
            and bool(guardrail_by_policy.get(label, {}).get("per_beam_power_guardrail_pass"))
            and bool(guardrail_by_policy.get(label, {}).get("inactive_beam_zero_w_guardrail_pass"))
            for label in present
        )
        qos_pass_for_accepted = bool(accepted) and all(
            bool(guardrail_by_policy[label]["p05_guardrail_pass"])
            and bool(guardrail_by_policy[label]["served_ratio_guardrail_pass"])
            and bool(guardrail_by_policy[label]["outage_guardrail_pass"])
            for label in accepted
        )
        ranking_separates_for_accepted = bool(accepted) and all(
            bool(
                ranking_by_policy.get(label, {}).get(
                    "association_policy_throughput_rescore_vs_EE_rescore_top_changes"
                )
            )
            for label in accepted
        )
        denominator_varies = bool(accepted) and all(
            bool(candidate_by_policy[label]["denominator_varies_in_eval"])
            for label in accepted
        )

        results[spec.name] = {
            "bucket": spec.name,
            "evaluation_seed_set": list(spec.evaluation_seed_set),
            "candidate_association_policies": list(settings.candidate_association_policies),
            "matched_control_association_policy": settings.matched_control_association_policy,
            "present_candidate_count": len(present),
            "noncollapsed_candidate_count": len(noncollapsed),
            "noncollapsed_candidate_policies": noncollapsed,
            "positive_EE_delta_candidate_count": len(positive),
            "positive_EE_delta_candidate_policies": positive,
            "accepted_candidate_count": len(accepted),
            "accepted_candidate_policies": accepted,
            "majority_noncollapsed_positive_EE_delta": (
                bool(noncollapsed) and len(positive) > len(noncollapsed) / 2.0
            ),
            "majority_noncollapsed_accepted": (
                bool(noncollapsed) and len(accepted) > len(noncollapsed) / 2.0
            ),
            "gains_not_concentrated_in_one_policy": len(positive) >= 2,
            "qos_guardrails_pass_for_accepted": qos_pass_for_accepted,
            "zero_budget_per_beam_inactive_power_violations": no_power_violations,
            "denominator_varies_for_accepted": denominator_varies,
            "active_set_contract_for_accepted": bool(accepted) and all(
                bool(candidate_by_policy[label]["active_set_contract_is_not_per_user_greedy"])
                for label in accepted
            ),
            "throughput_winner_vs_EE_winner_separate_for_accepted": (
                ranking_separates_for_accepted
            ),
            "one_active_beam_collapse_dominates": (
                len(present) > 0
                and len(present) - len(noncollapsed) > len(present) / 2.0
            ),
            "bucket_pass": (
                spec.name == CALIBRATION_BUCKET
                or (
                    bool(noncollapsed)
                    and len(positive) > len(noncollapsed) / 2.0
                    and len(accepted) > len(noncollapsed) / 2.0
                    and len(positive) >= 2
                    and qos_pass_for_accepted
                    and no_power_violations
                    and denominator_varies
                    and all(
                        bool(
                            candidate_by_policy[label][
                                "active_set_contract_is_not_per_user_greedy"
                            ]
                        )
                        for label in accepted
                    )
                )
            ),
        }
    return results


def _build_decision(
    *,
    summaries: list[dict[str, Any]],
    guardrail_checks: list[dict[str, Any]],
    bucket_results: dict[str, dict[str, Any]],
    include_oracle: bool,
) -> dict[str, Any]:
    held_out = bucket_results.get(HELD_OUT_BUCKET, {})
    candidate_summaries = [
        row for row in summaries if row["power_semantics"] == RA_EE_06_CANDIDATE
    ]
    candidate_guardrails = [
        row for row in guardrail_checks if row["power_semantics"] == RA_EE_06_CANDIDATE
    ]

    no_power_violations = all(
        bool(row.get("budget_guardrail_pass"))
        and bool(row.get("per_beam_power_guardrail_pass"))
        and bool(row.get("inactive_beam_zero_w_guardrail_pass"))
        for row in candidate_guardrails
    )
    learned_disabled = all(
        row.get("learned_association_enabled") in (None, False)
        for row in candidate_summaries
    )
    joint_disabled = all(
        row.get("joint_association_power_training_enabled") in (None, False)
        for row in candidate_summaries
    )
    active_set_contract = all(
        bool(row.get("active_set_contract_is_not_per_user_greedy"))
        for row in candidate_summaries
    )

    proof_flags = {
        "held_out_bucket_exists_and_reported_separately": bool(held_out),
        "association_counterfactual_only": True,
        "learned_association_disabled": learned_disabled,
        "joint_association_power_training_disabled": joint_disabled,
        "catfish_disabled": True,
        "multi_catfish_disabled": True,
        "rb_bandwidth_allocation_disabled": True,
        "old_EE_MODQN_continuation_disabled": True,
        "frozen_baseline_mutation": False,
        "active_set_served_set_proposal_contract": active_set_contract,
        "matched_control_uses_same_power_allocator": True,
        "safe_greedy_allocator_retained": True,
        "constrained_oracle_upper_bound_diagnostic_only": include_oracle,
        "majority_noncollapsed_held_out_positive_EE_delta": bool(
            held_out.get("majority_noncollapsed_positive_EE_delta")
        ),
        "majority_noncollapsed_held_out_accepted": bool(
            held_out.get("majority_noncollapsed_accepted")
        ),
        "held_out_gains_not_concentrated_in_one_policy": bool(
            held_out.get("gains_not_concentrated_in_one_policy")
        ),
        "p05_throughput_guardrail_pass_for_accepted_held_out": bool(
            held_out.get("qos_guardrails_pass_for_accepted")
        ),
        "served_ratio_does_not_drop_for_accepted_held_out": bool(
            held_out.get("qos_guardrails_pass_for_accepted")
        ),
        "outage_ratio_does_not_increase_for_accepted_held_out": bool(
            held_out.get("qos_guardrails_pass_for_accepted")
        ),
        "zero_budget_per_beam_inactive_power_violations": no_power_violations,
        "denominator_varies_for_accepted_held_out": bool(
            held_out.get("denominator_varies_for_accepted")
        ),
        "one_active_beam_collapse_avoided_for_accepted_held_out": not bool(
            held_out.get("one_active_beam_collapse_dominates")
        ),
        "scalar_reward_success_basis": False,
        "per_user_EE_credit_success_basis": False,
        "physical_energy_saving_claim": False,
        "hobs_optimizer_claim": False,
        "full_RA_EE_MODQN_claim": False,
    }
    stop_conditions = {
        "held_out_bucket_missing": not bool(held_out),
        "association_proposals_collapse_to_one_active_beam": bool(
            held_out.get("one_active_beam_collapse_dominates")
        ),
        "held_out_association_gains_disappear_or_concentrate": not bool(
            held_out.get("gains_not_concentrated_in_one_policy")
        ),
        "p05_throughput_guardrail_fails": any(
            row["evaluation_bucket"] == HELD_OUT_BUCKET
            and float(row["EE_system_delta_vs_matched_control"] or 0.0) > 0.0
            and not bool(row["QoS_guardrails_pass"])
            for row in candidate_guardrails
        ),
        "denominator_becomes_fixed_for_accepted": bool(
            held_out.get("accepted_candidate_count")
        )
        and not bool(held_out.get("denominator_varies_for_accepted")),
        "budget_or_inactive_power_violations": not no_power_violations,
        "per_user_greedy_used_as_candidate_contract": not active_set_contract,
        "learned_association_added": not learned_disabled,
        "joint_training_added": not joint_disabled,
        "catfish_added": False,
        "frozen_baseline_mutated": False,
        "oracle_used_as_candidate_claim": False,
    }
    required_true = (
        "held_out_bucket_exists_and_reported_separately",
        "association_counterfactual_only",
        "learned_association_disabled",
        "joint_association_power_training_disabled",
        "catfish_disabled",
        "multi_catfish_disabled",
        "rb_bandwidth_allocation_disabled",
        "old_EE_MODQN_continuation_disabled",
        "active_set_served_set_proposal_contract",
        "matched_control_uses_same_power_allocator",
        "safe_greedy_allocator_retained",
        "constrained_oracle_upper_bound_diagnostic_only",
        "majority_noncollapsed_held_out_positive_EE_delta",
        "majority_noncollapsed_held_out_accepted",
        "held_out_gains_not_concentrated_in_one_policy",
        "p05_throughput_guardrail_pass_for_accepted_held_out",
        "served_ratio_does_not_drop_for_accepted_held_out",
        "outage_ratio_does_not_increase_for_accepted_held_out",
        "zero_budget_per_beam_inactive_power_violations",
        "denominator_varies_for_accepted_held_out",
        "one_active_beam_collapse_avoided_for_accepted_held_out",
    )
    pass_required = all(bool(proof_flags[field]) for field in required_true)
    pass_required = (
        pass_required
        and proof_flags["scalar_reward_success_basis"] is False
        and proof_flags["per_user_EE_credit_success_basis"] is False
        and proof_flags["physical_energy_saving_claim"] is False
        and proof_flags["hobs_optimizer_claim"] is False
        and proof_flags["full_RA_EE_MODQN_claim"] is False
    )
    if any(bool(value) for value in stop_conditions.values()):
        decision = "BLOCKED"
    elif pass_required:
        decision = "PASS"
    else:
        decision = "NEEDS MORE EVIDENCE"

    return {
        "ra_ee_06_decision": decision,
        "proof_flags": proof_flags,
        "stop_conditions": stop_conditions,
        "allowed_claim": (
            "PASS only means the offline active-set association counterfactual "
            "gate passed against matched fixed-association + same-power-allocator "
            "control. It is not learned RA-EE-MODQN."
            if decision == "PASS"
            else "Do not proceed to learned hierarchical RA-EE training without resolving blockers."
        ),
    }


def _compact_summary_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "evaluation_bucket",
        "trajectory_policy",
        "candidate_association_policy",
        "source_association_policy_distribution",
        "association_role",
        "association_action_contract",
        "power_semantics",
        "power_allocator",
        "EE_system_aggregate_bps_per_w",
        "EE_system_step_mean_bps_per_w",
        "throughput_mean_user_step_bps",
        "throughput_p05_user_step_bps",
        "served_ratio",
        "outage_ratio",
        "handover_count",
        "active_beam_count_distribution",
        "active_set_size_distribution",
        "served_set_size_distribution",
        "selected_power_profile_distribution",
        "selected_power_vector_distribution",
        "total_active_beam_power_w_distribution",
        "denominator_varies_in_eval",
        "one_active_beam_step_ratio",
        "budget_violations",
        "per_beam_power_violations",
        "inactive_beam_nonzero_power_step_count",
        "throughput_vs_EE_system_correlation",
    )
    return [{field: row[field] for field in fields} for row in summaries]


def _review_lines(summary: dict[str, Any]) -> list[str]:
    decision = summary["ra_ee_06_decision"]
    proof = summary["proof_flags"]
    held_out = summary["bucket_results"][HELD_OUT_BUCKET]
    lines = [
        "# RA-EE-06 Association Counterfactual / Oracle Review",
        "",
        "Offline active-set / served-set association counterfactual gate only. "
        "No learned association, joint association + power training, Catfish, "
        "multi-Catfish, RB / bandwidth allocation, HOBS optimizer claim, "
        "physical energy-saving claim, or frozen baseline mutation was performed.",
        "",
        "## Protocol",
        "",
        f"- method label: `{summary['protocol']['method_label']}`",
        f"- implementation sublabel: `{summary['protocol']['implementation_sublabel']}`",
        f"- config: `{summary['inputs']['config_path']}`",
        f"- artifact namespace: `{summary['inputs']['output_dir']}`",
        f"- matched control: `{summary['protocol']['matched_control']}`",
        f"- candidate: `{summary['protocol']['candidate']}`",
        f"- oracle: `{summary['protocol']['oracle_upper_bound']}`",
        f"- candidate association policies: `{summary['protocol']['candidate_association_policies']}`",
        f"- diagnostic association policies: `{summary['protocol']['diagnostic_association_policies']}`",
        "",
        "## Held-Out Gate",
        "",
        f"- noncollapsed candidates: `{held_out['noncollapsed_candidate_policies']}`",
        f"- positive EE delta candidates: `{held_out['positive_EE_delta_candidate_policies']}`",
        f"- accepted candidates: `{held_out['accepted_candidate_policies']}`",
        f"- majority noncollapsed positive EE delta: `{held_out['majority_noncollapsed_positive_EE_delta']}`",
        f"- majority noncollapsed accepted: `{held_out['majority_noncollapsed_accepted']}`",
        f"- gains not concentrated in one policy: `{held_out['gains_not_concentrated_in_one_policy']}`",
        f"- QoS guardrails pass for accepted: `{held_out['qos_guardrails_pass_for_accepted']}`",
        f"- zero budget / per-beam / inactive-power violations: `{held_out['zero_budget_per_beam_inactive_power_violations']}`",
        f"- denominator varies for accepted: `{held_out['denominator_varies_for_accepted']}`",
        "",
        "## Gate Flags",
        "",
    ]
    for key, value in proof.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- RA-EE-06 decision: `{decision}`",
            f"- allowed claim: {summary['decision_detail']['allowed_claim']}",
            "",
            "## Forbidden Claims",
            "",
        ]
    )
    for claim in summary["forbidden_claims_still_active"]:
        lines.append(f"- {claim}")
    return lines


def export_ra_ee_06_association_counterfactual_oracle(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    calibration_seed_set: tuple[int, ...] | None = None,
    held_out_seed_set: tuple[int, ...] | None = None,
    candidate_association_policies: tuple[str, ...] | None = None,
    max_steps: int | None = None,
    include_oracle: bool = True,
    include_fixed_1w_control: bool = True,
) -> dict[str, Any]:
    """Export RA-EE-06 association counterfactual / oracle artifacts."""
    if max_steps is not None and max_steps < 1:
        raise ValueError(f"max_steps must be >= 1 when provided, got {max_steps!r}")

    cfg = load_training_yaml(config_path)
    env = build_environment(cfg)
    if env.power_surface_config.hobs_power_surface_mode != (
        HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK
    ):
        raise ValueError("RA-EE-06 config must opt into the power-codebook surface.")

    settings = _settings_from_config(cfg)
    bucket_specs: list[_BucketSpec] = []
    for spec in settings.bucket_specs:
        bucket_specs.append(
            _BucketSpec(
                name=spec.name,
                trajectory_families=spec.trajectory_families,
                evaluation_seed_set=(
                    tuple(calibration_seed_set)
                    if spec.name == CALIBRATION_BUCKET and calibration_seed_set is not None
                    else (
                        tuple(held_out_seed_set)
                        if spec.name == HELD_OUT_BUCKET and held_out_seed_set is not None
                        else spec.evaluation_seed_set
                    )
                ),
            )
        )
    run_settings = _RAEE06Settings(
        method_label=settings.method_label,
        implementation_sublabel=settings.implementation_sublabel,
        audit=settings.audit,
        bucket_specs=tuple(bucket_specs),
        matched_control_association_policy=settings.matched_control_association_policy,
        candidate_association_policies=(
            tuple(candidate_association_policies)
            if candidate_association_policies is not None
            else settings.candidate_association_policies
        ),
        diagnostic_association_policies=settings.diagnostic_association_policies,
        oracle_association_policies=settings.oracle_association_policies,
        min_active_beams=settings.min_active_beams,
        max_active_beams=settings.max_active_beams,
        target_users_per_active_beam=settings.target_users_per_active_beam,
        load_cap_overflow_users=settings.load_cap_overflow_users,
        candidate_max_demoted_beams=settings.candidate_max_demoted_beams,
        candidate_step_p05_guardrail_margin=(
            settings.candidate_step_p05_guardrail_margin
        ),
        max_one_active_beam_ratio_for_acceptance=(
            settings.max_one_active_beam_ratio_for_acceptance
        ),
    )
    _validate_policies("candidate policies", run_settings.candidate_association_policies)

    trajectories, association_metadata = _rollout_association_trajectories(
        cfg=cfg,
        settings=run_settings,
        max_steps=max_steps,
    )
    snapshots = _build_unit_power_snapshots(
        base_cfg=cfg,
        settings=run_settings.audit,
        trajectories=trajectories,
    )
    step_rows, user_throughputs_by_key = _evaluation_rows(
        snapshots=snapshots,
        settings=run_settings,
        include_oracle=include_oracle,
        include_fixed_1w_control=include_fixed_1w_control,
    )
    summaries = _summarize_all(
        rows=step_rows,
        user_throughputs_by_key=user_throughputs_by_key,
    )
    summaries = _augment_summaries(summaries, step_rows=step_rows)
    guardrail_checks = _build_guardrail_checks(
        summaries=summaries,
        settings=run_settings,
    )
    ranking_checks = _ranking_checks(summaries)
    oracle_gap_diagnostics = _oracle_gap_diagnostics(summaries)
    bucket_results = _bucket_results(
        settings=run_settings,
        summaries=summaries,
        guardrail_checks=guardrail_checks,
        ranking_checks=ranking_checks,
    )
    decision_detail = _build_decision(
        summaries=summaries,
        guardrail_checks=guardrail_checks,
        bucket_results=bucket_results,
        include_oracle=include_oracle,
    )

    out_dir = Path(output_dir)
    step_csv = _write_csv(
        out_dir / "ra_ee_06_step_metrics.csv",
        step_rows,
        fieldnames=list(step_rows[0].keys()),
    )
    compact_rows = _compact_summary_rows(summaries)
    summary_csv = _write_csv(
        out_dir / "ra_ee_06_candidate_summary.csv",
        compact_rows,
        fieldnames=list(compact_rows[0].keys()),
    )

    protocol = {
        "phase": "RA-EE-06",
        "method_label": run_settings.method_label,
        "method_family": "RA-EE hierarchical association + power design",
        "implementation_sublabel": run_settings.implementation_sublabel,
        "training": "none; offline counterfactual association proposals only",
        "learned_association": "disabled",
        "association_training": "disabled",
        "joint_association_power_training": "disabled",
        "catfish": "disabled",
        "multi_catfish": "disabled",
        "rb_bandwidth_allocation": "disabled/not-modeled",
        "old_EE_MODQN_continuation": "forbidden/not-performed",
        "frozen_baseline_mutation": "forbidden/not-performed",
        "hobs_optimizer_claim": "forbidden/not-made",
        "physical_energy_saving_claim": "forbidden/not-made",
        "association_action_contract": "centralized-active-set-served-set-proposal",
        "matched_control": RA_EE_06_MATCHED_CONTROL,
        "candidate": RA_EE_06_CANDIDATE,
        "greedy_diagnostic": RA_EE_06_GREEDY_DIAGNOSTIC,
        "oracle_upper_bound": RA_EE_06_ORACLE,
        "oracle_upper_bound_diagnostic_only": include_oracle,
        "matched_control_association_policy": (
            run_settings.matched_control_association_policy
        ),
        "candidate_association_policies": list(
            run_settings.candidate_association_policies
        ),
        "diagnostic_association_policies": list(
            run_settings.diagnostic_association_policies
        ),
        "oracle_association_policies": list(run_settings.oracle_association_policies),
        "power_allocator_embedding": "post-association optimizer",
        "power_allocator": "safe-greedy-power-allocator",
        "system_EE_primary": True,
        "scalar_reward_success_basis": False,
        "per_user_EE_credit_success_basis": False,
    }
    constraints = {
        "min_active_beams": run_settings.min_active_beams,
        "max_active_beams": run_settings.max_active_beams,
        "target_users_per_active_beam": run_settings.target_users_per_active_beam,
        "load_cap_overflow_users": run_settings.load_cap_overflow_users,
        "max_one_active_beam_ratio_for_acceptance": (
            run_settings.max_one_active_beam_ratio_for_acceptance
        ),
        "per_beam_max_power_w": run_settings.audit.per_beam_max_power_w,
        "total_power_budget_w": run_settings.audit.total_power_budget_w,
        "inactive_beam_policy": "zero-w",
        "codebook_levels_w": list(run_settings.audit.codebook_levels_w),
        "p05_throughput_min_ratio_vs_control": (
            run_settings.audit.p05_min_ratio_vs_control
        ),
        "served_ratio_min_delta_vs_control": (
            run_settings.audit.served_ratio_min_delta_vs_control
        ),
        "outage_ratio_max_delta_vs_control": (
            run_settings.audit.outage_ratio_max_delta_vs_control
        ),
        "candidate_max_demoted_beams": run_settings.candidate_max_demoted_beams,
        "candidate_step_p05_guardrail_margin": (
            run_settings.candidate_step_p05_guardrail_margin
        ),
        "power_repair": "not-used; requested and effective vectors are exported",
        "effective_power_vector_contract": (
            "same effective_power_vector_w feeds SINR numerator, throughput, "
            "EE denominator, audit logs, and budget checks"
        ),
    }
    summary = {
        "inputs": {
            "config_path": str(Path(config_path)),
            "output_dir": str(out_dir),
            "max_steps": max_steps,
        },
        "protocol": protocol,
        "constraints": constraints,
        "association_metadata": association_metadata,
        "candidate_summaries": summaries,
        "guardrail_checks": guardrail_checks,
        "ranking_separation_result": {
            "comparison_matched_control_vs_candidate": ranking_checks,
        },
        "bucket_results": bucket_results,
        "oracle_gap_diagnostics": oracle_gap_diagnostics,
        "decision_detail": decision_detail,
        "proof_flags": decision_detail["proof_flags"],
        "stop_conditions": decision_detail["stop_conditions"],
        "ra_ee_06_decision": decision_detail["ra_ee_06_decision"],
        "remaining_blockers": [
            "This is offline association counterfactual / oracle evidence only.",
            "No learned association or full RA-EE-MODQN policy exists.",
            "No joint association + power training exists.",
            "No RB / bandwidth allocation is included.",
            "The constrained association + power oracle is diagnostic only.",
        ],
        "forbidden_claims_still_active": [
            "Do not call RA-EE-06 full RA-EE-MODQN.",
            "Do not claim learned association effectiveness.",
            "Do not claim joint association + power training.",
            "Do not claim old EE-MODQN effectiveness.",
            "Do not claim HOBS optimizer behavior.",
            "Do not claim Catfish, multi-Catfish, or Catfish-EE-MODQN effectiveness.",
            "Do not add or claim RB / bandwidth allocation.",
            "Do not treat per-user EE credit as system EE.",
            "Do not use scalar reward alone as success evidence.",
            "Do not claim full paper-faithful reproduction.",
            "Do not claim physical energy saving.",
        ],
    }

    summary_path = write_json(
        out_dir / "ra_ee_06_association_counterfactual_oracle_summary.json",
        summary,
    )
    review_path = out_dir / "review.md"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text("\n".join(_review_lines(summary)) + "\n")

    return {
        "ra_ee_06_association_counterfactual_oracle_summary": summary_path,
        "ra_ee_06_candidate_summary_csv": summary_csv,
        "ra_ee_06_step_metrics": step_csv,
        "review_md": review_path,
        "summary": summary,
    }


__all__ = [
    "ACTIVE_SET_LOAD_SPREAD",
    "ACTIVE_SET_QUALITY_SPREAD",
    "ACTIVE_SET_STICKY_SPREAD",
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT_DIR",
    "FIXED_HOLD_CURRENT",
    "PER_USER_GREEDY_BEST_BEAM",
    "RA_EE_06_CANDIDATE",
    "RA_EE_06_GREEDY_DIAGNOSTIC",
    "RA_EE_06_MATCHED_CONTROL",
    "RA_EE_06_METHOD_LABEL",
    "RA_EE_06_ORACLE",
    "export_ra_ee_06_association_counterfactual_oracle",
]
