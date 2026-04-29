"""RA-EE-06B association proposal refinement / oracle distillation audit.

This exporter is deliberately offline and deterministic. It evaluates
proposal-rule association candidates against the same RA-EE-04/05
safe-greedy power allocator used by the matched fixed-association control,
while keeping association and power oracles diagnostic-only.
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
from .ra_ee_06_association_counterfactual_oracle import (
    FIXED_HOLD_CURRENT,
    PER_USER_GREEDY_BEST_BEAM,
)


DEFAULT_CONFIG = "configs/ra-ee-06b-association-proposal-refinement.resolved.yaml"
DEFAULT_OUTPUT_DIR = "artifacts/ra-ee-06b-association-proposal-refinement"

RA_EE_06B_METHOD_LABEL = "RA-EE association proposal refinement / oracle distillation audit"
RA_EE_06B_MATCHED_CONTROL = "matched-fixed-association+safe-greedy-power-allocator"
RA_EE_06B_CANDIDATE = "association-proposal-rule+safe-greedy-power-allocator"
RA_EE_06B_PROPOSAL_FIXED_1W = "association-proposal-rule+fixed-1w-diagnostic"
RA_EE_06B_GREEDY_DIAGNOSTIC = "per-user-greedy-best-beam+safe-greedy-power-allocator"
RA_EE_06B_ORACLE_SAFE_GREEDY = "association-oracle+same-safe-greedy-diagnostic"
RA_EE_06B_ORACLE_CONSTRAINED = "association-oracle+constrained-power-upper-bound"
RA_EE_06B_MATCHED_FIXED_CONSTRAINED = (
    "matched-fixed-association+constrained-power-oracle-isolation"
)

STICKY_ORACLE_COUNT_LOCAL_SEARCH = "sticky-oracle-count-local-search"
P05_SLACK_AWARE_ACTIVE_SET = "p05-slack-aware-active-set"
POWER_RESPONSE_AWARE_LOAD_BALANCE = "power-response-aware-load-balance"
BOUNDED_MOVE_SERVED_SET = "bounded-move-served-set"
ORACLE_SCORE_TOPK_ACTIVE_SET = "oracle-score-topk-active-set"

RA_EE_06B_PROPOSAL_POLICIES = (
    STICKY_ORACLE_COUNT_LOCAL_SEARCH,
    P05_SLACK_AWARE_ACTIVE_SET,
    POWER_RESPONSE_AWARE_LOAD_BALANCE,
    BOUNDED_MOVE_SERVED_SET,
    ORACLE_SCORE_TOPK_ACTIVE_SET,
)
SUPPORTED_ASSOCIATION_POLICIES = (
    FIXED_HOLD_CURRENT,
    *RA_EE_06B_PROPOSAL_POLICIES,
    PER_USER_GREEDY_BEST_BEAM,
)


@dataclass(frozen=True)
class _RAEE06BSettings:
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
    max_two_beam_overload_step_ratio: float
    max_moved_user_ratio: float
    max_moved_user_ratio_per_step: float
    min_oracle_gap_closed_ratio: float
    quality_margin_for_move: float
    local_search_swap_limit: int
    trace_top_k: int


@dataclass(frozen=True)
class _AssociationTrace:
    evaluation_bucket: str
    association_policy: str
    trajectory_policy: str
    evaluation_seed: int
    step_index: int
    current_assignments: np.ndarray
    selected_actions: np.ndarray
    active_beam_mask: np.ndarray
    beam_loads: np.ndarray
    load_cap: int
    selected_quality_by_user: np.ndarray
    top_k_quality_by_user: np.ndarray
    best_vs_selected_margin_by_user: np.ndarray
    valid_beam_count_by_user: np.ndarray
    beam_rank_distance_by_user: np.ndarray
    beam_offset_distance_by_user: np.ndarray
    moved_flags: np.ndarray
    tail_user_ids: tuple[int, ...]


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


def _numeric_distribution(values: list[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "distinct": [],
            "histogram": {},
        }
    arr = np.asarray(clean, dtype=np.float64)
    rounded = [round(float(value), 12) for value in clean]
    counts = Counter(str(value) for value in rounded)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "distinct": sorted({float(value) for value in rounded}),
        "histogram": {
            key: int(value)
            for key, value in sorted(counts.items(), key=lambda item: float(item[0]))
        },
    }


def _ra_ee_06b_value(cfg: dict[str, Any]) -> dict[str, Any]:
    value = (
        cfg.get("resolved_assumptions", {})
        .get("ra_ee_06b_association_proposal_refinement", {})
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
            trajectory_families=("association-proposal-refinement",),
            evaluation_seed_set=_tuple_ints(
                calibration.get("evaluation_seed_set"),
                _tuple_ints(seeds.get("evaluation_seed_set"), DEFAULT_CALIBRATION_SEEDS),
            ),
        ),
        _BucketSpec(
            name=HELD_OUT_BUCKET,
            trajectory_families=("association-proposal-refinement-heldout",),
            evaluation_seed_set=_tuple_ints(
                held_out.get("evaluation_seed_set"),
                DEFAULT_HELD_OUT_SEEDS,
            ),
        ),
    )


def _validate_policies(name: str, policies: tuple[str, ...]) -> None:
    unsupported = sorted(set(policies) - set(SUPPORTED_ASSOCIATION_POLICIES))
    if unsupported:
        raise ValueError(f"Unsupported RA-EE-06B {name}: {unsupported!r}")


def _settings_from_config(cfg: dict[str, Any]) -> _RAEE06BSettings:
    gate = _ra_ee_06b_value(cfg)
    power = _power_surface_value(cfg)
    seeds = get_seeds(cfg)

    levels_raw = gate.get(
        "codebook_levels_w",
        power.get("power_codebook_levels_w", [0.5, 0.75, 1.0, 1.5, 2.0]),
    )
    levels = tuple(float(level) for level in levels_raw)
    if not levels:
        raise ValueError("RA-EE-06B requires at least one power codebook level.")
    if tuple(sorted(levels)) != levels:
        raise ValueError(f"RA-EE-06B codebook levels must be sorted, got {levels!r}.")

    matched_control = str(
        gate.get("matched_control_association_policy", FIXED_HOLD_CURRENT)
    )
    candidates = _tuple_strings(
        gate.get("candidate_association_policies"),
        RA_EE_06B_PROPOSAL_POLICIES,
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
        raise ValueError("RA-EE-06B matched control must not also be a candidate.")

    min_active = int(gate.get("min_active_beams", 2))
    max_active = int(gate.get("max_active_beams", 8))
    if min_active < 1 or max_active < min_active:
        raise ValueError(
            f"Invalid RA-EE-06B active-beam bounds: min={min_active}, max={max_active}."
        )

    audit = _AuditSettings(
        method_label=str(gate.get("method_label", RA_EE_06B_METHOD_LABEL)),
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
    return _RAEE06BSettings(
        method_label=str(gate.get("method_label", RA_EE_06B_METHOD_LABEL)),
        implementation_sublabel=str(
            gate.get(
                "implementation_sublabel",
                "RA-EE-06B association proposal refinement / oracle distillation audit",
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
        max_two_beam_overload_step_ratio=float(
            gate.get("max_two_beam_overload_step_ratio", 0.10)
        ),
        max_moved_user_ratio=float(gate.get("max_moved_user_ratio", 0.20)),
        max_moved_user_ratio_per_step=float(
            gate.get("max_moved_user_ratio_per_step", 0.18)
        ),
        min_oracle_gap_closed_ratio=float(gate.get("min_oracle_gap_closed_ratio", 0.20)),
        quality_margin_for_move=float(gate.get("quality_margin_for_move", 0.0)),
        local_search_swap_limit=int(gate.get("local_search_swap_limit", 2)),
        trace_top_k=int(gate.get("trace_top_k", 3)),
    )


def _ra_ee_05_settings(settings: _RAEE06BSettings) -> _RAEE05Settings:
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


def _format_int_vector(values: np.ndarray) -> str:
    return " ".join(str(int(value)) for value in values.tolist())


def _format_bool_vector(values: np.ndarray) -> str:
    return " ".join("1" if bool(value) else "0" for value in values.tolist())


def _beam_set_from_assignments(assignments: np.ndarray) -> set[int]:
    return {int(value) for value in assignments.tolist() if int(value) >= 0}


def _all_valid_beams(masks: list[Any]) -> set[int]:
    return {
        int(beam_idx)
        for mask_obj in masks
        for beam_idx in _valid_indices(mask_obj).tolist()
    }


def _choose_best_valid_by_quality(state: Any, valid: np.ndarray) -> int:
    return int(
        max(valid.tolist(), key=lambda idx: (float(state.channel_quality[int(idx)]), -int(idx)))
    )


def _target_active_count(
    *,
    masks: list[Any],
    current_assignments: np.ndarray,
    settings: _RAEE06BSettings,
) -> int:
    valid_count = len(_all_valid_beams(masks))
    if valid_count <= 0:
        return 0
    current_active = len(
        {
            int(beam_idx)
            for beam_idx in current_assignments.tolist()
            if int(beam_idx) in _all_valid_beams(masks)
        }
    )
    target_from_load = int(math.ceil(len(masks) / settings.target_users_per_active_beam))
    target = max(settings.min_active_beams, target_from_load, current_active)
    target = min(settings.max_active_beams, target, valid_count)
    return max(1, target)


def _beam_proxy_scores(
    *,
    user_states: list[Any],
    masks: list[Any],
    current_assignments: np.ndarray,
    tail_user_ids: set[int] | None = None,
) -> dict[int, dict[str, float]]:
    current_counts = Counter(int(value) for value in current_assignments.tolist())
    scores: dict[int, dict[str, float]] = defaultdict(
        lambda: {
            "coverage": 0.0,
            "quality": 0.0,
            "tail_coverage": 0.0,
            "tail_quality": 0.0,
            "current_load": 0.0,
            "topk_quality": 0.0,
        }
    )
    for uid, (state, mask_obj) in enumerate(zip(user_states, masks)):
        valid = _valid_indices(mask_obj)
        if valid.size == 0:
            continue
        qualities = [(int(idx), float(state.channel_quality[int(idx)])) for idx in valid.tolist()]
        qualities.sort(key=lambda item: (-item[1], item[0]))
        topk = {idx for idx, _quality in qualities[:3]}
        for beam_idx, quality in qualities:
            entry = scores[beam_idx]
            entry["coverage"] += 1.0
            entry["quality"] += quality
            if beam_idx in topk:
                entry["topk_quality"] += quality
            if tail_user_ids is not None and uid in tail_user_ids:
                entry["tail_coverage"] += 1.0
                entry["tail_quality"] += quality
    for beam_idx, count in current_counts.items():
        scores[int(beam_idx)]["current_load"] = float(count)
    return scores


def _current_selected_quality(user_states: list[Any], current_assignments: np.ndarray) -> np.ndarray:
    values = np.zeros(len(user_states), dtype=np.float64)
    for uid, state in enumerate(user_states):
        beam_idx = int(current_assignments[uid])
        if 0 <= beam_idx < state.channel_quality.size:
            values[uid] = float(state.channel_quality[beam_idx])
    return values


def _tail_user_ids_from_current_quality(
    user_states: list[Any],
    current_assignments: np.ndarray,
) -> set[int]:
    current_quality = _current_selected_quality(user_states, current_assignments)
    if current_quality.size == 0:
        return set()
    threshold = float(np.percentile(current_quality, 5))
    return {int(uid) for uid, value in enumerate(current_quality.tolist()) if value <= threshold}


def _trim_active_set(
    active: set[int],
    *,
    target: int,
    protected: set[int],
    scores: dict[int, dict[str, float]],
) -> set[int]:
    while len(active) > target:
        removable = [beam_idx for beam_idx in active if beam_idx not in protected]
        if not removable:
            removable = list(active)
        victim = min(
            removable,
            key=lambda idx: (
                scores[idx]["tail_coverage"],
                scores[idx]["coverage"],
                scores[idx]["quality"],
                -idx,
            ),
        )
        active.remove(int(victim))
    return active


def _fill_active_set(
    active: set[int],
    *,
    target: int,
    all_valid: set[int],
    scores: dict[int, dict[str, float]],
    key_name: str,
) -> set[int]:
    while len(active) < target:
        choices = [beam_idx for beam_idx in all_valid if beam_idx not in active]
        if not choices:
            break
        selected = max(
            choices,
            key=lambda idx: (
                scores[idx].get(key_name, 0.0),
                scores[idx]["coverage"],
                scores[idx]["quality"],
                -idx,
            ),
        )
        active.add(int(selected))
    return active


def _active_set_score(
    active: set[int],
    *,
    user_states: list[Any],
    masks: list[Any],
    current_assignments: np.ndarray,
    settings: _RAEE06BSettings,
) -> float:
    if not active:
        return -math.inf
    assigned_loads = Counter()
    quality_sum = 0.0
    move_penalty = 0.0
    uncovered = 0
    for uid, (state, mask_obj) in enumerate(zip(user_states, masks)):
        valid = _valid_indices(mask_obj)
        candidates = [int(idx) for idx in valid.tolist() if int(idx) in active]
        if not candidates:
            uncovered += 1
            continue
        current = int(current_assignments[uid])
        selected = max(
            candidates,
            key=lambda idx: (
                idx == current,
                float(state.channel_quality[idx]),
                -assigned_loads[idx],
                -idx,
            ),
        )
        assigned_loads[selected] += 1
        quality_sum += float(state.channel_quality[selected])
        if selected != current:
            move_penalty += 1.0
    loads = np.asarray([assigned_loads[int(idx)] for idx in active], dtype=np.float64)
    load_gap = 0.0 if loads.size < 2 else float(np.max(loads) - np.min(loads))
    target_load = len(masks) / max(len(active), 1)
    overload = float(np.sum(np.maximum(loads - (target_load + settings.load_cap_overflow_users), 0.0)))
    return quality_sum - 0.05 * load_gap - 0.25 * overload - 0.1 * uncovered - 0.005 * move_penalty


def _local_search_active_set(
    active: set[int],
    *,
    all_valid: set[int],
    user_states: list[Any],
    masks: list[Any],
    current_assignments: np.ndarray,
    settings: _RAEE06BSettings,
) -> set[int]:
    current = set(active)
    for _ in range(max(settings.local_search_swap_limit, 0)):
        base_score = _active_set_score(
            current,
            user_states=user_states,
            masks=masks,
            current_assignments=current_assignments,
            settings=settings,
        )
        best_score = base_score
        best_set = current
        for add_idx in sorted(all_valid - current):
            for remove_idx in sorted(current):
                trial = set(current)
                trial.remove(int(remove_idx))
                trial.add(int(add_idx))
                score = _active_set_score(
                    trial,
                    user_states=user_states,
                    masks=masks,
                    current_assignments=current_assignments,
                    settings=settings,
                )
                if score > best_score + 1e-12:
                    best_score = score
                    best_set = trial
        if best_set == current:
            break
        current = best_set
    return current


def _assign_to_active_set(
    active_set: set[int],
    *,
    policy: str,
    user_states: list[Any],
    masks: list[Any],
    current_assignments: np.ndarray,
    settings: _RAEE06BSettings,
    move_budget: int,
    sticky: bool = True,
) -> np.ndarray:
    actions = np.zeros(len(masks), dtype=np.int32)
    assigned_counts: Counter[int] = Counter()
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
            candidates = [_choose_best_valid_by_quality(state, valid)]
        current = int(current_assignments[uid])
        if (
            sticky
            and current in candidates
            and assigned_counts[current] < load_cap
        ):
            selected = current
        else:
            selected = max(
                candidates,
                key=lambda idx: (
                    assigned_counts[idx] < load_cap,
                    -assigned_counts[idx],
                    float(state.channel_quality[idx]),
                    -idx,
                ),
            )
        actions[uid] = int(selected)
        assigned_counts[int(selected)] += 1

    moves = int(np.count_nonzero(actions != current_assignments))
    missing = [beam_idx for beam_idx in sorted(active_set) if assigned_counts[beam_idx] == 0]
    for beam_idx in missing:
        if moves >= move_budget:
            break
        options: list[tuple[float, int]] = []
        for uid, (state, mask_obj) in enumerate(zip(user_states, masks)):
            if not bool(mask_obj.mask[beam_idx]) or int(actions[uid]) == beam_idx:
                continue
            current_quality = (
                float(state.channel_quality[int(actions[uid])])
                if 0 <= int(actions[uid]) < state.channel_quality.size
                else 0.0
            )
            loss = current_quality - float(state.channel_quality[beam_idx])
            options.append((loss, uid))
        if not options:
            continue
        _loss, uid = min(options, key=lambda item: (item[0], item[1]))
        assigned_counts[int(actions[uid])] -= 1
        actions[uid] = int(beam_idx)
        assigned_counts[int(beam_idx)] += 1
        moves = int(np.count_nonzero(actions != current_assignments))

    return actions


def _bounded_load_balance_moves(
    actions: np.ndarray,
    *,
    active_set: set[int],
    user_states: list[Any],
    masks: list[Any],
    current_assignments: np.ndarray,
    settings: _RAEE06BSettings,
    move_budget: int,
) -> np.ndarray:
    result = actions.copy()
    if not active_set:
        return result
    target_load = int(math.ceil(len(result) / max(len(active_set), 1)))
    load_cap = target_load + settings.load_cap_overflow_users
    while int(np.count_nonzero(result != current_assignments)) < move_budget:
        counts = Counter(int(value) for value in result.tolist())
        overloaded = [
            beam_idx for beam_idx in sorted(active_set) if counts[beam_idx] > load_cap
        ]
        underloaded = [
            beam_idx for beam_idx in sorted(active_set) if counts[beam_idx] < target_load
        ]
        if not overloaded or not underloaded:
            break
        best_move: tuple[float, int, int] | None = None
        for uid, state in enumerate(user_states):
            old = int(result[uid])
            if old not in overloaded:
                continue
            old_quality = float(state.channel_quality[old])
            for new in underloaded:
                if not bool(masks[uid].mask[new]):
                    continue
                margin = float(state.channel_quality[new]) - old_quality
                if margin < settings.quality_margin_for_move:
                    continue
                candidate = (margin, uid, int(new))
                if best_move is None or candidate > best_move:
                    best_move = candidate
        if best_move is None:
            break
        _margin, uid, new_beam = best_move
        result[uid] = int(new_beam)
    return result


def _select_fixed_hold_current(
    *,
    masks: list[Any],
    current_assignments: np.ndarray,
) -> np.ndarray:
    actions = np.zeros(len(masks), dtype=np.int32)
    for uid, mask_obj in enumerate(masks):
        valid = _valid_indices(mask_obj)
        current = int(current_assignments[uid])
        if 0 <= current < mask_obj.mask.size and bool(mask_obj.mask[current]):
            actions[uid] = current
        elif valid.size:
            actions[uid] = int(valid[0])
    return actions


def _select_per_user_greedy_best_beam(
    *,
    user_states: list[Any],
    masks: list[Any],
) -> np.ndarray:
    actions = np.zeros(len(masks), dtype=np.int32)
    for uid, (state, mask_obj) in enumerate(zip(user_states, masks)):
        valid = _valid_indices(mask_obj)
        actions[uid] = _choose_best_valid_by_quality(state, valid) if valid.size else 0
    return actions


def _active_set_for_policy(
    policy: str,
    *,
    user_states: list[Any],
    masks: list[Any],
    current_assignments: np.ndarray,
    settings: _RAEE06BSettings,
) -> set[int]:
    all_valid = _all_valid_beams(masks)
    target = _target_active_count(
        masks=masks,
        current_assignments=current_assignments,
        settings=settings,
    )
    if target == 0:
        return set()
    current_active = {
        int(beam_idx)
        for beam_idx in current_assignments.tolist()
        if int(beam_idx) in all_valid
    }
    tail_users = _tail_user_ids_from_current_quality(user_states, current_assignments)
    scores = _beam_proxy_scores(
        user_states=user_states,
        masks=masks,
        current_assignments=current_assignments,
        tail_user_ids=tail_users,
    )

    if policy == P05_SLACK_AWARE_ACTIVE_SET:
        active = set(current_active)
        protected = set()
        for uid in tail_users:
            valid = _valid_indices(masks[uid])
            if valid.size == 0:
                continue
            best = _choose_best_valid_by_quality(user_states[uid], valid)
            active.add(best)
            protected.add(best)
        active = _fill_active_set(
            active,
            target=target,
            all_valid=all_valid,
            scores=scores,
            key_name="tail_quality",
        )
        return _trim_active_set(active, target=target, protected=protected, scores=scores)

    if policy == POWER_RESPONSE_AWARE_LOAD_BALANCE:
        active = set(current_active)
        active = _fill_active_set(
            active,
            target=target,
            all_valid=all_valid,
            scores=scores,
            key_name="coverage",
        )
        return _trim_active_set(active, target=target, protected=set(), scores=scores)

    if policy == ORACLE_SCORE_TOPK_ACTIVE_SET:
        ranked = sorted(
            all_valid,
            key=lambda idx: (
                scores[idx]["topk_quality"],
                scores[idx]["tail_quality"],
                scores[idx]["coverage"],
                idx in current_active,
                -idx,
            ),
            reverse=True,
        )
        return set(int(idx) for idx in ranked[:target])

    active = set(current_active)
    active = _fill_active_set(
        active,
        target=target,
        all_valid=all_valid,
        scores=scores,
        key_name="quality",
    )
    active = _trim_active_set(active, target=target, protected=set(), scores=scores)
    if policy == STICKY_ORACLE_COUNT_LOCAL_SEARCH:
        active = _local_search_active_set(
            active,
            all_valid=all_valid,
            user_states=user_states,
            masks=masks,
            current_assignments=current_assignments,
            settings=settings,
        )
    return active


def _select_actions_for_association_policy(
    policy: str,
    *,
    user_states: list[Any],
    masks: list[Any],
    current_assignments: np.ndarray,
    settings: _RAEE06BSettings,
) -> np.ndarray:
    if policy == FIXED_HOLD_CURRENT:
        return _select_fixed_hold_current(
            masks=masks,
            current_assignments=current_assignments,
        )
    if policy == PER_USER_GREEDY_BEST_BEAM:
        return _select_per_user_greedy_best_beam(user_states=user_states, masks=masks)
    if policy not in RA_EE_06B_PROPOSAL_POLICIES:
        raise ValueError(f"Unsupported RA-EE-06B association policy {policy!r}.")

    move_budget = max(1, int(math.floor(settings.max_moved_user_ratio_per_step * len(masks))))
    if policy == BOUNDED_MOVE_SERVED_SET:
        current_active = _beam_set_from_assignments(current_assignments)
        actions = _assign_to_active_set(
            current_active,
            policy=policy,
            user_states=user_states,
            masks=masks,
            current_assignments=current_assignments,
            settings=settings,
            move_budget=move_budget,
            sticky=True,
        )
        return _bounded_load_balance_moves(
            actions,
            active_set=current_active,
            user_states=user_states,
            masks=masks,
            current_assignments=current_assignments,
            settings=settings,
            move_budget=move_budget,
        )

    active_set = _active_set_for_policy(
        policy,
        user_states=user_states,
        masks=masks,
        current_assignments=current_assignments,
        settings=settings,
    )
    actions = _assign_to_active_set(
        active_set,
        policy=policy,
        user_states=user_states,
        masks=masks,
        current_assignments=current_assignments,
        settings=settings,
        move_budget=move_budget,
        sticky=policy
        in {
            STICKY_ORACLE_COUNT_LOCAL_SEARCH,
            P05_SLACK_AWARE_ACTIVE_SET,
            POWER_RESPONSE_AWARE_LOAD_BALANCE,
        },
    )
    if policy == POWER_RESPONSE_AWARE_LOAD_BALANCE:
        actions = _bounded_load_balance_moves(
            actions,
            active_set=active_set,
            user_states=user_states,
            masks=masks,
            current_assignments=current_assignments,
            settings=settings,
            move_budget=move_budget,
        )
    return actions


def _trace_for_actions(
    *,
    bucket: str,
    policy: str,
    trajectory_policy: str,
    evaluation_seed: int,
    step_index: int,
    user_states: list[Any],
    masks: list[Any],
    current_assignments: np.ndarray,
    actions: np.ndarray,
    settings: _RAEE06BSettings,
) -> _AssociationTrace:
    selected_quality = np.zeros(len(actions), dtype=np.float64)
    topk_quality = np.zeros(len(actions), dtype=np.float64)
    best_margin = np.zeros(len(actions), dtype=np.float64)
    valid_counts = np.zeros(len(actions), dtype=np.int32)
    rank_distance = np.zeros(len(actions), dtype=np.float64)
    offset_distance = np.abs(actions.astype(np.int32) - current_assignments.astype(np.int32)).astype(
        np.float64
    )
    for uid, (state, mask_obj) in enumerate(zip(user_states, masks)):
        selected = int(actions[uid])
        valid = _valid_indices(mask_obj)
        valid_counts[uid] = int(valid.size)
        if 0 <= selected < state.channel_quality.size:
            selected_quality[uid] = float(state.channel_quality[selected])
        if valid.size == 0:
            continue
        ranked = sorted(
            valid.tolist(),
            key=lambda idx: (-float(state.channel_quality[int(idx)]), int(idx)),
        )
        best = int(ranked[0])
        top = ranked[: max(settings.trace_top_k, 1)]
        topk_quality[uid] = float(
            np.mean([float(state.channel_quality[int(idx)]) for idx in top])
        )
        best_margin[uid] = float(state.channel_quality[best]) - selected_quality[uid]
        if selected in ranked:
            rank_distance[uid] = float(ranked.index(selected))
        else:
            rank_distance[uid] = float(valid.size)

    loads = np.bincount(actions.astype(np.int32), minlength=user_states[0].beam_loads.size)
    active_mask = loads > 0
    active_count = max(int(np.count_nonzero(active_mask)), 1)
    load_cap = int(math.ceil(len(actions) / active_count) + settings.load_cap_overflow_users)
    tail_threshold = float(np.percentile(selected_quality, 5)) if selected_quality.size else 0.0
    tail_ids = tuple(
        int(uid)
        for uid, value in enumerate(selected_quality.tolist())
        if float(value) <= tail_threshold
    )
    return _AssociationTrace(
        evaluation_bucket=bucket,
        association_policy=policy,
        trajectory_policy=trajectory_policy,
        evaluation_seed=int(evaluation_seed),
        step_index=int(step_index),
        current_assignments=current_assignments.astype(np.int32, copy=True),
        selected_actions=actions.astype(np.int32, copy=True),
        active_beam_mask=active_mask.astype(bool, copy=True),
        beam_loads=loads.astype(np.float64, copy=True),
        load_cap=load_cap,
        selected_quality_by_user=selected_quality,
        top_k_quality_by_user=topk_quality,
        best_vs_selected_margin_by_user=best_margin,
        valid_beam_count_by_user=valid_counts,
        beam_rank_distance_by_user=rank_distance,
        beam_offset_distance_by_user=offset_distance,
        moved_flags=(actions.astype(np.int32) != current_assignments.astype(np.int32)),
        tail_user_ids=tail_ids,
    )


def _policy_label(bucket: str, policy: str) -> str:
    return f"{bucket}:{policy}"


def _bucket_from_label(policy_label: str) -> str:
    return policy_label.split(":", 1)[0]


def _rollout_association_trajectories(
    *,
    cfg: dict[str, Any],
    settings: _RAEE06BSettings,
    max_steps: int | None,
) -> tuple[
    dict[str, dict[int, list[np.ndarray]]],
    dict[str, dict[str, Any]],
    dict[tuple[str, int, int], _AssociationTrace],
]:
    trajectories: dict[str, dict[int, list[np.ndarray]]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    trace_by_key: dict[tuple[str, int, int], _AssociationTrace] = {}
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
                    current_assignments = env.current_assignments()
                    actions = _select_actions_for_association_policy(
                        policy,
                        user_states=user_states,
                        masks=masks,
                        current_assignments=current_assignments,
                        settings=settings,
                    )
                    result = env.step(actions, env_rng)
                    rows_by_seed[int(eval_seed)].append(actions.copy())
                    trace_by_key[(label, int(eval_seed), int(result.step_index))] = (
                        _trace_for_actions(
                            bucket=spec.name,
                            policy=policy,
                            trajectory_policy=label,
                            evaluation_seed=int(eval_seed),
                            step_index=int(result.step_index),
                            user_states=user_states,
                            masks=masks,
                            current_assignments=current_assignments,
                            actions=actions,
                            settings=settings,
                        )
                    )
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
                        else "deterministic-active-set-served-set-proposal-rule"
                    )
                ),
                "candidate_role": (
                    "matched-control"
                    if policy == settings.matched_control_association_policy
                    else (
                        "diagnostic-greedy-comparator"
                        if policy in settings.diagnostic_association_policies
                        else "candidate-proposal-rule"
                    )
                ),
            }
    return trajectories, metadata, trace_by_key


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


def _safe_greedy_demotions(power_vector: np.ndarray, active_mask: np.ndarray, fixed_w: float) -> list[int]:
    return [
        int(idx)
        for idx in np.flatnonzero(active_mask).tolist()
        if float(power_vector[int(idx)]) < fixed_w - 1e-12
    ]


def _safe_greedy_step_row(
    *,
    snapshot: _StepSnapshot,
    settings: _RAEE06BSettings,
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
    demoted = _safe_greedy_demotions(
        requested,
        snapshot.active_mask,
        settings.audit.fixed_control_power_w,
    )
    lower_level_count = len(
        [
            level
            for level in settings.audit.codebook_levels_w
            if float(level) < settings.audit.fixed_control_power_w
        ]
    )
    attempted_demotions = int(np.count_nonzero(snapshot.active_mask)) * lower_level_count
    row["trajectory_policy"] = trajectory_policy
    row["source_association_policy"] = association_policy
    row["association_role"] = association_role
    row["association_action_contract"] = (
        "fixed-by-trajectory"
        if association_role == "matched-control"
        else (
            "per-user-one-hot-greedy-diagnostic"
            if association_policy == PER_USER_GREEDY_BEST_BEAM
            else "deterministic-active-set-served-set-proposal-rule"
        )
    )
    row["requested_power_vector_w"] = _format_vector(requested)
    row["effective_power_vector_w"] = row["beam_transmit_power_w"]
    row["power_allocator"] = "safe-greedy-power-allocator"
    row["power_repair_used"] = False
    row["safe_greedy_accepted_demotions"] = _format_int_vector(np.asarray(demoted, dtype=np.int32))
    row["safe_greedy_accepted_demotion_count"] = int(len(demoted))
    row["safe_greedy_rejected_demotion_count"] = max(
        0,
        int(attempted_demotions) - int(len(demoted)),
    )
    row["demoted_beams"] = row["safe_greedy_accepted_demotions"]
    row["demoted_beam_count"] = int(len(demoted))
    row["learned_association_enabled"] = False
    row["joint_association_power_training_enabled"] = False
    row["catfish_enabled"] = False
    row["rb_bandwidth_allocation_enabled"] = False
    row["active_set_size"] = row["active_beam_count"]
    row["served_set_size"] = row["served_count"]
    row["EE_denominator_w"] = row["total_active_beam_power_w"]
    return row


def _fixed_1w_proposal_step_row(
    *,
    snapshot: _StepSnapshot,
    settings: _RAEE06BSettings,
    trajectory_policy: str,
    association_policy: str,
) -> dict[str, Any]:
    powers = _power_vector_for_candidate(snapshot, settings.audit, "fixed-control")
    row = _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics=RA_EE_06B_PROPOSAL_FIXED_1W,
        selected_power_profile=f"fixed-{settings.audit.fixed_control_power_w:g}w-diagnostic",
        power_vector=powers,
        settings=settings.audit,
    )
    row["trajectory_policy"] = trajectory_policy
    row["source_association_policy"] = association_policy
    row["association_role"] = "diagnostic-proposal-fixed-1w"
    row["association_action_contract"] = "deterministic-active-set-served-set-proposal-rule"
    row["requested_power_vector_w"] = _format_vector(powers)
    row["effective_power_vector_w"] = row["beam_transmit_power_w"]
    row["power_allocator"] = "fixed-1w-per-active-beam-diagnostic"
    row["power_repair_used"] = False
    row["safe_greedy_accepted_demotions"] = ""
    row["safe_greedy_accepted_demotion_count"] = 0
    row["safe_greedy_rejected_demotion_count"] = "not-applicable"
    row["demoted_beams"] = ""
    row["demoted_beam_count"] = 0
    row["learned_association_enabled"] = False
    row["joint_association_power_training_enabled"] = False
    row["catfish_enabled"] = False
    row["rb_bandwidth_allocation_enabled"] = False
    row["active_set_size"] = row["active_beam_count"]
    row["served_set_size"] = row["served_count"]
    row["EE_denominator_w"] = row["total_active_beam_power_w"]
    return row


def _constrained_power_row(
    *,
    snapshot: _StepSnapshot,
    control_row: dict[str, Any],
    settings: _RAEE06BSettings,
    power_semantics: str,
    trajectory_policy: str,
    association_policy: str,
    association_role: str,
    association_action_contract: str,
) -> dict[str, Any]:
    row = _select_oracle_step(
        snapshot=snapshot,
        control_row=control_row,
        settings=settings.audit,
    )
    row["trajectory_policy"] = trajectory_policy
    row["power_semantics"] = power_semantics
    row["source_association_policy"] = association_policy
    row["association_role"] = association_role
    row["association_action_contract"] = association_action_contract
    row["requested_power_vector_w"] = row["beam_transmit_power_w"]
    row["effective_power_vector_w"] = row["beam_transmit_power_w"]
    row["power_allocator"] = "constrained-power-oracle-diagnostic"
    row["power_repair_used"] = False
    row["safe_greedy_accepted_demotions"] = ""
    row["safe_greedy_accepted_demotion_count"] = 0
    row["safe_greedy_rejected_demotion_count"] = "not-applicable"
    row["demoted_beams"] = ""
    row["demoted_beam_count"] = 0
    row["learned_association_enabled"] = False
    row["joint_association_power_training_enabled"] = False
    row["catfish_enabled"] = False
    row["rb_bandwidth_allocation_enabled"] = False
    row["active_set_size"] = row["active_beam_count"]
    row["served_set_size"] = row["served_count"]
    row["EE_denominator_w"] = row["total_active_beam_power_w"]
    return row


def _select_best_oracle_association_row(
    *,
    candidate_label: str,
    control_row: dict[str, Any],
    seed: int,
    step_index: int,
    snapshots_by_key: dict[tuple[str, int, int], _StepSnapshot],
    settings: _RAEE06BSettings,
    constrained_power: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    bucket = _bucket_from_label(candidate_label)
    best: dict[str, Any] | None = None
    best_policy: str | None = None
    for policy in settings.oracle_association_policies:
        proposal_label = _policy_label(bucket, policy)
        snapshot = snapshots_by_key.get((proposal_label, seed, step_index))
        if snapshot is None:
            continue
        if constrained_power:
            row = _constrained_power_row(
                snapshot=snapshot,
                control_row=control_row,
                settings=settings,
                power_semantics=RA_EE_06B_ORACLE_CONSTRAINED,
                trajectory_policy=candidate_label,
                association_policy=policy,
                association_role="diagnostic-association-power-oracle",
                association_action_contract="finite-active-set-oracle-diagnostic",
            )
        else:
            row = _safe_greedy_step_row(
                snapshot=snapshot,
                settings=settings,
                power_semantics=RA_EE_06B_ORACLE_SAFE_GREEDY,
                trajectory_policy=candidate_label,
                association_policy=policy,
                association_role="diagnostic-association-oracle-same-safe-greedy",
            )
            row["association_action_contract"] = "finite-active-set-oracle-diagnostic"
        row_ee = -math.inf if row["EE_system_bps_per_w"] is None else float(row["EE_system_bps_per_w"])
        if best is None:
            best = row
            best_policy = policy
            continue
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
    return best, best_policy


def _p05_ratio_and_slack(
    *,
    control_p05_bps: float,
    candidate_p05_bps: float,
    threshold_ratio: float,
) -> tuple[float | None, float]:
    threshold = float(threshold_ratio) * float(control_p05_bps)
    ratio = None if abs(float(control_p05_bps)) < 1e-12 else float(candidate_p05_bps) / abs(float(control_p05_bps))
    return ratio, float(candidate_p05_bps) - threshold


def _handover_burden(*, moved_user_count: int, user_step_count: int) -> dict[str, Any]:
    return {
        "moved_user_count": int(moved_user_count),
        "user_step_count": int(user_step_count),
        "moved_user_ratio": (
            0.0 if int(user_step_count) <= 0 else int(moved_user_count) / int(user_step_count)
        ),
    }


def _trace_fields(trace: _AssociationTrace | None) -> dict[str, Any]:
    if trace is None:
        return {
            "active_set_source_policy": "",
            "beam_load_distribution": {},
            "beam_load_max": None,
            "beam_load_min": None,
            "beam_load_std": None,
            "load_cap_slack": None,
            "beam_load_balance_gap": None,
            "per_user_selected_beam_quality": "",
            "per_user_top_k_quality": "",
            "selected_beam_quality_mean": None,
            "top_k_quality_mean": None,
            "best_vs_selected_margin": "",
            "best_vs_selected_margin_mean": None,
            "valid_beam_count": "",
            "valid_beam_count_mean": None,
            "current_beam": "",
            "control_beam": "",
            "oracle_beam": "",
            "selected_beam": "",
            "moved_flag": "",
            "moved_user_count": 0,
            "moved_user_ratio": 0.0,
            "beam_rank_distance": "",
            "beam_rank_distance_mean": None,
            "beam_offset_distance_proxy": "",
            "beam_offset_distance_proxy_mean": None,
            "tail_user_ids": "",
        }
    active_loads = trace.beam_loads[trace.active_beam_mask]
    load_gap = 0.0 if active_loads.size < 2 else float(np.max(active_loads) - np.min(active_loads))
    moved_count = int(np.count_nonzero(trace.moved_flags))
    return {
        "active_set_source_policy": trace.association_policy,
        "beam_load_distribution": _numeric_distribution(active_loads.tolist()),
        "beam_load_max": None if active_loads.size == 0 else float(np.max(active_loads)),
        "beam_load_min": None if active_loads.size == 0 else float(np.min(active_loads)),
        "beam_load_std": None if active_loads.size == 0 else float(np.std(active_loads)),
        "load_cap_slack": (
            None if active_loads.size == 0 else float(trace.load_cap - float(np.max(active_loads)))
        ),
        "beam_load_balance_gap": load_gap,
        "per_user_selected_beam_quality": _format_vector(trace.selected_quality_by_user),
        "per_user_top_k_quality": _format_vector(trace.top_k_quality_by_user),
        "selected_beam_quality_mean": float(np.mean(trace.selected_quality_by_user)),
        "top_k_quality_mean": float(np.mean(trace.top_k_quality_by_user)),
        "best_vs_selected_margin": _format_vector(trace.best_vs_selected_margin_by_user),
        "best_vs_selected_margin_mean": float(np.mean(trace.best_vs_selected_margin_by_user)),
        "valid_beam_count": _format_int_vector(trace.valid_beam_count_by_user),
        "valid_beam_count_mean": float(np.mean(trace.valid_beam_count_by_user)),
        "current_beam": _format_int_vector(trace.current_assignments),
        "control_beam": "",
        "oracle_beam": "",
        "selected_beam": _format_int_vector(trace.selected_actions),
        "moved_flag": _format_bool_vector(trace.moved_flags),
        "moved_user_count": moved_count,
        "moved_user_ratio": moved_count / max(trace.selected_actions.size, 1),
        "beam_rank_distance": _format_vector(trace.beam_rank_distance_by_user),
        "beam_rank_distance_mean": float(np.mean(trace.beam_rank_distance_by_user)),
        "beam_offset_distance_proxy": _format_vector(trace.beam_offset_distance_by_user),
        "beam_offset_distance_proxy_mean": float(np.mean(trace.beam_offset_distance_by_user)),
        "tail_user_ids": _format_int_vector(np.asarray(trace.tail_user_ids, dtype=np.int32)),
    }


def _add_trace_and_comparison_fields(
    row: dict[str, Any],
    *,
    trace: _AssociationTrace | None,
    control_trace: _AssociationTrace | None,
    oracle_trace: _AssociationTrace | None,
    matched_row: dict[str, Any],
    candidate_row: dict[str, Any],
    oracle_row: dict[str, Any] | None,
    settings: _RAEE06BSettings,
    diagnostic_only: bool,
) -> dict[str, Any]:
    row.update(_trace_fields(trace))
    if control_trace is not None:
        row["control_beam"] = _format_int_vector(control_trace.selected_actions)
    if oracle_trace is not None:
        row["oracle_beam"] = _format_int_vector(oracle_trace.selected_actions)
    own_p05 = float(row["throughput_p05_user_step_bps"])
    control_p05 = float(matched_row["throughput_p05_user_step_bps"])
    p05_ratio, p05_slack = _p05_ratio_and_slack(
        control_p05_bps=control_p05,
        candidate_p05_bps=own_p05,
        threshold_ratio=settings.audit.p05_min_ratio_vs_control,
    )
    row["p05_throughput_control_bps"] = control_p05
    row["p05_throughput_candidate_bps"] = float(candidate_row["throughput_p05_user_step_bps"])
    row["p05_throughput_oracle_bps"] = (
        None if oracle_row is None else float(oracle_row["throughput_p05_user_step_bps"])
    )
    row["p05_ratio_vs_matched_control"] = p05_ratio
    row["p05_slack_to_0_95_threshold_bps"] = p05_slack
    row["EE_delta_vs_matched_control"] = (
        None
        if row["EE_system_bps_per_w"] is None
        or matched_row["EE_system_bps_per_w"] is None
        else float(row["EE_system_bps_per_w"]) - float(matched_row["EE_system_bps_per_w"])
    )
    row["oracle_selected_association_policy"] = (
        "" if oracle_trace is None else oracle_trace.association_policy
    )
    row["oracle_power_profile"] = (
        "" if oracle_row is None else str(oracle_row["selected_power_profile"])
    )
    if diagnostic_only:
        row["accepted_flag"] = False
        row["rejection_reason"] = "diagnostic-only"
        return row

    reasons: list[str] = []
    if row["EE_delta_vs_matched_control"] is None or float(row["EE_delta_vs_matched_control"]) <= 0.0:
        reasons.append("nonpositive-ee-delta")
    if p05_ratio is None or float(p05_ratio) < settings.audit.p05_min_ratio_vs_control:
        reasons.append("p05-ratio-below-threshold")
    if float(row["served_ratio"]) < float(matched_row["served_ratio"]) + settings.audit.served_ratio_min_delta_vs_control:
        reasons.append("served-ratio-drop")
    if float(row["outage_ratio"]) > float(matched_row["outage_ratio"]) + settings.audit.outage_ratio_max_delta_vs_control:
        reasons.append("outage-ratio-increase")
    if bool(row["budget_violation"]):
        reasons.append("budget-violation")
    if bool(row["per_beam_power_violation"]):
        reasons.append("per-beam-power-violation")
    if bool(row["inactive_beam_nonzero_power"]):
        reasons.append("inactive-power-nonzero")
    if int(row["active_beam_count"]) <= 1:
        reasons.append("one-active-beam-collapse")
    if int(row["active_beam_count"]) < settings.min_active_beams or int(row["active_beam_count"]) > settings.max_active_beams:
        reasons.append("active-set-bounds")
    if float(row["moved_user_ratio"]) > settings.max_moved_user_ratio_per_step:
        reasons.append("step-handover-burden")
    row["accepted_flag"] = not reasons
    row["rejection_reason"] = "accepted" if not reasons else ";".join(reasons)
    return row


def _evaluation_rows(
    *,
    snapshots: list[_StepSnapshot],
    traces_by_key: dict[tuple[str, int, int], _AssociationTrace],
    settings: _RAEE06BSettings,
    include_oracle: bool,
    include_fixed_1w_diagnostic: bool,
    include_matched_fixed_constrained_isolation: bool,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[float]]]:
    by_key = _snapshot_index(snapshots)
    rows: list[dict[str, Any]] = []
    user_throughputs_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    oracle_cache: dict[tuple[str, int, int, str], tuple[dict[str, Any], str | None]] = {}
    fixed_constrained_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
    candidate_policies = (
        *settings.candidate_association_policies,
        *settings.diagnostic_association_policies,
    )

    for spec in settings.bucket_specs:
        control_label = _policy_label(spec.name, settings.matched_control_association_policy)
        for proposal_policy in candidate_policies:
            candidate_label = _policy_label(spec.name, proposal_policy)
            is_greedy = proposal_policy in settings.diagnostic_association_policies
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
                    matched = _safe_greedy_step_row(
                        snapshot=control_snapshot,
                        settings=settings,
                        power_semantics=RA_EE_06B_MATCHED_CONTROL,
                        trajectory_policy=candidate_label,
                        association_policy=settings.matched_control_association_policy,
                        association_role="matched-control",
                    )
                    candidate_semantics = (
                        RA_EE_06B_GREEDY_DIAGNOSTIC
                        if is_greedy
                        else RA_EE_06B_CANDIDATE
                    )
                    candidate = _safe_greedy_step_row(
                        snapshot=candidate_snapshot,
                        settings=settings,
                        power_semantics=candidate_semantics,
                        trajectory_policy=candidate_label,
                        association_policy=proposal_policy,
                        association_role=(
                            "diagnostic-greedy-comparator"
                            if is_greedy
                            else "candidate-proposal-rule"
                        ),
                    )
                    oracle_safe: dict[str, Any] | None = None
                    oracle_safe_policy: str | None = None
                    oracle_constrained: dict[str, Any] | None = None
                    oracle_constrained_policy: str | None = None
                    if include_oracle and not is_greedy:
                        safe_cache_key = (spec.name, int(seed), step_index, "safe")
                        if safe_cache_key not in oracle_cache:
                            safe_row, safe_policy = _select_best_oracle_association_row(
                                candidate_label=candidate_label,
                                control_row=matched,
                                seed=int(seed),
                                step_index=step_index,
                                snapshots_by_key=by_key,
                                settings=settings,
                                constrained_power=False,
                            )
                            if safe_row is not None:
                                oracle_cache[safe_cache_key] = (
                                    dict(safe_row),
                                    safe_policy,
                                )
                        if safe_cache_key in oracle_cache:
                            cached_row, oracle_safe_policy = oracle_cache[safe_cache_key]
                            oracle_safe = dict(cached_row)
                            oracle_safe["trajectory_policy"] = candidate_label

                        constrained_cache_key = (
                            spec.name,
                            int(seed),
                            step_index,
                            "constrained",
                        )
                        if constrained_cache_key not in oracle_cache:
                            constrained_row, constrained_policy = (
                                _select_best_oracle_association_row(
                                    candidate_label=candidate_label,
                                    control_row=matched,
                                    seed=int(seed),
                                    step_index=step_index,
                                    snapshots_by_key=by_key,
                                    settings=settings,
                                    constrained_power=True,
                                )
                            )
                            if constrained_row is not None:
                                oracle_cache[constrained_cache_key] = (
                                    dict(constrained_row),
                                    constrained_policy,
                                )
                        if constrained_cache_key in oracle_cache:
                            cached_row, oracle_constrained_policy = oracle_cache[
                                constrained_cache_key
                            ]
                            oracle_constrained = dict(cached_row)
                            oracle_constrained["trajectory_policy"] = candidate_label
                    comparison_oracle = oracle_constrained or oracle_safe
                    comparison_policy = oracle_constrained_policy or oracle_safe_policy
                    oracle_trace = (
                        None
                        if comparison_policy is None
                        else traces_by_key.get(
                            (
                                _policy_label(spec.name, comparison_policy),
                                int(seed),
                                step_index,
                            )
                        )
                    )
                    control_trace = traces_by_key.get((control_label, int(seed), step_index))
                    candidate_trace = traces_by_key.get((candidate_label, int(seed), step_index))

                    step_rows: list[tuple[dict[str, Any], _AssociationTrace | None, bool]] = [
                        (matched, control_trace, True),
                        (candidate, candidate_trace, is_greedy),
                    ]
                    if include_fixed_1w_diagnostic and not is_greedy:
                        fixed_1w = _fixed_1w_proposal_step_row(
                            snapshot=candidate_snapshot,
                            settings=settings,
                            trajectory_policy=candidate_label,
                            association_policy=proposal_policy,
                        )
                        step_rows.append((fixed_1w, candidate_trace, True))
                    if include_oracle and oracle_safe is not None:
                        step_rows.append((oracle_safe, oracle_trace, True))
                    if include_oracle and oracle_constrained is not None:
                        step_rows.append((oracle_constrained, oracle_trace, True))
                    if include_matched_fixed_constrained_isolation and not is_greedy:
                        fixed_cache_key = (spec.name, int(seed), step_index)
                        if fixed_cache_key not in fixed_constrained_cache:
                            fixed_constrained_cache[fixed_cache_key] = dict(
                                _constrained_power_row(
                                    snapshot=control_snapshot,
                                    control_row=matched,
                                    settings=settings,
                                    power_semantics=(
                                        RA_EE_06B_MATCHED_FIXED_CONSTRAINED
                                    ),
                                    trajectory_policy=candidate_label,
                                    association_policy=(
                                        settings.matched_control_association_policy
                                    ),
                                    association_role=(
                                        "diagnostic-matched-fixed-constrained-power-isolation"
                                    ),
                                    association_action_contract="fixed-by-trajectory",
                                )
                            )
                        fixed_constrained = dict(fixed_constrained_cache[fixed_cache_key])
                        fixed_constrained["trajectory_policy"] = candidate_label
                        step_rows.append((fixed_constrained, control_trace, True))

                    for row, trace, diagnostic_only in step_rows:
                        row["evaluation_bucket"] = spec.name
                        row["matched_control_association_policy"] = (
                            settings.matched_control_association_policy
                        )
                        row["candidate_association_policy"] = proposal_policy
                        _add_trace_and_comparison_fields(
                            row,
                            trace=trace,
                            control_trace=control_trace,
                            oracle_trace=oracle_trace,
                            matched_row=matched,
                            candidate_row=candidate,
                            oracle_row=comparison_oracle,
                            settings=settings,
                            diagnostic_only=diagnostic_only
                            or row["power_semantics"] != RA_EE_06B_CANDIDATE,
                        )
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
    moved_by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    moved_ratio_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    load_gap_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    source_policy_by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    rejection_by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    p05_ratio_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    accepted_step_by_key: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for row in step_rows:
        key = (str(row["trajectory_policy"]), str(row["power_semantics"]))
        vectors_by_key[key].append(str(row["effective_power_vector_w"]))
        active_sets_by_key[key].append(int(row["active_set_size"]))
        moved_by_key[key].append(int(row["moved_user_count"]))
        moved_ratio_by_key[key].append(float(row["moved_user_ratio"]))
        if row["beam_load_balance_gap"] is not None:
            load_gap_by_key[key].append(float(row["beam_load_balance_gap"]))
        source_policy_by_key[key].append(str(row["source_association_policy"]))
        rejection_by_key[key].append(str(row["rejection_reason"]))
        if row["p05_ratio_vs_matched_control"] is not None:
            p05_ratio_by_key[key].append(float(row["p05_ratio_vs_matched_control"]))
        accepted_step_by_key[key].append(bool(row["accepted_flag"]))
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
                "oracle_selected_association_policy": row[
                    "oracle_selected_association_policy"
                ],
                "oracle_power_profile": row["oracle_power_profile"],
            },
        )

    for summary in summaries:
        key = (str(summary["trajectory_policy"]), str(summary["power_semantics"]))
        summary.update(metadata_by_key[key])
        summary["selected_power_vector_distribution"] = _categorical_distribution(
            vectors_by_key[key]
        )
        summary["active_set_size_distribution"] = _numeric_distribution(
            [float(value) for value in active_sets_by_key[key]]
        )
        summary["moved_user_count_distribution"] = _numeric_distribution(
            [float(value) for value in moved_by_key[key]]
        )
        summary["moved_user_ratio_distribution"] = _numeric_distribution(
            moved_ratio_by_key[key]
        )
        summary["moved_user_count_total"] = int(sum(moved_by_key[key]))
        summary["user_step_count"] = int(
            summary["step_count"] * max(summary["served_ratio"] + summary["outage_ratio"], 1)
        )
        summary["handover_burden"] = _handover_burden(
            moved_user_count=int(sum(moved_by_key[key])),
            user_step_count=max(int(summary["step_count"]) * 100, 1),
        )
        summary["beam_load_balance_gap_distribution"] = _numeric_distribution(
            load_gap_by_key[key]
        )
        summary["source_association_policy_distribution"] = _categorical_distribution(
            source_policy_by_key[key]
        )
        summary["rejection_reason_distribution"] = _categorical_distribution(
            rejection_by_key[key]
        )
        summary["p05_ratio_vs_matched_control_distribution"] = _numeric_distribution(
            p05_ratio_by_key[key]
        )
        summary["accepted_step_count"] = int(sum(accepted_step_by_key[key]))
        summary["accepted_step_ratio"] = (
            int(sum(accepted_step_by_key[key])) / max(len(accepted_step_by_key[key]), 1)
        )
        summary["active_set_contract_is_proposal_rule"] = (
            summary["association_action_contract"]
            == "deterministic-active-set-served-set-proposal-rule"
        )
        summary["two_beam_overload_step_ratio"] = 0.0
    return summaries


def _pct_delta(reference: float | None, value: float | None) -> float | None:
    if reference is None or value is None or abs(float(reference)) < 1e-12:
        return None
    return float((float(value) - float(reference)) / abs(float(reference)))


def _guardrail_result(
    *,
    candidate: dict[str, Any],
    control: dict[str, Any],
    oracle: dict[str, Any] | None,
    settings: _RAEE06BSettings,
) -> dict[str, Any]:
    p05_threshold = settings.audit.p05_min_ratio_vs_control * float(
        control["throughput_p05_user_step_bps"]
    )
    p05_ratio, p05_slack = _p05_ratio_and_slack(
        control_p05_bps=float(control["throughput_p05_user_step_bps"]),
        candidate_p05_bps=float(candidate["throughput_p05_user_step_bps"]),
        threshold_ratio=settings.audit.p05_min_ratio_vs_control,
    )
    p05_pass = p05_ratio is not None and p05_ratio >= settings.audit.p05_min_ratio_vs_control
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
    oracle_gap_closed = None
    oracle_delta = None
    if (
        oracle is not None
        and oracle["EE_system_aggregate_bps_per_w"] is not None
        and control["EE_system_aggregate_bps_per_w"] is not None
        and ee_delta is not None
    ):
        oracle_delta = float(oracle["EE_system_aggregate_bps_per_w"]) - float(
            control["EE_system_aggregate_bps_per_w"]
        )
        if oracle_delta > 1e-12:
            oracle_gap_closed = float(ee_delta) / oracle_delta
    noncollapsed = (
        float(candidate["one_active_beam_step_ratio"])
        <= settings.max_one_active_beam_ratio_for_acceptance
    )
    two_beam_ok = (
        float(candidate.get("two_beam_overload_step_ratio", 0.0))
        <= settings.max_two_beam_overload_step_ratio
    )
    denominator_varies = bool(candidate["denominator_varies_in_eval"])
    handover_burden = candidate["handover_burden"]
    handover_pass = float(handover_burden["moved_user_ratio"]) <= settings.max_moved_user_ratio
    oracle_gap_pass = (
        oracle_gap_closed is not None
        and oracle_gap_closed >= settings.min_oracle_gap_closed_ratio
    )
    reasons: list[str] = []
    if ee_delta is None or ee_delta <= 0.0:
        reasons.append("nonpositive-ee-delta")
    if not p05_pass:
        reasons.append("p05-ratio-below-threshold")
    if not served_pass:
        reasons.append("served-ratio-drop")
    if not outage_pass:
        reasons.append("outage-ratio-increase")
    if not budget_pass:
        reasons.append("budget-violation")
    if not per_beam_pass:
        reasons.append("per-beam-power-violation")
    if not inactive_pass:
        reasons.append("inactive-power-nonzero")
    if not noncollapsed:
        reasons.append("one-active-beam-collapse")
    if not two_beam_ok:
        reasons.append("two-beam-overload-collapse")
    if not denominator_varies:
        reasons.append("denominator-fixed")
    if not handover_pass:
        reasons.append("handover-burden")
    if not oracle_gap_pass:
        reasons.append("oracle-gap-not-meaningfully-closed")
    accepted = not reasons
    return {
        "evaluation_bucket": candidate["evaluation_bucket"],
        "trajectory_policy": candidate["trajectory_policy"],
        "candidate_association_policy": candidate["candidate_association_policy"],
        "power_semantics": candidate["power_semantics"],
        "matched_control_power_semantics": control["power_semantics"],
        "diagnostic_oracle_power_semantics": (
            None if oracle is None else oracle["power_semantics"]
        ),
        "EE_system_delta_vs_matched_control": ee_delta,
        "EE_system_pct_delta_vs_matched_control": _pct_delta(
            control["EE_system_aggregate_bps_per_w"],
            candidate["EE_system_aggregate_bps_per_w"],
        ),
        "throughput_mean_pct_delta_vs_matched_control": _pct_delta(
            control["throughput_mean_user_step_bps"],
            candidate["throughput_mean_user_step_bps"],
        ),
        "throughput_p05_ratio_vs_matched_control": p05_ratio,
        "p05_threshold_bps": p05_threshold,
        "p05_slack_to_0_95_threshold_bps": p05_slack,
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
        "two_beam_overload_guardrail_pass": two_beam_ok,
        "denominator_varies_in_eval": denominator_varies,
        "handover_burden_guardrail_pass": handover_pass,
        "moved_user_count": handover_burden["moved_user_count"],
        "moved_user_ratio": handover_burden["moved_user_ratio"],
        "oracle_delta_vs_matched_control": oracle_delta,
        "oracle_gap_closed_ratio": oracle_gap_closed,
        "oracle_gap_closed_guardrail_pass": oracle_gap_pass,
        "active_set_contract_is_proposal_rule": bool(
            candidate["active_set_contract_is_proposal_rule"]
        ),
        "accepted": accepted,
        "rejection_reason": "accepted" if accepted else ";".join(reasons),
    }


def _build_guardrail_checks(
    *,
    summaries: list[dict[str, Any]],
    settings: _RAEE06BSettings,
) -> list[dict[str, Any]]:
    by_policy: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for summary in summaries:
        by_policy[str(summary["trajectory_policy"])][str(summary["power_semantics"])] = summary
    checks: list[dict[str, Any]] = []
    for _policy, rows in sorted(by_policy.items()):
        control = rows.get(RA_EE_06B_MATCHED_CONTROL)
        if control is None:
            continue
        oracle = rows.get(RA_EE_06B_ORACLE_CONSTRAINED) or rows.get(
            RA_EE_06B_ORACLE_SAFE_GREEDY
        )
        for semantics in (
            RA_EE_06B_CANDIDATE,
            RA_EE_06B_GREEDY_DIAGNOSTIC,
            RA_EE_06B_ORACLE_SAFE_GREEDY,
            RA_EE_06B_ORACLE_CONSTRAINED,
        ):
            candidate = rows.get(semantics)
            if candidate is None:
                continue
            checks.append(
                _guardrail_result(
                    candidate=candidate,
                    control=control,
                    oracle=oracle,
                    settings=settings,
                )
            )
    return checks


def _ranking_checks(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_policy: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summaries:
        by_policy[str(row["trajectory_policy"])][str(row["power_semantics"])] = row
    checks: list[dict[str, Any]] = []
    compared_semantics = (RA_EE_06B_MATCHED_CONTROL, RA_EE_06B_CANDIDATE)
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
        control = rows.get(RA_EE_06B_MATCHED_CONTROL)
        candidate = rows.get(RA_EE_06B_CANDIDATE)
        oracle_safe = rows.get(RA_EE_06B_ORACLE_SAFE_GREEDY)
        oracle_constrained = rows.get(RA_EE_06B_ORACLE_CONSTRAINED)
        if control is None or candidate is None:
            continue
        for oracle in (oracle_safe, oracle_constrained):
            if oracle is None:
                continue
            control_ee = control["EE_system_aggregate_bps_per_w"]
            candidate_ee = candidate["EE_system_aggregate_bps_per_w"]
            oracle_ee = oracle["EE_system_aggregate_bps_per_w"]
            candidate_delta = (
                None if candidate_ee is None or control_ee is None else float(candidate_ee) - float(control_ee)
            )
            oracle_delta = (
                None if oracle_ee is None or control_ee is None else float(oracle_ee) - float(control_ee)
            )
            diagnostics.append(
                {
                    "evaluation_bucket": candidate["evaluation_bucket"],
                    "trajectory_policy": policy,
                    "candidate_association_policy": candidate[
                        "candidate_association_policy"
                    ],
                    "candidate_power_semantics": RA_EE_06B_CANDIDATE,
                    "oracle_power_semantics": oracle["power_semantics"],
                    "oracle_is_diagnostic_only": True,
                    "control_EE_system_aggregate_bps_per_w": control_ee,
                    "candidate_EE_system_aggregate_bps_per_w": candidate_ee,
                    "oracle_EE_system_aggregate_bps_per_w": oracle_ee,
                    "candidate_EE_delta_vs_control": candidate_delta,
                    "oracle_EE_delta_vs_control": oracle_delta,
                    "oracle_gap_closed_ratio": (
                        None
                        if candidate_delta is None
                        or oracle_delta is None
                        or oracle_delta <= 1e-12
                        else candidate_delta / oracle_delta
                    ),
                    "oracle_EE_gap_vs_candidate_bps_per_w": (
                        None if candidate_ee is None or oracle_ee is None else float(oracle_ee) - float(candidate_ee)
                    ),
                }
            )
    return diagnostics


def _bucket_results(
    *,
    settings: _RAEE06BSettings,
    summaries: list[dict[str, Any]],
    guardrail_checks: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    candidate_by_policy = {
        str(row["trajectory_policy"]): row
        for row in summaries
        if row["power_semantics"] == RA_EE_06B_CANDIDATE
    }
    guardrail_by_policy = {
        str(row["trajectory_policy"]): row
        for row in guardrail_checks
        if row["power_semantics"] == RA_EE_06B_CANDIDATE
    }
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
            and float(candidate_by_policy[label].get("two_beam_overload_step_ratio", 0.0))
            <= settings.max_two_beam_overload_step_ratio
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
        denominator_varies = bool(accepted) and all(
            bool(candidate_by_policy[label]["denominator_varies_in_eval"])
            for label in accepted
        )
        handover_bounded = bool(accepted) and all(
            bool(guardrail_by_policy[label]["handover_burden_guardrail_pass"])
            for label in accepted
        )
        oracle_gap_closed = bool(accepted) and all(
            bool(guardrail_by_policy[label]["oracle_gap_closed_guardrail_pass"])
            for label in accepted
        )
        rejection_reasons = {
            label: guardrail_by_policy.get(label, {}).get("rejection_reason", "missing")
            for label in present
        }
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
            "rejection_reasons": rejection_reasons,
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
            "handover_burden_bounded_for_accepted": handover_bounded,
            "oracle_gap_closed_for_accepted": oracle_gap_closed,
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
                    and handover_bounded
                    and oracle_gap_closed
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
        row for row in summaries if row["power_semantics"] == RA_EE_06B_CANDIDATE
    ]
    candidate_guardrails = [
        row for row in guardrail_checks if row["power_semantics"] == RA_EE_06B_CANDIDATE
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
        bool(row.get("active_set_contract_is_proposal_rule"))
        for row in candidate_summaries
    )
    proof_flags = {
        "held_out_bucket_exists_and_reported_separately": bool(held_out),
        "offline_trace_export_only": True,
        "deterministic_proposal_refinement_only": True,
        "learned_hierarchical_RL_disabled": learned_disabled,
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
        "oracle_diagnostic_only": include_oracle,
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
        "one_active_beam_or_two_beam_overload_collapse_avoided": not bool(
            held_out.get("one_active_beam_collapse_dominates")
        ),
        "handover_burden_bounded_for_accepted_held_out": bool(
            held_out.get("handover_burden_bounded_for_accepted")
        ),
        "candidate_closes_meaningful_oracle_gap": bool(
            held_out.get("oracle_gap_closed_for_accepted")
        ),
        "scalar_reward_success_basis": False,
        "per_user_EE_credit_success_basis": False,
        "physical_energy_saving_claim": False,
        "hobs_optimizer_claim": False,
        "full_RA_EE_MODQN_claim": False,
    }
    stop_conditions = {
        "held_out_bucket_missing": not bool(held_out),
        "proposal_gains_require_constrained_oracle_power": not bool(
            held_out.get("majority_noncollapsed_positive_EE_delta")
        )
        and any(
            row["evaluation_bucket"] == HELD_OUT_BUCKET
            and row["power_semantics"] == RA_EE_06B_ORACLE_CONSTRAINED
            and float(row["EE_system_delta_vs_matched_control"] or 0.0) > 0.0
            for row in guardrail_checks
        ),
        "held_out_EE_delta_negative_or_concentrated": not bool(
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
        "proposal_collapses_to_one_active_beam_or_two_beam_overload": bool(
            held_out.get("one_active_beam_collapse_dominates")
        ),
        "reassignment_churn_is_source_of_apparent_gain": bool(
            held_out.get("accepted_candidate_count")
        )
        and not bool(held_out.get("handover_burden_bounded_for_accepted")),
        "budget_or_inactive_power_violations": not no_power_violations,
        "learned_RL_or_joint_training_or_catfish_or_RB_added": (
            not learned_disabled or not joint_disabled
        ),
        "frozen_baseline_mutated": False,
        "oracle_used_as_candidate_claim": False,
    }
    required_true = (
        "held_out_bucket_exists_and_reported_separately",
        "offline_trace_export_only",
        "deterministic_proposal_refinement_only",
        "learned_hierarchical_RL_disabled",
        "joint_association_power_training_disabled",
        "catfish_disabled",
        "multi_catfish_disabled",
        "rb_bandwidth_allocation_disabled",
        "active_set_served_set_proposal_contract",
        "matched_control_uses_same_power_allocator",
        "safe_greedy_allocator_retained",
        "oracle_diagnostic_only",
        "majority_noncollapsed_held_out_positive_EE_delta",
        "majority_noncollapsed_held_out_accepted",
        "held_out_gains_not_concentrated_in_one_policy",
        "p05_throughput_guardrail_pass_for_accepted_held_out",
        "served_ratio_does_not_drop_for_accepted_held_out",
        "outage_ratio_does_not_increase_for_accepted_held_out",
        "zero_budget_per_beam_inactive_power_violations",
        "denominator_varies_for_accepted_held_out",
        "one_active_beam_or_two_beam_overload_collapse_avoided",
        "handover_burden_bounded_for_accepted_held_out",
        "candidate_closes_meaningful_oracle_gap",
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
        "ra_ee_06b_decision": decision,
        "proof_flags": proof_flags,
        "stop_conditions": stop_conditions,
        "allowed_claim": (
            "PASS only means deterministic offline association proposal rules "
            "passed against matched fixed association plus the same safe-greedy "
            "power allocator. It is not learned hierarchical RA-EE-MODQN."
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
        "moved_user_count_total",
        "moved_user_count_distribution",
        "active_beam_count_distribution",
        "active_set_size_distribution",
        "selected_power_profile_distribution",
        "selected_power_vector_distribution",
        "total_active_beam_power_w_distribution",
        "denominator_varies_in_eval",
        "one_active_beam_step_ratio",
        "budget_violations",
        "per_beam_power_violations",
        "inactive_beam_nonzero_power_step_count",
        "p05_ratio_vs_matched_control_distribution",
        "rejection_reason_distribution",
        "throughput_vs_EE_system_correlation",
    )
    return [{field: row[field] for field in fields} for row in summaries]


def _review_lines(summary: dict[str, Any]) -> list[str]:
    decision = summary["ra_ee_06b_decision"]
    proof = summary["proof_flags"]
    held_out = summary["bucket_results"][HELD_OUT_BUCKET]
    lines = [
        "# RA-EE-06B Association Proposal Refinement Review",
        "",
        "Offline oracle-trace export and deterministic association proposal "
        "refinement only. No learned hierarchical RL, joint association + power "
        "training, Catfish, multi-Catfish, RB / bandwidth allocation, HOBS "
        "optimizer claim, physical energy-saving claim, or frozen baseline "
        "mutation was performed.",
        "",
        "## Protocol",
        "",
        f"- method label: `{summary['protocol']['method_label']}`",
        f"- implementation sublabel: `{summary['protocol']['implementation_sublabel']}`",
        f"- config: `{summary['inputs']['config_path']}`",
        f"- artifact namespace: `{summary['inputs']['output_dir']}`",
        f"- matched control: `{summary['protocol']['matched_control']}`",
        f"- primary candidate: `{summary['protocol']['candidate']}`",
        f"- diagnostic oracle: `{summary['protocol']['oracle_same_safe_greedy']}`",
        f"- upper bound oracle: `{summary['protocol']['oracle_upper_bound']}`",
        f"- proposal rules: `{summary['protocol']['candidate_association_policies']}`",
        "",
        "## Held-Out Gate",
        "",
        f"- noncollapsed candidates: `{held_out['noncollapsed_candidate_policies']}`",
        f"- positive EE delta candidates: `{held_out['positive_EE_delta_candidate_policies']}`",
        f"- accepted candidates: `{held_out['accepted_candidate_policies']}`",
        f"- rejection reasons: `{held_out['rejection_reasons']}`",
        f"- majority noncollapsed positive EE delta: `{held_out['majority_noncollapsed_positive_EE_delta']}`",
        f"- majority noncollapsed accepted: `{held_out['majority_noncollapsed_accepted']}`",
        f"- gains not concentrated in one policy: `{held_out['gains_not_concentrated_in_one_policy']}`",
        f"- QoS guardrails pass for accepted: `{held_out['qos_guardrails_pass_for_accepted']}`",
        f"- denominator varies for accepted: `{held_out['denominator_varies_for_accepted']}`",
        f"- handover burden bounded: `{held_out['handover_burden_bounded_for_accepted']}`",
        f"- oracle gap closed: `{held_out['oracle_gap_closed_for_accepted']}`",
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
            f"- RA-EE-06B decision: `{decision}`",
            f"- allowed claim: {summary['decision_detail']['allowed_claim']}",
            "",
            "## Forbidden Claims",
            "",
        ]
    )
    for claim in summary["forbidden_claims_still_active"]:
        lines.append(f"- {claim}")
    return lines


def export_ra_ee_06b_association_proposal_refinement(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    calibration_seed_set: tuple[int, ...] | None = None,
    held_out_seed_set: tuple[int, ...] | None = None,
    candidate_association_policies: tuple[str, ...] | None = None,
    max_steps: int | None = None,
    include_oracle: bool = True,
    include_fixed_1w_diagnostic: bool = True,
    include_matched_fixed_constrained_isolation: bool = True,
) -> dict[str, Any]:
    """Export RA-EE-06B proposal-refinement trace and summary artifacts."""
    if max_steps is not None and max_steps < 1:
        raise ValueError(f"max_steps must be >= 1 when provided, got {max_steps!r}")
    cfg = load_training_yaml(config_path)
    env = build_environment(cfg)
    if env.power_surface_config.hobs_power_surface_mode != (
        HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK
    ):
        raise ValueError("RA-EE-06B config must opt into the power-codebook surface.")

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
    run_settings = _RAEE06BSettings(
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
        max_two_beam_overload_step_ratio=settings.max_two_beam_overload_step_ratio,
        max_moved_user_ratio=settings.max_moved_user_ratio,
        max_moved_user_ratio_per_step=settings.max_moved_user_ratio_per_step,
        min_oracle_gap_closed_ratio=settings.min_oracle_gap_closed_ratio,
        quality_margin_for_move=settings.quality_margin_for_move,
        local_search_swap_limit=settings.local_search_swap_limit,
        trace_top_k=settings.trace_top_k,
    )
    _validate_policies("candidate policies", run_settings.candidate_association_policies)

    trajectories, association_metadata, traces_by_key = _rollout_association_trajectories(
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
        traces_by_key=traces_by_key,
        settings=run_settings,
        include_oracle=include_oracle,
        include_fixed_1w_diagnostic=include_fixed_1w_diagnostic,
        include_matched_fixed_constrained_isolation=(
            include_matched_fixed_constrained_isolation
        ),
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
    )
    decision_detail = _build_decision(
        summaries=summaries,
        guardrail_checks=guardrail_checks,
        bucket_results=bucket_results,
        include_oracle=include_oracle,
    )

    out_dir = Path(output_dir)
    trace_csv = _write_csv(
        out_dir / "ra_ee_06b_oracle_trace.csv",
        step_rows,
        fieldnames=list(step_rows[0].keys()),
    )
    compact_rows = _compact_summary_rows(summaries)
    summary_csv = _write_csv(
        out_dir / "ra_ee_06b_candidate_summary.csv",
        compact_rows,
        fieldnames=list(compact_rows[0].keys()),
    )
    guardrail_csv = _write_csv(
        out_dir / "ra_ee_06b_guardrail_checks.csv",
        guardrail_checks,
        fieldnames=list(guardrail_checks[0].keys()) if guardrail_checks else [],
    )

    protocol = {
        "phase": "RA-EE-06B",
        "method_label": run_settings.method_label,
        "method_family": "RA-EE association proposal refinement",
        "implementation_sublabel": run_settings.implementation_sublabel,
        "training": "none; offline trace export and deterministic proposal rules only",
        "learned_hierarchical_RL": "disabled",
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
        "association_action_contract": "deterministic-active-set-served-set-proposal-rule",
        "matched_control": RA_EE_06B_MATCHED_CONTROL,
        "candidate": RA_EE_06B_CANDIDATE,
        "proposal_fixed_1w_diagnostic": RA_EE_06B_PROPOSAL_FIXED_1W,
        "greedy_diagnostic": RA_EE_06B_GREEDY_DIAGNOSTIC,
        "oracle_same_safe_greedy": RA_EE_06B_ORACLE_SAFE_GREEDY,
        "oracle_upper_bound": RA_EE_06B_ORACLE_CONSTRAINED,
        "matched_fixed_constrained_isolation": RA_EE_06B_MATCHED_FIXED_CONSTRAINED,
        "oracle_diagnostic_only": include_oracle,
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
        "max_two_beam_overload_step_ratio": (
            run_settings.max_two_beam_overload_step_ratio
        ),
        "max_moved_user_ratio": run_settings.max_moved_user_ratio,
        "max_moved_user_ratio_per_step": run_settings.max_moved_user_ratio_per_step,
        "min_oracle_gap_closed_ratio": run_settings.min_oracle_gap_closed_ratio,
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
        "oracle_trace_schema_fields": list(step_rows[0].keys()),
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
        "ra_ee_06b_decision": decision_detail["ra_ee_06b_decision"],
        "remaining_blockers": [
            "This is offline proposal-rule and oracle-distillation evidence only.",
            "No learned hierarchical association or full RA-EE-MODQN policy exists.",
            "No joint association + power training exists.",
            "No RB / bandwidth allocation is included.",
            "The association and constrained-power oracles are diagnostic only.",
        ],
        "forbidden_claims_still_active": [
            "Do not call RA-EE-06B full RA-EE-MODQN.",
            "Do not claim learned hierarchical RL or learned association effectiveness.",
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
        out_dir / "ra_ee_06b_association_proposal_refinement_summary.json",
        summary,
    )
    review_path = out_dir / "review.md"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text("\n".join(_review_lines(summary)) + "\n")

    return {
        "ra_ee_06b_association_proposal_refinement_summary": summary_path,
        "ra_ee_06b_candidate_summary_csv": summary_csv,
        "ra_ee_06b_guardrail_checks_csv": guardrail_csv,
        "ra_ee_06b_oracle_trace": trace_csv,
        "review_md": review_path,
        "summary": summary,
    }


__all__ = [
    "BOUNDED_MOVE_SERVED_SET",
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT_DIR",
    "ORACLE_SCORE_TOPK_ACTIVE_SET",
    "P05_SLACK_AWARE_ACTIVE_SET",
    "POWER_RESPONSE_AWARE_LOAD_BALANCE",
    "RA_EE_06B_CANDIDATE",
    "RA_EE_06B_GREEDY_DIAGNOSTIC",
    "RA_EE_06B_MATCHED_CONTROL",
    "RA_EE_06B_METHOD_LABEL",
    "RA_EE_06B_ORACLE_CONSTRAINED",
    "RA_EE_06B_ORACLE_SAFE_GREEDY",
    "RA_EE_06B_PROPOSAL_FIXED_1W",
    "RA_EE_06B_PROPOSAL_POLICIES",
    "STICKY_ORACLE_COUNT_LOCAL_SEARCH",
    "_handover_burden",
    "_p05_ratio_and_slack",
    "_select_actions_for_association_policy",
    "_settings_from_config",
    "export_ra_ee_06b_association_proposal_refinement",
]
