"""RA-EE-07 constrained-power allocator distillation gate.

This module is an offline replay gate.  It keeps association fixed for the
primary comparison, evaluates deployable non-oracle power allocators against
the matched RA-EE-04/05 safe-greedy allocator, and keeps association/oracle
rows diagnostic-only.
"""

from __future__ import annotations

import csv
import itertools
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
    CALIBRATION_TRAJECTORIES,
    DEFAULT_CALIBRATION_SEEDS,
    DEFAULT_HELD_OUT_SEEDS,
    HELD_OUT_BUCKET,
    HELD_OUT_TRAJECTORIES,
    _BucketSpec,
    _RAEE05Settings,
    _rollout_fixed_association_trajectories,
    _safe_greedy_power_vector,
)
from .ra_ee_06_association_counterfactual_oracle import FIXED_HOLD_CURRENT
from .ra_ee_06b_association_proposal_refinement import (
    BOUNDED_MOVE_SERVED_SET,
    ORACLE_SCORE_TOPK_ACTIVE_SET,
    P05_SLACK_AWARE_ACTIVE_SET,
    POWER_RESPONSE_AWARE_LOAD_BALANCE,
    RA_EE_06B_PROPOSAL_POLICIES,
    STICKY_ORACLE_COUNT_LOCAL_SEARCH,
    _RAEE06BSettings,
    _policy_label,
    _rollout_association_trajectories,
)


DEFAULT_CONFIG = "configs/ra-ee-07-constrained-power-allocator-distillation.resolved.yaml"
DEFAULT_OUTPUT_DIR = "artifacts/ra-ee-07-constrained-power-allocator-distillation"

RA_EE_07_METHOD_LABEL = "RA-EE constrained-power allocator distillation gate"
RA_EE_07_FIXED_1W_DIAGNOSTIC = "matched-fixed-association+fixed-1w-diagnostic"
RA_EE_07_SAFE_GREEDY_CONTROL = (
    "matched-fixed-association+safe-greedy-power-allocator"
)
RA_EE_07_DEPLOYABLE = (
    "matched-fixed-association+deployable-stronger-power-allocator"
)
RA_EE_07_CONSTRAINED_ORACLE = (
    "matched-fixed-association+constrained-power-oracle-isolation"
)
RA_EE_07_ASSOC_PROPOSAL_DEPLOYABLE = (
    "association-proposal+deployable-stronger-power-allocator-diagnostic"
)
RA_EE_07_ASSOC_ORACLE_CONSTRAINED = (
    "association-oracle+constrained-power-oracle-upper-bound"
)

P05_SLACK_TRIM_TAIL_PROTECT = "p05-slack-aware-trim-tail-protect-boost"
BOUNDED_LOCAL_SEARCH = "bounded-local-search-codebook"
FINITE_CODEBOOK_DP = "finite-codebook-dp-knapsack"
DETERMINISTIC_HYBRID = "deterministic-hybrid-runtime"

DEPLOYABLE_ALLOCATORS = (
    P05_SLACK_TRIM_TAIL_PROTECT,
    BOUNDED_LOCAL_SEARCH,
    FINITE_CODEBOOK_DP,
    DETERMINISTIC_HYBRID,
)


@dataclass(frozen=True)
class _AllocatorResult:
    allocator_label: str
    power_vector: np.ndarray
    selected_power_profile: str
    accepted_move_count: int
    rejected_move_count: int
    rejection_reason: str
    selected_from: str
    evaluated_profile_count: int


@dataclass(frozen=True)
class _RAEE07Settings:
    method_label: str
    implementation_sublabel: str
    audit: _AuditSettings
    fixed_bucket_specs: tuple[_BucketSpec, ...]
    diagnostic_bucket_specs: tuple[_BucketSpec, ...]
    deployable_allocators: tuple[str, ...]
    primary_deployable_allocator: str
    candidate_max_demoted_beams: int
    candidate_step_p05_guardrail_margin: float
    local_search_max_moves: int
    p05_trim_max_moves: int
    dp_max_profile_count: int
    min_oracle_gap_closed_ratio: float
    association_diagnostic_policies: tuple[str, ...]
    min_active_beams: int
    max_active_beams: int
    target_users_per_active_beam: int
    load_cap_overflow_users: int
    max_moved_user_ratio_per_step: float
    max_moved_user_ratio: float
    max_one_active_beam_ratio_for_acceptance: float
    max_two_beam_overload_step_ratio: float
    diagnostic_max_steps: int | None


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


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            ordered.append(key)
    return ordered


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


def _ra_ee_07_value(cfg: dict[str, Any]) -> dict[str, Any]:
    value = (
        cfg.get("resolved_assumptions", {})
        .get("ra_ee_07_constrained_power_allocator_distillation", {})
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


def _fixed_bucket_specs_from_config(
    gate: dict[str, Any],
    seeds: dict[str, Any],
) -> tuple[_BucketSpec, ...]:
    buckets = gate.get("fixed_association_buckets", gate.get("evaluation_buckets", {}))
    if not isinstance(buckets, dict):
        buckets = {}
    calibration = buckets.get(CALIBRATION_BUCKET, {})
    if not isinstance(calibration, dict):
        calibration = {}
    held_out = buckets.get(HELD_OUT_BUCKET, buckets.get("held_out", {}))
    if not isinstance(held_out, dict):
        held_out = {}

    specs = (
        _BucketSpec(
            name=CALIBRATION_BUCKET,
            trajectory_families=_tuple_strings(
                calibration.get("trajectory_families"),
                CALIBRATION_TRAJECTORIES,
            ),
            evaluation_seed_set=_tuple_ints(
                calibration.get("evaluation_seed_set"),
                _tuple_ints(seeds.get("evaluation_seed_set"), DEFAULT_CALIBRATION_SEEDS),
            ),
        ),
        _BucketSpec(
            name=HELD_OUT_BUCKET,
            trajectory_families=_tuple_strings(
                held_out.get("trajectory_families"),
                HELD_OUT_TRAJECTORIES,
            ),
            evaluation_seed_set=_tuple_ints(
                held_out.get("evaluation_seed_set"),
                DEFAULT_HELD_OUT_SEEDS,
            ),
        ),
    )
    supported = set(CALIBRATION_TRAJECTORIES + HELD_OUT_TRAJECTORIES)
    for spec in specs:
        unsupported = sorted(set(spec.trajectory_families) - supported)
        if unsupported:
            raise ValueError(
                f"Unsupported RA-EE-07 fixed trajectories in {spec.name!r}: {unsupported!r}"
            )
    return specs


def _diagnostic_bucket_specs_from_config(
    gate: dict[str, Any],
    seeds: dict[str, Any],
) -> tuple[_BucketSpec, ...]:
    buckets = gate.get("diagnostic_association_buckets", {})
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


def _settings_from_config(cfg: dict[str, Any]) -> _RAEE07Settings:
    gate = _ra_ee_07_value(cfg)
    power = _power_surface_value(cfg)
    seeds = get_seeds(cfg)

    levels_raw = gate.get(
        "codebook_levels_w",
        power.get("power_codebook_levels_w", [0.5, 0.75, 1.0, 1.5, 2.0]),
    )
    levels = tuple(float(level) for level in levels_raw)
    if not levels:
        raise ValueError("RA-EE-07 requires at least one power codebook level.")
    if tuple(sorted(levels)) != levels:
        raise ValueError(f"RA-EE-07 codebook levels must be sorted, got {levels!r}.")

    deployable = _tuple_strings(
        gate.get("deployable_allocator_candidates"),
        DEPLOYABLE_ALLOCATORS,
    )
    unsupported = sorted(set(deployable) - set(DEPLOYABLE_ALLOCATORS))
    if unsupported:
        raise ValueError(f"Unsupported RA-EE-07 deployable allocators: {unsupported!r}")
    primary = str(gate.get("primary_deployable_allocator", DETERMINISTIC_HYBRID))
    if primary not in deployable:
        raise ValueError(
            f"RA-EE-07 primary allocator {primary!r} must be in candidates {deployable!r}."
        )

    association_policies = _tuple_strings(
        gate.get("diagnostic_association_policies"),
        RA_EE_06B_PROPOSAL_POLICIES,
    )
    unsupported_assoc = sorted(set(association_policies) - set(RA_EE_06B_PROPOSAL_POLICIES))
    if unsupported_assoc:
        raise ValueError(
            f"Unsupported RA-EE-07 diagnostic association policies: {unsupported_assoc!r}"
        )

    audit = _AuditSettings(
        method_label=str(gate.get("method_label", RA_EE_07_METHOD_LABEL)),
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
    return _RAEE07Settings(
        method_label=str(gate.get("method_label", RA_EE_07_METHOD_LABEL)),
        implementation_sublabel=str(
            gate.get(
                "implementation_sublabel",
                "RA-EE-07 constrained-power allocator distillation gate",
            )
        ),
        audit=audit,
        fixed_bucket_specs=_fixed_bucket_specs_from_config(gate, seeds),
        diagnostic_bucket_specs=_diagnostic_bucket_specs_from_config(gate, seeds),
        deployable_allocators=deployable,
        primary_deployable_allocator=primary,
        candidate_max_demoted_beams=int(gate.get("candidate_max_demoted_beams", 3)),
        candidate_step_p05_guardrail_margin=float(
            gate.get("candidate_step_p05_guardrail_margin", 0.005)
        ),
        local_search_max_moves=int(gate.get("local_search_max_moves", 8)),
        p05_trim_max_moves=int(gate.get("p05_trim_max_moves", 6)),
        dp_max_profile_count=int(gate.get("dp_max_profile_count", 20000)),
        min_oracle_gap_closed_ratio=float(gate.get("min_oracle_gap_closed_ratio", 0.20)),
        association_diagnostic_policies=association_policies,
        min_active_beams=int(gate.get("min_active_beams", 2)),
        max_active_beams=int(gate.get("max_active_beams", 8)),
        target_users_per_active_beam=int(gate.get("target_users_per_active_beam", 16)),
        load_cap_overflow_users=int(gate.get("load_cap_overflow_users", 2)),
        max_moved_user_ratio_per_step=float(
            gate.get("max_moved_user_ratio_per_step", 0.18)
        ),
        max_moved_user_ratio=float(gate.get("max_moved_user_ratio", 0.20)),
        max_one_active_beam_ratio_for_acceptance=float(
            gate.get("max_one_active_beam_ratio_for_acceptance", 0.25)
        ),
        max_two_beam_overload_step_ratio=float(
            gate.get("max_two_beam_overload_step_ratio", 0.10)
        ),
        diagnostic_max_steps=(
            None
            if gate.get("diagnostic_max_steps") in (None, "none")
            else int(gate.get("diagnostic_max_steps"))
        ),
    )


def _ra_ee_05_settings(settings: _RAEE07Settings) -> _RAEE05Settings:
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


def _ra_ee_06b_settings(settings: _RAEE07Settings) -> _RAEE06BSettings:
    return _RAEE06BSettings(
        method_label=settings.method_label,
        implementation_sublabel=settings.implementation_sublabel,
        audit=settings.audit,
        bucket_specs=settings.diagnostic_bucket_specs,
        matched_control_association_policy=FIXED_HOLD_CURRENT,
        candidate_association_policies=settings.association_diagnostic_policies,
        diagnostic_association_policies=(),
        oracle_association_policies=settings.association_diagnostic_policies,
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
        quality_margin_for_move=0.0,
        local_search_swap_limit=2,
        trace_top_k=3,
    )


def _active_indices(snapshot: _StepSnapshot) -> list[int]:
    return [int(idx) for idx in np.flatnonzero(snapshot.active_mask).tolist()]


def _format_int_vector(values: list[int] | np.ndarray) -> str:
    array = np.asarray(values, dtype=np.int32)
    return " ".join(str(int(value)) for value in array.tolist())


def _lower_or_equal_levels(levels: tuple[float, ...], limit: float) -> tuple[float, ...]:
    values = tuple(float(level) for level in levels if float(level) <= float(limit) + 1e-12)
    return values if values else (float(levels[0]),)


def _adjacent_levels(levels: tuple[float, ...], value: float) -> tuple[float, ...]:
    rounded = [float(level) for level in levels]
    current = min(range(len(rounded)), key=lambda idx: abs(rounded[idx] - float(value)))
    candidates: list[float] = []
    if current > 0:
        candidates.append(rounded[current - 1])
    if current + 1 < len(rounded):
        candidates.append(rounded[current + 1])
    return tuple(candidates)


def _qos_guardrails_pass_vs_control(
    *,
    candidate: dict[str, Any],
    control: dict[str, Any],
    settings: _RAEE07Settings,
) -> bool:
    return (
        float(candidate["throughput_p05_user_step_bps"])
        >= settings.audit.p05_min_ratio_vs_control
        * float(control["throughput_p05_user_step_bps"])
        and float(candidate["served_ratio"])
        >= float(control["served_ratio"]) + settings.audit.served_ratio_min_delta_vs_control
        and float(candidate["outage_ratio"])
        <= float(control["outage_ratio"]) + settings.audit.outage_ratio_max_delta_vs_control
    )


def _power_contract_pass(row: dict[str, Any]) -> bool:
    return (
        not bool(row["budget_violation"])
        and not bool(row["per_beam_power_violation"])
        and not bool(row["inactive_beam_nonzero_power"])
    )


def _candidate_step_passes(
    *,
    row: dict[str, Any],
    matched_safe_row: dict[str, Any],
    settings: _RAEE07Settings,
) -> bool:
    return _power_contract_pass(row) and _qos_guardrails_pass_vs_control(
        candidate=row,
        control=matched_safe_row,
        settings=settings,
    )


def _row_ee(row: dict[str, Any]) -> float:
    value = row["EE_system_bps_per_w"]
    return -math.inf if value is None else float(value)


def _evaluate_candidate_vector(
    snapshot: _StepSnapshot,
    vector: np.ndarray,
    *,
    power_semantics: str,
    selected_power_profile: str,
    settings: _RAEE07Settings,
) -> dict[str, Any]:
    return _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics=power_semantics,
        selected_power_profile=selected_power_profile,
        power_vector=vector,
        settings=settings.audit,
    )


def _best_row(
    rows: list[tuple[dict[str, Any], np.ndarray, str]],
) -> tuple[dict[str, Any], np.ndarray, str]:
    return max(
        rows,
        key=lambda item: (
            _row_ee(item[0]),
            float(item[0]["sum_user_throughput_bps"]),
            item[2],
        ),
    )


def _tail_user_ids(row: dict[str, Any]) -> tuple[int, ...]:
    throughputs = row.get("_user_throughputs")
    if throughputs is None:
        return ()
    values = np.asarray(throughputs, dtype=np.float64)
    if values.size == 0:
        return ()
    threshold = float(np.percentile(values, 5))
    return tuple(int(idx) for idx, value in enumerate(values.tolist()) if value <= threshold)


def _p05_trim_tail_protect_allocator(
    snapshot: _StepSnapshot,
    matched_safe_row: dict[str, Any],
    safe_vector: np.ndarray,
    settings: _RAEE07Settings,
) -> _AllocatorResult:
    active = _active_indices(snapshot)
    current = safe_vector.copy()
    current_row = _evaluate_candidate_vector(
        snapshot,
        current,
        power_semantics=P05_SLACK_TRIM_TAIL_PROTECT,
        selected_power_profile=f"{P05_SLACK_TRIM_TAIL_PROTECT}:start",
        settings=settings,
    )
    tail_ids = _tail_user_ids(matched_safe_row)
    protected = {
        int(snapshot.assignments[uid])
        for uid in tail_ids
        if 0 <= int(snapshot.assignments[uid]) < snapshot.beam_loads.size
    }
    accepted = 0
    rejected = 0
    lower_levels = [level for level in settings.audit.codebook_levels_w if level < settings.audit.fixed_control_power_w]

    for _ in range(max(settings.p05_trim_max_moves, 0)):
        trials: list[tuple[dict[str, Any], np.ndarray, str]] = []
        candidates = sorted(
            active,
            key=lambda idx: (
                idx in protected,
                float(snapshot.beam_loads[idx]),
                int(idx),
            ),
        )
        for beam_idx in candidates:
            current_level = float(current[beam_idx])
            possible = [level for level in lower_levels if level < current_level - 1e-12]
            if not possible:
                continue
            trial = current.copy()
            trial[beam_idx] = max(possible)
            profile = (
                f"{P05_SLACK_TRIM_TAIL_PROTECT}:beam-{beam_idx}-to-{trial[beam_idx]:g}w"
            )
            row = _evaluate_candidate_vector(
                snapshot,
                trial,
                power_semantics=P05_SLACK_TRIM_TAIL_PROTECT,
                selected_power_profile=profile,
                settings=settings,
            )
            if (
                _candidate_step_passes(
                    row=row,
                    matched_safe_row=matched_safe_row,
                    settings=settings,
                )
                and _row_ee(row) > _row_ee(current_row) + 1e-12
            ):
                trials.append((row, trial, profile))
            else:
                rejected += 1
        if not trials:
            break
        current_row, current, profile = _best_row(trials)
        accepted += 1

    return _AllocatorResult(
        allocator_label=P05_SLACK_TRIM_TAIL_PROTECT,
        power_vector=current,
        selected_power_profile=(
            f"{P05_SLACK_TRIM_TAIL_PROTECT}:{_format_vector(current[snapshot.active_mask])}"
        ),
        accepted_move_count=accepted,
        rejected_move_count=rejected,
        rejection_reason="accepted" if accepted else "no-safe-trim-improved-ee",
        selected_from=P05_SLACK_TRIM_TAIL_PROTECT,
        evaluated_profile_count=accepted + rejected + 1,
    )


def _bounded_local_search_allocator(
    snapshot: _StepSnapshot,
    matched_safe_row: dict[str, Any],
    safe_vector: np.ndarray,
    settings: _RAEE07Settings,
) -> _AllocatorResult:
    current = safe_vector.copy()
    current_row = _evaluate_candidate_vector(
        snapshot,
        current,
        power_semantics=BOUNDED_LOCAL_SEARCH,
        selected_power_profile=f"{BOUNDED_LOCAL_SEARCH}:start",
        settings=settings,
    )
    accepted = 0
    rejected = 0
    for _ in range(max(settings.local_search_max_moves, 0)):
        trials: list[tuple[dict[str, Any], np.ndarray, str]] = []
        for beam_idx in _active_indices(snapshot):
            for level in _adjacent_levels(settings.audit.codebook_levels_w, current[beam_idx]):
                if abs(float(level) - float(current[beam_idx])) <= 1e-12:
                    continue
                trial = current.copy()
                trial[beam_idx] = float(level)
                profile = f"{BOUNDED_LOCAL_SEARCH}:beam-{beam_idx}-to-{float(level):g}w"
                row = _evaluate_candidate_vector(
                    snapshot,
                    trial,
                    power_semantics=BOUNDED_LOCAL_SEARCH,
                    selected_power_profile=profile,
                    settings=settings,
                )
                if (
                    _candidate_step_passes(
                        row=row,
                        matched_safe_row=matched_safe_row,
                        settings=settings,
                    )
                    and _row_ee(row) > _row_ee(current_row) + 1e-12
                ):
                    trials.append((row, trial, profile))
                else:
                    rejected += 1
        if not trials:
            break
        current_row, current, _profile = _best_row(trials)
        accepted += 1

    return _AllocatorResult(
        allocator_label=BOUNDED_LOCAL_SEARCH,
        power_vector=current,
        selected_power_profile=f"{BOUNDED_LOCAL_SEARCH}:{_format_vector(current[snapshot.active_mask])}",
        accepted_move_count=accepted,
        rejected_move_count=rejected,
        rejection_reason="accepted" if accepted else "no-local-move-improved-ee",
        selected_from=BOUNDED_LOCAL_SEARCH,
        evaluated_profile_count=accepted + rejected + 1,
    )


def _dp_candidate_profiles(
    snapshot: _StepSnapshot,
    safe_vector: np.ndarray,
    settings: _RAEE07Settings,
) -> list[np.ndarray]:
    active = _active_indices(snapshot)
    if not active:
        return [safe_vector.copy()]
    levels_by_beam = [
        _lower_or_equal_levels(
            settings.audit.codebook_levels_w,
            max(settings.audit.fixed_control_power_w, float(safe_vector[beam_idx])),
        )
        for beam_idx in active
    ]
    count = 1
    for levels in levels_by_beam:
        count *= len(levels)
    if count > settings.dp_max_profile_count:
        ranked = sorted(active, key=lambda idx: (-float(snapshot.beam_loads[idx]), idx))
        active = ranked[: max(1, int(math.log(settings.dp_max_profile_count, 3)))]
        levels_by_beam = [
            _lower_or_equal_levels(
                settings.audit.codebook_levels_w,
                max(settings.audit.fixed_control_power_w, float(safe_vector[beam_idx])),
            )
            for beam_idx in active
        ]

    profiles: list[np.ndarray] = []
    for combo in itertools.product(*levels_by_beam):
        vector = safe_vector.copy()
        for beam_idx, level in zip(active, combo):
            vector[beam_idx] = float(level)
        profiles.append(vector)
    return profiles


def _finite_codebook_dp_allocator(
    snapshot: _StepSnapshot,
    matched_safe_row: dict[str, Any],
    safe_vector: np.ndarray,
    settings: _RAEE07Settings,
) -> _AllocatorResult:
    best_vector = safe_vector.copy()
    best_row = _evaluate_candidate_vector(
        snapshot,
        best_vector,
        power_semantics=FINITE_CODEBOOK_DP,
        selected_power_profile=f"{FINITE_CODEBOOK_DP}:safe-start",
        settings=settings,
    )
    rejected = 0
    evaluated = 0
    for vector in _dp_candidate_profiles(snapshot, safe_vector, settings):
        evaluated += 1
        row = _evaluate_candidate_vector(
            snapshot,
            vector,
            power_semantics=FINITE_CODEBOOK_DP,
            selected_power_profile=f"{FINITE_CODEBOOK_DP}:{_format_vector(vector[snapshot.active_mask])}",
            settings=settings,
        )
        if not _candidate_step_passes(
            row=row,
            matched_safe_row=matched_safe_row,
            settings=settings,
        ):
            rejected += 1
            continue
        if (
            _row_ee(row) > _row_ee(best_row) + 1e-12
            or (
                abs(_row_ee(row) - _row_ee(best_row)) <= 1e-12
                and float(row["sum_user_throughput_bps"])
                > float(best_row["sum_user_throughput_bps"])
            )
        ):
            best_row = row
            best_vector = vector
    accepted = int(np.count_nonzero(np.abs(best_vector - safe_vector) > 1e-12))
    return _AllocatorResult(
        allocator_label=FINITE_CODEBOOK_DP,
        power_vector=best_vector,
        selected_power_profile=f"{FINITE_CODEBOOK_DP}:{_format_vector(best_vector[snapshot.active_mask])}",
        accepted_move_count=accepted,
        rejected_move_count=rejected,
        rejection_reason="accepted" if accepted else "safe-vector-remained-best",
        selected_from=FINITE_CODEBOOK_DP,
        evaluated_profile_count=evaluated,
    )


def _deployable_allocator_results(
    snapshot: _StepSnapshot,
    matched_safe_row: dict[str, Any],
    safe_vector: np.ndarray,
    settings: _RAEE07Settings,
) -> dict[str, _AllocatorResult]:
    results: dict[str, _AllocatorResult] = {}
    if P05_SLACK_TRIM_TAIL_PROTECT in settings.deployable_allocators:
        results[P05_SLACK_TRIM_TAIL_PROTECT] = _p05_trim_tail_protect_allocator(
            snapshot,
            matched_safe_row,
            safe_vector,
            settings,
        )
    if BOUNDED_LOCAL_SEARCH in settings.deployable_allocators:
        results[BOUNDED_LOCAL_SEARCH] = _bounded_local_search_allocator(
            snapshot,
            matched_safe_row,
            safe_vector,
            settings,
        )
    if FINITE_CODEBOOK_DP in settings.deployable_allocators:
        results[FINITE_CODEBOOK_DP] = _finite_codebook_dp_allocator(
            snapshot,
            matched_safe_row,
            safe_vector,
            settings,
        )
    if DETERMINISTIC_HYBRID in settings.deployable_allocators:
        selectable: list[tuple[dict[str, Any], _AllocatorResult]] = []
        for label, result in results.items():
            row = _evaluate_candidate_vector(
                snapshot,
                result.power_vector,
                power_semantics=label,
                selected_power_profile=result.selected_power_profile,
                settings=settings,
            )
            if _candidate_step_passes(
                row=row,
                matched_safe_row=matched_safe_row,
                settings=settings,
            ):
                selectable.append((row, result))
        if selectable:
            _row, selected = max(
                selectable,
                key=lambda item: (
                    _row_ee(item[0]),
                    float(item[0]["sum_user_throughput_bps"]),
                    item[1].allocator_label,
                ),
            )
            hybrid = _AllocatorResult(
                allocator_label=DETERMINISTIC_HYBRID,
                power_vector=selected.power_vector.copy(),
                selected_power_profile=(
                    f"{DETERMINISTIC_HYBRID}:selected-{selected.allocator_label}:"
                    f"{_format_vector(selected.power_vector[snapshot.active_mask])}"
                ),
                accepted_move_count=selected.accepted_move_count,
                rejected_move_count=sum(item.rejected_move_count for item in results.values()),
                rejection_reason="accepted",
                selected_from=selected.allocator_label,
                evaluated_profile_count=sum(
                    item.evaluated_profile_count for item in results.values()
                ),
            )
        else:
            hybrid = _AllocatorResult(
                allocator_label=DETERMINISTIC_HYBRID,
                power_vector=safe_vector.copy(),
                selected_power_profile=f"{DETERMINISTIC_HYBRID}:fallback-safe-greedy",
                accepted_move_count=0,
                rejected_move_count=sum(item.rejected_move_count for item in results.values()),
                rejection_reason="no-deployable-candidate-passed-runtime-guardrails",
                selected_from="safe-greedy-fallback",
                evaluated_profile_count=sum(
                    item.evaluated_profile_count for item in results.values()
                ),
            )
        results[DETERMINISTIC_HYBRID] = hybrid
    return results


def _power_delta_fields(vector: np.ndarray, baseline: np.ndarray, active_mask: np.ndarray) -> dict[str, Any]:
    boosted = [
        int(idx)
        for idx in np.flatnonzero(active_mask).tolist()
        if float(vector[int(idx)]) > float(baseline[int(idx)]) + 1e-12
    ]
    demoted = [
        int(idx)
        for idx in np.flatnonzero(active_mask).tolist()
        if float(vector[int(idx)]) < float(baseline[int(idx)]) - 1e-12
    ]
    return {
        "boosted_beams": _format_int_vector(boosted),
        "boosted_beam_count": int(len(boosted)),
        "demoted_beams": _format_int_vector(demoted),
        "demoted_beam_count": int(len(demoted)),
    }


def _load_stats(snapshot: _StepSnapshot) -> dict[str, Any]:
    active_loads = snapshot.beam_loads[snapshot.active_mask]
    if active_loads.size == 0:
        return {
            "beam_load_distribution": {},
            "beam_load_max": None,
            "beam_load_min": None,
            "beam_load_std": None,
            "beam_load_balance_gap": None,
        }
    return {
        "beam_load_distribution": _numeric_distribution(active_loads.tolist()),
        "beam_load_max": float(np.max(active_loads)),
        "beam_load_min": float(np.min(active_loads)),
        "beam_load_std": float(np.std(active_loads)),
        "beam_load_balance_gap": (
            0.0 if active_loads.size < 2 else float(np.max(active_loads) - np.min(active_loads))
        ),
    }


def _decorated_power_row(
    snapshot: _StepSnapshot,
    *,
    row: dict[str, Any],
    requested_power_vector: np.ndarray,
    baseline_power_vector: np.ndarray,
    bucket: str,
    association_policy: str,
    association_role: str,
    association_action_contract: str,
    allocator_label: str,
    allocator_result: _AllocatorResult | None,
    diagnostic_only: bool,
    primary_candidate: bool,
) -> dict[str, Any]:
    row["evaluation_bucket"] = bucket
    row["association_policy"] = association_policy
    row["source_association_policy"] = association_policy
    row["association_role"] = association_role
    row["association_action_contract"] = association_action_contract
    row["active_set_size"] = row["active_beam_count"]
    row["served_set_size"] = row["served_count"]
    row["requested_power_vector_w"] = _format_vector(requested_power_vector)
    row["effective_power_vector_w"] = row["beam_transmit_power_w"]
    row["selected_profile"] = row["selected_power_profile"]
    row["allocator_label"] = allocator_label
    row["selected_allocator_candidate"] = (
        "" if allocator_result is None else allocator_result.selected_from
    )
    row["accepted_allocator_move_count"] = (
        0 if allocator_result is None else allocator_result.accepted_move_count
    )
    row["rejected_allocator_move_count"] = (
        0 if allocator_result is None else allocator_result.rejected_move_count
    )
    row["allocator_rejection_reason"] = (
        "diagnostic-only" if allocator_result is None else allocator_result.rejection_reason
    )
    row["evaluated_allocator_profile_count"] = (
        1 if allocator_result is None else allocator_result.evaluated_profile_count
    )
    row.update(_power_delta_fields(requested_power_vector, baseline_power_vector, snapshot.active_mask))
    row.update(_load_stats(snapshot))
    row["per_user_quality"] = _format_vector(snapshot.unit_snr_by_user)
    row["valid_beam_count"] = ""
    row["valid_beam_count_mean"] = None
    row["tail_user_ids"] = _format_int_vector(_tail_user_ids(row))
    row["total_active_power"] = row["total_active_beam_power_w"]
    row["EE_denominator_w"] = row["total_active_beam_power_w"]
    row["oracle_profile"] = ""
    row["oracle_gap_bps_per_w"] = None
    row["oracle_gap_closed_ratio"] = None
    row["candidate_regret_bps_per_w"] = None
    row["diagnostic_only"] = diagnostic_only
    row["primary_candidate"] = primary_candidate
    row["accepted_flag"] = False
    row["rejection_reason"] = "diagnostic-only" if diagnostic_only else ""
    row["learned_association_enabled"] = False
    row["learned_hierarchical_RL_enabled"] = False
    row["joint_association_power_training_enabled"] = False
    row["catfish_enabled"] = False
    row["multi_catfish_enabled"] = False
    row["rb_bandwidth_allocation_enabled"] = False
    row["oracle_labels_used_for_runtime_decision"] = False
    row["future_outcomes_used_for_runtime_decision"] = False
    row["held_out_answers_used_for_runtime_decision"] = False
    row["scalar_reward_success_basis"] = False
    row["physical_energy_saving_claim"] = False
    row["hobs_optimizer_claim"] = False
    return row


def _fixed_1w_row(
    snapshot: _StepSnapshot,
    settings: _RAEE07Settings,
    *,
    bucket: str,
    association_policy: str,
    association_role: str,
    association_action_contract: str,
) -> tuple[dict[str, Any], np.ndarray]:
    vector = _power_vector_for_candidate(snapshot, settings.audit, "fixed-control")
    row = _evaluate_candidate_vector(
        snapshot,
        vector,
        power_semantics=RA_EE_07_FIXED_1W_DIAGNOSTIC,
        selected_power_profile=f"fixed-{settings.audit.fixed_control_power_w:g}w-diagnostic",
        settings=settings,
    )
    return (
        _decorated_power_row(
            snapshot,
            row=row,
            requested_power_vector=vector,
            baseline_power_vector=vector,
            bucket=bucket,
            association_policy=association_policy,
            association_role=association_role,
            association_action_contract=association_action_contract,
            allocator_label="fixed-1w-per-active-beam-diagnostic",
            allocator_result=None,
            diagnostic_only=True,
            primary_candidate=False,
        ),
        vector,
    )


def _safe_greedy_row(
    snapshot: _StepSnapshot,
    settings: _RAEE07Settings,
    *,
    bucket: str,
    association_policy: str,
    association_role: str,
    association_action_contract: str,
    power_semantics: str = RA_EE_07_SAFE_GREEDY_CONTROL,
) -> tuple[dict[str, Any], np.ndarray]:
    vector, label = _safe_greedy_power_vector(snapshot, _ra_ee_05_settings(settings))
    row = _evaluate_candidate_vector(
        snapshot,
        vector,
        power_semantics=power_semantics,
        selected_power_profile=label,
        settings=settings,
    )
    fixed = _power_vector_for_candidate(snapshot, settings.audit, "fixed-control")
    return (
        _decorated_power_row(
            snapshot,
            row=row,
            requested_power_vector=vector,
            baseline_power_vector=fixed,
            bucket=bucket,
            association_policy=association_policy,
            association_role=association_role,
            association_action_contract=association_action_contract,
            allocator_label="safe-greedy-power-allocator",
            allocator_result=None,
            diagnostic_only=True,
            primary_candidate=False,
        ),
        vector,
    )


def _deployable_row(
    snapshot: _StepSnapshot,
    settings: _RAEE07Settings,
    *,
    result: _AllocatorResult,
    matched_safe_row: dict[str, Any],
    safe_vector: np.ndarray,
    bucket: str,
    association_policy: str,
    association_role: str,
    association_action_contract: str,
    power_semantics: str,
    diagnostic_only: bool,
    primary_candidate: bool,
) -> dict[str, Any]:
    row = _evaluate_candidate_vector(
        snapshot,
        result.power_vector,
        power_semantics=power_semantics,
        selected_power_profile=result.selected_power_profile,
        settings=settings,
    )
    row = _decorated_power_row(
        snapshot,
        row=row,
        requested_power_vector=result.power_vector,
        baseline_power_vector=safe_vector,
        bucket=bucket,
        association_policy=association_policy,
        association_role=association_role,
        association_action_contract=association_action_contract,
        allocator_label=result.allocator_label,
        allocator_result=result,
        diagnostic_only=diagnostic_only,
        primary_candidate=primary_candidate,
    )
    if not diagnostic_only:
        reasons: list[str] = []
        if _row_ee(row) <= _row_ee(matched_safe_row) + 1e-12:
            reasons.append("nonpositive-ee-delta-vs-safe-greedy")
        if not _qos_guardrails_pass_vs_control(
            candidate=row,
            control=matched_safe_row,
            settings=settings,
        ):
            reasons.append("qos-guardrail-failed")
        if bool(row["budget_violation"]):
            reasons.append("budget-violation")
        if bool(row["per_beam_power_violation"]):
            reasons.append("per-beam-power-violation")
        if bool(row["inactive_beam_nonzero_power"]):
            reasons.append("inactive-power-nonzero")
        row["accepted_flag"] = not reasons
        row["rejection_reason"] = "accepted" if not reasons else ";".join(reasons)
    return row


def _oracle_row_from_best_available(
    snapshot: _StepSnapshot,
    settings: _RAEE07Settings,
    *,
    matched_safe_row: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    bucket: str,
    association_policy: str,
    association_role: str,
    association_action_contract: str,
) -> dict[str, Any]:
    oracle = _select_oracle_step(
        snapshot=snapshot,
        control_row=matched_safe_row,
        settings=settings.audit,
    )
    oracle["power_semantics"] = RA_EE_07_CONSTRAINED_ORACLE
    best = oracle
    for candidate in candidate_rows:
        if not _candidate_step_passes(
            row=candidate,
            matched_safe_row=matched_safe_row,
            settings=settings,
        ):
            continue
        if _row_ee(candidate) > _row_ee(best) + 1e-12:
            best = dict(candidate)
            best["selected_power_profile"] = (
                f"oracle:includes-{candidate['allocator_label']}:"
                f"{candidate['effective_power_vector_w']}"
            )
            best["beam_transmit_power_w"] = candidate["effective_power_vector_w"]
    vector = np.fromstring(str(best["beam_transmit_power_w"]), sep=" ", dtype=np.float64)
    if vector.size != snapshot.beam_loads.size:
        vector = _power_vector_for_candidate(snapshot, settings.audit, "fixed-control")
    row = _evaluate_candidate_vector(
        snapshot,
        vector,
        power_semantics=RA_EE_07_CONSTRAINED_ORACLE,
        selected_power_profile=str(best["selected_power_profile"]),
        settings=settings,
    )
    fixed = _power_vector_for_candidate(snapshot, settings.audit, "fixed-control")
    row = _decorated_power_row(
        snapshot,
        row=row,
        requested_power_vector=vector,
        baseline_power_vector=fixed,
        bucket=bucket,
        association_policy=association_policy,
        association_role=association_role,
        association_action_contract=association_action_contract,
        allocator_label="constrained-power-oracle-diagnostic",
        allocator_result=None,
        diagnostic_only=True,
        primary_candidate=False,
    )
    row["oracle_profile"] = row["selected_power_profile"]
    row["rejection_reason"] = "diagnostic-only"
    return row


def _add_oracle_gap_fields(
    rows: list[dict[str, Any]],
    *,
    matched_safe_row: dict[str, Any],
    deployable_row: dict[str, Any],
    oracle_row: dict[str, Any],
) -> None:
    control_ee = _row_ee(matched_safe_row)
    oracle_ee = _row_ee(oracle_row)
    for row in rows:
        row["oracle_profile"] = oracle_row["selected_power_profile"]
        row["oracle_gap_bps_per_w"] = None if oracle_ee == -math.inf else oracle_ee - _row_ee(row)
        row["candidate_regret_bps_per_w"] = row["oracle_gap_bps_per_w"]
        oracle_delta = oracle_ee - control_ee
        candidate_delta = _row_ee(row) - control_ee
        row["oracle_gap_closed_ratio"] = (
            None
            if oracle_delta <= 1e-12
            else candidate_delta / oracle_delta
        )
    deployable_row["candidate_regret_bps_per_w"] = oracle_ee - _row_ee(deployable_row)


def _evaluation_rows_for_fixed_snapshots(
    *,
    snapshots: list[_StepSnapshot],
    bucket_by_policy: dict[str, str],
    settings: _RAEE07Settings,
    include_oracle: bool,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[float]]]:
    rows: list[dict[str, Any]] = []
    user_throughputs_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    for snapshot in snapshots:
        bucket = bucket_by_policy[str(snapshot.trajectory_policy)]
        fixed, _fixed_vector = _fixed_1w_row(
            snapshot,
            settings,
            bucket=bucket,
            association_policy=str(snapshot.trajectory_policy),
            association_role="fixed-association-diagnostic-control",
            association_action_contract="fixed-by-trajectory",
        )
        safe, safe_vector = _safe_greedy_row(
            snapshot,
            settings,
            bucket=bucket,
            association_policy=str(snapshot.trajectory_policy),
            association_role="matched-fixed-association-primary-control",
            association_action_contract="fixed-by-trajectory",
        )
        allocator_results = _deployable_allocator_results(
            snapshot,
            safe,
            safe_vector,
            settings,
        )
        deployable_rows: list[dict[str, Any]] = []
        for allocator in settings.deployable_allocators:
            result = allocator_results[allocator]
            deployable_rows.append(
                _deployable_row(
                    snapshot,
                    settings,
                    result=result,
                    matched_safe_row=safe,
                    safe_vector=safe_vector,
                    bucket=bucket,
                    association_policy=str(snapshot.trajectory_policy),
                    association_role="fixed-association-deployable-power-candidate",
                    association_action_contract="fixed-by-trajectory",
                    power_semantics=(
                        RA_EE_07_DEPLOYABLE
                        if allocator == settings.primary_deployable_allocator
                        else f"deployable-candidate::{allocator}"
                    ),
                    diagnostic_only=allocator != settings.primary_deployable_allocator,
                    primary_candidate=allocator == settings.primary_deployable_allocator,
                )
            )
        primary = next(row for row in deployable_rows if row["primary_candidate"])
        step_rows = [fixed, safe, *deployable_rows]
        oracle = None
        if include_oracle:
            oracle = _oracle_row_from_best_available(
                snapshot,
                settings,
                matched_safe_row=safe,
                candidate_rows=deployable_rows,
                bucket=bucket,
                association_policy=str(snapshot.trajectory_policy),
                association_role="fixed-association-constrained-power-oracle-diagnostic",
                association_action_contract="fixed-by-trajectory",
            )
            step_rows.append(oracle)
            _add_oracle_gap_fields(
                step_rows,
                matched_safe_row=safe,
                deployable_row=primary,
                oracle_row=oracle,
            )
        for row in step_rows:
            throughputs = row.pop("_user_throughputs")
            rows.append(row)
            user_throughputs_by_key[
                (str(row["trajectory_policy"]), str(row["power_semantics"]))
            ].extend(float(value) for value in throughputs.tolist())
    return rows, user_throughputs_by_key


def _best_association_oracle_row(
    *,
    snapshots_by_key: dict[tuple[str, int, int], _StepSnapshot],
    spec_name: str,
    seed: int,
    step_index: int,
    matched_safe_row: dict[str, Any],
    settings: _RAEE07Settings,
) -> tuple[dict[str, Any] | None, str]:
    best: dict[str, Any] | None = None
    best_policy = ""
    for policy in settings.association_diagnostic_policies:
        label = _policy_label(spec_name, policy)
        snapshot = snapshots_by_key.get((label, int(seed), int(step_index)))
        if snapshot is None:
            continue
        oracle = _select_oracle_step(
            snapshot=snapshot,
            control_row=matched_safe_row,
            settings=settings.audit,
        )
        oracle["power_semantics"] = RA_EE_07_ASSOC_ORACLE_CONSTRAINED
        if best is None or _row_ee(oracle) > _row_ee(best) + 1e-12:
            best = oracle
            best_policy = policy
    return best, best_policy


def _evaluation_rows_for_association_diagnostics(
    *,
    cfg: dict[str, Any],
    settings: _RAEE07Settings,
    max_steps: int | None,
    include_oracle: bool,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[float]]]:
    diag_settings = _ra_ee_06b_settings(settings)
    trajectories, _metadata, _traces = _rollout_association_trajectories(
        cfg=cfg,
        settings=diag_settings,
        max_steps=max_steps,
    )
    snapshots = _build_unit_power_snapshots(
        base_cfg=cfg,
        settings=settings.audit,
        trajectories=trajectories,
    )
    snapshots_by_key = {
        (
            str(snapshot.trajectory_policy),
            int(snapshot.evaluation_seed),
            int(snapshot.step_index),
        ): snapshot
        for snapshot in snapshots
    }
    rows: list[dict[str, Any]] = []
    user_throughputs_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    for spec in settings.diagnostic_bucket_specs:
        control_label = _policy_label(spec.name, FIXED_HOLD_CURRENT)
        for proposal_policy in settings.association_diagnostic_policies:
            candidate_label = _policy_label(spec.name, proposal_policy)
            for seed in spec.evaluation_seed_set:
                step_indices = sorted(
                    step
                    for label, eval_seed, step in snapshots_by_key
                    if label == candidate_label and eval_seed == int(seed)
                )
                for step_index in step_indices:
                    control_snapshot = snapshots_by_key.get(
                        (control_label, int(seed), int(step_index))
                    )
                    candidate_snapshot = snapshots_by_key.get(
                        (candidate_label, int(seed), int(step_index))
                    )
                    if control_snapshot is None or candidate_snapshot is None:
                        continue
                    safe, safe_vector = _safe_greedy_row(
                        control_snapshot,
                        settings,
                        bucket=spec.name,
                        association_policy=FIXED_HOLD_CURRENT,
                        association_role="diagnostic-matched-fixed-association-control",
                        association_action_contract="fixed-by-trajectory",
                        power_semantics=RA_EE_07_SAFE_GREEDY_CONTROL,
                    )
                    allocator_results = _deployable_allocator_results(
                        candidate_snapshot,
                        safe,
                        safe_vector,
                        settings,
                    )
                    selected = allocator_results[settings.primary_deployable_allocator]
                    proposal = _deployable_row(
                        candidate_snapshot,
                        settings,
                        result=selected,
                        matched_safe_row=safe,
                        safe_vector=safe_vector,
                        bucket=spec.name,
                        association_policy=proposal_policy,
                        association_role="diagnostic-association-proposal",
                        association_action_contract=(
                            "deterministic-active-set-served-set-proposal-rule"
                        ),
                        power_semantics=RA_EE_07_ASSOC_PROPOSAL_DEPLOYABLE,
                        diagnostic_only=True,
                        primary_candidate=False,
                    )
                    step_rows = [safe, proposal]
                    if include_oracle:
                        oracle_base, oracle_policy = _best_association_oracle_row(
                            snapshots_by_key=snapshots_by_key,
                            spec_name=spec.name,
                            seed=int(seed),
                            step_index=int(step_index),
                            matched_safe_row=safe,
                            settings=settings,
                        )
                        if oracle_base is not None:
                            vector = np.fromstring(
                                str(oracle_base["beam_transmit_power_w"]),
                                sep=" ",
                                dtype=np.float64,
                            )
                            oracle_snapshot = snapshots_by_key[
                                (_policy_label(spec.name, oracle_policy), int(seed), int(step_index))
                            ]
                            oracle = _evaluate_candidate_vector(
                                oracle_snapshot,
                                vector,
                                power_semantics=RA_EE_07_ASSOC_ORACLE_CONSTRAINED,
                                selected_power_profile=str(
                                    oracle_base["selected_power_profile"]
                                ),
                                settings=settings,
                            )
                            oracle = _decorated_power_row(
                                oracle_snapshot,
                                row=oracle,
                                requested_power_vector=vector,
                                baseline_power_vector=safe_vector,
                                bucket=spec.name,
                                association_policy=oracle_policy,
                                association_role=(
                                    "diagnostic-association-oracle-constrained-power"
                                ),
                                association_action_contract="finite-active-set-oracle-diagnostic",
                                allocator_label="constrained-power-oracle-diagnostic",
                                allocator_result=None,
                                diagnostic_only=True,
                                primary_candidate=False,
                            )
                            step_rows.append(oracle)
                            _add_oracle_gap_fields(
                                step_rows,
                                matched_safe_row=safe,
                                deployable_row=proposal,
                                oracle_row=oracle,
                            )
                    for row in step_rows:
                        row["trajectory_policy"] = candidate_label
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
    allocators_by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    selected_by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    rejection_by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    accepted_move_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    rejected_move_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    oracle_gap_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    regret_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in step_rows:
        key = (str(row["trajectory_policy"]), str(row["power_semantics"]))
        vectors_by_key[key].append(str(row["effective_power_vector_w"]))
        allocators_by_key[key].append(str(row["allocator_label"]))
        selected_by_key[key].append(str(row["selected_allocator_candidate"]))
        rejection_by_key[key].append(str(row["rejection_reason"]))
        accepted_move_by_key[key].append(float(row["accepted_allocator_move_count"]))
        rejected_move_by_key[key].append(float(row["rejected_allocator_move_count"]))
        if row["oracle_gap_closed_ratio"] is not None:
            oracle_gap_by_key[key].append(float(row["oracle_gap_closed_ratio"]))
        if row["candidate_regret_bps_per_w"] is not None:
            regret_by_key[key].append(float(row["candidate_regret_bps_per_w"]))
        metadata_by_key.setdefault(
            key,
            {
                "evaluation_bucket": row["evaluation_bucket"],
                "association_policy": row["association_policy"],
                "association_role": row["association_role"],
                "association_action_contract": row["association_action_contract"],
                "allocator_label": row["allocator_label"],
                "diagnostic_only": row["diagnostic_only"],
                "primary_candidate": row["primary_candidate"],
                "oracle_labels_used_for_runtime_decision": row[
                    "oracle_labels_used_for_runtime_decision"
                ],
                "future_outcomes_used_for_runtime_decision": row[
                    "future_outcomes_used_for_runtime_decision"
                ],
                "held_out_answers_used_for_runtime_decision": row[
                    "held_out_answers_used_for_runtime_decision"
                ],
            },
        )
    for summary in summaries:
        key = (str(summary["trajectory_policy"]), str(summary["power_semantics"]))
        summary.update(metadata_by_key[key])
        summary["selected_power_vector_distribution"] = _categorical_distribution(
            vectors_by_key[key]
        )
        summary["allocator_label_distribution"] = _categorical_distribution(
            allocators_by_key[key]
        )
        summary["selected_allocator_candidate_distribution"] = _categorical_distribution(
            selected_by_key[key]
        )
        summary["rejection_reason_distribution"] = _categorical_distribution(
            rejection_by_key[key]
        )
        summary["accepted_allocator_move_count_distribution"] = _numeric_distribution(
            accepted_move_by_key[key]
        )
        summary["rejected_allocator_move_count_distribution"] = _numeric_distribution(
            rejected_move_by_key[key]
        )
        summary["oracle_gap_closed_ratio_distribution"] = _numeric_distribution(
            oracle_gap_by_key[key]
        )
        summary["candidate_regret_bps_per_w_distribution"] = _numeric_distribution(
            regret_by_key[key]
        )
    return summaries


def _pct_delta(reference: float | None, value: float | None) -> float | None:
    if reference is None or value is None or abs(float(reference)) < 1e-12:
        return None
    return float((float(value) - float(reference)) / abs(float(reference)))


def _group_summaries(summaries: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for summary in summaries:
        grouped[str(summary["trajectory_policy"])][str(summary["power_semantics"])] = summary
    return grouped


def _guardrail_result(
    *,
    candidate: dict[str, Any],
    control: dict[str, Any],
    oracle: dict[str, Any] | None,
    settings: _RAEE07Settings,
) -> dict[str, Any]:
    p05_threshold = settings.audit.p05_min_ratio_vs_control * float(
        control["throughput_p05_user_step_bps"]
    )
    p05_ratio = (
        None
        if control["throughput_p05_user_step_bps"] is None
        or abs(float(control["throughput_p05_user_step_bps"])) < 1e-12
        else float(candidate["throughput_p05_user_step_bps"])
        / abs(float(control["throughput_p05_user_step_bps"]))
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
    oracle_delta = None
    oracle_gap_closed = None
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

    denominator_varies = bool(candidate["denominator_varies_in_eval"])
    profile_varies = int(candidate["selected_profile_distinct_count"]) > 1
    vector_varies = int(candidate["selected_power_vector_distribution"]["distinct_count"]) > 1
    active_power_varies = (
        len(candidate["total_active_beam_power_w_distribution"]["distinct"]) > 1
    )
    oracle_gap_pass = (
        oracle_gap_closed is None
        or oracle_gap_closed >= settings.min_oracle_gap_closed_ratio
    )
    reasons: list[str] = []
    if ee_delta is None or ee_delta <= 0.0:
        reasons.append("nonpositive-ee-delta-vs-safe-greedy")
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
    if not denominator_varies:
        reasons.append("denominator-fixed")
    if not profile_varies:
        reasons.append("selected-profile-single-point")
    if not vector_varies:
        reasons.append("selected-power-vector-single-point")
    if not active_power_varies:
        reasons.append("total-active-power-single-point")
    if not oracle_gap_pass:
        reasons.append("oracle-gap-not-meaningfully-closed")
    accepted = not reasons
    return {
        "evaluation_bucket": candidate["evaluation_bucket"],
        "trajectory_policy": candidate["trajectory_policy"],
        "power_semantics": candidate["power_semantics"],
        "allocator_label": candidate["allocator_label"],
        "matched_control_power_semantics": control["power_semantics"],
        "diagnostic_oracle_power_semantics": (
            None if oracle is None else oracle["power_semantics"]
        ),
        "EE_system_delta_vs_matched_safe_greedy": ee_delta,
        "EE_system_pct_delta_vs_matched_safe_greedy": _pct_delta(
            control["EE_system_aggregate_bps_per_w"],
            candidate["EE_system_aggregate_bps_per_w"],
        ),
        "throughput_mean_pct_delta_vs_matched_safe_greedy": _pct_delta(
            control["throughput_mean_user_step_bps"],
            candidate["throughput_mean_user_step_bps"],
        ),
        "throughput_p05_ratio_vs_matched_safe_greedy": p05_ratio,
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
        "denominator_varies_in_eval": denominator_varies,
        "selected_profiles_not_single_point": profile_varies,
        "selected_power_vectors_not_single_point": vector_varies,
        "total_active_power_not_single_point": active_power_varies,
        "oracle_delta_vs_matched_safe_greedy": oracle_delta,
        "oracle_gap_closed_ratio": oracle_gap_closed,
        "oracle_gap_closed_guardrail_pass": oracle_gap_pass,
        "oracle_is_diagnostic_only": True,
        "candidate_runtime_uses_oracle_labels": bool(
            candidate["oracle_labels_used_for_runtime_decision"]
        ),
        "candidate_runtime_uses_future_outcomes": bool(
            candidate["future_outcomes_used_for_runtime_decision"]
        ),
        "candidate_runtime_uses_held_out_answers": bool(
            candidate["held_out_answers_used_for_runtime_decision"]
        ),
        "accepted": accepted,
        "rejection_reason": "accepted" if accepted else ";".join(reasons),
    }


def _build_guardrail_checks(
    *,
    summaries: list[dict[str, Any]],
    settings: _RAEE07Settings,
) -> list[dict[str, Any]]:
    grouped = _group_summaries(summaries)
    checks: list[dict[str, Any]] = []
    for _policy, rows in sorted(grouped.items()):
        control = rows.get(RA_EE_07_SAFE_GREEDY_CONTROL)
        if control is None:
            continue
        oracle = rows.get(RA_EE_07_CONSTRAINED_ORACLE) or rows.get(
            RA_EE_07_ASSOC_ORACLE_CONSTRAINED
        )
        for semantics in (
            RA_EE_07_DEPLOYABLE,
            f"deployable-candidate::{P05_SLACK_TRIM_TAIL_PROTECT}",
            f"deployable-candidate::{BOUNDED_LOCAL_SEARCH}",
            f"deployable-candidate::{FINITE_CODEBOOK_DP}",
            RA_EE_07_ASSOC_PROPOSAL_DEPLOYABLE,
            RA_EE_07_CONSTRAINED_ORACLE,
            RA_EE_07_ASSOC_ORACLE_CONSTRAINED,
        ):
            candidate = rows.get(semantics)
            if candidate is None:
                continue
            result = _guardrail_result(
                candidate=candidate,
                control=control,
                oracle=oracle,
                settings=settings,
            )
            if semantics in {
                RA_EE_07_CONSTRAINED_ORACLE,
                RA_EE_07_ASSOC_ORACLE_CONSTRAINED,
                RA_EE_07_ASSOC_PROPOSAL_DEPLOYABLE,
            }:
                result["accepted"] = False
                result["rejection_reason"] = "diagnostic-only"
            checks.append(result)
    return checks


def _ranking_checks(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_summaries(summaries)
    checks: list[dict[str, Any]] = []
    for policy, rows in sorted(grouped.items()):
        compared = [
            rows[key]
            for key in (RA_EE_07_SAFE_GREEDY_CONTROL, RA_EE_07_DEPLOYABLE)
            if key in rows
        ]
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
                "compared_power_semantics": [
                    RA_EE_07_SAFE_GREEDY_CONTROL,
                    RA_EE_07_DEPLOYABLE,
                ],
                "throughput_rescore_ranking": throughput_ranking,
                "EE_rescore_ranking": ee_ranking,
                "throughput_rescore_winner": throughput_ranking[0],
                "EE_rescore_winner": ee_ranking[0],
                "throughput_rescore_vs_EE_rescore_ranking_changes": (
                    throughput_ranking != ee_ranking
                ),
                "throughput_rescore_vs_EE_rescore_top_changes": (
                    throughput_ranking[0] != ee_ranking[0]
                ),
            }
        )
    return checks


def _oracle_gap_diagnostics(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_summaries(summaries)
    diagnostics: list[dict[str, Any]] = []
    for policy, rows in sorted(grouped.items()):
        control = rows.get(RA_EE_07_SAFE_GREEDY_CONTROL)
        candidate = rows.get(RA_EE_07_DEPLOYABLE) or rows.get(
            RA_EE_07_ASSOC_PROPOSAL_DEPLOYABLE
        )
        oracle = rows.get(RA_EE_07_CONSTRAINED_ORACLE) or rows.get(
            RA_EE_07_ASSOC_ORACLE_CONSTRAINED
        )
        if control is None or candidate is None or oracle is None:
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
                "candidate_power_semantics": candidate["power_semantics"],
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


def _seed_level_results(
    *,
    step_rows: list[dict[str, Any]],
    bucket: str,
) -> dict[str, Any]:
    by_key: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in step_rows:
        if row["evaluation_bucket"] != bucket:
            continue
        if row["power_semantics"] not in {
            RA_EE_07_SAFE_GREEDY_CONTROL,
            RA_EE_07_DEPLOYABLE,
        }:
            continue
        by_key[(int(row["evaluation_seed"]), str(row["power_semantics"]))].append(row)
    seeds = sorted({seed for seed, _semantics in by_key})
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        control = by_key.get((seed, RA_EE_07_SAFE_GREEDY_CONTROL), [])
        candidate = by_key.get((seed, RA_EE_07_DEPLOYABLE), [])
        if not control or not candidate:
            continue
        control_thr = sum(float(row["sum_user_throughput_bps"]) for row in control)
        control_power = sum(float(row["total_active_beam_power_w"]) for row in control)
        cand_thr = sum(float(row["sum_user_throughput_bps"]) for row in candidate)
        cand_power = sum(float(row["total_active_beam_power_w"]) for row in candidate)
        control_ee = None if control_power <= 0.0 else control_thr / control_power
        cand_ee = None if cand_power <= 0.0 else cand_thr / cand_power
        delta = None if control_ee is None or cand_ee is None else cand_ee - control_ee
        rows.append(
            {
                "seed": int(seed),
                "control_EE_system_aggregate_bps_per_w": control_ee,
                "candidate_EE_system_aggregate_bps_per_w": cand_ee,
                "EE_system_delta_vs_matched_safe_greedy": delta,
                "positive": delta is not None and delta > 0.0,
            }
        )
    positive = [row for row in rows if bool(row["positive"])]
    positive_delta_sum = sum(
        max(0.0, float(row["EE_system_delta_vs_matched_safe_greedy"] or 0.0))
        for row in rows
    )
    max_share = (
        0.0
        if positive_delta_sum <= 1e-12
        else max(
            max(0.0, float(row["EE_system_delta_vs_matched_safe_greedy"] or 0.0))
            / positive_delta_sum
            for row in rows
        )
    )
    return {
        "bucket": bucket,
        "seed_results": rows,
        "positive_seed_count": len(positive),
        "seed_count": len(rows),
        "majority_positive_seeds": bool(rows) and len(positive) > len(rows) / 2.0,
        "gains_not_concentrated_in_one_seed": len(positive) >= 2 and max_share < 0.80,
        "max_positive_seed_delta_share": max_share,
    }


def _bucket_results(
    *,
    settings: _RAEE07Settings,
    summaries: list[dict[str, Any]],
    guardrail_checks: list[dict[str, Any]],
    ranking_checks: list[dict[str, Any]],
    oracle_gap_diagnostics: list[dict[str, Any]],
    seed_results: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    candidate_by_policy = {
        str(row["trajectory_policy"]): row
        for row in summaries
        if row["power_semantics"] == RA_EE_07_DEPLOYABLE
    }
    guardrail_by_policy = {
        str(row["trajectory_policy"]): row
        for row in guardrail_checks
        if row["power_semantics"] == RA_EE_07_DEPLOYABLE
    }
    ranking_by_policy = {
        str(row["trajectory_policy"]): row for row in ranking_checks
    }
    oracle_gap_by_policy = {
        str(row["trajectory_policy"]): row
        for row in oracle_gap_diagnostics
        if row["candidate_power_semantics"] == RA_EE_07_DEPLOYABLE
    }
    results: dict[str, dict[str, Any]] = {}
    for spec in settings.fixed_bucket_specs:
        present = [
            policy
            for policy in spec.trajectory_families
            if policy in candidate_by_policy
        ]
        noncollapsed = [
            policy
            for policy in present
            if float(candidate_by_policy[policy]["one_active_beam_step_ratio"]) < 1.0
        ]
        positive = [
            policy
            for policy in noncollapsed
            if float(
                guardrail_by_policy.get(policy, {}).get(
                    "EE_system_delta_vs_matched_safe_greedy",
                    -math.inf,
                )
                or -math.inf
            )
            > 0.0
        ]
        accepted = [
            policy
            for policy in positive
            if bool(guardrail_by_policy.get(policy, {}).get("accepted"))
        ]
        no_power_violations = all(
            bool(guardrail_by_policy.get(policy, {}).get("budget_guardrail_pass"))
            and bool(guardrail_by_policy.get(policy, {}).get("per_beam_power_guardrail_pass"))
            and bool(guardrail_by_policy.get(policy, {}).get("inactive_beam_zero_w_guardrail_pass"))
            for policy in present
        )
        qos_pass = bool(accepted) and all(
            bool(guardrail_by_policy[policy]["p05_guardrail_pass"])
            and bool(guardrail_by_policy[policy]["served_ratio_guardrail_pass"])
            and bool(guardrail_by_policy[policy]["outage_guardrail_pass"])
            for policy in accepted
        )
        denominator_varies = bool(accepted) and all(
            bool(candidate_by_policy[policy]["denominator_varies_in_eval"])
            for policy in accepted
        )
        profile_varies = bool(accepted) and all(
            bool(guardrail_by_policy[policy]["selected_profiles_not_single_point"])
            for policy in accepted
        )
        vector_varies = bool(accepted) and all(
            bool(guardrail_by_policy[policy]["selected_power_vectors_not_single_point"])
            for policy in accepted
        )
        active_power_varies = bool(accepted) and all(
            bool(guardrail_by_policy[policy]["total_active_power_not_single_point"])
            for policy in accepted
        )
        ranking_or_gap_clear = bool(accepted) and all(
            bool(
                ranking_by_policy.get(policy, {}).get(
                    "throughput_rescore_vs_EE_rescore_top_changes"
                )
            )
            or (
                oracle_gap_by_policy.get(policy, {}).get("oracle_gap_closed_ratio")
                is not None
                and float(oracle_gap_by_policy[policy]["oracle_gap_closed_ratio"])
                >= settings.min_oracle_gap_closed_ratio
            )
            for policy in accepted
        )
        positive_delta_sum = sum(
            max(
                0.0,
                float(
                    guardrail_by_policy.get(policy, {}).get(
                        "EE_system_delta_vs_matched_safe_greedy",
                        0.0,
                    )
                    or 0.0
                ),
            )
            for policy in present
        )
        max_traj_share = (
            0.0
            if positive_delta_sum <= 1e-12
            else max(
                max(
                    0.0,
                    float(
                        guardrail_by_policy.get(policy, {}).get(
                            "EE_system_delta_vs_matched_safe_greedy",
                            0.0,
                        )
                        or 0.0
                    ),
                )
                / positive_delta_sum
                for policy in present
            )
        )
        gap_rows = [
            oracle_gap_by_policy[policy]
            for policy in present
            if policy in oracle_gap_by_policy
            and oracle_gap_by_policy[policy]["oracle_EE_delta_vs_control"] is not None
            and float(oracle_gap_by_policy[policy]["oracle_EE_delta_vs_control"]) > 1e-12
        ]
        gap_numerator = sum(
            max(0.0, float(row["candidate_EE_delta_vs_control"] or 0.0))
            for row in gap_rows
        )
        gap_denominator = sum(
            max(0.0, float(row["oracle_EE_delta_vs_control"] or 0.0))
            for row in gap_rows
        )
        aggregate_gap_closure = (
            None if gap_denominator <= 1e-12 else gap_numerator / gap_denominator
        )
        seed_ok = (
            spec.name != HELD_OUT_BUCKET
            or bool(seed_results.get("gains_not_concentrated_in_one_seed"))
        )
        results[spec.name] = {
            "bucket": spec.name,
            "evaluation_seed_set": list(spec.evaluation_seed_set),
            "trajectory_families": list(spec.trajectory_families),
            "present_trajectory_count": len(present),
            "noncollapsed_trajectory_count": len(noncollapsed),
            "noncollapsed_trajectories": noncollapsed,
            "positive_EE_delta_trajectory_count": len(positive),
            "positive_EE_delta_trajectories": positive,
            "accepted_candidate_trajectory_count": len(accepted),
            "accepted_candidate_trajectories": accepted,
            "rejection_reasons": {
                policy: guardrail_by_policy.get(policy, {}).get("rejection_reason", "missing")
                for policy in present
            },
            "majority_noncollapsed_positive_EE_delta": (
                bool(noncollapsed) and len(positive) > len(noncollapsed) / 2.0
            ),
            "majority_noncollapsed_accepted": (
                bool(noncollapsed) and len(accepted) > len(noncollapsed) / 2.0
            ),
            "gains_not_concentrated_in_one_trajectory": (
                len(positive) >= 2 and max_traj_share < 0.80
            ),
            "max_positive_trajectory_delta_share": max_traj_share,
            "gains_not_concentrated_in_one_seed": seed_ok,
            "qos_guardrails_pass_for_accepted": qos_pass,
            "zero_budget_per_beam_inactive_power_violations": no_power_violations,
            "denominator_varies_for_accepted": denominator_varies,
            "selected_profiles_not_single_point_for_accepted": profile_varies,
            "selected_power_vectors_not_single_point_for_accepted": vector_varies,
            "total_active_power_not_single_point_for_accepted": active_power_varies,
            "ranking_separates_or_oracle_gap_reduction_clear": ranking_or_gap_clear,
            "aggregate_oracle_gap_closed_ratio": aggregate_gap_closure,
            "candidate_closes_meaningful_oracle_gap": (
                aggregate_gap_closure is not None
                and aggregate_gap_closure >= settings.min_oracle_gap_closed_ratio
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
                    and max_traj_share < 0.80
                    and seed_ok
                    and qos_pass
                    and no_power_violations
                    and denominator_varies
                    and profile_varies
                    and vector_varies
                    and active_power_varies
                    and ranking_or_gap_clear
                    and aggregate_gap_closure is not None
                    and aggregate_gap_closure >= settings.min_oracle_gap_closed_ratio
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
    include_association_diagnostics: bool,
) -> dict[str, Any]:
    held_out = bucket_results.get(HELD_OUT_BUCKET, {})
    primary_summaries = [
        row for row in summaries if row["power_semantics"] == RA_EE_07_DEPLOYABLE
    ]
    primary_guardrails = [
        row for row in guardrail_checks if row["power_semantics"] == RA_EE_07_DEPLOYABLE
    ]
    no_power_violations = all(
        bool(row.get("budget_guardrail_pass"))
        and bool(row.get("per_beam_power_guardrail_pass"))
        and bool(row.get("inactive_beam_zero_w_guardrail_pass"))
        for row in primary_guardrails
    )
    no_leakage = all(
        not bool(row.get("oracle_labels_used_for_runtime_decision"))
        and not bool(row.get("future_outcomes_used_for_runtime_decision"))
        and not bool(row.get("held_out_answers_used_for_runtime_decision"))
        for row in primary_summaries
    )
    fixed_only = all(
        row.get("association_action_contract") == "fixed-by-trajectory"
        for row in primary_summaries
    )
    proof_flags = {
        "held_out_bucket_exists_and_reported_separately": bool(held_out),
        "offline_replay_only": True,
        "fixed_association_primary_only": fixed_only,
        "diagnostic_association_buckets_reported_separately": (
            include_association_diagnostics
        ),
        "deployable_non_oracle_power_allocator_comparison_only": True,
        "learned_association_disabled": True,
        "learned_hierarchical_RL_disabled": True,
        "joint_association_power_training_disabled": True,
        "catfish_disabled": True,
        "multi_catfish_disabled": True,
        "rb_bandwidth_allocation_disabled": True,
        "old_EE_MODQN_continuation_disabled": True,
        "frozen_baseline_mutation": False,
        "oracle_diagnostic_only": include_oracle,
        "candidate_does_not_use_oracle_labels_or_future_or_heldout_answers": no_leakage,
        "same_effective_power_vector_feeds_numerator_denominator_audit": True,
        "majority_noncollapsed_held_out_positive_EE_delta": bool(
            held_out.get("majority_noncollapsed_positive_EE_delta")
        ),
        "majority_noncollapsed_held_out_accepted": bool(
            held_out.get("majority_noncollapsed_accepted")
        ),
        "held_out_gains_not_concentrated_in_one_trajectory": bool(
            held_out.get("gains_not_concentrated_in_one_trajectory")
        ),
        "held_out_gains_not_concentrated_in_one_seed": bool(
            held_out.get("gains_not_concentrated_in_one_seed")
        ),
        "candidate_closes_meaningful_oracle_gap": bool(
            held_out.get("candidate_closes_meaningful_oracle_gap")
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
        "selected_profiles_not_single_point_for_accepted_held_out": bool(
            held_out.get("selected_profiles_not_single_point_for_accepted")
        ),
        "total_active_power_not_single_point_for_accepted_held_out": bool(
            held_out.get("total_active_power_not_single_point_for_accepted")
        ),
        "ranking_separates_or_oracle_gap_reduction_clear": bool(
            held_out.get("ranking_separates_or_oracle_gap_reduction_clear")
        ),
        "scalar_reward_success_basis": False,
        "per_user_EE_credit_success_basis": False,
        "physical_energy_saving_claim": False,
        "hobs_optimizer_claim": False,
        "full_RA_EE_MODQN_claim": False,
    }
    stop_conditions = {
        "held_out_bucket_missing": not bool(held_out),
        "stronger_allocator_cannot_beat_safe_greedy_on_held_out": not bool(
            held_out.get("majority_noncollapsed_positive_EE_delta")
        ),
        "gains_require_constrained_oracle_rows": (
            not bool(held_out.get("majority_noncollapsed_positive_EE_delta"))
            and any(
                row["evaluation_bucket"] == HELD_OUT_BUCKET
                and row["power_semantics"] == RA_EE_07_CONSTRAINED_ORACLE
                and float(row["EE_system_delta_vs_matched_safe_greedy"] or 0.0) > 0.0
                for row in guardrail_checks
            )
        ),
        "oracle_gap_closure_zero_negative_or_concentrated": not bool(
            held_out.get("candidate_closes_meaningful_oracle_gap")
        ),
        "held_out_gains_concentrated": not (
            bool(held_out.get("gains_not_concentrated_in_one_trajectory"))
            and bool(held_out.get("gains_not_concentrated_in_one_seed"))
        ),
        "p05_served_or_outage_guardrail_fails": any(
            row["evaluation_bucket"] == HELD_OUT_BUCKET
            and float(row["EE_system_delta_vs_matched_safe_greedy"] or 0.0) > 0.0
            and not bool(row["QoS_guardrails_pass"])
            for row in primary_guardrails
        ),
        "budget_or_inactive_power_violations": not no_power_violations,
        "denominator_or_profile_collapses": bool(held_out.get("accepted_candidate_trajectory_count"))
        and not (
            bool(held_out.get("denominator_varies_for_accepted"))
            and bool(held_out.get("selected_profiles_not_single_point_for_accepted"))
            and bool(held_out.get("total_active_power_not_single_point_for_accepted"))
        ),
        "candidate_uses_oracle_labels_future_or_hidden_leakage": not no_leakage,
        "learned_association_joint_training_catfish_or_RB_added": False,
        "frozen_baseline_mutated": False,
        "oracle_used_as_candidate_claim": False,
    }
    required_true = (
        "held_out_bucket_exists_and_reported_separately",
        "offline_replay_only",
        "fixed_association_primary_only",
        "deployable_non_oracle_power_allocator_comparison_only",
        "learned_association_disabled",
        "learned_hierarchical_RL_disabled",
        "joint_association_power_training_disabled",
        "catfish_disabled",
        "multi_catfish_disabled",
        "rb_bandwidth_allocation_disabled",
        "oracle_diagnostic_only",
        "candidate_does_not_use_oracle_labels_or_future_or_heldout_answers",
        "majority_noncollapsed_held_out_positive_EE_delta",
        "majority_noncollapsed_held_out_accepted",
        "held_out_gains_not_concentrated_in_one_trajectory",
        "held_out_gains_not_concentrated_in_one_seed",
        "candidate_closes_meaningful_oracle_gap",
        "p05_throughput_guardrail_pass_for_accepted_held_out",
        "served_ratio_does_not_drop_for_accepted_held_out",
        "outage_ratio_does_not_increase_for_accepted_held_out",
        "zero_budget_per_beam_inactive_power_violations",
        "denominator_varies_for_accepted_held_out",
        "selected_profiles_not_single_point_for_accepted_held_out",
        "total_active_power_not_single_point_for_accepted_held_out",
        "ranking_separates_or_oracle_gap_reduction_clear",
    )
    pass_required = all(bool(proof_flags[field]) for field in required_true)
    pass_required = (
        pass_required
        and proof_flags["scalar_reward_success_basis"] is False
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
        "ra_ee_07_decision": decision,
        "proof_flags": proof_flags,
        "stop_conditions": stop_conditions,
        "candidate_guardrail_checks": primary_guardrails,
        "allowed_claim": (
            "PASS only means a deployable non-oracle power allocator beat the "
            "matched fixed-association safe-greedy allocator on the RA-EE-07 "
            "offline held-out gate. It is not learned association or full RA-EE-MODQN."
            if decision == "PASS"
            else "Do not promote RA-EE-07 beyond a blocked or inconclusive offline replay gate."
        ),
    }


def _compact_summary_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "evaluation_bucket",
        "trajectory_policy",
        "association_policy",
        "association_role",
        "power_semantics",
        "allocator_label",
        "selected_allocator_candidate_distribution",
        "EE_system_aggregate_bps_per_w",
        "EE_system_step_mean_bps_per_w",
        "throughput_mean_user_step_bps",
        "throughput_p05_user_step_bps",
        "served_ratio",
        "outage_ratio",
        "handover_count",
        "active_beam_count_distribution",
        "selected_power_profile_distribution",
        "selected_power_vector_distribution",
        "total_active_beam_power_w_distribution",
        "denominator_varies_in_eval",
        "one_active_beam_step_ratio",
        "budget_violations",
        "per_beam_power_violations",
        "inactive_beam_nonzero_power_step_count",
        "accepted_allocator_move_count_distribution",
        "rejected_allocator_move_count_distribution",
        "oracle_gap_closed_ratio_distribution",
        "candidate_regret_bps_per_w_distribution",
        "throughput_vs_EE_system_correlation",
        "diagnostic_only",
        "primary_candidate",
    )
    return [{field: row[field] for field in fields} for row in summaries]


def _review_lines(summary: dict[str, Any]) -> list[str]:
    decision = summary["ra_ee_07_decision"]
    proof = summary["proof_flags"]
    held_out = summary["bucket_results"][HELD_OUT_BUCKET]
    lines = [
        "# RA-EE-07 Constrained-Power Allocator Distillation Review",
        "",
        "Offline fixed-association deployable power-allocator comparison only. "
        "Association proposal buckets and oracle rows are diagnostic-only. No "
        "learned association, hierarchical RL, joint association + power "
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
        f"- primary control: `{summary['protocol']['primary_control']}`",
        f"- primary candidate: `{summary['protocol']['primary_candidate']}`",
        f"- deployable allocator candidates: `{summary['protocol']['deployable_allocator_candidates']}`",
        f"- diagnostic fixed 1W: `{summary['protocol']['fixed_1w_diagnostic']}`",
        f"- diagnostic constrained oracle: `{summary['protocol']['constrained_power_oracle_isolation']}`",
        f"- diagnostic association proposal: `{summary['protocol']['association_proposal_diagnostic']}`",
        f"- diagnostic association oracle upper bound: `{summary['protocol']['association_oracle_upper_bound']}`",
        "",
        "## Held-Out Gate",
        "",
        f"- noncollapsed held-out trajectories: `{held_out['noncollapsed_trajectories']}`",
        f"- positive EE delta trajectories: `{held_out['positive_EE_delta_trajectories']}`",
        f"- accepted candidate trajectories: `{held_out['accepted_candidate_trajectories']}`",
        f"- rejection reasons: `{held_out['rejection_reasons']}`",
        f"- aggregate oracle gap closure: `{held_out['aggregate_oracle_gap_closed_ratio']}`",
        f"- gains not concentrated in one trajectory: `{held_out['gains_not_concentrated_in_one_trajectory']}`",
        f"- gains not concentrated in one seed: `{held_out['gains_not_concentrated_in_one_seed']}`",
        f"- QoS guardrails pass for accepted: `{held_out['qos_guardrails_pass_for_accepted']}`",
        f"- zero budget / per-beam / inactive-power violations: `{held_out['zero_budget_per_beam_inactive_power_violations']}`",
        f"- ranking separates or oracle gap reduction clear: `{held_out['ranking_separates_or_oracle_gap_reduction_clear']}`",
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
            f"- RA-EE-07 decision: `{decision}`",
            f"- allowed claim: {summary['decision_detail']['allowed_claim']}",
            "",
            "## Forbidden Claims",
            "",
        ]
    )
    for claim in summary["forbidden_claims_still_active"]:
        lines.append(f"- {claim}")
    return lines


def export_ra_ee_07_constrained_power_allocator_distillation(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    calibration_seed_set: tuple[int, ...] | None = None,
    held_out_seed_set: tuple[int, ...] | None = None,
    calibration_policies: tuple[str, ...] | None = None,
    held_out_policies: tuple[str, ...] | None = None,
    diagnostic_seed_set: tuple[int, ...] | None = None,
    diagnostic_association_policies: tuple[str, ...] | None = None,
    max_steps: int | None = None,
    include_oracle: bool = True,
    include_association_diagnostics: bool = True,
) -> dict[str, Any]:
    """Export RA-EE-07 constrained-power allocator artifacts."""
    if max_steps is not None and max_steps < 1:
        raise ValueError(f"max_steps must be >= 1 when provided, got {max_steps!r}")
    cfg = load_training_yaml(config_path)
    env = build_environment(cfg)
    if env.power_surface_config.hobs_power_surface_mode != (
        HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK
    ):
        raise ValueError("RA-EE-07 config must opt into the power-codebook surface.")

    settings = _settings_from_config(cfg)
    fixed_specs: list[_BucketSpec] = []
    for spec in settings.fixed_bucket_specs:
        fixed_specs.append(
            _BucketSpec(
                name=spec.name,
                trajectory_families=(
                    tuple(calibration_policies)
                    if spec.name == CALIBRATION_BUCKET and calibration_policies is not None
                    else (
                        tuple(held_out_policies)
                        if spec.name == HELD_OUT_BUCKET and held_out_policies is not None
                        else spec.trajectory_families
                    )
                ),
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
    diagnostic_specs = tuple(
        _BucketSpec(
            name=spec.name,
            trajectory_families=spec.trajectory_families,
            evaluation_seed_set=(
                tuple(diagnostic_seed_set)
                if diagnostic_seed_set is not None
                else spec.evaluation_seed_set
            ),
        )
        for spec in settings.diagnostic_bucket_specs
    )
    run_settings = _RAEE07Settings(
        method_label=settings.method_label,
        implementation_sublabel=settings.implementation_sublabel,
        audit=settings.audit,
        fixed_bucket_specs=tuple(fixed_specs),
        diagnostic_bucket_specs=diagnostic_specs,
        deployable_allocators=settings.deployable_allocators,
        primary_deployable_allocator=settings.primary_deployable_allocator,
        candidate_max_demoted_beams=settings.candidate_max_demoted_beams,
        candidate_step_p05_guardrail_margin=settings.candidate_step_p05_guardrail_margin,
        local_search_max_moves=settings.local_search_max_moves,
        p05_trim_max_moves=settings.p05_trim_max_moves,
        dp_max_profile_count=settings.dp_max_profile_count,
        min_oracle_gap_closed_ratio=settings.min_oracle_gap_closed_ratio,
        association_diagnostic_policies=(
            tuple(diagnostic_association_policies)
            if diagnostic_association_policies is not None
            else settings.association_diagnostic_policies
        ),
        min_active_beams=settings.min_active_beams,
        max_active_beams=settings.max_active_beams,
        target_users_per_active_beam=settings.target_users_per_active_beam,
        load_cap_overflow_users=settings.load_cap_overflow_users,
        max_moved_user_ratio_per_step=settings.max_moved_user_ratio_per_step,
        max_moved_user_ratio=settings.max_moved_user_ratio,
        max_one_active_beam_ratio_for_acceptance=(
            settings.max_one_active_beam_ratio_for_acceptance
        ),
        max_two_beam_overload_step_ratio=settings.max_two_beam_overload_step_ratio,
        diagnostic_max_steps=settings.diagnostic_max_steps,
    )

    trajectories, bucket_by_policy = _rollout_fixed_association_trajectories(
        cfg=cfg,
        bucket_specs=run_settings.fixed_bucket_specs,
        max_steps=max_steps,
    )
    snapshots = _build_unit_power_snapshots(
        base_cfg=cfg,
        settings=run_settings.audit,
        trajectories=trajectories,
    )
    fixed_rows, fixed_user_throughputs = _evaluation_rows_for_fixed_snapshots(
        snapshots=snapshots,
        bucket_by_policy=bucket_by_policy,
        settings=run_settings,
        include_oracle=include_oracle,
    )
    diagnostic_rows: list[dict[str, Any]] = []
    diagnostic_user_throughputs: dict[tuple[str, str], list[float]] = defaultdict(list)
    if include_association_diagnostics:
        diagnostic_rows, diagnostic_user_throughputs = (
            _evaluation_rows_for_association_diagnostics(
                cfg=cfg,
                settings=run_settings,
                max_steps=(
                    max_steps
                    if max_steps is not None
                    else run_settings.diagnostic_max_steps
                ),
                include_oracle=include_oracle,
            )
        )
    step_rows = fixed_rows + diagnostic_rows
    user_throughputs_by_key = defaultdict(list)
    for source in (fixed_user_throughputs, diagnostic_user_throughputs):
        for key, values in source.items():
            user_throughputs_by_key[key].extend(values)
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
    seed_results = _seed_level_results(step_rows=step_rows, bucket=HELD_OUT_BUCKET)
    bucket_results = _bucket_results(
        settings=run_settings,
        summaries=summaries,
        guardrail_checks=guardrail_checks,
        ranking_checks=ranking_checks,
        oracle_gap_diagnostics=oracle_gap_diagnostics,
        seed_results=seed_results,
    )
    decision_detail = _build_decision(
        summaries=summaries,
        guardrail_checks=guardrail_checks,
        bucket_results=bucket_results,
        include_oracle=include_oracle,
        include_association_diagnostics=include_association_diagnostics,
    )

    out_dir = Path(output_dir)
    trace_csv = _write_csv(
        out_dir / "ra_ee_07_step_metrics.csv",
        step_rows,
        fieldnames=_fieldnames(step_rows),
    )
    compact_rows = _compact_summary_rows(summaries)
    summary_csv = _write_csv(
        out_dir / "ra_ee_07_candidate_summary.csv",
        compact_rows,
        fieldnames=list(compact_rows[0].keys()),
    )
    guardrail_csv = _write_csv(
        out_dir / "ra_ee_07_guardrail_checks.csv",
        guardrail_checks,
        fieldnames=list(guardrail_checks[0].keys()) if guardrail_checks else [],
    )

    protocol = {
        "phase": "RA-EE-07",
        "method_label": run_settings.method_label,
        "method_family": "RA-EE constrained fixed-association power allocation",
        "implementation_sublabel": run_settings.implementation_sublabel,
        "training": "none; offline replay only",
        "offline_replay_only": True,
        "primary_association_scope": "fixed association only",
        "diagnostic_association_buckets": include_association_diagnostics,
        "diagnostic_association_max_steps": (
            max_steps if max_steps is not None else run_settings.diagnostic_max_steps
        ),
        "learned_association": "disabled",
        "learned_hierarchical_RL": "disabled",
        "association_training": "disabled",
        "joint_association_power_training": "disabled",
        "catfish": "disabled",
        "multi_catfish": "disabled",
        "rb_bandwidth_allocation": "disabled/not-modeled",
        "old_EE_MODQN_continuation": "forbidden/not-performed",
        "frozen_baseline_mutation": "forbidden/not-performed",
        "hobs_optimizer_claim": "forbidden/not-made",
        "physical_energy_saving_claim": "forbidden/not-made",
        "fixed_1w_diagnostic": RA_EE_07_FIXED_1W_DIAGNOSTIC,
        "primary_control": RA_EE_07_SAFE_GREEDY_CONTROL,
        "primary_candidate": RA_EE_07_DEPLOYABLE,
        "deployable_allocator_candidates": list(run_settings.deployable_allocators),
        "primary_deployable_allocator": run_settings.primary_deployable_allocator,
        "constrained_power_oracle_isolation": RA_EE_07_CONSTRAINED_ORACLE,
        "association_proposal_diagnostic": RA_EE_07_ASSOC_PROPOSAL_DEPLOYABLE,
        "association_oracle_upper_bound": RA_EE_07_ASSOC_ORACLE_CONSTRAINED,
        "oracle_diagnostic_only": include_oracle,
        "system_EE_primary": True,
        "scalar_reward_success_basis": False,
        "per_user_EE_credit_success_basis": False,
    }
    constraints = {
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
        "min_oracle_gap_closed_ratio": run_settings.min_oracle_gap_closed_ratio,
        "candidate_step_p05_guardrail_margin": (
            run_settings.candidate_step_p05_guardrail_margin
        ),
        "power_repair": "not-used; requested and effective vectors are exported",
        "effective_power_vector_contract": (
            "same effective_power_vector_w feeds SINR/SNR numerator, throughput, "
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
        "candidate_summaries": summaries,
        "guardrail_checks": guardrail_checks,
        "ranking_separation_result": {
            "comparison_safe_greedy_vs_deployable": ranking_checks,
        },
        "bucket_results": bucket_results,
        "seed_level_results": seed_results,
        "oracle_gap_diagnostics": oracle_gap_diagnostics,
        "decision_detail": decision_detail,
        "proof_flags": decision_detail["proof_flags"],
        "stop_conditions": decision_detail["stop_conditions"],
        "ra_ee_07_decision": decision_detail["ra_ee_07_decision"],
        "remaining_blockers": [
            "This is offline fixed-association power-allocator evidence only.",
            "No learned association, hierarchical RL, or full RA-EE-MODQN policy exists.",
            "No joint association + power training or RB / bandwidth allocation exists.",
            "Association proposal and oracle rows remain diagnostic-only.",
            "A PASS does not claim HOBS optimizer behavior or physical energy saving.",
        ],
        "forbidden_claims_still_active": [
            "Do not call RA-EE-07 full RA-EE-MODQN.",
            "Do not claim learned association or hierarchical RL effectiveness.",
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
        out_dir / "ra_ee_07_constrained_power_allocator_distillation_summary.json",
        summary,
    )
    review_path = out_dir / "review.md"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text("\n".join(_review_lines(summary)) + "\n")

    return {
        "ra_ee_07_constrained_power_allocator_distillation_summary": summary_path,
        "ra_ee_07_candidate_summary_csv": summary_csv,
        "ra_ee_07_guardrail_checks_csv": guardrail_csv,
        "ra_ee_07_step_metrics": trace_csv,
        "review_md": review_path,
        "summary": summary,
    }


__all__ = [
    "BOUNDED_LOCAL_SEARCH",
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT_DIR",
    "DEPLOYABLE_ALLOCATORS",
    "DETERMINISTIC_HYBRID",
    "FINITE_CODEBOOK_DP",
    "P05_SLACK_TRIM_TAIL_PROTECT",
    "RA_EE_07_ASSOC_ORACLE_CONSTRAINED",
    "RA_EE_07_ASSOC_PROPOSAL_DEPLOYABLE",
    "RA_EE_07_CONSTRAINED_ORACLE",
    "RA_EE_07_DEPLOYABLE",
    "RA_EE_07_FIXED_1W_DIAGNOSTIC",
    "RA_EE_07_METHOD_LABEL",
    "RA_EE_07_SAFE_GREEDY_CONTROL",
    "_candidate_step_passes",
    "_settings_from_config",
    "export_ra_ee_07_constrained_power_allocator_distillation",
]
