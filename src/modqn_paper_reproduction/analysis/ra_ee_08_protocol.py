"""RA-EE-08 protocol constants and config parsing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config_loader import get_seeds
from .ra_ee_02_oracle_power_allocation import _AuditSettings
from .ra_ee_05_fixed_association_robustness import (
    CALIBRATION_BUCKET,
    DEFAULT_CALIBRATION_SEEDS,
    DEFAULT_HELD_OUT_SEEDS,
    HELD_OUT_BUCKET,
    _BucketSpec,
)
from .ra_ee_06_association_counterfactual_oracle import (
    ACTIVE_SET_LOAD_SPREAD,
    ACTIVE_SET_POLICIES,
    ACTIVE_SET_QUALITY_SPREAD,
    ACTIVE_SET_STICKY_SPREAD,
    FIXED_HOLD_CURRENT,
    PER_USER_GREEDY_BEST_BEAM,
    _RAEE06Settings,
    _select_actions_for_association_policy as _select_ra_ee_06_actions,
)
from .ra_ee_06b_association_proposal_refinement import (
    BOUNDED_MOVE_SERVED_SET,
    ORACLE_SCORE_TOPK_ACTIVE_SET,
    P05_SLACK_AWARE_ACTIVE_SET,
    POWER_RESPONSE_AWARE_LOAD_BALANCE,
    RA_EE_06B_PROPOSAL_POLICIES,
    STICKY_ORACLE_COUNT_LOCAL_SEARCH,
    _RAEE06BSettings,
    _select_actions_for_association_policy as _select_ra_ee_06b_actions,
)
from .ra_ee_07_constrained_power_allocator_distillation import (
    DEPLOYABLE_ALLOCATORS,
    DETERMINISTIC_HYBRID,
)

DEFAULT_CONFIG = "configs/ra-ee-08-offline-association-reevaluation.resolved.yaml"
DEFAULT_OUTPUT_DIR = "artifacts/ra-ee-08-offline-association-reevaluation"

RA_EE_08_METHOD_LABEL = "RA-EE offline association re-evaluation gate"
RA_EE_08_FIXED_DEPLOYABLE_CONTROL = (
    "matched-fixed-association+deployable-stronger-power-allocator"
)
RA_EE_08_CANDIDATE = (
    "association-proposal+same-deployable-stronger-power-allocator"
)
RA_EE_08_PROPOSAL_SAFE_GREEDY = "association-proposal+safe-greedy-diagnostic"
RA_EE_08_FIXED_SAFE_GREEDY = "matched-fixed-association+safe-greedy-diagnostic"
RA_EE_08_FIXED_CONSTRAINED_ORACLE = (
    "matched-fixed-association+constrained-power-oracle-diagnostic"
)
RA_EE_08_ASSOC_ORACLE_CONSTRAINED = (
    "association-oracle+constrained-power-oracle-diagnostic"
)
RA_EE_08_ASSOC_ORACLE_DEPLOYABLE = (
    "association-oracle+deployable-stronger-power-allocator-diagnostic"
)

RA_EE_08_PROPOSAL_POLICIES = (
    ACTIVE_SET_LOAD_SPREAD,
    ACTIVE_SET_QUALITY_SPREAD,
    ACTIVE_SET_STICKY_SPREAD,
    STICKY_ORACLE_COUNT_LOCAL_SEARCH,
    P05_SLACK_AWARE_ACTIVE_SET,
    POWER_RESPONSE_AWARE_LOAD_BALANCE,
    BOUNDED_MOVE_SERVED_SET,
    ORACLE_SCORE_TOPK_ACTIVE_SET,
)
SUPPORTED_ASSOCIATION_POLICIES = (
    FIXED_HOLD_CURRENT,
    *RA_EE_08_PROPOSAL_POLICIES,
    PER_USER_GREEDY_BEST_BEAM,
)


@dataclass(frozen=True)
class _RAEE08Settings:
    method_label: str
    implementation_sublabel: str
    audit: _AuditSettings
    bucket_specs: tuple[_BucketSpec, ...]
    matched_control_association_policy: str
    candidate_association_policies: tuple[str, ...]
    oracle_association_policies: tuple[str, ...]
    predeclared_primary_association_policy: str | None
    deployable_allocators: tuple[str, ...]
    primary_deployable_allocator: str
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
    p05_trim_max_moves: int
    local_search_max_moves: int
    dp_max_profile_count: int

def _ra_ee_08_value(cfg: dict[str, Any]) -> dict[str, Any]:
    value = (
        cfg.get("resolved_assumptions", {})
        .get("ra_ee_08_offline_association_reevaluation", {})
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
            trajectory_families=("offline-association-reevaluation",),
            evaluation_seed_set=_tuple_ints(
                calibration.get("evaluation_seed_set"),
                _tuple_ints(seeds.get("evaluation_seed_set"), DEFAULT_CALIBRATION_SEEDS),
            ),
        ),
        _BucketSpec(
            name=HELD_OUT_BUCKET,
            trajectory_families=("offline-association-reevaluation-heldout",),
            evaluation_seed_set=_tuple_ints(
                held_out.get("evaluation_seed_set"),
                DEFAULT_HELD_OUT_SEEDS,
            ),
        ),
    )


def _validate_association_policies(name: str, policies: tuple[str, ...]) -> None:
    unsupported = sorted(set(policies) - set(SUPPORTED_ASSOCIATION_POLICIES))
    if unsupported:
        raise ValueError(f"Unsupported RA-EE-08 {name}: {unsupported!r}")


def _settings_from_config(cfg: dict[str, Any]) -> _RAEE08Settings:
    gate = _ra_ee_08_value(cfg)
    power = _power_surface_value(cfg)
    seeds = get_seeds(cfg)

    levels_raw = gate.get(
        "codebook_levels_w",
        power.get("power_codebook_levels_w", [0.5, 0.75, 1.0, 1.5, 2.0]),
    )
    levels = tuple(float(level) for level in levels_raw)
    if not levels:
        raise ValueError("RA-EE-08 requires at least one power codebook level.")
    if tuple(sorted(levels)) != levels:
        raise ValueError(f"RA-EE-08 codebook levels must be sorted, got {levels!r}.")

    matched_control = str(
        gate.get("matched_control_association_policy", FIXED_HOLD_CURRENT)
    )
    candidates = _tuple_strings(
        gate.get("candidate_association_policies"),
        RA_EE_08_PROPOSAL_POLICIES,
    )
    oracle_policies = _tuple_strings(
        gate.get("oracle_association_policies"),
        candidates,
    )
    _validate_association_policies("matched control policy", (matched_control,))
    _validate_association_policies("candidate policies", candidates)
    _validate_association_policies("oracle policies", oracle_policies)
    if matched_control in candidates:
        raise ValueError("RA-EE-08 matched control must not also be a candidate.")

    deployable = _tuple_strings(
        gate.get("deployable_allocator_candidates"),
        DEPLOYABLE_ALLOCATORS,
    )
    unsupported_allocators = sorted(set(deployable) - set(DEPLOYABLE_ALLOCATORS))
    if unsupported_allocators:
        raise ValueError(
            f"Unsupported RA-EE-08 deployable allocators: {unsupported_allocators!r}"
        )
    primary_allocator = str(
        gate.get("primary_deployable_allocator", DETERMINISTIC_HYBRID)
    )
    if primary_allocator not in deployable:
        raise ValueError(
            "RA-EE-08 primary deployable allocator must be listed in candidates."
        )

    primary_policy = gate.get("predeclared_primary_association_policy")
    primary_policy = None if primary_policy in (None, "none") else str(primary_policy)
    if primary_policy is not None and primary_policy not in candidates:
        raise ValueError(
            "RA-EE-08 predeclared primary association policy must be one of "
            f"{candidates!r}, got {primary_policy!r}."
        )

    min_active = int(gate.get("min_active_beams", 2))
    max_active = int(gate.get("max_active_beams", 8))
    if min_active < 1 or max_active < min_active:
        raise ValueError(
            f"Invalid RA-EE-08 active-beam bounds: min={min_active}, max={max_active}."
        )

    audit = _AuditSettings(
        method_label=str(gate.get("method_label", RA_EE_08_METHOD_LABEL)),
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
    return _RAEE08Settings(
        method_label=str(gate.get("method_label", RA_EE_08_METHOD_LABEL)),
        implementation_sublabel=str(
            gate.get(
                "implementation_sublabel",
                "RA-EE-08 offline association re-evaluation gate",
            )
        ),
        audit=audit,
        bucket_specs=_bucket_specs_from_config(gate, seeds),
        matched_control_association_policy=matched_control,
        candidate_association_policies=candidates,
        oracle_association_policies=oracle_policies,
        predeclared_primary_association_policy=primary_policy,
        deployable_allocators=deployable,
        primary_deployable_allocator=primary_allocator,
        min_active_beams=min_active,
        max_active_beams=max_active,
        target_users_per_active_beam=int(gate.get("target_users_per_active_beam", 16)),
        load_cap_overflow_users=int(gate.get("load_cap_overflow_users", 2)),
        candidate_max_demoted_beams=int(gate.get("candidate_max_demoted_beams", 3)),
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
        p05_trim_max_moves=int(gate.get("p05_trim_max_moves", 6)),
        local_search_max_moves=int(gate.get("local_search_max_moves", 8)),
        dp_max_profile_count=int(gate.get("dp_max_profile_count", 512)),
    )


def _ra_ee_06_settings(settings: _RAEE08Settings) -> _RAEE06Settings:
    return _RAEE06Settings(
        method_label=settings.method_label,
        implementation_sublabel=settings.implementation_sublabel,
        audit=settings.audit,
        bucket_specs=settings.bucket_specs,
        matched_control_association_policy=settings.matched_control_association_policy,
        candidate_association_policies=tuple(
            policy
            for policy in settings.candidate_association_policies
            if policy in ACTIVE_SET_POLICIES
        ),
        diagnostic_association_policies=(),
        oracle_association_policies=tuple(
            policy for policy in settings.oracle_association_policies if policy in ACTIVE_SET_POLICIES
        ),
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


def _ra_ee_06b_settings(settings: _RAEE08Settings) -> _RAEE06BSettings:
    return _RAEE06BSettings(
        method_label=settings.method_label,
        implementation_sublabel=settings.implementation_sublabel,
        audit=settings.audit,
        bucket_specs=settings.bucket_specs,
        matched_control_association_policy=settings.matched_control_association_policy,
        candidate_association_policies=tuple(
            policy
            for policy in settings.candidate_association_policies
            if policy in RA_EE_06B_PROPOSAL_POLICIES
        ),
        diagnostic_association_policies=(),
        oracle_association_policies=tuple(
            policy
            for policy in settings.oracle_association_policies
            if policy in RA_EE_06B_PROPOSAL_POLICIES
        ),
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


def _select_actions_for_association_policy(
    policy: str,
    *,
    user_states: list[Any],
    masks: list[Any],
    current_assignments: np.ndarray,
    settings: _RAEE08Settings,
) -> np.ndarray:
    if policy in ACTIVE_SET_POLICIES or policy in {
        FIXED_HOLD_CURRENT,
        PER_USER_GREEDY_BEST_BEAM,
    }:
        return _select_ra_ee_06_actions(
            policy,
            user_states=user_states,
            masks=masks,
            current_assignments=current_assignments,
            settings=_ra_ee_06_settings(settings),
        )
    if policy in RA_EE_06B_PROPOSAL_POLICIES:
        return _select_ra_ee_06b_actions(
            policy,
            user_states=user_states,
            masks=masks,
            current_assignments=current_assignments,
            settings=_ra_ee_06b_settings(settings),
        )
    raise ValueError(f"Unsupported RA-EE-08 association policy {policy!r}.")


def _association_action_contract(policy: str, settings: _RAEE08Settings) -> str:
    if policy == settings.matched_control_association_policy:
        return "fixed-by-trajectory"
    if policy == PER_USER_GREEDY_BEST_BEAM:
        return "per-user-one-hot-greedy-diagnostic"
    return "deterministic-active-set-served-set-proposal-rule"

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT_DIR",
    "RA_EE_08_ASSOC_ORACLE_CONSTRAINED",
    "RA_EE_08_ASSOC_ORACLE_DEPLOYABLE",
    "RA_EE_08_CANDIDATE",
    "RA_EE_08_FIXED_CONSTRAINED_ORACLE",
    "RA_EE_08_FIXED_DEPLOYABLE_CONTROL",
    "RA_EE_08_FIXED_SAFE_GREEDY",
    "RA_EE_08_METHOD_LABEL",
    "RA_EE_08_PROPOSAL_POLICIES",
    "RA_EE_08_PROPOSAL_SAFE_GREEDY",
    "SUPPORTED_ASSOCIATION_POLICIES",
    "_RAEE08Settings",
    "_association_action_contract",
    "_ra_ee_06_settings",
    "_ra_ee_06b_settings",
    "_select_actions_for_association_policy",
    "_settings_from_config",
    "_validate_association_policies",
]
