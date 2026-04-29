"""Resource-share allocation, throughput recompute, and accounting for RA-EE-09."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ._ra_ee_09_common import (
    RA_EE_09_CANDIDATE,
    RA_EE_09_CONTROL,
    RA_EE_09_POWER_ALLOCATOR_ID,
    _RAEE09Settings,
    _ResourceSettings,
    _hash_array,
    _hash_int_array,
)
from .ra_ee_02_oracle_power_allocation import (
    _StepSnapshot,
    _evaluate_power_vector,
    _format_vector,
)
from .ra_ee_07_constrained_power_allocator_distillation import (
    _load_stats,
    _power_delta_fields,
)


def _equal_share_resource_fractions(snapshot: _StepSnapshot) -> np.ndarray:
    shares = np.zeros(snapshot.assignments.shape, dtype=np.float64)
    for uid, beam_idx_raw in enumerate(snapshot.assignments.tolist()):
        beam_idx = int(beam_idx_raw)
        if beam_idx < 0 or beam_idx >= snapshot.beam_loads.size:
            continue
        load = max(float(snapshot.beam_loads[beam_idx]), 1.0)
        shares[uid] = 1.0 / load
    return shares


def _project_resource_shares_to_bounds(
    proposed: np.ndarray,
    *,
    lower: float,
    upper: float,
    target_sum: float,
    tolerance: float,
) -> np.ndarray:
    shares = np.clip(np.asarray(proposed, dtype=np.float64), lower, upper)
    if shares.size == 0:
        return shares

    for _ in range(shares.size + 2):
        residual = float(target_sum - np.sum(shares, dtype=np.float64))
        if abs(residual) <= tolerance:
            break
        if residual > 0.0:
            capacity = upper - shares
            eligible = np.flatnonzero(capacity > tolerance)
            sign = 1.0
        else:
            capacity = shares - lower
            eligible = np.flatnonzero(capacity > tolerance)
            sign = -1.0
        if eligible.size == 0:
            break
        total_capacity = float(np.sum(capacity[eligible], dtype=np.float64))
        if total_capacity <= tolerance:
            break
        delta = min(abs(residual), total_capacity)
        shares[eligible] += sign * delta * capacity[eligible] / total_capacity

    residual = float(target_sum - np.sum(shares, dtype=np.float64))
    if abs(residual) > 0.0:
        if residual > 0.0:
            capacity = upper - shares
            sign = 1.0
        else:
            capacity = shares - lower
            sign = -1.0
        for idx in np.flatnonzero(capacity >= abs(residual) - tolerance).tolist():
            shares[int(idx)] += residual
            residual = float(target_sum - np.sum(shares, dtype=np.float64))
            if abs(residual) <= tolerance:
                break
        if abs(residual) > tolerance:
            for idx in np.flatnonzero(capacity > tolerance).tolist():
                step = sign * min(abs(residual), float(capacity[int(idx)]))
                shares[int(idx)] += step
                residual = float(target_sum - np.sum(shares, dtype=np.float64))
                if abs(residual) <= tolerance:
                    break

    return np.clip(shares, lower, upper)


def _bounded_qos_slack_resource_share_allocator(
    snapshot: _StepSnapshot,
    power_vector: np.ndarray,
    settings: _ResourceSettings,
) -> np.ndarray:
    """Allocate per-user resource shares from current fixed-association state only."""
    shares = np.zeros(snapshot.assignments.shape, dtype=np.float64)
    tol = settings.resource_sum_tolerance
    assignments = snapshot.assignments.astype(np.int32, copy=False)
    for beam_idx in np.flatnonzero(snapshot.active_mask).tolist():
        user_indices = np.flatnonzero(assignments == int(beam_idx))
        if user_indices.size == 0:
            continue
        load = float(user_indices.size)
        equal_share = 1.0 / load
        lower = settings.per_user_min_equal_share_multiplier * equal_share
        upper = min(settings.per_user_max_equal_share_multiplier * equal_share, 1.0)
        equal = np.full(user_indices.shape, equal_share, dtype=np.float64)
        if user_indices.size == 1:
            shares[user_indices] = 1.0
            continue

        power_w = max(float(power_vector[int(beam_idx)]), 0.0)
        gamma = snapshot.unit_snr_by_user[user_indices] * power_w
        spectral_efficiency = np.log2(1.0 + np.maximum(gamma, 0.0))
        equal_throughputs = snapshot.bandwidth_hz * equal * spectral_efficiency
        center = float(np.median(equal_throughputs))
        receiver_deficit = np.maximum(center - equal_throughputs, 0.0)
        donor_slack = np.maximum(equal_throughputs - center, 0.0)

        proposed = equal.copy()
        receiver_mask = receiver_deficit > tol
        donor_mask = donor_slack > tol
        if np.any(receiver_mask) and np.any(donor_mask):
            receiver_capacity = np.maximum(upper - equal, 0.0) * receiver_mask
            donor_capacity = np.maximum(equal - lower, 0.0) * donor_mask
            total_receiver_capacity = float(
                np.sum(receiver_capacity, dtype=np.float64)
            )
            total_donor_capacity = float(np.sum(donor_capacity, dtype=np.float64))
            if total_receiver_capacity > tol and total_donor_capacity > tol:
                pressure = float(
                    np.sum(receiver_deficit, dtype=np.float64)
                    / max(
                        np.sum(receiver_deficit + donor_slack, dtype=np.float64),
                        tol,
                    )
                )
                move_budget = min(
                    total_receiver_capacity,
                    total_donor_capacity,
                ) * min(max(pressure, 0.0), 1.0)
                receiver_weights = receiver_deficit / max(
                    float(np.sum(receiver_deficit, dtype=np.float64)),
                    tol,
                )
                donor_weights = donor_slack / max(
                    float(np.sum(donor_slack, dtype=np.float64)),
                    tol,
                )
                proposed = proposed + move_budget * receiver_weights
                proposed = proposed - move_budget * donor_weights

        shares[user_indices] = _project_resource_shares_to_bounds(
            proposed,
            lower=lower,
            upper=upper,
            target_sum=settings.per_beam_budget,
            tolerance=tol,
        )
    return shares


def _compute_user_throughputs_from_resource(
    snapshot: _StepSnapshot,
    power_vector: np.ndarray,
    user_resource_fractions: np.ndarray,
) -> np.ndarray:
    throughputs = np.zeros(snapshot.assignments.shape, dtype=np.float64)
    for uid, beam_idx_raw in enumerate(snapshot.assignments.tolist()):
        beam_idx = int(beam_idx_raw)
        if beam_idx < 0 or beam_idx >= snapshot.beam_loads.size:
            continue
        power_w = max(float(power_vector[beam_idx]), 0.0)
        snr = float(snapshot.unit_snr_by_user[uid]) * power_w
        rho = max(float(user_resource_fractions[uid]), 0.0)
        throughputs[uid] = snapshot.bandwidth_hz * rho * math.log2(1.0 + snr)
    return throughputs


def _audit_resource_accounting(
    snapshot: _StepSnapshot,
    user_resource_fractions: np.ndarray,
    settings: _ResourceSettings,
    *,
    inactive_beam_resource_usage: np.ndarray | None = None,
) -> dict[str, Any]:
    active_mask = snapshot.active_mask.astype(bool, copy=False)
    tol = settings.resource_sum_tolerance
    per_beam_usage = np.zeros(snapshot.beam_loads.shape, dtype=np.float64)
    for uid, beam_idx_raw in enumerate(snapshot.assignments.tolist()):
        beam_idx = int(beam_idx_raw)
        if 0 <= beam_idx < per_beam_usage.size:
            per_beam_usage[beam_idx] += float(user_resource_fractions[uid])

    inactive_usage = np.zeros(snapshot.beam_loads.shape, dtype=np.float64)
    if inactive_beam_resource_usage is not None:
        inactive_usage = np.asarray(inactive_beam_resource_usage, dtype=np.float64)
        if inactive_usage.shape != per_beam_usage.shape:
            raise ValueError(
                "inactive_beam_resource_usage must match beam shape, got "
                f"{inactive_usage.shape!r} expected {per_beam_usage.shape!r}."
            )
    inactive_total_usage = per_beam_usage + inactive_usage

    active_sums = per_beam_usage[active_mask]
    active_errors = (
        np.abs(active_sums - settings.per_beam_budget)
        if active_sums.size
        else np.zeros(0, dtype=np.float64)
    )
    overuse = np.maximum(active_sums - settings.per_beam_budget, 0.0)
    underuse = np.maximum(settings.per_beam_budget - active_sums, 0.0)
    inactive_nonzero = bool(np.any(np.abs(inactive_total_usage[~active_mask]) > tol))

    min_violations = 0
    max_violations = 0
    for uid, beam_idx_raw in enumerate(snapshot.assignments.tolist()):
        beam_idx = int(beam_idx_raw)
        if beam_idx < 0 or beam_idx >= snapshot.beam_loads.size:
            continue
        load = max(float(snapshot.beam_loads[beam_idx]), 1.0)
        equal_share = 1.0 / load
        share = float(user_resource_fractions[uid])
        min_share = settings.per_user_min_equal_share_multiplier * equal_share
        max_share = min(settings.per_user_max_equal_share_multiplier * equal_share, 1.0)
        if share < min_share - tol:
            min_violations += 1
        if share > max_share + tol:
            max_violations += 1

    resource_sum_violation_count = int(np.sum(active_errors > tol))
    budget_violation_count = (
        resource_sum_violation_count
        + int(inactive_nonzero)
        + int(min_violations)
        + int(max_violations)
    )
    total_budget = settings.per_beam_budget * int(np.count_nonzero(active_mask))
    total_usage = float(np.sum(per_beam_usage[active_mask], dtype=np.float64))
    return {
        "resource_unit": settings.resource_unit,
        "resource_allocator_id": settings.resource_allocator_id,
        "resource_fractions": _format_vector(user_resource_fractions),
        "per_beam_resource_sum": _format_vector(per_beam_usage),
        "active_beam_resource_sum_min": (
            None if active_sums.size == 0 else float(np.min(active_sums))
        ),
        "active_beam_resource_sum_max": (
            None if active_sums.size == 0 else float(np.max(active_sums))
        ),
        "active_beam_resource_sum_max_abs_error": (
            0.0 if active_errors.size == 0 else float(np.max(active_errors))
        ),
        "active_beam_resource_sum_exact": bool(
            active_errors.size == 0 or np.max(active_errors) <= tol
        ),
        "active_beam_resource_sum_violation_count": resource_sum_violation_count,
        "inactive_beam_resource_usage": _format_vector(inactive_total_usage),
        "inactive_beam_nonzero_resource": inactive_nonzero,
        "inactive_beam_nonzero_resource_count": int(
            np.sum(np.abs(inactive_total_usage[~active_mask]) > tol)
        ),
        "per_user_min_resource_violation_count": int(min_violations),
        "per_user_max_resource_violation_count": int(max_violations),
        "resource_budget_violation": bool(budget_violation_count > 0),
        "resource_budget_violation_count": int(budget_violation_count),
        "resource_overuse_max": 0.0 if overuse.size == 0 else float(np.max(overuse)),
        "resource_underuse_max": 0.0 if underuse.size == 0 else float(np.max(underuse)),
        "resource_unused_total": float(np.sum(underuse, dtype=np.float64)),
        "total_resource_budget": total_budget,
        "total_resource_usage": total_usage,
        "per_beam_budget": settings.per_beam_budget,
    }


def _resource_control_row(
    *,
    snapshot: _StepSnapshot,
    bucket: str,
    power_vector: np.ndarray,
    selected_power_profile: str,
    selected_allocator_candidate: str,
    allocator_rejection_reason: str,
    evaluated_allocator_profile_count: int,
    safe_vector: np.ndarray,
    settings: _RAEE09Settings,
) -> dict[str, Any]:
    base = _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics=RA_EE_09_CONTROL,
        selected_power_profile=selected_power_profile,
        power_vector=power_vector,
        settings=settings.power_settings.audit,
    )
    existing_throughputs = np.asarray(base["_user_throughputs"], dtype=np.float64)
    shares = _equal_share_resource_fractions(snapshot)
    resource_throughputs = _compute_user_throughputs_from_resource(
        snapshot,
        power_vector,
        shares,
    )
    accounting = _audit_resource_accounting(snapshot, shares, settings.resource)
    parity_abs = np.abs(resource_throughputs - existing_throughputs)
    parity_max_abs = 0.0 if parity_abs.size == 0 else float(np.max(parity_abs))
    parity_sum_delta = float(
        np.sum(resource_throughputs, dtype=np.float64)
        - np.sum(existing_throughputs, dtype=np.float64)
    )
    existing_p05 = float(np.percentile(existing_throughputs, 5))
    resource_p05 = float(np.percentile(resource_throughputs, 5))
    total_active_power = float(base["total_active_beam_power_w"])
    existing_ee = (
        None
        if total_active_power <= 0.0
        else float(np.sum(existing_throughputs, dtype=np.float64) / total_active_power)
    )
    resource_ee = (
        None
        if total_active_power <= 0.0
        else float(np.sum(resource_throughputs, dtype=np.float64) / total_active_power)
    )

    beam_throughputs = np.zeros(snapshot.beam_loads.shape, dtype=np.float64)
    for uid, beam_idx_raw in enumerate(snapshot.assignments.tolist()):
        beam_idx = int(beam_idx_raw)
        if 0 <= beam_idx < beam_throughputs.size:
            beam_throughputs[beam_idx] += float(resource_throughputs[uid])
    active_thr = beam_throughputs[snapshot.active_mask]
    active_gap = (
        0.0
        if active_thr.size < 2
        else float(np.max(active_thr) - np.min(active_thr))
    )

    row = dict(base)
    row["_user_throughputs"] = resource_throughputs
    row["sum_user_throughput_bps"] = float(np.sum(resource_throughputs, dtype=np.float64))
    row["throughput_mean_user_step_bps"] = (
        row["sum_user_throughput_bps"] / max(resource_throughputs.size, 1)
    )
    row["throughput_p05_user_step_bps"] = resource_p05
    row["EE_system_bps_per_w"] = resource_ee
    row["active_beam_throughput_gap_bps"] = active_gap
    row["evaluation_bucket"] = bucket
    row["association_mode"] = "fixed-replay"
    row["association_policy"] = str(snapshot.trajectory_policy)
    row["source_association_policy"] = str(snapshot.trajectory_policy)
    row["association_role"] = "fixed-association-equal-share-control"
    row["association_action_contract"] = "fixed-by-trajectory"
    row["assignment_hash"] = _hash_int_array(snapshot.assignments)
    row["association_trajectory_id"] = str(snapshot.trajectory_policy)
    row["association_trajectory_hash"] = row["assignment_hash"]
    row["resource_allocation_changes_assignment"] = False
    row["resource_allocation_changes_handover_trajectory"] = False
    row["active_set_size"] = row["active_beam_count"]
    row["served_set_size"] = row["served_count"]
    row["requested_power_vector_w"] = _format_vector(power_vector)
    row["effective_power_vector_w"] = row["beam_transmit_power_w"]
    row["power_vector_hash"] = _hash_array(power_vector)
    row["control_power_vector_hash"] = row["power_vector_hash"]
    row["same_power_vector_as_control"] = True
    row["power_allocator"] = "deployable-stronger-power-allocator"
    row["power_allocator_id"] = RA_EE_09_POWER_ALLOCATOR_ID
    row["selected_profile"] = row["selected_power_profile"]
    row["allocator_label"] = settings.power_settings.primary_deployable_allocator
    row["selected_allocator_candidate"] = selected_allocator_candidate
    row["allocator_rejection_reason"] = allocator_rejection_reason
    row["evaluated_allocator_profile_count"] = int(evaluated_allocator_profile_count)
    row.update(_power_delta_fields(power_vector, safe_vector, snapshot.active_mask))
    row.update(_load_stats(snapshot))
    row["per_user_quality"] = _format_vector(snapshot.unit_snr_by_user)
    row["total_active_power"] = row["total_active_beam_power_w"]
    row["EE_denominator_w"] = row["total_active_beam_power_w"]
    row["diagnostic_only"] = False
    row["primary_candidate"] = False
    row["candidate_allocator_enabled"] = False
    row["resource_accounting_enabled"] = True
    row["rb_bandwidth_allocation_enabled"] = True
    row["learned_association_enabled"] = False
    row["learned_hierarchical_RL_enabled"] = False
    row["joint_association_power_training_enabled"] = False
    row["catfish_enabled"] = False
    row["multi_catfish_enabled"] = False
    row["phase03c_continuation_enabled"] = False
    row["oracle_labels_used_for_runtime_decision"] = False
    row["future_outcomes_used_for_runtime_decision"] = False
    row["held_out_answers_used_for_runtime_decision"] = False
    row["resource_allocation_after_power_vector_selection"] = True
    row["resource_allocation_feedback_to_power_decision"] = False
    row["scalar_reward_success_basis"] = False
    row["per_user_EE_credit_success_basis"] = False
    row["physical_energy_saving_claim"] = False
    row["hobs_optimizer_claim"] = False
    row["full_RA_EE_MODQN_claim"] = False
    row["throughput_formula_version"] = settings.resource.throughput_formula_version
    row["control_formula"] = "R_i(t) = B / N_b(t) * log2(1 + gamma_i,b(t))"
    row["generalized_formula"] = "R_i(t) = B * rho_i,b(t) * log2(1 + gamma_i,b(t))"
    row["equal_share_formula"] = "rho_i,b(t) = 1 / N_b(t)"
    row["equal_share_throughput_parity"] = bool(
        parity_max_abs <= settings.resource.resource_sum_tolerance
    )
    row["resource_throughput_max_abs_delta_vs_existing_bps"] = parity_max_abs
    row["resource_throughput_sum_delta_vs_existing_bps"] = parity_sum_delta
    row["resource_throughput_p05_delta_vs_existing_bps"] = resource_p05 - existing_p05
    row["resource_EE_system_delta_vs_existing_bps_per_w"] = (
        None if existing_ee is None or resource_ee is None else resource_ee - existing_ee
    )
    row.update(accounting)
    row["accepted_flag"] = True
    row["rejection_reason"] = "control-parity-row"
    return row


def _resource_candidate_row(
    *,
    snapshot: _StepSnapshot,
    bucket: str,
    power_vector: np.ndarray,
    selected_power_profile: str,
    selected_allocator_candidate: str,
    allocator_rejection_reason: str,
    evaluated_allocator_profile_count: int,
    safe_vector: np.ndarray,
    settings: _RAEE09Settings,
) -> dict[str, Any]:
    base = _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics=RA_EE_09_CANDIDATE,
        selected_power_profile=selected_power_profile,
        power_vector=power_vector,
        settings=settings.power_settings.audit,
    )
    existing_throughputs = np.asarray(base["_user_throughputs"], dtype=np.float64)
    shares = _bounded_qos_slack_resource_share_allocator(
        snapshot,
        power_vector,
        settings.resource,
    )
    resource_throughputs = _compute_user_throughputs_from_resource(
        snapshot,
        power_vector,
        shares,
    )
    accounting = _audit_resource_accounting(snapshot, shares, settings.resource)
    resource_abs = np.abs(resource_throughputs - existing_throughputs)
    resource_max_abs = 0.0 if resource_abs.size == 0 else float(np.max(resource_abs))
    resource_sum_delta = float(
        np.sum(resource_throughputs, dtype=np.float64)
        - np.sum(existing_throughputs, dtype=np.float64)
    )
    existing_p05 = float(np.percentile(existing_throughputs, 5))
    resource_p05 = float(np.percentile(resource_throughputs, 5))
    total_active_power = float(base["total_active_beam_power_w"])
    existing_ee = (
        None
        if total_active_power <= 0.0
        else float(np.sum(existing_throughputs, dtype=np.float64) / total_active_power)
    )
    resource_ee = (
        None
        if total_active_power <= 0.0
        else float(np.sum(resource_throughputs, dtype=np.float64) / total_active_power)
    )

    beam_throughputs = np.zeros(snapshot.beam_loads.shape, dtype=np.float64)
    for uid, beam_idx_raw in enumerate(snapshot.assignments.tolist()):
        beam_idx = int(beam_idx_raw)
        if 0 <= beam_idx < beam_throughputs.size:
            beam_throughputs[beam_idx] += float(resource_throughputs[uid])
    active_thr = beam_throughputs[snapshot.active_mask]
    active_gap = (
        0.0
        if active_thr.size < 2
        else float(np.max(active_thr) - np.min(active_thr))
    )

    power_hash = _hash_array(power_vector)
    assignment_hash = _hash_int_array(snapshot.assignments)
    row = dict(base)
    row["_user_throughputs"] = resource_throughputs
    row["sum_user_throughput_bps"] = float(np.sum(resource_throughputs, dtype=np.float64))
    row["throughput_mean_user_step_bps"] = (
        row["sum_user_throughput_bps"] / max(resource_throughputs.size, 1)
    )
    row["throughput_p05_user_step_bps"] = resource_p05
    row["EE_system_bps_per_w"] = resource_ee
    row["active_beam_throughput_gap_bps"] = active_gap
    row["evaluation_bucket"] = bucket
    row["association_mode"] = "fixed-replay"
    row["association_policy"] = str(snapshot.trajectory_policy)
    row["source_association_policy"] = str(snapshot.trajectory_policy)
    row["association_role"] = "fixed-association-bounded-resource-candidate"
    row["association_action_contract"] = "fixed-by-trajectory"
    row["assignment_hash"] = assignment_hash
    row["association_trajectory_id"] = str(snapshot.trajectory_policy)
    row["association_trajectory_hash"] = assignment_hash
    row["assignment_hash_before_resource_allocation"] = assignment_hash
    row["assignment_hash_after_resource_allocation"] = assignment_hash
    row["resource_allocation_changes_assignment"] = False
    row["resource_allocation_changes_handover_trajectory"] = False
    row["handover_count_before_resource_allocation"] = int(snapshot.handover_count)
    row["handover_count_after_resource_allocation"] = int(snapshot.handover_count)
    row["active_set_size"] = row["active_beam_count"]
    row["served_set_size"] = row["served_count"]
    row["requested_power_vector_w"] = _format_vector(power_vector)
    row["effective_power_vector_w"] = row["beam_transmit_power_w"]
    row["power_vector_hash"] = power_hash
    row["control_power_vector_hash"] = power_hash
    row["power_vector_hash_before_resource_allocation"] = power_hash
    row["power_vector_hash_after_resource_allocation"] = power_hash
    row["same_power_vector_as_control"] = True
    row["power_allocator"] = "deployable-stronger-power-allocator"
    row["power_allocator_id"] = RA_EE_09_POWER_ALLOCATOR_ID
    row["selected_profile"] = row["selected_power_profile"]
    row["allocator_label"] = settings.power_settings.primary_deployable_allocator
    row["selected_allocator_candidate"] = selected_allocator_candidate
    row["allocator_rejection_reason"] = allocator_rejection_reason
    row["evaluated_allocator_profile_count"] = int(evaluated_allocator_profile_count)
    row.update(_power_delta_fields(power_vector, safe_vector, snapshot.active_mask))
    row.update(_load_stats(snapshot))
    row["per_user_quality"] = _format_vector(snapshot.unit_snr_by_user)
    row["total_active_power"] = row["total_active_beam_power_w"]
    row["EE_denominator_w"] = row["total_active_beam_power_w"]
    row["diagnostic_only"] = False
    row["primary_candidate"] = True
    row["candidate_allocator_enabled"] = True
    row["resource_accounting_enabled"] = True
    row["rb_bandwidth_allocation_enabled"] = True
    row["learned_association_enabled"] = False
    row["learned_hierarchical_RL_enabled"] = False
    row["joint_association_power_training_enabled"] = False
    row["catfish_enabled"] = False
    row["multi_catfish_enabled"] = False
    row["phase03c_continuation_enabled"] = False
    row["oracle_labels_used_for_runtime_decision"] = False
    row["future_outcomes_used_for_runtime_decision"] = False
    row["held_out_answers_used_for_runtime_decision"] = False
    row["resource_allocation_after_power_vector_selection"] = True
    row["resource_allocation_feedback_to_power_decision"] = False
    row["resource_allocator_decision_basis"] = (
        "current_gamma_and_equal_share_throughput_only"
    )
    row["scalar_reward_success_basis"] = False
    row["per_user_EE_credit_success_basis"] = False
    row["physical_energy_saving_claim"] = False
    row["hobs_optimizer_claim"] = False
    row["full_RA_EE_MODQN_claim"] = False
    row["throughput_formula_version"] = settings.resource.throughput_formula_version
    row["control_formula"] = "R_i(t) = B / N_b(t) * log2(1 + gamma_i,b(t))"
    row["generalized_formula"] = "R_i(t) = B * rho_i,b(t) * log2(1 + gamma_i,b(t))"
    row["equal_share_formula"] = "rho_i,b(t) = 1 / N_b(t)"
    row["equal_share_throughput_parity"] = False
    row["resource_throughput_max_abs_delta_vs_existing_bps"] = resource_max_abs
    row["resource_throughput_sum_delta_vs_existing_bps"] = resource_sum_delta
    row["resource_throughput_p05_delta_vs_existing_bps"] = resource_p05 - existing_p05
    row["resource_EE_system_delta_vs_existing_bps_per_w"] = (
        None if existing_ee is None or resource_ee is None else resource_ee - existing_ee
    )
    row.update(accounting)
    row["accepted_flag"] = True
    row["rejection_reason"] = "slice-09d-candidate-awaits-09e-effectiveness-gate"
    return row
