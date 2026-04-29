"""RA-EE-08 deterministic replay and row construction."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from ..config_loader import build_environment
from .ra_ee_02_oracle_power_allocation import (
    _StepSnapshot,
    _evaluate_power_vector,
    _format_vector,
    _power_vector_for_candidate,
    _select_oracle_step,
)
from .ra_ee_06b_association_proposal_refinement import (
    _AssociationTrace,
    _p05_ratio_and_slack,
    _policy_label,
    _trace_fields,
    _trace_for_actions,
)
from .ra_ee_07_constrained_power_allocator_distillation import (
    _deployable_allocator_results,
    _deployable_row,
    _load_stats,
    _power_delta_fields,
    _row_ee,
    _safe_greedy_row,
)
from .ra_ee_08_protocol import (
    FIXED_HOLD_CURRENT,
    PER_USER_GREEDY_BEST_BEAM,
    RA_EE_08_ASSOC_ORACLE_CONSTRAINED,
    RA_EE_08_ASSOC_ORACLE_DEPLOYABLE,
    RA_EE_08_CANDIDATE,
    RA_EE_08_FIXED_CONSTRAINED_ORACLE,
    RA_EE_08_FIXED_DEPLOYABLE_CONTROL,
    RA_EE_08_FIXED_SAFE_GREEDY,
    RA_EE_08_PROPOSAL_SAFE_GREEDY,
    _RAEE08Settings,
    _association_action_contract,
    _ra_ee_06b_settings,
    _select_actions_for_association_policy,
)

def _rollout_association_trajectories(
    *,
    cfg: dict[str, Any],
    settings: _RAEE08Settings,
    max_steps: int | None,
) -> tuple[
    dict[str, dict[int, list[np.ndarray]]],
    dict[str, dict[str, Any]],
    dict[tuple[str, int, int], _AssociationTrace],
]:
    trajectories: dict[str, dict[int, list[np.ndarray]]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    traces_by_key: dict[tuple[str, int, int], _AssociationTrace] = {}
    base_policies = (
        settings.matched_control_association_policy,
        *settings.candidate_association_policies,
        *settings.oracle_association_policies,
    )
    unique_policies = tuple(dict.fromkeys(base_policies))
    trace_settings = _ra_ee_06b_settings(settings)

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
                    traces_by_key[(label, int(eval_seed), int(result.step_index))] = (
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
                            settings=trace_settings,
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
                "association_action_contract": _association_action_contract(
                    policy,
                    settings,
                ),
                "candidate_role": (
                    "matched-control"
                    if policy == settings.matched_control_association_policy
                    else "candidate-proposal-rule"
                ),
            }
    return trajectories, metadata, traces_by_key


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


def _power_allocator_label(power_semantics: str) -> str:
    if power_semantics in {RA_EE_08_FIXED_SAFE_GREEDY, RA_EE_08_PROPOSAL_SAFE_GREEDY}:
        return "safe-greedy-power-allocator"
    if power_semantics == RA_EE_08_FIXED_DEPLOYABLE_CONTROL:
        return "deployable-stronger-power-allocator"
    if power_semantics == RA_EE_08_CANDIDATE:
        return "deployable-stronger-power-allocator"
    if power_semantics == RA_EE_08_ASSOC_ORACLE_DEPLOYABLE:
        return "deployable-stronger-power-allocator"
    if power_semantics in {
        RA_EE_08_FIXED_CONSTRAINED_ORACLE,
        RA_EE_08_ASSOC_ORACLE_CONSTRAINED,
    }:
        return "constrained-power-oracle-diagnostic"
    return "unknown"


def _decorate_power_row(
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
    row["selected_allocator_candidate"] = ""
    row["accepted_allocator_move_count"] = 0
    row["rejected_allocator_move_count"] = 0
    row["allocator_rejection_reason"] = "diagnostic-only"
    row["evaluated_allocator_profile_count"] = 1
    row.update(
        _power_delta_fields(
            requested_power_vector,
            baseline_power_vector,
            snapshot.active_mask,
        )
    )
    row.update(_load_stats(snapshot))
    row["per_user_quality"] = _format_vector(snapshot.unit_snr_by_user)
    row["valid_beam_count"] = ""
    row["valid_beam_count_mean"] = None
    row["tail_user_ids"] = ""
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


def _constrained_oracle_row(
    *,
    snapshot: _StepSnapshot,
    control_row: dict[str, Any],
    settings: _RAEE08Settings,
    power_semantics: str,
    bucket: str,
    association_policy: str,
    association_role: str,
    association_action_contract: str,
) -> dict[str, Any]:
    base = _select_oracle_step(
        snapshot=snapshot,
        control_row=control_row,
        settings=settings.audit,
    )
    vector = np.fromstring(str(base["beam_transmit_power_w"]), sep=" ", dtype=np.float64)
    if vector.size != snapshot.beam_loads.size:
        vector = _power_vector_for_candidate(snapshot, settings.audit, "fixed-control")
    row = _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics=power_semantics,
        selected_power_profile=str(base["selected_power_profile"]),
        power_vector=vector,
        settings=settings.audit,
    )
    fixed = _power_vector_for_candidate(snapshot, settings.audit, "fixed-control")
    row = _decorate_power_row(
        snapshot,
        row=row,
        requested_power_vector=vector,
        baseline_power_vector=fixed,
        bucket=bucket,
        association_policy=association_policy,
        association_role=association_role,
        association_action_contract=association_action_contract,
        allocator_label="constrained-power-oracle-diagnostic",
        diagnostic_only=True,
        primary_candidate=False,
    )
    row["oracle_profile"] = row["selected_power_profile"]
    return row


def _copy_row_for_semantics(
    row: dict[str, Any],
    *,
    trajectory_policy: str,
    power_semantics: str,
    association_policy: str,
    association_role: str,
    association_action_contract: str,
    diagnostic_only: bool,
    primary_candidate: bool,
) -> dict[str, Any]:
    copied = dict(row)
    copied["trajectory_policy"] = trajectory_policy
    copied["power_semantics"] = power_semantics
    copied["association_policy"] = association_policy
    copied["source_association_policy"] = association_policy
    copied["association_role"] = association_role
    copied["association_action_contract"] = association_action_contract
    copied["diagnostic_only"] = diagnostic_only
    copied["primary_candidate"] = primary_candidate
    copied["accepted_flag"] = False
    copied["rejection_reason"] = "diagnostic-only" if diagnostic_only else ""
    copied["power_allocator"] = _power_allocator_label(power_semantics)
    return copied


def _add_oracle_gap_fields(
    rows: list[dict[str, Any]],
    *,
    matched_control_row: dict[str, Any],
    candidate_row: dict[str, Any],
    oracle_row: dict[str, Any] | None,
) -> None:
    if oracle_row is None:
        return
    control_ee = _row_ee(matched_control_row)
    oracle_ee = _row_ee(oracle_row)
    oracle_delta = oracle_ee - control_ee
    for row in rows:
        row["oracle_profile"] = oracle_row["selected_power_profile"]
        row["oracle_gap_bps_per_w"] = (
            None if oracle_ee == -math.inf else oracle_ee - _row_ee(row)
        )
        row["candidate_regret_bps_per_w"] = row["oracle_gap_bps_per_w"]
        row_delta = _row_ee(row) - control_ee
        row["oracle_gap_closed_ratio"] = (
            None if oracle_delta <= 1e-12 else row_delta / oracle_delta
        )
    candidate_row["candidate_regret_bps_per_w"] = oracle_ee - _row_ee(candidate_row)


def _add_comparison_fields(
    row: dict[str, Any],
    *,
    trace: _AssociationTrace | None,
    control_trace: _AssociationTrace | None,
    oracle_trace: _AssociationTrace | None,
    matched_control_row: dict[str, Any],
    primary_candidate_row: dict[str, Any],
    oracle_row: dict[str, Any] | None,
    settings: _RAEE08Settings,
    diagnostic_only: bool,
) -> dict[str, Any]:
    row.update(_trace_fields(trace))
    if control_trace is not None:
        row["control_beam"] = " ".join(
            str(int(value)) for value in control_trace.selected_actions.tolist()
        )
    if oracle_trace is not None:
        row["oracle_beam"] = " ".join(
            str(int(value)) for value in oracle_trace.selected_actions.tolist()
        )
    control_p05 = float(matched_control_row["throughput_p05_user_step_bps"])
    own_p05 = float(row["throughput_p05_user_step_bps"])
    p05_ratio, p05_slack = _p05_ratio_and_slack(
        control_p05_bps=control_p05,
        candidate_p05_bps=own_p05,
        threshold_ratio=settings.audit.p05_min_ratio_vs_control,
    )
    row["matched_control_power_semantics"] = RA_EE_08_FIXED_DEPLOYABLE_CONTROL
    row["matched_control_power_allocator"] = "deployable-stronger-power-allocator"
    row["candidate_power_allocator"] = _power_allocator_label(str(row["power_semantics"]))
    row["power_allocator"] = _power_allocator_label(str(row["power_semantics"]))
    row["same_deployable_allocator_pairing"] = row["power_semantics"] in {
        RA_EE_08_FIXED_DEPLOYABLE_CONTROL,
        RA_EE_08_CANDIDATE,
    }
    row["p05_throughput_control_bps"] = control_p05
    row["p05_throughput_candidate_bps"] = float(
        primary_candidate_row["throughput_p05_user_step_bps"]
    )
    row["p05_throughput_oracle_bps"] = (
        None if oracle_row is None else float(oracle_row["throughput_p05_user_step_bps"])
    )
    row["p05_ratio_vs_matched_control"] = p05_ratio
    row["p05_slack_to_0_95_threshold_bps"] = p05_slack
    row["served_ratio_delta_vs_matched_control"] = float(row["served_ratio"]) - float(
        matched_control_row["served_ratio"]
    )
    row["outage_ratio_delta_vs_matched_control"] = float(row["outage_ratio"]) - float(
        matched_control_row["outage_ratio"]
    )
    row["EE_delta_vs_matched_control"] = (
        None
        if row["EE_system_bps_per_w"] is None
        or matched_control_row["EE_system_bps_per_w"] is None
        else float(row["EE_system_bps_per_w"])
        - float(matched_control_row["EE_system_bps_per_w"])
    )
    row["oracle_selected_association_policy"] = (
        "" if oracle_trace is None else oracle_trace.association_policy
    )
    row["oracle_power_profile"] = (
        "" if oracle_row is None else str(oracle_row["selected_power_profile"])
    )
    row["primary_comparison_no_step_cap_mismatch"] = True
    row["oracle_labels_used_for_runtime_decision"] = False
    row["future_outcomes_used_for_runtime_decision"] = False
    row["held_out_answers_used_for_runtime_decision"] = False
    row["learned_association_enabled"] = False
    row["learned_hierarchical_RL_enabled"] = False
    row["joint_association_power_training_enabled"] = False
    row["catfish_enabled"] = False
    row["multi_catfish_enabled"] = False
    row["rb_bandwidth_allocation_enabled"] = False
    row["scalar_reward_success_basis"] = False
    row["per_user_EE_credit_success_basis"] = False
    row["full_RA_EE_MODQN_claim"] = False
    if diagnostic_only:
        row["accepted_flag"] = False
        row["rejection_reason"] = "diagnostic-only"
        return row
    if row["power_semantics"] == RA_EE_08_FIXED_DEPLOYABLE_CONTROL:
        row["accepted_flag"] = False
        row["rejection_reason"] = "primary-control"
        return row

    reasons: list[str] = []
    if row["EE_delta_vs_matched_control"] is None or float(row["EE_delta_vs_matched_control"]) <= 0.0:
        reasons.append("nonpositive-ee-delta-vs-fixed-deployable")
    if p05_ratio is None or float(p05_ratio) < settings.audit.p05_min_ratio_vs_control:
        reasons.append("p05-ratio-below-threshold")
    if float(row["served_ratio"]) < float(matched_control_row["served_ratio"]) + settings.audit.served_ratio_min_delta_vs_control:
        reasons.append("served-ratio-drop")
    if float(row["outage_ratio"]) > float(matched_control_row["outage_ratio"]) + settings.audit.outage_ratio_max_delta_vs_control:
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
    settings: _RAEE08Settings,
    include_oracle: bool,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[float]]]:
    by_key = _snapshot_index(snapshots)
    rows: list[dict[str, Any]] = []
    user_throughputs_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    eval_cache: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    constrained_oracle_cache: dict[tuple[str, int, int], tuple[dict[str, Any], str]] = {}
    deployable_oracle_cache: dict[tuple[str, int, int], tuple[dict[str, Any], str]] = {}

    def proposal_eval(
        *,
        spec_name: str,
        policy: str,
        seed: int,
        step_index: int,
    ) -> dict[str, Any] | None:
        cache_key = (spec_name, policy, int(seed), int(step_index))
        if cache_key in eval_cache:
            return eval_cache[cache_key]
        label = _policy_label(spec_name, policy)
        snapshot = by_key.get((label, int(seed), int(step_index)))
        if snapshot is None:
            return None
        safe, safe_vector = _safe_greedy_row(
            snapshot,
            settings,
            bucket=spec_name,
            association_policy=policy,
            association_role="diagnostic-proposal-safe-greedy",
            association_action_contract="deterministic-active-set-served-set-proposal-rule",
            power_semantics=RA_EE_08_PROPOSAL_SAFE_GREEDY,
        )
        allocator_results = _deployable_allocator_results(
            snapshot,
            safe,
            safe_vector,
            settings,
        )
        selected = allocator_results[settings.primary_deployable_allocator]
        deployable = _deployable_row(
            snapshot,
            settings,
            result=selected,
            matched_safe_row=safe,
            safe_vector=safe_vector,
            bucket=spec_name,
            association_policy=policy,
            association_role="association-proposal-primary-candidate",
            association_action_contract="deterministic-active-set-served-set-proposal-rule",
            power_semantics=RA_EE_08_CANDIDATE,
            diagnostic_only=False,
            primary_candidate=True,
        )
        safe["power_allocator"] = "safe-greedy-power-allocator"
        deployable["power_allocator"] = "deployable-stronger-power-allocator"
        eval_cache[cache_key] = {
            "snapshot": snapshot,
            "safe": safe,
            "safe_vector": safe_vector,
            "deployable": deployable,
            "trace": traces_by_key.get((label, int(seed), int(step_index))),
        }
        return eval_cache[cache_key]

    def best_association_constrained_oracle(
        *,
        spec_name: str,
        seed: int,
        step_index: int,
        matched_control_row: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str | None]:
        cache_key = (spec_name, int(seed), int(step_index))
        if cache_key in constrained_oracle_cache:
            row, policy = constrained_oracle_cache[cache_key]
            return dict(row), policy
        best: dict[str, Any] | None = None
        best_policy: str | None = None
        for policy in settings.oracle_association_policies:
            label = _policy_label(spec_name, policy)
            snapshot = by_key.get((label, int(seed), int(step_index)))
            if snapshot is None:
                continue
            row = _constrained_oracle_row(
                snapshot=snapshot,
                control_row=matched_control_row,
                settings=settings,
                power_semantics=RA_EE_08_ASSOC_ORACLE_CONSTRAINED,
                bucket=spec_name,
                association_policy=policy,
                association_role="diagnostic-association-constrained-power-oracle",
                association_action_contract="finite-active-set-oracle-diagnostic",
            )
            if best is None or _row_ee(row) > _row_ee(best) + 1e-12:
                best = row
                best_policy = policy
        if best is not None and best_policy is not None:
            constrained_oracle_cache[cache_key] = (dict(best), best_policy)
        return best, best_policy

    def best_association_deployable_oracle(
        *,
        spec_name: str,
        seed: int,
        step_index: int,
    ) -> tuple[dict[str, Any] | None, str | None]:
        cache_key = (spec_name, int(seed), int(step_index))
        if cache_key in deployable_oracle_cache:
            row, policy = deployable_oracle_cache[cache_key]
            return dict(row), policy
        best: dict[str, Any] | None = None
        best_policy: str | None = None
        for policy in settings.oracle_association_policies:
            evaluated = proposal_eval(
                spec_name=spec_name,
                policy=policy,
                seed=seed,
                step_index=step_index,
            )
            if evaluated is None:
                continue
            row = dict(evaluated["deployable"])
            row["power_semantics"] = RA_EE_08_ASSOC_ORACLE_DEPLOYABLE
            row["association_policy"] = policy
            row["source_association_policy"] = policy
            row["association_role"] = "diagnostic-association-oracle-deployable"
            row["association_action_contract"] = "finite-active-set-oracle-diagnostic"
            row["diagnostic_only"] = True
            row["primary_candidate"] = False
            row["accepted_flag"] = False
            row["rejection_reason"] = "diagnostic-only"
            if best is None or _row_ee(row) > _row_ee(best) + 1e-12:
                best = row
                best_policy = policy
        if best is not None and best_policy is not None:
            deployable_oracle_cache[cache_key] = (dict(best), best_policy)
        return best, best_policy

    for spec in settings.bucket_specs:
        control_label = _policy_label(
            spec.name,
            settings.matched_control_association_policy,
        )
        for proposal_policy in settings.candidate_association_policies:
            candidate_label = _policy_label(spec.name, proposal_policy)
            for seed in spec.evaluation_seed_set:
                step_indices = sorted(
                    step
                    for label, eval_seed, step in by_key
                    if label == candidate_label and eval_seed == int(seed)
                )
                for step_index in step_indices:
                    control_snapshot = by_key.get((control_label, int(seed), step_index))
                    candidate_eval = proposal_eval(
                        spec_name=spec.name,
                        policy=proposal_policy,
                        seed=int(seed),
                        step_index=int(step_index),
                    )
                    if control_snapshot is None or candidate_eval is None:
                        continue
                    control_trace = traces_by_key.get(
                        (control_label, int(seed), int(step_index))
                    )
                    candidate_trace = candidate_eval["trace"]

                    fixed_safe, fixed_safe_vector = _safe_greedy_row(
                        control_snapshot,
                        settings,
                        bucket=spec.name,
                        association_policy=settings.matched_control_association_policy,
                        association_role="diagnostic-fixed-safe-greedy",
                        association_action_contract="fixed-by-trajectory",
                        power_semantics=RA_EE_08_FIXED_SAFE_GREEDY,
                    )
                    fixed_safe["power_allocator"] = "safe-greedy-power-allocator"
                    control_allocators = _deployable_allocator_results(
                        control_snapshot,
                        fixed_safe,
                        fixed_safe_vector,
                        settings,
                    )
                    selected_control = control_allocators[
                        settings.primary_deployable_allocator
                    ]
                    fixed_deployable = _deployable_row(
                        control_snapshot,
                        settings,
                        result=selected_control,
                        matched_safe_row=fixed_safe,
                        safe_vector=fixed_safe_vector,
                        bucket=spec.name,
                        association_policy=settings.matched_control_association_policy,
                        association_role="matched-fixed-association-primary-control",
                        association_action_contract="fixed-by-trajectory",
                        power_semantics=RA_EE_08_FIXED_DEPLOYABLE_CONTROL,
                        diagnostic_only=False,
                        primary_candidate=False,
                    )
                    fixed_deployable["power_allocator"] = (
                        "deployable-stronger-power-allocator"
                    )
                    proposal_safe = dict(candidate_eval["safe"])
                    proposal_deployable = dict(candidate_eval["deployable"])

                    fixed_oracle: dict[str, Any] | None = None
                    assoc_oracle_constrained: dict[str, Any] | None = None
                    assoc_oracle_constrained_policy: str | None = None
                    assoc_oracle_deployable: dict[str, Any] | None = None
                    assoc_oracle_deployable_policy: str | None = None
                    if include_oracle:
                        fixed_oracle = _constrained_oracle_row(
                            snapshot=control_snapshot,
                            control_row=fixed_deployable,
                            settings=settings,
                            power_semantics=RA_EE_08_FIXED_CONSTRAINED_ORACLE,
                            bucket=spec.name,
                            association_policy=settings.matched_control_association_policy,
                            association_role=(
                                "diagnostic-fixed-association-constrained-power-oracle"
                            ),
                            association_action_contract="fixed-by-trajectory",
                        )
                        assoc_oracle_constrained, assoc_oracle_constrained_policy = (
                            best_association_constrained_oracle(
                                spec_name=spec.name,
                                seed=int(seed),
                                step_index=int(step_index),
                                matched_control_row=fixed_deployable,
                            )
                        )
                        assoc_oracle_deployable, assoc_oracle_deployable_policy = (
                            best_association_deployable_oracle(
                                spec_name=spec.name,
                                seed=int(seed),
                                step_index=int(step_index),
                            )
                        )
                    oracle_trace = (
                        None
                        if assoc_oracle_constrained_policy is None
                        else traces_by_key.get(
                            (
                                _policy_label(spec.name, assoc_oracle_constrained_policy),
                                int(seed),
                                int(step_index),
                            )
                        )
                    )
                    deployable_oracle_trace = (
                        None
                        if assoc_oracle_deployable_policy is None
                        else traces_by_key.get(
                            (
                                _policy_label(spec.name, assoc_oracle_deployable_policy),
                                int(seed),
                                int(step_index),
                            )
                        )
                    )

                    step_rows = [
                        (fixed_safe, control_trace, True),
                        (fixed_deployable, control_trace, False),
                        (proposal_safe, candidate_trace, True),
                        (proposal_deployable, candidate_trace, False),
                    ]
                    if fixed_oracle is not None:
                        step_rows.append((fixed_oracle, control_trace, True))
                    if assoc_oracle_deployable is not None:
                        step_rows.append((assoc_oracle_deployable, deployable_oracle_trace, True))
                    if assoc_oracle_constrained is not None:
                        step_rows.append((assoc_oracle_constrained, oracle_trace, True))

                    if assoc_oracle_constrained is not None:
                        _add_oracle_gap_fields(
                            [row for row, _trace, _diag in step_rows],
                            matched_control_row=fixed_deployable,
                            candidate_row=proposal_deployable,
                            oracle_row=assoc_oracle_constrained,
                        )
                    for row, trace, diagnostic_only in step_rows:
                        row = _copy_row_for_semantics(
                            row,
                            trajectory_policy=candidate_label,
                            power_semantics=str(row["power_semantics"]),
                            association_policy=str(row["association_policy"]),
                            association_role=str(row["association_role"]),
                            association_action_contract=str(
                                row["association_action_contract"]
                            ),
                            diagnostic_only=diagnostic_only
                            or str(row["power_semantics"])
                            not in {
                                RA_EE_08_FIXED_DEPLOYABLE_CONTROL,
                                RA_EE_08_CANDIDATE,
                            },
                            primary_candidate=str(row["power_semantics"]) == RA_EE_08_CANDIDATE,
                        )
                        row["candidate_association_policy"] = proposal_policy
                        row["matched_control_association_policy"] = (
                            settings.matched_control_association_policy
                        )
                        row["evaluation_bucket"] = spec.name
                        _add_comparison_fields(
                            row,
                            trace=trace,
                            control_trace=control_trace,
                            oracle_trace=oracle_trace,
                            matched_control_row=fixed_deployable,
                            primary_candidate_row=proposal_deployable,
                            oracle_row=assoc_oracle_constrained,
                            settings=settings,
                            diagnostic_only=bool(row["diagnostic_only"]),
                        )
                        throughputs = row.pop("_user_throughputs")
                        rows.append(row)
                        user_throughputs_by_key[
                            (str(row["trajectory_policy"]), str(row["power_semantics"]))
                        ].extend(float(value) for value in throughputs.tolist())
    return rows, user_throughputs_by_key

__all__ = [
    "_evaluation_rows",
    "_power_allocator_label",
    "_rollout_association_trajectories",
]
