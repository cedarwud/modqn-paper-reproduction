"""Control and candidate replay artifact exports for RA-EE-09."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..config_loader import build_environment, load_training_yaml
from ..env.step import HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK
from ._common import write_json
from ._ra_ee_09_common import (
    DEFAULT_CANDIDATE_OUTPUT_DIR,
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    RA_EE_09_CANDIDATE,
    RA_EE_09_CANDIDATE_ALLOCATOR,
    RA_EE_09_CONTROL,
    RA_EE_09_EQUAL_SHARE_ALLOCATOR,
    RA_EE_09_GATE_ID,
    RA_EE_09_POWER_ALLOCATOR_ID,
    RA_EE_09_RESOURCE_UNIT,
    _RAEE09Settings,
    _candidate_settings_from_control,
    _settings_from_config,
    _write_csv,
)
from ._ra_ee_09_resource import (
    _resource_candidate_row,
    _resource_control_row,
)
from .ra_ee_02_oracle_power_allocation import (
    _StepSnapshot,
    _build_unit_power_snapshots,
    _summarize_all,
)
from .ra_ee_05_fixed_association_robustness import (
    CALIBRATION_BUCKET,
    HELD_OUT_BUCKET,
    _BucketSpec,
    _rollout_fixed_association_trajectories,
)
from .ra_ee_07_constrained_power_allocator_distillation import (
    _RAEE07Settings,
    _categorical_distribution,
    _deployable_allocator_results,
    _fieldnames,
    _numeric_distribution,
    _safe_greedy_row,
)


def _evaluation_rows_for_control_snapshots(
    *,
    snapshots: list[_StepSnapshot],
    bucket_by_policy: dict[str, str],
    settings: _RAEE09Settings,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[float]]]:
    rows: list[dict[str, Any]] = []
    user_throughputs_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    for snapshot in snapshots:
        bucket = bucket_by_policy[str(snapshot.trajectory_policy)]
        safe, safe_vector = _safe_greedy_row(
            snapshot,
            settings.power_settings,
            bucket=bucket,
            association_policy=str(snapshot.trajectory_policy),
            association_role="matched-fixed-association-power-boundary-helper",
            association_action_contract="fixed-by-trajectory",
        )
        allocator_results = _deployable_allocator_results(
            snapshot,
            safe,
            safe_vector,
            settings.power_settings,
        )
        deployable = allocator_results[settings.power_settings.primary_deployable_allocator]
        row = _resource_control_row(
            snapshot=snapshot,
            bucket=bucket,
            power_vector=deployable.power_vector,
            selected_power_profile=deployable.selected_power_profile,
            selected_allocator_candidate=deployable.selected_from,
            allocator_rejection_reason=deployable.rejection_reason,
            evaluated_allocator_profile_count=deployable.evaluated_profile_count,
            safe_vector=safe_vector,
            settings=settings,
        )
        throughputs = row.pop("_user_throughputs")
        rows.append(row)
        user_throughputs_by_key[
            (str(row["trajectory_policy"]), str(row["power_semantics"]))
        ].extend(float(value) for value in throughputs.tolist())
    return rows, user_throughputs_by_key


def _evaluation_rows_for_candidate_snapshots(
    *,
    snapshots: list[_StepSnapshot],
    bucket_by_policy: dict[str, str],
    control_settings: _RAEE09Settings,
    candidate_settings: _RAEE09Settings,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[str, str], list[float]],
    dict[tuple[str, str], list[float]],
]:
    control_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    control_throughputs_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    candidate_throughputs_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    for snapshot in snapshots:
        bucket = bucket_by_policy[str(snapshot.trajectory_policy)]
        safe, safe_vector = _safe_greedy_row(
            snapshot,
            control_settings.power_settings,
            bucket=bucket,
            association_policy=str(snapshot.trajectory_policy),
            association_role="matched-fixed-association-power-boundary-helper",
            association_action_contract="fixed-by-trajectory",
        )
        allocator_results = _deployable_allocator_results(
            snapshot,
            safe,
            safe_vector,
            control_settings.power_settings,
        )
        deployable = allocator_results[
            control_settings.power_settings.primary_deployable_allocator
        ]
        control_row = _resource_control_row(
            snapshot=snapshot,
            bucket=bucket,
            power_vector=deployable.power_vector,
            selected_power_profile=deployable.selected_power_profile,
            selected_allocator_candidate=deployable.selected_from,
            allocator_rejection_reason=deployable.rejection_reason,
            evaluated_allocator_profile_count=deployable.evaluated_profile_count,
            safe_vector=safe_vector,
            settings=control_settings,
        )
        candidate_row = _resource_candidate_row(
            snapshot=snapshot,
            bucket=bucket,
            power_vector=deployable.power_vector,
            selected_power_profile=deployable.selected_power_profile,
            selected_allocator_candidate=deployable.selected_from,
            allocator_rejection_reason=deployable.rejection_reason,
            evaluated_allocator_profile_count=deployable.evaluated_profile_count,
            safe_vector=safe_vector,
            settings=candidate_settings,
        )
        control_throughputs = control_row.pop("_user_throughputs")
        candidate_throughputs = candidate_row.pop("_user_throughputs")
        control_rows.append(control_row)
        candidate_rows.append(candidate_row)
        control_throughputs_by_key[
            (str(control_row["trajectory_policy"]), str(control_row["power_semantics"]))
        ].extend(float(value) for value in control_throughputs.tolist())
        candidate_throughputs_by_key[
            (
                str(candidate_row["trajectory_policy"]),
                str(candidate_row["power_semantics"]),
            )
        ].extend(float(value) for value in candidate_throughputs.tolist())
    return (
        control_rows,
        candidate_rows,
        control_throughputs_by_key,
        candidate_throughputs_by_key,
    )


def _augment_summaries(
    summaries: list[dict[str, Any]],
    *,
    step_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in step_rows:
        rows_by_key[(str(row["trajectory_policy"]), str(row["power_semantics"]))].append(
            row
        )

    augmented: list[dict[str, Any]] = []
    for summary in summaries:
        key = (str(summary["trajectory_policy"]), str(summary["power_semantics"]))
        rows = rows_by_key[key]
        summary = dict(summary)
        summary["evaluation_bucket"] = rows[0]["evaluation_bucket"]
        summary["association_mode"] = "fixed-replay"
        summary["power_allocator_id"] = RA_EE_09_POWER_ALLOCATOR_ID
        summary["resource_unit"] = RA_EE_09_RESOURCE_UNIT
        summary["resource_allocator_id"] = str(rows[0]["resource_allocator_id"])
        summary["candidate_allocator_enabled"] = any(
            bool(row["candidate_allocator_enabled"]) for row in rows
        )
        summary["resource_accounting_enabled"] = True
        summary["equal_share_throughput_parity"] = all(
            bool(row["equal_share_throughput_parity"]) for row in rows
        )
        summary["resource_throughput_max_abs_delta_vs_existing_bps"] = max(
            float(row["resource_throughput_max_abs_delta_vs_existing_bps"])
            for row in rows
        )
        summary["resource_throughput_sum_delta_vs_existing_bps"] = float(
            np.sum(
                [
                    float(row["resource_throughput_sum_delta_vs_existing_bps"])
                    for row in rows
                ],
                dtype=np.float64,
            )
        )
        summary["active_beam_resource_sum_exact"] = all(
            bool(row["active_beam_resource_sum_exact"]) for row in rows
        )
        summary["active_beam_resource_sum_max_abs_error"] = max(
            float(row["active_beam_resource_sum_max_abs_error"]) for row in rows
        )
        summary["inactive_beam_zero_resource"] = not any(
            bool(row["inactive_beam_nonzero_resource"]) for row in rows
        )
        summary["resource_budget_violations"] = {
            "step_count": int(sum(bool(row["resource_budget_violation"]) for row in rows)),
            "violation_count": int(
                sum(int(row["resource_budget_violation_count"]) for row in rows)
            ),
            "max_active_beam_sum_error": summary[
                "active_beam_resource_sum_max_abs_error"
            ],
            "max_overuse": max(float(row["resource_overuse_max"]) for row in rows),
            "max_underuse": max(float(row["resource_underuse_max"]) for row in rows),
        }
        summary["per_beam_resource_sum_distribution"] = _numeric_distribution(
            [
                value
                for row in rows
                for value in np.fromstring(
                    str(row["per_beam_resource_sum"]),
                    sep=" ",
                    dtype=np.float64,
                ).tolist()
            ]
        )
        summary["selected_power_vector_distribution"] = _categorical_distribution(
            [str(row["effective_power_vector_w"]) for row in rows]
        )
        summary["same_power_vector_as_control"] = all(
            bool(row["same_power_vector_as_control"]) for row in rows
        )
        summary["fixed_association_enforced"] = all(
            row["association_action_contract"] == "fixed-by-trajectory"
            and not bool(row["resource_allocation_changes_assignment"])
            and not bool(row["resource_allocation_changes_handover_trajectory"])
            for row in rows
        )
        summary["resource_allocation_after_power_vector_selection"] = all(
            bool(row["resource_allocation_after_power_vector_selection"]) for row in rows
        )
        summary["resource_allocation_feedback_to_power_decision"] = any(
            bool(row["resource_allocation_feedback_to_power_decision"]) for row in rows
        )
        augmented.append(summary)
    return augmented


def _resource_budget_report(
    step_rows: list[dict[str, Any]],
    *,
    resource_allocator_id: str = RA_EE_09_EQUAL_SHARE_ALLOCATOR,
    equal_share_control: bool = True,
) -> dict[str, Any]:
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in step_rows:
        by_bucket[str(row["evaluation_bucket"])].append(row)

    def bucket_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "step_count": len(rows),
            "active_beam_resource_sum_exact": all(
                bool(row["active_beam_resource_sum_exact"]) for row in rows
            ),
            "inactive_beam_zero_resource": not any(
                bool(row["inactive_beam_nonzero_resource"]) for row in rows
            ),
            "resource_budget_violation_count": int(
                sum(int(row["resource_budget_violation_count"]) for row in rows)
            ),
            "resource_budget_violation_step_count": int(
                sum(bool(row["resource_budget_violation"]) for row in rows)
            ),
            "max_active_beam_resource_sum_abs_error": max(
                float(row["active_beam_resource_sum_max_abs_error"]) for row in rows
            ),
            "max_resource_overuse": max(float(row["resource_overuse_max"]) for row in rows),
            "max_resource_underuse": max(float(row["resource_underuse_max"]) for row in rows),
        }

    all_rows = list(step_rows)
    return {
        "resource_unit": RA_EE_09_RESOURCE_UNIT,
        "resource_allocator_id": resource_allocator_id,
        "equal_share_control": equal_share_control,
        "overall": bucket_report(all_rows) if all_rows else {},
        "by_bucket": {bucket: bucket_report(rows) for bucket, rows in sorted(by_bucket.items())},
    }


def _build_decision(
    *,
    summaries: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
    resource_budget_report: dict[str, Any],
) -> dict[str, Any]:
    parity = all(bool(row["equal_share_throughput_parity"]) for row in step_rows)
    fixed_association = all(
        row["association_action_contract"] == "fixed-by-trajectory"
        and not bool(row["resource_allocation_changes_assignment"])
        for row in step_rows
    )
    same_power = all(bool(row["same_power_vector_as_control"]) for row in step_rows)
    resource_audit = bool(resource_budget_report.get("overall"))
    active_sum_exact = all(
        bool(row["active_beam_resource_sum_exact"]) for row in step_rows
    )
    inactive_zero = not any(bool(row["inactive_beam_nonzero_resource"]) for row in step_rows)
    no_resource_violations = not any(bool(row["resource_budget_violation"]) for row in step_rows)
    no_forbidden_runtime = all(
        not bool(row["learned_association_enabled"])
        and not bool(row["learned_hierarchical_RL_enabled"])
        and not bool(row["joint_association_power_training_enabled"])
        and not bool(row["catfish_enabled"])
        and not bool(row["phase03c_continuation_enabled"])
        and not bool(row["oracle_labels_used_for_runtime_decision"])
        and not bool(row["future_outcomes_used_for_runtime_decision"])
        and not bool(row["held_out_answers_used_for_runtime_decision"])
        for row in step_rows
    )
    no_power_feedback = not any(
        bool(row["resource_allocation_feedback_to_power_decision"]) for row in step_rows
    )
    proof_flags = {
        "ra_ee_09_config_namespace_enabled": True,
        "disabled_by_default_outside_explicit_ra_ee_09_namespace": True,
        "offline_replay_only": True,
        "candidate_allocator_disabled": True,
        "resource_accounting_enabled": True,
        "equal_share_control_only": True,
        "equal_share_throughput_parity": parity,
        "fixed_association_enforced": fixed_association,
        "same_power_vector_as_control": same_power,
        "power_allocator_id_matches_ra_ee_07": all(
            row["power_allocator_id"] == RA_EE_09_POWER_ALLOCATOR_ID
            for row in step_rows
        ),
        "resource_allocation_after_power_vector_selection": all(
            bool(row["resource_allocation_after_power_vector_selection"])
            for row in step_rows
        ),
        "resource_allocation_feedback_to_power_decision": False,
        "resource_accounting_auditable": resource_audit,
        "active_beam_resource_sum_exact": active_sum_exact,
        "inactive_beam_zero_resource": inactive_zero,
        "zero_resource_budget_violations": no_resource_violations,
        "learned_association_disabled": True,
        "hierarchical_RL_disabled": True,
        "joint_association_power_training_disabled": True,
        "catfish_disabled": True,
        "phase03c_continuation_disabled": True,
        "oracle_labels_future_or_heldout_answers_disabled": no_forbidden_runtime,
        "frozen_baseline_mutation": False,
        "scalar_reward_success_basis": False,
        "per_user_EE_credit_success_basis": False,
        "full_RA_EE_MODQN_claim": False,
    }
    stop_conditions = {
        "equal_share_control_cannot_reproduce_existing_throughput": not parity,
        "frozen_baseline_semantics_must_change": False,
        "same_association_cannot_be_proven": not fixed_association,
        "same_power_boundary_cannot_be_proven": not same_power,
        "resource_accounting_cannot_audit_per_beam_or_inactive_zero": not (
            resource_audit and active_sum_exact and inactive_zero
        ),
        "resource_budget_violations_present": not no_resource_violations,
        "depends_on_learned_association_catfish_or_oracle_future_labels": (
            not no_forbidden_runtime
        ),
        "resource_allocation_feeds_back_into_power": not no_power_feedback,
    }
    can_proceed = (
        not any(bool(value) for value in stop_conditions.values())
        and proof_flags["scalar_reward_success_basis"] is False
        and proof_flags["full_RA_EE_MODQN_claim"] is False
    )
    return {
        "ra_ee_09_slice_09a_09c_decision": (
            "PASS_TO_SLICE_09D" if can_proceed else "BLOCKED"
        ),
        "proof_flags": proof_flags,
        "stop_conditions": stop_conditions,
        "allowed_claim": (
            "Slices 09A-09C establish only opt-in config/metadata, resource "
            "accounting auditability, equal-share throughput parity, and "
            "fixed-association control replay on the RA-EE-07 deployable power "
            "boundary. They do not establish RB / bandwidth effectiveness."
        ),
        "summary_count": len(summaries),
    }


def _build_candidate_decision(
    *,
    summaries: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
    resource_budget_report: dict[str, Any],
) -> dict[str, Any]:
    fixed_association = all(
        row["association_action_contract"] == "fixed-by-trajectory"
        and not bool(row["resource_allocation_changes_assignment"])
        and str(row["assignment_hash_before_resource_allocation"])
        == str(row["assignment_hash_after_resource_allocation"])
        for row in step_rows
    )
    handover_fixed = all(
        not bool(row["resource_allocation_changes_handover_trajectory"])
        and int(row["handover_count_before_resource_allocation"])
        == int(row["handover_count_after_resource_allocation"])
        for row in step_rows
    )
    same_power = all(
        bool(row["same_power_vector_as_control"])
        and str(row["power_vector_hash_before_resource_allocation"])
        == str(row["power_vector_hash_after_resource_allocation"])
        for row in step_rows
    )
    resource_audit = bool(resource_budget_report.get("overall"))
    active_sum_exact = all(
        bool(row["active_beam_resource_sum_exact"]) for row in step_rows
    )
    inactive_zero = not any(bool(row["inactive_beam_nonzero_resource"]) for row in step_rows)
    no_resource_violations = not any(bool(row["resource_budget_violation"]) for row in step_rows)
    no_forbidden_runtime = all(
        not bool(row["learned_association_enabled"])
        and not bool(row["learned_hierarchical_RL_enabled"])
        and not bool(row["joint_association_power_training_enabled"])
        and not bool(row["catfish_enabled"])
        and not bool(row["phase03c_continuation_enabled"])
        and not bool(row["oracle_labels_used_for_runtime_decision"])
        and not bool(row["future_outcomes_used_for_runtime_decision"])
        and not bool(row["held_out_answers_used_for_runtime_decision"])
        for row in step_rows
    )
    no_power_feedback = not any(
        bool(row["resource_allocation_feedback_to_power_decision"]) for row in step_rows
    )
    candidate_enabled = all(
        bool(row["candidate_allocator_enabled"])
        and row["resource_allocator_id"] == RA_EE_09_CANDIDATE_ALLOCATOR
        for row in step_rows
    )
    proof_flags = {
        "ra_ee_09_config_namespace_enabled": True,
        "offline_replay_only": True,
        "candidate_allocator_enabled": candidate_enabled,
        "resource_allocator_id_matches_candidate": candidate_enabled,
        "resource_accounting_enabled": True,
        "fixed_association_enforced": fixed_association,
        "handover_trajectory_unchanged": handover_fixed,
        "same_power_vector_as_control": same_power,
        "power_allocator_id_matches_ra_ee_07": all(
            row["power_allocator_id"] == RA_EE_09_POWER_ALLOCATOR_ID
            for row in step_rows
        ),
        "resource_allocation_after_power_vector_selection": all(
            bool(row["resource_allocation_after_power_vector_selection"])
            for row in step_rows
        ),
        "resource_allocation_feedback_to_power_decision": False,
        "resource_accounting_auditable": resource_audit,
        "active_beam_resource_sum_exact": active_sum_exact,
        "inactive_beam_zero_resource": inactive_zero,
        "zero_resource_budget_violations": no_resource_violations,
        "learned_association_disabled": True,
        "hierarchical_RL_disabled": True,
        "joint_association_power_training_disabled": True,
        "catfish_disabled": True,
        "phase03c_continuation_disabled": True,
        "oracle_labels_future_or_heldout_answers_disabled": no_forbidden_runtime,
        "frozen_baseline_mutation": False,
        "scalar_reward_success_basis": False,
        "per_user_EE_credit_success_basis": False,
        "full_RA_EE_MODQN_claim": False,
        "effectiveness_evidence_established": False,
    }
    stop_conditions = {
        "frozen_baseline_semantics_must_change": False,
        "same_association_cannot_be_proven": not fixed_association,
        "same_handover_trajectory_cannot_be_proven": not handover_fixed,
        "same_power_boundary_cannot_be_proven": not same_power,
        "resource_accounting_cannot_audit_per_beam_or_inactive_zero": not (
            resource_audit and active_sum_exact and inactive_zero
        ),
        "resource_budget_violations_present": not no_resource_violations,
        "depends_on_learned_association_catfish_or_oracle_future_labels": (
            not no_forbidden_runtime
        ),
        "resource_allocation_feeds_back_into_power": not no_power_feedback,
        "candidate_allocator_not_enabled": not candidate_enabled,
    }
    can_proceed = (
        not any(bool(value) for value in stop_conditions.values())
        and proof_flags["scalar_reward_success_basis"] is False
        and proof_flags["full_RA_EE_MODQN_claim"] is False
    )
    return {
        "ra_ee_09_slice_09d_decision": (
            "PASS_TO_SLICE_09E" if can_proceed else "BLOCKED"
        ),
        "proof_flags": proof_flags,
        "stop_conditions": stop_conditions,
        "allowed_claim": (
            "Slice 09D establishes only a deterministic bounded resource-share "
            "candidate implementation and accounting boundary tests. It does "
            "not establish RB / bandwidth effectiveness; Slice 09E remains the "
            "matched replay and QoS/effectiveness gate."
        ),
        "summary_count": len(summaries),
    }


def _compact_summary_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "evaluation_bucket",
        "trajectory_policy",
        "power_semantics",
        "association_mode",
        "power_allocator_id",
        "resource_unit",
        "resource_allocator_id",
        "step_count",
        "EE_system_aggregate_bps_per_w",
        "throughput_mean_user_step_bps",
        "throughput_p05_user_step_bps",
        "served_ratio",
        "outage_ratio",
        "handover_count",
        "active_beam_count_distribution",
        "selected_power_profile_distribution",
        "selected_power_vector_distribution",
        "total_active_beam_power_w_distribution",
        "equal_share_throughput_parity",
        "resource_throughput_max_abs_delta_vs_existing_bps",
        "active_beam_resource_sum_exact",
        "active_beam_resource_sum_max_abs_error",
        "inactive_beam_zero_resource",
        "resource_budget_violations",
        "same_power_vector_as_control",
        "fixed_association_enforced",
        "candidate_allocator_enabled",
    )
    return [{field: row[field] for field in fields} for row in summaries]


def _review_lines(summary: dict[str, Any]) -> list[str]:
    decision = summary["ra_ee_09_slice_09a_09c_decision"]
    proof = summary["proof_flags"]
    lines = [
        "# RA-EE-09 Fixed-Association RB / Bandwidth Control Replay Review",
        "",
        "Slices 09A-09C only. This is explicit RA-EE-09 metadata, equal-share "
        "normalized per-beam resource accounting, and fixed-association control "
        "replay on the RA-EE-07 deployable power boundary. No candidate "
        "allocator, training, learned association, hierarchical RL, Catfish, "
        "Phase 03C continuation, or full RA-EE-MODQN claim was performed.",
        "",
        "## Protocol",
        "",
        f"- method label: `{summary['metadata']['method_label']}`",
        f"- implementation sublabel: `{summary['metadata']['implementation_sublabel']}`",
        f"- config: `{summary['inputs']['config_path']}`",
        f"- artifact namespace: `{summary['inputs']['output_dir']}`",
        f"- resource unit: `{summary['metadata']['resource_unit']}`",
        f"- resource allocator: `{summary['metadata']['resource_allocator_id']}`",
        f"- power allocator: `{summary['metadata']['power_allocator_id']}`",
        f"- association mode: `{summary['metadata']['association_mode']}`",
        "",
        "## Proof Flags",
        "",
    ]
    for key, value in proof.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- RA-EE-09 Slice 09A-09C decision: `{decision}`",
            f"- allowed claim: {summary['decision_detail']['allowed_claim']}",
            "",
            "## Forbidden Claims",
            "",
        ]
    )
    for claim in summary["forbidden_claims_still_active"]:
        lines.append(f"- {claim}")
    return lines


def _candidate_review_lines(summary: dict[str, Any]) -> list[str]:
    decision = summary["ra_ee_09_slice_09d_decision"]
    proof = summary["proof_flags"]
    lines = [
        "# RA-EE-09 Slice 09D Bounded Resource-Share Candidate Review",
        "",
        "Slice 09D only. This adds the deterministic bounded resource-share "
        "candidate after RA-EE-07 power vector selection. No training, learned "
        "association, hierarchical RL, Catfish, Phase 03C continuation, or full "
        "RA-EE-MODQN claim was performed.",
        "",
        "## Protocol",
        "",
        f"- method label: `{summary['metadata']['method_label']}`",
        f"- implementation sublabel: `{summary['metadata']['implementation_sublabel']}`",
        f"- config: `{summary['inputs']['config_path']}`",
        f"- artifact namespace: `{summary['inputs']['output_dir']}`",
        f"- resource unit: `{summary['metadata']['resource_unit']}`",
        f"- resource allocator: `{summary['metadata']['resource_allocator_id']}`",
        f"- power allocator: `{summary['metadata']['power_allocator_id']}`",
        f"- association mode: `{summary['metadata']['association_mode']}`",
        "",
        "## Proof Flags",
        "",
    ]
    for key, value in proof.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- RA-EE-09 Slice 09D decision: `{decision}`",
            f"- allowed claim: {summary['decision_detail']['allowed_claim']}",
            "",
            "## Forbidden Claims",
            "",
        ]
    )
    for claim in summary["forbidden_claims_still_active"]:
        lines.append(f"- {claim}")
    return lines


def export_ra_ee_09_fixed_association_rb_bandwidth_control(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    calibration_seed_set: tuple[int, ...] | None = None,
    held_out_seed_set: tuple[int, ...] | None = None,
    calibration_policies: tuple[str, ...] | None = None,
    held_out_policies: tuple[str, ...] | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Export RA-EE-09 Slice 09A-09C control replay artifacts."""
    if max_steps is not None and max_steps < 1:
        raise ValueError(f"max_steps must be >= 1 when provided, got {max_steps!r}")
    cfg = load_training_yaml(config_path)
    env = build_environment(cfg)
    if env.power_surface_config.hobs_power_surface_mode != (
        HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK
    ):
        raise ValueError("RA-EE-09 config must opt into the RA-EE-07 power surface.")

    settings = _settings_from_config(cfg, config_path=config_path)
    fixed_specs: list[_BucketSpec] = []
    for spec in settings.power_settings.fixed_bucket_specs:
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
    run_power_settings = _RAEE07Settings(
        method_label=settings.power_settings.method_label,
        implementation_sublabel=settings.power_settings.implementation_sublabel,
        audit=settings.power_settings.audit,
        fixed_bucket_specs=tuple(fixed_specs),
        diagnostic_bucket_specs=(),
        deployable_allocators=settings.power_settings.deployable_allocators,
        primary_deployable_allocator=settings.power_settings.primary_deployable_allocator,
        candidate_max_demoted_beams=settings.power_settings.candidate_max_demoted_beams,
        candidate_step_p05_guardrail_margin=(
            settings.power_settings.candidate_step_p05_guardrail_margin
        ),
        local_search_max_moves=settings.power_settings.local_search_max_moves,
        p05_trim_max_moves=settings.power_settings.p05_trim_max_moves,
        dp_max_profile_count=settings.power_settings.dp_max_profile_count,
        min_oracle_gap_closed_ratio=settings.power_settings.min_oracle_gap_closed_ratio,
        association_diagnostic_policies=(),
        min_active_beams=settings.power_settings.min_active_beams,
        max_active_beams=settings.power_settings.max_active_beams,
        target_users_per_active_beam=settings.power_settings.target_users_per_active_beam,
        load_cap_overflow_users=settings.power_settings.load_cap_overflow_users,
        max_moved_user_ratio_per_step=(
            settings.power_settings.max_moved_user_ratio_per_step
        ),
        max_moved_user_ratio=settings.power_settings.max_moved_user_ratio,
        max_one_active_beam_ratio_for_acceptance=(
            settings.power_settings.max_one_active_beam_ratio_for_acceptance
        ),
        max_two_beam_overload_step_ratio=(
            settings.power_settings.max_two_beam_overload_step_ratio
        ),
        diagnostic_max_steps=None,
    )
    run_settings = _RAEE09Settings(
        method_label=settings.method_label,
        implementation_sublabel=settings.implementation_sublabel,
        power_settings=run_power_settings,
        resource=settings.resource,
        metadata=settings.metadata,
    )

    trajectories, bucket_by_policy = _rollout_fixed_association_trajectories(
        cfg=cfg,
        bucket_specs=run_settings.power_settings.fixed_bucket_specs,
        max_steps=max_steps,
    )
    snapshots = _build_unit_power_snapshots(
        base_cfg=cfg,
        settings=run_settings.power_settings.audit,
        trajectories=trajectories,
    )
    step_rows, user_throughputs_by_key = _evaluation_rows_for_control_snapshots(
        snapshots=snapshots,
        bucket_by_policy=bucket_by_policy,
        settings=run_settings,
    )
    summaries = _summarize_all(
        rows=step_rows,
        user_throughputs_by_key=user_throughputs_by_key,
    )
    summaries = _augment_summaries(summaries, step_rows=step_rows)
    budget_report = _resource_budget_report(step_rows)
    decision_detail = _build_decision(
        summaries=summaries,
        step_rows=step_rows,
        resource_budget_report=budget_report,
    )

    out_dir = Path(output_dir)
    step_csv = _write_csv(
        out_dir / "ra_ee_09_step_resource_trace.csv",
        step_rows,
        fieldnames=_fieldnames(step_rows),
    )
    compact_rows = _compact_summary_rows(summaries)
    summary_csv = _write_csv(
        out_dir / "ra_ee_09_control_summary.csv",
        compact_rows,
        fieldnames=list(compact_rows[0].keys()) if compact_rows else [],
    )
    budget_path = write_json(
        out_dir / "resource_budget_report.json",
        budget_report,
    )

    protocol = {
        "phase": RA_EE_09_GATE_ID,
        "method_label": run_settings.method_label,
        "method_family": "RA-EE fixed-association resource accounting",
        "implementation_sublabel": run_settings.implementation_sublabel,
        "training": "none; offline replay only",
        "offline_replay_only": True,
        "association_mode": "fixed-replay",
        "association_training": "disabled",
        "learned_association": "disabled",
        "hierarchical_RL": "disabled",
        "joint_association_power_training": "disabled",
        "catfish": "disabled",
        "phase03c_continuation": "disabled",
        "candidate_allocator": "disabled/not-implemented-in-slices-09a-09c",
        "primary_control": RA_EE_09_CONTROL,
        "power_allocator_id": RA_EE_09_POWER_ALLOCATOR_ID,
        "primary_deployable_allocator": (
            run_settings.power_settings.primary_deployable_allocator
        ),
        "resource_unit": run_settings.resource.resource_unit,
        "resource_allocator_id": run_settings.resource.resource_allocator_id,
        "throughput_formula_version": run_settings.resource.throughput_formula_version,
        "resource_allocation_order": "after-power-vector-selection",
        "resource_allocation_feedback_to_power_decision": False,
        "same_power_vector_as_control_required": True,
        "system_EE_primary": True,
        "scalar_reward_success_basis": False,
        "per_user_EE_credit_success_basis": False,
    }
    constraints = {
        "per_beam_budget": run_settings.resource.per_beam_budget,
        "total_resource_budget": "active_beam_count",
        "inactive_beam_resource_policy": (
            run_settings.resource.inactive_beam_resource_policy
        ),
        "per_user_min": (
            f"{run_settings.resource.per_user_min_equal_share_multiplier:g}/N_b"
        ),
        "per_user_max": (
            f"{run_settings.resource.per_user_max_equal_share_multiplier:g}/N_b"
        ),
        "resource_sum_tolerance": run_settings.resource.resource_sum_tolerance,
        "per_beam_max_power_w": run_settings.power_settings.audit.per_beam_max_power_w,
        "total_power_budget_w": run_settings.power_settings.audit.total_power_budget_w,
        "codebook_levels_w": list(run_settings.power_settings.audit.codebook_levels_w),
    }
    summary = {
        "inputs": {
            "config_path": str(Path(config_path)),
            "output_dir": str(out_dir),
            "max_steps": max_steps,
        },
        "metadata": run_settings.metadata,
        "protocol": protocol,
        "constraints": constraints,
        "step_resource_trace_schema_fields": list(step_rows[0].keys()) if step_rows else [],
        "control_summaries": summaries,
        "resource_budget_report": budget_report,
        "decision_detail": decision_detail,
        "proof_flags": decision_detail["proof_flags"],
        "stop_conditions": decision_detail["stop_conditions"],
        "ra_ee_09_slice_09a_09c_decision": (
            decision_detail["ra_ee_09_slice_09a_09c_decision"]
        ),
        "remaining_blockers": [
            "No RA-EE-09 candidate resource allocator exists in Slice 09A-09C.",
            "No RB / bandwidth effectiveness evidence exists yet.",
            "No learned association, hierarchical RL, joint training, or Catfish path exists.",
            "Physical 3GPP RB semantics and integer RB rounding remain out of scope.",
        ],
        "forbidden_claims_still_active": [
            "Do not call RA-EE-09 full RA-EE-MODQN.",
            "Do not claim learned association effectiveness.",
            "Do not claim old EE-MODQN effectiveness.",
            "Do not claim HOBS optimizer behavior.",
            "Do not claim physical energy saving.",
            "Do not claim Catfish-EE or Catfish repair.",
            "Do not claim RB / bandwidth allocation effectiveness yet.",
            "Do not claim full paper-faithful reproduction.",
        ],
    }
    summary_path = write_json(
        out_dir / "ra_ee_09_fixed_association_rb_bandwidth_control_summary.json",
        summary,
    )
    review_path = out_dir / "review.md"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text("\n".join(_review_lines(summary)) + "\n")
    return {
        "ra_ee_09_fixed_association_rb_bandwidth_control_summary": summary_path,
        "ra_ee_09_control_summary_csv": summary_csv,
        "ra_ee_09_step_resource_trace": step_csv,
        "resource_budget_report": budget_path,
        "review_md": review_path,
        "summary": summary,
    }


def export_ra_ee_09_fixed_association_rb_bandwidth_candidate(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_CANDIDATE_OUTPUT_DIR,
    *,
    calibration_seed_set: tuple[int, ...] | None = None,
    held_out_seed_set: tuple[int, ...] | None = None,
    calibration_policies: tuple[str, ...] | None = None,
    held_out_policies: tuple[str, ...] | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Export RA-EE-09 Slice 09D candidate replay artifacts."""
    if max_steps is not None and max_steps < 1:
        raise ValueError(f"max_steps must be >= 1 when provided, got {max_steps!r}")
    cfg = load_training_yaml(config_path)
    env = build_environment(cfg)
    if env.power_surface_config.hobs_power_surface_mode != (
        HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK
    ):
        raise ValueError("RA-EE-09 config must opt into the RA-EE-07 power surface.")

    settings = _settings_from_config(cfg, config_path=config_path)
    fixed_specs: list[_BucketSpec] = []
    for spec in settings.power_settings.fixed_bucket_specs:
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
    run_power_settings = _RAEE07Settings(
        method_label=settings.power_settings.method_label,
        implementation_sublabel="RA-EE-09 Slice 09D bounded resource-share candidate",
        audit=settings.power_settings.audit,
        fixed_bucket_specs=tuple(fixed_specs),
        diagnostic_bucket_specs=(),
        deployable_allocators=settings.power_settings.deployable_allocators,
        primary_deployable_allocator=settings.power_settings.primary_deployable_allocator,
        candidate_max_demoted_beams=settings.power_settings.candidate_max_demoted_beams,
        candidate_step_p05_guardrail_margin=(
            settings.power_settings.candidate_step_p05_guardrail_margin
        ),
        local_search_max_moves=settings.power_settings.local_search_max_moves,
        p05_trim_max_moves=settings.power_settings.p05_trim_max_moves,
        dp_max_profile_count=settings.power_settings.dp_max_profile_count,
        min_oracle_gap_closed_ratio=settings.power_settings.min_oracle_gap_closed_ratio,
        association_diagnostic_policies=(),
        min_active_beams=settings.power_settings.min_active_beams,
        max_active_beams=settings.power_settings.max_active_beams,
        target_users_per_active_beam=settings.power_settings.target_users_per_active_beam,
        load_cap_overflow_users=settings.power_settings.load_cap_overflow_users,
        max_moved_user_ratio_per_step=(
            settings.power_settings.max_moved_user_ratio_per_step
        ),
        max_moved_user_ratio=settings.power_settings.max_moved_user_ratio,
        max_one_active_beam_ratio_for_acceptance=(
            settings.power_settings.max_one_active_beam_ratio_for_acceptance
        ),
        max_two_beam_overload_step_ratio=(
            settings.power_settings.max_two_beam_overload_step_ratio
        ),
        diagnostic_max_steps=None,
    )
    control_settings = _RAEE09Settings(
        method_label=settings.method_label,
        implementation_sublabel="RA-EE-09 Slice 09D matched equal-share control",
        power_settings=run_power_settings,
        resource=settings.resource,
        metadata=settings.metadata,
    )
    candidate_settings = _candidate_settings_from_control(control_settings)

    trajectories, bucket_by_policy = _rollout_fixed_association_trajectories(
        cfg=cfg,
        bucket_specs=control_settings.power_settings.fixed_bucket_specs,
        max_steps=max_steps,
    )
    snapshots = _build_unit_power_snapshots(
        base_cfg=cfg,
        settings=control_settings.power_settings.audit,
        trajectories=trajectories,
    )
    (
        control_rows,
        candidate_rows,
        control_user_throughputs_by_key,
        candidate_user_throughputs_by_key,
    ) = _evaluation_rows_for_candidate_snapshots(
        snapshots=snapshots,
        bucket_by_policy=bucket_by_policy,
        control_settings=control_settings,
        candidate_settings=candidate_settings,
    )
    control_summaries = _summarize_all(
        rows=control_rows,
        user_throughputs_by_key=control_user_throughputs_by_key,
    )
    control_summaries = _augment_summaries(
        control_summaries,
        step_rows=control_rows,
    )
    candidate_summaries = _summarize_all(
        rows=candidate_rows,
        user_throughputs_by_key=candidate_user_throughputs_by_key,
    )
    candidate_summaries = _augment_summaries(
        candidate_summaries,
        step_rows=candidate_rows,
    )
    budget_report = _resource_budget_report(
        candidate_rows,
        resource_allocator_id=RA_EE_09_CANDIDATE_ALLOCATOR,
        equal_share_control=False,
    )
    decision_detail = _build_candidate_decision(
        summaries=candidate_summaries,
        step_rows=candidate_rows,
        resource_budget_report=budget_report,
    )

    out_dir = Path(output_dir)
    step_csv = _write_csv(
        out_dir / "ra_ee_09_candidate_step_resource_trace.csv",
        candidate_rows,
        fieldnames=_fieldnames(candidate_rows),
    )
    compact_rows = _compact_summary_rows(candidate_summaries)
    summary_csv = _write_csv(
        out_dir / "ra_ee_09_candidate_summary.csv",
        compact_rows,
        fieldnames=list(compact_rows[0].keys()) if compact_rows else [],
    )
    budget_path = write_json(
        out_dir / "resource_budget_report.json",
        budget_report,
    )

    protocol = {
        "phase": RA_EE_09_GATE_ID,
        "method_label": candidate_settings.method_label,
        "method_family": "RA-EE fixed-association resource allocation",
        "implementation_sublabel": candidate_settings.implementation_sublabel,
        "training": "none; offline replay only",
        "offline_replay_only": True,
        "association_mode": "fixed-replay",
        "association_training": "disabled",
        "learned_association": "disabled",
        "hierarchical_RL": "disabled",
        "joint_association_power_training": "disabled",
        "catfish": "disabled",
        "phase03c_continuation": "disabled",
        "candidate_allocator": RA_EE_09_CANDIDATE_ALLOCATOR,
        "matched_control": RA_EE_09_CONTROL,
        "primary_candidate": RA_EE_09_CANDIDATE,
        "power_allocator_id": RA_EE_09_POWER_ALLOCATOR_ID,
        "primary_deployable_allocator": (
            candidate_settings.power_settings.primary_deployable_allocator
        ),
        "resource_unit": candidate_settings.resource.resource_unit,
        "resource_allocator_id": candidate_settings.resource.resource_allocator_id,
        "throughput_formula_version": (
            candidate_settings.resource.throughput_formula_version
        ),
        "resource_allocation_order": "after-power-vector-selection",
        "resource_allocation_feedback_to_power_decision": False,
        "same_power_vector_as_control_required": True,
        "system_EE_primary": True,
        "scalar_reward_success_basis": False,
        "per_user_EE_credit_success_basis": False,
    }
    constraints = {
        "per_beam_budget": candidate_settings.resource.per_beam_budget,
        "total_resource_budget": "active_beam_count",
        "inactive_beam_resource_policy": (
            candidate_settings.resource.inactive_beam_resource_policy
        ),
        "per_user_min": "0.25/N_b",
        "per_user_max": "min(4/N_b, 1.0)",
        "resource_sum_tolerance": candidate_settings.resource.resource_sum_tolerance,
        "per_beam_max_power_w": (
            candidate_settings.power_settings.audit.per_beam_max_power_w
        ),
        "total_power_budget_w": (
            candidate_settings.power_settings.audit.total_power_budget_w
        ),
        "codebook_levels_w": list(
            candidate_settings.power_settings.audit.codebook_levels_w
        ),
    }
    summary = {
        "inputs": {
            "config_path": str(Path(config_path)),
            "output_dir": str(out_dir),
            "max_steps": max_steps,
        },
        "metadata": candidate_settings.metadata,
        "protocol": protocol,
        "constraints": constraints,
        "step_resource_trace_schema_fields": (
            list(candidate_rows[0].keys()) if candidate_rows else []
        ),
        "control_summaries": control_summaries,
        "candidate_summaries": candidate_summaries,
        "resource_budget_report": budget_report,
        "decision_detail": decision_detail,
        "proof_flags": decision_detail["proof_flags"],
        "stop_conditions": decision_detail["stop_conditions"],
        "ra_ee_09_slice_09d_decision": (
            decision_detail["ra_ee_09_slice_09d_decision"]
        ),
        "remaining_blockers": [
            "No RA-EE-09 matched held-out effectiveness replay has been run.",
            "No RB / bandwidth effectiveness claim is authorized before Slice 09E.",
            "No learned association, hierarchical RL, joint training, or Catfish path exists.",
            "Physical 3GPP RB semantics and integer RB rounding remain out of scope.",
        ],
        "forbidden_claims_still_active": [
            "Do not call RA-EE-09 full RA-EE-MODQN.",
            "Do not claim learned association effectiveness.",
            "Do not claim old EE-MODQN effectiveness.",
            "Do not claim HOBS optimizer behavior.",
            "Do not claim physical energy saving.",
            "Do not claim Catfish-EE or Catfish repair.",
            "Do not claim RB / bandwidth allocation effectiveness yet.",
            "Do not claim full paper-faithful reproduction.",
        ],
    }
    summary_path = write_json(
        out_dir / "ra_ee_09_fixed_association_rb_bandwidth_candidate_summary.json",
        summary,
    )
    review_path = out_dir / "review.md"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text("\n".join(_candidate_review_lines(summary)) + "\n")
    return {
        "ra_ee_09_fixed_association_rb_bandwidth_candidate_summary": summary_path,
        "ra_ee_09_candidate_summary_csv": summary_csv,
        "ra_ee_09_candidate_step_resource_trace": step_csv,
        "resource_budget_report": budget_path,
        "review_md": review_path,
        "summary": summary,
    }
