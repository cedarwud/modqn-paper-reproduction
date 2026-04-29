"""Matched held-out comparison and Slice 09E decision logic for RA-EE-09."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..config_loader import build_environment, load_training_yaml
from ..env.step import HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK
from ._common import write_json
from ._ra_ee_09_common import (
    DEFAULT_COMPARISON_OUTPUT_DIR,
    DEFAULT_CONFIG,
    HELD_OUT_BUCKET,
    RA_EE_09_CANDIDATE,
    RA_EE_09_CANDIDATE_ALLOCATOR,
    RA_EE_09_CONTROL,
    RA_EE_09_EQUAL_SHARE_ALLOCATOR,
    RA_EE_09_GATE_ID,
    RA_EE_09_MAX_POSITIVE_GAIN_CONTRIBUTION_SHARE,
    RA_EE_09_POWER_ALLOCATOR_ID,
    RA_EE_09_PREDECLARED_RESOURCE_EFFICIENCY_METRIC,
    _RAEE09Settings,
    _candidate_settings_from_control,
    _correlation_or_none,
    _parse_vector,
    _pct_delta,
    _run_power_settings_with_fixed_specs,
    _safe_mean,
    _safe_percentile,
    _safe_ratio,
    _settings_from_config,
    _stable_hash,
    _write_csv,
)
from ._ra_ee_09_replay import (
    _augment_summaries,
    _evaluation_rows_for_candidate_snapshots,
    _resource_budget_report,
)
from .ra_ee_02_oracle_power_allocation import (
    _build_unit_power_snapshots,
    _summarize_all,
)
from .ra_ee_05_fixed_association_robustness import (
    _BucketSpec,
    _rollout_fixed_association_trajectories,
)
from .ra_ee_07_constrained_power_allocator_distillation import (
    _fieldnames,
    _numeric_distribution,
)


def _step_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row["trajectory_policy"]),
        int(row["evaluation_seed"]),
        int(row["step_index"]),
    )


def _build_paired_step_rows(
    *,
    control_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    control_by_key = {_step_key(row): row for row in control_rows}
    candidate_by_key = {_step_key(row): row for row in candidate_rows}
    if set(control_by_key) != set(candidate_by_key):
        missing_candidate = sorted(set(control_by_key) - set(candidate_by_key))
        missing_control = sorted(set(candidate_by_key) - set(control_by_key))
        raise ValueError(
            "RA-EE-09 matched comparison requires identical schedules; "
            f"missing_candidate={missing_candidate!r} missing_control={missing_control!r}"
        )

    paired: list[dict[str, Any]] = []
    for key in sorted(control_by_key):
        control = control_by_key[key]
        candidate = candidate_by_key[key]
        active_resource_budget = float(candidate["total_resource_budget"])
        control_step_p05_per_resource = _safe_ratio(
            float(control["throughput_p05_user_step_bps"]),
            active_resource_budget,
        )
        candidate_step_p05_per_resource = _safe_ratio(
            float(candidate["throughput_p05_user_step_bps"]),
            active_resource_budget,
        )
        ee_delta = (
            None
            if control["EE_system_bps_per_w"] is None
            or candidate["EE_system_bps_per_w"] is None
            else float(candidate["EE_system_bps_per_w"])
            - float(control["EE_system_bps_per_w"])
        )
        paired.append(
            {
                "evaluation_bucket": candidate["evaluation_bucket"],
                "trajectory_policy": key[0],
                "evaluation_seed": key[1],
                "step_index": key[2],
                "matched_step_key": f"{key[0]}:{key[1]}:{key[2]}",
                "control_power_semantics": control["power_semantics"],
                "candidate_power_semantics": candidate["power_semantics"],
                "control_sum_throughput_bps": control["sum_user_throughput_bps"],
                "candidate_sum_throughput_bps": candidate["sum_user_throughput_bps"],
                "sum_throughput_delta_bps": float(
                    candidate["sum_user_throughput_bps"]
                )
                - float(control["sum_user_throughput_bps"]),
                "sum_throughput_pct_delta": _pct_delta(
                    float(control["sum_user_throughput_bps"]),
                    float(candidate["sum_user_throughput_bps"]),
                ),
                "control_mean_throughput_bps": control[
                    "throughput_mean_user_step_bps"
                ],
                "candidate_mean_throughput_bps": candidate[
                    "throughput_mean_user_step_bps"
                ],
                "mean_throughput_delta_bps": float(
                    candidate["throughput_mean_user_step_bps"]
                )
                - float(control["throughput_mean_user_step_bps"]),
                "control_p05_throughput_bps": control[
                    "throughput_p05_user_step_bps"
                ],
                "candidate_p05_throughput_bps": candidate[
                    "throughput_p05_user_step_bps"
                ],
                "p05_throughput_delta_bps": float(
                    candidate["throughput_p05_user_step_bps"]
                )
                - float(control["throughput_p05_user_step_bps"]),
                "p05_throughput_ratio": _safe_ratio(
                    float(candidate["throughput_p05_user_step_bps"]),
                    float(control["throughput_p05_user_step_bps"]),
                ),
                "control_EE_system_bps_per_w": control["EE_system_bps_per_w"],
                "candidate_EE_system_bps_per_w": candidate["EE_system_bps_per_w"],
                "EE_system_delta_bps_per_w": ee_delta,
                "EE_system_pct_delta": _pct_delta(
                    None
                    if control["EE_system_bps_per_w"] is None
                    else float(control["EE_system_bps_per_w"]),
                    None
                    if candidate["EE_system_bps_per_w"] is None
                    else float(candidate["EE_system_bps_per_w"]),
                ),
                "active_resource_budget": active_resource_budget,
                "control_p05_throughput_per_active_resource_budget": (
                    control_step_p05_per_resource
                ),
                "candidate_p05_throughput_per_active_resource_budget": (
                    candidate_step_p05_per_resource
                ),
                "p05_throughput_per_active_resource_budget_delta": (
                    None
                    if control_step_p05_per_resource is None
                    or candidate_step_p05_per_resource is None
                    else candidate_step_p05_per_resource
                    - control_step_p05_per_resource
                ),
                "control_served_ratio": control["served_ratio"],
                "candidate_served_ratio": candidate["served_ratio"],
                "served_ratio_delta": float(candidate["served_ratio"])
                - float(control["served_ratio"]),
                "control_outage_ratio": control["outage_ratio"],
                "candidate_outage_ratio": candidate["outage_ratio"],
                "outage_ratio_delta": float(candidate["outage_ratio"])
                - float(control["outage_ratio"]),
                "control_handover_count": control["handover_count"],
                "candidate_handover_count": candidate["handover_count"],
                "handover_count_delta": int(candidate["handover_count"])
                - int(control["handover_count"]),
                "control_active_beam_count": control["active_beam_count"],
                "candidate_active_beam_count": candidate["active_beam_count"],
                "active_beam_count_delta": int(candidate["active_beam_count"])
                - int(control["active_beam_count"]),
                "control_total_active_power_w": control["total_active_power"],
                "candidate_total_active_power_w": candidate["total_active_power"],
                "total_active_power_delta_w": float(candidate["total_active_power"])
                - float(control["total_active_power"]),
                "control_assignment_hash": control["assignment_hash"],
                "candidate_assignment_hash": candidate["assignment_hash"],
                "assignment_hash_match": str(control["assignment_hash"])
                == str(candidate["assignment_hash"]),
                "control_association_trajectory_hash": control[
                    "association_trajectory_hash"
                ],
                "candidate_association_trajectory_hash": candidate[
                    "association_trajectory_hash"
                ],
                "association_trajectory_hash_match": str(
                    control["association_trajectory_hash"]
                )
                == str(candidate["association_trajectory_hash"]),
                "control_power_vector_hash": control["power_vector_hash"],
                "candidate_power_vector_hash": candidate["power_vector_hash"],
                "effective_power_vector_hash_match": str(
                    control["power_vector_hash"]
                )
                == str(candidate["power_vector_hash"]),
                "control_effective_power_vector_w": control[
                    "effective_power_vector_w"
                ],
                "candidate_effective_power_vector_w": candidate[
                    "effective_power_vector_w"
                ],
                "effective_power_vector_match": str(
                    control["effective_power_vector_w"]
                )
                == str(candidate["effective_power_vector_w"]),
                "same_power_vector_as_control": bool(
                    candidate["same_power_vector_as_control"]
                )
                and str(control["power_vector_hash"])
                == str(candidate["power_vector_hash"]),
                "power_allocator_id": candidate["power_allocator_id"],
                "selected_allocator_candidate": candidate[
                    "selected_allocator_candidate"
                ],
                "control_resource_allocator_id": control["resource_allocator_id"],
                "candidate_resource_allocator_id": candidate["resource_allocator_id"],
                "control_resource_fractions": control["resource_fractions"],
                "candidate_resource_fractions": candidate["resource_fractions"],
                "control_per_beam_resource_sum": control[
                    "per_beam_resource_sum"
                ],
                "candidate_per_beam_resource_sum": candidate[
                    "per_beam_resource_sum"
                ],
                "control_inactive_beam_resource_usage": control[
                    "inactive_beam_resource_usage"
                ],
                "candidate_inactive_beam_resource_usage": candidate[
                    "inactive_beam_resource_usage"
                ],
                "control_resource_budget_violation_count": control[
                    "resource_budget_violation_count"
                ],
                "candidate_resource_budget_violation_count": candidate[
                    "resource_budget_violation_count"
                ],
                "control_budget_violation": control["budget_violation"],
                "candidate_budget_violation": candidate["budget_violation"],
                "control_per_beam_power_violation": control[
                    "per_beam_power_violation"
                ],
                "candidate_per_beam_power_violation": candidate[
                    "per_beam_power_violation"
                ],
                "control_inactive_beam_nonzero_power": control[
                    "inactive_beam_nonzero_power"
                ],
                "candidate_inactive_beam_nonzero_power": candidate[
                    "inactive_beam_nonzero_power"
                ],
                "control_inactive_beam_nonzero_resource": control[
                    "inactive_beam_nonzero_resource"
                ],
                "candidate_inactive_beam_nonzero_resource": candidate[
                    "inactive_beam_nonzero_resource"
                ],
                "resource_allocation_feedback_to_power_decision": bool(
                    candidate["resource_allocation_feedback_to_power_decision"]
                ),
                "scalar_reward_success_basis": False,
            }
        )
    return paired


def _aggregate_replay_metrics(
    rows: list[dict[str, Any]],
    *,
    user_throughputs: list[float],
) -> dict[str, Any]:
    total_throughput = float(
        np.sum([float(row["sum_user_throughput_bps"]) for row in rows], dtype=np.float64)
    )
    total_power = float(
        np.sum([float(row["total_active_power"]) for row in rows], dtype=np.float64)
    )
    total_resource_budget = float(
        np.sum([float(row["total_resource_budget"]) for row in rows], dtype=np.float64)
    )
    active_resource_budget_step_mean = _safe_mean(
        [float(row["total_resource_budget"]) for row in rows]
    )
    p05 = _safe_percentile(user_throughputs, 5)
    served_count = int(sum(int(row["served_count"]) for row in rows))
    outage_count = int(sum(int(row["outage_count"]) for row in rows))
    step_pairs = [
        (float(row["sum_user_throughput_bps"]), float(row["EE_system_bps_per_w"]))
        for row in rows
        if row["EE_system_bps_per_w"] is not None
    ]
    return {
        "step_count": int(len(rows)),
        "evaluation_bucket": HELD_OUT_BUCKET,
        "trajectory_policies": sorted({str(row["trajectory_policy"]) for row in rows}),
        "evaluation_seeds": sorted({int(row["evaluation_seed"]) for row in rows}),
        "simulated_EE_system_aggregate_bps_per_w": (
            None if total_power <= 0.0 else total_throughput / total_power
        ),
        "simulated_EE_system_step_mean_bps_per_w": _safe_mean(
            [
                float(row["EE_system_bps_per_w"])
                for row in rows
                if row["EE_system_bps_per_w"] is not None
            ]
        ),
        "sum_throughput_bps": total_throughput,
        "mean_throughput_bps": _safe_mean(user_throughputs),
        "p05_throughput_bps": p05,
        "served_ratio": served_count / max(served_count + outage_count, 1),
        "outage_ratio": outage_count / max(served_count + outage_count, 1),
        "handover_count": int(sum(int(row["handover_count"]) for row in rows)),
        "active_beam_count_distribution": _numeric_distribution(
            [float(row["active_beam_count"]) for row in rows]
        ),
        "total_active_power_w_sum": total_power,
        "total_active_power_w_distribution": _numeric_distribution(
            [float(row["total_active_power"]) for row in rows]
        ),
        "active_resource_budget_sum": total_resource_budget,
        "active_resource_budget_step_mean": active_resource_budget_step_mean,
        RA_EE_09_PREDECLARED_RESOURCE_EFFICIENCY_METRIC: _safe_ratio(
            p05,
            active_resource_budget_step_mean,
        ),
        "sum_throughput_per_active_resource_budget": _safe_ratio(
            total_throughput,
            total_resource_budget,
        ),
        "per_user_resource_share_distribution": _numeric_distribution(
            [
                value
                for row in rows
                for value in _parse_vector(row["resource_fractions"])
            ]
        ),
        "per_beam_resource_sum_distribution": _numeric_distribution(
            [
                value
                for row in rows
                for value in _parse_vector(row["per_beam_resource_sum"])
            ]
        ),
        "inactive_beam_resource_usage_distribution": _numeric_distribution(
            [
                value
                for row in rows
                for value in _parse_vector(row["inactive_beam_resource_usage"])
            ]
        ),
        "resource_budget_violation_count": int(
            sum(int(row["resource_budget_violation_count"]) for row in rows)
        ),
        "resource_budget_violation_step_count": int(
            sum(bool(row["resource_budget_violation"]) for row in rows)
        ),
        "power_budget_violation_step_count": int(
            sum(bool(row["budget_violation"]) for row in rows)
        ),
        "per_beam_power_violation_step_count": int(
            sum(bool(row["per_beam_power_violation"]) for row in rows)
        ),
        "inactive_beam_nonzero_power_step_count": int(
            sum(bool(row["inactive_beam_nonzero_power"]) for row in rows)
        ),
        "throughput_vs_EE_system_correlation": _correlation_or_none(
            [throughput for throughput, _ee in step_pairs],
            [ee for _throughput, ee in step_pairs],
        ),
    }


def _comparison_deltas(
    *,
    control: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "simulated_EE_system_delta_bps_per_w": (
            None
            if control["simulated_EE_system_aggregate_bps_per_w"] is None
            or candidate["simulated_EE_system_aggregate_bps_per_w"] is None
            else float(candidate["simulated_EE_system_aggregate_bps_per_w"])
            - float(control["simulated_EE_system_aggregate_bps_per_w"])
        ),
        "simulated_EE_system_pct_delta": _pct_delta(
            control["simulated_EE_system_aggregate_bps_per_w"],
            candidate["simulated_EE_system_aggregate_bps_per_w"],
        ),
        "sum_throughput_delta_bps": float(candidate["sum_throughput_bps"])
        - float(control["sum_throughput_bps"]),
        "sum_throughput_pct_delta": _pct_delta(
            control["sum_throughput_bps"],
            candidate["sum_throughput_bps"],
        ),
        "mean_throughput_delta_bps": (
            None
            if control["mean_throughput_bps"] is None
            or candidate["mean_throughput_bps"] is None
            else float(candidate["mean_throughput_bps"])
            - float(control["mean_throughput_bps"])
        ),
        "p05_throughput_delta_bps": (
            None
            if control["p05_throughput_bps"] is None
            or candidate["p05_throughput_bps"] is None
            else float(candidate["p05_throughput_bps"])
            - float(control["p05_throughput_bps"])
        ),
        "p05_throughput_ratio": _safe_ratio(
            candidate["p05_throughput_bps"],
            control["p05_throughput_bps"],
        ),
        "served_ratio_delta": float(candidate["served_ratio"])
        - float(control["served_ratio"]),
        "outage_ratio_delta": float(candidate["outage_ratio"])
        - float(control["outage_ratio"]),
        "handover_count_delta": int(candidate["handover_count"])
        - int(control["handover_count"]),
        "total_active_power_w_sum_delta": float(candidate["total_active_power_w_sum"])
        - float(control["total_active_power_w_sum"]),
        "active_resource_budget_sum_delta": float(
            candidate["active_resource_budget_sum"]
        )
        - float(control["active_resource_budget_sum"]),
        "predeclared_resource_efficiency_metric": (
            RA_EE_09_PREDECLARED_RESOURCE_EFFICIENCY_METRIC
        ),
        "predeclared_resource_efficiency_delta": (
            None
            if control[RA_EE_09_PREDECLARED_RESOURCE_EFFICIENCY_METRIC] is None
            or candidate[RA_EE_09_PREDECLARED_RESOURCE_EFFICIENCY_METRIC] is None
            else float(candidate[RA_EE_09_PREDECLARED_RESOURCE_EFFICIENCY_METRIC])
            - float(control[RA_EE_09_PREDECLARED_RESOURCE_EFFICIENCY_METRIC])
        ),
    }


def _gain_concentration(
    paired_rows: list[dict[str, Any]],
    *,
    gain_field: str,
) -> dict[str, Any]:
    positive_rows = [
        row
        for row in paired_rows
        if row[gain_field] is not None and float(row[gain_field]) > 0.0
    ]
    total_positive = float(
        np.sum([float(row[gain_field]) for row in positive_rows], dtype=np.float64)
    )

    def group_concentration(group_key: str) -> dict[str, Any]:
        grouped: dict[str, float] = defaultdict(float)
        for row in positive_rows:
            grouped[str(row[group_key])] += float(row[gain_field])
        if total_positive <= 0.0:
            return {
                "positive_group_count": 0,
                "max_positive_contribution_share": None,
                "max_positive_contribution_group": None,
                "histogram": {},
                "passes_max_share_threshold": False,
            }
        shares = {
            key: value / total_positive for key, value in sorted(grouped.items())
        }
        max_group = max(shares, key=shares.get) if shares else None
        max_share = None if max_group is None else float(shares[max_group])
        return {
            "positive_group_count": int(len(shares)),
            "max_positive_contribution_share": max_share,
            "max_positive_contribution_group": max_group,
            "histogram": shares,
            "passes_max_share_threshold": bool(
                max_share is not None
                and max_share < RA_EE_09_MAX_POSITIVE_GAIN_CONTRIBUTION_SHARE
            ),
        }

    by_seed = group_concentration("evaluation_seed")
    by_trajectory = group_concentration("trajectory_policy")
    return {
        "gain_field": gain_field,
        "max_positive_contribution_share_threshold": (
            RA_EE_09_MAX_POSITIVE_GAIN_CONTRIBUTION_SHARE
        ),
        "positive_step_count": int(len(positive_rows)),
        "total_positive_gain": total_positive,
        "by_seed": by_seed,
        "by_trajectory": by_trajectory,
        "passes": bool(
            total_positive > 0.0
            and by_seed["passes_max_share_threshold"]
            and by_trajectory["passes_max_share_threshold"]
        ),
    }


def _throughput_vs_ee_ranking_comparison(
    *,
    control_overall: dict[str, Any],
    candidate_overall: dict[str, Any],
    paired_rows: list[dict[str, Any]],
    control_summaries: list[dict[str, Any]],
    candidate_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    overall = [
        {
            "method": "equal-share-control",
            "sum_throughput_bps": control_overall["sum_throughput_bps"],
            "simulated_EE_system_aggregate_bps_per_w": control_overall[
                "simulated_EE_system_aggregate_bps_per_w"
            ],
        },
        {
            "method": RA_EE_09_CANDIDATE_ALLOCATOR,
            "sum_throughput_bps": candidate_overall["sum_throughput_bps"],
            "simulated_EE_system_aggregate_bps_per_w": candidate_overall[
                "simulated_EE_system_aggregate_bps_per_w"
            ],
        },
    ]
    throughput_ranking = [
        row["method"]
        for row in sorted(
            overall,
            key=lambda row: float(row["sum_throughput_bps"]),
            reverse=True,
        )
    ]
    ee_ranking = [
        row["method"]
        for row in sorted(
            overall,
            key=lambda row: float(
                row["simulated_EE_system_aggregate_bps_per_w"] or -math.inf
            ),
            reverse=True,
        )
    ]
    control_by_policy = {
        str(row["trajectory_policy"]): row for row in control_summaries
    }
    candidate_by_policy = {
        str(row["trajectory_policy"]): row for row in candidate_summaries
    }
    by_trajectory: list[dict[str, Any]] = []
    for policy in sorted(set(control_by_policy) & set(candidate_by_policy)):
        control = control_by_policy[policy]
        candidate = candidate_by_policy[policy]
        compared = [
            {
                "method": "equal-share-control",
                "mean_throughput_bps": control["throughput_mean_user_step_bps"],
                "EE_system_aggregate_bps_per_w": control[
                    "EE_system_aggregate_bps_per_w"
                ],
            },
            {
                "method": RA_EE_09_CANDIDATE_ALLOCATOR,
                "mean_throughput_bps": candidate[
                    "throughput_mean_user_step_bps"
                ],
                "EE_system_aggregate_bps_per_w": candidate[
                    "EE_system_aggregate_bps_per_w"
                ],
            },
        ]
        policy_throughput_ranking = [
            row["method"]
            for row in sorted(
                compared,
                key=lambda row: float(row["mean_throughput_bps"] or -math.inf),
                reverse=True,
            )
        ]
        policy_ee_ranking = [
            row["method"]
            for row in sorted(
                compared,
                key=lambda row: float(row["EE_system_aggregate_bps_per_w"] or -math.inf),
                reverse=True,
            )
        ]
        by_trajectory.append(
            {
                "trajectory_policy": policy,
                "throughput_ranking": policy_throughput_ranking,
                "EE_system_ranking": policy_ee_ranking,
                "throughput_vs_EE_ranking_changes": (
                    policy_throughput_ranking != policy_ee_ranking
                ),
                "throughput_winner": policy_throughput_ranking[0],
                "EE_system_winner": policy_ee_ranking[0],
            }
        )

    return {
        "overall_throughput_ranking": throughput_ranking,
        "overall_EE_system_ranking": ee_ranking,
        "overall_throughput_vs_EE_ranking_changes": throughput_ranking != ee_ranking,
        "paired_step_throughput_delta_vs_EE_delta_correlation": _correlation_or_none(
            [
                float(row["sum_throughput_delta_bps"])
                for row in paired_rows
                if row["EE_system_delta_bps_per_w"] is not None
            ],
            [
                float(row["EE_system_delta_bps_per_w"])
                for row in paired_rows
                if row["EE_system_delta_bps_per_w"] is not None
            ],
        ),
        "by_trajectory": by_trajectory,
    }


def _matched_boundary_proof(
    *,
    paired_rows: list[dict[str, Any]],
    held_out_seed_set: tuple[int, ...],
    held_out_policies: tuple[str, ...],
) -> dict[str, Any]:
    schedule = [
        {
            "trajectory_policy": row["trajectory_policy"],
            "evaluation_seed": int(row["evaluation_seed"]),
            "step_index": int(row["step_index"]),
        }
        for row in paired_rows
    ]
    control_assignment_schedule = [
        row["control_assignment_hash"] for row in paired_rows
    ]
    candidate_assignment_schedule = [
        row["candidate_assignment_hash"] for row in paired_rows
    ]
    control_power_schedule = [row["control_power_vector_hash"] for row in paired_rows]
    candidate_power_schedule = [
        row["candidate_power_vector_hash"] for row in paired_rows
    ]
    return {
        "held_out_bucket_id": HELD_OUT_BUCKET,
        "evaluation_seed_set": list(held_out_seed_set),
        "fixed_association_trajectory_families": list(held_out_policies),
        "matched_step_count": int(len(paired_rows)),
        "same_evaluation_schedule": True,
        "evaluation_schedule_hash": _stable_hash(schedule),
        "same_association_hash_per_step": all(
            bool(row["assignment_hash_match"]) for row in paired_rows
        ),
        "control_association_schedule_hash": _stable_hash(
            control_assignment_schedule
        ),
        "candidate_association_schedule_hash": _stable_hash(
            candidate_assignment_schedule
        ),
        "same_association_schedule_hash": _stable_hash(control_assignment_schedule)
        == _stable_hash(candidate_assignment_schedule),
        "same_power_vector_hash_per_step": all(
            bool(row["effective_power_vector_hash_match"]) for row in paired_rows
        ),
        "control_effective_power_schedule_hash": _stable_hash(
            control_power_schedule
        ),
        "candidate_effective_power_schedule_hash": _stable_hash(
            candidate_power_schedule
        ),
        "same_effective_power_schedule_hash": _stable_hash(control_power_schedule)
        == _stable_hash(candidate_power_schedule),
        "same_effective_power_vector_text_per_step": all(
            bool(row["effective_power_vector_match"]) for row in paired_rows
        ),
        "same_power_allocator_id": all(
            row["power_allocator_id"] == RA_EE_09_POWER_ALLOCATOR_ID
            for row in paired_rows
        ),
        "resource_allocation_feedback_to_power_decision": any(
            bool(row["resource_allocation_feedback_to_power_decision"])
            for row in paired_rows
        ),
    }


def _build_09e_decision(
    *,
    comparison: dict[str, Any],
    matched_boundary: dict[str, Any],
    control_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    control_budget_report: dict[str, Any],
    candidate_budget_report: dict[str, Any],
    settings: _RAEE09Settings,
) -> dict[str, Any]:
    deltas = comparison["deltas"]
    ee_delta = deltas["simulated_EE_system_delta_bps_per_w"]
    resource_efficiency_delta = deltas["predeclared_resource_efficiency_delta"]
    ee_positive = ee_delta is not None and float(ee_delta) > 0.0
    resource_efficiency_positive = (
        resource_efficiency_delta is not None
        and float(resource_efficiency_delta) > 0.0
    )
    selected_gain_basis = (
        "EE_system_delta_bps_per_w"
        if ee_positive
        else (
            "p05_throughput_per_active_resource_budget_delta"
            if resource_efficiency_positive
            else "none"
        )
    )
    selected_concentration = (
        _gain_concentration(paired_rows, gain_field=selected_gain_basis)
        if selected_gain_basis != "none"
        else {
            "gain_field": "none",
            "passes": False,
            "by_seed": {"max_positive_contribution_share": None},
            "by_trajectory": {"max_positive_contribution_share": None},
        }
    )
    p05_ratio = deltas["p05_throughput_ratio"]
    p05_pass = (
        p05_ratio is not None
        and float(p05_ratio) >= settings.power_settings.audit.p05_min_ratio_vs_control
    )
    served_pass = (
        float(deltas["served_ratio_delta"])
        >= settings.power_settings.audit.served_ratio_min_delta_vs_control
    )
    outage_pass = (
        float(deltas["outage_ratio_delta"])
        <= settings.power_settings.audit.outage_ratio_max_delta_vs_control
    )
    handover_pass = int(deltas["handover_count_delta"]) == 0
    same_power = (
        matched_boundary["same_power_vector_hash_per_step"]
        and matched_boundary["same_effective_power_schedule_hash"]
        and not matched_boundary["resource_allocation_feedback_to_power_decision"]
    )
    same_association = (
        matched_boundary["same_association_hash_per_step"]
        and matched_boundary["same_association_schedule_hash"]
    )
    no_power_violations = all(
        not bool(row["budget_violation"])
        and not bool(row["per_beam_power_violation"])
        and not bool(row["inactive_beam_nonzero_power"])
        for row in control_rows + candidate_rows
    )
    no_resource_violations = (
        int(control_budget_report["overall"]["resource_budget_violation_count"]) == 0
        and int(candidate_budget_report["overall"]["resource_budget_violation_count"]) == 0
    )
    inactive_resource_zero = (
        bool(control_budget_report["overall"]["inactive_beam_zero_resource"])
        and bool(candidate_budget_report["overall"]["inactive_beam_zero_resource"])
    )
    active_resource_exact = (
        bool(control_budget_report["overall"]["active_beam_resource_sum_exact"])
        and bool(candidate_budget_report["overall"]["active_beam_resource_sum_exact"])
    )
    no_forbidden_runtime = all(
        not bool(row["learned_association_enabled"])
        and not bool(row["learned_hierarchical_RL_enabled"])
        and not bool(row["joint_association_power_training_enabled"])
        and not bool(row["catfish_enabled"])
        and not bool(row["phase03c_continuation_enabled"])
        and not bool(row["oracle_labels_used_for_runtime_decision"])
        and not bool(row["future_outcomes_used_for_runtime_decision"])
        and not bool(row["held_out_answers_used_for_runtime_decision"])
        for row in control_rows + candidate_rows
    )
    positive_effectiveness_basis = ee_positive or resource_efficiency_positive
    proof_flags = {
        "offline_fixed_association_replay_only": True,
        "matched_control_vs_candidate_comparison": True,
        "same_seeds": matched_boundary["evaluation_seed_set"],
        "same_fixed_association_trajectories": (
            matched_boundary["fixed_association_trajectory_families"]
        ),
        "same_evaluation_schedule": matched_boundary["same_evaluation_schedule"],
        "same_association_schedule_hash": matched_boundary[
            "same_association_schedule_hash"
        ],
        "same_effective_power_vector_hash_per_step": matched_boundary[
            "same_power_vector_hash_per_step"
        ],
        "same_effective_power_schedule_hash": matched_boundary[
            "same_effective_power_schedule_hash"
        ],
        "same_RA_EE_07_power_boundary": same_power,
        "candidate_does_not_change_association": same_association,
        "candidate_does_not_change_handover": handover_pass,
        "resource_allocation_after_power_vector_selection": all(
            bool(row["resource_allocation_after_power_vector_selection"])
            for row in candidate_rows
        ),
        "resource_allocation_feedback_to_power_decision": False,
        "positive_held_out_simulated_EE_system_delta": ee_positive,
        "predeclared_resource_efficiency_metric": (
            RA_EE_09_PREDECLARED_RESOURCE_EFFICIENCY_METRIC
        ),
        "positive_predeclared_resource_efficiency_delta": (
            resource_efficiency_positive
        ),
        "p05_throughput_ratio_guardrail_pass": p05_pass,
        "served_ratio_does_not_decrease": served_pass,
        "outage_ratio_does_not_increase": outage_pass,
        "handover_count_unchanged": handover_pass,
        "zero_power_violations": no_power_violations,
        "zero_resource_budget_violations": no_resource_violations,
        "active_beam_resource_sum_exact": active_resource_exact,
        "inactive_beam_resource_usage_zero": inactive_resource_zero,
        "gain_concentration_max_share_lt_0_80": bool(
            selected_concentration["passes"]
        ),
        "selected_gain_basis": selected_gain_basis,
        "scalar_reward_success_basis": False,
        "learned_association_disabled": True,
        "hierarchical_RL_disabled": True,
        "joint_association_power_training_disabled": True,
        "catfish_disabled": True,
        "phase03c_continuation_disabled": True,
        "oracle_labels_future_or_heldout_answers_disabled": no_forbidden_runtime,
        "full_RA_EE_MODQN_claim": False,
    }
    stop_conditions = {
        "equal_share_control_no_longer_matches_existing_throughput_formula": not all(
            bool(row["equal_share_throughput_parity"]) for row in control_rows
        ),
        "candidate_changes_association_or_handover_trajectory": not (
            same_association and handover_pass
        ),
        "candidate_changes_or_feeds_back_into_power_allocation": not same_power,
        "resource_accounting_cannot_be_audited": not (
            no_resource_violations and active_resource_exact and inactive_resource_zero
        ),
        "QoS_regression_explains_gain": bool(
            positive_effectiveness_basis and not (p05_pass and served_pass and outage_pass)
        ),
        "gains_appear_only_in_scalar_reward": False,
        "gains_concentrate_in_one_seed_or_trajectory": bool(
            positive_effectiveness_basis and not selected_concentration["passes"]
        ),
        "requires_catfish_learned_association_or_trainer_changes": (
            not no_forbidden_runtime
        ),
    }
    acceptance = {
        "positive_held_out_simulated_EE_system_delta_or_predeclared_resource_efficiency_delta": (
            positive_effectiveness_basis
        ),
        "p05_throughput_ratio_at_least_0_95": p05_pass,
        "served_ratio_does_not_decrease": served_pass,
        "outage_ratio_does_not_increase": outage_pass,
        "handover_count_unchanged": handover_pass,
        "all_power_resource_per_beam_per_user_inactive_violations_zero": (
            no_power_violations
            and no_resource_violations
            and active_resource_exact
            and inactive_resource_zero
        ),
        "gains_not_concentrated": bool(selected_concentration["passes"]),
        "metadata_proves_same_association_power_schedule": (
            same_association and same_power and matched_boundary["same_evaluation_schedule"]
        ),
        "scalar_reward_not_success_basis": True,
    }
    if any(bool(value) for value in stop_conditions.values()):
        decision = "BLOCK"
    elif all(bool(value) for value in acceptance.values()):
        decision = "PASS"
    else:
        decision = "NEEDS MORE EVIDENCE"
    if decision == "PASS" and ee_positive:
        allowed_claim = (
            "RA-EE-09 Slice 09E provides matched held-out offline evidence "
            "that the bounded resource-share candidate improves simulated "
            "EE_system under fixed association and the same RA-EE-07 effective "
            "power vector schedule. This remains fixed-association resource "
            "allocation evidence only."
        )
    elif decision == "PASS":
        allowed_claim = (
            "RA-EE-09 Slice 09E provides matched held-out offline evidence "
            f"that the bounded resource-share candidate improves the predeclared "
            f"{RA_EE_09_PREDECLARED_RESOURCE_EFFICIENCY_METRIC} metric under "
            "fixed association and the same RA-EE-07 effective power vector "
            "schedule. This does not establish simulated EE_system improvement."
        )
    else:
        allowed_claim = (
            "RA-EE-09 Slice 09E does not promote RB / bandwidth allocation "
            "effectiveness beyond the matched held-out evidence reported here."
        )
    return {
        "ra_ee_09_slice_09e_decision": decision,
        "proof_flags": proof_flags,
        "acceptance_criteria": acceptance,
        "stop_conditions": stop_conditions,
        "selected_gain_concentration": selected_concentration,
        "allowed_claim": allowed_claim,
    }


def _matched_comparison_summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    comparison = summary["matched_comparison"]
    rows = [
        {
            "scope": "overall",
            "trajectory_policy": "ALL_HELD_OUT",
            "control_EE_system_aggregate_bps_per_w": comparison["control_overall"][
                "simulated_EE_system_aggregate_bps_per_w"
            ],
            "candidate_EE_system_aggregate_bps_per_w": comparison[
                "candidate_overall"
            ]["simulated_EE_system_aggregate_bps_per_w"],
            "EE_system_delta_bps_per_w": comparison["deltas"][
                "simulated_EE_system_delta_bps_per_w"
            ],
            "control_sum_throughput_bps": comparison["control_overall"][
                "sum_throughput_bps"
            ],
            "candidate_sum_throughput_bps": comparison["candidate_overall"][
                "sum_throughput_bps"
            ],
            "sum_throughput_delta_bps": comparison["deltas"][
                "sum_throughput_delta_bps"
            ],
            "control_mean_throughput_bps": comparison["control_overall"][
                "mean_throughput_bps"
            ],
            "candidate_mean_throughput_bps": comparison["candidate_overall"][
                "mean_throughput_bps"
            ],
            "control_p05_throughput_bps": comparison["control_overall"][
                "p05_throughput_bps"
            ],
            "candidate_p05_throughput_bps": comparison["candidate_overall"][
                "p05_throughput_bps"
            ],
            "p05_throughput_ratio": comparison["deltas"][
                "p05_throughput_ratio"
            ],
            "control_served_ratio": comparison["control_overall"]["served_ratio"],
            "candidate_served_ratio": comparison["candidate_overall"]["served_ratio"],
            "served_ratio_delta": comparison["deltas"]["served_ratio_delta"],
            "control_outage_ratio": comparison["control_overall"]["outage_ratio"],
            "candidate_outage_ratio": comparison["candidate_overall"]["outage_ratio"],
            "outage_ratio_delta": comparison["deltas"]["outage_ratio_delta"],
            "handover_count_delta": comparison["deltas"]["handover_count_delta"],
            "total_active_power_w_sum_delta": comparison["deltas"][
                "total_active_power_w_sum_delta"
            ],
            "predeclared_resource_efficiency_metric": comparison["deltas"][
                "predeclared_resource_efficiency_metric"
            ],
            "predeclared_resource_efficiency_delta": comparison["deltas"][
                "predeclared_resource_efficiency_delta"
            ],
            "decision": summary["ra_ee_09_slice_09e_decision"],
        }
    ]
    rows.extend(summary["matched_comparison"]["by_trajectory"])
    return rows


def _matched_comparison_review_lines(summary: dict[str, Any]) -> list[str]:
    decision = summary["ra_ee_09_slice_09e_decision"]
    proof = summary["proof_flags"]
    deltas = summary["matched_comparison"]["deltas"]
    concentration = summary["decision_detail"]["selected_gain_concentration"]
    lines = [
        "# RA-EE-09 Slice 09E Matched Held-Out Replay Review",
        "",
        "Slice 09E only. This is an offline fixed-association matched replay "
        "comparing equal-share resource control against the bounded QoS-slack "
        "resource-share candidate. It uses the same held-out seeds, same fixed "
        "association trajectories, same evaluation schedule, and the same "
        "RA-EE-07 deployable stronger power allocator. No training, learned "
        "association, hierarchical RL, Catfish, Phase 03C continuation, or "
        "full RA-EE-MODQN claim was performed.",
        "",
        "## Protocol",
        "",
        f"- config: `{summary['inputs']['config_path']}`",
        f"- artifact namespace: `{summary['inputs']['output_dir']}`",
        f"- held-out seeds: `{summary['matched_boundary_proof']['evaluation_seed_set']}`",
        "- held-out trajectories: "
        f"`{summary['matched_boundary_proof']['fixed_association_trajectory_families']}`",
        f"- power allocator: `{summary['metadata']['power_allocator_id']}`",
        f"- control allocator: `{RA_EE_09_EQUAL_SHARE_ALLOCATOR}`",
        f"- candidate allocator: `{RA_EE_09_CANDIDATE_ALLOCATOR}`",
        f"- resource-efficiency metric: `{RA_EE_09_PREDECLARED_RESOURCE_EFFICIENCY_METRIC}`",
        "",
        "## Matched Comparison",
        "",
        f"- EE_system delta: `{deltas['simulated_EE_system_delta_bps_per_w']}`",
        f"- sum throughput delta: `{deltas['sum_throughput_delta_bps']}`",
        f"- p05 throughput ratio: `{deltas['p05_throughput_ratio']}`",
        f"- served ratio delta: `{deltas['served_ratio_delta']}`",
        f"- outage ratio delta: `{deltas['outage_ratio_delta']}`",
        f"- handover count delta: `{deltas['handover_count_delta']}`",
        "- predeclared resource-efficiency delta: "
        f"`{deltas['predeclared_resource_efficiency_delta']}`",
        "",
        "## Boundary Proof",
        "",
        "- evaluation schedule hash: "
        f"`{summary['matched_boundary_proof']['evaluation_schedule_hash']}`",
        "- control power schedule hash: "
        f"`{summary['matched_boundary_proof']['control_effective_power_schedule_hash']}`",
        "- candidate power schedule hash: "
        f"`{summary['matched_boundary_proof']['candidate_effective_power_schedule_hash']}`",
        "- same effective power schedule hash: "
        f"`{summary['matched_boundary_proof']['same_effective_power_schedule_hash']}`",
        "- same association schedule hash: "
        f"`{summary['matched_boundary_proof']['same_association_schedule_hash']}`",
        "",
        "## Proof Flags",
        "",
    ]
    for key, value in proof.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Gain Concentration",
            "",
            f"- gain basis: `{concentration['gain_field']}`",
            f"- passes: `{concentration['passes']}`",
            "- max seed share: "
            f"`{concentration['by_seed']['max_positive_contribution_share']}`",
            "- max trajectory share: "
            f"`{concentration['by_trajectory']['max_positive_contribution_share']}`",
            "",
            "## Decision",
            "",
            f"- RA-EE-09 Slice 09E decision: `{decision}`",
            f"- allowed claim: {summary['decision_detail']['allowed_claim']}",
            "",
            "## Forbidden Claims",
            "",
        ]
    )
    for claim in summary["forbidden_claims_still_active"]:
        lines.append(f"- {claim}")
    return lines


def export_ra_ee_09_fixed_association_rb_bandwidth_matched_comparison(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_COMPARISON_OUTPUT_DIR,
    *,
    held_out_seed_set: tuple[int, ...] | None = None,
    held_out_policies: tuple[str, ...] | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Export RA-EE-09 Slice 09E held-out matched control/candidate replay."""
    if max_steps is not None and max_steps < 1:
        raise ValueError(f"max_steps must be >= 1 when provided, got {max_steps!r}")
    cfg = load_training_yaml(config_path)
    env = build_environment(cfg)
    if env.power_surface_config.hobs_power_surface_mode != (
        HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK
    ):
        raise ValueError("RA-EE-09 config must opt into the RA-EE-07 power surface.")

    settings = _settings_from_config(cfg, config_path=config_path)
    base_held_out = next(
        spec
        for spec in settings.power_settings.fixed_bucket_specs
        if spec.name == HELD_OUT_BUCKET
    )
    held_spec = _BucketSpec(
        name=HELD_OUT_BUCKET,
        trajectory_families=(
            tuple(held_out_policies)
            if held_out_policies is not None
            else base_held_out.trajectory_families
        ),
        evaluation_seed_set=(
            tuple(int(seed) for seed in held_out_seed_set)
            if held_out_seed_set is not None
            else base_held_out.evaluation_seed_set
        ),
    )
    if not held_spec.evaluation_seed_set:
        raise ValueError("RA-EE-09 Slice 09E requires at least one held-out seed.")
    if not held_spec.trajectory_families:
        raise ValueError(
            "RA-EE-09 Slice 09E requires at least one held-out trajectory."
        )

    run_power_settings = _run_power_settings_with_fixed_specs(
        settings,
        fixed_specs=(held_spec,),
        implementation_sublabel="RA-EE-09 Slice 09E matched held-out replay",
    )
    control_metadata = dict(settings.metadata)
    control_metadata.update(
        {
            "implementation_sublabel": (
                "RA-EE-09 Slice 09E matched held-out equal-share control"
            ),
            "candidate_allocator_enabled": False,
            "scalar_reward_success_basis": False,
        }
    )
    control_settings = _RAEE09Settings(
        method_label=settings.method_label,
        implementation_sublabel=control_metadata["implementation_sublabel"],
        power_settings=run_power_settings,
        resource=settings.resource,
        metadata=control_metadata,
    )
    candidate_settings = _candidate_settings_from_control(control_settings)
    candidate_metadata = dict(candidate_settings.metadata)
    candidate_metadata.update(
        {
            "implementation_sublabel": (
                "RA-EE-09 Slice 09E matched held-out bounded resource-share "
                "candidate"
            ),
            "held_out_bucket_id": HELD_OUT_BUCKET,
            "eval_seeds": list(held_spec.evaluation_seed_set),
            "fixed_association_trajectory_families": list(
                held_spec.trajectory_families
            ),
            "predeclared_resource_efficiency_metric": (
                RA_EE_09_PREDECLARED_RESOURCE_EFFICIENCY_METRIC
            ),
            "candidate_allocator_enabled": True,
            "scalar_reward_success_basis": False,
        }
    )
    candidate_settings = replace(
        candidate_settings,
        implementation_sublabel=candidate_metadata["implementation_sublabel"],
        metadata=candidate_metadata,
    )

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
    paired_rows = _build_paired_step_rows(
        control_rows=control_rows,
        candidate_rows=candidate_rows,
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
    control_budget_report = _resource_budget_report(
        control_rows,
        resource_allocator_id=RA_EE_09_EQUAL_SHARE_ALLOCATOR,
        equal_share_control=True,
    )
    candidate_budget_report = _resource_budget_report(
        candidate_rows,
        resource_allocator_id=RA_EE_09_CANDIDATE_ALLOCATOR,
        equal_share_control=False,
    )
    all_control_user_throughputs = [
        float(value)
        for values in control_user_throughputs_by_key.values()
        for value in values
    ]
    all_candidate_user_throughputs = [
        float(value)
        for values in candidate_user_throughputs_by_key.values()
        for value in values
    ]
    control_overall = _aggregate_replay_metrics(
        control_rows,
        user_throughputs=all_control_user_throughputs,
    )
    candidate_overall = _aggregate_replay_metrics(
        candidate_rows,
        user_throughputs=all_candidate_user_throughputs,
    )
    deltas = _comparison_deltas(
        control=control_overall,
        candidate=candidate_overall,
    )
    control_by_policy = {
        str(row["trajectory_policy"]): row for row in control_summaries
    }
    candidate_by_policy = {
        str(row["trajectory_policy"]): row for row in candidate_summaries
    }
    by_trajectory: list[dict[str, Any]] = []
    for policy in sorted(set(control_by_policy) & set(candidate_by_policy)):
        control = control_by_policy[policy]
        candidate = candidate_by_policy[policy]
        by_trajectory.append(
            {
                "scope": "trajectory",
                "trajectory_policy": policy,
                "control_EE_system_aggregate_bps_per_w": control[
                    "EE_system_aggregate_bps_per_w"
                ],
                "candidate_EE_system_aggregate_bps_per_w": candidate[
                    "EE_system_aggregate_bps_per_w"
                ],
                "EE_system_delta_bps_per_w": (
                    None
                    if control["EE_system_aggregate_bps_per_w"] is None
                    or candidate["EE_system_aggregate_bps_per_w"] is None
                    else float(candidate["EE_system_aggregate_bps_per_w"])
                    - float(control["EE_system_aggregate_bps_per_w"])
                ),
                "control_sum_throughput_bps": control["total_throughput_bps"],
                "candidate_sum_throughput_bps": candidate["total_throughput_bps"],
                "sum_throughput_delta_bps": float(candidate["total_throughput_bps"])
                - float(control["total_throughput_bps"]),
                "control_mean_throughput_bps": control[
                    "throughput_mean_user_step_bps"
                ],
                "candidate_mean_throughput_bps": candidate[
                    "throughput_mean_user_step_bps"
                ],
                "control_p05_throughput_bps": control[
                    "throughput_p05_user_step_bps"
                ],
                "candidate_p05_throughput_bps": candidate[
                    "throughput_p05_user_step_bps"
                ],
                "p05_throughput_ratio": _safe_ratio(
                    candidate["throughput_p05_user_step_bps"],
                    control["throughput_p05_user_step_bps"],
                ),
                "control_served_ratio": control["served_ratio"],
                "candidate_served_ratio": candidate["served_ratio"],
                "served_ratio_delta": float(candidate["served_ratio"])
                - float(control["served_ratio"]),
                "control_outage_ratio": control["outage_ratio"],
                "candidate_outage_ratio": candidate["outage_ratio"],
                "outage_ratio_delta": float(candidate["outage_ratio"])
                - float(control["outage_ratio"]),
                "handover_count_delta": int(candidate["handover_count"])
                - int(control["handover_count"]),
                "total_active_power_w_sum_delta": float(
                    candidate["total_active_beam_power_sum_w"]
                )
                - float(control["total_active_beam_power_sum_w"]),
                "predeclared_resource_efficiency_metric": (
                    RA_EE_09_PREDECLARED_RESOURCE_EFFICIENCY_METRIC
                ),
                "predeclared_resource_efficiency_delta": (
                    None
                    if control["throughput_p05_user_step_bps"] is None
                    or candidate["throughput_p05_user_step_bps"] is None
                    else float(candidate["throughput_p05_user_step_bps"])
                    / float(candidate["active_beam_count_distribution"]["mean"])
                    - float(control["throughput_p05_user_step_bps"])
                    / float(control["active_beam_count_distribution"]["mean"])
                ),
            }
        )

    matched_boundary = _matched_boundary_proof(
        paired_rows=paired_rows,
        held_out_seed_set=held_spec.evaluation_seed_set,
        held_out_policies=held_spec.trajectory_families,
    )
    ranking = _throughput_vs_ee_ranking_comparison(
        control_overall=control_overall,
        candidate_overall=candidate_overall,
        paired_rows=paired_rows,
        control_summaries=control_summaries,
        candidate_summaries=candidate_summaries,
    )
    comparison = {
        "control_overall": control_overall,
        "candidate_overall": candidate_overall,
        "deltas": deltas,
        "by_trajectory": by_trajectory,
        "throughput_vs_EE_ranking_separation": ranking,
    }
    decision_detail = _build_09e_decision(
        comparison=comparison,
        matched_boundary=matched_boundary,
        control_rows=control_rows,
        candidate_rows=candidate_rows,
        paired_rows=paired_rows,
        control_budget_report=control_budget_report,
        candidate_budget_report=candidate_budget_report,
        settings=control_settings,
    )

    protocol = {
        "phase": RA_EE_09_GATE_ID,
        "implementation_slice": "09E",
        "method_label": candidate_settings.method_label,
        "method_family": "RA-EE fixed-association resource allocation",
        "implementation_sublabel": candidate_settings.implementation_sublabel,
        "training": "none; offline replay only",
        "offline_replay_only": True,
        "held_out_bucket_id": HELD_OUT_BUCKET,
        "evaluation_seed_set": list(held_spec.evaluation_seed_set),
        "fixed_association_trajectories": list(held_spec.trajectory_families),
        "association_mode": "fixed-replay",
        "association_training": "disabled",
        "learned_association": "disabled",
        "hierarchical_RL": "disabled",
        "joint_association_power_training": "disabled",
        "catfish": "disabled",
        "phase03c_continuation": "disabled",
        "matched_control": RA_EE_09_CONTROL,
        "primary_candidate": RA_EE_09_CANDIDATE,
        "power_allocator_id": RA_EE_09_POWER_ALLOCATOR_ID,
        "primary_deployable_allocator": (
            candidate_settings.power_settings.primary_deployable_allocator
        ),
        "resource_unit": candidate_settings.resource.resource_unit,
        "control_resource_allocator_id": RA_EE_09_EQUAL_SHARE_ALLOCATOR,
        "candidate_resource_allocator_id": RA_EE_09_CANDIDATE_ALLOCATOR,
        "throughput_formula_version": (
            candidate_settings.resource.throughput_formula_version
        ),
        "resource_allocation_order": "after-power-vector-selection",
        "resource_allocation_feedback_to_power_decision": False,
        "same_power_vector_as_control_required": True,
        "system_EE_primary": True,
        "predeclared_resource_efficiency_metric": (
            RA_EE_09_PREDECLARED_RESOURCE_EFFICIENCY_METRIC
        ),
        "scalar_reward_success_basis": False,
        "per_user_EE_credit_success_basis": False,
    }
    constraints = {
        "per_beam_budget": candidate_settings.resource.per_beam_budget,
        "total_resource_budget": "active_beam_count",
        "inactive_beam_resource_policy": (
            candidate_settings.resource.inactive_beam_resource_policy
        ),
        "candidate_per_user_min": "0.25/N_b",
        "candidate_per_user_max": "min(4/N_b, 1.0)",
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
        "p05_throughput_min_ratio_vs_control": (
            candidate_settings.power_settings.audit.p05_min_ratio_vs_control
        ),
        "served_ratio_min_delta_vs_control": (
            candidate_settings.power_settings.audit.served_ratio_min_delta_vs_control
        ),
        "outage_ratio_max_delta_vs_control": (
            candidate_settings.power_settings.audit.outage_ratio_max_delta_vs_control
        ),
        "max_positive_gain_contribution_share": (
            RA_EE_09_MAX_POSITIVE_GAIN_CONTRIBUTION_SHARE
        ),
    }
    resource_budget_report = {
        "control": control_budget_report,
        "candidate": candidate_budget_report,
    }
    summary = {
        "inputs": {
            "config_path": str(Path(config_path)),
            "output_dir": str(Path(output_dir)),
            "max_steps": max_steps,
        },
        "metadata": candidate_settings.metadata,
        "protocol": protocol,
        "constraints": constraints,
        "matched_boundary_proof": matched_boundary,
        "matched_comparison": comparison,
        "control_summaries": control_summaries,
        "candidate_summaries": candidate_summaries,
        "resource_budget_report": resource_budget_report,
        "decision_detail": decision_detail,
        "proof_flags": decision_detail["proof_flags"],
        "acceptance_criteria": decision_detail["acceptance_criteria"],
        "stop_conditions": decision_detail["stop_conditions"],
        "ra_ee_09_slice_09e_decision": (
            decision_detail["ra_ee_09_slice_09e_decision"]
        ),
        "remaining_blockers": (
            []
            if decision_detail["ra_ee_09_slice_09e_decision"] == "PASS"
            else [
                "RA-EE-09 RB / bandwidth allocation effectiveness is not promoted unless all Slice 09E acceptance criteria pass.",
                "No learned association, hierarchical RL, joint training, or Catfish path exists.",
                "Physical 3GPP RB semantics and integer RB rounding remain out of scope.",
            ]
        ),
        "forbidden_claims_still_active": [
            "Do not call RA-EE-09 full RA-EE-MODQN.",
            "Do not claim learned association effectiveness.",
            "Do not claim old EE-MODQN effectiveness.",
            "Do not claim HOBS optimizer behavior.",
            "Do not claim physical energy saving.",
            "Do not claim Catfish-EE or Catfish repair.",
            "Do not claim full paper-faithful reproduction.",
        ],
    }

    out_dir = Path(output_dir)
    paired_step_csv = _write_csv(
        out_dir / "ra_ee_09_matched_step_comparison.csv",
        paired_rows,
        fieldnames=_fieldnames(paired_rows),
    )
    summary_rows = _matched_comparison_summary_rows(summary)
    summary_csv = _write_csv(
        out_dir / "ra_ee_09_matched_summary.csv",
        summary_rows,
        fieldnames=_fieldnames(summary_rows),
    )
    budget_path = write_json(
        out_dir / "resource_budget_report.json",
        resource_budget_report,
    )
    comparison_path = write_json(
        out_dir / "paired_comparison.json",
        summary,
    )
    review_path = out_dir / "review.md"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text("\n".join(_matched_comparison_review_lines(summary)) + "\n")
    return {
        "paired_comparison": comparison_path,
        "ra_ee_09_matched_step_comparison": paired_step_csv,
        "ra_ee_09_matched_summary_csv": summary_csv,
        "resource_budget_report": budget_path,
        "review_md": review_path,
        "summary": summary,
    }
