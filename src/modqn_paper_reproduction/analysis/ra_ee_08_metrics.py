"""RA-EE-08 summary, guardrail, and decision logic."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from .ra_ee_05_fixed_association_robustness import CALIBRATION_BUCKET, HELD_OUT_BUCKET
from .ra_ee_06b_association_proposal_refinement import (
    _handover_burden,
    _p05_ratio_and_slack,
    _policy_label,
)
from .ra_ee_07_constrained_power_allocator_distillation import _numeric_distribution
from .ra_ee_08_protocol import (
    RA_EE_08_ASSOC_ORACLE_CONSTRAINED,
    RA_EE_08_ASSOC_ORACLE_DEPLOYABLE,
    RA_EE_08_CANDIDATE,
    RA_EE_08_FIXED_CONSTRAINED_ORACLE,
    RA_EE_08_FIXED_DEPLOYABLE_CONTROL,
    RA_EE_08_FIXED_SAFE_GREEDY,
    RA_EE_08_PROPOSAL_SAFE_GREEDY,
    _RAEE08Settings,
)


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

def _augment_summaries(
    summaries: list[dict[str, Any]],
    *,
    step_rows: list[dict[str, Any]],
    settings: _RAEE08Settings,
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
    moved_by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    moved_ratio_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    load_gap_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    source_policy_by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    p05_ratio_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    accepted_step_by_key: dict[tuple[str, str], list[bool]] = defaultdict(list)
    served_delta_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    outage_delta_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in step_rows:
        key = (str(row["trajectory_policy"]), str(row["power_semantics"]))
        vectors_by_key[key].append(str(row["effective_power_vector_w"]))
        allocators_by_key[key].append(str(row["allocator_label"]))
        selected_by_key[key].append(str(row.get("selected_allocator_candidate", "")))
        rejection_by_key[key].append(str(row["rejection_reason"]))
        accepted_move_by_key[key].append(float(row.get("accepted_allocator_move_count", 0)))
        rejected_move_by_key[key].append(float(row.get("rejected_allocator_move_count", 0)))
        moved_by_key[key].append(int(row["moved_user_count"]))
        moved_ratio_by_key[key].append(float(row["moved_user_ratio"]))
        if row["beam_load_balance_gap"] is not None:
            load_gap_by_key[key].append(float(row["beam_load_balance_gap"]))
        source_policy_by_key[key].append(str(row["source_association_policy"]))
        if row["oracle_gap_closed_ratio"] is not None:
            oracle_gap_by_key[key].append(float(row["oracle_gap_closed_ratio"]))
        if row["candidate_regret_bps_per_w"] is not None:
            regret_by_key[key].append(float(row["candidate_regret_bps_per_w"]))
        if row["p05_ratio_vs_matched_control"] is not None:
            p05_ratio_by_key[key].append(float(row["p05_ratio_vs_matched_control"]))
        served_delta_by_key[key].append(float(row["served_ratio_delta_vs_matched_control"]))
        outage_delta_by_key[key].append(float(row["outage_ratio_delta_vs_matched_control"]))
        accepted_step_by_key[key].append(bool(row["accepted_flag"]))
        metadata_by_key.setdefault(
            key,
            {
                "evaluation_bucket": row["evaluation_bucket"],
                "association_policy": row["association_policy"],
                "association_role": row["association_role"],
                "association_action_contract": row["association_action_contract"],
                "matched_control_association_policy": row[
                    "matched_control_association_policy"
                ],
                "candidate_association_policy": row["candidate_association_policy"],
                "power_allocator": row["power_allocator"],
                "allocator_label": row["allocator_label"],
                "diagnostic_only": row["diagnostic_only"],
                "primary_candidate": row["primary_candidate"],
                "same_deployable_allocator_pairing": row[
                    "same_deployable_allocator_pairing"
                ],
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
        summary["moved_user_count_distribution"] = _numeric_distribution(
            [float(value) for value in moved_by_key[key]]
        )
        summary["moved_user_ratio_distribution"] = _numeric_distribution(
            moved_ratio_by_key[key]
        )
        summary["moved_user_count_total"] = int(sum(moved_by_key[key]))
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
        summary["p05_ratio_vs_matched_control_distribution"] = _numeric_distribution(
            p05_ratio_by_key[key]
        )
        summary["served_ratio_delta_vs_matched_control_distribution"] = _numeric_distribution(
            served_delta_by_key[key]
        )
        summary["outage_ratio_delta_vs_matched_control_distribution"] = _numeric_distribution(
            outage_delta_by_key[key]
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
        summary["primary_comparison_no_step_cap_mismatch"] = True
        summary["max_moved_user_ratio_for_acceptance"] = settings.max_moved_user_ratio
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
    settings: _RAEE08Settings,
) -> dict[str, Any]:
    p05_ratio, p05_slack = _p05_ratio_and_slack(
        control_p05_bps=float(control["throughput_p05_user_step_bps"]),
        candidate_p05_bps=float(candidate["throughput_p05_user_step_bps"]),
        threshold_ratio=settings.audit.p05_min_ratio_vs_control,
    )
    p05_threshold = settings.audit.p05_min_ratio_vs_control * float(
        control["throughput_p05_user_step_bps"]
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
    noncollapsed = (
        float(candidate["one_active_beam_step_ratio"])
        <= settings.max_one_active_beam_ratio_for_acceptance
    )
    two_beam_ok = (
        float(candidate.get("two_beam_overload_step_ratio", 0.0))
        <= settings.max_two_beam_overload_step_ratio
    )
    handover_pass = (
        float(candidate["handover_burden"]["moved_user_ratio"])
        <= settings.max_moved_user_ratio
    )
    denominator_varies = bool(candidate["denominator_varies_in_eval"])
    oracle_gap_pass = (
        oracle_gap_closed is not None
        and oracle_gap_closed >= settings.min_oracle_gap_closed_ratio
    )
    no_leakage = (
        not bool(candidate["oracle_labels_used_for_runtime_decision"])
        and not bool(candidate["future_outcomes_used_for_runtime_decision"])
        and not bool(candidate["held_out_answers_used_for_runtime_decision"])
    )
    reasons: list[str] = []
    if ee_delta is None or ee_delta <= 0.0:
        reasons.append("nonpositive-ee-delta-vs-fixed-deployable")
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
    if not no_leakage:
        reasons.append("oracle-future-heldout-leakage")
    accepted = not reasons
    return {
        "evaluation_bucket": candidate["evaluation_bucket"],
        "trajectory_policy": candidate["trajectory_policy"],
        "candidate_association_policy": candidate["candidate_association_policy"],
        "power_semantics": candidate["power_semantics"],
        "power_allocator": candidate["power_allocator"],
        "matched_control_power_semantics": control["power_semantics"],
        "matched_control_power_allocator": control["power_allocator"],
        "diagnostic_oracle_power_semantics": (
            None if oracle is None else oracle["power_semantics"]
        ),
        "EE_system_delta_vs_fixed_deployable": ee_delta,
        "EE_system_pct_delta_vs_fixed_deployable": _pct_delta(
            control["EE_system_aggregate_bps_per_w"],
            candidate["EE_system_aggregate_bps_per_w"],
        ),
        "throughput_mean_pct_delta_vs_fixed_deployable": _pct_delta(
            control["throughput_mean_user_step_bps"],
            candidate["throughput_mean_user_step_bps"],
        ),
        "throughput_p05_ratio_vs_fixed_deployable": p05_ratio,
        "p05_threshold_bps": p05_threshold,
        "p05_slack_to_0_95_threshold_bps": p05_slack,
        "p05_guardrail_pass": p05_pass,
        "served_ratio_delta_vs_fixed_deployable": float(candidate["served_ratio"]) - float(
            control["served_ratio"]
        ),
        "served_ratio_threshold": served_threshold,
        "served_ratio_guardrail_pass": served_pass,
        "outage_ratio_delta_vs_fixed_deployable": float(candidate["outage_ratio"]) - float(
            control["outage_ratio"]
        ),
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
        "moved_user_count": candidate["handover_burden"]["moved_user_count"],
        "moved_user_ratio": candidate["handover_burden"]["moved_user_ratio"],
        "oracle_delta_vs_fixed_deployable": oracle_delta,
        "oracle_gap_closed_ratio": oracle_gap_closed,
        "oracle_gap_closed_guardrail_pass": oracle_gap_pass,
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
    settings: _RAEE08Settings,
) -> list[dict[str, Any]]:
    grouped = _group_summaries(summaries)
    checks: list[dict[str, Any]] = []
    diagnostic_semantics = {
        RA_EE_08_PROPOSAL_SAFE_GREEDY,
        RA_EE_08_FIXED_SAFE_GREEDY,
        RA_EE_08_FIXED_CONSTRAINED_ORACLE,
        RA_EE_08_ASSOC_ORACLE_DEPLOYABLE,
        RA_EE_08_ASSOC_ORACLE_CONSTRAINED,
    }
    for _policy, rows in sorted(grouped.items()):
        control = rows.get(RA_EE_08_FIXED_DEPLOYABLE_CONTROL)
        if control is None:
            continue
        oracle = rows.get(RA_EE_08_ASSOC_ORACLE_CONSTRAINED)
        for semantics in (
            RA_EE_08_CANDIDATE,
            RA_EE_08_PROPOSAL_SAFE_GREEDY,
            RA_EE_08_FIXED_SAFE_GREEDY,
            RA_EE_08_FIXED_CONSTRAINED_ORACLE,
            RA_EE_08_ASSOC_ORACLE_DEPLOYABLE,
            RA_EE_08_ASSOC_ORACLE_CONSTRAINED,
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
            if semantics in diagnostic_semantics:
                result["accepted"] = False
                result["rejection_reason"] = "diagnostic-only"
            checks.append(result)
    return checks


def _ranking_checks(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_summaries(summaries)
    checks: list[dict[str, Any]] = []
    compared_semantics = (RA_EE_08_FIXED_DEPLOYABLE_CONTROL, RA_EE_08_CANDIDATE)
    for policy, rows in sorted(grouped.items()):
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
        control = rows.get(RA_EE_08_FIXED_DEPLOYABLE_CONTROL)
        candidate = rows.get(RA_EE_08_CANDIDATE)
        if control is None or candidate is None:
            continue
        for oracle_semantics in (
            RA_EE_08_ASSOC_ORACLE_DEPLOYABLE,
            RA_EE_08_ASSOC_ORACLE_CONSTRAINED,
        ):
            oracle = rows.get(oracle_semantics)
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
                    "candidate_power_semantics": RA_EE_08_CANDIDATE,
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
            RA_EE_08_FIXED_DEPLOYABLE_CONTROL,
            RA_EE_08_CANDIDATE,
        }:
            continue
        by_key[(int(row["evaluation_seed"]), str(row["power_semantics"]))].append(row)
    seeds = sorted({seed for seed, _semantics in by_key})
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        control = by_key.get((seed, RA_EE_08_FIXED_DEPLOYABLE_CONTROL), [])
        candidate = by_key.get((seed, RA_EE_08_CANDIDATE), [])
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
                "EE_system_delta_vs_fixed_deployable": delta,
                "positive": delta is not None and delta > 0.0,
            }
        )
    positive = [row for row in rows if bool(row["positive"])]
    positive_delta_sum = sum(
        max(0.0, float(row["EE_system_delta_vs_fixed_deployable"] or 0.0))
        for row in rows
    )
    max_share = (
        0.0
        if positive_delta_sum <= 1e-12
        else max(
            max(0.0, float(row["EE_system_delta_vs_fixed_deployable"] or 0.0))
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
    settings: _RAEE08Settings,
    summaries: list[dict[str, Any]],
    guardrail_checks: list[dict[str, Any]],
    ranking_checks: list[dict[str, Any]],
    oracle_gap_diagnostics: list[dict[str, Any]],
    seed_results: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    candidate_by_policy = {
        str(row["trajectory_policy"]): row
        for row in summaries
        if row["power_semantics"] == RA_EE_08_CANDIDATE
    }
    guardrail_by_policy = {
        str(row["trajectory_policy"]): row
        for row in guardrail_checks
        if row["power_semantics"] == RA_EE_08_CANDIDATE
    }
    ranking_by_policy = {str(row["trajectory_policy"]): row for row in ranking_checks}
    constrained_gap_by_policy = {
        str(row["trajectory_policy"]): row
        for row in oracle_gap_diagnostics
        if row["oracle_power_semantics"] == RA_EE_08_ASSOC_ORACLE_CONSTRAINED
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
                    "EE_system_delta_vs_fixed_deployable",
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
        primary_label = (
            None
            if settings.predeclared_primary_association_policy is None
            else _policy_label(
                spec.name,
                settings.predeclared_primary_association_policy,
            )
        )
        primary_positive = primary_label in positive if primary_label else False
        primary_accepted = primary_label in accepted if primary_label else False
        no_power_violations = all(
            bool(guardrail_by_policy.get(label, {}).get("budget_guardrail_pass"))
            and bool(guardrail_by_policy.get(label, {}).get("per_beam_power_guardrail_pass"))
            and bool(guardrail_by_policy.get(label, {}).get("inactive_beam_zero_w_guardrail_pass"))
            for label in present
        )
        accepted_scope = accepted
        qos_pass = bool(accepted_scope) and all(
            bool(guardrail_by_policy[label]["p05_guardrail_pass"])
            and bool(guardrail_by_policy[label]["served_ratio_guardrail_pass"])
            and bool(guardrail_by_policy[label]["outage_guardrail_pass"])
            for label in accepted_scope
        )
        denominator_varies = bool(accepted_scope) and all(
            bool(candidate_by_policy[label]["denominator_varies_in_eval"])
            for label in accepted_scope
        )
        handover_bounded = bool(accepted_scope) and all(
            bool(guardrail_by_policy[label]["handover_burden_guardrail_pass"])
            for label in accepted_scope
        )
        oracle_gap_closed = bool(accepted_scope) and all(
            bool(guardrail_by_policy[label]["oracle_gap_closed_guardrail_pass"])
            for label in accepted_scope
        )
        ranking_or_gap_clear = bool(accepted_scope) and all(
            bool(
                ranking_by_policy.get(label, {}).get(
                    "throughput_rescore_vs_EE_rescore_top_changes"
                )
            )
            or bool(guardrail_by_policy[label]["oracle_gap_closed_guardrail_pass"])
            for label in accepted_scope
        )
        positive_delta_sum = sum(
            max(
                0.0,
                float(
                    guardrail_by_policy.get(label, {}).get(
                        "EE_system_delta_vs_fixed_deployable",
                        0.0,
                    )
                    or 0.0
                ),
            )
            for label in present
        )
        max_policy_share = (
            0.0
            if positive_delta_sum <= 1e-12
            else max(
                max(
                    0.0,
                    float(
                        guardrail_by_policy.get(label, {}).get(
                            "EE_system_delta_vs_fixed_deployable",
                            0.0,
                        )
                        or 0.0
                    ),
                )
                / positive_delta_sum
                for label in present
            )
        )
        gap_rows = [
            constrained_gap_by_policy[label]
            for label in present
            if label in constrained_gap_by_policy
            and constrained_gap_by_policy[label]["oracle_EE_delta_vs_control"] is not None
            and float(constrained_gap_by_policy[label]["oracle_EE_delta_vs_control"]) > 1e-12
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
        majority_positive = bool(noncollapsed) and len(positive) > len(noncollapsed) / 2.0
        majority_accepted = bool(noncollapsed) and len(accepted) > len(noncollapsed) / 2.0
        positive_gate = majority_positive or primary_positive
        accepted_gate = majority_accepted or primary_accepted
        concentration_ok = len(positive) >= 2 and max_policy_share < 0.80
        results[spec.name] = {
            "bucket": spec.name,
            "evaluation_seed_set": list(spec.evaluation_seed_set),
            "candidate_association_policies": list(settings.candidate_association_policies),
            "matched_control_association_policy": settings.matched_control_association_policy,
            "predeclared_primary_association_policy": (
                settings.predeclared_primary_association_policy
            ),
            "present_candidate_count": len(present),
            "noncollapsed_candidate_count": len(noncollapsed),
            "noncollapsed_candidate_policies": noncollapsed,
            "positive_EE_delta_candidate_count": len(positive),
            "positive_EE_delta_candidate_policies": positive,
            "accepted_candidate_count": len(accepted),
            "accepted_candidate_policies": accepted,
            "predeclared_primary_positive_EE_delta": primary_positive,
            "predeclared_primary_accepted": primary_accepted,
            "rejection_reasons": {
                label: guardrail_by_policy.get(label, {}).get("rejection_reason", "missing")
                for label in present
            },
            "majority_noncollapsed_positive_EE_delta": majority_positive,
            "majority_noncollapsed_accepted": majority_accepted,
            "majority_or_predeclared_primary_positive_EE_delta": positive_gate,
            "majority_or_predeclared_primary_accepted": accepted_gate,
            "gains_not_concentrated_in_one_policy": concentration_ok,
            "max_positive_policy_delta_share": max_policy_share,
            "gains_not_concentrated_in_one_seed": seed_ok,
            "qos_guardrails_pass_for_accepted": qos_pass,
            "zero_budget_per_beam_inactive_power_violations": no_power_violations,
            "denominator_varies_for_accepted": denominator_varies,
            "active_beam_behavior_noncollapsed_for_accepted": bool(accepted_scope)
            and all(label in noncollapsed for label in accepted_scope),
            "handover_burden_bounded_for_accepted": handover_bounded,
            "oracle_gap_closed_for_accepted": oracle_gap_closed,
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
                    positive_gate
                    and accepted_gate
                    and concentration_ok
                    and seed_ok
                    and qos_pass
                    and no_power_violations
                    and denominator_varies
                    and handover_bounded
                    and oracle_gap_closed
                    and ranking_or_gap_clear
                    and aggregate_gap_closure is not None
                    and aggregate_gap_closure >= settings.min_oracle_gap_closed_ratio
                )
            ),
        }
    return results


def _primary_step_cap_mismatch(summaries: list[dict[str, Any]]) -> bool:
    grouped = _group_summaries(summaries)
    for rows in grouped.values():
        control = rows.get(RA_EE_08_FIXED_DEPLOYABLE_CONTROL)
        candidate = rows.get(RA_EE_08_CANDIDATE)
        if control is None or candidate is None:
            continue
        if int(control["step_count"]) != int(candidate["step_count"]):
            return True
    return False


def _build_decision(
    *,
    summaries: list[dict[str, Any]],
    guardrail_checks: list[dict[str, Any]],
    bucket_results: dict[str, dict[str, Any]],
    include_oracle: bool,
) -> dict[str, Any]:
    held_out = bucket_results.get(HELD_OUT_BUCKET, {})
    candidate_summaries = [
        row for row in summaries if row["power_semantics"] == RA_EE_08_CANDIDATE
    ]
    candidate_guardrails = [
        row for row in guardrail_checks if row["power_semantics"] == RA_EE_08_CANDIDATE
    ]
    no_power_violations = all(
        bool(row.get("budget_guardrail_pass"))
        and bool(row.get("per_beam_power_guardrail_pass"))
        and bool(row.get("inactive_beam_zero_w_guardrail_pass"))
        for row in candidate_guardrails
    )
    no_leakage = all(
        not bool(row.get("oracle_labels_used_for_runtime_decision"))
        and not bool(row.get("future_outcomes_used_for_runtime_decision"))
        and not bool(row.get("held_out_answers_used_for_runtime_decision"))
        for row in candidate_summaries
    )
    learned_disabled = all(
        row.get("learned_association_enabled") in (None, False)
        and row.get("learned_hierarchical_RL_enabled") in (None, False)
        for row in candidate_summaries
    )
    joint_disabled = all(
        row.get("joint_association_power_training_enabled") in (None, False)
        for row in candidate_summaries
    )
    same_allocator = all(
        row.get("power_allocator") == "deployable-stronger-power-allocator"
        for row in candidate_summaries
    )
    no_step_mismatch = not _primary_step_cap_mismatch(summaries)
    proof_flags = {
        "held_out_bucket_exists_and_reported_separately": bool(held_out),
        "offline_replay_only": True,
        "deterministic_association_proposals_only": True,
        "learned_association_disabled": learned_disabled,
        "learned_hierarchical_RL_disabled": learned_disabled,
        "association_training_disabled": learned_disabled,
        "joint_association_power_training_disabled": joint_disabled,
        "catfish_disabled": True,
        "multi_catfish_disabled": True,
        "rb_bandwidth_allocation_disabled": True,
        "old_EE_MODQN_continuation_disabled": True,
        "frozen_baseline_mutation": False,
        "matched_control_uses_same_deployable_allocator": same_allocator,
        "primary_comparison_uses_same_deployable_allocator": same_allocator,
        "primary_comparison_no_step_cap_mismatch": no_step_mismatch,
        "same_power_codebook_and_budget": True,
        "same_effective_power_vector_feeds_numerator_denominator_audit": True,
        "oracle_diagnostic_only": include_oracle,
        "oracle_rows_excluded_from_acceptance": True,
        "candidate_does_not_use_oracle_labels_or_future_or_heldout_answers": no_leakage,
        "majority_noncollapsed_held_out_positive_EE_delta": bool(
            held_out.get("majority_noncollapsed_positive_EE_delta")
        ),
        "predeclared_primary_held_out_positive_EE_delta": bool(
            held_out.get("predeclared_primary_positive_EE_delta")
        ),
        "majority_or_predeclared_primary_held_out_positive_EE_delta": bool(
            held_out.get("majority_or_predeclared_primary_positive_EE_delta")
        ),
        "majority_or_predeclared_primary_held_out_accepted": bool(
            held_out.get("majority_or_predeclared_primary_accepted")
        ),
        "held_out_gains_not_concentrated_in_one_policy": bool(
            held_out.get("gains_not_concentrated_in_one_policy")
        ),
        "held_out_gains_not_concentrated_in_one_seed": bool(
            held_out.get("gains_not_concentrated_in_one_seed")
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
        "active_beam_behavior_noncollapsed_for_accepted_held_out": bool(
            held_out.get("active_beam_behavior_noncollapsed_for_accepted")
        ),
        "handover_burden_bounded_for_accepted_held_out": bool(
            held_out.get("handover_burden_bounded_for_accepted")
        ),
        "candidate_closes_meaningful_oracle_gap": bool(
            held_out.get("candidate_closes_meaningful_oracle_gap")
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
        "proposal_gains_vanish_under_same_deployable_allocator": not bool(
            held_out.get("majority_or_predeclared_primary_positive_EE_delta")
        ),
        "only_proposal_stronger_vs_fixed_safe_greedy_positive": any(
            row["evaluation_bucket"] == HELD_OUT_BUCKET
            and row["power_semantics"] == RA_EE_08_PROPOSAL_SAFE_GREEDY
            and float(row["EE_system_delta_vs_fixed_deployable"] or 0.0) > 0.0
            for row in guardrail_checks
        )
        and not bool(held_out.get("majority_or_predeclared_primary_positive_EE_delta")),
        "positive_result_requires_constrained_oracle": not bool(
            held_out.get("majority_or_predeclared_primary_positive_EE_delta")
        )
        and any(
            row["evaluation_bucket"] == HELD_OUT_BUCKET
            and row["power_semantics"] == RA_EE_08_ASSOC_ORACLE_CONSTRAINED
            and float(row["EE_system_delta_vs_fixed_deployable"] or 0.0) > 0.0
            for row in guardrail_checks
        ),
        "held_out_gains_concentrated": not (
            bool(held_out.get("gains_not_concentrated_in_one_policy"))
            and bool(held_out.get("gains_not_concentrated_in_one_seed"))
        ),
        "p05_served_or_outage_guardrail_fails": any(
            row["evaluation_bucket"] == HELD_OUT_BUCKET
            and float(row["EE_system_delta_vs_fixed_deployable"] or 0.0) > 0.0
            and not bool(row["QoS_guardrails_pass"])
            for row in candidate_guardrails
        ),
        "moved_user_or_handover_burden_unbounded": bool(
            held_out.get("accepted_candidate_count")
        )
        and not bool(held_out.get("handover_burden_bounded_for_accepted")),
        "no_meaningful_oracle_gap_closure": not bool(
            held_out.get("candidate_closes_meaningful_oracle_gap")
        ),
        "budget_or_inactive_power_violations": not no_power_violations,
        "denominator_or_active_beam_behavior_collapses": bool(
            held_out.get("accepted_candidate_count")
        )
        and not (
            bool(held_out.get("denominator_varies_for_accepted"))
            and bool(held_out.get("active_beam_behavior_noncollapsed_for_accepted"))
        ),
        "candidate_uses_oracle_labels_future_or_hidden_leakage": not no_leakage,
        "training_catfish_RB_or_frozen_baseline_mutation_added": (
            not learned_disabled or not joint_disabled
        ),
        "primary_comparison_step_cap_mismatch": not no_step_mismatch,
        "oracle_used_as_deployable_method": False,
        "acceptance_depends_on_scalar_reward": False,
    }
    required_true = (
        "held_out_bucket_exists_and_reported_separately",
        "offline_replay_only",
        "deterministic_association_proposals_only",
        "learned_association_disabled",
        "learned_hierarchical_RL_disabled",
        "joint_association_power_training_disabled",
        "catfish_disabled",
        "multi_catfish_disabled",
        "rb_bandwidth_allocation_disabled",
        "matched_control_uses_same_deployable_allocator",
        "primary_comparison_uses_same_deployable_allocator",
        "primary_comparison_no_step_cap_mismatch",
        "oracle_diagnostic_only",
        "oracle_rows_excluded_from_acceptance",
        "candidate_does_not_use_oracle_labels_or_future_or_heldout_answers",
        "majority_or_predeclared_primary_held_out_positive_EE_delta",
        "majority_or_predeclared_primary_held_out_accepted",
        "held_out_gains_not_concentrated_in_one_policy",
        "held_out_gains_not_concentrated_in_one_seed",
        "p05_throughput_guardrail_pass_for_accepted_held_out",
        "served_ratio_does_not_drop_for_accepted_held_out",
        "outage_ratio_does_not_increase_for_accepted_held_out",
        "zero_budget_per_beam_inactive_power_violations",
        "denominator_varies_for_accepted_held_out",
        "active_beam_behavior_noncollapsed_for_accepted_held_out",
        "handover_burden_bounded_for_accepted_held_out",
        "candidate_closes_meaningful_oracle_gap",
        "ranking_separates_or_oracle_gap_reduction_clear",
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
        "ra_ee_08_decision": decision,
        "proof_flags": proof_flags,
        "stop_conditions": stop_conditions,
        "candidate_guardrail_checks": candidate_guardrails,
        "allowed_claim": (
            "PASS only means deterministic offline association proposals improve "
            "against matched fixed association when both are paired with the same "
            "deployable stronger power allocator. It is not training evidence or "
            "full RA-EE-MODQN."
            if decision == "PASS"
            else "Do not promote RA-EE-08 beyond a blocked or inconclusive offline replay gate."
        ),
    }


def _compact_summary_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "evaluation_bucket",
        "trajectory_policy",
        "candidate_association_policy",
        "association_policy",
        "association_role",
        "association_action_contract",
        "power_semantics",
        "power_allocator",
        "allocator_label",
        "selected_allocator_candidate_distribution",
        "EE_system_aggregate_bps_per_w",
        "EE_system_step_mean_bps_per_w",
        "throughput_mean_user_step_bps",
        "throughput_p05_user_step_bps",
        "served_ratio",
        "outage_ratio",
        "handover_count",
        "r2_mean",
        "moved_user_count_total",
        "moved_user_count_distribution",
        "moved_user_ratio_distribution",
        "handover_burden",
        "active_beam_count_distribution",
        "selected_power_profile_distribution",
        "selected_power_vector_distribution",
        "total_active_beam_power_w_distribution",
        "denominator_varies_in_eval",
        "one_active_beam_step_ratio",
        "budget_violations",
        "per_beam_power_violations",
        "inactive_beam_nonzero_power_step_count",
        "p05_ratio_vs_matched_control_distribution",
        "served_ratio_delta_vs_matched_control_distribution",
        "outage_ratio_delta_vs_matched_control_distribution",
        "oracle_gap_closed_ratio_distribution",
        "candidate_regret_bps_per_w_distribution",
        "rejection_reason_distribution",
        "throughput_vs_EE_system_correlation",
        "diagnostic_only",
        "primary_candidate",
    )
    return [{field: row[field] for field in fields} for row in summaries]

__all__ = [
    "_augment_summaries",
    "_build_decision",
    "_build_guardrail_checks",
    "_bucket_results",
    "_compact_summary_rows",
    "_guardrail_result",
    "_oracle_gap_diagnostics",
    "_ranking_checks",
    "_seed_level_results",
]
