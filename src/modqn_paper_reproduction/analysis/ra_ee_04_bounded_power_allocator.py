"""RA-EE-04 fixed-association centralized power-allocation pilot.

This module keeps association trajectories fixed and evaluates only a
centralized per-active-beam power allocator. It does not train handover,
does not invoke Catfish, and does not mutate the frozen MODQN baseline.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config_loader import build_environment, get_seeds, load_training_yaml
from ..env.step import HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK
from ._common import write_json
from .ra_ee_02_oracle_power_allocation import (
    COUNTERFACTUAL_POLICIES,
    _AuditSettings,
    _StepSnapshot,
    _build_guardrail_checks,
    _build_ranking_checks,
    _build_unit_power_snapshots,
    _evaluate_power_vector,
    _format_vector,
    _power_vector_for_candidate,
    _qos_guardrails_pass,
    _rollout_counterfactual_trajectories,
    _select_oracle_step,
    _summarize_all,
)


DEFAULT_CONTROL_CONFIG = (
    "configs/ra-ee-04-bounded-power-allocator-control.resolved.yaml"
)
DEFAULT_CANDIDATE_CONFIG = (
    "configs/ra-ee-04-bounded-power-allocator-candidate.resolved.yaml"
)
DEFAULT_CONTROL_OUTPUT_DIR = (
    "artifacts/ra-ee-04-bounded-power-allocator-control-pilot"
)
DEFAULT_CANDIDATE_OUTPUT_DIR = (
    "artifacts/ra-ee-04-bounded-power-allocator-candidate-pilot"
)
DEFAULT_COMPARISON_OUTPUT_DIR = (
    "artifacts/ra-ee-04-bounded-power-allocator-candidate-pilot/"
    "paired-comparison-vs-control"
)

RA_EE_04_POLICIES = COUNTERFACTUAL_POLICIES
RA_EE_04_CANDIDATE = "safe-greedy-power-allocator"
RA_EE_04_ORACLE = "constrained-oracle-upper-bound"


@dataclass(frozen=True)
class _RAEE04Settings:
    method_label: str
    implementation_sublabel: str
    audit: _AuditSettings
    training_episodes: int
    train_seed: int
    environment_seed: int
    mobility_seed: int
    evaluation_seed_set: tuple[int, ...]
    primary_policies: tuple[str, ...]
    candidate_max_demoted_beams: int


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


def _ra_ee_04_value(cfg: dict[str, Any]) -> dict[str, Any]:
    value = (
        cfg.get("resolved_assumptions", {})
        .get("ra_ee_04_bounded_power_allocator", {})
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


def _settings_from_config(cfg: dict[str, Any]) -> _RAEE04Settings:
    pilot = _ra_ee_04_value(cfg)
    power = _power_surface_value(cfg)
    seeds = get_seeds(cfg)

    levels_raw = pilot.get(
        "codebook_levels_w",
        power.get("power_codebook_levels_w", [0.5, 0.75, 1.0, 1.5, 2.0]),
    )
    levels = tuple(float(level) for level in levels_raw)
    if not levels:
        raise ValueError("RA-EE-04 requires at least one power codebook level.")
    if tuple(sorted(levels)) != levels:
        raise ValueError(f"RA-EE-04 codebook levels must be sorted, got {levels!r}.")

    policies_raw = pilot.get("primary_fixed_association_trajectories", RA_EE_04_POLICIES)
    policies = tuple(str(policy) for policy in policies_raw)
    unsupported = sorted(set(policies) - set(RA_EE_04_POLICIES))
    if unsupported:
        raise ValueError(f"Unsupported RA-EE-04 fixed trajectory policies: {unsupported!r}")

    training_episodes = int(pilot.get("training_episodes", 20))
    if training_episodes != 20:
        raise ValueError(
            "RA-EE-04 bounded pilot must use exactly 20 training episodes, "
            f"got {training_episodes}."
        )

    eval_seed_set = tuple(
        int(seed)
        for seed in pilot.get(
            "evaluation_seed_set",
            seeds.get("evaluation_seed_set", [100, 200, 300, 400, 500]),
        )
    )
    if not eval_seed_set:
        raise ValueError("RA-EE-04 requires evaluation seeds.")

    audit = _AuditSettings(
        method_label=str(pilot.get("method_label", "RA-EE-MDP")),
        codebook_levels_w=levels,
        fixed_control_power_w=float(pilot.get("fixed_control_power_w", 1.0)),
        total_power_budget_w=float(
            pilot.get("total_active_power_budget_w", power.get("total_power_budget_w", 8.0))
        ),
        per_beam_max_power_w=float(
            pilot.get("per_beam_max_power_w", power.get("max_power_w", 2.0))
        ),
        active_base_power_w=float(
            pilot.get("active_base_power_w", power.get("active_base_power_w", 0.25))
        ),
        load_scale_power_w=float(
            pilot.get("load_scale_power_w", power.get("load_scale_power_w", 0.35))
        ),
        load_exponent=float(
            pilot.get("load_exponent", power.get("load_exponent", 0.5))
        ),
        p05_min_ratio_vs_control=float(
            pilot.get("p05_throughput_min_ratio_vs_control", 0.95)
        ),
        served_ratio_min_delta_vs_control=float(
            pilot.get("served_ratio_min_delta_vs_control", 0.0)
        ),
        outage_ratio_max_delta_vs_control=float(
            pilot.get("outage_ratio_max_delta_vs_control", 0.0)
        ),
        oracle_max_demoted_beams=int(pilot.get("oracle_max_demoted_beams", 3)),
    )
    return _RAEE04Settings(
        method_label=str(pilot.get("method_label", "RA-EE-MDP")),
        implementation_sublabel=str(
            pilot.get(
                "implementation_sublabel",
                "RA-EE-04 fixed-association power-allocation pilot",
            )
        ),
        audit=audit,
        training_episodes=training_episodes,
        train_seed=int(pilot.get("train_seed", seeds.get("train_seed", 42))),
        environment_seed=int(
            pilot.get("environment_seed", seeds.get("environment_seed", 1337))
        ),
        mobility_seed=int(pilot.get("mobility_seed", seeds.get("mobility_seed", 7))),
        evaluation_seed_set=eval_seed_set,
        primary_policies=policies,
        candidate_max_demoted_beams=int(
            pilot.get("candidate_max_demoted_beams", audit.oracle_max_demoted_beams)
        ),
    )


def _power_vector_key(power_vector: np.ndarray, active_mask: np.ndarray) -> str:
    return _format_vector(power_vector[active_mask])


def _safe_greedy_power_vector(
    snapshot: _StepSnapshot,
    settings: _RAEE04Settings,
) -> tuple[np.ndarray, str]:
    """Select a bounded power vector by greedy QoS-safe demotion.

    The allocator starts from fixed 1 W per active beam and repeatedly tries one
    finite-codebook demotion. It accepts only actions that improve system EE and
    keep p05 throughput, served ratio, outage, budget, per-beam power, and
    inactive-beam-zero constraints satisfied versus the matched fixed control.
    """
    audit = settings.audit
    control_vector = _power_vector_for_candidate(snapshot, audit, "fixed-control")
    control_row = _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics="fixed-control",
        selected_power_profile=f"fixed-{audit.fixed_control_power_w:g}w-control",
        power_vector=control_vector,
        settings=audit,
    )
    current = control_vector.copy()
    current_row = control_row
    active_indices = [int(idx) for idx in np.flatnonzero(snapshot.active_mask).tolist()]
    lower_levels = [
        float(level)
        for level in audit.codebook_levels_w
        if float(level) < audit.fixed_control_power_w
    ]
    demoted: set[int] = set()

    while len(demoted) < settings.candidate_max_demoted_beams:
        best_row: dict[str, Any] | None = None
        best_vector: np.ndarray | None = None
        best_idx: int | None = None
        for beam_idx in active_indices:
            if beam_idx in demoted:
                continue
            for level in lower_levels:
                requested = current.copy()
                requested[beam_idx] = float(level)
                row = _evaluate_power_vector(
                    snapshot=snapshot,
                    power_semantics=RA_EE_04_CANDIDATE,
                    selected_power_profile=(
                        f"safe-greedy:{_power_vector_key(requested, snapshot.active_mask)}"
                    ),
                    power_vector=requested,
                    settings=audit,
                )
                constraints_ok = (
                    not bool(row["budget_violation"])
                    and not bool(row["per_beam_power_violation"])
                    and not bool(row["inactive_beam_nonzero_power"])
                    and _qos_guardrails_pass(
                        candidate=row,
                        control=control_row,
                        settings=audit,
                    )
                )
                if not constraints_ok:
                    continue
                row_ee = float(row["EE_system_bps_per_w"] or -math.inf)
                current_ee = float(current_row["EE_system_bps_per_w"] or -math.inf)
                if row_ee <= current_ee + 1e-12:
                    continue
                if best_row is None:
                    best_row = row
                    best_vector = requested
                    best_idx = beam_idx
                    continue
                best_ee = float(best_row["EE_system_bps_per_w"] or -math.inf)
                if (
                    row_ee > best_ee + 1e-12
                    or (
                        abs(row_ee - best_ee) <= 1e-12
                        and float(row["sum_user_throughput_bps"])
                        > float(best_row["sum_user_throughput_bps"])
                    )
                ):
                    best_row = row
                    best_vector = requested
                    best_idx = beam_idx

        if best_row is None or best_vector is None or best_idx is None:
            break
        current = best_vector
        current_row = best_row
        demoted.add(best_idx)

    label = f"safe-greedy:{_power_vector_key(current, snapshot.active_mask)}"
    return current, label


def _candidate_step_row(
    snapshot: _StepSnapshot,
    settings: _RAEE04Settings,
) -> dict[str, Any]:
    requested, label = _safe_greedy_power_vector(snapshot, settings)
    row = _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics=RA_EE_04_CANDIDATE,
        selected_power_profile=label,
        power_vector=requested,
        settings=settings.audit,
    )
    row["requested_power_vector_w"] = _format_vector(requested)
    row["effective_power_vector_w"] = row["beam_transmit_power_w"]
    row["power_repair_used"] = False
    row["association_control"] = "fixed-by-trajectory"
    row["learned_association_enabled"] = False
    return row


def _control_step_row(
    snapshot: _StepSnapshot,
    settings: _RAEE04Settings,
) -> dict[str, Any]:
    powers = _power_vector_for_candidate(snapshot, settings.audit, "fixed-control")
    row = _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics="fixed-control",
        selected_power_profile=f"fixed-{settings.audit.fixed_control_power_w:g}w-control",
        power_vector=powers,
        settings=settings.audit,
    )
    row["requested_power_vector_w"] = _format_vector(powers)
    row["effective_power_vector_w"] = row["beam_transmit_power_w"]
    row["power_repair_used"] = False
    row["association_control"] = "fixed-by-trajectory"
    row["learned_association_enabled"] = False
    return row


def _oracle_step_row(
    snapshot: _StepSnapshot,
    control_row: dict[str, Any],
    settings: _RAEE04Settings,
) -> dict[str, Any]:
    row = _select_oracle_step(
        snapshot=snapshot,
        control_row=control_row,
        settings=settings.audit,
    )
    row["power_semantics"] = RA_EE_04_ORACLE
    row["requested_power_vector_w"] = row["beam_transmit_power_w"]
    row["effective_power_vector_w"] = row["beam_transmit_power_w"]
    row["power_repair_used"] = False
    row["association_control"] = "fixed-by-trajectory"
    row["learned_association_enabled"] = False
    return row


def _evaluation_rows(
    *,
    snapshots: list[_StepSnapshot],
    settings: _RAEE04Settings,
    include_oracle: bool,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[float]]]:
    rows: list[dict[str, Any]] = []
    user_throughputs_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    for snapshot in snapshots:
        control = _control_step_row(snapshot, settings)
        candidate = _candidate_step_row(snapshot, settings)
        step_rows = [control, candidate]
        if include_oracle:
            step_rows.append(_oracle_step_row(snapshot, control, settings))
        for row in step_rows:
            throughputs = row.pop("_user_throughputs")
            rows.append(row)
            user_throughputs_by_key[
                (str(row["trajectory_policy"]), str(row["power_semantics"]))
            ].extend(float(value) for value in throughputs.tolist())
    return rows, user_throughputs_by_key


def _comparison_ranking_checks(
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_policy: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summaries:
        by_policy[str(row["trajectory_policy"])][str(row["power_semantics"])] = row

    checks: list[dict[str, Any]] = []
    for policy, rows in sorted(by_policy.items()):
        compared = [
            rows[key]
            for key in ("fixed-control", RA_EE_04_CANDIDATE)
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
                "trajectory_policy": policy,
                "compared_power_semantics": ["fixed-control", RA_EE_04_CANDIDATE],
                "throughput_rescore_ranking": throughput_ranking,
                "EE_rescore_ranking": ee_ranking,
                "throughput_rescore_winner": throughput_ranking[0],
                "EE_rescore_winner": ee_ranking[0],
                "same_policy_throughput_rescore_vs_EE_rescore_ranking_changes": (
                    throughput_ranking != ee_ranking
                ),
                "same_policy_throughput_rescore_vs_EE_rescore_top_changes": (
                    throughput_ranking[0] != ee_ranking[0]
                ),
            }
        )
    return checks


def _training_schedule(settings: _RAEE04Settings) -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    for episode_idx in range(settings.training_episodes):
        policy = settings.primary_policies[episode_idx % len(settings.primary_policies)]
        schedule.append(
            {
                "episode": episode_idx + 1,
                "trajectory_policy": policy,
                "fixed_association_seed": settings.train_seed + episode_idx,
            }
        )
    return schedule


def _training_metrics(
    *,
    cfg: dict[str, Any],
    settings: _RAEE04Settings,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _training_schedule(settings):
        policy = str(item["trajectory_policy"])
        seed = int(item["fixed_association_seed"])
        trajectories = _rollout_counterfactual_trajectories(
            cfg=cfg,
            evaluation_seed_set=(seed,),
            max_steps=None,
            policies=(policy,),
        )
        snapshots = _build_unit_power_snapshots(
            base_cfg=cfg,
            settings=settings.audit,
            trajectories=trajectories,
        )
        eval_rows, user_throughputs_by_key = _evaluation_rows(
            snapshots=snapshots,
            settings=settings,
            include_oracle=False,
        )
        summaries = _summarize_all(
            rows=eval_rows,
            user_throughputs_by_key=user_throughputs_by_key,
        )
        candidate = next(
            row for row in summaries if row["power_semantics"] == RA_EE_04_CANDIDATE
        )
        rows.append(
            {
                "episode": item["episode"],
                "trajectory_policy": policy,
                "fixed_association_seed": seed,
                "association_training": "disabled",
                "catfish": "disabled",
                "candidate_EE_system_aggregate_bps_per_w": candidate[
                    "EE_system_aggregate_bps_per_w"
                ],
                "candidate_throughput_mean_user_step_bps": candidate[
                    "throughput_mean_user_step_bps"
                ],
                "candidate_throughput_p05_user_step_bps": candidate[
                    "throughput_p05_user_step_bps"
                ],
                "candidate_selected_profile_distinct_count": candidate[
                    "selected_profile_distinct_count"
                ],
                "candidate_denominator_varies_in_episode": candidate[
                    "denominator_varies_in_eval"
                ],
                "scalar_reward_diagnostic": candidate[
                    "EE_system_step_mean_bps_per_w"
                ],
            }
        )
    return rows


def _candidate_guardrails(
    *,
    summaries: list[dict[str, Any]],
    guardrail_checks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate_by_policy = {
        str(row["trajectory_policy"]): row
        for row in summaries
        if row["power_semantics"] == RA_EE_04_CANDIDATE
    }
    return [
        row
        for row in guardrail_checks
        if row["power_semantics"] == RA_EE_04_CANDIDATE
        and str(row["trajectory_policy"]) in candidate_by_policy
    ]


def _build_decision(
    *,
    summaries: list[dict[str, Any]],
    guardrail_checks: list[dict[str, Any]],
    ranking_checks: list[dict[str, Any]],
    settings: _RAEE04Settings,
) -> dict[str, Any]:
    candidate_summaries = [
        row for row in summaries if row["power_semantics"] == RA_EE_04_CANDIDATE
    ]
    candidate_guardrails = _candidate_guardrails(
        summaries=summaries,
        guardrail_checks=guardrail_checks,
    )
    candidate_by_policy = {
        str(row["trajectory_policy"]): row for row in candidate_summaries
    }
    guardrail_by_policy = {
        str(row["trajectory_policy"]): row for row in candidate_guardrails
    }
    ranking_by_policy = {
        str(row["trajectory_policy"]): row for row in ranking_checks
    }

    expected_policies = set(settings.primary_policies)
    has_all_policies = expected_policies.issubset(candidate_by_policy)
    fixed_association_only = all(
        row.get("association_control") in (None, "fixed-by-trajectory")
        for row in candidate_summaries
    )
    learned_association_disabled = all(
        row.get("learned_association_enabled") in (None, False)
        for row in candidate_summaries
    )
    denominator_varies = all(
        bool(candidate_by_policy[policy]["denominator_varies_in_eval"])
        for policy in expected_policies
        if policy in candidate_by_policy
    )
    selected_profile_not_single = all(
        int(candidate_by_policy[policy]["selected_profile_distinct_count"]) > 1
        for policy in expected_policies
        if policy in candidate_by_policy
    )
    active_power_not_single = all(
        len(candidate_by_policy[policy]["total_active_beam_power_w_distribution"]["distinct"])
        > 1
        for policy in expected_policies
        if policy in candidate_by_policy
    )
    not_collapsed = all(
        float(candidate_by_policy[policy]["one_active_beam_step_ratio"]) < 1.0
        for policy in expected_policies
        if policy in candidate_by_policy
    )
    ee_improves = all(
        bool(guardrail_by_policy.get(policy, {}).get("accepted"))
        and float(guardrail_by_policy[policy]["EE_system_delta_vs_fixed_control"]) > 0.0
        for policy in expected_policies
        if policy in guardrail_by_policy
    )
    qos_pass = all(
        bool(guardrail_by_policy.get(policy, {}).get("QoS_guardrails_pass"))
        for policy in expected_policies
        if policy in guardrail_by_policy
    )
    no_power_violations = all(
        bool(guardrail_by_policy.get(policy, {}).get("budget_guardrail_pass"))
        and bool(guardrail_by_policy[policy]["per_beam_power_guardrail_pass"])
        and bool(guardrail_by_policy[policy]["inactive_beam_zero_w_guardrail_pass"])
        for policy in expected_policies
        if policy in guardrail_by_policy
    )
    ranking_separates = all(
        bool(
            ranking_by_policy.get(policy, {}).get(
                "same_policy_throughput_rescore_vs_EE_rescore_ranking_changes"
            )
        )
        for policy in expected_policies
        if policy in ranking_by_policy
    )

    proof_flags = {
        "fixed_association_only": fixed_association_only,
        "learned_association_disabled": learned_association_disabled,
        "catfish_disabled": True,
        "multi_catfish_disabled": True,
        "training_episodes_is_20": settings.training_episodes == 20,
        "all_primary_noncollapsed_trajectories_present": has_all_policies,
        "denominator_varies_in_eval": denominator_varies,
        "selected_power_vector_not_single_point": selected_profile_not_single,
        "total_active_power_not_single_point": active_power_not_single,
        "not_all_one_active_beam": not_collapsed,
        "EE_system_improves_vs_fixed_control": ee_improves,
        "QoS_guardrails_pass": qos_pass,
        "zero_budget_per_beam_inactive_power_violations": no_power_violations,
        "ranking_separates_or_rescore_changes": ranking_separates,
        "scalar_reward_success_basis": False,
        "per_user_EE_credit_success_basis": False,
    }

    stop_conditions = {
        "association_learned_jointly": not learned_association_disabled,
        "trajectory_collapsed_to_one_active_beam": not not_collapsed,
        "power_action_collapsed_to_one_point": not selected_profile_not_single,
        "denominator_remains_fixed": not denominator_varies,
        "EE_gain_from_QoS_collapse": any(
            float(row["EE_system_delta_vs_fixed_control"] or 0.0) > 0.0
            and not bool(row["QoS_guardrails_pass"])
            for row in candidate_guardrails
        ),
        "budget_or_inactive_power_violation": not no_power_violations,
        "catfish_introduced": False,
        "frozen_baseline_mutation": False,
        "only_scalar_reward_improves": False,
    }

    required_true_fields = (
        "fixed_association_only",
        "learned_association_disabled",
        "catfish_disabled",
        "multi_catfish_disabled",
        "training_episodes_is_20",
        "all_primary_noncollapsed_trajectories_present",
        "denominator_varies_in_eval",
        "selected_power_vector_not_single_point",
        "total_active_power_not_single_point",
        "not_all_one_active_beam",
        "EE_system_improves_vs_fixed_control",
        "QoS_guardrails_pass",
        "zero_budget_per_beam_inactive_power_violations",
        "ranking_separates_or_rescore_changes",
    )
    pass_required = all(bool(proof_flags[field]) for field in required_true_fields)
    pass_required = (
        pass_required
        and proof_flags["scalar_reward_success_basis"] is False
        and proof_flags["per_user_EE_credit_success_basis"] is False
    )

    if pass_required:
        decision = "PASS"
    elif any(bool(value) for value in stop_conditions.values()):
        decision = "BLOCKED"
    else:
        decision = "NEEDS MORE EVIDENCE"

    return {
        "ra_ee_04_decision": decision,
        "proof_flags": proof_flags,
        "stop_conditions": stop_conditions,
        "candidate_guardrail_checks": candidate_guardrails,
        "allowed_claim": (
            "PASS only means this bounded fixed-association centralized "
            "power-allocation pilot passed its implementation gate."
            if decision == "PASS"
            else "Do not promote RA-EE-04 beyond bounded pilot evidence."
        ),
    }


def _compact_summary_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "trajectory_policy": row["trajectory_policy"],
            "power_semantics": row["power_semantics"],
            "EE_system_aggregate_bps_per_w": row["EE_system_aggregate_bps_per_w"],
            "EE_system_step_mean_bps_per_w": row["EE_system_step_mean_bps_per_w"],
            "throughput_mean_user_step_bps": row["throughput_mean_user_step_bps"],
            "throughput_p05_user_step_bps": row["throughput_p05_user_step_bps"],
            "served_ratio": row["served_ratio"],
            "outage_ratio": row["outage_ratio"],
            "active_beam_count_distribution": row["active_beam_count_distribution"],
            "selected_power_profile_distribution": row[
                "selected_power_profile_distribution"
            ],
            "total_active_beam_power_w_distribution": row[
                "total_active_beam_power_w_distribution"
            ],
            "denominator_varies_in_eval": row["denominator_varies_in_eval"],
            "one_active_beam_step_ratio": row["one_active_beam_step_ratio"],
            "budget_violation_step_count": row["budget_violations"]["step_count"],
            "per_beam_power_violation_step_count": row[
                "per_beam_power_violations"
            ]["step_count"],
            "inactive_beam_nonzero_power_step_count": row[
                "inactive_beam_nonzero_power_step_count"
            ],
            "throughput_EE_pearson": row[
                "throughput_vs_EE_system_correlation"
            ]["pearson"],
            "throughput_EE_spearman": row[
                "throughput_vs_EE_system_correlation"
            ]["spearman"],
        }
        for row in summaries
    ]


def _review_lines(summary: dict[str, Any]) -> list[str]:
    decision = summary["ra_ee_04_decision"]
    proof = summary["proof_flags"]
    lines = [
        "# RA-EE-04 Bounded Power-Allocator Pilot Review",
        "",
        "Fixed-association centralized power-allocation pilot only. No learned "
        "association, Catfish, multi-Catfish, old EE-MODQN continuation, long "
        "training, HOBS optimizer claim, physical energy-saving claim, or frozen "
        "baseline mutation was performed.",
        "",
        "## Protocol",
        "",
        f"- method label: `{summary['protocol']['method_label']}`",
        f"- implementation sublabel: `{summary['protocol']['implementation_sublabel']}`",
        f"- training episodes: `{summary['protocol']['training_episodes']}`",
        f"- evaluation seeds: `{summary['protocol']['evaluation_seed_set']}`",
        f"- fixed trajectories: `{summary['protocol']['fixed_association_trajectories']}`",
        f"- action: `{summary['protocol']['candidate_action_contract']}`",
        f"- levels W: `{summary['constraints']['codebook_levels_w']}`",
        f"- total active budget W: `{summary['constraints']['total_power_budget_w']}`",
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
            f"- RA-EE-04 decision: `{decision}`",
            f"- allowed claim: {summary['decision_detail']['allowed_claim']}",
            "",
            "## Forbidden Claims",
            "",
        ]
    )
    for claim in summary["forbidden_claims_still_active"]:
        lines.append(f"- {claim}")
    return lines


def export_ra_ee_04_bounded_power_allocator_pilot(
    control_config_path: str | Path = DEFAULT_CONTROL_CONFIG,
    candidate_config_path: str | Path = DEFAULT_CANDIDATE_CONFIG,
    control_output_dir: str | Path = DEFAULT_CONTROL_OUTPUT_DIR,
    candidate_output_dir: str | Path = DEFAULT_CANDIDATE_OUTPUT_DIR,
    comparison_output_dir: str | Path = DEFAULT_COMPARISON_OUTPUT_DIR,
    *,
    evaluation_seed_set: tuple[int, ...] | None = None,
    max_steps: int | None = None,
    policies: tuple[str, ...] = RA_EE_04_POLICIES,
    include_oracle: bool = True,
) -> dict[str, Any]:
    """Run the RA-EE-04 bounded fixed-association power-allocation pilot."""
    if max_steps is not None and max_steps < 1:
        raise ValueError(f"max_steps must be >= 1 when provided, got {max_steps!r}")

    control_cfg = load_training_yaml(control_config_path)
    candidate_cfg = load_training_yaml(candidate_config_path)
    control_env = build_environment(control_cfg)
    candidate_env = build_environment(candidate_cfg)
    if (
        control_env.power_surface_config.hobs_power_surface_mode
        != HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK
        or candidate_env.power_surface_config.hobs_power_surface_mode
        != HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK
    ):
        raise ValueError("RA-EE-04 configs must opt into the power-codebook surface.")

    settings = _settings_from_config(candidate_cfg)
    eval_seeds = (
        tuple(int(seed) for seed in evaluation_seed_set)
        if evaluation_seed_set is not None
        else settings.evaluation_seed_set
    )
    run_settings = _RAEE04Settings(
        method_label=settings.method_label,
        implementation_sublabel=settings.implementation_sublabel,
        audit=settings.audit,
        training_episodes=settings.training_episodes,
        train_seed=settings.train_seed,
        environment_seed=settings.environment_seed,
        mobility_seed=settings.mobility_seed,
        evaluation_seed_set=eval_seeds,
        primary_policies=tuple(policies),
        candidate_max_demoted_beams=settings.candidate_max_demoted_beams,
    )

    training_rows = _training_metrics(cfg=candidate_cfg, settings=run_settings)

    trajectories = _rollout_counterfactual_trajectories(
        cfg=candidate_cfg,
        evaluation_seed_set=eval_seeds,
        max_steps=max_steps,
        policies=tuple(policies),
    )
    snapshots = _build_unit_power_snapshots(
        base_cfg=candidate_cfg,
        settings=run_settings.audit,
        trajectories=trajectories,
    )
    step_rows, user_throughputs_by_key = _evaluation_rows(
        snapshots=snapshots,
        settings=run_settings,
        include_oracle=include_oracle,
    )
    summaries = _summarize_all(
        rows=step_rows,
        user_throughputs_by_key=user_throughputs_by_key,
    )
    all_ranking_checks = _build_ranking_checks(summaries)
    comparison_ranking_checks = _comparison_ranking_checks(summaries)
    guardrail_checks = _build_guardrail_checks(
        candidate_summaries=summaries,
        settings=run_settings.audit,
    )
    decision_detail = _build_decision(
        summaries=summaries,
        guardrail_checks=guardrail_checks,
        ranking_checks=comparison_ranking_checks,
        settings=run_settings,
    )

    control_dir = Path(control_output_dir)
    candidate_dir = Path(candidate_output_dir)
    comparison_dir = Path(comparison_output_dir)
    control_rows = [row for row in step_rows if row["power_semantics"] == "fixed-control"]
    candidate_rows = [
        row for row in step_rows if row["power_semantics"] == RA_EE_04_CANDIDATE
    ]

    control_step_csv = _write_csv(
        control_dir / "ra_ee_04_control_step_metrics.csv",
        control_rows,
        fieldnames=list(control_rows[0].keys()),
    )
    candidate_step_csv = _write_csv(
        candidate_dir / "ra_ee_04_candidate_step_metrics.csv",
        candidate_rows,
        fieldnames=list(candidate_rows[0].keys()),
    )
    training_csv = _write_csv(
        candidate_dir / "ra_ee_04_training_metrics.csv",
        training_rows,
        fieldnames=list(training_rows[0].keys()),
    )
    comparison_step_csv = _write_csv(
        comparison_dir / "ra_ee_04_paired_step_metrics.csv",
        step_rows,
        fieldnames=list(step_rows[0].keys()),
    )
    summary_csv = _write_csv(
        comparison_dir / "ra_ee_04_candidate_summary.csv",
        _compact_summary_rows(summaries),
        fieldnames=list(_compact_summary_rows(summaries)[0].keys()),
    )

    common_protocol = {
        "phase": "RA-EE-04",
        "method_label": run_settings.method_label,
        "method_family": "RA-EE-MDP",
        "implementation_sublabel": run_settings.implementation_sublabel,
        "training": "20-episode bounded calibration only",
        "training_episodes": run_settings.training_episodes,
        "train_seed": run_settings.train_seed,
        "environment_seed": run_settings.environment_seed,
        "mobility_seed": run_settings.mobility_seed,
        "evaluation_seed_set": list(eval_seeds),
        "fixed_association_trajectories": list(policies),
        "learned_association": "disabled",
        "association_control": "fixed-by-trajectory",
        "catfish": "disabled",
        "multi_catfish": "disabled",
        "old_EE_MODQN_continuation": "forbidden/not-performed",
        "frozen_baseline_mutation": "forbidden/not-performed",
        "hobs_optimizer_claim": "forbidden/not-made",
        "physical_energy_saving_claim": "forbidden/not-made",
        "candidate_action_contract": (
            "centralized per-active-beam discrete power vector; inactive beams 0 W"
        ),
        "candidate_allocator": RA_EE_04_CANDIDATE,
        "control_allocator": "fixed-control-1w-per-active-beam",
        "oracle_upper_bound_exported": include_oracle,
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
        "candidate_max_demoted_beams": run_settings.candidate_max_demoted_beams,
        "invalid_budget_actions": "rejected by action mask/evaluator",
        "power_repair": "not-used; requested and effective vectors are still exported",
    }
    summary = {
        "inputs": {
            "control_config_path": str(Path(control_config_path)),
            "candidate_config_path": str(Path(candidate_config_path)),
            "control_output_dir": str(control_dir),
            "candidate_output_dir": str(candidate_dir),
            "comparison_output_dir": str(comparison_dir),
            "max_steps": max_steps,
        },
        "protocol": common_protocol,
        "constraints": constraints,
        "candidate_summaries": summaries,
        "guardrail_checks": guardrail_checks,
        "ranking_separation_result": {
            "comparison_control_vs_candidate": comparison_ranking_checks,
            "all_power_semantics": all_ranking_checks,
            "ranking_separates_or_rescore_changes": decision_detail["proof_flags"][
                "ranking_separates_or_rescore_changes"
            ],
        },
        "decision_detail": decision_detail,
        "proof_flags": decision_detail["proof_flags"],
        "stop_conditions": decision_detail["stop_conditions"],
        "ra_ee_04_decision": decision_detail["ra_ee_04_decision"],
        "remaining_blockers": [
            "This is fixed-association centralized power-allocation evidence only.",
            "It is not learned association, not old EE-MODQN, not Catfish, and not a HOBS optimizer.",
            "A PASS does not claim full paper-faithful reproduction or physical energy saving.",
        ],
        "forbidden_claims_still_active": [
            "Do not claim old EE-MODQN effectiveness.",
            "Do not claim HOBS optimizer behavior.",
            "Do not claim Catfish, multi-Catfish, or Catfish-EE-MODQN effectiveness.",
            "Do not treat per-user EE credit as system EE.",
            "Do not use scalar reward alone as success evidence.",
            "Do not claim full paper-faithful reproduction.",
            "Do not claim physical energy saving.",
        ],
    }

    control_metadata = {
        "protocol": common_protocol | {"comparison_role": "paired-control"},
        "constraints": constraints,
        "artifacts": {
            "step_metrics": str(control_step_csv),
        },
    }
    candidate_metadata = {
        "protocol": common_protocol | {"comparison_role": "paired-candidate"},
        "constraints": constraints,
        "artifacts": {
            "training_metrics": str(training_csv),
            "step_metrics": str(candidate_step_csv),
        },
    }
    control_summary_path = write_json(
        control_dir / "ra_ee_04_control_summary.json",
        control_metadata,
    )
    candidate_summary_path = write_json(
        candidate_dir / "ra_ee_04_candidate_summary.json",
        candidate_metadata,
    )
    summary_path = write_json(
        comparison_dir / "ra_ee_04_bounded_power_allocator_summary.json",
        summary,
    )
    review_path = comparison_dir / "review.md"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text("\n".join(_review_lines(summary)) + "\n")

    return {
        "ra_ee_04_bounded_power_allocator_summary": summary_path,
        "ra_ee_04_control_summary": control_summary_path,
        "ra_ee_04_candidate_summary": candidate_summary_path,
        "ra_ee_04_control_step_metrics": control_step_csv,
        "ra_ee_04_candidate_step_metrics": candidate_step_csv,
        "ra_ee_04_training_metrics": training_csv,
        "ra_ee_04_paired_step_metrics": comparison_step_csv,
        "ra_ee_04_candidate_summary_csv": summary_csv,
        "review_md": review_path,
        "summary": summary,
    }


__all__ = [
    "DEFAULT_CANDIDATE_CONFIG",
    "DEFAULT_CANDIDATE_OUTPUT_DIR",
    "DEFAULT_COMPARISON_OUTPUT_DIR",
    "DEFAULT_CONTROL_CONFIG",
    "DEFAULT_CONTROL_OUTPUT_DIR",
    "RA_EE_04_CANDIDATE",
    "RA_EE_04_ORACLE",
    "export_ra_ee_04_bounded_power_allocator_pilot",
]
