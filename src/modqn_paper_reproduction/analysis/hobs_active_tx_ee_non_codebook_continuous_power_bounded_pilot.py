"""CP-base non-codebook continuous-power bounded matched pilot.

This module is the bounded pilot runner authorized by the CP-base prompt only.
It compares exactly two roles: throughput-control versus ee-candidate. Both
roles share the analytic continuous power surface and QoS-sticky structural
guard; the only intended behavioral difference is r1_reward_mode.
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..config_loader import build_environment, build_trainer_config, get_seeds, load_training_yaml
from ..env.step import HOBS_POWER_SURFACE_NON_CODEBOOK_CONTINUOUS_POWER
from ..runtime.trainer_spec import (
    HOBS_ACTIVE_TX_EE_NON_CODEBOOK_CONTINUOUS_POWER_BOUNDED_PILOT_KIND,
    R1_REWARD_MODE_HOBS_ACTIVE_TX_EE,
    R1_REWARD_MODE_THROUGHPUT,
)
from ._common import write_json
from .hobs_active_tx_ee_anti_collapse import (
    HANDOVER_DELTA_MAX,
    MAX_TINY_EPISODES,
    OUTAGE_RATIO_MAX_DELTA,
    P05_THROUGHPUT_RATIO_MIN,
    R2_MEAN_DELTA_MIN,
    SERVED_RATIO_MIN_DELTA,
    _anti_collapse_subset,
    _config_power_surface_value,
    _distribution,
    _train_and_evaluate_arm,
    _unique_sorted,
    predeclared_tolerances,
)

THROUGHPUT_CONTROL_ROLE = "throughput-control"
EE_CANDIDATE_ROLE = "ee-candidate"
ROLE_ORDER = (THROUGHPUT_CONTROL_ROLE, EE_CANDIDATE_ROLE)
ROLE_CONFIGS: dict[str, Path] = {
    THROUGHPUT_CONTROL_ROLE: Path(
        "configs/hobs-active-tx-ee-non-codebook-continuous-power-"
        "bounded-pilot-throughput-control.resolved.yaml"
    ),
    EE_CANDIDATE_ROLE: Path(
        "configs/hobs-active-tx-ee-non-codebook-continuous-power-"
        "bounded-pilot-ee-candidate.resolved.yaml"
    ),
}
ARTIFACT_ROOTS: dict[str, Path] = {
    THROUGHPUT_CONTROL_ROLE: Path(
        "artifacts/hobs-active-tx-ee-non-codebook-continuous-power-"
        "bounded-pilot-throughput-control"
    ),
    EE_CANDIDATE_ROLE: Path(
        "artifacts/hobs-active-tx-ee-non-codebook-continuous-power-"
        "bounded-pilot-ee-candidate"
    ),
}
PAIRED_COMPARISON_DIR = (
    ARTIFACT_ROOTS[EE_CANDIDATE_ROLE]
    / "paired-comparison-vs-throughput-control"
)
SUMMARY_DIR = Path(
    "artifacts/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot-summary"
)
REPORT_PATH = Path(
    "docs/research/catfish-ee-modqn/"
    "hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot.execution-report.md"
)


MEAN_KEYS = (
    "EE_system",
    "EE_system_eval_aggregate",
    "EE_system_step_mean",
    "eta_EE_active_TX",
    "eta_EE_active_TX_step_mean",
    "raw_throughput_mean_bps",
    "raw_episode_throughput_mean_bps",
    "p05_throughput_bps",
    "served_ratio",
    "outage_ratio",
    "handover_count",
    "r2_mean",
    "load_balance_metric",
    "r3_mean",
    "scalar_reward_diagnostic_mean",
    "episode_scalar_reward_diagnostic_mean",
    "power_control_activity_rate",
    "continuous_power_activity_rate",
    "throughput_vs_ee_pearson",
    "throughput_vs_ee_spearman",
)
SUM_KEYS = (
    "budget_violation_count",
    "per_beam_power_violation_count",
    "inactive_beam_nonzero_power_step_count",
    "overflow_steps",
    "overflow_user_count",
    "sticky_override_count",
    "nonsticky_move_count",
    "qos_guard_reject_count",
    "handover_guard_reject_count",
)


def run_bounded_pilot() -> dict[str, Any]:
    """Run the six authorized bounded pilot runs and write artifacts."""
    boundary = prove_bounded_pilot_boundary()
    if not bool(boundary["matched_boundary_pass"]):
        summary = _blocked_boundary_summary(boundary)
        _write_all_summary_artifacts(summary, [])
        return summary

    rows: list[dict[str, Any]] = []
    for role in ROLE_ORDER:
        config_path = ROLE_CONFIGS[role]
        for seed_index, seed_triplet in enumerate(boundary["seed_triplets"]):
            run_dir = _role_seed_dir(role, seed_index, seed_triplet)
            materialized_config = _materialize_seed_config(
                config_path,
                role=role,
                seed_index=seed_index,
                seed_triplet=seed_triplet,
                output_dir=run_dir,
            )
            arm_summary = _train_and_evaluate_arm(
                materialized_config,
                run_dir,
                route_label="CP-base-EE-MODQN bounded matched pilot",
            )
            rows.append(
                {
                    "role": role,
                    "seed_index": seed_index,
                    "seed_triplet": list(seed_triplet),
                    "run_dir": str(run_dir),
                    "config_path": str(materialized_config),
                    "summary_path": arm_summary["artifact_paths"]["summary_json"],
                    "summary": arm_summary,
                }
            )

    summary = summarize_bounded_pilot_runs(rows, boundary)
    _write_all_summary_artifacts(summary, rows)
    return summary


def prove_bounded_pilot_boundary(
    role_configs: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Prove the two-role CP-base pilot boundary before metrics are interpreted."""
    role_configs = dict(role_configs or ROLE_CONFIGS)
    loaded = {role: load_training_yaml(path) for role, path in role_configs.items()}
    trainers = {role: build_trainer_config(cfg) for role, cfg in loaded.items()}
    envs = {role: build_environment(cfg) for role, cfg in loaded.items()}
    powers = {role: _config_power_surface_value(cfg) for role, cfg in loaded.items()}
    constraints = {role: _anti_collapse_subset(trainers[role]) for role in trainers}
    seed_triplets = _protocol_seed_triplets(loaded[THROUGHPUT_CONTROL_ROLE])
    eval_seeds = get_seeds(loaded[THROUGHPUT_CONTROL_ROLE])["evaluation_seed_set"]
    forbidden = _forbidden_mode_flags(loaded, trainers, powers)

    checks = {
        "required_roles_present": set(role_configs) == set(ROLE_ORDER),
        "same_training_experiment_kind": len(
            {trainer.training_experiment_kind for trainer in trainers.values()}
        )
        == 1
        and all(
            trainer.training_experiment_kind
            == HOBS_ACTIVE_TX_EE_NON_CODEBOOK_CONTINUOUS_POWER_BOUNDED_PILOT_KIND
            for trainer in trainers.values()
        ),
        "same_phase": len({trainer.phase for trainer in trainers.values()}) == 1,
        "same_episode_budget": len({trainer.episodes for trainer in trainers.values()}) == 1,
        "tiny_episode_budget": all(
            int(trainer.episodes) == MAX_TINY_EPISODES
            for trainer in trainers.values()
        ),
        "same_eval_seeds": all(
            get_seeds(cfg)["evaluation_seed_set"] == eval_seeds
            for cfg in loaded.values()
        ),
        "same_seed_triplets": all(
            _protocol_seed_triplets(cfg) == seed_triplets for cfg in loaded.values()
        ),
        "exact_required_seed_triplets": seed_triplets
        == ((42, 1337, 7), (43, 1338, 8), (44, 1339, 9)),
        "exact_required_eval_seeds": tuple(int(seed) for seed in eval_seeds)
        == (100, 200, 300, 400, 500),
        "same_objective_weights": len(
            {trainer.objective_weights for trainer in trainers.values()}
        )
        == 1,
        "same_trainer_hyperparameters": _same_training_hyperparameters(trainers),
        "same_checkpoint_protocol": _same_checkpoint_protocol(trainers),
        "same_environment_boundary": _same_environment_boundary(envs),
        "same_continuous_power_surface": (
            powers[THROUGHPUT_CONTROL_ROLE] == powers[EE_CANDIDATE_ROLE]
        ),
        "continuous_power_mode": all(
            env.power_surface_config.hobs_power_surface_mode
            == HOBS_POWER_SURFACE_NON_CODEBOOK_CONTINUOUS_POWER
            for env in envs.values()
        ),
        "same_qos_sticky_guard": (
            constraints[THROUGHPUT_CONTROL_ROLE] == constraints[EE_CANDIDATE_ROLE]
            and all(
                trainer.anti_collapse_action_constraint_enabled
                for trainer in trainers.values()
            )
            and constraints[THROUGHPUT_CONTROL_ROLE]["mode"]
            == "qos-sticky-overflow-reassignment"
        ),
        "same_nonsticky_handover_protections": all(
            not constraint["allow_nonsticky_moves"]
            and int(constraint["nonsticky_move_budget"]) == 0
            and int(constraint["min_active_beams_target"]) == 0
            for constraint in constraints.values()
        ),
        "throughput_control_r1": (
            trainers[THROUGHPUT_CONTROL_ROLE].r1_reward_mode
            == R1_REWARD_MODE_THROUGHPUT
        ),
        "ee_candidate_r1": (
            trainers[EE_CANDIDATE_ROLE].r1_reward_mode
            == R1_REWARD_MODE_HOBS_ACTIVE_TX_EE
        ),
        "only_intended_difference_is_r1_reward_mode": (
            trainers[THROUGHPUT_CONTROL_ROLE].r1_reward_mode
            != trainers[EE_CANDIDATE_ROLE].r1_reward_mode
            and _behavioral_boundary_subset(
                loaded[THROUGHPUT_CONTROL_ROLE],
                trainers[THROUGHPUT_CONTROL_ROLE],
                powers[THROUGHPUT_CONTROL_ROLE],
                constraints[THROUGHPUT_CONTROL_ROLE],
            )
            == _behavioral_boundary_subset(
                loaded[EE_CANDIDATE_ROLE],
                trainers[EE_CANDIDATE_ROLE],
                powers[EE_CANDIDATE_ROLE],
                constraints[EE_CANDIDATE_ROLE],
            )
        ),
        "finite_codebook_levels_absent": all(
            "power_codebook_levels_w" not in power
            and "finite_codebook_levels_w" not in power
            for power in powers.values()
        ),
        "selected_power_profile_absent": all(
            "selected_power_profile" not in power
            and "power_codebook_profile" not in power
            for power in powers.values()
        ),
        "forbidden_modes_disabled": all(
            not bool(value) for value in forbidden.values()
        ),
    }
    return {
        "matched_boundary_pass": all(bool(value) for value in checks.values()),
        "checks": checks,
        "forbidden_mode_flags": forbidden,
        "roles": list(ROLE_ORDER),
        "role_config_paths": {role: str(path) for role, path in role_configs.items()},
        "allowed_difference": "r1_reward_mode only",
        "seed_triplets": [list(row) for row in seed_triplets],
        "evaluation_seed_set": [int(seed) for seed in eval_seeds],
        "episode_budget": int(trainers[THROUGHPUT_CONTROL_ROLE].episodes),
        "power_surface_constants": {
            key: powers[THROUGHPUT_CONTROL_ROLE].get(key)
            for key in (
                "p_active_lo_w",
                "p_active_hi_w",
                "alpha",
                "beta",
                "kappa",
                "bias",
                "q_ref",
                "n_qos",
                "max_power_w",
                "total_power_budget_w",
            )
        },
        "role_constraints": constraints,
        "predeclared_tolerances": predeclared_tolerances(),
    }


def summarize_bounded_pilot_runs(
    rows: list[dict[str, Any]],
    boundary: dict[str, Any],
) -> dict[str, Any]:
    """Summarize aggregate and per-seed CP-base pilot metrics."""
    by_role: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_role.setdefault(row["role"], []).append(row)
    role_summaries = {
        role: _aggregate_role(by_role.get(role, [])) for role in ROLE_ORDER
    }
    aggregate_comparison = _candidate_vs_control(
        role_summaries[THROUGHPUT_CONTROL_ROLE],
        role_summaries[EE_CANDIDATE_ROLE],
        "aggregate",
    )
    per_seed_comparisons = _per_seed_comparisons(rows)
    acceptance = _acceptance(boundary, role_summaries, aggregate_comparison, per_seed_comparisons)
    stop_conditions = _stop_conditions(boundary, role_summaries, aggregate_comparison, acceptance)

    return {
        "phase": "hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot",
        "method_label": "CP-base-EE-MODQN bounded matched pilot",
        "status": acceptance["status"],
        "boundary_proof": boundary,
        "protocol": {
            "roles": list(ROLE_ORDER),
            "seed_triplets": boundary["seed_triplets"],
            "evaluation_seed_set": boundary["evaluation_seed_set"],
            "bounded_episode_budget": boundary["episode_budget"],
            "expected_required_runs": 6,
            "actual_required_runs": len(rows),
            "checkpoint_protocol": "final plus best-weighted-reward-on-eval",
            "scalar_reward_diagnostic_only": True,
        },
        "role_summaries": role_summaries,
        "role_table": [_role_table_row(role, role_summaries[role]) for role in ROLE_ORDER],
        "aggregate_comparison": aggregate_comparison,
        "per_seed_comparisons": per_seed_comparisons,
        "acceptance": acceptance,
        "stop_conditions_triggered": stop_conditions,
        "required_metrics": {
            role: _required_metric_view(role_summaries[role]) for role in ROLE_ORDER
        },
        "run_artifacts": [
            {
                "role": row["role"],
                "seed_triplet": row["seed_triplet"],
                "run_dir": row["run_dir"],
                "summary_path": row["summary_path"],
            }
            for row in rows
        ],
        "forbidden_claims_still_active": _forbidden_claims(),
        "deviations_or_blockers": acceptance["reasons"],
    }


def _aggregate_role(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"run_count": 0}
    diags = [row["summary"]["diagnostics"] for row in rows]
    distinct_total = _unique_sorted(
        [
            value
            for diag in diags
            for value in diag.get("distinct_total_active_power_w_values", [])
        ]
    )
    distinct_active = _unique_sorted(
        [
            value
            for diag in diags
            for value in diag.get("distinct_active_power_w_values", [])
        ]
    )
    aggregate: dict[str, Any] = {
        "run_count": len(rows),
        "seed_triplets": [row["seed_triplet"] for row in rows],
        "steps_evaluated": int(sum(int(d.get("steps_evaluated", 0)) for d in diags)),
        "denominator_varies_in_eval": all(
            bool(d.get("denominator_varies_in_eval", False)) for d in diags
        ),
        "all_evaluated_steps_one_active_beam": all(
            bool(d.get("all_evaluated_steps_one_active_beam", True)) for d in diags
        ),
        "per_seed_one_active_beam_count": int(
            sum(bool(d.get("all_evaluated_steps_one_active_beam", True)) for d in diags)
        ),
        "active_beam_count_distribution": _merge_distributions(
            d.get("active_beam_count_distribution", {}) for d in diags
        ),
        "total_active_power_distribution": _merge_distributions(
            d.get("total_active_power_distribution", {}) for d in diags
        ),
        "selected_power_profile_distribution": _merge_distributions(
            d.get("selected_power_profile_distribution", {}) for d in diags
        ),
        "selected_power_profile_absent": all(
            bool(d.get("selected_power_profile_absent", False)) for d in diags
        ),
        "active_power_single_point_distribution": len(distinct_active) <= 1,
        "distinct_total_active_power_w_values": distinct_total,
        "distinct_active_power_w_values": distinct_active,
        "same_policy_throughput_vs_ee_rescore_ranking_change": all(
            bool(d.get("same_policy_throughput_vs_ee_rescore_ranking_change", False))
            for d in diags
        ),
        "same_policy_throughput_vs_ee_rescore_ranking_change_any_seed": any(
            bool(d.get("same_policy_throughput_vs_ee_rescore_ranking_change", False))
            for d in diags
        ),
    }
    for key in MEAN_KEYS:
        values = [d.get(key) for d in diags if d.get(key) is not None]
        aggregate[key] = None if not values else float(np.mean(values))
    for key in SUM_KEYS:
        aggregate[key] = int(sum(int(d.get(key, 0)) for d in diags))
    aggregate["per_seed_metrics"] = [
        {
            "seed_triplet": row["seed_triplet"],
            **_required_metric_view(row["summary"]["diagnostics"]),
        }
        for row in rows
    ]
    return aggregate


def _candidate_vs_control(
    control: dict[str, Any],
    candidate: dict[str, Any],
    seed_label: str,
) -> dict[str, Any]:
    control_p05 = float(control.get("p05_throughput_bps") or 0.0)
    candidate_p05 = float(candidate.get("p05_throughput_bps") or 0.0)
    p05_ratio = (
        candidate_p05 / control_p05
        if control_p05 > 0.0
        else (1.0 if candidate_p05 == 0.0 else float("inf"))
    )
    return {
        "seed_label": seed_label,
        "candidate_vs_control_EE_system_delta": float(
            float(candidate.get("EE_system") or 0.0) - float(control.get("EE_system") or 0.0)
        ),
        "candidate_vs_control_EE_system_step_mean_delta": float(
            float(candidate.get("EE_system_step_mean") or 0.0)
            - float(control.get("EE_system_step_mean") or 0.0)
        ),
        "candidate_vs_control_throughput_delta_bps": float(
            float(candidate.get("raw_throughput_mean_bps") or 0.0)
            - float(control.get("raw_throughput_mean_bps") or 0.0)
        ),
        "p05_throughput_ratio_vs_control": float(p05_ratio),
        "p05_throughput_delta_bps": float(candidate_p05 - control_p05),
        "served_ratio_delta_vs_control": float(
            float(candidate.get("served_ratio") or 0.0)
            - float(control.get("served_ratio") or 0.0)
        ),
        "outage_ratio_delta_vs_control": float(
            float(candidate.get("outage_ratio") or 0.0)
            - float(control.get("outage_ratio") or 0.0)
        ),
        "handover_delta_vs_control": float(
            float(candidate.get("handover_count") or 0.0)
            - float(control.get("handover_count") or 0.0)
        ),
        "r2_delta_vs_control": float(
            float(candidate.get("r2_mean") or 0.0)
            - float(control.get("r2_mean") or 0.0)
        ),
        "load_balance_metric_delta_vs_control": float(
            float(candidate.get("load_balance_metric") or 0.0)
            - float(control.get("load_balance_metric") or 0.0)
        ),
        "scalar_reward_diagnostic_delta_vs_control": float(
            float(candidate.get("episode_scalar_reward_diagnostic_mean") or 0.0)
            - float(control.get("episode_scalar_reward_diagnostic_mean") or 0.0)
        ),
    }


def _per_seed_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed_role = {
        (int(row["seed_index"]), row["role"]): row["summary"]["diagnostics"]
        for row in rows
    }
    comparisons: list[dict[str, Any]] = []
    for seed_index in sorted({int(row["seed_index"]) for row in rows}):
        control = by_seed_role[(seed_index, THROUGHPUT_CONTROL_ROLE)]
        candidate = by_seed_role[(seed_index, EE_CANDIDATE_ROLE)]
        comparisons.append(
            {
                "seed_index": seed_index,
                "seed_triplet": _seed_triplet_for_row(rows, seed_index),
                **_candidate_vs_control(control, candidate, f"seed-{seed_index:02d}"),
            }
        )
    return comparisons


def _acceptance(
    boundary: dict[str, Any],
    role_summaries: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    per_seed_comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate = role_summaries[EE_CANDIDATE_ROLE]
    reasons: list[str] = []
    if not bool(boundary.get("matched_boundary_pass", False)):
        reasons.append("matched_boundary_pass=false")
    if bool(candidate.get("all_evaluated_steps_one_active_beam", True)):
        reasons.append("candidate_all_evaluated_steps_one_active_beam=true")
    if int(candidate.get("per_seed_one_active_beam_count") or 0) > 0:
        reasons.append("at least one candidate seed triplet remains one-active-beam")
    if not bool(candidate.get("denominator_varies_in_eval", False)):
        reasons.append("candidate_denominator_varies_in_eval=false")
    if bool(candidate.get("active_power_single_point_distribution", True)):
        reasons.append("candidate_active_power_single_point_distribution=true")
    if not bool(candidate.get("selected_power_profile_absent", False)):
        reasons.append("candidate_selected_power_profile_absent=false")
    if comparison["candidate_vs_control_EE_system_delta"] <= 0.0:
        reasons.append("candidate loses EE_system to throughput same-guard same-power control")
    if comparison["p05_throughput_ratio_vs_control"] < P05_THROUGHPUT_RATIO_MIN:
        reasons.append("candidate p05 throughput ratio below 0.95")
    if comparison["served_ratio_delta_vs_control"] < SERVED_RATIO_MIN_DELTA:
        reasons.append("candidate served ratio decreases")
    if comparison["outage_ratio_delta_vs_control"] > OUTAGE_RATIO_MAX_DELTA:
        reasons.append("candidate outage ratio increases")
    if comparison["handover_delta_vs_control"] > HANDOVER_DELTA_MAX:
        reasons.append("candidate handover delta exceeds +25")
    if comparison["r2_delta_vs_control"] < R2_MEAN_DELTA_MIN:
        reasons.append("candidate r2 delta below -0.05")
    for key in (
        "budget_violation_count",
        "per_beam_power_violation_count",
        "inactive_beam_nonzero_power_step_count",
    ):
        if int(candidate.get(key) or 0) != 0:
            reasons.append(f"candidate {key} != 0")

    positive_seed_count = sum(
        row["candidate_vs_control_EE_system_delta"] > 0.0
        for row in per_seed_comparisons
    )
    concentration_status = "PASS"
    if comparison["candidate_vs_control_EE_system_delta"] > 0.0 and positive_seed_count < 2:
        reasons.append("positive EE gain is concentrated in fewer than two seed triplets")
        concentration_status = "NEEDS MORE DESIGN"

    scalar_reward_success_basis = False
    if (
        comparison["scalar_reward_diagnostic_delta_vs_control"] > 0.0
        and comparison["candidate_vs_control_EE_system_delta"] <= 0.0
    ):
        scalar_reward_success_basis = True
        reasons.append("candidate wins only scalar reward")

    if any("concentrated" in reason for reason in reasons) and len(reasons) == 1:
        status = "NEEDS MORE DESIGN"
    elif concentration_status == "NEEDS MORE DESIGN" and all(
        reason == "positive EE gain is concentrated in fewer than two seed triplets"
        for reason in reasons
    ):
        status = "NEEDS MORE DESIGN"
    else:
        status = "PASS" if not reasons else "BLOCK"

    return {
        "status": status,
        "acceptance_pass": status == "PASS",
        "reasons": reasons,
        "positive_seed_triplet_count": int(positive_seed_count),
        "total_seed_triplet_count": len(per_seed_comparisons),
        "scalar_reward_success_basis": scalar_reward_success_basis,
        "criteria": {
            "matched_boundary_pass": bool(boundary.get("matched_boundary_pass", False)),
            "candidate_all_evaluated_steps_one_active_beam": bool(
                candidate.get("all_evaluated_steps_one_active_beam", True)
            ),
            "candidate_denominator_varies_in_eval": bool(
                candidate.get("denominator_varies_in_eval", False)
            ),
            "candidate_active_power_single_point_distribution": bool(
                candidate.get("active_power_single_point_distribution", True)
            ),
            "candidate_selected_power_profile_absent": bool(
                candidate.get("selected_power_profile_absent", False)
            ),
            "candidate_vs_throughput_same_guard_same_power_control_EE_system_delta": (
                comparison["candidate_vs_control_EE_system_delta"]
            ),
            "candidate_vs_throughput_same_guard_same_power_control_p05_ratio": (
                comparison["p05_throughput_ratio_vs_control"]
            ),
            "candidate_vs_throughput_same_guard_same_power_control_served_ratio_delta": (
                comparison["served_ratio_delta_vs_control"]
            ),
            "candidate_vs_throughput_same_guard_same_power_control_outage_ratio_delta": (
                comparison["outage_ratio_delta_vs_control"]
            ),
            "candidate_vs_throughput_same_guard_same_power_control_handover_delta": (
                comparison["handover_delta_vs_control"]
            ),
            "candidate_vs_throughput_same_guard_same_power_control_r2_delta": (
                comparison["r2_delta_vs_control"]
            ),
            "budget_per_beam_inactive_power_violations": [
                int(candidate.get("budget_violation_count") or 0),
                int(candidate.get("per_beam_power_violation_count") or 0),
                int(candidate.get("inactive_beam_nonzero_power_step_count") or 0),
            ],
            "scalar_reward_success_basis": scalar_reward_success_basis,
            "positive_EE_not_concentrated_in_single_seed_triplet": (
                positive_seed_count >= 2
            ),
        },
    }


def _stop_conditions(
    boundary: dict[str, Any],
    role_summaries: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    acceptance: dict[str, Any],
) -> list[str]:
    candidate = role_summaries.get(EE_CANDIDATE_ROLE, {})
    triggered: list[str] = []
    checks = boundary.get("checks", {})
    if not bool(boundary.get("matched_boundary_pass", False)):
        triggered.append("matched boundary proof failed")
    if not bool(checks.get("only_intended_difference_is_r1_reward_mode", False)):
        triggered.append("candidate/control differ by more than r1_reward_mode")
    if not bool(checks.get("throughput_control_r1", False)):
        triggered.append("throughput + same guard + same continuous power control is missing")
    if not bool(checks.get("same_continuous_power_surface", False)):
        triggered.append("continuous power is not shared between roles")
    if not bool(checks.get("selected_power_profile_absent", False)):
        triggered.append("power profile selector is present")
    if bool(candidate.get("all_evaluated_steps_one_active_beam", True)):
        triggered.append("candidate still has all evaluated steps with one active beam")
    if not bool(candidate.get("denominator_varies_in_eval", False)):
        triggered.append("candidate denominator does not vary in eval")
    if comparison.get("candidate_vs_control_EE_system_delta", 0.0) <= 0.0:
        triggered.append("candidate loses EE_system to throughput same-guard same-power control")
    if bool(acceptance.get("scalar_reward_success_basis", False)):
        triggered.append("candidate wins only scalar reward")
    if any(
        phrase in reason
        for reason in acceptance.get("reasons", [])
        for phrase in ("p05", "served", "outage", "handover", "r2", "violation")
    ):
        triggered.append("candidate violates protected guardrails")
    if not bool(checks.get("forbidden_modes_disabled", False)):
        triggered.append("a forbidden mode is enabled")
    return triggered or ["None"]


def _required_metric_view(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "EE_system",
        "EE_system_step_mean",
        "raw_throughput_mean_bps",
        "p05_throughput_bps",
        "served_ratio",
        "outage_ratio",
        "handover_count",
        "r2_mean",
        "load_balance_metric",
        "r3_mean",
        "episode_scalar_reward_diagnostic_mean",
        "all_evaluated_steps_one_active_beam",
        "per_seed_one_active_beam_count",
        "active_beam_count_distribution",
        "denominator_varies_in_eval",
        "active_power_single_point_distribution",
        "distinct_total_active_power_w_values",
        "distinct_active_power_w_values",
        "selected_power_profile_distribution",
        "selected_power_profile_absent",
        "power_control_activity_rate",
        "continuous_power_activity_rate",
        "budget_violation_count",
        "per_beam_power_violation_count",
        "inactive_beam_nonzero_power_step_count",
        "throughput_vs_ee_pearson",
        "throughput_vs_ee_spearman",
        "same_policy_throughput_vs_ee_rescore_ranking_change",
    )
    return {key: summary.get(key) for key in keys}


def _role_table_row(role: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": role,
        "r1": "throughput" if role == THROUGHPUT_CONTROL_ROLE else "hobs-active-tx-ee",
        "anti_collapse": True,
        "continuous_power_surface": "same",
        **_required_metric_view(summary),
    }


def _write_all_summary_artifacts(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    PAIRED_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = write_json(SUMMARY_DIR / "summary.json", summary)
    role_table_path = _write_csv(SUMMARY_DIR / "role_table.csv", summary.get("role_table", []))
    per_seed_path = _write_csv(
        SUMMARY_DIR / "per_seed_comparisons.csv",
        summary.get("per_seed_comparisons", []),
    )
    paired_summary_path = write_json(PAIRED_COMPARISON_DIR / "summary.json", summary)
    report_path = _write_execution_report(REPORT_PATH, summary)
    summary["artifact_paths"] = {
        "summary_json": str(summary_path),
        "role_table_csv": str(role_table_path),
        "per_seed_comparisons_csv": str(per_seed_path),
        "paired_comparison_summary_json": str(paired_summary_path),
        "execution_report_md": str(report_path),
    }
    if rows:
        summary["run_artifacts"] = [
            {
                "role": row["role"],
                "seed_triplet": row["seed_triplet"],
                "run_dir": row["run_dir"],
                "summary_path": row["summary_path"],
            }
            for row in rows
        ]
    write_json(SUMMARY_DIR / "summary.json", summary)
    write_json(PAIRED_COMPARISON_DIR / "summary.json", summary)


def _write_execution_report(path: Path, summary: dict[str, Any]) -> Path:
    boundary = summary.get("boundary_proof", {})
    acceptance = summary.get("acceptance", {})
    role_table = summary.get("role_table", [])
    role_summaries = summary.get("role_summaries", {})
    comparison = summary.get("aggregate_comparison", {})
    lines = [
        "# HOBS Active-TX EE Non-Codebook Continuous-Power Bounded Pilot Execution Report",
        "",
        "**Date:** `2026-05-02`",
        f"**Status:** `{summary.get('status')}`",
        "**Method label:** `CP-base-EE-MODQN bounded matched pilot`",
        "**Scope:** bounded matched pilot only; not Catfish-EE, not physical energy saving, not HOBS optimizer reproduction, not full RA-EE-MODQN, and not a general EE-MODQN effectiveness claim.",
        "",
        "## Changed Files",
        "",
        "```text",
        "configs/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot-throughput-control.resolved.yaml",
        "configs/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot-ee-candidate.resolved.yaml",
        "scripts/run_hobs_active_tx_ee_non_codebook_continuous_power_bounded_pilot.py",
        "src/modqn_paper_reproduction/analysis/hobs_active_tx_ee_non_codebook_continuous_power_bounded_pilot.py",
        "src/modqn_paper_reproduction/config_loader.py",
        "src/modqn_paper_reproduction/runtime/trainer_spec.py",
        "src/modqn_paper_reproduction/analysis/hobs_active_tx_ee_anti_collapse.py",
        "tests/test_hobs_active_tx_ee_non_codebook_continuous_power_bounded_pilot.py",
        "docs/research/catfish-ee-modqn/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot.execution-report.md",
        "```",
        "",
        "## What Was Implemented",
        "",
        "- Added the bounded-pilot config namespace and two matched resolved configs.",
        "- Added a bounded-pilot runner that executes exactly the authorized two roles and three seed triplets.",
        "- Added matched-boundary proof, paired comparison, summary CSV/JSON, and execution-report emission.",
        "- Extended the CP-base continuous-power diagnostics with profile absence, continuous-power activity, and Spearman correlation fields.",
        "- Added focused tests for config loading, matched metadata, acceptance, and scalar-only BLOCK enforcement.",
        "",
        "## Protocol / Roles",
        "",
        "- Roles: `throughput-control`, `ee-candidate`.",
        "- Seed triplets: `[42, 1337, 7]`, `[43, 1338, 8]`, `[44, 1339, 9]`.",
        "- Eval seeds: `[100, 200, 300, 400, 500]`.",
        "- Episode budget: `5` per role / seed triplet.",
        "- Scalar reward is diagnostic only.",
        "",
        "## Matched Boundary Proof",
        "",
        f"`matched_boundary_pass={boundary.get('matched_boundary_pass')}`",
        "",
        "```text",
    ]
    for key, value in boundary.get("checks", {}).items():
        lines.append(f"{key} = {value}")
    lines.extend(["```", "", "## Metrics", ""])
    if role_table:
        lines.extend(
            [
                "| Role | EE_system | EE step mean | throughput | p05 | served | outage | handovers | r2 | r3 | one-beam | denom varies | selected profile absent |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
            ]
        )
        for row in role_table:
            lines.append(
                f"| `{row['role']}` | `{row.get('EE_system')}` | `{row.get('EE_system_step_mean')}` | "
                f"`{row.get('raw_throughput_mean_bps')}` | `{row.get('p05_throughput_bps')}` | "
                f"`{row.get('served_ratio')}` | `{row.get('outage_ratio')}` | "
                f"`{row.get('handover_count')}` | `{row.get('r2_mean')}` | "
                f"`{row.get('load_balance_metric')}` | `{row.get('all_evaluated_steps_one_active_beam')}` | "
                f"`{row.get('denominator_varies_in_eval')}` | `{row.get('selected_power_profile_absent')}` |"
            )
    if comparison:
        lines.extend(
            [
                "",
                "Aggregate candidate vs throughput-control:",
                "",
                "```text",
                f"candidate_vs_control_EE_system_delta = {comparison.get('candidate_vs_control_EE_system_delta')}",
                f"p05_throughput_ratio_vs_control = {comparison.get('p05_throughput_ratio_vs_control')}",
                f"served_ratio_delta_vs_control = {comparison.get('served_ratio_delta_vs_control')}",
                f"outage_ratio_delta_vs_control = {comparison.get('outage_ratio_delta_vs_control')}",
                f"handover_delta_vs_control = {comparison.get('handover_delta_vs_control')}",
                f"r2_delta_vs_control = {comparison.get('r2_delta_vs_control')}",
                f"scalar_reward_diagnostic_delta_vs_control = {comparison.get('scalar_reward_diagnostic_delta_vs_control')}",
                "```",
            ]
        )
    if summary.get("per_seed_comparisons"):
        lines.extend(["", "Per-seed EE deltas:", "", "```text"])
        for row in summary["per_seed_comparisons"]:
            lines.append(
                f"{row['seed_triplet']}: EE_delta={row['candidate_vs_control_EE_system_delta']}, "
                f"p05_ratio={row['p05_throughput_ratio_vs_control']}, "
                f"handover_delta={row['handover_delta_vs_control']}, "
                f"r2_delta={row['r2_delta_vs_control']}"
            )
        lines.append("```")
    if role_summaries:
        lines.extend(["", "Per-seed role metrics:", ""])
        lines.extend(
            [
                "| Role | Seed triplet | EE_system | throughput | p05 | served | outage | handovers | r2 | r3 | scalar diagnostic | denom varies | profile absent | power violations |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
            ]
        )
        for role in ("throughput-control", "ee-candidate"):
            for seed_row in role_summaries.get(role, {}).get("per_seed_metrics", []):
                violations = (
                    seed_row.get("budget_violation_count"),
                    seed_row.get("per_beam_power_violation_count"),
                    seed_row.get("inactive_beam_nonzero_power_step_count"),
                )
                lines.append(
                    f"| `{role}` | `{seed_row.get('seed_triplet')}` | `{seed_row.get('EE_system')}` | "
                    f"`{seed_row.get('raw_throughput_mean_bps')}` | `{seed_row.get('p05_throughput_bps')}` | "
                    f"`{seed_row.get('served_ratio')}` | `{seed_row.get('outage_ratio')}` | "
                    f"`{seed_row.get('handover_count')}` | `{seed_row.get('r2_mean')}` | "
                    f"`{seed_row.get('load_balance_metric')}` | `{seed_row.get('episode_scalar_reward_diagnostic_mean')}` | "
                    f"`{seed_row.get('denominator_varies_in_eval')}` | `{seed_row.get('selected_power_profile_absent')}` | "
                    f"`{violations}` |"
                )
        lines.extend(["", "Power and correlation diagnostics:", "", "```text"])
        for role in ("throughput-control", "ee-candidate"):
            role_summary = role_summaries.get(role, {})
            total_power_values = role_summary.get("distinct_total_active_power_w_values", [])
            active_power_values = role_summary.get("distinct_active_power_w_values", [])
            lines.extend(
                [
                    f"{role}.active_beam_count_distribution = {role_summary.get('active_beam_count_distribution')}",
                    f"{role}.selected_power_profile_distribution = {role_summary.get('selected_power_profile_distribution')}",
                    f"{role}.distinct_total_active_power_w_value_count = {len(total_power_values)}",
                    f"{role}.distinct_active_power_w_value_count = {len(active_power_values)}",
                    f"{role}.power_control_activity_rate = {role_summary.get('power_control_activity_rate')}",
                    f"{role}.continuous_power_activity_rate = {role_summary.get('continuous_power_activity_rate')}",
                    f"{role}.budget_violation_count = {role_summary.get('budget_violation_count')}",
                    f"{role}.per_beam_power_violation_count = {role_summary.get('per_beam_power_violation_count')}",
                    f"{role}.inactive_beam_nonzero_power_step_count = {role_summary.get('inactive_beam_nonzero_power_step_count')}",
                    f"{role}.throughput_vs_ee_pearson = {role_summary.get('throughput_vs_ee_pearson')}",
                    f"{role}.throughput_vs_ee_spearman = {role_summary.get('throughput_vs_ee_spearman')}",
                    f"{role}.same_policy_throughput_vs_ee_rescore_ranking_change = {role_summary.get('same_policy_throughput_vs_ee_rescore_ranking_change')}",
                    "",
                ]
            )
        lines.extend(
            [
                "Full active-power value lists and per-seed distributions are in artifacts/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot-summary/summary.json.",
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "## Acceptance Result",
            "",
            f"`PASS / BLOCK / NEEDS MORE DESIGN: {summary.get('status')}`",
            "",
            "```text",
        ]
    )
    for key, value in acceptance.get("criteria", {}).items():
        lines.append(f"{key} = {value}")
    lines.extend(["```", "", "## Stop Conditions Triggered", ""])
    lines.extend(f"- {item}" for item in summary.get("stop_conditions_triggered", []))
    lines.extend(["", "## Tests / Checks Run", ""])
    lines.extend(
        [
            "```text",
            ".venv/bin/python -m pytest tests/test_hobs_active_tx_ee_non_codebook_continuous_power_bounded_pilot.py tests/test_hobs_active_tx_ee_non_codebook_continuous_power.py -q",
            ".venv/bin/python scripts/run_hobs_active_tx_ee_non_codebook_continuous_power_bounded_pilot.py",
            "git diff --check",
            "```",
        ]
    )
    lines.extend(["", "## Artifacts", ""])
    for artifact in [
        "artifacts/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot-throughput-control/",
        "artifacts/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot-ee-candidate/",
        "artifacts/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot-ee-candidate/paired-comparison-vs-throughput-control/",
        "artifacts/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot-summary/",
    ]:
        lines.append(f"- `{artifact}`")
    lines.extend(["", "## Forbidden Claims Still Active", ""])
    lines.extend(f"- {claim}" for claim in _forbidden_claims())
    lines.extend(["", "## Deviations / Blockers", ""])
    reasons = acceptance.get("reasons", [])
    lines.extend(f"- {reason}" for reason in reasons or ["None within the bounded protocol."])
    lines.extend(["", "## PASS / BLOCK / NEEDS MORE DESIGN", "", f"`{summary.get('status')}`"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _materialize_seed_config(
    config_path: Path,
    *,
    role: str,
    seed_index: int,
    seed_triplet: list[int] | tuple[int, int, int],
    output_dir: Path,
) -> Path:
    cfg = copy.deepcopy(load_training_yaml(config_path))
    cfg.pop("inherits_from", None)
    train_seed, env_seed, mobility_seed = (int(value) for value in seed_triplet)
    seed_value = cfg.setdefault("resolved_assumptions", {}).setdefault(
        "seed_and_rng_policy",
        {"assumption_id": "ASSUME-MODQN-REP-018", "value": {}},
    ).setdefault("value", {})
    seed_value["train_seed"] = train_seed
    seed_value["environment_seed"] = env_seed
    seed_value["mobility_seed"] = mobility_seed
    seed_value["evaluation_seed_set"] = [100, 200, 300, 400, 500]
    track = cfg.setdefault("track", {})
    track["bounded_pilot_role"] = role
    track["seed_index"] = seed_index
    track["materialized_for_artifact_dir"] = str(output_dir)
    experiment = cfg.setdefault("training_experiment", {})
    experiment["experiment_id"] = (
        f"{experiment.get('experiment_id', 'CP-BASE-BOUNDED-PILOT')}"
        f"-SEED-{seed_index:02d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.resolved.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _role_seed_dir(
    role: str,
    seed_index: int,
    seed_triplet: list[int] | tuple[int, int, int],
) -> Path:
    train, env, mobility = (int(value) for value in seed_triplet)
    return ARTIFACT_ROOTS[role] / f"seed-{seed_index:02d}-t{train}-e{env}-m{mobility}"


def _protocol_seed_triplets(cfg: dict[str, Any]) -> tuple[tuple[int, int, int], ...]:
    block = cfg.get("bounded_pilot_protocol", {})
    rows = block.get("seed_triplets", []) if isinstance(block, dict) else []
    return tuple(tuple(int(value) for value in row) for row in rows)


def _same_training_hyperparameters(trainers: dict[str, Any]) -> bool:
    keys = (
        "hidden_layers",
        "activation",
        "learning_rate",
        "discount_factor",
        "batch_size",
        "epsilon_start",
        "epsilon_end",
        "epsilon_decay_episodes",
        "target_update_every_episodes",
        "replay_capacity",
        "policy_sharing_mode",
        "snr_encoding",
        "offset_scale_km",
        "load_normalization",
    )
    return all(len({getattr(cfg, key) for cfg in trainers.values()}) == 1 for key in keys)


def _same_checkpoint_protocol(trainers: dict[str, Any]) -> bool:
    return (
        len({cfg.checkpoint_assumption_id for cfg in trainers.values()}) == 1
        and len({cfg.checkpoint_primary_report for cfg in trainers.values()}) == 1
        and len({cfg.checkpoint_secondary_report for cfg in trainers.values()}) == 1
    )


def _same_environment_boundary(envs: dict[str, Any]) -> bool:
    first = envs[THROUGHPUT_CONTROL_ROLE]
    return all(
        env.config == first.config
        and env.orbit.config == first.orbit.config
        and env.beam_pattern.config == first.beam_pattern.config
        and env.channel_config == first.channel_config
        for env in envs.values()
    )


def _behavioral_boundary_subset(
    cfg: dict[str, Any],
    trainer: Any,
    power: dict[str, Any],
    constraint: dict[str, Any],
) -> dict[str, Any]:
    seeds = get_seeds(cfg)
    return {
        "training_experiment_kind": trainer.training_experiment_kind,
        "phase": trainer.phase,
        "episodes": trainer.episodes,
        "objective_weights": list(trainer.objective_weights),
        "trainer_hyperparameters": {
            "hidden_layers": list(trainer.hidden_layers),
            "activation": trainer.activation,
            "learning_rate": trainer.learning_rate,
            "discount_factor": trainer.discount_factor,
            "batch_size": trainer.batch_size,
            "epsilon_start": trainer.epsilon_start,
            "epsilon_end": trainer.epsilon_end,
            "epsilon_decay_episodes": trainer.epsilon_decay_episodes,
            "target_update_every_episodes": trainer.target_update_every_episodes,
            "replay_capacity": trainer.replay_capacity,
            "policy_sharing_mode": trainer.policy_sharing_mode,
            "snr_encoding": trainer.snr_encoding,
            "offset_scale_km": trainer.offset_scale_km,
            "load_normalization": trainer.load_normalization,
        },
        "checkpoint": {
            "primary": trainer.checkpoint_primary_report,
            "secondary": trainer.checkpoint_secondary_report,
        },
        "seeds": {
            "train_seed": int(seeds["train_seed"]),
            "environment_seed": int(seeds["environment_seed"]),
            "mobility_seed": int(seeds["mobility_seed"]),
            "evaluation_seed_set": [int(seed) for seed in seeds["evaluation_seed_set"]],
        },
        "power_surface": power,
        "anti_collapse": constraint,
        "reward_calibration_enabled": trainer.reward_calibration_enabled,
        "catfish_enabled": trainer.catfish_enabled,
    }


def _forbidden_mode_flags(
    loaded: dict[str, dict[str, Any]],
    trainers: dict[str, Any],
    powers: dict[str, dict[str, Any]],
) -> dict[str, bool]:
    phase_or_kind = " ".join(
        [
            str(trainer.training_experiment_kind)
            + " "
            + str(trainer.phase)
            + " "
            + str(loaded[role].get("track", {}).get("label", ""))
            for role, trainer in trainers.items()
        ]
    ).lower()
    return {
        "catfish_enabled": any(trainer.catfish_enabled for trainer in trainers.values()),
        "multi_catfish_enabled": "multi-catfish" in phase_or_kind,
        "phase_03c_enabled": "phase-03c" in phase_or_kind,
        "ra_ee_learned_association_enabled": "ra-ee" in phase_or_kind
        or "learned-association" in phase_or_kind,
        "oracle_enabled": "oracle" in phase_or_kind,
        "future_information_enabled": "future-information" in phase_or_kind,
        "offline_replay_oracle_enabled": "offline-replay-oracle" in phase_or_kind,
        "hobs_optimizer_enabled": "hobs-optimizer" in phase_or_kind,
        "finite_codebook_enabled": any(
            "power_codebook_levels_w" in power or "finite_codebook_levels_w" in power
            for power in powers.values()
        ),
        "runtime_profile_selector_enabled": any(
            power.get("power_codebook_profile") == "runtime-ee-selector"
            or "selected_power_profile" in power
            for power in powers.values()
        ),
    }


def _merge_distributions(distributions: Any) -> dict[str, int]:
    merged: dict[str, int] = {}
    for dist in distributions:
        for key, value in dict(dist).items():
            merged[str(key)] = merged.get(str(key), 0) + int(value)
    return dict(sorted(merged.items()))


def _seed_triplet_for_row(rows: list[dict[str, Any]], seed_index: int) -> list[int]:
    for row in rows:
        if int(row["seed_index"]) == seed_index:
            return list(row["seed_triplet"])
    return []


def _blocked_boundary_summary(boundary: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": "hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot",
        "method_label": "CP-base-EE-MODQN bounded matched pilot",
        "status": "BLOCK",
        "boundary_proof": boundary,
        "protocol": {
            "roles": list(ROLE_ORDER),
            "expected_required_runs": 0,
            "actual_required_runs": 0,
            "scalar_reward_diagnostic_only": True,
        },
        "role_summaries": {},
        "role_table": [],
        "aggregate_comparison": {},
        "per_seed_comparisons": [],
        "acceptance": {
            "status": "BLOCK",
            "acceptance_pass": False,
            "reasons": ["matched boundary proof failed before metrics interpretation"],
            "criteria": {"matched_boundary_pass": False},
            "scalar_reward_success_basis": False,
        },
        "stop_conditions_triggered": ["matched boundary proof failed"],
        "required_metrics": {},
        "run_artifacts": [],
        "forbidden_claims_still_active": _forbidden_claims(),
        "deviations_or_blockers": ["matched boundary proof failed"],
    }


def _forbidden_claims() -> list[str]:
    return [
        "general EE-MODQN effectiveness",
        "Catfish-EE readiness",
        "Catfish / Multi-Catfish effectiveness",
        "physical energy saving",
        "HOBS optimizer reproduction",
        "full RA-EE-MODQN",
        "learned association effectiveness",
        "RB / bandwidth allocation effectiveness",
        "Phase 03D failure is overturned",
        "Phase 03C selector route is reopened",
        "scalar reward success",
        "QoS-sticky anti-collapse as EE objective contribution",
        "denominator variability alone proves energy-aware learning",
        "same-throughput-less-physical-power",
    ]


__all__ = [
    "ARTIFACT_ROOTS",
    "EE_CANDIDATE_ROLE",
    "PAIRED_COMPARISON_DIR",
    "REPORT_PATH",
    "ROLE_CONFIGS",
    "ROLE_ORDER",
    "SUMMARY_DIR",
    "THROUGHPUT_CONTROL_ROLE",
    "prove_bounded_pilot_boundary",
    "run_bounded_pilot",
    "summarize_bounded_pilot_runs",
]
