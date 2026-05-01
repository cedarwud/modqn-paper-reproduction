"""Bounded QoS-sticky HOBS active-TX EE broader-effectiveness gate.

This gate compares the primary QoS-sticky EE candidate against throughput and
EE controls under the same DPC sidecar. It is scoped to active-TX EE evidence
only and does not reopen the frozen baseline, Catfish, Phase 03C, or RA-EE.
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any

import yaml

from ..config_loader import (
    build_environment,
    build_trainer_config,
    get_seeds,
    load_training_yaml,
)
from ..env.step import HOBS_POWER_SURFACE_DPC_SIDECAR
from ..runtime.trainer_spec import (
    HOBS_ACTIVE_TX_EE_QOS_STICKY_BROADER_EFFECTIVENESS_KIND,
    R1_REWARD_MODE_HOBS_ACTIVE_TX_EE,
    R1_REWARD_MODE_THROUGHPUT,
)
from ._common import write_json
from .hobs_active_tx_ee_anti_collapse import (
    HANDOVER_DELTA_MAX,
    MAX_TINY_EPISODES,
    P05_THROUGHPUT_RATIO_MIN,
    R2_MEAN_DELTA_MIN,
    _anti_collapse_subset,
    _config_power_surface_value,
    _dpc_parameter_subset,
    _train_and_evaluate_arm,
    predeclared_tolerances,
)
from .hobs_active_tx_ee_qos_sticky_robustness import (
    _aggregate_role,
    _same_checkpoint_rule,
    _same_training_hyperparameters,
)

MATCHED_THROUGHPUT_CONTROL_ROLE = "matched-throughput-control"
HOBS_EE_NO_ANTI_COLLAPSE_ROLE = "hobs-ee-control-no-anti-collapse"
QOS_STICKY_EE_CANDIDATE_ROLE = "qos-sticky-ee-candidate"
ANTI_COLLAPSE_THROUGHPUT_CONTROL_ROLE = "anti-collapse-throughput-control"

ROLE_CONFIGS: dict[str, Path] = {
    MATCHED_THROUGHPUT_CONTROL_ROLE: Path(
        "configs/hobs-active-tx-ee-qos-sticky-broader-effectiveness-"
        "matched-throughput-control.resolved.yaml"
    ),
    HOBS_EE_NO_ANTI_COLLAPSE_ROLE: Path(
        "configs/hobs-active-tx-ee-qos-sticky-broader-effectiveness-"
        "hobs-ee-control-no-anti-collapse.resolved.yaml"
    ),
    QOS_STICKY_EE_CANDIDATE_ROLE: Path(
        "configs/hobs-active-tx-ee-qos-sticky-broader-effectiveness-"
        "qos-sticky-ee-candidate.resolved.yaml"
    ),
    ANTI_COLLAPSE_THROUGHPUT_CONTROL_ROLE: Path(
        "configs/hobs-active-tx-ee-qos-sticky-broader-effectiveness-"
        "anti-collapse-throughput-control.resolved.yaml"
    ),
}
ROLE_ORDER = tuple(ROLE_CONFIGS.keys())
CONTROL_ROLES = (
    MATCHED_THROUGHPUT_CONTROL_ROLE,
    HOBS_EE_NO_ANTI_COLLAPSE_ROLE,
    ANTI_COLLAPSE_THROUGHPUT_CONTROL_ROLE,
)
ANTI_COLLAPSE_ROLES = (
    QOS_STICKY_EE_CANDIDATE_ROLE,
    ANTI_COLLAPSE_THROUGHPUT_CONTROL_ROLE,
)
ARTIFACT_ROOT = Path(
    "artifacts/hobs-active-tx-ee-qos-sticky-broader-effectiveness-summary"
)
REPORT_PATH = Path(
    "docs/research/catfish-ee-modqn/"
    "hobs-active-tx-ee-qos-sticky-broader-effectiveness-gate.execution-report.md"
)
EE_OBJECTIVE_EPSILON = 1e-9


def run_qos_sticky_broader_effectiveness_gate() -> dict[str, Any]:
    """Run the bounded broader-effectiveness roles and write artifacts."""
    boundary = prove_broader_effectiveness_boundary()
    if not bool(boundary["matched_boundary_pass"]):
        summary = {
            "phase": "hobs-active-tx-ee-qos-sticky-broader-effectiveness",
            "status": "NEEDS MORE DESIGN",
            "boundary_proof": boundary,
            "stop_conditions_triggered": [
                "matched boundary cannot isolate EE objective from anti-collapse"
            ],
        }
        write_json(ARTIFACT_ROOT / "summary.json", summary)
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
            arm_summary = _train_and_evaluate_arm(materialized_config, run_dir)
            rows.append({
                "role": role,
                "seed_index": seed_index,
                "seed_triplet": list(seed_triplet),
                "run_dir": str(run_dir),
                "config_path": str(materialized_config),
                "summary_path": arm_summary["artifact_paths"]["summary_json"],
                "summary": arm_summary,
            })

    summary = summarize_broader_effectiveness_runs(rows, boundary)
    summary_path = write_json(ARTIFACT_ROOT / "summary.json", summary)
    role_csv = _write_role_table(
        ARTIFACT_ROOT / "role_table.csv",
        summary["role_table"],
    )
    comparison_csv = _write_comparison_table(
        ARTIFACT_ROOT / "control_comparisons.csv",
        summary["control_comparisons"],
    )
    report_path = _write_execution_report(REPORT_PATH, summary)
    summary["artifact_paths"] = {
        "summary_json": str(summary_path),
        "role_table_csv": str(role_csv),
        "control_comparisons_csv": str(comparison_csv),
        "execution_report_md": str(report_path),
    }
    write_json(ARTIFACT_ROOT / "summary.json", summary)
    return summary


def prove_broader_effectiveness_boundary(
    role_configs: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Prove that the four-role gate matches except r1 and anti-collapse toggles."""
    role_configs = dict(role_configs or ROLE_CONFIGS)
    loaded = {role: load_training_yaml(path) for role, path in role_configs.items()}
    trainers = {role: build_trainer_config(cfg) for role, cfg in loaded.items()}
    envs = {role: build_environment(cfg) for role, cfg in loaded.items()}
    powers = {
        role: _dpc_parameter_subset(_config_power_surface_value(cfg))
        for role, cfg in loaded.items()
    }
    constraints = {role: _anti_collapse_subset(trainers[role]) for role in trainers}
    seed_triplets = _protocol_seed_triplets(loaded[QOS_STICKY_EE_CANDIDATE_ROLE])
    eval_seeds = get_seeds(
        loaded[QOS_STICKY_EE_CANDIDATE_ROLE]
    )["evaluation_seed_set"]

    checks = {
        "required_roles_present": set(role_configs) == set(ROLE_ORDER),
        "all_training_experiment_kind_match": all(
            cfg.training_experiment_kind
            == HOBS_ACTIVE_TX_EE_QOS_STICKY_BROADER_EFFECTIVENESS_KIND
            for cfg in trainers.values()
        ),
        "all_phase_match": len({cfg.phase for cfg in trainers.values()}) == 1,
        "same_episode_budget": len({cfg.episodes for cfg in trainers.values()}) == 1,
        "tiny_episode_budget": all(
            int(cfg.episodes) == MAX_TINY_EPISODES for cfg in trainers.values()
        ),
        "same_objective_weights": (
            len({cfg.objective_weights for cfg in trainers.values()}) == 1
        ),
        "same_training_hyperparameters": _same_training_hyperparameters(trainers),
        "same_checkpoint_rule": _same_checkpoint_rule(trainers),
        "same_eval_seed_set": all(
            get_seeds(cfg)["evaluation_seed_set"] == eval_seeds
            for cfg in loaded.values()
        ),
        "same_seed_triplets_declared": all(
            _protocol_seed_triplets(cfg) == seed_triplets for cfg in loaded.values()
        ),
        "at_least_three_seed_triplets": len(seed_triplets) >= 3,
        "same_dpc_sidecar": len({tuple(sorted(p.items())) for p in powers.values()}) == 1,
        "dpc_sidecar_enabled": all(
            env.power_surface_config.hobs_power_surface_mode
            == HOBS_POWER_SURFACE_DPC_SIDECAR
            for env in envs.values()
        ),
        "same_environment_boundary": _same_environment_boundary(envs),
        "matched_throughput_control_r1": (
            trainers[MATCHED_THROUGHPUT_CONTROL_ROLE].r1_reward_mode
            == R1_REWARD_MODE_THROUGHPUT
        ),
        "hobs_ee_control_r1": (
            trainers[HOBS_EE_NO_ANTI_COLLAPSE_ROLE].r1_reward_mode
            == R1_REWARD_MODE_HOBS_ACTIVE_TX_EE
        ),
        "qos_sticky_candidate_r1": (
            trainers[QOS_STICKY_EE_CANDIDATE_ROLE].r1_reward_mode
            == R1_REWARD_MODE_HOBS_ACTIVE_TX_EE
        ),
        "anti_collapse_throughput_control_r1": (
            trainers[ANTI_COLLAPSE_THROUGHPUT_CONTROL_ROLE].r1_reward_mode
            == R1_REWARD_MODE_THROUGHPUT
        ),
        "no_anti_collapse_controls_disabled": all(
            not trainers[role].anti_collapse_action_constraint_enabled
            for role in (
                MATCHED_THROUGHPUT_CONTROL_ROLE,
                HOBS_EE_NO_ANTI_COLLAPSE_ROLE,
            )
        ),
        "anti_collapse_roles_enabled": all(
            trainers[role].anti_collapse_action_constraint_enabled
            for role in ANTI_COLLAPSE_ROLES
        ),
        "enabled_roles_are_qos_sticky": all(
            constraints[role]["mode"] == "qos-sticky-overflow-reassignment"
            for role in ANTI_COLLAPSE_ROLES
        ),
        "same_constraint_parameters": len(
            {tuple(sorted(value.items())) for value in constraints.values()}
        ) == 1,
        "no_forced_min_active_beams_target": all(
            int(value["min_active_beams_target"]) == 0 for value in constraints.values()
        ),
        "all_roles_disable_nonsticky_moves": all(
            not bool(value["allow_nonsticky_moves"]) for value in constraints.values()
        ),
        "all_nonsticky_budgets_zero": all(
            int(value["nonsticky_move_budget"]) == 0 for value in constraints.values()
        ),
        "method_label_is_variant_not_frozen_baseline": all(
            cfg.method_family == "QoS-sticky HOBS-active-TX EE-MODQN"
            for cfg in trainers.values()
        ),
        "throughput_controls_are_dpc_matched_not_frozen_baseline": all(
            trainers[role].method_family != "MODQN-baseline"
            and trainers[role].comparison_role == role
            for role in (
                MATCHED_THROUGHPUT_CONTROL_ROLE,
                ANTI_COLLAPSE_THROUGHPUT_CONTROL_ROLE,
            )
        ),
        "no_forbidden_runtime_routes_enabled": all(
            not getattr(cfg, "catfish_enabled", False) for cfg in trainers.values()
        ),
    }
    return {
        "matched_boundary_pass": all(bool(value) for value in checks.values()),
        "checks": checks,
        "roles": list(ROLE_ORDER),
        "role_config_paths": {role: str(path) for role, path in role_configs.items()},
        "allowed_differences": [
            "r1_reward_mode is throughput for the two throughput controls",
            "r1_reward_mode is hobs-active-tx-ee for EE control and candidate",
            "qos-sticky-overflow-reassignment is enabled only for the candidate and anti-collapse throughput control",
        ],
        "seed_triplets": [list(row) for row in seed_triplets],
        "evaluation_seed_set": [int(seed) for seed in eval_seeds],
        "episode_budget": int(trainers[QOS_STICKY_EE_CANDIDATE_ROLE].episodes),
        "dpc_sidecar_parameters": powers[QOS_STICKY_EE_CANDIDATE_ROLE],
        "role_constraints": constraints,
        "predeclared_tolerances": {
            **predeclared_tolerances(),
            "ee_delta_vs_hobs_no_anti_collapse_min": 0.0,
            "ee_delta_vs_anti_collapse_throughput_control_min": 0.0,
        },
    }


def summarize_broader_effectiveness_runs(
    rows: list[dict[str, Any]],
    boundary: dict[str, Any],
) -> dict[str, Any]:
    """Summarize completed role / seed artifacts into required verdicts."""
    role_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        role_rows.setdefault(row["role"], []).append(row)
    role_summaries = {
        role: _aggregate_role(role_rows.get(role, [])) for role in ROLE_ORDER
    }
    comparisons = [
        _candidate_vs_control(
            role_summaries[control_role],
            role_summaries[QOS_STICKY_EE_CANDIDATE_ROLE],
            control_role,
        )
        for control_role in CONTROL_ROLES
    ]
    anti_verdict = _anti_collapse_mechanism_verdict(
        boundary,
        role_summaries[QOS_STICKY_EE_CANDIDATE_ROLE],
        comparisons,
    )
    ee_verdict = _ee_objective_contribution_verdict(
        boundary,
        role_summaries,
        comparisons,
        anti_verdict,
    )
    stop_conditions = _stop_conditions(
        boundary,
        role_summaries,
        comparisons,
        anti_verdict,
        ee_verdict,
    )
    overall_status = _overall_status(anti_verdict, ee_verdict)

    return {
        "phase": "hobs-active-tx-ee-qos-sticky-broader-effectiveness",
        "method_label": "QoS-sticky HOBS-active-TX EE-MODQN",
        "method_markings": [
            "new extension / method variant",
            "bounded active-TX EE validation only",
            "not full EE-MODQN",
            "not physical energy saving",
            "not HOBS optimizer reproduction",
            "not RA-EE association",
            "not Catfish-EE",
        ],
        "status": overall_status,
        "boundary_proof": boundary,
        "protocol": {
            "role_count": len(ROLE_ORDER),
            "roles": list(ROLE_ORDER),
            "seed_triplets": boundary["seed_triplets"],
            "evaluation_seed_set": boundary["evaluation_seed_set"],
            "bounded_episode_budget": boundary["episode_budget"],
            "eval_schedule": "unchanged target_update_every_episodes from configs",
            "checkpoint_protocol": "unchanged final plus best-weighted-reward-on-eval",
            "scalar_reward_diagnostic_only": True,
        },
        "role_summaries": role_summaries,
        "role_table": [_role_table_row(role, role_summaries[role]) for role in ROLE_ORDER],
        "required_metrics": {
            role: _required_metric_view(role_summaries[role]) for role in ROLE_ORDER
        },
        "control_comparisons": comparisons,
        "anti_collapse_mechanism_verdict": anti_verdict,
        "ee_objective_contribution_verdict": ee_verdict,
        "acceptance_result": {
            "anti_collapse_mechanism": anti_verdict["status"],
            "ee_objective_contribution": ee_verdict["status"],
            "overall": overall_status,
        },
        "stop_conditions_triggered": stop_conditions,
        "allowed_claim_boundary": [
            "QoS-sticky overflow reassignment can be discussed only as a bounded anti-collapse mechanism if that verdict passes.",
            "Any EE statement is limited to active-TX EE under the disclosed HOBS-inspired DPC sidecar and this bounded protocol.",
            "The matched-throughput-control is a DPC-matched throughput-objective control, not the frozen MODQN baseline.",
            "Scalar reward remains diagnostic only.",
        ],
        "forbidden_claims_still_active": [
            "general EE-MODQN effectiveness",
            "physical energy saving",
            "HOBS optimizer reproduction",
            "full RA-EE-MODQN",
            "learned association effectiveness",
            "Catfish / Multi-Catfish / Catfish-EE repair",
            "scalar reward success",
            "denominator variability alone proves energy-aware learning",
            "QoS-sticky robustness PASS means general EE-MODQN effectiveness",
            "Phase 03D failure has been overturned",
        ],
        "run_artifacts": [
            {
                "role": row["role"],
                "seed_triplet": row["seed_triplet"],
                "run_dir": row["run_dir"],
                "summary_path": row["summary_path"],
            }
            for row in rows
        ],
    }


def _candidate_vs_control(
    control: dict[str, Any],
    candidate: dict[str, Any],
    control_role: str,
) -> dict[str, Any]:
    control_p05 = float(control.get("p05_throughput_bps") or 0.0)
    candidate_p05 = float(candidate.get("p05_throughput_bps") or 0.0)
    p05_ratio = (
        candidate_p05 / control_p05
        if control_p05 > 0.0 else (1.0 if candidate_p05 == 0.0 else float("inf"))
    )
    return {
        "candidate_role": QOS_STICKY_EE_CANDIDATE_ROLE,
        "control_role": control_role,
        "candidate_vs_control_EE_system_delta": float(
            float(candidate.get("EE_system") or 0.0) - float(control.get("EE_system") or 0.0)
        ),
        "candidate_vs_control_eta_EE_active_TX_delta": float(
            float(candidate.get("eta_EE_active_TX") or 0.0)
            - float(control.get("eta_EE_active_TX") or 0.0)
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


def _anti_collapse_mechanism_verdict(
    boundary: dict[str, Any],
    candidate: dict[str, Any],
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    reasons: list[str] = []
    if not bool(boundary.get("matched_boundary_pass", False)):
        return {"status": "NEEDS MORE DESIGN", "reasons": ["matched_boundary_pass=false"]}
    if bool(candidate.get("all_evaluated_steps_one_active_beam", True)):
        reasons.append("candidate all_evaluated_steps_one_active_beam=true")
    if not bool(candidate.get("denominator_varies_in_eval", False)):
        reasons.append("candidate denominator_varies_in_eval=false")
    if bool(candidate.get("active_power_single_point_distribution", True)):
        reasons.append("candidate active_power_single_point_distribution=true")
    if int(candidate.get("nonsticky_move_count") or 0) != 0:
        reasons.append("candidate nonsticky_move_count != 0")
    for key in (
        "budget_violation_count",
        "per_beam_power_violation_count",
        "inactive_beam_nonzero_power_step_count",
    ):
        if int(candidate.get(key) or 0) != 0:
            reasons.append(f"candidate {key} != 0")
    for comparison in comparisons:
        control_role = comparison["control_role"]
        if comparison["p05_throughput_ratio_vs_control"] < P05_THROUGHPUT_RATIO_MIN:
            reasons.append(f"p05 throughput guard failed vs {control_role}")
        if comparison["served_ratio_delta_vs_control"] < 0.0:
            reasons.append(f"served ratio guard failed vs {control_role}")
        if comparison["outage_ratio_delta_vs_control"] > 0.0:
            reasons.append(f"outage ratio guard failed vs {control_role}")
        if comparison["handover_delta_vs_control"] > HANDOVER_DELTA_MAX:
            reasons.append(f"handover guard failed vs {control_role}")
        if comparison["r2_delta_vs_control"] < R2_MEAN_DELTA_MIN:
            reasons.append(f"r2 guard failed vs {control_role}")
    return {"status": "PASS" if not reasons else "BLOCK", "reasons": reasons}


def _ee_objective_contribution_verdict(
    boundary: dict[str, Any],
    role_summaries: dict[str, dict[str, Any]],
    comparisons: list[dict[str, Any]],
    anti_verdict: dict[str, Any],
) -> dict[str, Any]:
    if not bool(boundary.get("matched_boundary_pass", False)):
        return {
            "status": "NEEDS MORE DESIGN",
            "reasons": ["matched boundary cannot isolate EE objective contribution"],
        }
    reasons: list[str] = []
    by_control = {row["control_role"]: row for row in comparisons}
    hobs_no = by_control[HOBS_EE_NO_ANTI_COLLAPSE_ROLE]
    anti_throughput = by_control[ANTI_COLLAPSE_THROUGHPUT_CONTROL_ROLE]
    candidate = role_summaries[QOS_STICKY_EE_CANDIDATE_ROLE]
    anti_control = role_summaries[ANTI_COLLAPSE_THROUGHPUT_CONTROL_ROLE]
    if anti_verdict["status"] != "PASS":
        reasons.append("anti-collapse / guardrail verdict is not PASS")
    if (
        hobs_no["candidate_vs_control_EE_system_delta"]
        < -EE_OBJECTIVE_EPSILON
    ):
        reasons.append("candidate active-TX EE is worse than hobs-ee-control-no-anti-collapse")
    candidate_beats_anti_throughput = (
        anti_throughput["candidate_vs_control_EE_system_delta"]
        > EE_OBJECTIVE_EPSILON
    )
    ranking_separation_beyond_anti_throughput = bool(
        candidate.get("same_policy_throughput_vs_ee_rescore_ranking_change", False)
    ) and not bool(
        anti_control.get("same_policy_throughput_vs_ee_rescore_ranking_change", False)
    )
    if not (candidate_beats_anti_throughput or ranking_separation_beyond_anti_throughput):
        reasons.append(
            "anti-collapse-throughput-control explains the EE/ranking gain boundary"
        )
    if (
        anti_throughput["scalar_reward_diagnostic_delta_vs_control"] > 0.0
        and anti_throughput["candidate_vs_control_EE_system_delta"]
        <= EE_OBJECTIVE_EPSILON
    ):
        reasons.append("candidate wins only scalar reward vs anti-collapse-throughput-control")
    return {
        "status": "PASS" if not reasons else "BLOCK",
        "reasons": reasons,
        "candidate_beats_anti_collapse_throughput_control_on_EE": (
            candidate_beats_anti_throughput
        ),
        "ranking_separation_beyond_anti_collapse_throughput_control": (
            ranking_separation_beyond_anti_throughput
        ),
    }


def _stop_conditions(
    boundary: dict[str, Any],
    role_summaries: dict[str, dict[str, Any]],
    comparisons: list[dict[str, Any]],
    anti_verdict: dict[str, Any],
    ee_verdict: dict[str, Any],
) -> list[str]:
    triggered: list[str] = []
    candidate = role_summaries[QOS_STICKY_EE_CANDIDATE_ROLE]
    if not bool(boundary.get("matched_boundary_pass", False)):
        triggered.append("matched boundary cannot isolate EE objective from anti-collapse")
    if bool(candidate.get("all_evaluated_steps_one_active_beam", True)):
        triggered.append("candidate remains all_evaluated_steps_one_active_beam=true")
    if any(
        "guard failed" in reason for reason in anti_verdict.get("reasons", [])
    ):
        triggered.append("candidate harms protected QoS / handover / r2 guardrails")
    if any("wins only scalar reward" in reason for reason in ee_verdict.get("reasons", [])):
        triggered.append("candidate wins only scalar reward")
    if any(
        "anti-collapse-throughput-control explains" in reason
        for reason in ee_verdict.get("reasons", [])
    ):
        triggered.append("anti-collapse-throughput-control explains all gains")
    if len(ROLE_ORDER) * len(boundary.get("seed_triplets", [])) > 16:
        triggered.append("role count / seed count too large for a bounded gate")
    if not triggered:
        triggered.append("None")
    return triggered


def _overall_status(
    anti_verdict: dict[str, Any],
    ee_verdict: dict[str, Any],
) -> str:
    if "NEEDS MORE DESIGN" in {anti_verdict["status"], ee_verdict["status"]}:
        return "NEEDS MORE DESIGN"
    if anti_verdict["status"] == "PASS" and ee_verdict["status"] == "PASS":
        return "PASS"
    return "BLOCK"


def _role_table_row(role: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": role,
        "r1": (
            "throughput"
            if role in {
                MATCHED_THROUGHPUT_CONTROL_ROLE,
                ANTI_COLLAPSE_THROUGHPUT_CONTROL_ROLE,
            }
            else "hobs-active-tx-ee"
        ),
        "dpc_sidecar": "same",
        "anti_collapse": role in ANTI_COLLAPSE_ROLES,
        "nonsticky_moves_disabled": True,
        **_required_metric_view(summary),
    }


def _required_metric_view(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "all_evaluated_steps_one_active_beam",
        "active_beam_count_distribution",
        "overflow_steps",
        "overflow_user_count",
        "sticky_override_count",
        "nonsticky_move_count",
        "qos_guard_reject_count",
        "handover_guard_reject_count",
        "denominator_varies_in_eval",
        "active_power_single_point_distribution",
        "distinct_total_active_power_w_values",
        "power_control_activity_rate",
        "eta_EE_active_TX",
        "EE_system",
        "throughput_vs_ee_pearson",
        "same_policy_throughput_vs_ee_rescore_ranking_change",
        "raw_throughput_mean_bps",
        "p05_throughput_bps",
        "served_ratio",
        "outage_ratio",
        "handover_count",
        "r2_mean",
        "load_balance_metric",
        "budget_violation_count",
        "per_beam_power_violation_count",
        "inactive_beam_nonzero_power_step_count",
        "episode_scalar_reward_diagnostic_mean",
    )
    return {key: summary.get(key) for key in keys}


def _protocol_seed_triplets(cfg: dict[str, Any]) -> tuple[tuple[int, int, int], ...]:
    block = cfg.get("broader_effectiveness_protocol", {})
    rows = block.get("seed_triplets", []) if isinstance(block, dict) else []
    return tuple(tuple(int(value) for value in row) for row in rows)


def _same_environment_boundary(envs: dict[str, Any]) -> bool:
    first = envs[MATCHED_THROUGHPUT_CONTROL_ROLE]
    return all(
        env.config == first.config
        and env.orbit.config == first.orbit.config
        and env.beam_pattern.config == first.beam_pattern.config
        and env.channel_config == first.channel_config
        for env in envs.values()
    )


def _role_seed_dir(
    role: str,
    seed_index: int,
    seed_triplet: list[int] | tuple[int, int, int],
) -> Path:
    train, env, mobility = (int(value) for value in seed_triplet)
    return (
        Path(f"artifacts/hobs-active-tx-ee-qos-sticky-broader-effectiveness-{role}")
        / f"seed-{seed_index:02d}-t{train}-e{env}-m{mobility}"
    )


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
    seed_value = cfg.setdefault("resolved_assumptions", {}).setdefault(
        "seed_and_rng_policy",
        {"assumption_id": "ASSUME-MODQN-REP-018", "value": {}},
    ).setdefault("value", {})
    train_seed, env_seed, mobility_seed = (int(value) for value in seed_triplet)
    seed_value["train_seed"] = train_seed
    seed_value["environment_seed"] = env_seed
    seed_value["mobility_seed"] = mobility_seed
    seed_value["evaluation_seed_set"] = [100, 200, 300, 400, 500]
    track = cfg.setdefault("track", {})
    track["broader_effectiveness_role"] = role
    track["seed_index"] = seed_index
    track["materialized_for_artifact_dir"] = str(output_dir)
    experiment = cfg.setdefault("training_experiment", {})
    experiment["experiment_id"] = (
        f"{experiment.get('experiment_id', 'HOBS-QOS-STICKY-BROADER')}"
        f"-SEED-{seed_index:02d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.resolved.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _write_role_table(path: Path, rows: list[dict[str, Any]]) -> Path:
    fieldnames = [
        "role",
        "r1",
        "anti_collapse",
        "all_evaluated_steps_one_active_beam",
        "active_beam_count_distribution",
        "denominator_varies_in_eval",
        "active_power_single_point_distribution",
        "eta_EE_active_TX",
        "EE_system",
        "raw_throughput_mean_bps",
        "p05_throughput_bps",
        "served_ratio",
        "outage_ratio",
        "handover_count",
        "r2_mean",
        "load_balance_metric",
        "nonsticky_move_count",
        "budget_violation_count",
        "per_beam_power_violation_count",
        "inactive_beam_nonzero_power_step_count",
    ]
    return _write_csv(path, rows, fieldnames)


def _write_comparison_table(path: Path, rows: list[dict[str, Any]]) -> Path:
    fieldnames = [
        "candidate_role",
        "control_role",
        "candidate_vs_control_EE_system_delta",
        "candidate_vs_control_eta_EE_active_TX_delta",
        "candidate_vs_control_throughput_delta_bps",
        "p05_throughput_ratio_vs_control",
        "served_ratio_delta_vs_control",
        "outage_ratio_delta_vs_control",
        "handover_delta_vs_control",
        "r2_delta_vs_control",
        "load_balance_metric_delta_vs_control",
        "scalar_reward_diagnostic_delta_vs_control",
    ]
    return _write_csv(path, rows, fieldnames)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return path


def _write_execution_report(path: Path, summary: dict[str, Any]) -> Path:
    anti = summary["anti_collapse_mechanism_verdict"]
    ee = summary["ee_objective_contribution_verdict"]
    lines = [
        "# HOBS Active-TX EE QoS-Sticky Broader-Effectiveness Gate Execution Report",
        "",
        "**Date:** `2026-05-01`",
        f"**Status:** `{summary['status']}`",
        "**Method label:** `QoS-sticky HOBS-active-TX EE-MODQN`",
        "**Scope:** new extension / method variant; bounded active-TX EE",
        "validation only; not full EE-MODQN; not physical energy saving; not",
        "HOBS optimizer reproduction; not RA-EE association; not Catfish-EE.",
        "",
        "## Protocol",
        "",
        "- Roles: `matched-throughput-control`, `hobs-ee-control-no-anti-collapse`, `qos-sticky-ee-candidate`, `anti-collapse-throughput-control`.",
        "- Seed triplets: `[42, 1337, 7]`, `[43, 1338, 8]`, `[44, 1339, 9]`.",
        "- Evaluation seeds: `[100, 200, 300, 400, 500]`.",
        "- Episode budget: `5` per role / seed triplet.",
        "- Scalar reward is diagnostic only.",
        "",
        "## Matched Boundary Proof",
        "",
        f"`matched_boundary_pass={summary['boundary_proof']['matched_boundary_pass']}`.",
        "All roles use the same environment boundary, HOBS-inspired DPC sidecar,",
        "seed triplets, eval seeds, bounded training budget, eval schedule,",
        "checkpoint protocol, objective weights, and hyperparameters. The only",
        "intended differences are `r1_reward_mode` and whether the QoS-sticky",
        "anti-collapse hook is enabled. The matched-throughput-control is a",
        "DPC-matched throughput control, not the frozen MODQN baseline.",
        "",
        "## Role Table",
        "",
        "| Role | r1 | anti-collapse | EE_system | throughput | p05 | served | outage | handovers | r2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["role_table"]:
        lines.append(
            f"| `{row['role']}` | `{row['r1']}` | `{row['anti_collapse']}` | "
            f"`{row.get('EE_system')}` | `{row.get('raw_throughput_mean_bps')}` | "
            f"`{row.get('p05_throughput_bps')}` | `{row.get('served_ratio')}` | "
            f"`{row.get('outage_ratio')}` | `{row.get('handover_count')}` | "
            f"`{row.get('r2_mean')}` |"
        )
    lines.extend([
        "",
        "## Control Comparisons",
        "",
        "| Control | EE delta | throughput delta | p05 ratio | served delta | outage delta | handover delta | r2 delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary["control_comparisons"]:
        lines.append(
            f"| `{row['control_role']}` | "
            f"`{row['candidate_vs_control_EE_system_delta']}` | "
            f"`{row['candidate_vs_control_throughput_delta_bps']}` | "
            f"`{row['p05_throughput_ratio_vs_control']}` | "
            f"`{row['served_ratio_delta_vs_control']}` | "
            f"`{row['outage_ratio_delta_vs_control']}` | "
            f"`{row['handover_delta_vs_control']}` | "
            f"`{row['r2_delta_vs_control']}` |"
        )
    lines.extend([
        "",
        "## Verdicts",
        "",
        f"- Anti-collapse mechanism: `{anti['status']}`",
        f"- EE objective contribution: `{ee['status']}`",
        "",
        "## Acceptance Result",
        "",
        f"`PASS / BLOCK / NEEDS MORE DESIGN: {summary['status']}`",
        "",
        "## Stop Conditions Triggered",
        "",
    ])
    lines.extend(f"- {item}" for item in summary["stop_conditions_triggered"])
    lines.extend([
        "",
        "## Allowed Claim Boundary",
        "",
    ])
    lines.extend(f"- {item}" for item in summary["allowed_claim_boundary"])
    lines.extend([
        "",
        "## Forbidden Claims Still Active",
        "",
    ])
    lines.extend(f"- {item}" for item in summary["forbidden_claims_still_active"])
    lines.extend([
        "",
        "## Artifact Paths",
        "",
        "- `artifacts/hobs-active-tx-ee-qos-sticky-broader-effectiveness-*/`",
        "- `artifacts/hobs-active-tx-ee-qos-sticky-broader-effectiveness-summary/summary.json`",
        "- `artifacts/hobs-active-tx-ee-qos-sticky-broader-effectiveness-summary/role_table.csv`",
        "- `artifacts/hobs-active-tx-ee-qos-sticky-broader-effectiveness-summary/control_comparisons.csv`",
    ])
    if anti["reasons"] or ee["reasons"]:
        lines.extend(["", "## Reasons", ""])
        lines.extend(f"- Anti-collapse: {item}" for item in anti["reasons"])
        lines.extend(f"- EE objective: {item}" for item in ee["reasons"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


__all__ = [
    "ANTI_COLLAPSE_THROUGHPUT_CONTROL_ROLE",
    "ARTIFACT_ROOT",
    "CONTROL_ROLES",
    "HOBS_EE_NO_ANTI_COLLAPSE_ROLE",
    "MATCHED_THROUGHPUT_CONTROL_ROLE",
    "QOS_STICKY_EE_CANDIDATE_ROLE",
    "REPORT_PATH",
    "ROLE_CONFIGS",
    "ROLE_ORDER",
    "prove_broader_effectiveness_boundary",
    "run_qos_sticky_broader_effectiveness_gate",
    "summarize_broader_effectiveness_runs",
]
