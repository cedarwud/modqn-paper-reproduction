"""Bounded QoS-sticky HOBS active-TX EE robustness gate.

This runner keeps the existing tiny anti-collapse training budget and repeats
the QoS-sticky overflow mechanism across matched seed triplets and bounded
ablation roles.  It is an attribution / robustness gate, not an EE-MODQN
effectiveness promotion.
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..config_loader import (
    build_environment,
    build_trainer_config,
    get_seeds,
    load_training_yaml,
)
from ..env.step import HOBS_POWER_SURFACE_DPC_SIDECAR
from ..runtime.trainer_spec import (
    HOBS_ACTIVE_TX_EE_ANTI_COLLAPSE_KIND,
    R1_REWARD_MODE_HOBS_ACTIVE_TX_EE,
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

ROLE_CONFIGS: dict[str, Path] = {
    "matched-control": Path(
        "configs/hobs-active-tx-ee-qos-sticky-robustness-matched-control.resolved.yaml"
    ),
    "primary-qos-sticky": Path(
        "configs/hobs-active-tx-ee-qos-sticky-robustness-primary.resolved.yaml"
    ),
    "no-qos-guard-ablation": Path(
        "configs/hobs-active-tx-ee-qos-sticky-robustness-no-qos-guard.resolved.yaml"
    ),
    "stricter-qos-ablation": Path(
        "configs/hobs-active-tx-ee-qos-sticky-robustness-stricter-qos.resolved.yaml"
    ),
    "threshold-sensitivity-45": Path(
        "configs/hobs-active-tx-ee-qos-sticky-robustness-threshold-45.resolved.yaml"
    ),
    "threshold-sensitivity-55": Path(
        "configs/hobs-active-tx-ee-qos-sticky-robustness-threshold-55.resolved.yaml"
    ),
}
ROLE_ORDER = tuple(ROLE_CONFIGS.keys())
CONTROL_ROLE = "matched-control"
PRIMARY_ROLE = "primary-qos-sticky"
ARTIFACT_ROOT = Path("artifacts/hobs-active-tx-ee-qos-sticky-robustness-summary")
REPORT_PATH = Path(
    "docs/research/catfish-ee-modqn/"
    "hobs-active-tx-ee-qos-sticky-robustness-gate.execution-report.md"
)

MEAN_DIAGNOSTIC_KEYS = (
    "power_control_activity_rate",
    "throughput_vs_ee_pearson",
    "eta_EE_active_TX",
    "eta_EE_active_TX_eval_aggregate",
    "eta_EE_active_TX_step_mean",
    "EE_system",
    "EE_system_eval_aggregate",
    "EE_system_step_mean",
    "raw_throughput_mean_bps",
    "raw_episode_throughput_mean_bps",
    "p05_throughput_bps",
    "served_ratio",
    "outage_ratio",
    "handover_count",
    "r2",
    "r2_mean",
    "load_balance_metric",
    "r3_mean",
    "active_beam_load_gap_mean",
    "scalar_reward_diagnostic_mean",
    "episode_scalar_reward_diagnostic_mean",
)
SUM_DIAGNOSTIC_KEYS = (
    "budget_violation_count",
    "per_beam_power_violation_count",
    "inactive_beam_nonzero_power_step_count",
    "overflow_steps",
    "overflow_user_count",
    "sticky_override_count",
    "nonsticky_move_count",
    "qos_guard_reject_count",
    "handover_guard_reject_count",
    "dpc_sign_flip_count",
    "dpc_step_count",
    "dpc_qos_guard_count",
    "dpc_per_beam_cap_clip_count",
    "dpc_sat_cap_clip_count",
)


def run_qos_sticky_robustness_gate() -> dict[str, Any]:
    """Run all bounded robustness roles and write aggregate artifacts."""
    boundary = prove_robustness_boundary()
    if not bool(boundary["matched_boundary_pass"]):
        summary = {
            "boundary_proof": boundary,
            "status": "NEEDS MORE DESIGN",
            "reason": "matched boundary could not be proven before runs",
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

    summary = summarize_qos_sticky_robustness_runs(rows, boundary)
    summary_path = write_json(ARTIFACT_ROOT / "summary.json", summary)
    per_seed_csv = _write_per_seed_csv(
        ARTIFACT_ROOT / "per_seed_pass_fail_table.csv",
        summary["per_seed_pass_fail_table"],
    )
    aggregate_csv = _write_aggregate_csv(
        ARTIFACT_ROOT / "aggregate_pass_fail_table.csv",
        summary["aggregate_pass_fail_table"],
    )
    report_path = _write_execution_report(REPORT_PATH, summary)
    summary["artifact_paths"] = {
        "summary_json": str(summary_path),
        "per_seed_pass_fail_csv": str(per_seed_csv),
        "aggregate_pass_fail_csv": str(aggregate_csv),
        "execution_report_md": str(report_path),
    }
    write_json(ARTIFACT_ROOT / "summary.json", summary)
    return summary


def prove_robustness_boundary(
    role_configs: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Prove that robustness roles share the declared matched boundary."""
    role_configs = dict(role_configs or ROLE_CONFIGS)
    loaded = {role: load_training_yaml(path) for role, path in role_configs.items()}
    trainers = {role: build_trainer_config(cfg) for role, cfg in loaded.items()}
    envs = {role: build_environment(cfg) for role, cfg in loaded.items()}
    powers = {
        role: _dpc_parameter_subset(_config_power_surface_value(cfg))
        for role, cfg in loaded.items()
    }
    constraints = {role: _anti_collapse_subset(trainers[role]) for role in trainers}
    seed_triplets = _protocol_seed_triplets(loaded[CONTROL_ROLE])
    eval_seeds = get_seeds(loaded[CONTROL_ROLE])["evaluation_seed_set"]

    primary_constraint = constraints[PRIMARY_ROLE]
    no_qos_constraint = constraints["no-qos-guard-ablation"]
    stricter_constraint = constraints["stricter-qos-ablation"]
    threshold_45 = constraints["threshold-sensitivity-45"]
    threshold_55 = constraints["threshold-sensitivity-55"]

    checks = {
        "required_roles_present": set(role_configs) == set(ROLE_ORDER),
        "all_r1_are_hobs_active_tx_ee": all(
            cfg.r1_reward_mode == R1_REWARD_MODE_HOBS_ACTIVE_TX_EE
            for cfg in trainers.values()
        ),
        "all_training_experiment_kind_match": all(
            cfg.training_experiment_kind == HOBS_ACTIVE_TX_EE_ANTI_COLLAPSE_KIND
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
        "at_least_three_seed_triplets": len(seed_triplets) >= 3,
        "same_seed_triplets_declared": all(
            _protocol_seed_triplets(cfg) == seed_triplets for cfg in loaded.values()
        ),
        "same_dpc_sidecar": len({tuple(sorted(p.items())) for p in powers.values()}) == 1,
        "dpc_sidecar_enabled": all(
            env.power_surface_config.hobs_power_surface_mode
            == HOBS_POWER_SURFACE_DPC_SIDECAR
            for env in envs.values()
        ),
        "same_environment_boundary": _same_environment_boundary(envs),
        "matched_control_constraint_disabled": (
            not trainers[CONTROL_ROLE].anti_collapse_action_constraint_enabled
        ),
        "candidate_roles_constraint_enabled": all(
            trainers[role].anti_collapse_action_constraint_enabled
            for role in ROLE_ORDER
            if role != CONTROL_ROLE
        ),
        "all_candidate_modes_are_qos_sticky": all(
            constraints[role]["mode"] == "qos-sticky-overflow-reassignment"
            for role in ROLE_ORDER
            if role != CONTROL_ROLE
        ),
        "no_forced_min_active_beams_target": all(
            int(value["min_active_beams_target"]) == 0 for value in constraints.values()
        ),
        "all_roles_disable_nonsticky_moves": all(
            not bool(value["allow_nonsticky_moves"]) for value in constraints.values()
        ),
        "all_nonsticky_budgets_zero": all(
            int(value["nonsticky_move_budget"]) == 0 for value in constraints.values()
        ),
        "primary_qos_ratio_min_0_95": primary_constraint["qos_ratio_min"] == 0.95,
        "no_qos_guard_relaxes_qos_ratio": (
            no_qos_constraint["qos_ratio_min"] < primary_constraint["qos_ratio_min"]
        ),
        "stricter_qos_raises_qos_ratio": (
            stricter_constraint["qos_ratio_min"] > primary_constraint["qos_ratio_min"]
        ),
        "threshold_sensitivity_nearby": (
            threshold_45["overload_threshold_users_per_beam"] == 45
            and primary_constraint["overload_threshold_users_per_beam"] == 50
            and threshold_55["overload_threshold_users_per_beam"] == 55
        ),
        "no_forbidden_claim_terms_in_configs": not _configs_contain_forbidden_terms(
            loaded
        ),
    }
    return {
        "matched_boundary_pass": all(bool(value) for value in checks.values()),
        "checks": checks,
        "roles": list(ROLE_ORDER),
        "role_config_paths": {
            role: str(path) for role, path in role_configs.items()
        },
        "allowed_differences": [
            "matched-control disables the opt-in anti-collapse hook",
            "primary enables qos-sticky-overflow-reassignment at threshold=50 and qos_ratio_min=0.95",
            "no-qos-guard ablation relaxes qos_ratio_min to 0.01",
            "stricter-qos ablation raises qos_ratio_min to 1.05",
            "threshold ablations change overload_threshold_users_per_beam to 45 and 55",
        ],
        "seed_triplets": [list(row) for row in seed_triplets],
        "evaluation_seed_set": [int(seed) for seed in eval_seeds],
        "episode_budget": int(trainers[CONTROL_ROLE].episodes),
        "dpc_sidecar_parameters": powers[CONTROL_ROLE],
        "role_constraints": constraints,
        "predeclared_tolerances": predeclared_tolerances(),
    }


def summarize_qos_sticky_robustness_runs(
    rows: list[dict[str, Any]],
    boundary: dict[str, Any],
) -> dict[str, Any]:
    """Summarize completed role / seed artifacts into pass-fail tables."""
    role_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        role_rows.setdefault(row["role"], []).append(row)
    role_summaries = {
        role: _aggregate_role(role_rows.get(role, [])) for role in ROLE_ORDER
    }
    per_seed_table = _per_seed_pass_fail_table(rows)
    aggregate_table = _aggregate_pass_fail_table(role_summaries)
    mechanism = _mechanism_attribution(role_summaries, aggregate_table)
    acceptance = _overall_acceptance(boundary, per_seed_table, aggregate_table, mechanism)

    return {
        "phase": "hobs-active-tx-ee-qos-sticky-robustness",
        "mode": "qos-sticky-overflow-reassignment",
        "status": acceptance["status"],
        "acceptance": acceptance,
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
        "per_seed_pass_fail_table": per_seed_table,
        "aggregate_pass_fail_table": aggregate_table,
        "mechanism_attribution": mechanism,
        "run_artifacts": [
            {
                "role": row["role"],
                "seed_triplet": row["seed_triplet"],
                "run_dir": row["run_dir"],
                "summary_path": row["summary_path"],
            }
            for row in rows
        ],
        "forbidden_claims_still_active": [
            "EE-MODQN effectiveness",
            "physical energy saving",
            "HOBS optimizer reproduction",
            "full RA-EE-MODQN",
            "learned association effectiveness",
            "Catfish / Multi-Catfish / Catfish-EE repair",
            "Phase 03C selector or reward tuning",
            "scalar reward as success evidence",
        ],
        "deviations_or_blockers": acceptance["deviations_or_blockers"],
    }


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
    )
    return all(
        len({getattr(cfg, key) for cfg in trainers.values()}) == 1 for key in keys
    )


def _same_checkpoint_rule(trainers: dict[str, Any]) -> bool:
    return (
        len({cfg.checkpoint_assumption_id for cfg in trainers.values()}) == 1
        and len({cfg.checkpoint_primary_report for cfg in trainers.values()}) == 1
        and len({cfg.checkpoint_secondary_report for cfg in trainers.values()}) == 1
    )


def _same_environment_boundary(envs: dict[str, Any]) -> bool:
    first = envs[CONTROL_ROLE]
    return all(
        env.config == first.config
        and env.orbit.config == first.orbit.config
        and env.beam_pattern.config == first.beam_pattern.config
        and env.channel_config == first.channel_config
        for env in envs.values()
    )


def _configs_contain_forbidden_terms(configs: dict[str, dict[str, Any]]) -> bool:
    serialized = yaml.safe_dump(configs, sort_keys=True)
    forbidden = (
        "Catfish",
        "Multi-Catfish",
        "Catfish-EE",
        "RA-EE",
        "phase-03c",
        "physical energy saving",
    )
    return any(term in serialized for term in forbidden)


def _protocol_seed_triplets(cfg: dict[str, Any]) -> tuple[tuple[int, int, int], ...]:
    block = cfg.get("robustness_protocol", {})
    rows = block.get("seed_triplets", []) if isinstance(block, dict) else []
    return tuple(tuple(int(value) for value in row) for row in rows)


def _role_seed_dir(
    role: str,
    seed_index: int,
    seed_triplet: list[int] | tuple[int, int, int],
) -> Path:
    train, env, mobility = (int(value) for value in seed_triplet)
    return (
        Path(f"artifacts/hobs-active-tx-ee-qos-sticky-robustness-{role}")
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
    track["robustness_role"] = role
    track["seed_index"] = seed_index
    track["materialized_for_artifact_dir"] = str(output_dir)
    experiment = cfg.setdefault("training_experiment", {})
    experiment["experiment_id"] = (
        f"{experiment.get('experiment_id', 'HOBS-QOS-STICKY-ROBUSTNESS')}"
        f"-SEED-{seed_index:02d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.resolved.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _aggregate_role(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"run_count": 0}
    diags = [row["summary"]["diagnostics"] for row in rows]
    active_dist = _merge_distributions(diag["active_beam_count_distribution"] for diag in diags)
    total_power_dist = _merge_distributions(diag["total_active_power_distribution"] for diag in diags)
    distinct_total = _unique_union(
        value
        for diag in diags
        for value in diag.get("distinct_total_active_power_w_values", [])
    )
    distinct_active = _unique_union(
        value
        for diag in diags
        for value in diag.get("distinct_active_power_w_values", [])
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
        "active_beam_count_distribution": active_dist,
        "total_active_power_distribution": total_power_dist,
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
    for key in MEAN_DIAGNOSTIC_KEYS:
        values = [d.get(key) for d in diags if d.get(key) is not None]
        aggregate[key] = None if not values else float(np.mean(values))
    for key in SUM_DIAGNOSTIC_KEYS:
        aggregate[key] = int(sum(int(d.get(key, 0)) for d in diags))
    aggregate["per_seed_metrics"] = [
        {
            "seed_triplet": row["seed_triplet"],
            **_selected_metric_view(row["summary"]["diagnostics"]),
        }
        for row in rows
    ]
    return aggregate


def _selected_metric_view(diag: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "all_evaluated_steps_one_active_beam",
        "active_beam_count_distribution",
        "denominator_varies_in_eval",
        "active_power_single_point_distribution",
        "power_control_activity_rate",
        "throughput_vs_ee_pearson",
        "same_policy_throughput_vs_ee_rescore_ranking_change",
        "eta_EE_active_TX",
        "EE_system",
        "raw_throughput_mean_bps",
        "p05_throughput_bps",
        "served_ratio",
        "outage_ratio",
        "handover_count",
        "r2_mean",
        "load_balance_metric",
        "episode_scalar_reward_diagnostic_mean",
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
    return {key: diag.get(key) for key in keys}


def _per_seed_pass_fail_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed_role = {
        (int(row["seed_index"]), row["role"]): row["summary"]["diagnostics"]
        for row in rows
    }
    table: list[dict[str, Any]] = []
    seed_indices = sorted({int(row["seed_index"]) for row in rows})
    for seed_index in seed_indices:
        control_diag = by_seed_role[(seed_index, CONTROL_ROLE)]
        table.append({
            "seed_index": seed_index,
            "seed_triplet": _seed_triplet_for_row(rows, seed_index),
            "role": CONTROL_ROLE,
            "status": "REFERENCE",
            "reasons": [],
            **_selected_metric_view(control_diag),
        })
        for role in ROLE_ORDER:
            if role == CONTROL_ROLE:
                continue
            diag = by_seed_role[(seed_index, role)]
            verdict = _candidate_verdict(control_diag, diag)
            table.append({
                "seed_index": seed_index,
                "seed_triplet": _seed_triplet_for_row(rows, seed_index),
                "role": role,
                "status": verdict["status"],
                "reasons": verdict["reasons"],
                **_selected_metric_view(diag),
                **verdict["deltas"],
            })
    return table


def _aggregate_pass_fail_table(
    role_summaries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    control = role_summaries[CONTROL_ROLE]
    table: list[dict[str, Any]] = []
    for role in ROLE_ORDER:
        summary = role_summaries[role]
        if role == CONTROL_ROLE:
            table.append({
                "role": role,
                "status": "REFERENCE",
                "reasons": [],
                **_aggregate_metric_view(summary),
            })
            continue
        verdict = _candidate_verdict(control, summary)
        table.append({
            "role": role,
            "status": verdict["status"],
            "reasons": verdict["reasons"],
            **_aggregate_metric_view(summary),
            **verdict["deltas"],
        })
    return table


def _aggregate_metric_view(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "all_evaluated_steps_one_active_beam",
        "per_seed_one_active_beam_count",
        "active_beam_count_distribution",
        "denominator_varies_in_eval",
        "active_power_single_point_distribution",
        "distinct_total_active_power_w_values",
        "power_control_activity_rate",
        "throughput_vs_ee_pearson",
        "same_policy_throughput_vs_ee_rescore_ranking_change",
        "eta_EE_active_TX",
        "EE_system",
        "raw_throughput_mean_bps",
        "p05_throughput_bps",
        "served_ratio",
        "outage_ratio",
        "handover_count",
        "r2_mean",
        "load_balance_metric",
        "episode_scalar_reward_diagnostic_mean",
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
    return {key: summary.get(key) for key in keys}


def _candidate_verdict(
    control_diag: dict[str, Any],
    candidate_diag: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    if bool(candidate_diag.get("all_evaluated_steps_one_active_beam", True)):
        reasons.append("all_evaluated_steps_one_active_beam=true")
    if not bool(candidate_diag.get("denominator_varies_in_eval", False)):
        reasons.append("denominator_varies_in_eval=false")
    if bool(candidate_diag.get("active_power_single_point_distribution", True)):
        reasons.append("active_power_single_point_distribution=true")

    control_p05 = float(control_diag.get("p05_throughput_bps") or 0.0)
    candidate_p05 = float(candidate_diag.get("p05_throughput_bps") or 0.0)
    p05_ratio = (
        candidate_p05 / control_p05
        if control_p05 > 0.0 else (1.0 if candidate_p05 == 0.0 else float("inf"))
    )
    served_delta = float(candidate_diag.get("served_ratio") or 0.0) - float(
        control_diag.get("served_ratio") or 0.0
    )
    outage_delta = float(candidate_diag.get("outage_ratio") or 0.0) - float(
        control_diag.get("outage_ratio") or 0.0
    )
    handover_delta = float(candidate_diag.get("handover_count") or 0.0) - float(
        control_diag.get("handover_count") or 0.0
    )
    r2_delta = float(candidate_diag.get("r2_mean") or 0.0) - float(
        control_diag.get("r2_mean") or 0.0
    )
    if p05_ratio < P05_THROUGHPUT_RATIO_MIN:
        reasons.append("p05_throughput_ratio_vs_control below 0.95")
    if served_delta < 0.0:
        reasons.append("served_ratio_delta below 0")
    if outage_delta > 0.0:
        reasons.append("outage_ratio_delta above 0")
    if handover_delta > HANDOVER_DELTA_MAX:
        reasons.append("handover_delta above +25")
    if r2_delta < R2_MEAN_DELTA_MIN:
        reasons.append("r2_delta below -0.05")
    for key in (
        "budget_violation_count",
        "per_beam_power_violation_count",
        "inactive_beam_nonzero_power_step_count",
    ):
        if int(candidate_diag.get(key, 0) or 0) > 0:
            reasons.append(f"{key} > 0")

    scalar_delta = (
        float(candidate_diag.get("episode_scalar_reward_diagnostic_mean") or 0.0)
        - float(control_diag.get("episode_scalar_reward_diagnostic_mean") or 0.0)
    )
    if scalar_delta > 0.0 and reasons:
        reasons.append("scalar reward improves but predeclared guards fail")

    return {
        "status": "PASS" if not reasons else "BLOCK",
        "reasons": reasons,
        "deltas": {
            "p05_throughput_ratio_vs_control": float(p05_ratio),
            "p05_throughput_delta_bps": float(candidate_p05 - control_p05),
            "served_ratio_delta": float(served_delta),
            "outage_ratio_delta": float(outage_delta),
            "handover_delta": float(handover_delta),
            "r2_delta": float(r2_delta),
            "load_balance_metric_delta": float(
                float(candidate_diag.get("load_balance_metric") or 0.0)
                - float(control_diag.get("load_balance_metric") or 0.0)
            ),
            "scalar_reward_diagnostic_delta": float(scalar_delta),
        },
    }


def _mechanism_attribution(
    role_summaries: dict[str, dict[str, Any]],
    aggregate_table: list[dict[str, Any]],
) -> dict[str, Any]:
    by_role = {row["role"]: row for row in aggregate_table}
    primary = by_role[PRIMARY_ROLE]
    no_qos = by_role["no-qos-guard-ablation"]
    strict = by_role["stricter-qos-ablation"]
    threshold_rows = [
        by_role["threshold-sensitivity-45"],
        by_role["threshold-sensitivity-55"],
    ]

    no_qos_passes = no_qos["status"] == "PASS"
    no_qos_scalar_wins = (
        float(no_qos["scalar_reward_diagnostic_delta"])
        > float(primary["scalar_reward_diagnostic_delta"])
    )
    no_qos_protection_failure = no_qos["status"] == "BLOCK" and any(
        "p05" in reason or "r2" in reason or "handover" in reason
        for reason in no_qos["reasons"]
    )
    strict_lower_intervention = (
        int(role_summaries["stricter-qos-ablation"]["sticky_override_count"])
        <= int(role_summaries[PRIMARY_ROLE]["sticky_override_count"])
    )
    threshold_fragility = any(row["status"] == "BLOCK" for row in threshold_rows)

    notes: list[str] = []
    if no_qos_passes:
        notes.append(
            "Relaxed-QoS ablation also passes; QoS guard may not be the active "
            "mechanism on this bounded surface."
        )
    elif no_qos_scalar_wins and no_qos_protection_failure:
        notes.append(
            "Relaxed-QoS ablation improves scalar diagnostics while failing "
            "p05/r2/handover protection; QoS guard is necessary."
        )
    elif no_qos_protection_failure:
        notes.append("Relaxed-QoS ablation fails protected metrics; QoS guard matters.")
    else:
        notes.append("Relaxed-QoS ablation does not explain the primary pass.")

    if strict["status"] == "PASS" and strict_lower_intervention:
        notes.append("Stricter-QoS preserves pass with lower or equal sticky intervention.")
    elif strict["status"] == "PASS":
        notes.append("Stricter-QoS preserves pass but does not lower intervention.")
    else:
        notes.append("Stricter-QoS blocks or weakens the mechanism.")

    if threshold_fragility:
        notes.append("Nearby threshold change flips at least one arm to BLOCK.")
    else:
        notes.append("Both nearby threshold sensitivity arms preserve aggregate pass.")

    return {
        "no_qos_guard_passes_without_protected_metric_failure": no_qos_passes,
        "no_qos_guard_scalar_wins": no_qos_scalar_wins,
        "no_qos_guard_protection_failure": no_qos_protection_failure,
        "stricter_qos_passes": strict["status"] == "PASS",
        "stricter_qos_lower_or_equal_intervention": strict_lower_intervention,
        "threshold_sensitivity_passes": not threshold_fragility,
        "threshold_fragility_detected": threshold_fragility,
        "notes": notes,
    }


def _overall_acceptance(
    boundary: dict[str, Any],
    per_seed_table: list[dict[str, Any]],
    aggregate_table: list[dict[str, Any]],
    mechanism: dict[str, Any],
) -> dict[str, Any]:
    by_role = {row["role"]: row for row in aggregate_table}
    primary = by_role[PRIMARY_ROLE]
    primary_per_seed = [
        row for row in per_seed_table if row["role"] == PRIMARY_ROLE
    ]
    reasons: list[str] = []
    blockers: list[str] = []
    if not bool(boundary.get("matched_boundary_pass", False)):
        blockers.append("matched boundary cannot be proven")
    if primary["status"] != "PASS":
        blockers.append("primary aggregate acceptance failed")
    if any(row["status"] != "PASS" for row in primary_per_seed):
        blockers.append("primary has at least one per-seed failure")
    if int(primary.get("per_seed_one_active_beam_count") or 0) > 0:
        blockers.append("primary has per-seed one-active-beam collapse")
    if bool(mechanism["threshold_fragility_detected"]):
        reasons.append("threshold sensitivity detected; needs more design")

    if blockers:
        status = "BLOCK"
    elif reasons:
        status = "NEEDS MORE DESIGN"
    else:
        status = "PASS"
    return {
        "status": status,
        "primary_aggregate_status": primary["status"],
        "primary_all_seed_statuses_pass": all(
            row["status"] == "PASS" for row in primary_per_seed
        ),
        "matched_boundary_pass": bool(boundary.get("matched_boundary_pass", False)),
        "threshold_fragility_detected": bool(
            mechanism["threshold_fragility_detected"]
        ),
        "reasons": reasons,
        "blockers": blockers,
        "deviations_or_blockers": blockers + reasons,
    }


def _seed_triplet_for_row(rows: list[dict[str, Any]], seed_index: int) -> list[int]:
    for row in rows:
        if int(row["seed_index"]) == seed_index:
            return row["seed_triplet"]
    return []


def _merge_distributions(distributions: Any) -> dict[str, int]:
    merged: dict[str, int] = {}
    for distribution in distributions:
        for key, value in distribution.items():
            merged[str(key)] = merged.get(str(key), 0) + int(value)
    return dict(sorted(merged.items(), key=lambda item: float(item[0])))


def _unique_union(values: Any, *, places: int = 9) -> list[float]:
    return sorted({round(float(value), places) for value in values})


def _write_per_seed_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    fieldnames = [
        "seed_index",
        "seed_triplet",
        "role",
        "status",
        "p05_throughput_ratio_vs_control",
        "served_ratio_delta",
        "outage_ratio_delta",
        "handover_delta",
        "r2_delta",
        "all_evaluated_steps_one_active_beam",
        "active_beam_count_distribution",
        "denominator_varies_in_eval",
        "active_power_single_point_distribution",
        "nonsticky_move_count",
        "qos_guard_reject_count",
        "handover_guard_reject_count",
        "reasons",
    ]
    return _write_csv(path, rows, fieldnames)


def _write_aggregate_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    fieldnames = [
        "role",
        "status",
        "p05_throughput_ratio_vs_control",
        "served_ratio_delta",
        "outage_ratio_delta",
        "handover_delta",
        "r2_delta",
        "all_evaluated_steps_one_active_beam",
        "per_seed_one_active_beam_count",
        "active_beam_count_distribution",
        "denominator_varies_in_eval",
        "active_power_single_point_distribution",
        "raw_throughput_mean_bps",
        "p05_throughput_bps",
        "served_ratio",
        "outage_ratio",
        "handover_count",
        "r2_mean",
        "load_balance_metric",
        "episode_scalar_reward_diagnostic_mean",
        "budget_violation_count",
        "per_beam_power_violation_count",
        "inactive_beam_nonzero_power_step_count",
        "overflow_steps",
        "overflow_user_count",
        "sticky_override_count",
        "nonsticky_move_count",
        "qos_guard_reject_count",
        "handover_guard_reject_count",
        "reasons",
    ]
    return _write_csv(path, rows, fieldnames)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serializable = {
                key: row.get(key) if not isinstance(row.get(key), (list, dict)) else row.get(key)
                for key in fieldnames
            }
            writer.writerow(serializable)
    return path


def _write_execution_report(path: Path, summary: dict[str, Any]) -> Path:
    acceptance = summary["acceptance"]
    primary = summary["role_summaries"][PRIMARY_ROLE]
    aggregate_by_role = {
        row["role"]: row for row in summary["aggregate_pass_fail_table"]
    }
    primary_row = aggregate_by_role[PRIMARY_ROLE]
    lines = [
        "# HOBS Active-TX EE QoS-Sticky Robustness Gate Execution Report",
        "",
        "**Date:** `2026-05-01`",
        f"**Status:** `{acceptance['status']}`",
        "**Scope:** bounded robustness / mechanism-attribution gate only. This",
        "report does not authorize EE-MODQN effectiveness, physical energy",
        "saving, HOBS optimizer, Catfish repair, RA-EE association, Phase `03C`",
        "continuation, or frozen baseline mutation.",
        "",
        "## Protocol",
        "",
        "- Roles: `matched-control`, `primary-qos-sticky`, `no-qos-guard-ablation`, `stricter-qos-ablation`, `threshold-sensitivity-45`, `threshold-sensitivity-55`.",
        "- Seed triplets: `[42, 1337, 7]`, `[43, 1338, 8]`, `[44, 1339, 9]`.",
        "- Evaluation seeds: `[100, 200, 300, 400, 500]`.",
        "- Episode budget: `5` per role / seed triplet.",
        "- Scalar reward is diagnostic only.",
        "",
        "## Matched Boundary Proof",
        "",
        f"`matched_boundary_pass={summary['boundary_proof']['matched_boundary_pass']}`.",
        "All roles use HOBS active-TX EE `r1`, the same HOBS-inspired DPC sidecar,",
        "the same training seed triplets, the same evaluation seeds, and the",
        "same bounded trainer / checkpoint protocol. Only the opt-in",
        "anti-collapse role or ablation knob differs.",
        "",
        "## Primary Aggregate Metrics",
        "",
        f"- `all_evaluated_steps_one_active_beam`: `{primary['all_evaluated_steps_one_active_beam']}`",
        f"- `active_beam_count_distribution`: `{primary['active_beam_count_distribution']}`",
        f"- `denominator_varies_in_eval`: `{primary['denominator_varies_in_eval']}`",
        f"- `active_power_single_point_distribution`: `{primary['active_power_single_point_distribution']}`",
        f"- `distinct_total_active_power_w_values`: `{primary['distinct_total_active_power_w_values']}`",
        f"- `power_control_activity_rate`: `{primary['power_control_activity_rate']}`",
        f"- `throughput_vs_ee_pearson`: `{primary['throughput_vs_ee_pearson']}`",
        f"- `same_policy_throughput_vs_ee_rescore_ranking_change`: `{primary['same_policy_throughput_vs_ee_rescore_ranking_change']}`",
        f"- `raw_throughput_mean_bps`: `{primary['raw_throughput_mean_bps']}`",
        f"- `p05_throughput_bps`: `{primary['p05_throughput_bps']}`",
        f"- `p05_throughput_ratio_vs_control`: `{primary_row['p05_throughput_ratio_vs_control']}`",
        f"- `served_ratio_delta`: `{primary_row['served_ratio_delta']}`",
        f"- `outage_ratio_delta`: `{primary_row['outage_ratio_delta']}`",
        f"- `handover_delta`: `{primary_row['handover_delta']}`",
        f"- `r2_delta`: `{primary_row['r2_delta']}`",
        f"- `budget/per-beam/inactive-power violations`: `{primary['budget_violation_count']}/{primary['per_beam_power_violation_count']}/{primary['inactive_beam_nonzero_power_step_count']}`",
        f"- `overflow/sticky/nonsticky/qos-reject/handover-reject`: `{primary['overflow_steps']}/{primary['sticky_override_count']}/{primary['nonsticky_move_count']}/{primary['qos_guard_reject_count']}/{primary['handover_guard_reject_count']}`",
        "",
        "## Aggregate Pass / Fail",
        "",
        "| Role | Status | p05 ratio | handover delta | r2 delta | one-beam seeds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["aggregate_pass_fail_table"]:
        lines.append(
            f"| `{row['role']}` | `{row['status']}` | "
            f"`{row.get('p05_throughput_ratio_vs_control')}` | "
            f"`{row.get('handover_delta')}` | `{row.get('r2_delta')}` | "
            f"`{row.get('per_seed_one_active_beam_count')}` |"
        )
    lines.extend([
        "",
        "## Mechanism Attribution",
        "",
    ])
    lines.extend(f"- {note}" for note in summary["mechanism_attribution"]["notes"])
    lines.extend([
        "",
        "## Acceptance Result",
        "",
        f"`PASS / BLOCK / NEEDS MORE DESIGN: {acceptance['status']}`",
        "",
        "## Artifacts",
        "",
        "- `artifacts/hobs-active-tx-ee-qos-sticky-robustness-*/`",
        "- `artifacts/hobs-active-tx-ee-qos-sticky-robustness-summary/summary.json`",
        "- `artifacts/hobs-active-tx-ee-qos-sticky-robustness-summary/per_seed_pass_fail_table.csv`",
        "- `artifacts/hobs-active-tx-ee-qos-sticky-robustness-summary/aggregate_pass_fail_table.csv`",
        "",
        "## Forbidden Claims Still Active",
        "",
    ])
    lines.extend(f"- {claim}" for claim in summary["forbidden_claims_still_active"])
    lines.extend([
        "",
        "## Deviations / Blockers",
        "",
    ])
    if summary["deviations_or_blockers"]:
        lines.extend(f"- {item}" for item in summary["deviations_or_blockers"])
    else:
        lines.append("- None within the bounded robustness protocol.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


__all__ = [
    "ARTIFACT_ROOT",
    "PRIMARY_ROLE",
    "REPORT_PATH",
    "ROLE_CONFIGS",
    "ROLE_ORDER",
    "prove_robustness_boundary",
    "run_qos_sticky_robustness_gate",
    "summarize_qos_sticky_robustness_runs",
]
