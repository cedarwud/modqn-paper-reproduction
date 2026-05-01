"""Tests for the QoS-sticky HOBS active-TX EE broader-effectiveness gate."""

from __future__ import annotations

from modqn_paper_reproduction.analysis.hobs_active_tx_ee_qos_sticky_broader_effectiveness import (
    ANTI_COLLAPSE_THROUGHPUT_CONTROL_ROLE,
    HOBS_EE_NO_ANTI_COLLAPSE_ROLE,
    MATCHED_THROUGHPUT_CONTROL_ROLE,
    QOS_STICKY_EE_CANDIDATE_ROLE,
    ROLE_CONFIGS,
    ROLE_ORDER,
    prove_broader_effectiveness_boundary,
    summarize_broader_effectiveness_runs,
)
from modqn_paper_reproduction.config_loader import (
    build_trainer_config,
    load_training_yaml,
)
from modqn_paper_reproduction.runtime.trainer_spec import (
    HOBS_ACTIVE_TX_EE_QOS_STICKY_BROADER_EFFECTIVENESS_KIND,
    R1_REWARD_MODE_HOBS_ACTIVE_TX_EE,
    R1_REWARD_MODE_THROUGHPUT,
)


def test_broader_effectiveness_configs_load_and_preserve_boundary() -> None:
    proof = prove_broader_effectiveness_boundary()
    trainers = {
        role: build_trainer_config(load_training_yaml(path))
        for role, path in ROLE_CONFIGS.items()
    }

    assert proof["matched_boundary_pass"] is True
    assert proof["checks"]["required_roles_present"] is True
    assert proof["checks"]["same_dpc_sidecar"] is True
    assert proof["checks"]["same_seed_triplets_declared"] is True
    assert proof["checks"]["throughput_controls_are_dpc_matched_not_frozen_baseline"]
    assert proof["episode_budget"] == 5
    assert proof["seed_triplets"] == [
        [42, 1337, 7],
        [43, 1338, 8],
        [44, 1339, 9],
    ]
    assert all(
        cfg.training_experiment_kind
        == HOBS_ACTIVE_TX_EE_QOS_STICKY_BROADER_EFFECTIVENESS_KIND
        for cfg in trainers.values()
    )
    assert (
        trainers[MATCHED_THROUGHPUT_CONTROL_ROLE].r1_reward_mode
        == R1_REWARD_MODE_THROUGHPUT
    )
    assert (
        trainers[ANTI_COLLAPSE_THROUGHPUT_CONTROL_ROLE].r1_reward_mode
        == R1_REWARD_MODE_THROUGHPUT
    )
    assert (
        trainers[HOBS_EE_NO_ANTI_COLLAPSE_ROLE].r1_reward_mode
        == R1_REWARD_MODE_HOBS_ACTIVE_TX_EE
    )
    assert (
        trainers[QOS_STICKY_EE_CANDIDATE_ROLE].r1_reward_mode
        == R1_REWARD_MODE_HOBS_ACTIVE_TX_EE
    )
    assert not trainers[MATCHED_THROUGHPUT_CONTROL_ROLE].anti_collapse_action_constraint_enabled
    assert not trainers[HOBS_EE_NO_ANTI_COLLAPSE_ROLE].anti_collapse_action_constraint_enabled
    assert trainers[QOS_STICKY_EE_CANDIDATE_ROLE].anti_collapse_action_constraint_enabled
    assert trainers[ANTI_COLLAPSE_THROUGHPUT_CONTROL_ROLE].anti_collapse_action_constraint_enabled


def _row(role: str, seed_index: int, **diag_overrides):
    is_anti = role in {
        QOS_STICKY_EE_CANDIDATE_ROLE,
        ANTI_COLLAPSE_THROUGHPUT_CONTROL_ROLE,
    }
    base_diag = {
        "steps_evaluated": 50,
        "denominator_varies_in_eval": True,
        "all_evaluated_steps_one_active_beam": not is_anti,
        "active_beam_count_distribution": {"7.0": 50} if is_anti else {"1.0": 50},
        "total_active_power_distribution": {"8.0": 25, "8.1": 25},
        "active_power_single_point_distribution": False,
        "distinct_total_active_power_w_values": [8.0, 8.1],
        "distinct_active_power_w_values": [1.0, 1.1],
        "power_control_activity_rate": 1.0,
        "throughput_vs_ee_pearson": 0.5,
        "same_policy_throughput_vs_ee_rescore_ranking_change": True,
        "eta_EE_active_TX": 20.0,
        "eta_EE_active_TX_eval_aggregate": 20.0,
        "eta_EE_active_TX_step_mean": 20.0,
        "EE_system": 20.0,
        "EE_system_eval_aggregate": 20.0,
        "EE_system_step_mean": 20.0,
        "raw_throughput_mean_bps": 1000.0,
        "raw_episode_throughput_mean_bps": 10000.0,
        "p05_throughput_bps": 10.0,
        "served_ratio": 1.0,
        "outage_ratio": 0.0,
        "handover_count": 100,
        "r2": -0.1,
        "r2_mean": -0.1,
        "load_balance_metric": -10.0,
        "r3_mean": -10.0,
        "active_beam_load_gap_mean": 10.0,
        "scalar_reward_diagnostic_mean": 10.0,
        "episode_scalar_reward_diagnostic_mean": 100.0,
        "budget_violation_count": 0,
        "per_beam_power_violation_count": 0,
        "inactive_beam_nonzero_power_step_count": 0,
        "overflow_steps": 50 if is_anti else 0,
        "overflow_user_count": 1000 if is_anti else 0,
        "sticky_override_count": 800 if is_anti else 0,
        "nonsticky_move_count": 0,
        "qos_guard_reject_count": 0,
        "handover_guard_reject_count": 200 if is_anti else 0,
        "dpc_sign_flip_count": 10,
        "dpc_step_count": 50,
        "dpc_qos_guard_count": 0,
        "dpc_per_beam_cap_clip_count": 0,
        "dpc_sat_cap_clip_count": 0,
    }
    base_diag.update(diag_overrides)
    return {
        "role": role,
        "seed_index": seed_index,
        "seed_triplet": [42 + seed_index, 1337 + seed_index, 7 + seed_index],
        "run_dir": f"artifacts/test/{role}/{seed_index}",
        "config_path": "unused",
        "summary_path": "unused",
        "summary": {"diagnostics": base_diag},
    }


def test_broader_effectiveness_blocks_ee_when_anti_collapse_control_explains_gain() -> None:
    boundary = prove_broader_effectiveness_boundary()
    rows = []
    for seed_index in range(3):
        rows.append(_row(MATCHED_THROUGHPUT_CONTROL_ROLE, seed_index))
        rows.append(_row(HOBS_EE_NO_ANTI_COLLAPSE_ROLE, seed_index))
        rows.append(
            _row(
                QOS_STICKY_EE_CANDIDATE_ROLE,
                seed_index,
                EE_system=90.0,
                eta_EE_active_TX=90.0,
                raw_throughput_mean_bps=7000.0,
                p05_throughput_bps=30.0,
                handover_count=80,
                r2=-0.08,
                r2_mean=-0.08,
                load_balance_metric=-5.0,
            )
        )
        rows.append(
            _row(
                ANTI_COLLAPSE_THROUGHPUT_CONTROL_ROLE,
                seed_index,
                EE_system=90.0,
                eta_EE_active_TX=90.0,
                raw_throughput_mean_bps=7000.0,
                p05_throughput_bps=30.0,
                handover_count=80,
                r2=-0.08,
                r2_mean=-0.08,
                load_balance_metric=-5.0,
            )
        )

    summary = summarize_broader_effectiveness_runs(rows, boundary)

    assert summary["anti_collapse_mechanism_verdict"]["status"] == "PASS"
    assert summary["ee_objective_contribution_verdict"]["status"] == "BLOCK"
    assert (
        "anti-collapse-throughput-control explains all gains"
        in summary["stop_conditions_triggered"]
    )
    assert summary["status"] == "BLOCK"
