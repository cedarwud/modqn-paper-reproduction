"""Tests for the QoS-sticky HOBS active-TX EE robustness gate."""

from __future__ import annotations

from modqn_paper_reproduction.analysis.hobs_active_tx_ee_qos_sticky_robustness import (
    PRIMARY_ROLE,
    ROLE_CONFIGS,
    ROLE_ORDER,
    prove_robustness_boundary,
    summarize_qos_sticky_robustness_runs,
)
from modqn_paper_reproduction.config_loader import (
    build_trainer_config,
    load_training_yaml,
)


def test_qos_sticky_robustness_configs_load_and_preserve_matched_boundary() -> None:
    proof = prove_robustness_boundary()

    assert proof["matched_boundary_pass"] is True
    assert proof["checks"]["required_roles_present"] is True
    assert proof["checks"]["all_r1_are_hobs_active_tx_ee"] is True
    assert proof["checks"]["same_dpc_sidecar"] is True
    assert proof["checks"]["same_seed_triplets_declared"] is True
    assert proof["checks"]["at_least_three_seed_triplets"] is True
    assert proof["checks"]["no_forced_min_active_beams_target"] is True
    assert proof["checks"]["all_roles_disable_nonsticky_moves"] is True
    assert proof["episode_budget"] == 5
    assert proof["seed_triplets"] == [
        [42, 1337, 7],
        [43, 1338, 8],
        [44, 1339, 9],
    ]


def test_qos_sticky_robustness_ablation_knobs_are_labeled() -> None:
    configs = {
        role: build_trainer_config(load_training_yaml(path))
        for role, path in ROLE_CONFIGS.items()
    }

    assert not configs["matched-control"].anti_collapse_action_constraint_enabled
    assert configs[PRIMARY_ROLE].anti_collapse_action_constraint_enabled
    assert configs[PRIMARY_ROLE].anti_collapse_overload_threshold_users_per_beam == 50
    assert configs[PRIMARY_ROLE].anti_collapse_qos_ratio_min == 0.95
    assert configs["no-qos-guard-ablation"].anti_collapse_qos_ratio_min == 0.01
    assert configs["stricter-qos-ablation"].anti_collapse_qos_ratio_min == 1.05
    assert (
        configs["threshold-sensitivity-45"]
        .anti_collapse_overload_threshold_users_per_beam
        == 45
    )
    assert (
        configs["threshold-sensitivity-55"]
        .anti_collapse_overload_threshold_users_per_beam
        == 55
    )
    assert all(
        not cfg.anti_collapse_allow_nonsticky_moves for cfg in configs.values()
    )


def _row(role: str, seed_index: int, **diag_overrides):
    base_diag = {
        "steps_evaluated": 50,
        "denominator_varies_in_eval": True,
        "all_evaluated_steps_one_active_beam": False,
        "active_beam_count_distribution": {"7.0": 50},
        "total_active_power_distribution": {"7.5": 10, "8.0": 40},
        "active_power_single_point_distribution": False,
        "distinct_total_active_power_w_values": [7.5, 8.0],
        "distinct_active_power_w_values": [1.0, 1.1],
        "power_control_activity_rate": 1.0,
        "throughput_vs_ee_pearson": 0.5,
        "same_policy_throughput_vs_ee_rescore_ranking_change": True,
        "raw_throughput_mean_bps": 1000.0,
        "raw_episode_throughput_mean_bps": 10000.0,
        "p05_throughput_bps": 100.0,
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
        "overflow_steps": 10,
        "overflow_user_count": 100,
        "sticky_override_count": 80,
        "nonsticky_move_count": 0,
        "qos_guard_reject_count": 0,
        "handover_guard_reject_count": 20,
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


def test_qos_sticky_robustness_summary_blocks_threshold_fragility() -> None:
    boundary = prove_robustness_boundary()
    rows = []
    for seed_index in range(3):
        rows.append(_row("matched-control", seed_index, active_beam_count_distribution={"1.0": 50}, p05_throughput_bps=100.0))
        for role in ROLE_ORDER:
            if role == "matched-control":
                continue
            overrides = {}
            if role == "threshold-sensitivity-55":
                overrides = {
                    "p05_throughput_bps": 50.0,
                    "handover_count": 200,
                    "r2": -0.2,
                    "r2_mean": -0.2,
                }
            rows.append(_row(role, seed_index, **overrides))

    summary = summarize_qos_sticky_robustness_runs(rows, boundary)

    assert summary["acceptance"]["primary_aggregate_status"] == "PASS"
    assert summary["mechanism_attribution"]["threshold_fragility_detected"] is True
    assert summary["status"] == "NEEDS MORE DESIGN"
