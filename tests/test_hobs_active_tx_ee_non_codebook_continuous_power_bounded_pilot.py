from __future__ import annotations

import copy

import pytest

from modqn_paper_reproduction.analysis.hobs_active_tx_ee_non_codebook_continuous_power_bounded_pilot import (
    EE_CANDIDATE_ROLE,
    ROLE_CONFIGS,
    THROUGHPUT_CONTROL_ROLE,
    prove_bounded_pilot_boundary,
    summarize_bounded_pilot_runs,
)
from modqn_paper_reproduction.config_loader import (
    ConfigValidationError,
    build_trainer_config,
    load_training_yaml,
)
from modqn_paper_reproduction.runtime.trainer_spec import (
    HOBS_ACTIVE_TX_EE_NON_CODEBOOK_CONTINUOUS_POWER_BOUNDED_PILOT_KIND,
    R1_REWARD_MODE_HOBS_ACTIVE_TX_EE,
    R1_REWARD_MODE_THROUGHPUT,
)


def test_bounded_pilot_configs_load_and_prove_matched_boundary() -> None:
    proof = prove_bounded_pilot_boundary()
    trainers = {
        role: build_trainer_config(load_training_yaml(path))
        for role, path in ROLE_CONFIGS.items()
    }

    assert proof["matched_boundary_pass"] is True
    assert proof["checks"]["only_intended_difference_is_r1_reward_mode"] is True
    assert proof["checks"]["same_continuous_power_surface"] is True
    assert proof["checks"]["same_qos_sticky_guard"] is True
    assert proof["checks"]["same_seed_triplets"] is True
    assert proof["checks"]["same_eval_seeds"] is True
    assert proof["checks"]["finite_codebook_levels_absent"] is True
    assert proof["checks"]["selected_power_profile_absent"] is True
    assert proof["checks"]["forbidden_modes_disabled"] is True
    assert proof["seed_triplets"] == [
        [42, 1337, 7],
        [43, 1338, 8],
        [44, 1339, 9],
    ]
    assert proof["evaluation_seed_set"] == [100, 200, 300, 400, 500]
    assert proof["episode_budget"] == 5
    assert (
        trainers[THROUGHPUT_CONTROL_ROLE].training_experiment_kind
        == HOBS_ACTIVE_TX_EE_NON_CODEBOOK_CONTINUOUS_POWER_BOUNDED_PILOT_KIND
    )
    assert trainers[THROUGHPUT_CONTROL_ROLE].r1_reward_mode == R1_REWARD_MODE_THROUGHPUT
    assert trainers[EE_CANDIDATE_ROLE].r1_reward_mode == R1_REWARD_MODE_HOBS_ACTIVE_TX_EE


def test_bounded_pilot_namespace_rejects_wrong_phase() -> None:
    cfg = load_training_yaml(ROLE_CONFIGS[EE_CANDIDATE_ROLE])
    bad = copy.deepcopy(cfg)
    bad["track"]["phase"] = "hobs-active-tx-ee-qos-sticky-broader-effectiveness"

    with pytest.raises(ConfigValidationError):
        load_training_yaml_from_dict_for_test(bad)


def load_training_yaml_from_dict_for_test(cfg):
    from modqn_paper_reproduction.config_loader import build_power_surface_config

    build_power_surface_config(cfg)
    return build_trainer_config(cfg)


def _diag(**overrides):
    base = {
        "steps_evaluated": 50,
        "denominator_varies_in_eval": True,
        "all_evaluated_steps_one_active_beam": False,
        "active_beam_count_distribution": {"7.0": 50},
        "total_active_power_distribution": {"1.55": 25, "1.56": 25},
        "selected_power_profile_distribution": {"": 50},
        "selected_power_profile_absent": True,
        "active_power_single_point_distribution": False,
        "distinct_total_active_power_w_values": [1.55, 1.56],
        "distinct_active_power_w_values": [0.21, 0.22, 0.23],
        "power_control_activity_rate": 1.0,
        "continuous_power_activity_rate": 1.0,
        "throughput_vs_ee_pearson": 0.5,
        "throughput_vs_ee_spearman": 0.4,
        "same_policy_throughput_vs_ee_rescore_ranking_change": True,
        "EE_system": 900.0,
        "EE_system_eval_aggregate": 900.0,
        "EE_system_step_mean": 900.0,
        "eta_EE_active_TX": 900.0,
        "eta_EE_active_TX_step_mean": 900.0,
        "raw_throughput_mean_bps": 7000.0,
        "raw_episode_throughput_mean_bps": 70000.0,
        "p05_throughput_bps": 15.0,
        "served_ratio": 1.0,
        "outage_ratio": 0.0,
        "handover_count": 200,
        "r2_mean": -0.2,
        "load_balance_metric": -40.0,
        "r3_mean": -40.0,
        "scalar_reward_diagnostic_mean": 10.0,
        "episode_scalar_reward_diagnostic_mean": 100.0,
        "budget_violation_count": 0,
        "per_beam_power_violation_count": 0,
        "inactive_beam_nonzero_power_step_count": 0,
        "overflow_steps": 50,
        "overflow_user_count": 2000,
        "sticky_override_count": 1000,
        "nonsticky_move_count": 0,
        "qos_guard_reject_count": 0,
        "handover_guard_reject_count": 100,
    }
    base.update(overrides)
    return base


def _row(role: str, seed_index: int, **diag_overrides):
    return {
        "role": role,
        "seed_index": seed_index,
        "seed_triplet": [42 + seed_index, 1337 + seed_index, 7 + seed_index],
        "run_dir": f"artifacts/test/{role}/{seed_index}",
        "config_path": "unused",
        "summary_path": "unused",
        "summary": {"diagnostics": _diag(**diag_overrides)},
    }


def test_bounded_pilot_summary_passes_only_when_candidate_beats_matched_control() -> None:
    boundary = prove_bounded_pilot_boundary()
    rows = []
    for seed_index in range(3):
        rows.append(
            _row(
                THROUGHPUT_CONTROL_ROLE,
                seed_index,
                EE_system=880.0,
                EE_system_step_mean=880.0,
                p05_throughput_bps=14.0,
                handover_count=210,
                r2_mean=-0.21,
            )
        )
        rows.append(_row(EE_CANDIDATE_ROLE, seed_index))

    summary = summarize_bounded_pilot_runs(rows, boundary)

    assert summary["status"] == "PASS"
    assert summary["acceptance"]["criteria"]["scalar_reward_success_basis"] is False
    assert (
        summary["aggregate_comparison"]["candidate_vs_control_EE_system_delta"]
        > 0.0
    )


def test_bounded_pilot_blocks_scalar_only_candidate() -> None:
    boundary = prove_bounded_pilot_boundary()
    rows = []
    for seed_index in range(3):
        rows.append(_row(THROUGHPUT_CONTROL_ROLE, seed_index, EE_system=900.0))
        rows.append(
            _row(
                EE_CANDIDATE_ROLE,
                seed_index,
                EE_system=899.0,
                episode_scalar_reward_diagnostic_mean=110.0,
            )
        )

    summary = summarize_bounded_pilot_runs(rows, boundary)

    assert summary["status"] == "BLOCK"
    assert summary["acceptance"]["scalar_reward_success_basis"] is True
    assert "candidate wins only scalar reward" in summary["stop_conditions_triggered"]
