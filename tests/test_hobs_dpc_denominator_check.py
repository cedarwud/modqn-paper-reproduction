"""Route D tiny matched DPC denominator-check tests."""

from __future__ import annotations

import copy

from modqn_paper_reproduction.analysis.hobs_dpc_denominator_check import (
    CANDIDATE_CONFIG,
    CONTROL_CONFIG,
    interpret_route_d_verdict,
    prove_matched_boundary,
)
from modqn_paper_reproduction.config_loader import (
    build_environment,
    build_trainer_config,
    load_training_yaml,
)
from modqn_paper_reproduction.env.step import HOBS_POWER_SURFACE_DPC_SIDECAR
from modqn_paper_reproduction.runtime.trainer_spec import (
    R1_REWARD_MODE_HOBS_ACTIVE_TX_EE,
    R1_REWARD_MODE_THROUGHPUT,
)


def _summary(**overrides):
    diagnostics = {
        "denominator_varies_in_eval": True,
        "all_evaluated_steps_one_active_beam": False,
        "active_power_single_point_distribution": False,
        "throughput_proxy_risk_flag": False,
        "throughput_vs_ee_pearson": 0.7,
        "raw_throughput_mean_bps": 1000.0,
        "p05_throughput_bps": 100.0,
        "served_ratio": 1.0,
        "handover_count": 10,
        "episode_scalar_reward_diagnostic_mean": 5.0,
    }
    diagnostics.update(overrides)
    return {"diagnostics": diagnostics}


def test_route_d_control_and_candidate_configs_are_gated() -> None:
    control_cfg = load_training_yaml(CONTROL_CONFIG)
    candidate_cfg = load_training_yaml(CANDIDATE_CONFIG)
    control_trainer = build_trainer_config(control_cfg)
    candidate_trainer = build_trainer_config(candidate_cfg)
    control_env = build_environment(control_cfg)
    candidate_env = build_environment(candidate_cfg)

    assert control_trainer.r1_reward_mode == R1_REWARD_MODE_THROUGHPUT
    assert candidate_trainer.r1_reward_mode == R1_REWARD_MODE_HOBS_ACTIVE_TX_EE
    assert control_trainer.episodes == candidate_trainer.episodes == 5
    assert (
        control_env.power_surface_config.hobs_power_surface_mode
        == candidate_env.power_surface_config.hobs_power_surface_mode
        == HOBS_POWER_SURFACE_DPC_SIDECAR
    )
    assert (
        control_trainer.checkpoint_secondary_report
        == candidate_trainer.checkpoint_secondary_report
        == "best-weighted-reward-on-eval"
    )


def test_route_d_matched_boundary_proves_same_dpc_eval_and_checkpoint() -> None:
    proof = prove_matched_boundary(CONTROL_CONFIG, CANDIDATE_CONFIG)
    assert proof["matched_boundary_pass"] is True
    assert proof["checks"]["same_dpc_sidecar"] is True
    assert proof["checks"]["same_seed_block"] is True
    assert proof["checks"]["same_checkpoint_rule"] is True
    assert proof["checks"]["same_episode_budget"] is True
    assert proof["checks"]["tiny_episode_budget"] is True
    assert proof["control"]["r1_reward_mode"] == R1_REWARD_MODE_THROUGHPUT
    assert proof["candidate"]["r1_reward_mode"] == R1_REWARD_MODE_HOBS_ACTIVE_TX_EE


def test_route_d_verdict_passes_when_denominator_and_qos_guards_pass() -> None:
    boundary = {"matched_boundary_pass": True}
    verdict = interpret_route_d_verdict(
        _summary(),
        _summary(
            raw_throughput_mean_bps=990.0,
            p05_throughput_bps=98.0,
            served_ratio=1.0,
            episode_scalar_reward_diagnostic_mean=4.0,
        ),
        boundary,
    )
    assert verdict["route_d_status"] == "PASS"
    assert verdict["candidate_denominator_diagnostics_pass"] is True
    assert verdict["material_throughput_collapse"] is False


def test_route_d_verdict_blocks_denominator_collapse() -> None:
    boundary = {"matched_boundary_pass": True}
    candidate = _summary(
        denominator_varies_in_eval=False,
        active_power_single_point_distribution=True,
        throughput_proxy_risk_flag=True,
        episode_scalar_reward_diagnostic_mean=6.0,
    )
    verdict = interpret_route_d_verdict(_summary(), candidate, boundary)
    assert verdict["route_d_status"] == "BLOCK"
    assert any("denominator_varies_in_eval=false" in r for r in verdict["reasons"])
    assert verdict["scalar_only_improvement_without_denominator"] is True


def test_route_d_verdict_blocks_near_throughput_proxy_correlation() -> None:
    boundary = {"matched_boundary_pass": True}
    verdict = interpret_route_d_verdict(
        _summary(),
        _summary(throughput_vs_ee_pearson=0.99),
        boundary,
    )
    assert verdict["route_d_status"] == "BLOCK"
    assert "throughput_vs_ee_pearson > 0.95" in verdict["reasons"]


def test_route_d_verdict_blocks_material_p05_or_served_collapse() -> None:
    boundary = {"matched_boundary_pass": True}
    verdict = interpret_route_d_verdict(
        _summary(p05_throughput_bps=100.0, served_ratio=1.0),
        _summary(p05_throughput_bps=80.0, served_ratio=0.99),
        boundary,
    )
    assert verdict["route_d_status"] == "BLOCK"
    assert verdict["material_throughput_collapse"] is True


def test_route_d_verdict_needs_more_design_when_boundary_unproven() -> None:
    boundary = copy.deepcopy(prove_matched_boundary(CONTROL_CONFIG, CANDIDATE_CONFIG))
    boundary["matched_boundary_pass"] = False
    verdict = interpret_route_d_verdict(_summary(), _summary(), boundary)
    assert verdict["route_d_status"] == "NEEDS MORE DESIGN"
    assert "candidate/control boundary cannot be proven matched" in verdict["reasons"]
