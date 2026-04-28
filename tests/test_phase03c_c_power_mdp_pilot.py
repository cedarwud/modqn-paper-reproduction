from __future__ import annotations

import numpy as np

from modqn_paper_reproduction.analysis.phase03c_c_power_mdp_pilot import (
    export_phase03c_c_power_mdp_paired_validation,
)
from modqn_paper_reproduction.cli import train_main
from modqn_paper_reproduction.config_loader import (
    build_environment,
    build_trainer_config,
    load_training_yaml,
)
from modqn_paper_reproduction.env.beam import BeamConfig
from modqn_paper_reproduction.env.channel import ChannelConfig
from modqn_paper_reproduction.env.orbit import OrbitConfig
from modqn_paper_reproduction.env.step import (
    HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK,
    HOBS_POWER_SURFACE_STATIC_CONFIG,
    POWER_CODEBOOK_RUNTIME_SELECTOR_PROFILE,
    PowerSurfaceConfig,
    StepConfig,
    StepEnvironment,
)


BASELINE_CONFIG = "configs/modqn-paper-baseline.resolved-template.yaml"
CONTROL_CONFIG = "configs/ee-modqn-phase-03c-c-power-mdp-control.resolved.yaml"
CANDIDATE_CONFIG = "configs/ee-modqn-phase-03c-c-power-mdp-candidate.resolved.yaml"


def _small_codebook_env(
    *,
    profile: str,
    budget_w: float = 4.0,
) -> StepEnvironment:
    return StepEnvironment(
        step_config=StepConfig(num_users=5, user_lat_deg=0.0, user_lon_deg=0.0),
        orbit_config=OrbitConfig(num_satellites=2),
        beam_config=BeamConfig(),
        channel_config=ChannelConfig(tx_power_w=2.0),
        power_surface_config=PowerSurfaceConfig(
            hobs_power_surface_mode=HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK,
            inactive_beam_policy="zero-w",
            power_codebook_profile=profile,
            power_codebook_levels_w=(0.5, 1.0, 2.0),
            max_power_w=2.0,
            total_power_budget_w=budget_w,
        ),
    )


def test_phase03c_c_configs_are_opt_in_and_leave_baseline_defaults() -> None:
    baseline_cfg = load_training_yaml(BASELINE_CONFIG)
    baseline_env = build_environment(baseline_cfg)
    baseline_trainer = build_trainer_config(baseline_cfg)

    control_cfg = load_training_yaml(CONTROL_CONFIG)
    candidate_cfg = load_training_yaml(CANDIDATE_CONFIG)
    control_env = build_environment(control_cfg)
    candidate_env = build_environment(candidate_cfg)
    control_trainer = build_trainer_config(control_cfg)
    candidate_trainer = build_trainer_config(candidate_cfg)

    assert baseline_env.power_surface_config.hobs_power_surface_mode == (
        HOBS_POWER_SURFACE_STATIC_CONFIG
    )
    assert baseline_trainer.training_experiment_kind == "baseline"
    assert baseline_trainer.r1_reward_mode == "throughput"
    assert baseline_trainer.objective_weights == (0.5, 0.3, 0.2)

    assert control_trainer.training_experiment_kind == "phase-03c-c-power-mdp-pilot"
    assert candidate_trainer.training_experiment_kind == "phase-03c-c-power-mdp-pilot"
    assert control_trainer.phase == candidate_trainer.phase == "phase-03c-c"
    assert control_trainer.episodes == candidate_trainer.episodes == 20
    assert control_trainer.objective_weights == candidate_trainer.objective_weights
    assert control_trainer.r1_reward_mode == "throughput"
    assert candidate_trainer.r1_reward_mode == "per-user-beam-ee-credit"

    assert control_env.power_surface_config.power_codebook_profile == "fixed-mid"
    assert candidate_env.power_surface_config.power_codebook_profile == (
        POWER_CODEBOOK_RUNTIME_SELECTOR_PROFILE
    )
    assert control_env.power_surface_config.total_power_budget_w == (
        candidate_env.power_surface_config.total_power_budget_w
    )
    assert (
        control_cfg["resolved_assumptions"]["phase_03c_c_power_mdp_pilot"]["value"][
            "adaptive_power_decision"
        ]
        == "disabled"
    )
    assert (
        candidate_cfg["resolved_assumptions"]["phase_03c_c_power_mdp_pilot"]["value"][
            "adaptive_power_decision"
        ]
        == "enabled"
    )


def test_phase03c_c_control_has_no_adaptive_power_decision() -> None:
    env = _small_codebook_env(profile="fixed-mid")
    rng = np.random.default_rng(42)
    env.reset(rng)
    result = env.step(np.arange(5, dtype=np.int32), rng)
    active_power = result.beam_transmit_power_w[result.active_beam_mask]

    assert result.selected_power_profile == "fixed-mid"
    np.testing.assert_allclose(active_power, 1.0)


def test_phase03c_c_candidate_logs_selected_power_decisions() -> None:
    env = _small_codebook_env(
        profile=POWER_CODEBOOK_RUNTIME_SELECTOR_PROFILE,
        budget_w=8.0,
    )
    rng = np.random.default_rng(42)
    env.reset(rng)

    concentrated = env.step(np.zeros(5, dtype=np.int32), rng)
    spread = env.step(np.arange(5, dtype=np.int32), rng)

    assert concentrated.selected_power_profile == "fixed-low"
    assert spread.selected_power_profile == "budget-trim"
    assert concentrated.total_active_beam_power_w > 0.0
    assert spread.total_active_beam_power_w > 0.0
    assert concentrated.selected_power_profile != spread.selected_power_profile


def test_phase03c_c_budget_violation_detection_works() -> None:
    env = _small_codebook_env(profile="fixed-high", budget_w=4.0)
    rng = np.random.default_rng(42)
    env.reset(rng)
    result = env.step(np.arange(5, dtype=np.int32), rng)

    assert result.total_active_beam_power_w == 10.0
    assert result.power_budget_violation is True
    assert result.power_budget_excess_w == 6.0


def test_phase03c_c_comparison_summary_includes_required_fields(tmp_path) -> None:
    control_run = tmp_path / "control"
    candidate_run = tmp_path / "candidate"
    comparison_dir = tmp_path / "comparison"

    assert train_main(
        [
            "--config",
            CONTROL_CONFIG,
            "--episodes",
            "1",
            "--progress-every",
            "0",
            "--output-dir",
            str(control_run),
        ]
    ) == 0
    assert train_main(
        [
            "--config",
            CANDIDATE_CONFIG,
            "--episodes",
            "1",
            "--progress-every",
            "0",
            "--output-dir",
            str(candidate_run),
        ]
    ) == 0

    outputs = export_phase03c_c_power_mdp_paired_validation(
        control_run_dir=control_run,
        candidate_run_dir=candidate_run,
        output_dir=comparison_dir,
        evaluation_seed_set=(100,),
    )
    summary = outputs["summary"]
    primary = summary["primary_comparison"]
    candidate = next(
        row
        for row in summary["checkpoint_summaries"]
        if row["method"] == "candidate"
        and row["checkpoint_role"] == primary["checkpoint_role"]
    )

    assert outputs["phase03c_c_power_mdp_summary"].exists()
    assert outputs["phase03c_c_power_mdp_step_metrics"].exists()
    assert outputs["phase03c_c_power_mdp_checkpoint_summary"].exists()
    for field in (
        "EE_system_aggregate_bps_per_w",
        "EE_system_step_mean_bps_per_w",
        "throughput_mean_user_step_bps",
        "throughput_p05_user_step_bps",
        "served_ratio",
        "outage_ratio",
        "handover_count",
        "r2_mean",
        "r3_mean",
        "denominator_varies_in_eval",
        "one_active_beam_step_ratio",
        "selected_profile_distinct_count",
        "throughput_vs_EE_system_correlation",
        "budget_violations",
    ):
        assert field in candidate
    assert "acceptance_gate" in primary
    assert "ranking_check" in primary
    assert "phase_03c_c_decision" in primary
