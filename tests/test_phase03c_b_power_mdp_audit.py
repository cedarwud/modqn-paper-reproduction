from __future__ import annotations

import numpy as np

from modqn_paper_reproduction.analysis.phase03c_b_power_mdp_audit import (
    export_phase03c_b_power_mdp_audit,
)
from modqn_paper_reproduction.config_loader import build_environment, load_training_yaml
from modqn_paper_reproduction.env.beam import BeamConfig
from modqn_paper_reproduction.env.channel import ChannelConfig
from modqn_paper_reproduction.env.orbit import OrbitConfig
from modqn_paper_reproduction.env.step import (
    HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK,
    HOBS_POWER_SURFACE_STATIC_CONFIG,
    PowerSurfaceConfig,
    StepConfig,
    StepEnvironment,
)


PHASE03C_B_CONFIG = "configs/ee-modqn-phase-03c-b-power-mdp-audit.resolved.yaml"


def _small_codebook_env(profile: str = "fixed-mid") -> StepEnvironment:
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
            total_power_budget_w=4.0,
        ),
    )


def test_phase03c_b_power_codebook_is_opt_in() -> None:
    baseline_cfg = load_training_yaml("configs/modqn-paper-baseline.resolved-template.yaml")
    baseline_env = build_environment(baseline_cfg)

    phase_cfg = load_training_yaml(PHASE03C_B_CONFIG)
    phase_env = build_environment(phase_cfg)

    assert baseline_env.power_surface_config.hobs_power_surface_mode == (
        HOBS_POWER_SURFACE_STATIC_CONFIG
    )
    assert phase_env.power_surface_config.hobs_power_surface_mode == (
        HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK
    )
    assert phase_env.power_surface_config.power_codebook_profile == "fixed-mid"


def test_phase03c_b_power_codebook_inactive_beams_are_zero_w() -> None:
    env = _small_codebook_env("fixed-mid")
    rng = np.random.default_rng(42)
    env.reset(rng)
    result = env.step(np.zeros(5, dtype=np.int32), rng)

    assert bool(result.active_beam_mask[0])
    assert np.count_nonzero(result.active_beam_mask) == 1
    assert result.beam_transmit_power_w[0] == 1.0
    np.testing.assert_allclose(result.beam_transmit_power_w[1:], 0.0)


def test_phase03c_b_power_codebook_active_power_respects_bounds() -> None:
    env = _small_codebook_env("fixed-high")
    rng = np.random.default_rng(42)
    env.reset(rng)
    result = env.step(np.array([0, 1, 2, 3, 4], dtype=np.int32), rng)
    active_power = result.beam_transmit_power_w[result.active_beam_mask]

    assert active_power.size == 5
    assert np.all(active_power > 0.0)
    assert np.all(active_power <= env.power_surface_config.max_power_w)


def test_phase03c_b_budget_violation_detection_and_trim(tmp_path) -> None:
    outputs = export_phase03c_b_power_mdp_audit(
        PHASE03C_B_CONFIG,
        tmp_path / "phase03c-b-audit",
        include_learned=False,
        evaluation_seed_set=(100,),
        max_steps=2,
        policies=("spread-valid",),
        power_semantics=("fixed-low", "fixed-high", "budget-trim"),
    )
    summaries = {
        row["power_semantics"]: row
        for row in outputs["summary"]["candidate_summaries"]
    }

    assert summaries["fixed-high"]["budget_violations"]["step_count"] > 0
    assert summaries["budget-trim"]["budget_violations"]["step_count"] == 0
    assert summaries["fixed-low"]["denominator_varies_in_eval"] is False
    assert outputs["summary"]["denominator_variability_result"][
        "fixed_denominator_audit_catches_fixed_denominator"
    ] is True


def test_phase03c_b_audit_artifacts_include_required_fields(tmp_path) -> None:
    outputs = export_phase03c_b_power_mdp_audit(
        PHASE03C_B_CONFIG,
        tmp_path / "phase03c-b-audit",
        include_learned=False,
        evaluation_seed_set=(100,),
        max_steps=2,
        policies=("hold-current", "spread-valid"),
        power_semantics=("fixed-2w", "phase-02b-proxy", "fixed-mid", "budget-trim"),
    )

    assert outputs["phase03c_b_power_mdp_audit_summary"].exists()
    assert outputs["phase03c_b_power_mdp_step_metrics"].exists()
    assert outputs["phase03c_b_power_mdp_candidate_summary"].exists()
    assert outputs["review_md"].exists()

    candidate = outputs["summary"]["candidate_summaries"][0]
    for field in (
        "EE_system_aggregate_bps_per_w",
        "EE_system_step_mean_bps_per_w",
        "active_beam_count_distribution",
        "total_active_beam_power_w_distribution",
        "beam_power_vector_distinct_count",
        "denominator_varies_in_eval",
        "one_active_beam_step_ratio",
        "throughput_mean_user_step_bps",
        "throughput_p05_user_step_bps",
        "served_ratio",
        "outage_ratio",
        "handover_count",
        "budget_violations",
        "throughput_vs_EE_system_correlation",
    ):
        assert field in candidate

    ranking = outputs["summary"]["ranking_separation_result"]
    assert "ranking_checks" in ranking
    assert "ranking_separates_under_same_policy_rescore" in ranking
