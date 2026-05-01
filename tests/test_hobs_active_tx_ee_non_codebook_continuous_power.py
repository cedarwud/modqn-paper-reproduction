from __future__ import annotations

import numpy as np
import pytest
from numpy.random import default_rng

from modqn_paper_reproduction.analysis.hobs_active_tx_ee_non_codebook_continuous_power_boundary_audit import (
    DEFAULT_CANDIDATE_CONFIG,
    DEFAULT_CONTROL_CONFIG,
    deterministic_step_wiring_samples,
    prove_candidate_control_boundary,
)
from modqn_paper_reproduction.config_loader import (
    ConfigValidationError,
    build_environment,
    build_power_surface_config,
    build_trainer_config,
    load_training_yaml,
)
from modqn_paper_reproduction.env.beam import BeamConfig
from modqn_paper_reproduction.env.channel import ChannelConfig
from modqn_paper_reproduction.env.orbit import OrbitConfig
from modqn_paper_reproduction.env.step import (
    HOBS_POWER_SURFACE_NON_CODEBOOK_CONTINUOUS_POWER,
    HOBS_POWER_SURFACE_STATIC_CONFIG,
    PowerSurfaceConfig,
    StepConfig,
    StepEnvironment,
)


def _continuous_power_config(
    *,
    lo: float = 0.05,
    hi: float = 0.25,
) -> PowerSurfaceConfig:
    return PowerSurfaceConfig(
        hobs_power_surface_mode=HOBS_POWER_SURFACE_NON_CODEBOOK_CONTINUOUS_POWER,
        inactive_beam_policy="zero-w",
        continuous_p_active_lo_w=lo,
        continuous_p_active_hi_w=hi,
        continuous_alpha=0.85,
        continuous_beta=0.35,
        continuous_kappa=0.60,
        continuous_bias=-2.0,
        continuous_q_ref=0.0,
        continuous_n_qos=50.0,
        max_power_w=hi,
        total_power_budget_w=8.0,
    )


def _small_continuous_env(
    *,
    lo: float = 0.05,
    hi: float = 0.25,
) -> StepEnvironment:
    return StepEnvironment(
        step_config=StepConfig(num_users=5, user_lat_deg=0.0, user_lon_deg=0.0),
        orbit_config=OrbitConfig(num_satellites=2),
        beam_config=BeamConfig(),
        channel_config=ChannelConfig(),
        power_surface_config=_continuous_power_config(lo=lo, hi=hi),
    )


def _mixed_valid_actions(masks) -> np.ndarray:
    first = int(np.flatnonzero(masks[0].mask)[0])
    second = int(np.flatnonzero(masks[0].mask)[1])
    return np.array([first, first, first, second, second], dtype=np.int32)


def _rotating_valid_actions(masks) -> np.ndarray:
    actions = np.zeros(len(masks), dtype=np.int32)
    for uid, mask in enumerate(masks):
        valid = np.flatnonzero(mask.mask)
        actions[uid] = int(valid[uid % int(valid.size)])
    return actions


def _one_step(env: StepEnvironment, actions_builder=_mixed_valid_actions):
    rng = default_rng(123)
    mobility_rng = default_rng(456)
    _states, masks, _diag = env.reset(rng, mobility_rng)
    return env.step(actions_builder(masks), rng)


def test_baseline_static_config_is_unchanged_when_continuous_power_disabled() -> None:
    env = StepEnvironment(
        step_config=StepConfig(num_users=5, user_lat_deg=0.0, user_lon_deg=0.0),
        orbit_config=OrbitConfig(num_satellites=2),
        beam_config=BeamConfig(),
        channel_config=ChannelConfig(tx_power_w=2.0),
        power_surface_config=PowerSurfaceConfig(),
    )
    result = _one_step(env)

    assert env.power_surface_config.hobs_power_surface_mode == HOBS_POWER_SURFACE_STATIC_CONFIG
    assert result.selected_power_profile == HOBS_POWER_SURFACE_STATIC_CONFIG
    assert np.allclose(result.beam_transmit_power_w, 2.0)


def test_config_namespace_gates_continuous_power_mode() -> None:
    candidate_cfg = load_training_yaml(DEFAULT_CANDIDATE_CONFIG)
    power_cfg = build_power_surface_config(candidate_cfg)
    trainer_cfg = build_trainer_config(candidate_cfg)

    assert power_cfg.hobs_power_surface_mode == HOBS_POWER_SURFACE_NON_CODEBOOK_CONTINUOUS_POWER
    assert trainer_cfg.comparison_role == "ee-candidate"

    bad_cfg = dict(candidate_cfg)
    bad_cfg["track"] = dict(candidate_cfg["track"])
    bad_cfg["track"]["phase"] = "baseline"
    with pytest.raises(ConfigValidationError):
        build_power_surface_config(bad_cfg)


def test_continuous_power_formula_outputs_non_codebook_values_and_zero_inactive_beams() -> None:
    result = _one_step(_small_continuous_env())
    active_power = result.beam_transmit_power_w[result.active_beam_mask]
    inactive_power = result.beam_transmit_power_w[~result.active_beam_mask]

    assert result.selected_power_profile == ""
    assert active_power.size == 2
    assert len({round(float(value), 12) for value in active_power}) > 1
    assert not any(round(float(value), 12) in {0.5, 1.0, 2.0} for value in active_power)
    assert np.allclose(inactive_power, 0.0)


def test_active_beams_stay_within_per_beam_and_total_power_bounds() -> None:
    cfg = _continuous_power_config()
    result = _one_step(_small_continuous_env())
    active_power = result.beam_transmit_power_w[result.active_beam_mask]

    assert np.all(active_power >= cfg.continuous_p_active_lo_w)
    assert np.all(active_power <= cfg.continuous_p_active_hi_w)
    assert result.total_active_beam_power_w <= float(cfg.total_power_budget_w)
    assert not result.power_budget_violation


def test_continuous_power_feeds_throughput_before_hobs_ee_reward() -> None:
    low_power = _one_step(_small_continuous_env(lo=0.02, hi=0.05))
    high_power = _one_step(_small_continuous_env(lo=0.10, hi=0.25))

    low_throughput = sum(float(rw.r1_throughput) for rw in low_power.rewards)
    high_throughput = sum(float(rw.r1_throughput) for rw in high_power.rewards)
    assert high_throughput > low_throughput

    expected_ee = high_throughput / (high_power.total_active_beam_power_w + 1e-9)
    assert high_power.rewards[0].r1_hobs_active_tx_ee == pytest.approx(expected_ee)


def test_policy_action_consequences_change_continuous_power() -> None:
    env_a = _small_continuous_env()
    env_b = _small_continuous_env()
    first_case = _one_step(env_a, actions_builder=_mixed_valid_actions)
    rotating_case = _one_step(env_b, actions_builder=_rotating_valid_actions)

    assert first_case.beam_transmit_power_w.tolist() != rotating_case.beam_transmit_power_w.tolist()


def test_candidate_control_boundary_metadata_has_only_r1_as_intended_difference() -> None:
    control_cfg = load_training_yaml(DEFAULT_CONTROL_CONFIG)
    candidate_cfg = load_training_yaml(DEFAULT_CANDIDATE_CONFIG)
    proof = prove_candidate_control_boundary(control_cfg, candidate_cfg)

    assert proof["matched_boundary_pass"] is True
    assert proof["checks"]["only_intended_difference_is_r1_reward_mode"] is True
    assert proof["checks"]["same_continuous_power_surface"] is True
    assert proof["checks"]["same_anti_collapse_guard"] is True
    assert proof["checks"]["qos_sticky_guard_shared_and_structural_only"] is True
    assert proof["checks"]["forbidden_modes_disabled"] is True


def test_boundary_audit_samples_prove_same_power_vector_and_forbidden_absences() -> None:
    control_cfg = load_training_yaml(DEFAULT_CONTROL_CONFIG)
    candidate_cfg = load_training_yaml(DEFAULT_CANDIDATE_CONFIG)
    samples = deterministic_step_wiring_samples(control_cfg, candidate_cfg)

    assert samples["same_power_vector_for_candidate_and_control"] is True
    assert samples["same_throughput_for_candidate_and_control"] is True
    assert samples["ee_denominator_reuses_step_power_vector"] is True
    assert samples["policy_action_consequences_change_power"] is True
    assert samples["active_power_nonconstant"] is True
    assert samples["active_power_non_codebook"] is True
    assert samples["selected_power_profile_absent"] is True
    assert samples["inactive_beams_zero_w"] is True
    assert samples["power_budget_violations"] == 0


def test_configs_do_not_enable_training_forbidden_modes_or_profiles() -> None:
    for path in (DEFAULT_CONTROL_CONFIG, DEFAULT_CANDIDATE_CONFIG):
        cfg = load_training_yaml(path)
        trainer = build_trainer_config(cfg)
        power = cfg["resolved_assumptions"]["hobs_power_surface"]["value"]

        assert trainer.catfish_enabled is False
        assert trainer.catfish_phase05b_variant == "not-applicable"
        assert trainer.training_experiment_kind.endswith("implementation-readiness")
        assert "power_codebook_profile" not in power
        assert "power_codebook_levels_w" not in power
        assert "selected_power_profile" not in power
        assert power["finite_codebook_levels_absent"] is True
        assert power["selected_power_profile_absent"] is True
        assert build_environment(cfg).power_surface_config.hobs_power_surface_mode == (
            HOBS_POWER_SURFACE_NON_CODEBOOK_CONTINUOUS_POWER
        )
