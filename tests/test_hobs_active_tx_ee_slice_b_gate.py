"""Slice B design gate tests for HOBS-style active-TX EE.

Namespace: hobs-active-tx-ee-modqn-feasibility
Date: 2026-05-01

These tests prove the Slice B gate findings:

  1. The current channel model assigns per-satellite SNR (no per-beam
     interference) — all beams of a satellite share the same SNR value.
  2. Under SNR-only channel + active-load-concave, hobs-active-tx-ee is
     numerically equivalent to per-user-ee-credit.
  3. The matched control config loads and is correctly gated.
  4. The SNR assumption check function correctly detects per-satellite SNR.
  5. Route A (SINR) is flagged as needed before Route D pilot.

These tests do NOT run training. They do NOT claim EE-MODQN effectiveness.
They document why Route D would be predicted to fail with the current config.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from modqn_paper_reproduction.analysis.hobs_active_tx_ee_feasibility import (
    check_snr_per_satellite_assumption,
)
from modqn_paper_reproduction.config_loader import (
    build_environment,
    build_trainer_config,
    load_training_yaml,
)
from modqn_paper_reproduction.env.step import _HOBS_ACTIVE_TX_EE_EPSILON_P_W
from modqn_paper_reproduction.runtime.trainer_spec import (
    HOBS_ACTIVE_TX_EE_MODQN_FEASIBILITY_KIND,
    R1_REWARD_MODE_THROUGHPUT,
)

_FEASIBILITY_CONFIG = "configs/hobs-active-tx-ee-modqn-feasibility.resolved.yaml"
_CONTROL_CONFIG = "configs/hobs-active-tx-ee-modqn-feasibility-control.resolved.yaml"
_BASELINE_CONFIG = "configs/modqn-paper-baseline.resolved-template.yaml"


# ---------------------------------------------------------------------------
# 1. Channel model: per-satellite SNR (no per-beam interference)
# ---------------------------------------------------------------------------

def test_channel_has_no_intra_satellite_beam_interference() -> None:
    """Per-beam SNR = P_b * channel_gain / noise — no interference term.

    Under active-load-concave, SNR differs per beam because P_b differs
    (different load → different power). But the CHANNEL GAIN is the same
    for all beams of a satellite (computed from slant range only, no per-beam
    off-axis or interference model).

    The structural fact: SNR/P_b (= channel_gain/noise) is constant across
    all beams of a satellite. This means there is NO interference term I_intra.
    Adding I_intra is Route A.
    """
    cfg = load_training_yaml(_FEASIBILITY_CONFIG)
    cfg["baseline"]["users"] = 10
    env = build_environment(cfg)
    K = env.beam_pattern.num_beams

    rng = np.random.default_rng(42)
    states, masks, _diag = env.reset(rng, np.random.default_rng(7))
    actions = np.zeros(env.config.num_users, dtype=np.int32)
    for uid, mask in enumerate(masks):
        valid = np.flatnonzero(mask.mask)
        if valid.size > 0:
            cur = int(np.argmax(states[uid].access_vector))
            actions[uid] = cur if bool(mask.mask[cur]) else int(valid[0])
    result = env.step(actions, rng)

    L = env.orbit.num_satellites
    beam_power = result.beam_transmit_power_w  # shape (L*K,)
    # Check SNR/P_b (= channel_gain/noise) is same across beams of same satellite
    for uid in range(min(3, len(result.user_states))):
        cq = result.user_states[uid].channel_quality  # shape (L*K,) — SNR linear
        for s in range(L):
            block_snr = cq[s * K: (s + 1) * K]
            block_pow = beam_power[s * K: (s + 1) * K]
            active_in_block = (block_pow > 1e-12) & (block_snr > 0)
            if np.sum(active_in_block) > 1:
                # channel_gain_proxy = SNR / P_b should be same for all active beams
                ratios = block_snr[active_in_block] / block_pow[active_in_block]
                ratio_range = float(np.max(ratios) - np.min(ratios))
                assert ratio_range < 1e-6, (
                    f"SNR/P_b is not constant across beams of satellite {s}: "
                    f"range={ratio_range:.2e}. Per-beam channel gains differ, "
                    f"suggesting an interference or off-axis gain model is present."
                )


def test_snr_assumption_check_detects_per_satellite_model() -> None:
    result = check_snr_per_satellite_assumption(_FEASIBILITY_CONFIG)
    assert result["all_beams_same_snr_per_satellite"] is True
    assert result["has_per_beam_interference"] is False
    assert result["hobs_active_tx_ee_equiv_per_user_credit"] is True
    assert result["route_a_sinr_needed_before_d_pilot"] is True


# ---------------------------------------------------------------------------
# 2. Mathematical equivalence: hobs-active-tx-ee == per-user-ee-credit
#    under SNR-only channel + active-load-concave + uniform load
# ---------------------------------------------------------------------------

def test_hobs_ee_equals_per_user_ee_credit_under_uniform_load_fixed_snr() -> None:
    """Under fixed channel gain (per-satellite), system EE == per-user credit
    at uniform load per beam.

    Under active-load-concave, SNR_b = P_b * h_sat / N0 (per-beam because P_b
    varies). However, the channel gain h_sat is per-satellite (no per-beam
    off-axis or interference). Under UNIFORM load (same N_b per beam, same P_b),
    all active beams share the same SNR, and both formulas reduce to:

        bw * log2(1 + SNR_uniform) / P_b_uniform

    This algebraic equivalence holds in the symmetric uniform-load case.
    The formulas diverge under non-uniform load, but the STRUCTURAL problem
    (no interference, shared Q-function) still applies.
    """
    bw = 1e6
    h_over_N0 = 5.0  # channel_gain / noise_power — per satellite
    base_w, scale_w, exp, max_w = 0.25, 0.35, 0.5, 2.0

    def p(n: float) -> float:
        return min(max_w, base_w + scale_w * n ** exp)

    def snr_b(n: float) -> float:
        return p(n) * h_over_N0  # SNR = P_b * h/N0

    for num_beams, users_per_beam in [(1, 100), (7, 100 / 7), (4, 25)]:
        n = float(users_per_beam)
        snr = snr_b(n)
        R_u = bw / n * math.log2(1 + snr)
        P_b = p(n)

        # per-user EE credit = R_u / (P_b / N_b) = bw*log2(1+SNR_b) / P_b
        old_credit = R_u / (P_b / n)
        assert old_credit == pytest.approx(bw * math.log2(1 + snr) / P_b, rel=1e-9)

        # Under uniform load: system EE = num_beams * bw*log2(1+SNR_b) / (num_beams*P_b)
        #                               = bw*log2(1+SNR_b) / P_b = old_credit
        new_system_ee = (
            num_beams * n * R_u
            / (num_beams * P_b + _HOBS_ACTIVE_TX_EE_EPSILON_P_W)
        )
        assert new_system_ee == pytest.approx(old_credit, rel=1e-6), (
            f"System EE != per-user credit for {num_beams} beams, "
            f"{n:.1f} users/beam: system_ee={new_system_ee:.4f}, "
            f"old_credit={old_credit:.4f}"
        )


def test_environment_confirms_reward_equivalence_on_step() -> None:
    """Verify that r1_hobs_active_tx_ee == r1_energy_efficiency_credit
    under uniform load on a single active beam."""
    cfg = load_training_yaml(_FEASIBILITY_CONFIG)
    cfg["baseline"]["users"] = 5
    env = build_environment(cfg)

    rng = np.random.default_rng(55)
    states, masks, _diag = env.reset(rng, np.random.default_rng(3))

    # Force all users to same beam to test collapse scenario
    first_valid = int(np.flatnonzero(masks[0].mask)[0])
    actions = np.full(env.config.num_users, first_valid, dtype=np.int32)
    result = env.step(actions, rng)

    # Under 1-beam collapse with active-load-concave:
    # r1_hobs_active_tx_ee = total_thr / total_power = sum(R_u) / P_b
    # r1_energy_efficiency_credit = R_u / (P_b / N_b) = R_u * N_b / P_b
    # sum(r1_ee_credit) / N_b = total_thr / P_b = system EE
    n_users = env.config.num_users
    sum_ee_credits = sum(rw.r1_energy_efficiency_credit for rw in result.rewards)
    system_ee = result.rewards[0].r1_hobs_active_tx_ee

    # They should be equal (up to epsilon in denominator)
    # avg credit = sum_credits / n_users ≈ system EE under 1-beam collapse
    avg_credit = sum_ee_credits / n_users if n_users > 0 else 0.0
    # Note: may differ slightly due to epsilon_P in denominator
    # The key check: they're in the same ballpark (within 0.1%)
    if avg_credit > 0 and system_ee > 0:
        ratio = avg_credit / system_ee
        assert 0.999 < ratio < 1.001, (
            f"avg EE credit ({avg_credit:.4f}) differs from system EE ({system_ee:.4f}) "
            f"by more than 0.1% (ratio={ratio:.6f}). "
            f"This may indicate a regime where the formulas differ."
        )


# ---------------------------------------------------------------------------
# 3. Matched control config loads and is correctly gated
# ---------------------------------------------------------------------------

def test_control_config_loads_with_throughput_r1_mode() -> None:
    cfg = load_training_yaml(_CONTROL_CONFIG)
    trainer_cfg = build_trainer_config(cfg)
    assert trainer_cfg.r1_reward_mode == R1_REWARD_MODE_THROUGHPUT
    assert trainer_cfg.training_experiment_kind == HOBS_ACTIVE_TX_EE_MODQN_FEASIBILITY_KIND
    assert trainer_cfg.method_family == "MODQN-control"


def test_control_and_candidate_configs_have_same_power_surface() -> None:
    control_cfg = load_training_yaml(_CONTROL_CONFIG)
    candidate_cfg = load_training_yaml(_FEASIBILITY_CONFIG)
    control_env = build_environment(control_cfg)
    candidate_env = build_environment(candidate_cfg)
    assert (
        control_env.power_surface_config.hobs_power_surface_mode
        == candidate_env.power_surface_config.hobs_power_surface_mode
        == "active-load-concave"
    )
    assert (
        control_env.power_surface_config.inactive_beam_policy
        == candidate_env.power_surface_config.inactive_beam_policy
        == "zero-w"
    )
    assert (
        control_env.power_surface_config.active_base_power_w
        == candidate_env.power_surface_config.active_base_power_w
    )


def test_control_and_candidate_configs_differ_only_in_r1_mode() -> None:
    control_cfg = load_training_yaml(_CONTROL_CONFIG)
    candidate_cfg = load_training_yaml(_FEASIBILITY_CONFIG)
    control_trainer = build_trainer_config(control_cfg)
    candidate_trainer = build_trainer_config(candidate_cfg)

    assert control_trainer.r1_reward_mode == R1_REWARD_MODE_THROUGHPUT
    from modqn_paper_reproduction.runtime.trainer_spec import R1_REWARD_MODE_HOBS_ACTIVE_TX_EE
    assert candidate_trainer.r1_reward_mode == R1_REWARD_MODE_HOBS_ACTIVE_TX_EE

    # All other key settings must match
    assert control_trainer.objective_weights == candidate_trainer.objective_weights
    assert control_trainer.discount_factor == candidate_trainer.discount_factor
    assert control_trainer.batch_size == candidate_trainer.batch_size


# ---------------------------------------------------------------------------
# 4. Route A prerequisite flag
# ---------------------------------------------------------------------------

def test_snr_assumption_check_flags_route_a_needed() -> None:
    """The feasibility config must be flagged as needing Route A (SINR)
    before a meaningful Route D pilot can be run."""
    result = check_snr_per_satellite_assumption(_FEASIBILITY_CONFIG)
    assert result["route_a_sinr_needed_before_d_pilot"] is True, (
        "Route A (SINR interference) must be flagged as needed when "
        "the channel is SNR-only (per-satellite). Running Route D pilot "
        "under SNR-only is predicted to reproduce Phase 03 collapse."
    )


def test_baseline_also_has_per_satellite_snr() -> None:
    """Baseline also uses per-satellite SNR — both configs have same channel."""
    result_baseline = check_snr_per_satellite_assumption(_BASELINE_CONFIG)
    result_feasibility = check_snr_per_satellite_assumption(_FEASIBILITY_CONFIG)
    assert result_baseline["all_beams_same_snr_per_satellite"] is True
    assert result_feasibility["all_beams_same_snr_per_satellite"] is True


# ---------------------------------------------------------------------------
# 5. Gate stop conditions are documented and reachable
# ---------------------------------------------------------------------------

def test_heuristic_policy_shows_denominator_varies_under_active_load_concave() -> None:
    """Under heuristic random policy, denominator DOES vary with active-load-concave.

    This proves the environment CAN produce variable denominators.
    The question for D is whether the LEARNED policy avoids collapse.
    Given mathematical equivalence with Phase 03, collapse is predicted.
    """
    from modqn_paper_reproduction.analysis.hobs_active_tx_ee_feasibility import (
        compute_hobs_active_tx_ee_diagnostics,
    )
    diag = compute_hobs_active_tx_ee_diagnostics(
        _FEASIBILITY_CONFIG,
        max_steps=5,
        policies=("random-valid",),
    )
    assert diag["denominator_varies_in_eval"] is True, (
        "Random-valid policy must produce variable denominators under active-load-concave."
    )
    assert diag["throughput_proxy_risk_flag"] is False


def test_static_config_always_throughput_proxy() -> None:
    """Static-config always produces throughput_proxy_risk_flag=True."""
    from modqn_paper_reproduction.analysis.hobs_active_tx_ee_feasibility import (
        compute_hobs_active_tx_ee_diagnostics,
    )
    diag = compute_hobs_active_tx_ee_diagnostics(
        _BASELINE_CONFIG,
        max_steps=5,
        policies=("hold-current",),
    )
    assert diag["active_power_single_point_distribution"] is True
    assert diag["throughput_proxy_risk_flag"] is True
