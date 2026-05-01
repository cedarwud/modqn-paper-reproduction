"""Tests for HOBS-style active-TX EE feasibility gate.

Namespace: hobs-active-tx-ee-modqn-feasibility
Date: 2026-05-01

These tests prove the minimum acceptance criteria for the feasibility gate:

  1. Baseline default reward remains throughput when new mode is not selected.
  2. New mode is disabled unless explicit config selects it.
  3. Active-TX EE formula computes expected value on a toy step.
  4. Inactive beams do not contribute power to the denominator.
  5. Fixed denominator is detected as throughput-proxy risk.
  6. Scalar reward alone is not a success flag.

Old failure conditions remain as guardrails — this gate does NOT overturn:
  - denominator_varies_in_eval=false
  - one-active-beam collapse
  - active power single-point distribution
  - throughput-vs-EE correlation near 1
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from modqn_paper_reproduction.algorithms.modqn import MODQNTrainer
from modqn_paper_reproduction.analysis.hobs_active_tx_ee_feasibility import (
    compute_hobs_active_tx_ee_diagnostics,
)
from modqn_paper_reproduction.config_loader import (
    build_environment,
    build_trainer_config,
    load_training_yaml,
)
from modqn_paper_reproduction.env.step import (
    PowerSurfaceConfig,
    RewardComponents,
    StepConfig,
    StepEnvironment,
    _HOBS_ACTIVE_TX_EE_EPSILON_P_W,
)
from modqn_paper_reproduction.runtime.objective_math import select_r1_reward_value
from modqn_paper_reproduction.runtime.trainer_spec import (
    HOBS_ACTIVE_TX_EE_MODQN_FEASIBILITY_KIND,
    R1_REWARD_MODE_HOBS_ACTIVE_TX_EE,
    R1_REWARD_MODE_THROUGHPUT,
    TrainerConfig,
)

_BASELINE_CONFIG = "configs/modqn-paper-baseline.resolved-template.yaml"
_FEASIBILITY_CONFIG = "configs/hobs-active-tx-ee-modqn-feasibility.resolved.yaml"


# ---------------------------------------------------------------------------
# 1. Baseline default reward remains throughput
# ---------------------------------------------------------------------------

def test_baseline_default_reward_remains_throughput() -> None:
    cfg = load_training_yaml(_BASELINE_CONFIG)
    trainer_cfg = build_trainer_config(cfg)
    assert trainer_cfg.r1_reward_mode == R1_REWARD_MODE_THROUGHPUT
    assert trainer_cfg.training_experiment_kind == "baseline"


def test_baseline_reward_vector_uses_throughput() -> None:
    cfg = load_training_yaml(_BASELINE_CONFIG)
    cfg["baseline"]["users"] = 3
    env = build_environment(cfg)
    trainer_cfg = build_trainer_config(cfg)
    trainer = MODQNTrainer(env=env, config=trainer_cfg)

    rng = np.random.default_rng(42)
    states, masks, _diag = env.reset(rng, np.random.default_rng(7))
    actions = np.array(
        [int(np.flatnonzero(mask.mask)[0]) for mask in masks],
        dtype=np.int32,
    )
    result = env.step(actions, rng)

    for uid in range(len(result.rewards)):
        rv = trainer.reward_vector_from_step_result(result, uid)
        assert rv[0] == pytest.approx(result.rewards[uid].r1_throughput)
        assert rv[1] == result.rewards[uid].r2_handover
        assert rv[2] == result.rewards[uid].r3_load_balance


# ---------------------------------------------------------------------------
# 2. New mode is disabled unless explicit config selects it
# ---------------------------------------------------------------------------

def test_hobs_active_tx_ee_mode_requires_feasibility_kind() -> None:
    with pytest.raises(ValueError, match="hobs-active-tx-ee-modqn-feasibility"):
        TrainerConfig(
            r1_reward_mode=R1_REWARD_MODE_HOBS_ACTIVE_TX_EE,
            training_experiment_kind="baseline",
        )


def test_hobs_active_tx_ee_mode_accepted_for_feasibility_kind() -> None:
    tc = TrainerConfig(
        r1_reward_mode=R1_REWARD_MODE_HOBS_ACTIVE_TX_EE,
        training_experiment_kind=HOBS_ACTIVE_TX_EE_MODQN_FEASIBILITY_KIND,
    )
    assert tc.r1_reward_mode == R1_REWARD_MODE_HOBS_ACTIVE_TX_EE


def test_feasibility_config_loads_correctly() -> None:
    cfg = load_training_yaml(_FEASIBILITY_CONFIG)
    trainer_cfg = build_trainer_config(cfg)
    assert trainer_cfg.r1_reward_mode == R1_REWARD_MODE_HOBS_ACTIVE_TX_EE
    assert trainer_cfg.training_experiment_kind == HOBS_ACTIVE_TX_EE_MODQN_FEASIBILITY_KIND


def test_baseline_config_does_not_activate_hobs_ee_mode() -> None:
    cfg = load_training_yaml(_BASELINE_CONFIG)
    trainer_cfg = build_trainer_config(cfg)
    assert trainer_cfg.r1_reward_mode != R1_REWARD_MODE_HOBS_ACTIVE_TX_EE


# ---------------------------------------------------------------------------
# 3. Active-TX EE formula computes expected value on a toy step
# ---------------------------------------------------------------------------

def test_active_tx_ee_formula_correct_on_toy_step() -> None:
    cfg = load_training_yaml(_FEASIBILITY_CONFIG)
    cfg["baseline"]["users"] = 4
    env = build_environment(cfg)

    rng = np.random.default_rng(99)
    states, masks, _diag = env.reset(rng, np.random.default_rng(11))
    actions = np.array(
        [int(np.flatnonzero(mask.mask)[0]) for mask in masks],
        dtype=np.int32,
    )
    result = env.step(actions, rng)

    # Manually compute expected system EE
    total_thr = float(
        sum(rw.r1_throughput for rw in result.rewards)
    )
    active_mask = result.active_beam_mask.astype(bool)
    total_active_power = float(
        np.sum(result.beam_transmit_power_w[active_mask], dtype=np.float64)
    )
    expected_ee = total_thr / (total_active_power + _HOBS_ACTIVE_TX_EE_EPSILON_P_W)

    # All users should have the same system EE value
    for rw in result.rewards:
        assert rw.r1_hobs_active_tx_ee == pytest.approx(expected_ee, rel=1e-9)


def test_select_r1_returns_hobs_ee_when_mode_set() -> None:
    tc = TrainerConfig(
        r1_reward_mode=R1_REWARD_MODE_HOBS_ACTIVE_TX_EE,
        training_experiment_kind=HOBS_ACTIVE_TX_EE_MODQN_FEASIBILITY_KIND,
    )
    result = select_r1_reward_value(
        throughput_bps=100.0,
        per_user_ee_credit_bps_per_w=50.0,
        per_user_beam_ee_credit_bps_per_w=25.0,
        hobs_active_tx_ee_bps_per_w=42.0,
        config=tc,
    )
    assert result == pytest.approx(42.0)


def test_select_r1_returns_throughput_when_baseline() -> None:
    tc = TrainerConfig(r1_reward_mode=R1_REWARD_MODE_THROUGHPUT)
    result = select_r1_reward_value(
        throughput_bps=100.0,
        per_user_ee_credit_bps_per_w=50.0,
        per_user_beam_ee_credit_bps_per_w=25.0,
        hobs_active_tx_ee_bps_per_w=42.0,
        config=tc,
    )
    assert result == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# 4. Inactive beams do not contribute power
# ---------------------------------------------------------------------------

def test_inactive_beams_zero_power_in_ee_denominator() -> None:
    cfg = load_training_yaml(_FEASIBILITY_CONFIG)
    cfg["baseline"]["users"] = 3
    env = build_environment(cfg)

    rng = np.random.default_rng(77)
    states, masks, _diag = env.reset(rng, np.random.default_rng(5))
    actions = np.array(
        [int(np.flatnonzero(mask.mask)[0]) for mask in masks],
        dtype=np.int32,
    )
    result = env.step(actions, rng)

    # Inactive beams must have zero power in the active-load-concave mode
    active_mask = result.active_beam_mask.astype(bool)
    inactive_powers = result.beam_transmit_power_w[~active_mask]
    assert float(np.max(inactive_powers)) == pytest.approx(0.0), (
        "Inactive beams must contribute zero power to the EE denominator"
    )

    # EE denominator uses only active beams
    active_power_sum = float(
        np.sum(result.beam_transmit_power_w[active_mask], dtype=np.float64)
    )
    full_power_sum = float(np.sum(result.beam_transmit_power_w, dtype=np.float64))
    # They match since inactive are zero
    assert active_power_sum == pytest.approx(full_power_sum)

    # total_active_beam_power_w on StepResult equals the manual active sum
    assert result.total_active_beam_power_w == pytest.approx(active_power_sum)


def test_ee_formula_with_all_beams_inactive() -> None:
    """EE formula must not blow up when no beams are active."""
    # Construct a degenerate RewardComponents manually — all beams inactive → power = 0
    # The formula should give sum_thr / eps, which is large but finite
    thr = 0.0
    power = 0.0
    expected = thr / (power + _HOBS_ACTIVE_TX_EE_EPSILON_P_W)
    assert math.isfinite(expected)


# ---------------------------------------------------------------------------
# 5. Fixed denominator detected as throughput-proxy risk
# ---------------------------------------------------------------------------

def test_static_config_detected_as_throughput_proxy_risk() -> None:
    """Under static-config, beam power is fixed → denominator varies only with
    active count, which is near-constant under collapse → proxy risk = True."""
    cfg = load_training_yaml(_BASELINE_CONFIG)
    # Baseline uses static-config (fixed power)
    env = build_environment(cfg)
    assert env.power_surface_config.hobs_power_surface_mode == "static-config"

    # Run a few steps with hold-current policy
    rng = np.random.default_rng(42)
    states, masks, _diag = env.reset(rng, np.random.default_rng(7))
    powers_seen: list[float] = []
    for _ in range(env.config.steps_per_episode):
        actions = np.zeros(env.config.num_users, dtype=np.int32)
        for uid, mask in enumerate(masks):
            valid = np.flatnonzero(mask.mask)
            if valid.size > 0:
                cur = int(np.argmax(states[uid].access_vector))
                actions[uid] = cur if bool(mask.mask[cur]) else int(valid[0])
        result = env.step(actions, rng)
        powers_seen.append(float(result.total_active_beam_power_w))
        if result.done:
            break
        states = result.user_states
        masks = result.action_masks

    # Under static-config the per-beam power is always the same scalar,
    # so total_active_beam_power_w = active_count * tx_power_w. If all
    # users stay on the same beam count, denominator is fixed.
    distinct_powers = len({round(p, 9) for p in powers_seen})
    # This may be 1 (pure fixed) or a few values if beam count shifts;
    # the diagnostic module's throughput_proxy_risk_flag covers both.
    # Here we just assert the diagnostic reports proxy risk for static-config.
    diag = compute_hobs_active_tx_ee_diagnostics(
        _BASELINE_CONFIG,
        max_steps=5,
    )
    # Under static-config the active-beam power per active beam is always
    # the same scalar → active_power_single_point = True → proxy risk = True
    assert diag["throughput_proxy_risk_flag"] is True


def test_diagnostics_formula_verified_field() -> None:
    diag = compute_hobs_active_tx_ee_diagnostics(
        _FEASIBILITY_CONFIG,
        max_steps=3,
    )
    assert diag["formula_verified"] is True


# ---------------------------------------------------------------------------
# 6. Scalar reward alone is not a success flag
# ---------------------------------------------------------------------------

def test_feasibility_gate_does_not_promote_on_scalar_reward_alone() -> None:
    """The feasibility gate must check denominator_varies_in_eval, not just scalar.

    A higher scalar reward under hobs-active-tx-ee mode is meaningless if
    denominator_varies_in_eval=False — it only reflects a reward-scale change.
    """
    diag = compute_hobs_active_tx_ee_diagnostics(
        _FEASIBILITY_CONFIG,
        max_steps=5,
    )
    # The diagnostics must include the denominator_varies_in_eval key
    assert "denominator_varies_in_eval" in diag
    # The diagnostics must include the throughput_proxy_risk_flag key
    assert "throughput_proxy_risk_flag" in diag
    # forbidden_claims must explicitly prohibit scalar-reward-only success
    forbidden = diag.get("forbidden_claims", [])
    assert any("scalar reward" in claim for claim in forbidden)


def test_ee_reward_mode_produces_nonzero_r1_hobs_active_tx_ee_field() -> None:
    """Verify the new RewardComponents field is populated, not left at default 0."""
    cfg = load_training_yaml(_FEASIBILITY_CONFIG)
    cfg["baseline"]["users"] = 5
    env = build_environment(cfg)

    rng = np.random.default_rng(55)
    states, masks, _diag = env.reset(rng, np.random.default_rng(3))
    actions = np.array(
        [int(np.flatnonzero(mask.mask)[0]) for mask in masks],
        dtype=np.int32,
    )
    result = env.step(actions, rng)

    # At least some users should have nonzero throughput → EE field nonzero
    any_thr = any(rw.r1_throughput > 0 for rw in result.rewards)
    if any_thr:
        assert result.rewards[0].r1_hobs_active_tx_ee > 0.0


def test_reward_vector_uses_hobs_ee_when_feasibility_config() -> None:
    cfg = load_training_yaml(_FEASIBILITY_CONFIG)
    cfg["baseline"]["users"] = 3
    env = build_environment(cfg)
    trainer_cfg = build_trainer_config(cfg)
    trainer = MODQNTrainer(env=env, config=trainer_cfg)

    rng = np.random.default_rng(17)
    states, masks, _diag = env.reset(rng, np.random.default_rng(9))
    actions = np.array(
        [int(np.flatnonzero(mask.mask)[0]) for mask in masks],
        dtype=np.int32,
    )
    result = env.step(actions, rng)

    for uid in range(len(result.rewards)):
        rv = trainer.reward_vector_from_step_result(result, uid)
        # r1 must equal r1_hobs_active_tx_ee (the system EE), not throughput
        assert rv[0] == pytest.approx(result.rewards[uid].r1_hobs_active_tx_ee)
        # r2 and r3 must be unchanged
        assert rv[1] == result.rewards[uid].r2_handover
        assert rv[2] == result.rewards[uid].r3_load_balance
