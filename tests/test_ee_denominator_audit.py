"""Phase 02 EE denominator audit tests."""

from __future__ import annotations

import csv

import numpy as np
from numpy.random import default_rng

from modqn_paper_reproduction.analysis.ee_denominator import (
    build_fixed_power_proxy_row,
    export_ee_denominator_audit,
)
from modqn_paper_reproduction.env.beam import BeamConfig
from modqn_paper_reproduction.env.channel import ChannelConfig
from modqn_paper_reproduction.env.orbit import OrbitConfig
from modqn_paper_reproduction.env.step import (
    PowerSurfaceConfig,
    StepConfig,
    StepEnvironment,
)


def _small_env() -> StepEnvironment:
    return StepEnvironment(
        step_config=StepConfig(num_users=5, user_lat_deg=0.0, user_lon_deg=0.0),
        orbit_config=OrbitConfig(num_satellites=2),
        beam_config=BeamConfig(),
        channel_config=ChannelConfig(tx_power_w=2.0),
    )


def _small_power_surface_env() -> StepEnvironment:
    return StepEnvironment(
        step_config=StepConfig(num_users=5, user_lat_deg=0.0, user_lon_deg=0.0),
        orbit_config=OrbitConfig(num_satellites=2),
        beam_config=BeamConfig(),
        channel_config=ChannelConfig(tx_power_w=2.0),
        power_surface_config=PowerSurfaceConfig(
            hobs_power_surface_mode="active-load-concave",
            inactive_beam_policy="zero-w",
            active_base_power_w=0.25,
            load_scale_power_w=0.35,
            load_exponent=0.5,
            max_power_w=2.0,
        ),
    )


def test_fixed_power_proxy_row_maps_runtime_fields_without_mutating_rewards() -> None:
    env = _small_env()
    rng = default_rng(42)
    states, _masks, _diag = env.reset(rng)
    actions = np.array([int(np.argmax(state.access_vector)) for state in states])
    result = env.step(actions, rng)

    reward_before = [reward.r1_throughput for reward in result.rewards]
    beam_throughput_before = result.beam_throughputs.copy()
    row = build_fixed_power_proxy_row(
        result=result,
        policy="hold-current",
        evaluation_seed=42,
        tx_power_w=env.channel_config.tx_power_w,
    )

    active_count = int(np.count_nonzero(result.user_states[0].beam_loads > 0.0))
    assert row["active_beam_count"] == active_count
    assert row["tx_power_w_per_active_beam"] == 2.0
    assert row["total_active_tx_power_w_fixed_proxy"] == active_count * 2.0
    assert abs(row["throughput_delta_bps"]) < 1e-6
    assert [reward.r1_throughput for reward in result.rewards] == reward_before
    np.testing.assert_array_equal(result.beam_throughputs, beam_throughput_before)


def test_ee_denominator_audit_exports_fixed_power_trap_report(tmp_path) -> None:
    output_dir = tmp_path / "ee-audit"
    outputs = export_ee_denominator_audit(
        "configs/modqn-paper-baseline.resolved-template.yaml",
        output_dir,
        evaluation_seed=12345,
        max_steps=2,
    )

    summary = outputs["summary"]
    assert summary["phase_02_decision"]["status"] == "blocked"
    assert summary["phase_02_decision"]["phase_03_gate"] == "no-go"
    assert summary["phase_02_decision"]["hobs_ee_system_defensible"] is False
    assert summary["phase_02_decision"]["fixed_power_trap_diagnostic"] is True
    assert summary["runtime_mapping"]["P_b_t"]["available_as_allocated_power"] is False
    assert summary["runtime_mapping"]["P_b_t"]["varies_with_action"] is False
    assert summary["runtime_mapping"]["active_beams"]["derivable"] is True
    assert summary["power_unit_audit"]["unit"] == "linear W"
    assert summary["denominator_variability"]["tx_power_w_distinct_values"] == [2.0]
    assert summary["denominator_classification"] in {
        "fixed-power-active-beam-count-proxy",
        "fixed-power-constant-denominator",
    }

    csv_path = outputs["ee_denominator_audit_csv"]
    review_path = outputs["review_md"]
    assert csv_path.exists()
    assert review_path.exists()
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert {row["policy"] for row in rows} == {
        "hold-current",
        "random-valid",
        "first-valid",
    }


def test_default_power_surface_preserves_baseline_step_reward_behavior() -> None:
    env_default = _small_env()
    env_static = StepEnvironment(
        step_config=StepConfig(num_users=5, user_lat_deg=0.0, user_lon_deg=0.0),
        orbit_config=OrbitConfig(num_satellites=2),
        beam_config=BeamConfig(),
        channel_config=ChannelConfig(tx_power_w=2.0),
        power_surface_config=PowerSurfaceConfig(),
    )
    rng_default = default_rng(2026)
    rng_static = default_rng(2026)
    states_default, _masks_default, _diag_default = env_default.reset(rng_default)
    states_static, _masks_static, _diag_static = env_static.reset(rng_static)
    actions = np.array(
        [int(np.argmax(state.access_vector)) for state in states_default],
        dtype=np.int32,
    )

    result_default = env_default.step(actions, rng_default)
    result_static = env_static.step(actions, rng_static)

    np.testing.assert_allclose(
        [reward.r1_throughput for reward in result_default.rewards],
        [reward.r1_throughput for reward in result_static.rewards],
    )
    np.testing.assert_allclose(
        result_default.beam_throughputs,
        result_static.beam_throughputs,
    )
    np.testing.assert_array_equal(
        result_default.active_beam_mask,
        result_static.active_beam_mask,
    )
    np.testing.assert_allclose(result_default.beam_transmit_power_w, 2.0)


def test_active_load_power_surface_emits_linear_w_and_zero_inactive_beams() -> None:
    env = _small_power_surface_env()
    rng = default_rng(42)
    env.reset(rng)
    result = env.step(np.zeros(5, dtype=np.int32), rng)

    expected_active_power = 0.25 + 0.35 * np.sqrt(5.0)
    assert bool(result.active_beam_mask[0])
    assert np.count_nonzero(result.active_beam_mask) == 1
    assert np.isclose(result.beam_transmit_power_w[0], expected_active_power)
    np.testing.assert_allclose(result.beam_transmit_power_w[1:], 0.0)
    assert result.beam_transmit_power_w.dtype == np.float64


def test_active_load_power_surface_denominator_varies_with_actions() -> None:
    env = _small_power_surface_env()
    rng = default_rng(42)
    env.reset(rng)
    concentrated = env.step(np.zeros(5, dtype=np.int32), rng)
    spread = env.step(np.array([0, 1, 2, 3, 4], dtype=np.int32), rng)

    concentrated_power = float(
        np.sum(concentrated.beam_transmit_power_w[concentrated.active_beam_mask])
    )
    spread_power = float(np.sum(spread.beam_transmit_power_w[spread.active_beam_mask]))

    assert concentrated_power > 0.0
    assert spread_power > 0.0
    assert concentrated_power != spread_power


def test_phase_02b_audit_exports_hobs_power_surface_go_report(tmp_path) -> None:
    output_dir = tmp_path / "ee-power-surface"
    outputs = export_ee_denominator_audit(
        "configs/ee-modqn-power-surface-phase-02b.resolved.yaml",
        output_dir,
        evaluation_seed=12345,
        max_steps=2,
    )

    summary = outputs["summary"]
    assert summary["power_surface"]["mode"] == "active-load-concave"
    assert summary["power_surface"]["inactive_beam_policy"] == "zero-w"
    assert summary["phase_02b_decision"]["hobs_ee_system_defensible"] is True
    assert summary["phase_02b_decision"]["phase_03_gate"] == (
        "conditional-go-for-paired-phase-03-design"
    )
    assert summary["phase_02b_decision"]["ee_degenerates_to_throughput_scaling"] is False
    assert summary["denominator_variability"]["denominator_varies"] is True
    assert summary["runtime_mapping"]["P_b_t"]["available_as_allocated_power"] is True
    assert summary["power_unit_audit"]["beam_transmit_power_w_unit"] == "linear W"

    with outputs["ee_denominator_audit_csv"].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert "beam_transmit_power_w" in rows[0]
    assert "active_beam_mask" in rows[0]
    assert "total_active_beam_power_w" in rows[0]
    assert "ee_system_bps_per_w" in rows[0]
