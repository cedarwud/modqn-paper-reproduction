"""Unit tests for env.step — environment step logic.

Validates state assembly, action masking, reward computation,
episode semantics, and diagnostics output per Phase 01 SDD §3.2-3.4
and ASSUME-MODQN-REP-003/008/012.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
from numpy.random import default_rng

from modqn_paper_reproduction.env.orbit import OrbitConfig
from modqn_paper_reproduction.env.beam import BeamConfig
from modqn_paper_reproduction.env.channel import ChannelConfig, AtmosphericSignMode
from modqn_paper_reproduction.env.step import (
    ActionMask,
    DiagnosticsReport,
    RANDOM_WANDERING_MAX_TURN_RAD,
    RewardComponents,
    StepConfig,
    StepEnvironment,
    StepResult,
    UserState,
    USER_HEADING_STRIDE_RAD,
    USER_SCATTER_RADIUS_KM,
    _handover_penalty,
    _load_balance_gap,
    _local_tangent_offset,
    _reachable_beam_mask,
)


# ---------------------------------------------------------------------------
# Small scenario helper: 2 sats, 7 beams, 5 users for fast tests
# ---------------------------------------------------------------------------

def _small_env(num_users: int = 5, num_sats: int = 2) -> StepEnvironment:
    return StepEnvironment(
        step_config=StepConfig(num_users=num_users, user_lat_deg=0.0, user_lon_deg=0.0),
        orbit_config=OrbitConfig(num_satellites=num_sats),
        beam_config=BeamConfig(),
        channel_config=ChannelConfig(),
    )


# ---------------------------------------------------------------------------
# 1. StepConfig validation
# ---------------------------------------------------------------------------


class TestStepConfig(unittest.TestCase):

    def test_defaults(self) -> None:
        cfg = StepConfig()
        self.assertEqual(cfg.num_users, 100)
        self.assertAlmostEqual(cfg.slot_duration_s, 1.0)
        self.assertAlmostEqual(cfg.episode_duration_s, 10.0)
        self.assertEqual(cfg.steps_per_episode, 10)
        self.assertAlmostEqual(cfg.phi1, 0.5)
        self.assertAlmostEqual(cfg.phi2, 1.0)
        self.assertEqual(cfg.r3_gap_scope, "all-reachable-beams")
        self.assertEqual(
            cfg.action_mask_eligibility_mode,
            "satellite-visible-all-beams",
        )
        self.assertAlmostEqual(cfg.user_heading_stride_rad, USER_HEADING_STRIDE_RAD)
        self.assertAlmostEqual(cfg.user_scatter_radius_km, USER_SCATTER_RADIUS_KM)
        self.assertEqual(cfg.user_scatter_distribution, "uniform-circular")
        self.assertEqual(cfg.mobility_model, "deterministic-heading")
        self.assertAlmostEqual(
            cfg.random_wandering_max_turn_rad,
            RANDOM_WANDERING_MAX_TURN_RAD,
        )

    def test_rejects_zero_users(self) -> None:
        with self.assertRaises(ValueError):
            StepConfig(num_users=0)

    def test_rejects_invalid_phi_order(self) -> None:
        with self.assertRaises(ValueError):
            StepConfig(phi1=1.0, phi2=0.5)

    def test_rejects_zero_phi1(self) -> None:
        with self.assertRaises(ValueError):
            StepConfig(phi1=0.0, phi2=1.0)

    def test_steps_per_episode(self) -> None:
        cfg = StepConfig(slot_duration_s=0.5, episode_duration_s=5.0)
        self.assertEqual(cfg.steps_per_episode, 10)

    def test_rejects_rectangle_without_extent(self) -> None:
        with self.assertRaises(ValueError):
            StepConfig(user_scatter_distribution="uniform-rectangle")

    def test_rejects_unknown_mobility_model(self) -> None:
        with self.assertRaises(ValueError):
            StepConfig(mobility_model="unknown")

    def test_rejects_unknown_action_mask_eligibility_mode(self) -> None:
        with self.assertRaises(ValueError):
            StepConfig(action_mask_eligibility_mode="unknown")


# ---------------------------------------------------------------------------
# 2. State shape and content
# ---------------------------------------------------------------------------


class TestStateAssembly(unittest.TestCase):

    def test_state_shapes(self) -> None:
        env = _small_env(num_users=5, num_sats=2)
        rng = default_rng(42)
        states, masks, diag = env.reset(rng)

        LK = 2 * 7  # 14
        self.assertEqual(len(states), 5)
        for s in states:
            self.assertEqual(s.access_vector.shape, (LK,))
            self.assertEqual(s.channel_quality.shape, (LK,))
            self.assertEqual(s.beam_offsets.shape, (LK, 2))
            self.assertEqual(s.beam_loads.shape, (LK,))

    def test_access_vector_is_one_hot(self) -> None:
        env = _small_env(num_users=5, num_sats=2)
        rng = default_rng(42)
        states, _, _ = env.reset(rng)
        for s in states:
            self.assertAlmostEqual(float(np.sum(s.access_vector)), 1.0)
            self.assertTrue(np.all((s.access_vector == 0) | (s.access_vector == 1)))

    def test_beam_loads_sum_to_num_users(self) -> None:
        env = _small_env(num_users=10, num_sats=2)
        rng = default_rng(42)
        states, _, _ = env.reset(rng)
        # All users share the same beam_loads snapshot
        loads = states[0].beam_loads
        self.assertAlmostEqual(float(np.sum(loads)), 10.0)

    def test_channel_quality_nonnegative(self) -> None:
        env = _small_env(num_users=5, num_sats=2)
        rng = default_rng(42)
        states, _, _ = env.reset(rng)
        for s in states:
            self.assertTrue(np.all(s.channel_quality >= 0))

    def test_rectangle_area_positions_stay_within_bounds(self) -> None:
        cfg = StepConfig(
            num_users=20,
            user_lat_deg=40.0,
            user_lon_deg=116.0,
            user_scatter_distribution="uniform-rectangle",
            user_area_width_km=200.0,
            user_area_height_km=90.0,
        )
        env = StepEnvironment(step_config=cfg, orbit_config=OrbitConfig(num_satellites=2))
        rng = default_rng(42)
        env.reset(rng, default_rng(7))

        for lat, lon in env.current_user_positions():
            east_km, north_km = _local_tangent_offset(40.0, 116.0, lat, lon)
            self.assertLessEqual(abs(east_km), 100.0 + 1e-6)
            self.assertLessEqual(abs(north_km), 45.0 + 1e-6)

    def test_reset_can_start_from_nonzero_orbital_time(self) -> None:
        cfg = StepConfig(num_users=1, user_lat_deg=40.0, user_lon_deg=116.0)
        orbit = OrbitConfig(
            num_satellites=4,
            raan_deg=116.0,
            initial_true_anomaly_offset_deg=40.0,
        )
        env = StepEnvironment(step_config=cfg, orbit_config=orbit)
        rng = default_rng(42)

        _, masks_at_zero, _ = env.reset(rng, initial_time_s=0.0)
        visible_at_zero = [
            sat_index
            for sat_index in range(4)
            if np.any(masks_at_zero[0].mask[sat_index * 7: (sat_index + 1) * 7])
        ]
        self.assertEqual(visible_at_zero, [0])

        _, masks_at_offset, _ = env.reset(default_rng(42), initial_time_s=1063.0)
        visible_at_offset = [
            sat_index
            for sat_index in range(4)
            if np.any(masks_at_offset[0].mask[sat_index * 7: (sat_index + 1) * 7])
        ]
        self.assertEqual(visible_at_offset, [3])
        self.assertAlmostEqual(env.time_s, 1063.0)


# ---------------------------------------------------------------------------
# 3. Action mask
# ---------------------------------------------------------------------------


class TestActionMask(unittest.TestCase):

    def test_mask_shape(self) -> None:
        env = _small_env(num_users=3, num_sats=2)
        rng = default_rng(42)
        _, masks, _ = env.reset(rng)
        LK = 2 * 7
        for m in masks:
            self.assertEqual(m.mask.shape, (LK,))

    def test_at_least_one_valid_action(self) -> None:
        """Every user should have at least one visible beam."""
        env = _small_env(num_users=5, num_sats=4)
        rng = default_rng(42)
        _, masks, _ = env.reset(rng)
        for m in masks:
            self.assertGreater(m.num_valid, 0)

    def test_mask_is_satellite_granular(self) -> None:
        """If a satellite is visible, all 7 of its beams should be valid."""
        env = _small_env(num_users=1, num_sats=4)
        rng = default_rng(42)
        _, masks, _ = env.reset(rng)
        m = masks[0].mask
        K = 7
        for si in range(4):
            beam_slice = m[si * K: (si + 1) * K]
            # Either all True or all False
            self.assertTrue(
                np.all(beam_slice) or not np.any(beam_slice),
                f"Satellite {si}: mask should be all-or-nothing per satellite"
            )

    def test_mask_can_be_nearest_beam_per_visible_satellite(self) -> None:
        env = StepEnvironment(
            step_config=StepConfig(
                num_users=1,
                user_lat_deg=0.0,
                user_lon_deg=0.0,
                action_mask_eligibility_mode="nearest-beam-per-visible-satellite",
            ),
            orbit_config=OrbitConfig(num_satellites=4),
            beam_config=BeamConfig(),
            channel_config=ChannelConfig(),
        )
        rng = default_rng(42)
        _, masks, _ = env.reset(rng)
        m = masks[0].mask
        for si in range(4):
            beam_slice = m[si * 7: (si + 1) * 7]
            self.assertIn(
                int(np.sum(beam_slice)),
                {0, 1},
                f"Satellite {si}: nearest-beam mode should keep at most one beam valid",
            )


# ---------------------------------------------------------------------------
# 4. Handover penalty
# ---------------------------------------------------------------------------


class TestHandoverPenalty(unittest.TestCase):

    def test_no_handover(self) -> None:
        self.assertAlmostEqual(_handover_penalty(5, 5, 7, 0.5, 1.0), 0.0)

    def test_intra_satellite_beam_change(self) -> None:
        # beam 0 → beam 3 on satellite 0 (both in [0..6])
        self.assertAlmostEqual(_handover_penalty(0, 3, 7, 0.5, 1.0), -0.5)

    def test_inter_satellite_handover(self) -> None:
        # beam 0 (sat 0) → beam 7 (sat 1)
        self.assertAlmostEqual(_handover_penalty(0, 7, 7, 0.5, 1.0), -1.0)

    def test_inter_satellite_different_beams(self) -> None:
        # sat 0 beam 6 → sat 1 beam 3
        self.assertAlmostEqual(_handover_penalty(6, 10, 7, 0.5, 1.0), -1.0)

    def test_custom_phi_values(self) -> None:
        self.assertAlmostEqual(_handover_penalty(0, 1, 7, 0.3, 0.8), -0.3)
        self.assertAlmostEqual(_handover_penalty(0, 7, 7, 0.3, 0.8), -0.8)


# ---------------------------------------------------------------------------
# 5. Reward components
# ---------------------------------------------------------------------------


class TestRewardComputation(unittest.TestCase):

    def test_rewards_length_matches_users(self) -> None:
        env = _small_env(num_users=5, num_sats=2)
        rng = default_rng(42)
        states, masks, _ = env.reset(rng)

        # Take a step with current assignments (no change)
        actions = np.array([
            int(np.argmax(s.access_vector)) for s in states
        ], dtype=np.int32)
        result = env.step(actions, rng)

        self.assertEqual(len(result.rewards), 5)

    def test_no_handover_gives_zero_r2(self) -> None:
        """Staying on the same beam → r2 = 0."""
        env = _small_env(num_users=3, num_sats=2)
        rng = default_rng(42)
        states, _, _ = env.reset(rng)

        actions = np.array([
            int(np.argmax(s.access_vector)) for s in states
        ], dtype=np.int32)
        result = env.step(actions, rng)

        for r in result.rewards:
            self.assertAlmostEqual(r.r2_handover, 0.0)

    def test_r1_nonnegative(self) -> None:
        """Throughput should never be negative."""
        env = _small_env(num_users=5, num_sats=4)
        rng = default_rng(42)
        env.reset(rng)

        # Random valid actions
        _, masks, _ = env.reset(default_rng(42))
        actions = np.array([
            int(rng.choice(np.where(m.mask)[0])) if m.num_valid > 0 else 0
            for m in masks
        ], dtype=np.int32)
        result = env.step(actions, rng)

        for r in result.rewards:
            self.assertGreaterEqual(r.r1_throughput, 0.0)

    def test_r3_nonpositive(self) -> None:
        """Load balance reward is -(gap)/U, should be <= 0."""
        env = _small_env(num_users=10, num_sats=2)
        rng = default_rng(42)
        states, _, _ = env.reset(rng)

        actions = np.array([
            int(np.argmax(s.access_vector)) for s in states
        ], dtype=np.int32)
        result = env.step(actions, rng)

        for r in result.rewards:
            self.assertLessEqual(r.r3_load_balance, 0.0)

    def test_random_wandering_rectangle_motion_stays_in_area(self) -> None:
        cfg = StepConfig(
            num_users=3,
            user_lat_deg=40.0,
            user_lon_deg=116.0,
            user_speed_kmh=150.0,
            user_scatter_distribution="uniform-rectangle",
            user_area_width_km=20.0,
            user_area_height_km=10.0,
            mobility_model="random-wandering",
            random_wandering_max_turn_rad=math.pi / 6.0,
        )
        env = StepEnvironment(step_config=cfg, orbit_config=OrbitConfig(num_satellites=2))
        rng = default_rng(42)
        states, _, _ = env.reset(rng, default_rng(7))
        before = env.current_user_positions()
        actions = np.array([
            int(np.argmax(s.access_vector)) for s in states
        ], dtype=np.int32)

        for _ in range(10):
            env.step(actions, rng)
            for lat, lon in env.current_user_positions():
                east_km, north_km = _local_tangent_offset(40.0, 116.0, lat, lon)
                self.assertLessEqual(abs(east_km), 10.0 + 1e-6)
                self.assertLessEqual(abs(north_km), 5.0 + 1e-6)

        after = env.current_user_positions()
        self.assertNotEqual(before, after)

    def test_raw_reward_magnitudes_not_normalized(self) -> None:
        """r1 should be in natural bps units, not silently normalized to [0,1].

        Without antenna gain (paper-faithful), zenith SNR is ~-56 dB and
        throughput is on the order of ~1000 bps. This confirms no hidden
        normalization — values are raw.
        """
        env = _small_env(num_users=5, num_sats=4)
        rng = default_rng(42)
        states, _, _ = env.reset(rng)

        actions = np.array([
            int(np.argmax(s.access_vector)) for s in states
        ], dtype=np.int32)
        result = env.step(actions, rng)

        # r1 should be > 0 and in a plausible bps range for the paper's
        # FSPL-only channel (no antenna gain → ~1000 bps scale).
        max_r1 = max(r.r1_throughput for r in result.rewards)
        self.assertGreater(max_r1, 1.0, "r1 should be in bps, not [0,1] normalized")


# ---------------------------------------------------------------------------
# 6. Episode semantics
# ---------------------------------------------------------------------------


class TestEpisodeSemantics(unittest.TestCase):

    def test_episode_length(self) -> None:
        env = _small_env(num_users=3, num_sats=2)
        rng = default_rng(42)
        states, masks, _ = env.reset(rng)

        for step_i in range(10):
            actions = np.array([
                int(np.argmax(s.access_vector))
                for s in (states if step_i == 0 else result.user_states)
            ], dtype=np.int32)
            result = env.step(actions, rng)

            if step_i < 9:
                self.assertFalse(result.done, f"Step {step_i} should not be done")
            else:
                self.assertTrue(result.done, "Step 9 (10th) should be done")

    def test_time_advances_by_slot(self) -> None:
        env = _small_env(num_users=3, num_sats=2)
        rng = default_rng(42)
        states, _, _ = env.reset(rng)

        actions = np.array([
            int(np.argmax(s.access_vector)) for s in states
        ], dtype=np.int32)
        result = env.step(actions, rng)
        self.assertAlmostEqual(result.time_s, 1.0)

        result2 = env.step(actions, rng)
        self.assertAlmostEqual(result2.time_s, 2.0)

    def test_step_index_increments(self) -> None:
        env = _small_env(num_users=3, num_sats=2)
        rng = default_rng(42)
        states, _, _ = env.reset(rng)
        actions = np.array([
            int(np.argmax(s.access_vector)) for s in states
        ], dtype=np.int32)

        for expected_idx in range(1, 4):
            result = env.step(actions, rng)
            self.assertEqual(result.step_index, expected_idx)


# ---------------------------------------------------------------------------
# 7. Diagnostics
# ---------------------------------------------------------------------------


class TestDiagnostics(unittest.TestCase):

    def test_diagnostics_returned_on_reset(self) -> None:
        env = _small_env(num_users=5, num_sats=2)
        rng = default_rng(42)
        _, _, diag = env.reset(rng)
        self.assertIsInstance(diag, DiagnosticsReport)

    def test_zenith_snr_is_finite(self) -> None:
        env = _small_env(num_users=5, num_sats=2)
        rng = default_rng(42)
        _, _, diag = env.reset(rng)
        self.assertTrue(math.isfinite(diag.zenith_snr_db))

    def test_zenith_throughput_positive(self) -> None:
        env = _small_env(num_users=5, num_sats=2)
        rng = default_rng(42)
        _, _, diag = env.reset(rng)
        self.assertGreater(diag.zenith_throughput_bps, 0)

    def test_sample_r2_values_match_phi(self) -> None:
        cfg = StepConfig(phi1=0.5, phi2=1.0, num_users=5)
        env = StepEnvironment(step_config=cfg, orbit_config=OrbitConfig(num_satellites=2))
        rng = default_rng(42)
        _, _, diag = env.reset(rng)
        self.assertAlmostEqual(diag.sample_r2_beam_change, -0.5)
        self.assertAlmostEqual(diag.sample_r2_sat_change, -1.0)

    def test_dominance_warning_fires(self) -> None:
        """With paper default parameters, r1 (bps) should dominate r2 (unit scale),
        triggering a dominance warning."""
        env = _small_env(num_users=5, num_sats=2)
        rng = default_rng(42)
        _, _, diag = env.reset(rng)
        # r1 is ~10^8 bps, r2 is ~0.5 → ratio > 100x is expected
        self.assertGreater(len(diag.dominance_warnings), 0,
                           "Expected dominance warning for raw reward scales")

    def test_ratios_are_positive(self) -> None:
        env = _small_env(num_users=5, num_sats=2)
        rng = default_rng(42)
        _, _, diag = env.reset(rng)
        self.assertGreater(diag.r1_r2_ratio, 0)
        self.assertGreater(diag.r1_r3_ratio, 0)


# ---------------------------------------------------------------------------
# 8. Local tangent offset helper
# ---------------------------------------------------------------------------


class TestLocalTangentOffset(unittest.TestCase):

    def test_same_point_is_zero(self) -> None:
        e, n = _local_tangent_offset(10.0, 20.0, 10.0, 20.0)
        self.assertAlmostEqual(e, 0.0, places=10)
        self.assertAlmostEqual(n, 0.0, places=10)

    def test_north_positive(self) -> None:
        """Point to the north → positive north_km."""
        e, n = _local_tangent_offset(0.0, 0.0, 1.0, 0.0)
        self.assertAlmostEqual(e, 0.0, places=5)
        self.assertGreater(n, 0.0)
        # ~111 km per degree
        self.assertAlmostEqual(n, 111.32, places=0)

    def test_east_positive(self) -> None:
        """Point to the east → positive east_km."""
        e, n = _local_tangent_offset(0.0, 0.0, 0.0, 1.0)
        self.assertGreater(e, 0.0)
        self.assertAlmostEqual(n, 0.0, places=10)


# ---------------------------------------------------------------------------
# 9. Reproducibility with seed
# ---------------------------------------------------------------------------


class TestReproducibility(unittest.TestCase):

    def test_same_seed_same_states(self) -> None:
        env1 = _small_env(num_users=5, num_sats=2)
        env2 = _small_env(num_users=5, num_sats=2)
        s1, _, _ = env1.reset(default_rng(42))
        s2, _, _ = env2.reset(default_rng(42))

        for a, b in zip(s1, s2):
            np.testing.assert_array_equal(a.access_vector, b.access_vector)
            np.testing.assert_array_equal(a.channel_quality, b.channel_quality)

    def test_different_seed_different_states(self) -> None:
        env1 = _small_env(num_users=5, num_sats=2)
        env2 = _small_env(num_users=5, num_sats=2)
        s1, _, _ = env1.reset(default_rng(42))
        s2, _, _ = env2.reset(default_rng(99))

        # Channel quality should differ (different fading + user positions)
        any_diff = any(
            not np.array_equal(a.channel_quality, b.channel_quality)
            for a, b in zip(s1, s2)
        )
        self.assertTrue(any_diff)


# ---------------------------------------------------------------------------
# 10. r3 gap semantics — reachable beams (ASSUME-MODQN-REP-019)
# ---------------------------------------------------------------------------


class TestR3GapSemantics(unittest.TestCase):

    def test_single_active_beam_r3_nonzero(self) -> None:
        """If all users are on one beam and other beams are reachable,
        r3 must NOT be 0 — the gap should equal that beam's throughput."""
        env = _small_env(num_users=5, num_sats=2)
        rng = default_rng(42)
        states, masks, _ = env.reset(rng)

        # Force all users onto beam 0
        actions = np.zeros(5, dtype=np.int32)
        result = env.step(actions, rng)

        # At least one reachable beam is empty → gap > 0 → r3 < 0
        for r in result.rewards:
            self.assertLess(
                r.r3_load_balance, 0.0,
                "All users on one beam: r3 must be negative (gap > 0), "
                "not 0 (false perfect balance)."
            )

    def test_load_balance_gap_with_empty_reachable_beams(self) -> None:
        """_load_balance_gap should include empty reachable beams at 0."""
        beam_thr = np.array([100.0, 0.0, 50.0, 0.0])
        beam_loads = np.array([2.0, 0.0, 1.0, 0.0])
        reachable = np.array([True, True, True, False])
        # Reachable: [100, 0, 50] → gap = 100 - 0 = 100
        gap = _load_balance_gap(beam_thr, beam_loads, reachable)
        self.assertAlmostEqual(gap, 100.0)

    def test_load_balance_gap_all_occupied_no_change(self) -> None:
        """When all reachable beams have users, gap is still max-min."""
        beam_thr = np.array([80.0, 40.0, 60.0])
        beam_loads = np.array([1.0, 1.0, 1.0])
        reachable = np.array([True, True, True])
        gap = _load_balance_gap(beam_thr, beam_loads, reachable)
        self.assertAlmostEqual(gap, 40.0)

    def test_load_balance_gap_single_reachable_beam(self) -> None:
        """With only one reachable beam, gap should be 0."""
        beam_thr = np.array([100.0, 0.0])
        beam_loads = np.array([1.0, 0.0])
        reachable = np.array([True, False])
        gap = _load_balance_gap(beam_thr, beam_loads, reachable)
        self.assertAlmostEqual(gap, 0.0)

    def test_load_balance_gap_occupied_only_scope(self) -> None:
        """occupied-beams-only reproduces the legacy single-beam gap=0 behavior."""
        beam_thr = np.array([100.0, 0.0, 0.0])
        beam_loads = np.array([5.0, 0.0, 0.0])
        reachable = np.array([True, True, True])
        gap = _load_balance_gap(
            beam_thr,
            beam_loads,
            reachable,
            scope="occupied-beams-only",
        )
        self.assertAlmostEqual(gap, 0.0)

    def test_reachable_beam_mask_union(self) -> None:
        """_reachable_beam_mask is the union of all users' masks."""
        m1 = ActionMask(mask=np.array([True, False, True, False]))
        m2 = ActionMask(mask=np.array([False, True, False, False]))
        combined = _reachable_beam_mask([m1, m2])
        np.testing.assert_array_equal(
            combined, [True, True, True, False]
        )


# ---------------------------------------------------------------------------
# 11. Throughput consistency — r1 and r3 share single truth
# ---------------------------------------------------------------------------


class TestThroughputConsistency(unittest.TestCase):

    def test_r1_matches_beam_throughput_contribution(self) -> None:
        """Each user's r1 must be consistent with the per-beam throughput
        used for r3, i.e. sum of r1 for users on a beam == beam_throughput."""
        env = _small_env(num_users=10, num_sats=2)
        rng = default_rng(42)
        states, masks, _ = env.reset(rng)

        actions = np.array([
            int(np.argmax(s.access_vector)) for s in states
        ], dtype=np.int32)
        result = env.step(actions, rng)

        # Rebuild per-beam r1 sum from rewards
        LK = env.num_beams_total
        r1_per_beam = np.zeros(LK, dtype=np.float64)
        for uid, r in enumerate(result.rewards):
            beam = int(actions[uid])
            r1_per_beam[beam] += r.r1_throughput

        # Must match beam_throughputs (single truth path)
        for b in range(LK):
            self.assertAlmostEqual(
                r1_per_beam[b], result.beam_throughputs[b],
                places=6,
                msg=f"Beam {b}: r1 sum ({r1_per_beam[b]:.6e}) != "
                    f"beam_throughput ({result.beam_throughputs[b]:.6e})"
            )


# ---------------------------------------------------------------------------
# 12. Mobility assumption surfaces are explicit
# ---------------------------------------------------------------------------


class TestMobilityAssumptionSurfaces(unittest.TestCase):

    def test_heading_stride_constant_exists(self) -> None:
        """USER_HEADING_STRIDE_RAD must be an importable module-level constant."""
        self.assertIsInstance(USER_HEADING_STRIDE_RAD, float)
        self.assertAlmostEqual(USER_HEADING_STRIDE_RAD, 2.3998277)

    def test_scatter_radius_constant_exists(self) -> None:
        """USER_SCATTER_RADIUS_KM must be an importable module-level constant."""
        self.assertIsInstance(USER_SCATTER_RADIUS_KM, float)
        self.assertAlmostEqual(USER_SCATTER_RADIUS_KM, 50.0)

    def test_heading_stride_covers_circle(self) -> None:
        """10 users should get roughly uniform heading spread."""
        headings = [(uid * USER_HEADING_STRIDE_RAD) % (2.0 * math.pi)
                    for uid in range(10)]
        # All headings should be distinct and spread across [0, 2π)
        self.assertEqual(len(set(round(h, 4) for h in headings)), 10)
        # Standard deviation of uniform on [0, 2π] is ~1.81; quasi-uniform
        # stride should have std > 1.0 (not clustered)
        self.assertGreater(np.std(headings), 1.0)

    def test_step_config_override_controls_runtime_mobility(self) -> None:
        """ASSUME-MODQN-REP-020/021 values must flow through StepConfig at runtime."""
        env = StepEnvironment(
            step_config=StepConfig(
                num_users=2,
                user_lat_deg=0.0,
                user_lon_deg=0.0,
                user_scatter_radius_km=0.0,
                user_heading_stride_rad=0.0,
            ),
            orbit_config=OrbitConfig(num_satellites=2),
            beam_config=BeamConfig(),
            channel_config=ChannelConfig(),
        )
        rng = default_rng(42)
        states, _, _ = env.reset(rng, default_rng(7))

        self.assertEqual(env._user_positions[0], (0.0, 0.0))
        self.assertEqual(env._user_positions[1], (0.0, 0.0))

        actions = np.array([
            int(np.argmax(s.access_vector)) for s in states
        ], dtype=np.int32)
        env.step(actions, default_rng(99))
        self.assertEqual(env._user_positions[0], env._user_positions[1])


if __name__ == "__main__":
    unittest.main()
