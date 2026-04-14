"""Environment step logic for MODQN paper reproduction.

Assembles per-user state, action mask, reward vector, and episode
semantics per PAP-2024-MORL-MULTIBEAM and the Phase 01 SDD.

State (SDD §3.2):
    s_i(t) = (u_i(t), G_i(t), Γ(t), N(t))

    - u_i(t): one-hot access vector (current beam assignment), length L*K
    - G_i(t): channel quality (SNR linear) to all beams, length L*K
    - Γ(t):   beam-location surface — per-beam local-tangent offsets
              (east_km, north_km) relative to the user, length L*K*2
    - N(t):   per-beam user load (pre-decision counts), length L*K

Action (SDD §3.3):
    One-hot beam selection across L*K beams, satellite-major beam-minor.

Reward (SDD §3.4):
    r1: throughput — B/N_b * log2(1 + gamma)  (paper Eq. 3)
    r2: handover penalty — 0 / -phi1 / -phi2  (ASSUME-MODQN-REP-003)
    r3: load balance — -(max_beam_thr - min_beam_thr) / U

Step semantics:
    slot_duration_s = 1.0, episode_duration_s = 10.0 → 10 steps/episode.

Diagnostics:
    On first call or explicit request, emits zenith-case SNR(dB),
    baseline throughput magnitude, r1/r2/r3 scale ratios, and
    dominance warnings.

Provenance: mix of paper-backed (Eq. 1-3, Table I) and
reproduction-assumption (ASSUME-MODQN-REP-001/002/003/008/009/012).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.random import Generator

from .orbit import OrbitConfig, OrbitProxy, VisibilityResult, EARTH_RADIUS_KM, ground_eci
from .beam import BeamConfig, BeamPattern
from .channel import (
    ChannelConfig,
    ChannelResult,
    compute_channel,
    linear_to_db,
)

# ---------------------------------------------------------------------------
# Reproduction-assumption defaults (used only as typed config defaults)
# ---------------------------------------------------------------------------

USER_HEADING_STRIDE_RAD: float = 2.3998277
"""ASSUME-MODQN-REP-020: irrational-multiple stride for deterministic
per-user heading spread.  ``heading = (uid * stride) mod 2π``.
Chosen so that moderate user counts get quasi-uniform angular coverage
without an RNG draw.  Paper does not specify a mobility heading model."""

USER_SCATTER_RADIUS_KM: float = 50.0
"""ASSUME-MODQN-REP-021: radius of the circular area in which users are
uniformly scattered around the configured ground point.  Paper does not
specify user spatial distribution; 50 km is a reasonable urban/suburban
coverage assumption."""

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepConfig:
    """Top-level configuration for one environment run.

    Paper-backed: num_users, slot_duration_s, episode_duration_s,
    handover phi1/phi2 bounds.
    Reproduction-assumption: concrete phi1/phi2 (ASSUME-MODQN-REP-003).
    """

    num_users: int = 100
    slot_duration_s: float = 1.0
    episode_duration_s: float = 10.0
    user_speed_kmh: float = 30.0

    # ASSUME-MODQN-REP-003: handover penalty values
    phi1: float = 0.5   # intra-satellite beam change
    phi2: float = 1.0   # inter-satellite handover

    # User ground position — reproduction-assumption (not specified in paper).
    # Default: equator, visible to polar-orbit sats.
    user_lat_deg: float = 0.0
    user_lon_deg: float = 0.0

    # ASSUME-MODQN-REP-019: r3 gap semantics
    r3_gap_scope: str = "all-reachable-beams"
    r3_empty_beam_throughput: float = 0.0

    # ASSUME-MODQN-REP-020/021: user mobility and scatter
    user_heading_stride_rad: float = USER_HEADING_STRIDE_RAD
    user_scatter_radius_km: float = USER_SCATTER_RADIUS_KM
    user_scatter_distribution: str = "uniform-circular"

    def __post_init__(self) -> None:
        if self.num_users < 1:
            raise ValueError(f"num_users must be >= 1, got {self.num_users}")
        if self.slot_duration_s <= 0:
            raise ValueError(f"slot_duration_s must be > 0, got {self.slot_duration_s}")
        if self.episode_duration_s <= 0:
            raise ValueError(
                f"episode_duration_s must be > 0, got {self.episode_duration_s}"
            )
        if not (0 < self.phi1 < self.phi2):
            raise ValueError(
                f"Paper requires 0 < phi1 < phi2, got phi1={self.phi1}, phi2={self.phi2}"
            )
        if self.r3_gap_scope not in {
            "all-reachable-beams",
            "occupied-beams-only",
        }:
            raise ValueError(
                "r3_gap_scope must be one of "
                "{'all-reachable-beams', 'occupied-beams-only'}, "
                f"got {self.r3_gap_scope!r}"
            )
        if self.user_scatter_radius_km < 0:
            raise ValueError(
                "user_scatter_radius_km must be >= 0, "
                f"got {self.user_scatter_radius_km}"
            )
        if self.user_scatter_distribution != "uniform-circular":
            raise ValueError(
                "user_scatter_distribution must be 'uniform-circular', "
                f"got {self.user_scatter_distribution!r}"
            )

    @property
    def steps_per_episode(self) -> int:
        return int(self.episode_duration_s / self.slot_duration_s)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UserState:
    """Per-user observation vector s_i(t).

    All beam-indexed arrays use satellite-major, beam-minor ordering
    (ASSUME-MODQN-REP-012).
    """

    access_vector: np.ndarray
    """One-hot current beam assignment, shape (L*K,)."""

    channel_quality: np.ndarray
    """SNR (linear) to every beam, shape (L*K,)."""

    beam_offsets: np.ndarray
    """Per-beam (east_km, north_km) offsets relative to user, shape (L*K, 2)."""

    beam_loads: np.ndarray
    """Pre-decision user count per beam, shape (L*K,)."""


@dataclass(frozen=True)
class ActionMask:
    """Valid-action mask for one user at time t.

    mask[j] = True means action j is eligible (satellite visible above 0°).
    Per ASSUME-MODQN-REP-012, ineligible actions get Q = -inf before argmax.
    """

    mask: np.ndarray
    """Boolean array, shape (L*K,)."""

    @property
    def num_valid(self) -> int:
        return int(np.sum(self.mask))


@dataclass(frozen=True)
class RewardComponents:
    """Raw reward vector r1/r2/r3 for one user at one step.

    No normalization is applied. Values are in natural units:
    - r1: bits/s (throughput)
    - r2: dimensionless penalty (0, -phi1, or -phi2)
    - r3: dimensionless ratio (negative gap / num_users)
    """

    r1_throughput: float
    r2_handover: float
    r3_load_balance: float


@dataclass
class StepResult:
    """Output of one environment step."""

    time_s: float
    step_index: int
    done: bool

    user_states: list[UserState]
    action_masks: list[ActionMask]
    rewards: list[RewardComponents]

    # Per-beam aggregate throughput for load balance computation.
    beam_throughputs: np.ndarray
    """Total throughput per beam, shape (L*K,)."""


@dataclass
class DiagnosticsReport:
    """First-step diagnostics for reward scale inspection.

    Emitted once per environment reset to expose reward magnitudes
    before any hidden normalization could mask calibration issues.
    """

    zenith_snr_db: float
    zenith_throughput_bps: float
    sample_r1: float
    sample_r2_beam_change: float
    sample_r2_sat_change: float
    sample_r3: float
    r1_r2_ratio: float
    r1_r3_ratio: float
    dominance_warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class StepEnvironment:
    """Stateful environment implementing the MODQN paper step semantics.

    Manages user positions, satellite positions, beam assignments,
    and computes state/reward per slot.

    Usage::

        env = StepEnvironment(...)
        states, masks, diag = env.reset(rng)
        for _ in range(env.config.steps_per_episode):
            result = env.step(actions, rng)
            if result.done:
                break
    """

    def __init__(
        self,
        step_config: StepConfig | None = None,
        orbit_config: OrbitConfig | None = None,
        beam_config: BeamConfig | None = None,
        channel_config: ChannelConfig | None = None,
    ) -> None:
        self._step_cfg = step_config or StepConfig()
        self._orbit = OrbitProxy(orbit_config)
        self._beam = BeamPattern(beam_config)
        self._channel_cfg = channel_config or ChannelConfig()

        self._num_beams_total = (
            self._orbit.num_satellites * self._beam.num_beams
        )

        # Mutable state (set by reset)
        self._t_s: float = 0.0
        self._step_index: int = 0
        self._assignments: np.ndarray = np.zeros(0, dtype=np.int32)
        self._user_positions: list[tuple[float, float]] = []
        self._diagnostics_emitted: bool = False
        self._last_diagnostics: DiagnosticsReport | None = None

    # -- properties -----------------------------------------------------------

    @property
    def config(self) -> StepConfig:
        return self._step_cfg

    @property
    def num_beams_total(self) -> int:
        return self._num_beams_total

    @property
    def orbit(self) -> OrbitProxy:
        return self._orbit

    @property
    def beam_pattern(self) -> BeamPattern:
        return self._beam

    @property
    def channel_config(self) -> ChannelConfig:
        return self._channel_cfg

    @property
    def time_s(self) -> float:
        return self._t_s

    @property
    def step_index(self) -> int:
        return self._step_index

    def current_assignments(self) -> np.ndarray:
        """Current global beam assignment per user."""
        return self._assignments.copy()

    def current_user_positions(self) -> list[tuple[float, float]]:
        """Current user geodetic positions."""
        return list(self._user_positions)

    def current_satellites(self):
        """Current satellite snapshots in stable index order."""
        return self._orbit.all_satellites(self._t_s)

    # -- reset ----------------------------------------------------------------

    def reset(
        self,
        rng: Generator,
        mobility_rng: Optional[Generator] = None,
    ) -> tuple[list[UserState], list[ActionMask], DiagnosticsReport]:
        """Reset to t=0 and return initial states, masks, and diagnostics.

        Parameters
        ----------
        rng : Generator
            Environment RNG for channel fading.
        mobility_rng : Generator, optional
            Separate RNG for user position generation.
            If None, uses *rng* (acceptable for Phase 1).
        """
        self._t_s = 0.0
        self._step_index = 0

        # Generate user positions: scattered around the configured ground point.
        # Paper does not specify user distribution.
        # Reproduction-assumption: uniform random in a small area around the
        # configured ground point, radius ~ altitude/10 for reasonable coverage.
        mrng = mobility_rng or rng
        self._user_positions = _generate_user_positions(
            self._step_cfg.num_users,
            self._step_cfg.user_lat_deg,
            self._step_cfg.user_lon_deg,
            mrng,
            radius_km=self._step_cfg.user_scatter_radius_km,
            distribution=self._step_cfg.user_scatter_distribution,
        )

        # Initial assignment: each user to their nearest visible beam.
        self._assignments = self._initial_assignments()

        # Build state and masks
        states, masks, _beam_thr, _user_thr, _beam_loads = self._build_states_and_masks(rng)

        # Diagnostics on first reset only. The report depends on config, not on
        # mutable episode state, so later resets can safely reuse the cached
        # report instead of spamming stderr on every episode.
        if not self._diagnostics_emitted or self._last_diagnostics is None:
            diag = self._emit_diagnostics(rng)
            self._last_diagnostics = diag
            self._diagnostics_emitted = True
        else:
            diag = self._last_diagnostics

        return states, masks, diag

    # -- step -----------------------------------------------------------------

    def step(
        self,
        actions: np.ndarray,
        rng: Generator,
    ) -> StepResult:
        """Execute one 1-second slot.

        Parameters
        ----------
        actions : ndarray of int, shape (num_users,)
            Beam index (in satellite-major, beam-minor order) per user.
        rng : Generator
            For channel fading samples this step.

        Returns
        -------
        StepResult
        """
        if len(actions) != self._step_cfg.num_users:
            raise ValueError(
                f"Expected {self._step_cfg.num_users} actions, got {len(actions)}"
            )

        prev_assignments = self._assignments.copy()

        # Advance time
        self._t_s += self._step_cfg.slot_duration_s
        self._step_index += 1

        # Move users (simple linear mobility within episode)
        self._move_users()

        # Apply actions (respecting mask — caller should mask before choosing)
        self._assignments = actions.astype(np.int32)

        # Compute state and masks for the new time
        states, masks, beam_throughputs, user_throughputs, beam_loads = (
            self._build_states_and_masks(rng)
        )

        # Build reachable-beam mask: union of all users' action masks.
        # A beam is "reachable" if at least one user can see its satellite.
        # ASSUME-MODQN-REP-019: r3 gap uses reachable beams, not just
        # occupied beams, so an empty reachable beam contributes
        # throughput=0 to the min.
        reachable_mask = _reachable_beam_mask(masks)

        # Compute rewards — r1 taken from user_throughputs (single truth)
        rewards = self._compute_rewards(
            prev_assignments, self._assignments,
            beam_throughputs, beam_loads, user_throughputs, reachable_mask,
        )

        done = self._step_index >= self._step_cfg.steps_per_episode

        return StepResult(
            time_s=self._t_s,
            step_index=self._step_index,
            done=done,
            user_states=states,
            action_masks=masks,
            rewards=rewards,
            beam_throughputs=beam_throughputs,
        )

    # -- internal: state assembly ---------------------------------------------

    def _build_states_and_masks(
        self, rng: Generator
    ) -> tuple[list[UserState], list[ActionMask], np.ndarray, np.ndarray, np.ndarray]:
        """Build per-user states, masks, per-beam throughput, and per-user throughput.

        Returns
        -------
        states : list[UserState]
        masks : list[ActionMask]
        beam_throughputs : ndarray, shape (L*K,), float64
        user_throughputs : ndarray, shape (U,), float64
            Single-truth throughput for each user.  Used by both r1 and
            beam-level aggregation so that no float32/float64 recomputation
            drift can occur.
        beam_loads : ndarray, shape (L*K,), float32
            Pre-decision user count per beam. Used by both state `N(t)` and
            ASSUME-MODQN-REP-019 gap semantics so that empty-beam treatment is
            config-driven rather than hardwired.
        """
        L = self._orbit.num_satellites
        K = self._beam.num_beams
        LK = self._num_beams_total
        U = self._step_cfg.num_users

        sats = self._orbit.all_satellites(self._t_s)

        # Pre-compute beam centers for all satellites
        all_beam_centers = [self._beam.beam_centers_ground(s) for s in sats]

        # Per-beam user load (pre-decision)
        beam_loads = np.zeros(LK, dtype=np.float32)
        for uid in range(U):
            beam_idx = int(self._assignments[uid])
            if 0 <= beam_idx < LK:
                beam_loads[beam_idx] += 1

        states: list[UserState] = []
        masks: list[ActionMask] = []

        # Per-beam throughput accumulator
        beam_throughputs = np.zeros(LK, dtype=np.float64)
        # Per-user throughput — single truth for r1 and beam aggregation
        user_throughputs = np.zeros(U, dtype=np.float64)

        for uid in range(U):
            ulat, ulon = self._user_positions[uid]

            # -- visibility for all satellites --
            vis_all = self._orbit.all_visibility(ulat, ulon, self._t_s)

            # -- action mask (ASSUME-MODQN-REP-012) --
            mask_arr = np.zeros(LK, dtype=bool)
            for vr in vis_all:
                if vr.is_visible:
                    base = vr.sat_index * K
                    mask_arr[base: base + K] = True

            # -- channel quality: SNR to every beam --
            # NOTE (F3 follow-up): channel fading draws consume *rng*
            # sequentially across users within a step.  This couples the
            # per-user fading streams: user ordering affects which draws
            # each user sees.  Acceptable for Phase 1 single-policy
            # shared-parameter mode where user ordering is stable, but
            # should be revisited if independent per-user fading streams
            # are needed (e.g. for variance-sensitive ablation).
            snr_arr = np.zeros(LK, dtype=np.float64)
            for vr in vis_all:
                if vr.is_visible and vr.slant_range_km > 0:
                    ch = compute_channel(
                        slant_range_km=vr.slant_range_km,
                        altitude_km=self._orbit.config.altitude_km,
                        config=self._channel_cfg,
                        rng=rng,
                        fading=True,
                    )
                    base = vr.sat_index * K
                    # All beams of a visible satellite share the same
                    # slant-range-based SNR. The paper does not model
                    # per-beam off-axis gain — channel quality depends
                    # on user-satellite geometry only.
                    snr_arr[base: base + K] = ch.snr_linear

            # -- beam offsets Γ(t): local-tangent (east, north) km --
            offsets = np.zeros((LK, 2), dtype=np.float64)
            for si, sat in enumerate(sats):
                bcs = all_beam_centers[si]
                for bc in bcs:
                    gi = si * K + bc.local_beam_index
                    if not math.isnan(bc.lat_deg):
                        e_km, n_km = _local_tangent_offset(
                            ulat, ulon, bc.lat_deg, bc.lon_deg
                        )
                        offsets[gi, 0] = e_km
                        offsets[gi, 1] = n_km

            # -- access vector u_i(t) --
            access = np.zeros(LK, dtype=np.float32)
            cur_beam = int(self._assignments[uid])
            if 0 <= cur_beam < LK:
                access[cur_beam] = 1.0

            # -- user throughput (paper Eq. 3): R = B/N_b * log2(1 + gamma) --
            thr = 0.0
            if 0 <= cur_beam < LK and snr_arr[cur_beam] > 0:
                n_b = max(beam_loads[cur_beam], 1.0)
                bw = self._channel_cfg.bandwidth_hz
                thr = (bw / n_b) * math.log2(1.0 + snr_arr[cur_beam])
            user_throughputs[uid] = thr

            states.append(
                UserState(
                    access_vector=access,
                    channel_quality=snr_arr.astype(np.float32),
                    beam_offsets=offsets.astype(np.float32),
                    beam_loads=beam_loads.copy(),
                )
            )
            masks.append(ActionMask(mask=mask_arr))

        # Aggregate per-beam throughput
        for uid in range(U):
            beam_idx = int(self._assignments[uid])
            if 0 <= beam_idx < LK:
                beam_throughputs[beam_idx] += user_throughputs[uid]

        return states, masks, beam_throughputs, user_throughputs, beam_loads

    # -- internal: rewards ----------------------------------------------------

    def _compute_rewards(
        self,
        prev_assignments: np.ndarray,
        cur_assignments: np.ndarray,
        beam_throughputs: np.ndarray,
        beam_loads: np.ndarray,
        user_throughputs: np.ndarray,
        reachable_mask: np.ndarray,
    ) -> list[RewardComponents]:
        """Compute r1/r2/r3 for each user. No normalization.

        r1 is read directly from *user_throughputs* (the single-truth
        float64 throughput computed in _build_states_and_masks) to avoid
        the float32/float64 recomputation drift that existed before.

        r3 gap uses *reachable_mask* so that empty but reachable beams
        contribute throughput=0 to the min side of the gap
        (ASSUME-MODQN-REP-019).
        """
        U = self._step_cfg.num_users
        K = self._beam.num_beams
        phi1 = self._step_cfg.phi1
        phi2 = self._step_cfg.phi2

        # r3 load balance gap: max-min over reachable beams.
        # Reachable beams with no users have throughput 0, which pulls
        # the min down and exposes the imbalance.
        gap = _load_balance_gap(
            beam_throughputs=beam_throughputs,
            beam_loads=beam_loads,
            reachable_mask=reachable_mask,
            scope=self._step_cfg.r3_gap_scope,
            empty_beam_throughput=self._step_cfg.r3_empty_beam_throughput,
        )

        rewards: list[RewardComponents] = []
        for uid in range(U):
            # r1: single-truth throughput (float64, no recomputation)
            r1 = float(user_throughputs[uid])

            # r2: handover penalty
            prev_beam = int(prev_assignments[uid])
            cur_beam = int(cur_assignments[uid])
            r2 = _handover_penalty(prev_beam, cur_beam, K, phi1, phi2)

            # r3: load balance — -(max - min) / U
            r3 = -gap / U if U > 0 else 0.0

            rewards.append(RewardComponents(
                r1_throughput=r1,
                r2_handover=r2,
                r3_load_balance=r3,
            ))

        return rewards

    # -- internal: initial assignments ----------------------------------------

    def _initial_assignments(self) -> np.ndarray:
        """Assign each user to the nearest visible beam at t=0."""
        U = self._step_cfg.num_users
        K = self._beam.num_beams
        assignments = np.zeros(U, dtype=np.int32)

        sats = self._orbit.all_satellites(self._t_s)

        for uid in range(U):
            ulat, ulon = self._user_positions[uid]
            vis = self._orbit.visible_satellites(ulat, ulon, self._t_s)

            best_beam = 0
            best_range = math.inf

            for vr in vis:
                sat = sats[vr.sat_index]
                local_beam = self._beam.nearest_beam(sat, ulat, ulon)
                global_idx = vr.sat_index * K + local_beam
                if vr.slant_range_km < best_range:
                    best_range = vr.slant_range_km
                    best_beam = global_idx

            assignments[uid] = best_beam

        return assignments

    # -- internal: user mobility ----------------------------------------------

    def _move_users(self) -> None:
        """Simple linear user mobility per step.

        Moves users along a deterministic heading at the configured speed.
        Paper does not specify the mobility model.

        ASSUME-MODQN-REP-020: per-user heading is
        ``(uid * user_heading_stride_rad) mod 2π``. The resolved config owns
        the stride value; the module-level constant is only the default.
        """
        speed_km_s = self._step_cfg.user_speed_kmh / 3600.0
        dt = self._step_cfg.slot_duration_s
        displacement_km = speed_km_s * dt

        for uid in range(len(self._user_positions)):
            lat, lon = self._user_positions[uid]
            heading_rad = (
                uid * self._step_cfg.user_heading_stride_rad
            ) % (2.0 * math.pi)
            dlat = displacement_km * math.cos(heading_rad) / 111.32
            dlon = displacement_km * math.sin(heading_rad) / (
                111.32 * max(math.cos(math.radians(lat)), 0.01)
            )
            self._user_positions[uid] = (lat + dlat, lon + dlon)

    # -- diagnostics ----------------------------------------------------------

    def _emit_diagnostics(self, rng: Generator) -> DiagnosticsReport:
        """Produce a diagnostics report for reward-scale inspection.

        Uses a zenith-case channel computation and typical reward
        scenarios to expose magnitude relationships.
        """
        # Zenith case: slant_range = altitude (satellite directly overhead)
        alt = self._orbit.config.altitude_km
        zenith_ch = compute_channel(
            slant_range_km=alt,
            altitude_km=alt,
            config=self._channel_cfg,
            fading=False,
        )
        zenith_snr_db = zenith_ch.snr_db

        # Throughput for zenith case, single user on beam (N_b=1)
        bw = self._channel_cfg.bandwidth_hz
        zenith_thr = bw * math.log2(1.0 + zenith_ch.snr_linear)

        # Typical r1 with shared bandwidth (assume baseline 100 users, ~4 beams active)
        # Use actual first-step state if available
        U = self._step_cfg.num_users
        L = self._orbit.num_satellites
        K = self._beam.num_beams
        typical_users_per_beam = max(U / (L * K), 1.0)
        sample_r1 = (bw / typical_users_per_beam) * math.log2(
            1.0 + zenith_ch.snr_linear
        )

        sample_r2_beam = -self._step_cfg.phi1
        sample_r2_sat = -self._step_cfg.phi2

        # r3: worst case gap estimate — if one beam has all throughput
        sample_r3 = -zenith_thr / U if U > 0 else 0.0

        # Scale ratios
        r1_abs = abs(sample_r1) if sample_r1 != 0 else 1e-30
        r2_abs = abs(sample_r2_beam) if sample_r2_beam != 0 else 1e-30
        r3_abs = abs(sample_r3) if sample_r3 != 0 else 1e-30

        r1_r2 = r1_abs / r2_abs
        r1_r3 = r1_abs / r3_abs

        # Dominance warnings
        warnings: list[str] = []
        max_obj = max(r1_abs, r2_abs, r3_abs)
        min_obj = min(r1_abs, r2_abs, r3_abs)
        if min_obj > 0 and max_obj / min_obj > 100:
            warnings.append(
                f"DOMINANCE WARNING: reward scale ratio is {max_obj/min_obj:.1f}x "
                f"(>100x threshold). r1_abs={r1_abs:.4e}, r2_abs={r2_abs:.4e}, "
                f"r3_abs={r3_abs:.4e}. "
                f"Scalarized reward will be dominated by the largest objective. "
                f"TODO: evaluate whether reward calibration is needed after "
                f"first training run."
            )

        report = DiagnosticsReport(
            zenith_snr_db=zenith_snr_db,
            zenith_throughput_bps=zenith_thr,
            sample_r1=sample_r1,
            sample_r2_beam_change=sample_r2_beam,
            sample_r2_sat_change=sample_r2_sat,
            sample_r3=sample_r3,
            r1_r2_ratio=r1_r2,
            r1_r3_ratio=r1_r3,
            dominance_warnings=warnings,
        )

        _print_diagnostics(report)
        return report


# ---------------------------------------------------------------------------
# Helpers (module-level)
# ---------------------------------------------------------------------------


def _reachable_beam_mask(masks: list[ActionMask]) -> np.ndarray:
    """Union of all users' action masks → reachable-beam boolean array.

    A beam is reachable if at least one user has it marked eligible.
    Used by r3 gap computation (ASSUME-MODQN-REP-019).
    """
    combined = masks[0].mask.copy()
    for m in masks[1:]:
        combined |= m.mask
    return combined


def _load_balance_gap(
    beam_throughputs: np.ndarray,
    beam_loads: np.ndarray,
    reachable_mask: np.ndarray,
    scope: str = "all-reachable-beams",
    empty_beam_throughput: float = 0.0,
) -> float:
    """Max–min throughput gap over reachable beams (ASSUME-MODQN-REP-019).

    The resolved config owns the active gap semantics:

    - ``all-reachable-beams``:
      the gap spans the union of all users' visibility masks, and empty
      reachable beams contribute ``empty_beam_throughput``.
    - ``occupied-beams-only``:
      only currently occupied reachable beams contribute to the gap.
    """
    if scope == "all-reachable-beams":
        candidate_mask = reachable_mask
        candidate_thr = beam_throughputs[candidate_mask].astype(np.float64, copy=True)
        candidate_loads = beam_loads[candidate_mask]
        candidate_thr[candidate_loads <= 0] = empty_beam_throughput
    elif scope == "occupied-beams-only":
        candidate_mask = reachable_mask & (beam_loads > 0)
        candidate_thr = beam_throughputs[candidate_mask]
    else:
        raise ValueError(f"Unsupported r3 gap scope: {scope!r}")

    if len(candidate_thr) < 2:
        return 0.0
    return float(np.max(candidate_thr) - np.min(candidate_thr))


def _handover_penalty(
    prev_beam: int, cur_beam: int, beams_per_sat: int,
    phi1: float, phi2: float,
) -> float:
    """Compute r2 handover penalty.

    Returns:
      0.0    if beam unchanged (no handover)
      -phi1  if beam changed within the same satellite
      -phi2  if satellite changed (ASSUME-MODQN-REP-003)
    """
    if prev_beam == cur_beam:
        return 0.0

    prev_sat = prev_beam // beams_per_sat
    cur_sat = cur_beam // beams_per_sat

    if prev_sat == cur_sat:
        return -phi1
    return -phi2


def _generate_user_positions(
    num_users: int,
    center_lat: float,
    center_lon: float,
    rng: Generator,
    radius_km: float = USER_SCATTER_RADIUS_KM,
    distribution: str = "uniform-circular",
) -> list[tuple[float, float]]:
    """Generate user ground positions scattered around a center point.

    ASSUME-MODQN-REP-021: the resolved config owns the scatter radius and
    distribution. The module-level constant is only the default value.
    """
    if distribution != "uniform-circular":
        raise ValueError(f"Unsupported user scatter distribution: {distribution!r}")

    positions: list[tuple[float, float]] = []
    for _ in range(num_users):
        # Uniform in circle
        r = radius_km * math.sqrt(float(rng.random()))
        theta = float(rng.random()) * 2.0 * math.pi
        dlat = r * math.cos(theta) / 111.32
        dlon = r * math.sin(theta) / (
            111.32 * max(math.cos(math.radians(center_lat)), 0.01)
        )
        positions.append((center_lat + dlat, center_lon + dlon))
    return positions


def _local_tangent_offset(
    user_lat: float, user_lon: float,
    beam_lat: float, beam_lon: float,
) -> tuple[float, float]:
    """Compute (east_km, north_km) offset from user to beam center.

    Uses the simple equirectangular approximation (sufficient for
    local-scale offsets within a satellite footprint).
    """
    dlat = beam_lat - user_lat
    dlon = beam_lon - user_lon
    cos_lat = math.cos(math.radians(user_lat))

    north_km = dlat * 111.32
    east_km = dlon * 111.32 * cos_lat

    return east_km, north_km


def local_tangent_offset_km(
    user_lat: float,
    user_lon: float,
    target_lat: float,
    target_lon: float,
) -> tuple[float, float]:
    """Public local-tangent offset helper for export/replay surfaces."""
    return _local_tangent_offset(user_lat, user_lon, target_lat, target_lon)


def _print_diagnostics(report: DiagnosticsReport) -> None:
    """Print diagnostics report to stderr for inspection."""
    print("=" * 65, file=sys.stderr)
    print("  MODQN Step Environment — Reward Scale Diagnostics", file=sys.stderr)
    print("=" * 65, file=sys.stderr)
    print(f"  Zenith-case SNR:            {report.zenith_snr_db:+.2f} dB", file=sys.stderr)
    print(f"  Zenith throughput (1 user):  {report.zenith_throughput_bps:.4e} bps", file=sys.stderr)
    print(f"  Sample r1 (shared BW):       {report.sample_r1:.4e} bps", file=sys.stderr)
    print(f"  Sample r2 (beam change):     {report.sample_r2_beam_change:.4f}", file=sys.stderr)
    print(f"  Sample r2 (sat change):      {report.sample_r2_sat_change:.4f}", file=sys.stderr)
    print(f"  Sample r3 (load balance):    {report.sample_r3:.4e}", file=sys.stderr)
    print(f"  |r1| / |r2| ratio:          {report.r1_r2_ratio:.4e}", file=sys.stderr)
    print(f"  |r1| / |r3| ratio:          {report.r1_r3_ratio:.4e}", file=sys.stderr)
    if report.dominance_warnings:
        for w in report.dominance_warnings:
            print(f"  ** {w}", file=sys.stderr)
    else:
        print("  No dominance warnings.", file=sys.stderr)
    print("=" * 65, file=sys.stderr)
