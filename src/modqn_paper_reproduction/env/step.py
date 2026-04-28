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

RANDOM_WANDERING_MAX_TURN_RAD: float = math.pi / 4.0
"""ASSUME-MODQN-REP-023: per-slot bounded turn magnitude for the
follow-on `random wandering` mobility rule. The paper names the mobility
family but does not disclose the exact turn-law details."""

HOBS_POWER_SURFACE_STATIC_CONFIG = "static-config"
"""Default baseline path: keep the paper Table I scalar transmit power."""

HOBS_POWER_SURFACE_ACTIVE_LOAD_CONCAVE = "active-load-concave"
"""Opt-in Phase 02B proxy: active beam power is allocated from beam load."""

HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK = "phase-03c-b-power-codebook"
"""Opt-in Phase 03C-B new-extension: centralized discrete power codebook."""

POWER_CODEBOOK_PROFILES = {
    "fixed-low",
    "fixed-mid",
    "fixed-high",
    "load-concave",
    "qos-tail-boost",
    "budget-trim",
}
"""Supported Phase 03C-B HOBS-inspired power-codebook controller profiles."""

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
    action_mask_eligibility_mode: str = "satellite-visible-all-beams"
    user_heading_stride_rad: float = USER_HEADING_STRIDE_RAD
    user_scatter_radius_km: float = USER_SCATTER_RADIUS_KM
    user_scatter_distribution: str = "uniform-circular"
    user_area_width_km: float = 0.0
    user_area_height_km: float = 0.0
    mobility_model: str = "deterministic-heading"
    random_wandering_max_turn_rad: float = RANDOM_WANDERING_MAX_TURN_RAD

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
        if self.action_mask_eligibility_mode not in {
            "satellite-visible-all-beams",
            "nearest-beam-per-visible-satellite",
        }:
            raise ValueError(
                "action_mask_eligibility_mode must be one of "
                "{'satellite-visible-all-beams', "
                "'nearest-beam-per-visible-satellite'}, "
                f"got {self.action_mask_eligibility_mode!r}"
            )
        if self.user_scatter_radius_km < 0:
            raise ValueError(
                "user_scatter_radius_km must be >= 0, "
                f"got {self.user_scatter_radius_km}"
            )
        if self.user_scatter_distribution not in {
            "uniform-circular",
            "uniform-rectangle",
        }:
            raise ValueError(
                "user_scatter_distribution must be one of "
                "{'uniform-circular', 'uniform-rectangle'}, "
                f"got {self.user_scatter_distribution!r}"
            )
        if self.user_scatter_distribution == "uniform-rectangle":
            if self.user_area_width_km <= 0 or self.user_area_height_km <= 0:
                raise ValueError(
                    "uniform-rectangle requires positive user_area_width_km "
                    f"and user_area_height_km, got width={self.user_area_width_km}, "
                    f"height={self.user_area_height_km}"
                )
        if self.mobility_model not in {
            "deterministic-heading",
            "random-wandering",
        }:
            raise ValueError(
                "mobility_model must be one of "
                "{'deterministic-heading', 'random-wandering'}, "
                f"got {self.mobility_model!r}"
            )
        if self.random_wandering_max_turn_rad < 0:
            raise ValueError(
                "random_wandering_max_turn_rad must be >= 0, "
                f"got {self.random_wandering_max_turn_rad}"
            )

    @property
    def steps_per_episode(self) -> int:
        return int(self.episode_duration_s / self.slot_duration_s)


@dataclass(frozen=True)
class PowerSurfaceConfig:
    """Opt-in per-beam transmit-power surface.

    The default ``static-config`` mode preserves the frozen MODQN baseline:
    channel/SINR uses ``ChannelConfig.tx_power_w`` exactly as before.

    ``active-load-concave`` is a Phase 02B audit surface, not an EE reward. It
    emits explicit linear-W ``P_b(t)`` values from the current active beam load:

    ``P_b(t) = min(max_w, active_base_power_w + load_scale_power_w * N_b^exponent)``

    for active beams and ``0 W`` for inactive beams.
    """

    hobs_power_surface_mode: str = HOBS_POWER_SURFACE_STATIC_CONFIG
    inactive_beam_policy: str = "excluded-from-active-beams"
    active_base_power_w: float = 0.25
    load_scale_power_w: float = 0.35
    load_exponent: float = 0.5
    max_power_w: float | None = 2.0
    power_codebook_profile: str = "fixed-mid"
    power_codebook_levels_w: tuple[float, ...] = (0.5, 1.0, 2.0)
    total_power_budget_w: float | None = 8.0

    def __post_init__(self) -> None:
        if self.hobs_power_surface_mode not in {
            HOBS_POWER_SURFACE_STATIC_CONFIG,
            HOBS_POWER_SURFACE_ACTIVE_LOAD_CONCAVE,
            HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK,
        }:
            raise ValueError(
                "hobs_power_surface_mode must be one of "
                f"{{{HOBS_POWER_SURFACE_STATIC_CONFIG!r}, "
                f"{HOBS_POWER_SURFACE_ACTIVE_LOAD_CONCAVE!r}, "
                f"{HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK!r}}}, "
                f"got {self.hobs_power_surface_mode!r}"
            )
        if self.inactive_beam_policy not in {
            "excluded-from-active-beams",
            "zero-w",
        }:
            raise ValueError(
                "inactive_beam_policy must be one of "
                "{'excluded-from-active-beams', 'zero-w'}, "
                f"got {self.inactive_beam_policy!r}"
            )
        if self.active_base_power_w < 0:
            raise ValueError(
                "active_base_power_w must be >= 0, "
                f"got {self.active_base_power_w}"
            )
        if self.load_scale_power_w < 0:
            raise ValueError(
                "load_scale_power_w must be >= 0, "
                f"got {self.load_scale_power_w}"
            )
        if self.load_exponent <= 0:
            raise ValueError(f"load_exponent must be > 0, got {self.load_exponent}")
        if self.max_power_w is not None and self.max_power_w <= 0:
            raise ValueError(f"max_power_w must be > 0 when set, got {self.max_power_w}")
        if self.power_codebook_profile not in POWER_CODEBOOK_PROFILES:
            raise ValueError(
                "power_codebook_profile must be one of "
                f"{sorted(POWER_CODEBOOK_PROFILES)!r}, "
                f"got {self.power_codebook_profile!r}"
            )
        if not self.power_codebook_levels_w:
            raise ValueError("power_codebook_levels_w must contain at least one level.")
        if any(level <= 0 for level in self.power_codebook_levels_w):
            raise ValueError(
                "power_codebook_levels_w values must all be > 0, "
                f"got {self.power_codebook_levels_w!r}"
            )
        if tuple(self.power_codebook_levels_w) != tuple(sorted(self.power_codebook_levels_w)):
            raise ValueError(
                "power_codebook_levels_w must be sorted ascending, "
                f"got {self.power_codebook_levels_w!r}"
            )
        if (
            self.max_power_w is not None
            and max(self.power_codebook_levels_w) > float(self.max_power_w)
        ):
            raise ValueError(
                "power_codebook_levels_w must not exceed max_power_w, "
                f"got levels={self.power_codebook_levels_w!r}, "
                f"max_power_w={self.max_power_w!r}"
            )
        if self.total_power_budget_w is not None and self.total_power_budget_w <= 0:
            raise ValueError(
                "total_power_budget_w must be > 0 when set, "
                f"got {self.total_power_budget_w}"
            )
        if (
            self.hobs_power_surface_mode == HOBS_POWER_SURFACE_ACTIVE_LOAD_CONCAVE
            and self.inactive_beam_policy != "zero-w"
        ):
            raise ValueError(
                "active-load-concave requires inactive_beam_policy='zero-w'."
            )
        if (
            self.hobs_power_surface_mode
            == HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK
            and self.inactive_beam_policy != "zero-w"
        ):
            raise ValueError(
                "phase-03c-b-power-codebook requires inactive_beam_policy='zero-w'."
            )


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

    mask[j] = True means action j is eligible under the active
    visibility/beam-eligibility mode. Per ASSUME-MODQN-REP-012,
    ineligible actions get Q = -inf before argmax.
    """

    mask: np.ndarray
    """Boolean array, shape (L*K,)."""

    @property
    def num_valid(self) -> int:
        return int(np.sum(self.mask))


@dataclass(frozen=True)
class RewardComponents:
    """Raw reward vector r1/r2/r3 for one user at one step.

    The paper-backed MODQN r1 throughput is always preserved. Phase 03 may
    opt into the separate per-user EE credit as the trainer's first objective,
    but that credit is a modeling / credit-assignment assumption and not the
    system-level EE metric.

    No normalization is applied. Values are in natural units:
    - r1_throughput: bits/s (throughput)
    - r1_energy_efficiency_credit: bits/s/W using equal allocated
      ``P_b/N_b`` credit assignment (Phase 03 diagnostic or gated objective)
    - r1_beam_power_efficiency_credit: bits/s/W using full selected-beam
      ``P_b`` denominator (Phase 03B credit-assignment sensitivity)
    - r2: dimensionless penalty (0, -phi1, or -phi2)
    - r3: dimensionless ratio (negative gap / num_users)
    """

    r1_throughput: float
    r2_handover: float
    r3_load_balance: float
    r1_energy_efficiency_credit: float = 0.0
    r1_beam_power_efficiency_credit: float = 0.0


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

    active_beam_mask: np.ndarray
    """Boolean active-beam mask derived from post-action beam loads, shape (L*K,)."""

    beam_transmit_power_w: np.ndarray
    """Explicit downlink per-beam transmit power ``P_b(t)`` in linear W."""


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
        power_surface_config: PowerSurfaceConfig | None = None,
    ) -> None:
        self._step_cfg = step_config or StepConfig()
        self._orbit = OrbitProxy(orbit_config)
        self._beam = BeamPattern(beam_config)
        self._channel_cfg = channel_config or ChannelConfig()
        self._power_surface_cfg = power_surface_config or PowerSurfaceConfig()

        self._num_beams_total = (
            self._orbit.num_satellites * self._beam.num_beams
        )

        # Mutable state (set by reset)
        self._t_s: float = 0.0
        self._step_index: int = 0
        self._assignments: np.ndarray = np.zeros(0, dtype=np.int32)
        self._user_positions: list[tuple[float, float]] = []
        self._user_headings: list[float] = []
        self._mobility_rng: Generator | None = None
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
    def power_surface_config(self) -> PowerSurfaceConfig:
        return self._power_surface_cfg

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
        *,
        initial_time_s: float = 0.0,
    ) -> tuple[list[UserState], list[ActionMask], DiagnosticsReport]:
        """Reset to the requested start time and return states, masks, and diagnostics.

        Parameters
        ----------
        rng : Generator
            Environment RNG for channel fading.
        mobility_rng : Generator, optional
            Separate RNG for user position generation.
            If None, uses *rng* (acceptable for Phase 1).
        initial_time_s : float, optional
            Orbital time offset in seconds for the reset state. This keeps the
            producer in control of replay-window selection without changing the
            scenario or consumer truth source.
        """
        if not math.isfinite(initial_time_s) or initial_time_s < 0.0:
            raise ValueError(
                "initial_time_s must be a finite value >= 0.0, "
                f"got {initial_time_s!r}"
            )

        self._t_s = float(initial_time_s)
        self._step_index = 0

        # Generate user positions: scattered around the configured ground point.
        # Paper does not specify user distribution.
        # Reproduction-assumption: uniform random in a small area around the
        # configured ground point, radius ~ altitude/10 for reasonable coverage.
        mrng = mobility_rng or rng
        self._mobility_rng = mrng
        self._user_positions = _generate_user_positions(
            self._step_cfg.num_users,
            self._step_cfg.user_lat_deg,
            self._step_cfg.user_lon_deg,
            mrng,
            radius_km=self._step_cfg.user_scatter_radius_km,
            distribution=self._step_cfg.user_scatter_distribution,
            width_km=self._step_cfg.user_area_width_km,
            height_km=self._step_cfg.user_area_height_km,
        )
        self._user_headings = _generate_user_headings(self._step_cfg, mrng)

        # Initial assignment: each user to their nearest visible beam.
        self._assignments = self._initial_assignments()

        # Build state and masks
        states, masks, _beam_thr, _user_thr, _beam_loads, _beam_power_w = (
            self._build_states_and_masks(rng)
        )

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
        states, masks, beam_throughputs, user_throughputs, beam_loads, beam_power_w = (
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
            beam_transmit_power_w=beam_power_w,
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
            active_beam_mask=beam_loads > 0.0,
            beam_transmit_power_w=beam_power_w,
        )

    # -- internal: state assembly ---------------------------------------------

    def _build_states_and_masks(
        self, rng: Generator
    ) -> tuple[
        list[UserState],
        list[ActionMask],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
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
        beam_transmit_power_w : ndarray, shape (L*K,), float64
            Explicit per-beam downlink transmit power in linear W.
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

        beam_transmit_power_w = _beam_transmit_power_w(
            beam_loads,
            channel_tx_power_w=self._channel_cfg.tx_power_w,
            power_surface_config=self._power_surface_cfg,
        )

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
                if not vr.is_visible:
                    continue
                base = vr.sat_index * K
                if self._step_cfg.action_mask_eligibility_mode == "satellite-visible-all-beams":
                    mask_arr[base: base + K] = True
                else:
                    local_beam = self._beam.nearest_beam(
                        sats[vr.sat_index],
                        ulat,
                        ulon,
                    )
                    mask_arr[base + local_beam] = True

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
                    static_power_mode = (
                        self._power_surface_cfg.hobs_power_surface_mode
                        == HOBS_POWER_SURFACE_STATIC_CONFIG
                    )
                    ch = compute_channel(
                        slant_range_km=vr.slant_range_km,
                        altitude_km=self._orbit.config.altitude_km,
                        config=self._channel_cfg,
                        rng=rng,
                        fading=True,
                        tx_power_w=None if static_power_mode else 1.0,
                    )
                    base = vr.sat_index * K
                    # All beams of a visible satellite share the same
                    # slant-range-based SNR. The paper does not model
                    # per-beam off-axis gain. In opt-in HOBS power-surface
                    # mode, the per-beam transmit power is the explicit
                    # numerator power P_b(t); baseline mode keeps the old
                    # scalar config-power SNR path.
                    if static_power_mode:
                        snr_arr[base: base + K] = ch.snr_linear
                    else:
                        for local_beam in range(K):
                            beam_idx = base + local_beam
                            tx_power_w = float(beam_transmit_power_w[beam_idx])
                            snr_arr[beam_idx] = (
                                0.0
                                if tx_power_w <= 0.0
                                else (tx_power_w * ch.channel_gain) / ch.noise_power_w
                            )

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

        return (
            states,
            masks,
            beam_throughputs,
            user_throughputs,
            beam_loads,
            beam_transmit_power_w,
        )

    # -- internal: rewards ----------------------------------------------------

    def _compute_rewards(
        self,
        prev_assignments: np.ndarray,
        cur_assignments: np.ndarray,
        beam_throughputs: np.ndarray,
        beam_loads: np.ndarray,
        user_throughputs: np.ndarray,
        reachable_mask: np.ndarray,
        *,
        beam_transmit_power_w: np.ndarray,
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

            prev_beam = int(prev_assignments[uid])
            cur_beam = int(cur_assignments[uid])
            cur_load = (
                float(beam_loads[cur_beam])
                if 0 <= cur_beam < len(beam_loads)
                else 0.0
            )
            cur_power_w = (
                float(beam_transmit_power_w[cur_beam])
                if 0 <= cur_beam < len(beam_transmit_power_w)
                else 0.0
            )
            allocated_power_w = cur_power_w / cur_load if cur_load > 0.0 else 0.0
            r1_ee_credit = r1 / allocated_power_w if allocated_power_w > 0.0 else 0.0
            r1_beam_ee_credit = r1 / cur_power_w if cur_power_w > 0.0 else 0.0

            # r2: handover penalty
            r2 = _handover_penalty(prev_beam, cur_beam, K, phi1, phi2)

            # r3: load balance — -(max - min) / U
            r3 = -gap / U if U > 0 else 0.0

            rewards.append(RewardComponents(
                r1_throughput=r1,
                r2_handover=r2,
                r3_load_balance=r3,
                r1_energy_efficiency_credit=float(r1_ee_credit),
                r1_beam_power_efficiency_credit=float(r1_beam_ee_credit),
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

        The resolved config controls the active mobility family:

        - `deterministic-heading` keeps the original Phase 01 proxy
        - `random-wandering` maintains a per-user heading and applies a
          bounded random turn each slot
        """
        speed_km_s = self._step_cfg.user_speed_kmh / 3600.0
        dt = self._step_cfg.slot_duration_s
        displacement_km = speed_km_s * dt

        for uid in range(len(self._user_positions)):
            lat, lon = self._user_positions[uid]
            if self._step_cfg.mobility_model == "random-wandering":
                if self._mobility_rng is None:
                    raise ValueError("random-wandering mobility requires a mobility RNG")
                turn_delta = float(
                    self._mobility_rng.uniform(
                        -self._step_cfg.random_wandering_max_turn_rad,
                        self._step_cfg.random_wandering_max_turn_rad,
                    )
                )
                heading_rad = (self._user_headings[uid] + turn_delta) % (2.0 * math.pi)
            else:
                heading_rad = (
                    uid * self._step_cfg.user_heading_stride_rad
                ) % (2.0 * math.pi)

            east_step_km = displacement_km * math.sin(heading_rad)
            north_step_km = displacement_km * math.cos(heading_rad)

            if self._step_cfg.user_scatter_distribution == "uniform-rectangle":
                east_km, north_km = _local_tangent_offset(
                    self._step_cfg.user_lat_deg,
                    self._step_cfg.user_lon_deg,
                    lat,
                    lon,
                )
                half_width = self._step_cfg.user_area_width_km / 2.0
                half_height = self._step_cfg.user_area_height_km / 2.0
                next_east_km = east_km + east_step_km
                next_north_km = north_km + north_step_km

                if next_east_km < -half_width or next_east_km > half_width:
                    heading_rad = (-heading_rad) % (2.0 * math.pi)
                    east_step_km = -east_step_km
                    next_east_km = max(min(east_km + east_step_km, half_width), -half_width)

                if next_north_km < -half_height or next_north_km > half_height:
                    heading_rad = (math.pi - heading_rad) % (2.0 * math.pi)
                    north_step_km = -north_step_km
                    next_north_km = max(
                        min(north_km + north_step_km, half_height),
                        -half_height,
                    )

                self._user_positions[uid] = _offset_from_ground_point(
                    self._step_cfg.user_lat_deg,
                    self._step_cfg.user_lon_deg,
                    next_east_km,
                    next_north_km,
                )
            else:
                dlat = north_step_km / 111.32
                dlon = east_step_km / (
                    111.32 * max(math.cos(math.radians(lat)), 0.01)
                )
                self._user_positions[uid] = (lat + dlat, lon + dlon)

            self._user_headings[uid] = heading_rad

    # -- diagnostics ----------------------------------------------------------

    def _emit_diagnostics(self, rng: Generator) -> DiagnosticsReport:
        """Produce a diagnostics report for reward-scale inspection.

        Uses a zenith-case channel computation and typical reward
        scenarios to expose magnitude relationships.
        """
        # Zenith case: slant_range = altitude (satellite directly overhead)
        alt = self._orbit.config.altitude_km
        zenith_tx_power_w = _diagnostic_beam_power_w(
            1.0,
            channel_tx_power_w=self._channel_cfg.tx_power_w,
            power_surface_config=self._power_surface_cfg,
        )
        zenith_ch = compute_channel(
            slant_range_km=alt,
            altitude_km=alt,
            config=self._channel_cfg,
            fading=False,
            tx_power_w=zenith_tx_power_w,
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
        typical_tx_power_w = _diagnostic_beam_power_w(
            typical_users_per_beam,
            channel_tx_power_w=self._channel_cfg.tx_power_w,
            power_surface_config=self._power_surface_cfg,
        )
        typical_snr = (
            0.0
            if zenith_tx_power_w <= 0.0
            else zenith_ch.snr_linear * (typical_tx_power_w / zenith_tx_power_w)
        )
        sample_r1 = (bw / typical_users_per_beam) * math.log2(
            1.0 + typical_snr
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


def _beam_transmit_power_w(
    beam_loads: np.ndarray,
    *,
    channel_tx_power_w: float,
    power_surface_config: PowerSurfaceConfig,
) -> np.ndarray:
    """Return explicit per-beam transmit power in linear W.

    ``static-config`` deliberately mirrors the frozen MODQN baseline. Phase
    02B ``active-load-concave`` remains a synthesized proxy. Phase 03C-B
    ``phase-03c-b-power-codebook`` is a new-extension / HOBS-inspired
    centralized discrete controller, not a paper-backed HOBS optimizer.
    """
    if (
        power_surface_config.hobs_power_surface_mode
        == HOBS_POWER_SURFACE_STATIC_CONFIG
    ):
        return np.full_like(
            beam_loads,
            fill_value=float(channel_tx_power_w),
            dtype=np.float64,
        )

    if (
        power_surface_config.hobs_power_surface_mode
        == HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK
    ):
        return _power_codebook_beam_transmit_power_w(
            beam_loads,
            power_surface_config=power_surface_config,
        )

    if (
        power_surface_config.hobs_power_surface_mode
        != HOBS_POWER_SURFACE_ACTIVE_LOAD_CONCAVE
    ):
        raise ValueError(
            "Unsupported hobs_power_surface_mode: "
            f"{power_surface_config.hobs_power_surface_mode!r}"
        )

    loads = beam_loads.astype(np.float64, copy=False)
    powers = np.zeros_like(loads, dtype=np.float64)
    active = loads > 0.0
    if not np.any(active):
        return powers

    active_power = (
        power_surface_config.active_base_power_w
        + power_surface_config.load_scale_power_w
        * np.power(loads[active], power_surface_config.load_exponent)
    )
    if power_surface_config.max_power_w is not None:
        active_power = np.minimum(active_power, float(power_surface_config.max_power_w))
    powers[active] = active_power
    return powers


def _nearest_codebook_level(value: float, levels: tuple[float, ...]) -> float:
    """Return the closest configured discrete linear-W power level."""
    return min(levels, key=lambda level: (abs(float(level) - value), float(level)))


def _power_codebook_beam_transmit_power_w(
    beam_loads: np.ndarray,
    *,
    power_surface_config: PowerSurfaceConfig,
) -> np.ndarray:
    """Apply the Phase 03C-B centralized discrete codebook controller.

    The controller observes post-handover beam loads and chooses one linear-W
    level for each active beam. Inactive beams are always ``0 W``. This is
    intentionally a bounded audit/control surface, not a learned optimizer.
    """
    loads = beam_loads.astype(np.float64, copy=False)
    powers = np.zeros_like(loads, dtype=np.float64)
    active = loads > 0.0
    if not np.any(active):
        return powers

    levels = tuple(float(level) for level in power_surface_config.power_codebook_levels_w)
    low = levels[0]
    mid = levels[len(levels) // 2]
    high = levels[-1]
    profile = power_surface_config.power_codebook_profile

    active_loads = loads[active]
    if profile == "fixed-low":
        active_power = np.full_like(active_loads, low, dtype=np.float64)
    elif profile == "fixed-mid":
        active_power = np.full_like(active_loads, mid, dtype=np.float64)
    elif profile == "fixed-high":
        active_power = np.full_like(active_loads, high, dtype=np.float64)
    elif profile == "load-concave":
        raw = (
            power_surface_config.active_base_power_w
            + power_surface_config.load_scale_power_w
            * np.power(active_loads, power_surface_config.load_exponent)
        )
        if power_surface_config.max_power_w is not None:
            raw = np.minimum(raw, float(power_surface_config.max_power_w))
        active_power = np.asarray(
            [_nearest_codebook_level(float(value), levels) for value in raw],
            dtype=np.float64,
        )
    elif profile == "qos-tail-boost":
        active_power = np.full_like(active_loads, mid, dtype=np.float64)
        if active_loads.size == 1:
            active_power[0] = high
        else:
            tail_threshold = float(np.percentile(active_loads, 75))
            light_threshold = float(np.percentile(active_loads, 25))
            active_power[active_loads >= tail_threshold] = high
            active_power[active_loads <= light_threshold] = low
    elif profile == "budget-trim":
        active_power = _budget_trim_power_levels(
            active_loads,
            levels=levels,
            budget_w=power_surface_config.total_power_budget_w,
        )
    else:
        raise ValueError(f"Unsupported power_codebook_profile: {profile!r}")

    powers[active] = active_power
    return powers


def _budget_trim_power_levels(
    active_loads: np.ndarray,
    *,
    levels: tuple[float, ...],
    budget_w: float | None,
) -> np.ndarray:
    """Start high and demote lower-load beams until the budget can fit."""
    powers = np.full_like(active_loads, levels[-1], dtype=np.float64)
    if budget_w is None or float(np.sum(powers, dtype=np.float64)) <= budget_w:
        return powers

    level_indices = np.full(active_loads.shape, len(levels) - 1, dtype=np.int32)
    # Preserve capacity for the most loaded beams first; trim lower-load beams.
    demotion_order = np.argsort(active_loads, kind="mergesort")
    while float(np.sum(powers, dtype=np.float64)) > budget_w:
        changed = False
        for idx in demotion_order:
            if level_indices[idx] <= 0:
                continue
            level_indices[idx] -= 1
            powers[idx] = levels[int(level_indices[idx])]
            changed = True
            if float(np.sum(powers, dtype=np.float64)) <= budget_w:
                return powers
        if not changed:
            return powers
    return powers


def _diagnostic_beam_power_w(
    beam_load: float,
    *,
    channel_tx_power_w: float,
    power_surface_config: PowerSurfaceConfig,
) -> float:
    loads = np.array([float(beam_load)], dtype=np.float64)
    return float(
        _beam_transmit_power_w(
            loads,
            channel_tx_power_w=channel_tx_power_w,
            power_surface_config=power_surface_config,
        )[0]
    )


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
    width_km: float = 0.0,
    height_km: float = 0.0,
) -> list[tuple[float, float]]:
    """Generate user ground positions scattered around a center point.

    ASSUME-MODQN-REP-021: the resolved config owns the scatter radius and
    distribution. The module-level constant is only the default value.
    """
    if distribution not in {"uniform-circular", "uniform-rectangle"}:
        raise ValueError(f"Unsupported user scatter distribution: {distribution!r}")

    positions: list[tuple[float, float]] = []
    for _ in range(num_users):
        if distribution == "uniform-circular":
            r = radius_km * math.sqrt(float(rng.random()))
            theta = float(rng.random()) * 2.0 * math.pi
            dlat = r * math.cos(theta) / 111.32
            dlon = r * math.sin(theta) / (
                111.32 * max(math.cos(math.radians(center_lat)), 0.01)
            )
            positions.append((center_lat + dlat, center_lon + dlon))
        else:
            east_km = float(rng.uniform(-width_km / 2.0, width_km / 2.0))
            north_km = float(rng.uniform(-height_km / 2.0, height_km / 2.0))
            positions.append(
                _offset_from_ground_point(center_lat, center_lon, east_km, north_km)
            )
    return positions


def _generate_user_headings(
    config: StepConfig,
    rng: Generator,
) -> list[float]:
    """Generate per-user headings for the active mobility model."""
    if config.mobility_model == "random-wandering":
        return [
            float(rng.uniform(0.0, 2.0 * math.pi))
            for _ in range(config.num_users)
        ]
    return [
        (uid * config.user_heading_stride_rad) % (2.0 * math.pi)
        for uid in range(config.num_users)
    ]


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


def _offset_from_ground_point(
    center_lat: float,
    center_lon: float,
    east_km: float,
    north_km: float,
) -> tuple[float, float]:
    """Convert local-tangent offsets back to geodetic lat/lon."""
    lat = center_lat + north_km / 111.32
    lon = center_lon + east_km / (
        111.32 * max(math.cos(math.radians(center_lat)), 0.01)
    )
    return (lat, lon)


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
