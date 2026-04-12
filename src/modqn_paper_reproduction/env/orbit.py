"""Orbit geometry proxy for MODQN paper reproduction.

Implements the single circular orbital plane proxy defined by
ASSUME-MODQN-REP-001 (Walker-like δ(4/1/0) at 780 km) and
provides the visibility/eligibility surface required by
ASSUME-MODQN-REP-012 (elevation > 0° line-of-sight).

Satellite indices 0..N-1 are stable across time steps and form
the ordering basis for the satellite-major, beam-minor action
catalog defined in the SDD §3.3A.

Provenance: reproduction-assumption (orbit layout is not fully
disclosed in the paper; STK was used but exact shell parameters
are not given).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EARTH_RADIUS_KM: float = 6_371.0
"""WGS-84 mean Earth radius. Sufficient for the paper's FSPL-level model."""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrbitConfig:
    """Orbit configuration aligned with the resolved-run template.

    Paper-backed values: altitude_km, satellite_speed_km_s, num_satellites.
    Reproduction-assumption values: inclination_deg, raan_deg,
    min_elevation_deg, num_planes (all under ASSUME-MODQN-REP-001/012).
    """

    altitude_km: float = 780.0
    num_satellites: int = 4
    num_planes: int = 1
    satellite_speed_km_s: float = 7.4
    inclination_deg: float = 90.0
    raan_deg: float = 0.0
    min_elevation_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.num_satellites < 1:
            raise ValueError(f"num_satellites must be >= 1, got {self.num_satellites}")
        if self.num_planes != 1:
            raise ValueError(
                f"Only single-plane proxy is supported (ASSUME-MODQN-REP-001), "
                f"got num_planes={self.num_planes}"
            )
        if self.altitude_km <= 0:
            raise ValueError(f"altitude_km must be > 0, got {self.altitude_km}")

    @property
    def orbit_radius_km(self) -> float:
        return EARTH_RADIUS_KM + self.altitude_km

    @property
    def angular_velocity_rad_s(self) -> float:
        """Derived from paper-backed satellite speed, not Kepler μ."""
        return self.satellite_speed_km_s / self.orbit_radius_km

    @property
    def in_plane_spacing_deg(self) -> float:
        return 360.0 / self.num_satellites

    @property
    def orbital_period_s(self) -> float:
        return 2.0 * math.pi / self.angular_velocity_rad_s


@dataclass(frozen=True)
class SatelliteSnapshot:
    """Position snapshot of one satellite at a specific time.

    Coordinates are in a simplified ECI frame (no Earth rotation,
    no precession — sufficient for the paper's short-episode model).
    """

    index: int
    true_anomaly_deg: float
    x_km: float
    y_km: float
    z_km: float
    lat_deg: float
    lon_deg: float


@dataclass(frozen=True)
class VisibilityResult:
    """Elevation and range from a ground point to one satellite."""

    sat_index: int
    elevation_deg: float
    slant_range_km: float
    is_visible: bool


# ---------------------------------------------------------------------------
# Orbit proxy
# ---------------------------------------------------------------------------


class OrbitProxy:
    """Single-plane circular orbit proxy.

    Provides deterministic, reproducible satellite positions and
    visibility queries.  The satellite index ``0..N-1`` is stable
    across all time steps, satisfying the ordering contract for the
    ``satellite-major, beam-minor`` action catalog.

    The model deliberately omits Earth rotation, J2, and drag because:
    1. the paper uses 10-second episodes where these are negligible;
    2. the paper's own model uses STK output but does not describe
       perturbation settings.
    """

    def __init__(self, config: OrbitConfig | None = None) -> None:
        self._cfg = config or OrbitConfig()
        r = self._cfg.orbit_radius_km
        omega = self._cfg.angular_velocity_rad_s

        # Cache frequently used values
        self._r = r
        self._omega = omega

        # Build orbital-plane → ECI rotation matrix.
        # Convention: Rz(RAAN) · Rx(inclination)
        i_rad = math.radians(self._cfg.inclination_deg)
        raan_rad = math.radians(self._cfg.raan_deg)
        ci, si = math.cos(i_rad), math.sin(i_rad)
        cr, sr = math.cos(raan_rad), math.sin(raan_rad)

        # Row-major 3×3 stored as tuple-of-tuples for immutability.
        self._rot: tuple[tuple[float, float, float], ...] = (
            (cr, -sr * ci, sr * si),
            (sr, cr * ci, -cr * si),
            (0.0, si, ci),
        )

    # -- public properties ---------------------------------------------------

    @property
    def config(self) -> OrbitConfig:
        return self._cfg

    @property
    def num_satellites(self) -> int:
        return self._cfg.num_satellites

    # -- position queries ----------------------------------------------------

    def satellite_true_anomaly_deg(self, sat_index: int, t_s: float) -> float:
        """True anomaly of *sat_index* at time *t_s* (seconds), in [0, 360)."""
        spacing = self._cfg.in_plane_spacing_deg
        theta_0 = sat_index * spacing
        omega_deg_s = math.degrees(self._omega)
        return (theta_0 + omega_deg_s * t_s) % 360.0

    def satellite_position(self, sat_index: int, t_s: float) -> SatelliteSnapshot:
        """ECI position of one satellite at time *t_s*."""
        ta_deg = self.satellite_true_anomaly_deg(sat_index, t_s)
        ta_rad = math.radians(ta_deg)

        # Position in the orbital plane
        x_op = self._r * math.cos(ta_rad)
        y_op = self._r * math.sin(ta_rad)

        # Rotate to ECI
        R = self._rot
        x = R[0][0] * x_op + R[0][1] * y_op
        y = R[1][0] * x_op + R[1][1] * y_op
        z = R[2][0] * x_op + R[2][1] * y_op

        # Sub-satellite point (geocentric lat/lon)
        lon = math.degrees(math.atan2(y, x))
        lat = math.degrees(math.asin(max(-1.0, min(1.0, z / self._r))))

        return SatelliteSnapshot(
            index=sat_index,
            true_anomaly_deg=ta_deg,
            x_km=x,
            y_km=y,
            z_km=z,
            lat_deg=lat,
            lon_deg=lon,
        )

    def all_satellites(self, t_s: float) -> list[SatelliteSnapshot]:
        """All satellite positions at *t_s*, in stable index order 0..N-1."""
        return [
            self.satellite_position(i, t_s)
            for i in range(self._cfg.num_satellites)
        ]

    # -- visibility queries --------------------------------------------------

    def elevation_and_range(
        self,
        user_lat_deg: float,
        user_lon_deg: float,
        sat: SatelliteSnapshot,
    ) -> VisibilityResult:
        """Elevation angle and slant range from ground user to satellite."""
        ux, uy, uz = ground_eci(user_lat_deg, user_lon_deg)

        dx = sat.x_km - ux
        dy = sat.y_km - uy
        dz = sat.z_km - uz
        slant = math.sqrt(dx * dx + dy * dy + dz * dz)

        # Local zenith unit vector at user
        r_u = EARTH_RADIUS_KM  # user is on the surface
        zx, zy, zz = ux / r_u, uy / r_u, uz / r_u

        # sin(elevation) = cos(zenith_angle) = dot(d_hat, zenith_hat)
        sin_el = (dx * zx + dy * zy + dz * zz) / slant
        sin_el = max(-1.0, min(1.0, sin_el))
        el_deg = math.degrees(math.asin(sin_el))

        return VisibilityResult(
            sat_index=sat.index,
            elevation_deg=el_deg,
            slant_range_km=slant,
            is_visible=el_deg >= self._cfg.min_elevation_deg,
        )

    def visible_satellites(
        self,
        user_lat_deg: float,
        user_lon_deg: float,
        t_s: float,
    ) -> list[VisibilityResult]:
        """Visible satellites from a ground user, in stable index order.

        Only returns entries where ``elevation >= min_elevation_deg``.
        """
        return [
            vr
            for vr in self.all_visibility(user_lat_deg, user_lon_deg, t_s)
            if vr.is_visible
        ]

    def all_visibility(
        self,
        user_lat_deg: float,
        user_lon_deg: float,
        t_s: float,
    ) -> list[VisibilityResult]:
        """Visibility results for ALL satellites (including non-visible),
        in stable index order.  Used for building the full action mask."""
        sats = self.all_satellites(t_s)
        return [
            self.elevation_and_range(user_lat_deg, user_lon_deg, s)
            for s in sats
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ground_eci(lat_deg: float, lon_deg: float) -> tuple[float, float, float]:
    """Geocentric ECI position of a ground point on Earth's surface."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    R = EARTH_RADIUS_KM
    return (
        R * math.cos(lat) * math.cos(lon),
        R * math.cos(lat) * math.sin(lon),
        R * math.sin(lat),
    )
