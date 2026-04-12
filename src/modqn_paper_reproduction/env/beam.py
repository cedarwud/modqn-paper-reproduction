"""Hex-7 beam geometry for MODQN paper reproduction.

Implements the beam layout defined by ASSUME-MODQN-REP-002:
  - 7 beams per satellite: 1 nadir center + 6 ring beams
  - Ring beams indexed clockwise (viewed from satellite looking down)
  - Ring beams spaced θ_3dB from center beam boresight

Beam indices 0..6 are stable and form the beam-minor axis of the
``satellite-major, beam-minor`` action catalog (SDD §3.3A).

Note on beam gain: the paper (PAP-2024-MORL-MULTIBEAM) does *not*
model off-axis beam gain — channel quality depends on user-to-satellite
slant range, atmospheric loss, and Rician fading, but not on the
angular offset from beam boresight.  Off-axis angle is provided here
for state encoding (Γ(t)) and future extensions, not for channel
computation in Phase 1.

Provenance: reproduction-assumption (beam footprint geometry and
θ_3dB are not specified in the paper).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .orbit import EARTH_RADIUS_KM, SatelliteSnapshot, ground_eci

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Vec3 = tuple[float, float, float]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BeamConfig:
    """Beam pattern configuration.

    Paper-backed: beams_per_satellite = 7.
    Reproduction-assumption (ASSUME-MODQN-REP-002): theta_3db_deg.
    """

    beams_per_satellite: int = 7
    theta_3db_deg: float = 2.0

    def __post_init__(self) -> None:
        if self.beams_per_satellite != 7:
            raise ValueError(
                f"Only hex-7 layout is supported (ASSUME-MODQN-REP-002), "
                f"got beams_per_satellite={self.beams_per_satellite}"
            )
        if self.theta_3db_deg <= 0:
            raise ValueError(f"theta_3db_deg must be > 0, got {self.theta_3db_deg}")


@dataclass(frozen=True)
class BeamCenterGround:
    """Ground position of one beam center."""

    local_beam_index: int
    lat_deg: float
    lon_deg: float


# ---------------------------------------------------------------------------
# Beam pattern
# ---------------------------------------------------------------------------


_NUM_RING_BEAMS = 6
_RING_AZIMUTH_STEP_DEG = 360.0 / _NUM_RING_BEAMS  # 60°


class BeamPattern:
    """Hex-7 beam pattern per ASSUME-MODQN-REP-002.

    Beam ordering (stable, deterministic):
      - beam 0: center (nadir boresight)
      - beams 1-6: ring, clockwise from the local-frame reference axis

    The local frame at each satellite uses:
      - nadir: satellite → Earth center
      - e1: projection of the celestial north pole onto the plane ⊥ nadir
             (falls back to x-axis if satellite is near a pole)
      - e2: nadir × e1 (completes the right-handed frame)
    """

    def __init__(self, config: BeamConfig | None = None) -> None:
        self._cfg = config or BeamConfig()
        self._theta_rad = math.radians(self._cfg.theta_3db_deg)
        self._ring_azimuths_rad: tuple[float, ...] = tuple(
            math.radians(k * _RING_AZIMUTH_STEP_DEG) for k in range(_NUM_RING_BEAMS)
        )

    # -- properties ----------------------------------------------------------

    @property
    def config(self) -> BeamConfig:
        return self._cfg

    @property
    def num_beams(self) -> int:
        return self._cfg.beams_per_satellite

    # -- beam directions (ECI unit vectors) ----------------------------------

    def beam_directions_eci(self, sat: SatelliteSnapshot) -> list[Vec3]:
        """Unit vectors from satellite toward each beam center, in ECI.

        Returns 7 directions in stable beam-index order.
        """
        nadir, e1, e2 = _satellite_local_frame(sat)
        ct = math.cos(self._theta_rad)
        st = math.sin(self._theta_rad)

        dirs: list[Vec3] = [nadir]  # beam 0 = center
        for phi_rad in self._ring_azimuths_rad:
            cp, sp = math.cos(phi_rad), math.sin(phi_rad)
            dx = ct * nadir[0] + st * (cp * e1[0] + sp * e2[0])
            dy = ct * nadir[1] + st * (cp * e1[1] + sp * e2[1])
            dz = ct * nadir[2] + st * (cp * e1[2] + sp * e2[2])
            dirs.append((dx, dy, dz))
        return dirs

    # -- beam center ground positions ----------------------------------------

    def beam_centers_ground(
        self, sat: SatelliteSnapshot
    ) -> list[BeamCenterGround]:
        """Ground (lat, lon) of each beam center for one satellite.

        Returns 7 entries in stable beam-index order.
        Center beam ground position ≈ sub-satellite point.
        """
        dirs = self.beam_directions_eci(sat)
        origin = (sat.x_km, sat.y_km, sat.z_km)
        result: list[BeamCenterGround] = []
        for idx, d in enumerate(dirs):
            hit = _ray_earth_intersection(origin, d)
            if hit is not None:
                result.append(BeamCenterGround(idx, hit[0], hit[1]))
            else:
                # Beam misses the Earth — should not happen for reasonable θ_3dB
                result.append(BeamCenterGround(idx, float("nan"), float("nan")))
        return result

    # -- off-axis angle ------------------------------------------------------

    def off_axis_angle_deg(
        self,
        sat: SatelliteSnapshot,
        local_beam_index: int,
        user_lat_deg: float,
        user_lon_deg: float,
    ) -> float:
        """Angle between beam boresight and the satellite-to-user direction.

        This is provided for state encoding and future reference.
        The paper's channel model does not use beam gain based on
        off-axis angle.
        """
        ux, uy, uz = ground_eci(user_lat_deg, user_lon_deg)
        dx = ux - sat.x_km
        dy = uy - sat.y_km
        dz = uz - sat.z_km
        d_len = math.sqrt(dx * dx + dy * dy + dz * dz)
        dx, dy, dz = dx / d_len, dy / d_len, dz / d_len

        bx, by, bz = self.beam_directions_eci(sat)[local_beam_index]

        dot = dx * bx + dy * by + dz * bz
        dot = max(-1.0, min(1.0, dot))
        return math.degrees(math.acos(dot))

    # -- nearest beam --------------------------------------------------------

    def nearest_beam(
        self,
        sat: SatelliteSnapshot,
        user_lat_deg: float,
        user_lon_deg: float,
    ) -> int:
        """Local beam index whose boresight is closest to the user direction.

        Uses minimum off-axis angle (angular distance in 3D).
        """
        ux, uy, uz = ground_eci(user_lat_deg, user_lon_deg)
        dx = ux - sat.x_km
        dy = uy - sat.y_km
        dz = uz - sat.z_km
        d_len = math.sqrt(dx * dx + dy * dy + dz * dz)
        dx, dy, dz = dx / d_len, dy / d_len, dz / d_len

        dirs = self.beam_directions_eci(sat)
        best_idx = 0
        best_dot = -2.0
        for idx, (bx, by, bz) in enumerate(dirs):
            dot = dx * bx + dy * by + dz * bz
            if dot > best_dot:
                best_dot = dot
                best_idx = idx
        return best_idx


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _satellite_local_frame(
    sat: SatelliteSnapshot,
) -> tuple[Vec3, Vec3, Vec3]:
    """Build a right-handed local frame at the satellite.

    Returns ``(nadir, e1, e2)`` where *nadir* points from satellite
    toward Earth center and *e1*, *e2* span the plane perpendicular
    to nadir.
    """
    r = math.sqrt(sat.x_km ** 2 + sat.y_km ** 2 + sat.z_km ** 2)
    nx, ny, nz = -sat.x_km / r, -sat.y_km / r, -sat.z_km / r

    # Project celestial north (0, 0, 1) onto the plane ⊥ nadir.
    dot_zn = nz
    px, py, pz = -dot_zn * nx, -dot_zn * ny, 1.0 - dot_zn * nz
    p_len = math.sqrt(px * px + py * py + pz * pz)

    if p_len < 1e-10:
        # Satellite near a pole — fall back to x-axis projection.
        dot_xn = nx
        px, py, pz = 1.0 - dot_xn * nx, -dot_xn * ny, -dot_xn * nz
        p_len = math.sqrt(px * px + py * py + pz * pz)

    e1x, e1y, e1z = px / p_len, py / p_len, pz / p_len

    # e2 = nadir × e1 (completes right-handed frame)
    e2x = ny * e1z - nz * e1y
    e2y = nz * e1x - nx * e1z
    e2z = nx * e1y - ny * e1x

    return (nx, ny, nz), (e1x, e1y, e1z), (e2x, e2y, e2z)


def _ray_earth_intersection(
    origin: Vec3, direction: Vec3
) -> tuple[float, float] | None:
    """Intersect a ray with the Earth sphere.

    Returns ``(lat_deg, lon_deg)`` of the nearer intersection, or
    ``None`` if the ray misses.

    *direction* must be a unit vector.
    """
    ox, oy, oz = origin
    dx, dy, dz = direction
    R = EARTH_RADIUS_KM

    # Solve: |origin + t·direction|² = R²
    b_half = ox * dx + oy * dy + oz * dz
    c = ox * ox + oy * oy + oz * oz - R * R

    discriminant = b_half * b_half - c
    if discriminant < 0.0:
        return None

    sqrt_disc = math.sqrt(discriminant)
    t = -b_half - sqrt_disc  # nearer intersection
    if t < 0.0:
        t = -b_half + sqrt_disc
        if t < 0.0:
            return None

    px = ox + t * dx
    py = oy + t * dy
    pz = oz + t * dz

    lat = math.degrees(math.asin(max(-1.0, min(1.0, pz / R))))
    lon = math.degrees(math.atan2(py, px))
    return (lat, lon)
