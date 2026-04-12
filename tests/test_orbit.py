"""Unit tests for env.orbit — the single-plane circular orbit proxy.

Validates geometry, visibility, and ordering contracts required by
ASSUME-MODQN-REP-001 and ASSUME-MODQN-REP-012.
"""

from __future__ import annotations

import math
import unittest

from modqn_paper_reproduction.env.orbit import (
    EARTH_RADIUS_KM,
    OrbitConfig,
    OrbitProxy,
    SatelliteSnapshot,
)


class TestOrbitConfig(unittest.TestCase):
    """OrbitConfig defaults and derived properties."""

    def test_defaults_match_paper_baseline(self) -> None:
        cfg = OrbitConfig()
        self.assertEqual(cfg.altitude_km, 780.0)
        self.assertEqual(cfg.num_satellites, 4)
        self.assertEqual(cfg.num_planes, 1)
        self.assertEqual(cfg.satellite_speed_km_s, 7.4)
        self.assertEqual(cfg.min_elevation_deg, 0.0)

    def test_orbit_radius(self) -> None:
        cfg = OrbitConfig()
        self.assertAlmostEqual(cfg.orbit_radius_km, EARTH_RADIUS_KM + 780.0)

    def test_in_plane_spacing(self) -> None:
        cfg = OrbitConfig()
        self.assertAlmostEqual(cfg.in_plane_spacing_deg, 90.0)

    def test_angular_velocity_from_paper_speed(self) -> None:
        cfg = OrbitConfig()
        expected = 7.4 / (EARTH_RADIUS_KM + 780.0)
        self.assertAlmostEqual(cfg.angular_velocity_rad_s, expected, places=10)

    def test_orbital_period_positive(self) -> None:
        cfg = OrbitConfig()
        self.assertGreater(cfg.orbital_period_s, 0)
        # Roughly ~100 minutes for LEO
        self.assertGreater(cfg.orbital_period_s, 5000)
        self.assertLess(cfg.orbital_period_s, 7000)

    def test_rejects_zero_satellites(self) -> None:
        with self.assertRaises(ValueError):
            OrbitConfig(num_satellites=0)

    def test_rejects_multi_plane(self) -> None:
        with self.assertRaises(ValueError):
            OrbitConfig(num_planes=2)


class TestSatellitePositions(unittest.TestCase):
    """Satellite positioning and ordering."""

    def setUp(self) -> None:
        self.proxy = OrbitProxy()

    def test_four_satellites_at_t0(self) -> None:
        sats = self.proxy.all_satellites(0.0)
        self.assertEqual(len(sats), 4)
        for i, s in enumerate(sats):
            self.assertEqual(s.index, i)

    def test_stable_index_order(self) -> None:
        """Index order must be stable across time steps."""
        for t in [0.0, 1.0, 5.0, 100.0]:
            sats = self.proxy.all_satellites(t)
            indices = [s.index for s in sats]
            self.assertEqual(indices, [0, 1, 2, 3])

    def test_equally_spaced_at_t0(self) -> None:
        """Satellites should be 90° apart at t=0."""
        sats = self.proxy.all_satellites(0.0)
        anomalies = [s.true_anomaly_deg for s in sats]
        for i in range(4):
            expected = i * 90.0
            self.assertAlmostEqual(anomalies[i], expected, places=6)

    def test_all_on_orbit_radius(self) -> None:
        """Each satellite must lie on the orbit sphere."""
        r = self.proxy.config.orbit_radius_km
        for t in [0.0, 3.0, 7.5]:
            for s in self.proxy.all_satellites(t):
                dist = math.sqrt(s.x_km**2 + s.y_km**2 + s.z_km**2)
                self.assertAlmostEqual(dist, r, places=6)

    def test_motion_direction(self) -> None:
        """True anomaly should increase with time."""
        ta_0 = self.proxy.satellite_true_anomaly_deg(0, 0.0)
        ta_1 = self.proxy.satellite_true_anomaly_deg(0, 1.0)
        # For small dt, anomaly increases (no wrap expected in 1 second)
        self.assertGreater(ta_1, ta_0)

    def test_angular_velocity_matches_config(self) -> None:
        """Position change over 1 second matches config angular velocity."""
        cfg = self.proxy.config
        ta_0 = self.proxy.satellite_true_anomaly_deg(0, 0.0)
        ta_1 = self.proxy.satellite_true_anomaly_deg(0, 1.0)
        delta_deg = ta_1 - ta_0
        expected_deg = math.degrees(cfg.angular_velocity_rad_s)
        self.assertAlmostEqual(delta_deg, expected_deg, places=8)

    def test_full_orbit_wraps(self) -> None:
        """After one orbital period, satellite returns to (nearly) the same position."""
        T = self.proxy.config.orbital_period_s
        ta_start = self.proxy.satellite_true_anomaly_deg(0, 0.0)
        ta_end = self.proxy.satellite_true_anomaly_deg(0, T)
        # Should wrap back to ~0
        self.assertAlmostEqual(ta_end, ta_start, places=4)


class TestPolarOrbitGeometry(unittest.TestCase):
    """Geometry sanity for the default 90° inclination polar orbit."""

    def setUp(self) -> None:
        self.proxy = OrbitProxy()  # default: i=90°, RAAN=0°

    def test_sat0_at_t0_on_equator(self) -> None:
        """Sat 0 at t=0 should be at ascending node (equator, lon=0)."""
        s = self.proxy.satellite_position(0, 0.0)
        self.assertAlmostEqual(s.lat_deg, 0.0, places=6)
        self.assertAlmostEqual(s.lon_deg, 0.0, places=6)

    def test_sat1_at_t0_at_north_pole(self) -> None:
        """Sat 1 at t=0 should be at 90° true anomaly → north pole for polar orbit."""
        s = self.proxy.satellite_position(1, 0.0)
        self.assertAlmostEqual(s.lat_deg, 90.0, places=4)

    def test_sat2_at_t0_at_equator_180(self) -> None:
        """Sat 2 at t=0 should be at descending node (equator, lon=180)."""
        s = self.proxy.satellite_position(2, 0.0)
        self.assertAlmostEqual(s.lat_deg, 0.0, places=6)
        self.assertAlmostEqual(abs(s.lon_deg), 180.0, places=4)

    def test_sat3_at_t0_at_south_pole(self) -> None:
        """Sat 3 at t=0 should be at 270° true anomaly → south pole."""
        s = self.proxy.satellite_position(3, 0.0)
        self.assertAlmostEqual(s.lat_deg, -90.0, places=4)


class TestElevationAndVisibility(unittest.TestCase):
    """Elevation angle, slant range, and visibility filtering."""

    def setUp(self) -> None:
        self.proxy = OrbitProxy()

    def test_zenith_elevation(self) -> None:
        """Satellite directly overhead should have elevation ~90°."""
        # Sat 0 at t=0 is at (lat=0, lon=0) for polar orbit.
        # User at (0, 0) should see it at zenith.
        sat = self.proxy.satellite_position(0, 0.0)
        vr = self.proxy.elevation_and_range(0.0, 0.0, sat)
        self.assertAlmostEqual(vr.elevation_deg, 90.0, places=2)
        self.assertTrue(vr.is_visible)

    def test_zenith_slant_range_equals_altitude(self) -> None:
        """Slant range to zenith satellite equals the orbit altitude."""
        sat = self.proxy.satellite_position(0, 0.0)
        vr = self.proxy.elevation_and_range(0.0, 0.0, sat)
        self.assertAlmostEqual(vr.slant_range_km, 780.0, places=0)

    def test_antipodal_not_visible(self) -> None:
        """Satellite on the opposite side of Earth should not be visible."""
        # Sat 0 at (0, 0), user at (0, 180): opposite side
        sat = self.proxy.satellite_position(0, 0.0)
        vr = self.proxy.elevation_and_range(0.0, 180.0, sat)
        self.assertFalse(vr.is_visible)
        self.assertLess(vr.elevation_deg, 0.0)

    def test_horizon_geometry(self) -> None:
        """Near-horizon satellite should have elevation near 0° and
        slant range approximately sqrt(2*R*h + h^2)."""
        R = EARTH_RADIUS_KM
        h = 780.0
        expected_horizon_range = math.sqrt(2 * R * h + h * h)

        # Find a user position where sat 0 at t=0 is near the horizon.
        # Sat 0 is at lat=0, lon=0.  The half-angle subtended from Earth
        # center to the geometric horizon is arccos(R/(R+h)).
        half_angle_deg = math.degrees(math.acos(R / (R + h)))
        # User at (0, half_angle_deg) should be near the 0° elevation line.
        sat = self.proxy.satellite_position(0, 0.0)
        vr = self.proxy.elevation_and_range(0.0, half_angle_deg, sat)
        self.assertAlmostEqual(vr.elevation_deg, 0.0, delta=0.5)
        self.assertAlmostEqual(vr.slant_range_km, expected_horizon_range, delta=10.0)

    def test_visible_satellites_filters_correctly(self) -> None:
        """visible_satellites returns only those above min_elevation."""
        # User at (0, 0) at t=0: sat 0 is overhead, sat 2 is antipodal.
        vis = self.proxy.visible_satellites(0.0, 0.0, 0.0)
        vis_indices = [v.sat_index for v in vis]
        self.assertIn(0, vis_indices)
        self.assertNotIn(2, vis_indices)

    def test_all_visibility_returns_all_sats(self) -> None:
        """all_visibility always returns exactly N entries in index order."""
        all_v = self.proxy.all_visibility(0.0, 0.0, 0.0)
        self.assertEqual(len(all_v), 4)
        self.assertEqual([v.sat_index for v in all_v], [0, 1, 2, 3])

    def test_min_elevation_threshold(self) -> None:
        """A satellite near the horizon is visible at el_min=0° but
        excluded when the threshold is raised above its elevation."""
        # Sat 0 at t=0 is at sub-satellite point (0°N, 0°E).
        # A user at (0°N, 25°E) sees it at a low positive elevation
        # (horizon ≈ 27° central angle for h=780 km).
        sat = self.proxy.satellite_position(0, 0.0)
        vr = self.proxy.elevation_and_range(0.0, 25.0, sat)

        # Confirm: low positive elevation (geometry sanity)
        self.assertGreater(vr.elevation_deg, 0.0)
        self.assertLess(vr.elevation_deg, 20.0)
        self.assertTrue(vr.is_visible)

        # Strict threshold above the measured elevation → same geometry, not visible
        strict_cfg = OrbitConfig(min_elevation_deg=vr.elevation_deg + 5.0)
        strict_proxy = OrbitProxy(strict_cfg)
        strict_vr = strict_proxy.elevation_and_range(0.0, 25.0, sat)
        self.assertAlmostEqual(strict_vr.elevation_deg, vr.elevation_deg, places=6)
        self.assertFalse(strict_vr.is_visible)


class TestSatelliteCountSweep(unittest.TestCase):
    """Orbit proxy must support the paper's satellite-count sweep (2..8)."""

    def test_sweep_range(self) -> None:
        for n in [2, 3, 4, 5, 6, 7, 8]:
            cfg = OrbitConfig(num_satellites=n)
            proxy = OrbitProxy(cfg)
            sats = proxy.all_satellites(0.0)
            self.assertEqual(len(sats), n)
            # Spacing should be 360/n
            if n > 1:
                spacing = sats[1].true_anomaly_deg - sats[0].true_anomaly_deg
                self.assertAlmostEqual(spacing, 360.0 / n, places=6)


class TestReproducibility(unittest.TestCase):
    """Determinism: same config + time → identical output."""

    def test_deterministic(self) -> None:
        p1 = OrbitProxy()
        p2 = OrbitProxy()
        for t in [0.0, 1.0, 5.5, 100.0]:
            s1 = p1.all_satellites(t)
            s2 = p2.all_satellites(t)
            for a, b in zip(s1, s2):
                self.assertEqual(a, b)


class TestCoverageAreaVisibility(unittest.TestCase):
    """Smoke: user on the ground track sees satellites during an orbit."""

    def test_default_ground_track_visibility(self) -> None:
        """With default RAAN=0°, a user on the ground track (40°N, 0°E)
        should see at least one satellite during one orbital period."""
        proxy = OrbitProxy()
        T = proxy.config.orbital_period_s
        found = False
        steps = int(T / 10)
        for step in range(steps):
            t = step * 10.0
            vis = proxy.visible_satellites(40.0, 0.0, t)
            if len(vis) > 0:
                found = True
                break
        self.assertTrue(found, "No satellite visible from (40N, 0E) during an entire orbit")

    def test_raan_aligned_with_coverage_center(self) -> None:
        """With RAAN=116°, the ground track passes over the paper's
        coverage center (40°N, 116°E) and satellites are visible."""
        cfg = OrbitConfig(raan_deg=116.0)
        proxy = OrbitProxy(cfg)
        T = proxy.config.orbital_period_s
        found = False
        steps = int(T / 10)
        for step in range(steps):
            t = step * 10.0
            vis = proxy.visible_satellites(40.0, 116.0, t)
            if len(vis) > 0:
                found = True
                break
        self.assertTrue(
            found,
            "No satellite visible from (40N, 116E) with RAAN=116° during an entire orbit",
        )


if __name__ == "__main__":
    unittest.main()
