"""Unit tests for env.beam — hex-7 beam geometry.

Validates beam layout, ground projections, off-axis angles, and
ordering contracts required by ASSUME-MODQN-REP-002.
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
from modqn_paper_reproduction.env.beam import (
    BeamCenterGround,
    BeamConfig,
    BeamPattern,
)


class TestBeamConfig(unittest.TestCase):
    """BeamConfig defaults and validation."""

    def test_defaults(self) -> None:
        cfg = BeamConfig()
        self.assertEqual(cfg.beams_per_satellite, 7)
        self.assertEqual(cfg.theta_3db_deg, 2.0)

    def test_rejects_non_seven(self) -> None:
        with self.assertRaises(ValueError):
            BeamConfig(beams_per_satellite=4)

    def test_rejects_negative_theta(self) -> None:
        with self.assertRaises(ValueError):
            BeamConfig(theta_3db_deg=-1.0)


class TestBeamDirections(unittest.TestCase):
    """Beam direction unit vectors in ECI."""

    def setUp(self) -> None:
        self.proxy = OrbitProxy()
        self.pattern = BeamPattern()

    def test_seven_directions(self) -> None:
        sat = self.proxy.satellite_position(0, 0.0)
        dirs = self.pattern.beam_directions_eci(sat)
        self.assertEqual(len(dirs), 7)

    def test_all_unit_vectors(self) -> None:
        sat = self.proxy.satellite_position(0, 0.0)
        for dx, dy, dz in self.pattern.beam_directions_eci(sat):
            length = math.sqrt(dx * dx + dy * dy + dz * dz)
            self.assertAlmostEqual(length, 1.0, places=10)

    def test_center_beam_is_nadir(self) -> None:
        """Beam 0 direction should point from satellite toward Earth center."""
        sat = self.proxy.satellite_position(0, 0.0)
        dirs = self.pattern.beam_directions_eci(sat)
        # Nadir = -sat_pos / |sat_pos|
        r = math.sqrt(sat.x_km ** 2 + sat.y_km ** 2 + sat.z_km ** 2)
        expected = (-sat.x_km / r, -sat.y_km / r, -sat.z_km / r)
        for i in range(3):
            self.assertAlmostEqual(dirs[0][i], expected[i], places=10)

    def test_ring_beams_tilted_by_theta_3db(self) -> None:
        """Each ring beam should be tilted θ_3dB from nadir."""
        sat = self.proxy.satellite_position(0, 0.0)
        dirs = self.pattern.beam_directions_eci(sat)
        nadir = dirs[0]
        theta = self.pattern.config.theta_3db_deg

        for k in range(1, 7):
            dot = sum(a * b for a, b in zip(nadir, dirs[k]))
            angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
            self.assertAlmostEqual(angle_deg, theta, places=6)

    def test_ring_beams_equally_spaced(self) -> None:
        """Adjacent ring beams should be separated by 60° in azimuth.

        We check the angular distance between consecutive ring beams.
        Since they're all at the same tilt from nadir, consecutive beams
        subtend a chord determined by the 60° azimuth step.
        """
        sat = self.proxy.satellite_position(0, 0.0)
        dirs = self.pattern.beam_directions_eci(sat)
        ring_dirs = dirs[1:]

        # Angle between consecutive ring beams
        for i in range(6):
            j = (i + 1) % 6
            dot = sum(a * b for a, b in zip(ring_dirs[i], ring_dirs[j]))
            angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
            # Expected: from spherical law of cosines on a cone at tilt θ
            # with 60° azimuth step.
            # cos(sep) = cos²(θ) + sin²(θ)*cos(60°)
            theta_rad = math.radians(self.pattern.config.theta_3db_deg)
            expected_cos = (
                math.cos(theta_rad) ** 2
                + math.sin(theta_rad) ** 2 * math.cos(math.radians(60))
            )
            expected_deg = math.degrees(math.acos(expected_cos))
            self.assertAlmostEqual(angle_deg, expected_deg, places=4)


class TestBeamCentersGround(unittest.TestCase):
    """Beam center ground projections."""

    def setUp(self) -> None:
        self.proxy = OrbitProxy()
        self.pattern = BeamPattern()

    def test_seven_centers(self) -> None:
        sat = self.proxy.satellite_position(0, 0.0)
        centers = self.pattern.beam_centers_ground(sat)
        self.assertEqual(len(centers), 7)

    def test_stable_beam_index_order(self) -> None:
        sat = self.proxy.satellite_position(0, 0.0)
        centers = self.pattern.beam_centers_ground(sat)
        indices = [c.local_beam_index for c in centers]
        self.assertEqual(indices, list(range(7)))

    def test_center_beam_near_sub_satellite_point(self) -> None:
        """Beam 0 ground position should match the satellite's sub-satellite point."""
        sat = self.proxy.satellite_position(0, 0.0)
        centers = self.pattern.beam_centers_ground(sat)
        center = centers[0]
        self.assertAlmostEqual(center.lat_deg, sat.lat_deg, places=3)
        self.assertAlmostEqual(center.lon_deg, sat.lon_deg, places=3)

    def test_ring_beams_surround_center(self) -> None:
        """Ring beam ground positions should be roughly equidistant from center."""
        sat = self.proxy.satellite_position(0, 0.0)
        centers = self.pattern.beam_centers_ground(sat)
        c_lat, c_lon = centers[0].lat_deg, centers[0].lon_deg

        ring_distances = []
        for k in range(1, 7):
            dlat = centers[k].lat_deg - c_lat
            dlon = centers[k].lon_deg - c_lon
            # Approximate great-circle distance in degrees
            dist = math.sqrt(dlat ** 2 + dlon ** 2)
            ring_distances.append(dist)

        # All ring beams should be at roughly the same distance from center
        mean_dist = sum(ring_distances) / len(ring_distances)
        for d in ring_distances:
            self.assertAlmostEqual(d, mean_dist, delta=mean_dist * 0.15)

    def test_footprint_size_scales_with_theta(self) -> None:
        """Larger θ_3dB → larger beam footprint spacing."""
        sat = self.proxy.satellite_position(0, 0.0)

        narrow = BeamPattern(BeamConfig(theta_3db_deg=1.0))
        wide = BeamPattern(BeamConfig(theta_3db_deg=4.0))

        narrow_c = narrow.beam_centers_ground(sat)
        wide_c = wide.beam_centers_ground(sat)

        def ring_spread(centers: list[BeamCenterGround]) -> float:
            c = centers[0]
            return sum(
                math.sqrt(
                    (b.lat_deg - c.lat_deg) ** 2 + (b.lon_deg - c.lon_deg) ** 2
                )
                for b in centers[1:]
            )

        self.assertGreater(ring_spread(wide_c), ring_spread(narrow_c))


class TestOffAxisAngle(unittest.TestCase):
    """Off-axis angle from beam boresight to user direction."""

    def setUp(self) -> None:
        self.proxy = OrbitProxy()
        self.pattern = BeamPattern()

    def test_zero_at_sub_satellite_point(self) -> None:
        """User directly below satellite → 0° off-axis from center beam."""
        sat = self.proxy.satellite_position(0, 0.0)
        angle = self.pattern.off_axis_angle_deg(sat, 0, sat.lat_deg, sat.lon_deg)
        self.assertAlmostEqual(angle, 0.0, places=2)

    def test_center_beam_off_axis_increases_with_distance(self) -> None:
        """Moving the user away from SSP increases center-beam off-axis angle."""
        sat = self.proxy.satellite_position(0, 0.0)
        a1 = self.pattern.off_axis_angle_deg(sat, 0, 0.0, 0.5)
        a2 = self.pattern.off_axis_angle_deg(sat, 0, 0.0, 2.0)
        self.assertGreater(a2, a1)
        self.assertGreater(a1, 0.0)

    def test_ring_beam_minimum_near_its_ground_center(self) -> None:
        """Off-axis angle for a ring beam should be smallest at that beam's
        ground center (approximately)."""
        sat = self.proxy.satellite_position(0, 0.0)
        centers = self.pattern.beam_centers_ground(sat)

        for k in range(1, 7):
            # Angle at beam k's own ground center
            angle_at_center = self.pattern.off_axis_angle_deg(
                sat, k, centers[k].lat_deg, centers[k].lon_deg
            )
            # Angle at a point far from beam k's center (the sub-satellite point)
            angle_at_ssp = self.pattern.off_axis_angle_deg(
                sat, k, sat.lat_deg, sat.lon_deg
            )
            self.assertLess(angle_at_center, angle_at_ssp)


class TestNearestBeam(unittest.TestCase):
    """nearest_beam selection."""

    def setUp(self) -> None:
        self.proxy = OrbitProxy()
        self.pattern = BeamPattern()

    def test_sub_satellite_returns_center(self) -> None:
        """User at sub-satellite point → nearest beam is center (0)."""
        sat = self.proxy.satellite_position(0, 0.0)
        idx = self.pattern.nearest_beam(sat, sat.lat_deg, sat.lon_deg)
        self.assertEqual(idx, 0)

    def test_ring_beam_center_returns_that_beam(self) -> None:
        """User at ring beam k's ground center → nearest beam is k."""
        sat = self.proxy.satellite_position(0, 0.0)
        centers = self.pattern.beam_centers_ground(sat)
        for k in range(1, 7):
            idx = self.pattern.nearest_beam(sat, centers[k].lat_deg, centers[k].lon_deg)
            self.assertEqual(idx, k, f"Expected beam {k} at its own center")


class TestDeterminism(unittest.TestCase):
    """Same inputs → identical outputs."""

    def test_deterministic_across_instances(self) -> None:
        proxy = OrbitProxy()
        p1 = BeamPattern()
        p2 = BeamPattern()
        sat = proxy.satellite_position(0, 0.0)
        d1 = p1.beam_directions_eci(sat)
        d2 = p2.beam_directions_eci(sat)
        for a, b in zip(d1, d2):
            for i in range(3):
                self.assertEqual(a[i], b[i])


class TestDifferentSatellitePositions(unittest.TestCase):
    """Beam pattern should adapt correctly to different satellite positions."""

    def setUp(self) -> None:
        self.proxy = OrbitProxy()
        self.pattern = BeamPattern()

    def test_center_beam_tracks_ssp(self) -> None:
        """As satellite moves, center beam ground position follows SSP."""
        for t in [0.0, 1.0, 5.0]:
            sat = self.proxy.satellite_position(0, t)
            centers = self.pattern.beam_centers_ground(sat)
            self.assertAlmostEqual(centers[0].lat_deg, sat.lat_deg, places=3)
            self.assertAlmostEqual(centers[0].lon_deg, sat.lon_deg, places=3)

    def test_directions_change_with_satellite_position(self) -> None:
        """Beam directions in ECI should differ for different satellite positions."""
        sat_a = self.proxy.satellite_position(0, 0.0)
        sat_b = self.proxy.satellite_position(1, 0.0)  # 90° apart
        dirs_a = self.pattern.beam_directions_eci(sat_a)
        dirs_b = self.pattern.beam_directions_eci(sat_b)
        # Center beam directions should differ (different nadirs)
        self.assertNotAlmostEqual(dirs_a[0][0], dirs_b[0][0], places=3)


if __name__ == "__main__":
    unittest.main()
