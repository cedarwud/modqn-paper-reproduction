"""Unit tests for env.channel — paper signal model.

Validates FSPL, atmospheric attenuation, Rician fading, noise power,
and SNR assembly per PAP-2024-MORL-MULTIBEAM Eq. (1)-(2) and the
assumption-backed choices in ASSUME-MODQN-REP-008 / 009.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
from numpy.random import default_rng

from modqn_paper_reproduction.env.channel import (
    SPEED_OF_LIGHT_M_S,
    AtmosphericSignMode,
    ChannelConfig,
    ChannelResult,
    atmospheric_factor,
    compute_channel,
    compute_path_loss,
    db_to_linear,
    dbm_to_w,
    fspl_linear,
    linear_to_db,
    rician_fading_mean_power,
    rician_fading_sample,
    w_to_dbm,
)


# ---------------------------------------------------------------------------
# 1. FSPL monotonicity and correctness
# ---------------------------------------------------------------------------


class TestFSPL(unittest.TestCase):
    """Free-space path loss: (c / (4 pi d f_c))^2."""

    def test_fspl_decreases_with_distance(self) -> None:
        """FSPL (linear < 1) should decrease as slant range increases."""
        f = 20.0e9
        near = fspl_linear(780.0, f)
        far = fspl_linear(1500.0, f)
        self.assertGreater(near, far)
        self.assertGreater(near, 0)
        self.assertGreater(far, 0)

    def test_fspl_decreases_with_frequency(self) -> None:
        """Higher carrier frequency → more path loss (smaller linear value)."""
        d = 780.0
        low_f = fspl_linear(d, 10.0e9)
        high_f = fspl_linear(d, 20.0e9)
        self.assertGreater(low_f, high_f)

    def test_fspl_known_value(self) -> None:
        """Cross-check against manually computed FSPL at 780 km, 20 GHz.

        FSPL_dB = 20 log10(4 pi d f / c)
                = 20 log10(4 * pi * 780e3 * 20e9 / 299792458)
        """
        d_km = 780.0
        f_hz = 20.0e9
        d_m = d_km * 1000.0
        expected_db = 20.0 * math.log10(4.0 * math.pi * d_m * f_hz / SPEED_OF_LIGHT_M_S)
        # FSPL linear = 10^(-FSPL_dB/10)
        expected_linear = 10.0 ** (-expected_db / 10.0)

        computed = fspl_linear(d_km, f_hz)
        self.assertAlmostEqual(computed, expected_linear, places=20)

    def test_fspl_inverse_square_law(self) -> None:
        """Doubling the distance should reduce FSPL by 6 dB (factor of 4)."""
        f = 20.0e9
        d1 = 800.0
        d2 = 1600.0
        ratio = fspl_linear(d1, f) / fspl_linear(d2, f)
        self.assertAlmostEqual(ratio, 4.0, places=6)

    def test_fspl_is_less_than_one(self) -> None:
        """Path loss should always be < 1 for any real LEO distance."""
        for d in [500.0, 780.0, 1200.0, 3000.0]:
            self.assertLess(fspl_linear(d, 20.0e9), 1.0)


# ---------------------------------------------------------------------------
# 2. Noise semantics  (ASSUME-MODQN-REP-008)
# ---------------------------------------------------------------------------


class TestNoisePower(unittest.TestCase):
    """sigma^2 = N_0 * B  using linear units."""

    def test_noise_power_formula(self) -> None:
        """Verify sigma^2 = N_0(W/Hz) * B(Hz) for default paper values."""
        cfg = ChannelConfig()
        n0_w = dbm_to_w(-174.0)
        expected = n0_w * 500.0e6
        self.assertAlmostEqual(cfg.noise_power_w, expected, places=30)

    def test_noise_power_scales_with_bandwidth(self) -> None:
        """Doubling bandwidth doubles noise power."""
        cfg_narrow = ChannelConfig(bandwidth_hz=250.0e6)
        cfg_wide = ChannelConfig(bandwidth_hz=500.0e6)
        ratio = cfg_wide.noise_power_w / cfg_narrow.noise_power_w
        self.assertAlmostEqual(ratio, 2.0, places=10)

    def test_noise_power_uses_total_bandwidth(self) -> None:
        """Noise power should use total B, not B/N_users.

        Per-user bandwidth sharing stays in Eq. (3) rate only,
        not in the noise (ASSUME-MODQN-REP-008).
        """
        cfg = ChannelConfig()
        # Just verify it's N_0 * B, no user-count dependency
        self.assertEqual(cfg.noise_power_w, cfg.noise_psd_w_hz * cfg.bandwidth_hz)

    def test_noise_psd_conversion(self) -> None:
        """-174 dBm/Hz should be approximately 3.981e-21 W/Hz."""
        cfg = ChannelConfig()
        # 10^((-174-30)/10) = 10^(-20.4) ≈ 3.981e-21
        expected = 10.0 ** ((-174.0 - 30.0) / 10.0)
        self.assertAlmostEqual(cfg.noise_psd_w_hz, expected, places=30)


# ---------------------------------------------------------------------------
# 3. Atmospheric attenuation sign  (ASSUME-MODQN-REP-009)
# ---------------------------------------------------------------------------


class TestAtmosphericAttenuation(unittest.TestCase):
    """Published vs corrected atmospheric attenuation sign."""

    def test_published_sign_yields_gain(self) -> None:
        """Paper-published formula A(d) = 10^(+3 d chi / (10 h)) > 1."""
        a = atmospheric_factor(
            slant_range_km=780.0,
            altitude_km=780.0,
            chi_db_per_km=0.05,
            mode=AtmosphericSignMode.PAPER_PUBLISHED,
        )
        self.assertGreater(a, 1.0, "Paper-published sign should yield gain (A > 1)")

    def test_corrected_sign_yields_loss(self) -> None:
        """Corrected formula A(d) = 10^(-3 d chi / (10 h)) < 1."""
        a = atmospheric_factor(
            slant_range_km=780.0,
            altitude_km=780.0,
            chi_db_per_km=0.05,
            mode=AtmosphericSignMode.CORRECTED_LOSSY,
        )
        self.assertLess(a, 1.0, "Corrected sign should yield loss (A < 1)")

    def test_published_and_corrected_are_reciprocals(self) -> None:
        """A_published * A_corrected = 1 (they use opposite exponents)."""
        kwargs = dict(
            slant_range_km=1000.0,
            altitude_km=780.0,
            chi_db_per_km=0.05,
        )
        a_pub = atmospheric_factor(**kwargs, mode=AtmosphericSignMode.PAPER_PUBLISHED)
        a_cor = atmospheric_factor(**kwargs, mode=AtmosphericSignMode.CORRECTED_LOSSY)
        self.assertAlmostEqual(a_pub * a_cor, 1.0, places=12)

    def test_atmospheric_increases_with_distance(self) -> None:
        """For published sign, atmospheric factor grows with distance.
        For corrected sign, it shrinks with distance."""
        h = 780.0
        chi = 0.05
        a_near_pub = atmospheric_factor(500.0, h, chi, AtmosphericSignMode.PAPER_PUBLISHED)
        a_far_pub = atmospheric_factor(1500.0, h, chi, AtmosphericSignMode.PAPER_PUBLISHED)
        self.assertGreater(a_far_pub, a_near_pub)

        a_near_cor = atmospheric_factor(500.0, h, chi, AtmosphericSignMode.CORRECTED_LOSSY)
        a_far_cor = atmospheric_factor(1500.0, h, chi, AtmosphericSignMode.CORRECTED_LOSSY)
        self.assertLess(a_far_cor, a_near_cor)

    def test_zero_attenuation_coefficient(self) -> None:
        """With chi=0, atmospheric factor should be exactly 1.0 regardless of sign."""
        for mode in AtmosphericSignMode:
            a = atmospheric_factor(1000.0, 780.0, 0.0, mode)
            self.assertAlmostEqual(a, 1.0, places=15)

    def test_known_value(self) -> None:
        """Manual computation for d=780, h=780, chi=0.05.

        exponent = 3 * 780 * 0.05 / (10 * 780) = 117 / 7800 = 0.015
        A_pub = 10^0.015 ≈ 1.03514
        """
        a = atmospheric_factor(780.0, 780.0, 0.05, AtmosphericSignMode.PAPER_PUBLISHED)
        expected = 10.0 ** 0.015
        self.assertAlmostEqual(a, expected, places=10)

    def test_rejects_zero_altitude(self) -> None:
        with self.assertRaises(ValueError):
            atmospheric_factor(780.0, 0.0, 0.05, AtmosphericSignMode.PAPER_PUBLISHED)


# ---------------------------------------------------------------------------
# 4. Rician fading reproducibility
# ---------------------------------------------------------------------------


class TestRicianFading(unittest.TestCase):
    """Seeded fading produces reproducible and statistically correct results."""

    def test_seeded_reproducibility(self) -> None:
        """Same seed → identical fading samples."""
        k_lin = db_to_linear(20.0)
        s1 = rician_fading_sample(k_lin, default_rng(42), size=100)
        s2 = rician_fading_sample(k_lin, default_rng(42), size=100)
        np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_differ(self) -> None:
        """Different seeds → different realisations."""
        k_lin = db_to_linear(20.0)
        s1 = rician_fading_sample(k_lin, default_rng(42), size=100)
        s2 = rician_fading_sample(k_lin, default_rng(99), size=100)
        self.assertFalse(np.array_equal(s1, s2))

    def test_mean_power_near_unity(self) -> None:
        """E[|h|^2] ≈ 1 for the normalised Rician model (large sample)."""
        k_lin = db_to_linear(20.0)
        samples = rician_fading_sample(k_lin, default_rng(123), size=100_000)
        mean_pwr = float(np.mean(samples))
        self.assertAlmostEqual(mean_pwr, 1.0, delta=0.02)

    def test_theoretical_mean_is_one(self) -> None:
        """rician_fading_mean_power should return 1.0 for any K."""
        for k_db in [0.0, 10.0, 20.0, 30.0]:
            self.assertAlmostEqual(
                rician_fading_mean_power(db_to_linear(k_db)), 1.0, places=15
            )

    def test_high_k_concentrates_near_unity(self) -> None:
        """K=20 dB → strong LOS → variance is small, samples ≈ 1.0."""
        k_lin = db_to_linear(20.0)  # 100 in linear
        samples = rician_fading_sample(k_lin, default_rng(77), size=10_000)
        std = float(np.std(samples))
        self.assertLess(std, 0.15, "K=20 dB should have very tight distribution")

    def test_all_samples_positive(self) -> None:
        """Power samples must be >= 0."""
        k_lin = db_to_linear(20.0)
        samples = rician_fading_sample(k_lin, default_rng(55), size=10_000)
        self.assertTrue(np.all(samples >= 0))

    def test_low_k_higher_variance(self) -> None:
        """K=0 dB (Rayleigh limit) should have much higher variance than K=20 dB."""
        n = 50_000
        rng = default_rng(42)
        low_k = rician_fading_sample(db_to_linear(0.0), rng, size=n)
        high_k = rician_fading_sample(db_to_linear(20.0), rng, size=n)
        self.assertGreater(float(np.var(low_k)), float(np.var(high_k)))


# ---------------------------------------------------------------------------
# 5. Deterministic mode (fading off)
# ---------------------------------------------------------------------------


class TestDeterministicMode(unittest.TestCase):
    """With fading=False, output is fully deterministic and stable."""

    def test_no_fading_eta_is_one(self) -> None:
        """fading=False → eta = 1.0."""
        cfg = ChannelConfig()
        result = compute_channel(780.0, 780.0, cfg, fading=False)
        self.assertEqual(result.fading_eta, 1.0)

    def test_deterministic_reproducible(self) -> None:
        """Same inputs → identical outputs without RNG."""
        cfg = ChannelConfig()
        r1 = compute_channel(780.0, 780.0, cfg, fading=False)
        r2 = compute_channel(780.0, 780.0, cfg, fading=False)
        self.assertEqual(r1.channel_gain, r2.channel_gain)
        self.assertEqual(r1.snr_linear, r2.snr_linear)

    def test_fading_true_without_rng_raises(self) -> None:
        """fading=True with no rng → ValueError."""
        cfg = ChannelConfig()
        with self.assertRaises(ValueError):
            compute_channel(780.0, 780.0, cfg, rng=None, fading=True)

    def test_seeded_fading_reproducible(self) -> None:
        """Same seed → same faded channel result."""
        cfg = ChannelConfig()
        r1 = compute_channel(780.0, 780.0, cfg, rng=default_rng(42), fading=True)
        r2 = compute_channel(780.0, 780.0, cfg, rng=default_rng(42), fading=True)
        self.assertEqual(r1.channel_gain, r2.channel_gain)
        self.assertEqual(r1.snr_linear, r2.snr_linear)


# ---------------------------------------------------------------------------
# 6. Full channel assembly
# ---------------------------------------------------------------------------


class TestComputeChannel(unittest.TestCase):
    """End-to-end channel computation sanity."""

    def test_snr_formula(self) -> None:
        """gamma = (p * G) / sigma^2."""
        cfg = ChannelConfig()
        result = compute_channel(780.0, 780.0, cfg, fading=False)
        expected_snr = result.rx_power_w / result.noise_power_w
        self.assertAlmostEqual(result.snr_linear, expected_snr, places=15)

    def test_rx_power_formula(self) -> None:
        """rx_power = p * G."""
        cfg = ChannelConfig()
        result = compute_channel(780.0, 780.0, cfg, fading=False)
        expected = cfg.tx_power_w * result.channel_gain
        self.assertAlmostEqual(result.rx_power_w, expected, places=20)

    def test_channel_gain_composition(self) -> None:
        """G = FSPL * A(d) * eta."""
        cfg = ChannelConfig()
        result = compute_channel(780.0, 780.0, cfg, fading=False)
        expected = result.fspl_linear * result.atmospheric_factor * result.fading_eta
        self.assertAlmostEqual(result.channel_gain, expected, places=20)

    def test_snr_decreases_with_distance(self) -> None:
        """Farther satellite → lower SNR (for corrected-lossy sign)."""
        cfg = ChannelConfig(atmospheric_sign_mode=AtmosphericSignMode.CORRECTED_LOSSY)
        near = compute_channel(780.0, 780.0, cfg, fading=False)
        far = compute_channel(1500.0, 780.0, cfg, fading=False)
        self.assertGreater(near.snr_linear, far.snr_linear)

    def test_snr_positive_for_paper_baseline(self) -> None:
        """Paper baseline at 780 km should yield a positive finite SNR."""
        cfg = ChannelConfig()
        result = compute_channel(780.0, 780.0, cfg, fading=False)
        self.assertGreater(result.snr_linear, 0)
        self.assertTrue(math.isfinite(result.snr_db))

    def test_atmospheric_sign_affects_snr(self) -> None:
        """Published sign → higher SNR than corrected sign (same distance)."""
        cfg_pub = ChannelConfig(atmospheric_sign_mode=AtmosphericSignMode.PAPER_PUBLISHED)
        cfg_cor = ChannelConfig(atmospheric_sign_mode=AtmosphericSignMode.CORRECTED_LOSSY)
        r_pub = compute_channel(780.0, 780.0, cfg_pub, fading=False)
        r_cor = compute_channel(780.0, 780.0, cfg_cor, fading=False)
        self.assertGreater(r_pub.snr_linear, r_cor.snr_linear)

    def test_snr_db_property(self) -> None:
        """snr_db should be 10*log10(snr_linear)."""
        cfg = ChannelConfig()
        result = compute_channel(780.0, 780.0, cfg, fading=False)
        expected = 10.0 * math.log10(result.snr_linear)
        self.assertAlmostEqual(result.snr_db, expected, places=10)

    def test_noise_power_in_result(self) -> None:
        """Result noise_power_w should match config noise_power_w."""
        cfg = ChannelConfig()
        result = compute_channel(780.0, 780.0, cfg, fading=False)
        self.assertEqual(result.noise_power_w, cfg.noise_power_w)

    def test_sign_mode_recorded_in_result(self) -> None:
        """Result records which atmospheric sign mode was used."""
        for mode in AtmosphericSignMode:
            cfg = ChannelConfig(atmospheric_sign_mode=mode)
            result = compute_channel(780.0, 780.0, cfg, fading=False)
            self.assertEqual(result.atmospheric_sign_mode, mode)


# ---------------------------------------------------------------------------
# 7. Config validation
# ---------------------------------------------------------------------------


class TestChannelConfig(unittest.TestCase):
    """ChannelConfig defaults and validation."""

    def test_defaults_match_paper(self) -> None:
        cfg = ChannelConfig()
        self.assertEqual(cfg.carrier_frequency_hz, 20.0e9)
        self.assertEqual(cfg.bandwidth_hz, 500.0e6)
        self.assertEqual(cfg.tx_power_w, 2.0)
        self.assertEqual(cfg.noise_psd_dbm_hz, -174.0)
        self.assertEqual(cfg.rician_k_db, 20.0)
        self.assertEqual(cfg.attenuation_db_per_km, 0.05)

    def test_default_sign_mode_is_paper_published(self) -> None:
        cfg = ChannelConfig()
        self.assertEqual(cfg.atmospheric_sign_mode, AtmosphericSignMode.PAPER_PUBLISHED)

    def test_rician_k_linear(self) -> None:
        cfg = ChannelConfig()
        self.assertAlmostEqual(cfg.rician_k_linear, 100.0, places=10)

    def test_wavelength(self) -> None:
        cfg = ChannelConfig()
        expected = SPEED_OF_LIGHT_M_S / 20.0e9
        self.assertAlmostEqual(cfg.wavelength_m, expected, places=10)

    def test_rejects_zero_frequency(self) -> None:
        with self.assertRaises(ValueError):
            ChannelConfig(carrier_frequency_hz=0.0)

    def test_rejects_negative_bandwidth(self) -> None:
        with self.assertRaises(ValueError):
            ChannelConfig(bandwidth_hz=-1.0)

    def test_rejects_zero_power(self) -> None:
        with self.assertRaises(ValueError):
            ChannelConfig(tx_power_w=0.0)


# ---------------------------------------------------------------------------
# 8. Unit conversion helpers
# ---------------------------------------------------------------------------


class TestUnitConversions(unittest.TestCase):

    def test_dbm_to_w_roundtrip(self) -> None:
        for dbm in [-174.0, -100.0, 0.0, 30.0, 33.0]:
            w = dbm_to_w(dbm)
            self.assertAlmostEqual(w_to_dbm(w), dbm, places=10)

    def test_db_linear_roundtrip(self) -> None:
        for db in [-30.0, -3.0, 0.0, 10.0, 20.0]:
            lin = db_to_linear(db)
            self.assertAlmostEqual(linear_to_db(lin), db, places=10)

    def test_zero_dbm_is_one_mw(self) -> None:
        self.assertAlmostEqual(dbm_to_w(0.0), 1.0e-3, places=15)

    def test_thirty_dbm_is_one_w(self) -> None:
        self.assertAlmostEqual(dbm_to_w(30.0), 1.0, places=15)


# ---------------------------------------------------------------------------
# 9. Path loss result composition
# ---------------------------------------------------------------------------


class TestPathLossResult(unittest.TestCase):
    """compute_path_loss returns correct intermediate values."""

    def test_deterministic_gain_composition(self) -> None:
        cfg = ChannelConfig()
        pl = compute_path_loss(780.0, 780.0, cfg)
        self.assertAlmostEqual(
            pl.deterministic_gain, pl.fspl_linear * pl.atmospheric_factor, places=20
        )

    def test_published_sign_boosts_deterministic_gain(self) -> None:
        """With paper-published sign, deterministic_gain > FSPL alone."""
        cfg = ChannelConfig(atmospheric_sign_mode=AtmosphericSignMode.PAPER_PUBLISHED)
        pl = compute_path_loss(780.0, 780.0, cfg)
        self.assertGreater(pl.deterministic_gain, pl.fspl_linear)

    def test_corrected_sign_reduces_deterministic_gain(self) -> None:
        """With corrected-lossy sign, deterministic_gain < FSPL alone."""
        cfg = ChannelConfig(atmospheric_sign_mode=AtmosphericSignMode.CORRECTED_LOSSY)
        pl = compute_path_loss(780.0, 780.0, cfg)
        self.assertLess(pl.deterministic_gain, pl.fspl_linear)


if __name__ == "__main__":
    unittest.main()
