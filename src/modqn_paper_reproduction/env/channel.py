"""Channel model for MODQN paper reproduction.

Implements the signal model from PAP-2024-MORL-MULTIBEAM Eq. (1)-(2):

    G_{i,l,v}(t) = (c / (4 pi d f_c))^2  *  A(d)  *  eta
    gamma        = (p * G) / sigma^2

Components:
  - Free-space path loss:  FSPL = (c / (4 pi d f_c))^2
  - Atmospheric attenuation factor A(d) — see ASSUME-MODQN-REP-009
  - Rician small-scale fading eta — K-factor from Table I
  - Noise power:  sigma^2 = N_0 * B  (ASSUME-MODQN-REP-008)

Provenance:
  - carrier_frequency_hz, bandwidth_hz, tx_power_w, noise_psd_dbm_hz,
    rician_k_db, attenuation_db_per_km: paper-backed (Table I)
  - noise semantics (sigma^2 = N_0 * B): reproduction-assumption
    (ASSUME-MODQN-REP-008)
  - atmospheric sign choice: reproduction-assumption
    (ASSUME-MODQN-REP-009)
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.random import Generator

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

SPEED_OF_LIGHT_M_S: float = 299_792_458.0
"""Speed of light in m/s (exact by definition)."""


# ---------------------------------------------------------------------------
# Atmospheric sign mode  (ASSUME-MODQN-REP-009)
# ---------------------------------------------------------------------------


class AtmosphericSignMode(enum.Enum):
    """Controls which sign is used in the atmospheric attenuation exponent.

    ASSUME-MODQN-REP-009 requires:
      - primary run uses paper-published sign (positive exponent → gain)
      - sensitivity run uses corrected lossy sign (negative exponent → loss)
    """

    PAPER_PUBLISHED = "paper-published"
    """A(d) = 10^(+3 d chi / (10 h))  — yields gain for typical values."""

    CORRECTED_LOSSY = "corrected-lossy"
    """A(d) = 10^(-3 d chi / (10 h))  — yields loss (physically expected)."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelConfig:
    """Channel model parameters.

    Paper-backed values (Table I):
      carrier_frequency_hz, bandwidth_hz, tx_power_w,
      noise_psd_dbm_hz, rician_k_db, attenuation_db_per_km.

    Reproduction-assumption:
      atmospheric_sign_mode (ASSUME-MODQN-REP-009),
      noise semantics (ASSUME-MODQN-REP-008).
    """

    # -- paper-backed (SDD §3.1) --------------------------------------------
    carrier_frequency_hz: float = 20.0e9        # 20 GHz
    bandwidth_hz: float = 500.0e6               # 500 MHz
    tx_power_w: float = 2.0                     # 2 W per link
    noise_psd_dbm_hz: float = -174.0            # N_0 in dBm/Hz
    rician_k_db: float = 20.0                   # Rician K-factor
    attenuation_db_per_km: float = 0.05         # chi in dB/km

    # -- reproduction-assumption (ASSUME-MODQN-REP-009) ---------------------
    atmospheric_sign_mode: AtmosphericSignMode = AtmosphericSignMode.PAPER_PUBLISHED

    def __post_init__(self) -> None:
        if self.carrier_frequency_hz <= 0:
            raise ValueError(
                f"carrier_frequency_hz must be > 0, got {self.carrier_frequency_hz}"
            )
        if self.bandwidth_hz <= 0:
            raise ValueError(f"bandwidth_hz must be > 0, got {self.bandwidth_hz}")
        if self.tx_power_w <= 0:
            raise ValueError(f"tx_power_w must be > 0, got {self.tx_power_w}")

    # -- derived constants (linear units) -----------------------------------

    @property
    def noise_psd_w_hz(self) -> float:
        """N_0 in W/Hz (linear from dBm/Hz)."""
        return dbm_to_w(self.noise_psd_dbm_hz)

    @property
    def noise_power_w(self) -> float:
        """sigma^2 = N_0 * B  (ASSUME-MODQN-REP-008).

        Uses total bandwidth, not per-user share — bandwidth sharing
        enters only in the rate equation (Eq. 3), not in noise power.
        """
        return self.noise_psd_w_hz * self.bandwidth_hz

    @property
    def rician_k_linear(self) -> float:
        """Rician K-factor in linear scale."""
        return 10.0 ** (self.rician_k_db / 10.0)

    @property
    def wavelength_m(self) -> float:
        return SPEED_OF_LIGHT_M_S / self.carrier_frequency_hz


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PathLossResult:
    """Deterministic path-loss components (no fading)."""

    slant_range_km: float
    fspl_linear: float
    """Free-space path loss as a linear power ratio < 1."""
    atmospheric_factor: float
    """Atmospheric factor A(d).  >1 for paper-published sign, <1 for corrected."""
    deterministic_gain: float
    """FSPL * atmospheric_factor (linear).  This is the non-fading part of G."""


@dataclass(frozen=True)
class ChannelResult:
    """Full single-link channel computation result.

    All values are in linear (not dB) units unless the field name
    explicitly says ``_db`` or ``_dbm``.
    """

    slant_range_km: float
    altitude_km: float
    fspl_linear: float
    atmospheric_factor: float
    atmospheric_sign_mode: AtmosphericSignMode
    fading_eta: float
    """Rician fading realisation (linear power scale). 1.0 when fading is off."""
    channel_gain: float
    """G = FSPL * A(d) * eta  (linear)."""
    rx_power_w: float
    """p * G  (linear, watts)."""
    noise_power_w: float
    """sigma^2 = N_0 * B  (linear, watts)."""
    snr_linear: float
    """gamma = rx_power / noise_power."""

    @property
    def snr_db(self) -> float:
        if self.snr_linear <= 0:
            return -math.inf
        return 10.0 * math.log10(self.snr_linear)

    @property
    def channel_gain_db(self) -> float:
        if self.channel_gain <= 0:
            return -math.inf
        return 10.0 * math.log10(self.channel_gain)


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------


def fspl_linear(slant_range_km: float, carrier_frequency_hz: float) -> float:
    """Free-space path loss as a linear power ratio (< 1).

    FSPL = (c / (4 pi d f_c))^2

    where d is in metres and f_c in Hz.
    Paper Eq. (1) first term.
    """
    d_m = slant_range_km * 1_000.0
    numerator = SPEED_OF_LIGHT_M_S
    denominator = 4.0 * math.pi * d_m * carrier_frequency_hz
    return (numerator / denominator) ** 2


def atmospheric_factor(
    slant_range_km: float,
    altitude_km: float,
    chi_db_per_km: float,
    mode: AtmosphericSignMode,
) -> float:
    """Atmospheric attenuation / gain factor A(d).

    Paper-published (ASSUME-MODQN-REP-009 primary):
        A(d) = 10^(+3 d chi / (10 h))

    Corrected-lossy (ASSUME-MODQN-REP-009 sensitivity):
        A(d) = 10^(-3 d chi / (10 h))

    Parameters
    ----------
    slant_range_km : float
        Distance d between satellite and user (km).
    altitude_km : float
        Satellite altitude h (km).
    chi_db_per_km : float
        Atmospheric attenuation coefficient (dB/km).
    mode : AtmosphericSignMode
        Which sign convention to use.
    """
    if altitude_km <= 0:
        raise ValueError(f"altitude_km must be > 0, got {altitude_km}")

    exponent = 3.0 * slant_range_km * chi_db_per_km / (10.0 * altitude_km)

    if mode is AtmosphericSignMode.CORRECTED_LOSSY:
        exponent = -exponent

    return 10.0 ** exponent


def compute_path_loss(
    slant_range_km: float,
    altitude_km: float,
    config: ChannelConfig,
) -> PathLossResult:
    """Deterministic path-loss computation (FSPL + atmospheric, no fading)."""
    fspl = fspl_linear(slant_range_km, config.carrier_frequency_hz)
    atm = atmospheric_factor(
        slant_range_km,
        altitude_km,
        config.attenuation_db_per_km,
        config.atmospheric_sign_mode,
    )
    return PathLossResult(
        slant_range_km=slant_range_km,
        fspl_linear=fspl,
        atmospheric_factor=atm,
        deterministic_gain=fspl * atm,
    )


# ---------------------------------------------------------------------------
# Rician fading
# ---------------------------------------------------------------------------


def rician_fading_sample(
    k_linear: float,
    rng: Generator,
    size: int = 1,
) -> np.ndarray:
    """Sample Rician fading power realisations.

    The Rician envelope is generated from two Gaussian components:
        h = sqrt( (s + X)^2 + Y^2 )
    where X, Y ~ N(0, sigma_g^2) and s is the LOS amplitude.

    Normalisation: E[|h|^2] = 1  (mean channel power gain = 1).

    With K = s^2 / (2 sigma_g^2):
        sigma_g^2 = 1 / (2 (K + 1))
        s = sqrt(2 K / (2 (K + 1))) = sqrt(K / (K + 1))

    Returns power |h|^2 for each sample (linear scale).
    """
    sigma_g_sq = 1.0 / (2.0 * (k_linear + 1.0))
    sigma_g = math.sqrt(sigma_g_sq)
    s = math.sqrt(k_linear / (k_linear + 1.0))

    x = rng.normal(0.0, sigma_g, size=size)
    y = rng.normal(0.0, sigma_g, size=size)

    power: np.ndarray = (s + x) ** 2 + y ** 2
    return power


def rician_fading_mean_power(k_linear: float) -> float:
    """Theoretical mean power E[|h|^2] for the normalised Rician model.

    Should be exactly 1.0 for our normalisation.
    """
    # E[|h|^2] = s^2 + 2*sigma_g^2  = K/(K+1) + 1/(K+1) = 1
    _ = k_linear  # unused but kept for interface symmetry
    return 1.0


# ---------------------------------------------------------------------------
# Full channel computation
# ---------------------------------------------------------------------------


def compute_channel(
    slant_range_km: float,
    altitude_km: float,
    config: ChannelConfig,
    rng: Optional[Generator] = None,
    fading: bool = True,
    tx_power_w: float | None = None,
) -> ChannelResult:
    """Compute the full single-link channel from paper Eq. (1)-(2).

    G = FSPL * A(d) * eta
    gamma = (p * G) / sigma^2

    Parameters
    ----------
    slant_range_km : float
        Slant range d from user to satellite (km).
    altitude_km : float
        Satellite orbital altitude h (km).
    config : ChannelConfig
        Channel parameters.
    rng : numpy Generator, optional
        Random source for Rician fading.  Required when ``fading=True``.
    fading : bool
        If False, eta = 1.0 (deterministic mode for testing).
    tx_power_w : float, optional
        Linear-W transmit power override for opt-in per-beam power surfaces.
        When omitted, the paper-baseline ``config.tx_power_w`` path is used.

    Returns
    -------
    ChannelResult
        All intermediate and final values in linear units.
    """
    # Deterministic path
    pl = compute_path_loss(slant_range_km, altitude_km, config)

    # Stochastic fading
    if fading:
        if rng is None:
            raise ValueError("rng must be provided when fading=True")
        eta = float(rician_fading_sample(config.rician_k_linear, rng, size=1)[0])
    else:
        eta = 1.0

    # Channel gain  G = FSPL * A(d) * eta
    channel_gain = pl.fspl_linear * pl.atmospheric_factor * eta

    # Noise power  sigma^2 = N_0 * B  (ASSUME-MODQN-REP-008)
    noise_w = config.noise_power_w

    # Received power and SNR
    effective_tx_power_w = (
        config.tx_power_w if tx_power_w is None else float(tx_power_w)
    )
    if effective_tx_power_w < 0:
        raise ValueError(f"tx_power_w override must be >= 0, got {effective_tx_power_w}")
    rx_power = effective_tx_power_w * channel_gain
    snr = rx_power / noise_w if noise_w > 0 else math.inf

    return ChannelResult(
        slant_range_km=slant_range_km,
        altitude_km=altitude_km,
        fspl_linear=pl.fspl_linear,
        atmospheric_factor=pl.atmospheric_factor,
        atmospheric_sign_mode=config.atmospheric_sign_mode,
        fading_eta=eta,
        channel_gain=channel_gain,
        rx_power_w=rx_power,
        noise_power_w=noise_w,
        snr_linear=snr,
    )


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------


def dbm_to_w(dbm: float) -> float:
    """Convert dBm to watts."""
    return 10.0 ** ((dbm - 30.0) / 10.0)


def w_to_dbm(w: float) -> float:
    """Convert watts to dBm."""
    if w <= 0:
        return -math.inf
    return 10.0 * math.log10(w) + 30.0


def db_to_linear(db: float) -> float:
    """Convert dB (power ratio) to linear."""
    return 10.0 ** (db / 10.0)


def linear_to_db(x: float) -> float:
    """Convert linear power ratio to dB."""
    if x <= 0:
        return -math.inf
    return 10.0 * math.log10(x)
