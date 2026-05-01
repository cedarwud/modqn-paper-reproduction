"""Route A2: HOBS-style SINR channel-regime audit.

Namespace: hobs-sinr-channel-regime-audit
Date: 2026-05-01

This module audits whether any paper-defensible or sensitivity-defensible
channel parameter regime can make intra-satellite SINR interference
numerically observable, enabling Route D (tiny learned pilot) to be
scientifically meaningful.

Context:
  Route A showed intra-satellite SINR interference is structurally correct
  but numerically negligible (I_intra/N0 ≈ 1.1e-5) at the MODQN paper's
  operating point (~-56 dB SNR). This audit determines whether Path 1
  (adjust channel regime) or Path 2 (HOBS-style DPC sidecar) is the
  correct next gate before Route D.

Observability thresholds:
  I_intra/N0 >= 0.01   => weak but detectable
  I_intra/N0 >= 0.1    => observable
  I_intra/N0 >= 1.0    => strong
  SINR_drop_ratio <= 0.9 => >= 10% SINR degradation

Parameter source categories:
  "paper-backed"  : directly from MODQN Table I or cross-reference
  "s10-backed"    : from S10/MAAC-BHPOWER power caps (paper-backed for power only)
  "sensitivity"   : plausible variation, not from MODQN paper
  "extension"     : requires adding a model feature not in MODQN paper (antenna gain)
  "different-paper": from HOBS or other paper, different scenario
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..env.channel import dbm_to_w

# ---------------------------------------------------------------------------
# Observability thresholds
# ---------------------------------------------------------------------------

I_N0_DETECTABLE    = 0.01   # I_intra/N0 — weak but detectable
I_N0_OBSERVABLE    = 0.10   # I_intra/N0 — clearly observable
I_N0_STRONG        = 1.00   # I_intra/N0 — strong interference
SINR_DROP_THRESHOLD = 0.90  # SINR/SNR ratio — >=10% SINR degradation


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChannelRegimeScenario:
    """One row in the channel-regime sweep."""
    label: str
    source: str          # parameter-source category
    p_tx_w: float        # signal beam transmit power (W)
    bw_mhz: float        # bandwidth (MHz)
    n0_psd_dbm_hz: float # noise PSD (dBm/Hz)
    f_c_ghz: float       # carrier frequency (GHz)
    alt_km: float        # satellite altitude (km)
    n_neighbors: int     # number of co-channel active neighbor beams
    p_neighbor_w: float  # each neighbor beam's transmit power (W)
    antenna_gain_db: float = 0.0    # total antenna gain (transmit, dBi)
    notes: str = ""


def _compute_metrics(s: ChannelRegimeScenario) -> dict[str, Any]:
    """Compute SNR, I_intra/N0, and SINR_drop for one scenario."""
    n0_psd_w_hz = dbm_to_w(s.n0_psd_dbm_hz)
    n0_w = n0_psd_w_hz * s.bw_mhz * 1e6

    # FSPL at zenith (minimum slant range = altitude)
    d_m = s.alt_km * 1e3
    fspl = (3e8 / (4.0 * math.pi * d_m * s.f_c_ghz * 1e9)) ** 2
    ant_linear = 10.0 ** (s.antenna_gain_db / 10.0)
    channel_gain = fspl * ant_linear

    snr_linear = s.p_tx_w * channel_gain / n0_w
    snr_db = 10.0 * math.log10(snr_linear) if snr_linear > 0 else -math.inf

    # Intra-satellite interference from n_neighbors beams at p_neighbor_w each
    i_intra_w = s.n_neighbors * s.p_neighbor_w * channel_gain
    i_n0_ratio = i_intra_w / n0_w if n0_w > 0 else math.inf

    # SINR: signal / (I_intra + N0)
    sinr_linear = s.p_tx_w * channel_gain / (i_intra_w + n0_w)
    sinr_drop_ratio = sinr_linear / snr_linear if snr_linear > 0 else 0.0

    # Classify observability
    if i_n0_ratio >= I_N0_STRONG:
        status = "strong"
    elif i_n0_ratio >= I_N0_OBSERVABLE:
        status = "observable"
    elif i_n0_ratio >= I_N0_DETECTABLE:
        status = "detectable"
    else:
        status = "negligible"

    return {
        "label": s.label,
        "source": s.source,
        "p_tx_w": s.p_tx_w,
        "bw_mhz": s.bw_mhz,
        "f_c_ghz": s.f_c_ghz,
        "alt_km": s.alt_km,
        "n_neighbors": s.n_neighbors,
        "p_neighbor_w": s.p_neighbor_w,
        "antenna_gain_db": s.antenna_gain_db,
        "n0_w": n0_w,
        "channel_gain": channel_gain,
        "snr_db": round(snr_db, 2),
        "i_intra_n0_ratio": i_n0_ratio,
        "sinr_drop_ratio": round(sinr_drop_ratio, 6),
        "interference_status": status,
        "exceeds_observable_threshold": i_n0_ratio >= I_N0_OBSERVABLE,
        "exceeds_sinr_drop_threshold": sinr_drop_ratio <= SINR_DROP_THRESHOLD,
        "notes": s.notes,
    }


# ---------------------------------------------------------------------------
# Defined sweep scenarios
# ---------------------------------------------------------------------------

# Paper-backed constants
_MODQN_P_TX_W         = 2.0      # Table I: transmit power
_MODQN_BW_MHZ         = 500.0    # Table I: bandwidth
_MODQN_N0_PSD         = -174.0   # Table I: noise PSD
_MODQN_FC_GHZ         = 20.0     # Table I: carrier frequency
_MODQN_ALT_KM         = 780.0    # Table I: altitude
_S10_PER_BEAM_W       = 10.0     # S10/MAAC-BHPOWER: 10 dBW = 10W per beam
_S10_SAT_MAX_W        = 19.95    # S10/MAAC-BHPOWER: 13 dBW ≈ 20W sat total
_HOBS_FC_GHZ          = 28.0     # HOBS: Ka-band
_HOBS_ALT_KM          = 550.0    # HOBS: altitude
_HOBS_P_MAX_W         = 100.0    # HOBS: 50 dBm total TX
_HOBS_G_T_DB          = 40.0     # HOBS: transmit antenna gain
_HOBS_BW_MHZ          = 100.0    # HOBS: bandwidth
_MAX_NEIGHBORS        = 6        # MODQN: L*K-1 worst case on same satellite, K=7 beams


SWEEP_SCENARIOS: list[ChannelRegimeScenario] = [
    # ---------- MODQN paper-backed scenarios ----------
    ChannelRegimeScenario(
        label="MODQN paper exact (Table I)",
        source="paper-backed",
        p_tx_w=_MODQN_P_TX_W, bw_mhz=_MODQN_BW_MHZ,
        n0_psd_dbm_hz=_MODQN_N0_PSD, f_c_ghz=_MODQN_FC_GHZ,
        alt_km=_MODQN_ALT_KM, n_neighbors=_MAX_NEIGHBORS,
        p_neighbor_w=_MODQN_P_TX_W, antenna_gain_db=0.0,
        notes="Baseline: 2W, 500MHz, 20GHz, 780km — no antenna gain in MODQN",
    ),
    # ---------- S10 power caps (paper-backed power, same channel) ----------
    ChannelRegimeScenario(
        label="S10 per-beam max 10W (MAAC-BHPOWER)",
        source="s10-backed",
        p_tx_w=_S10_PER_BEAM_W, bw_mhz=_MODQN_BW_MHZ,
        n0_psd_dbm_hz=_MODQN_N0_PSD, f_c_ghz=_MODQN_FC_GHZ,
        alt_km=_MODQN_ALT_KM, n_neighbors=_MAX_NEIGHBORS,
        p_neighbor_w=_S10_PER_BEAM_W, antenna_gain_db=0.0,
        notes="S10/MAAC-BHPOWER per-beam cap 10 dBW=10W; same MODQN channel otherwise",
    ),
    ChannelRegimeScenario(
        label="S10 sat-max 20W / 7 beams",
        source="s10-backed",
        p_tx_w=_S10_SAT_MAX_W / 7, bw_mhz=_MODQN_BW_MHZ,
        n0_psd_dbm_hz=_MODQN_N0_PSD, f_c_ghz=_MODQN_FC_GHZ,
        alt_km=_MODQN_ALT_KM, n_neighbors=_MAX_NEIGHBORS,
        p_neighbor_w=_S10_SAT_MAX_W / 7, antenna_gain_db=0.0,
        notes="S10 total sat power 13 dBW≈20W split across 7 beams; same MODQN channel",
    ),
    # ---------- Sensitivity: different altitudes ----------
    ChannelRegimeScenario(
        label="LEO 550km altitude (HOBS altitude)",
        source="sensitivity",
        p_tx_w=_MODQN_P_TX_W, bw_mhz=_MODQN_BW_MHZ,
        n0_psd_dbm_hz=_MODQN_N0_PSD, f_c_ghz=_MODQN_FC_GHZ,
        alt_km=_HOBS_ALT_KM, n_neighbors=_MAX_NEIGHBORS,
        p_neighbor_w=_MODQN_P_TX_W, antenna_gain_db=0.0,
        notes="Different altitude (HOBS=550km vs MODQN=780km); 20GHz MODQN channel",
    ),
    ChannelRegimeScenario(
        label="LEO 300km (Starlink-class)",
        source="sensitivity",
        p_tx_w=_MODQN_P_TX_W, bw_mhz=_MODQN_BW_MHZ,
        n0_psd_dbm_hz=_MODQN_N0_PSD, f_c_ghz=_MODQN_FC_GHZ,
        alt_km=300.0, n_neighbors=_MAX_NEIGHBORS,
        p_neighbor_w=_MODQN_P_TX_W, antenna_gain_db=0.0,
        notes="Sensitivity only: not MODQN scenario; shorter range increases SNR",
    ),
    # ---------- Sensitivity: bandwidth variation ----------
    ChannelRegimeScenario(
        label="100MHz BW (HOBS-backed bandwidth)",
        source="sensitivity",
        p_tx_w=_MODQN_P_TX_W, bw_mhz=_HOBS_BW_MHZ,
        n0_psd_dbm_hz=_MODQN_N0_PSD, f_c_ghz=_MODQN_FC_GHZ,
        alt_km=_MODQN_ALT_KM, n_neighbors=_MAX_NEIGHBORS,
        p_neighbor_w=_MODQN_P_TX_W, antenna_gain_db=0.0,
        notes="HOBS bandwidth 100MHz; reduces noise but not MODQN paper parameter",
    ),
    ChannelRegimeScenario(
        label="10MHz BW (narrow, sensitivity)",
        source="sensitivity",
        p_tx_w=_MODQN_P_TX_W, bw_mhz=10.0,
        n0_psd_dbm_hz=_MODQN_N0_PSD, f_c_ghz=_MODQN_FC_GHZ,
        alt_km=_MODQN_ALT_KM, n_neighbors=_MAX_NEIGHBORS,
        p_neighbor_w=_MODQN_P_TX_W, antenna_gain_db=0.0,
        notes="Sensitivity only: narrow BW; not in MODQN paper",
    ),
    # ---------- Extension: adding antenna gain (not in MODQN paper) ----------
    ChannelRegimeScenario(
        label="MODQN + 30dBi antenna gain (extension)",
        source="extension",
        p_tx_w=_MODQN_P_TX_W, bw_mhz=_MODQN_BW_MHZ,
        n0_psd_dbm_hz=_MODQN_N0_PSD, f_c_ghz=_MODQN_FC_GHZ,
        alt_km=_MODQN_ALT_KM, n_neighbors=_MAX_NEIGHBORS,
        p_neighbor_w=_MODQN_P_TX_W, antenna_gain_db=30.0,
        notes="Model extension: MODQN has no antenna gain term; adding 30dBi",
    ),
    ChannelRegimeScenario(
        label="MODQN + 40dBi antenna gain (HOBS G_T, extension)",
        source="extension",
        p_tx_w=_MODQN_P_TX_W, bw_mhz=_MODQN_BW_MHZ,
        n0_psd_dbm_hz=_MODQN_N0_PSD, f_c_ghz=_MODQN_FC_GHZ,
        alt_km=_MODQN_ALT_KM, n_neighbors=_MAX_NEIGHBORS,
        p_neighbor_w=_MODQN_P_TX_W, antenna_gain_db=_HOBS_G_T_DB,
        notes="Extension: adds HOBS transmit antenna gain (40dBi) to MODQN; just crosses observable",
    ),
    # ---------- Different-paper: HOBS full parameters ----------
    ChannelRegimeScenario(
        label="HOBS paper exact (different scenario)",
        source="different-paper",
        p_tx_w=_HOBS_P_MAX_W, bw_mhz=_HOBS_BW_MHZ,
        n0_psd_dbm_hz=_MODQN_N0_PSD, f_c_ghz=_HOBS_FC_GHZ,
        alt_km=_HOBS_ALT_KM, n_neighbors=_MAX_NEIGHBORS,
        p_neighbor_w=_HOBS_P_MAX_W * 0.75, antenna_gain_db=_HOBS_G_T_DB,
        notes="HOBS full scenario: 28GHz, 550km, 100W, 40dBi, 100MHz; strong interference",
    ),
]


# ---------------------------------------------------------------------------
# Sweep execution
# ---------------------------------------------------------------------------

def run_channel_regime_sweep() -> list[dict[str, Any]]:
    """Run all sweep scenarios and return annotated results."""
    return [_compute_metrics(s) for s in SWEEP_SCENARIOS]


def minimum_antenna_gain_for_threshold(
    threshold: float,
    p_tx_w: float = _MODQN_P_TX_W,
    bw_mhz: float = _MODQN_BW_MHZ,
    n0_psd_dbm_hz: float = _MODQN_N0_PSD,
    f_c_ghz: float = _MODQN_FC_GHZ,
    alt_km: float = _MODQN_ALT_KM,
    n_neighbors: int = _MAX_NEIGHBORS,
    p_neighbor_w: float = _MODQN_P_TX_W,
) -> float:
    """Return antenna gain (dBi) needed to make I_intra/N0 >= threshold."""
    n0_psd_w_hz = dbm_to_w(n0_psd_dbm_hz)
    n0_w = n0_psd_w_hz * bw_mhz * 1e6
    d_m = alt_km * 1e3
    fspl = (3e8 / (4.0 * math.pi * d_m * f_c_ghz * 1e9)) ** 2
    # I/N0 = n_neighbors * p_neighbor * fspl * 10^(G/10) / N0 >= threshold
    # 10^(G/10) >= threshold * N0 / (n_neighbors * p_neighbor * fspl)
    ratio = threshold * n0_w / (n_neighbors * p_neighbor_w * fspl)
    if ratio <= 0:
        return -math.inf
    return 10.0 * math.log10(ratio)


def path1_path2_verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Assess Path 1 (channel regime) vs Path 2 (DPC sidecar) viability."""
    paper_backed = [r for r in rows if r["source"] == "paper-backed"]
    s10_backed   = [r for r in rows if r["source"] == "s10-backed"]
    sensitivity  = [r for r in rows if r["source"] == "sensitivity"]
    extension    = [r for r in rows if r["source"] == "extension"]
    diff_paper   = [r for r in rows if r["source"] == "different-paper"]

    paper_max_i_n0  = max((r["i_intra_n0_ratio"] for r in paper_backed), default=0)
    s10_max_i_n0    = max((r["i_intra_n0_ratio"] for r in s10_backed), default=0)
    sens_max_i_n0   = max((r["i_intra_n0_ratio"] for r in sensitivity), default=0)
    ext_max_i_n0    = max((r["i_intra_n0_ratio"] for r in extension), default=0)
    diff_max_i_n0   = max((r["i_intra_n0_ratio"] for r in diff_paper), default=0)

    any_paper_observable   = paper_max_i_n0 >= I_N0_OBSERVABLE
    any_s10_observable     = s10_max_i_n0   >= I_N0_OBSERVABLE
    any_sens_observable    = sens_max_i_n0  >= I_N0_OBSERVABLE
    any_ext_observable     = ext_max_i_n0   >= I_N0_OBSERVABLE
    any_diff_observable    = diff_max_i_n0  >= I_N0_OBSERVABLE

    # Required antenna gain for each threshold
    g_for_detectable  = minimum_antenna_gain_for_threshold(I_N0_DETECTABLE)
    g_for_observable  = minimum_antenna_gain_for_threshold(I_N0_OBSERVABLE)
    g_for_strong      = minimum_antenna_gain_for_threshold(I_N0_STRONG)

    path1_verdict = (
        "BLOCK"
        if not any_paper_observable and not any_s10_observable
        else "PASS"
    )
    path1_note = (
        "Path 1 is BLOCKED. No paper-backed or S10-backed parameter change "
        "makes I_intra/N0 >= 0.1. Only model extension (antenna gain ≥ "
        f"{g_for_observable:.1f} dBi) or a different paper's scenario "
        "crosses the observable threshold. "
        "Adding antenna gain would require a new model assumption not in "
        "the MODQN paper."
        if path1_verdict == "BLOCK"
        else "Path 1 may be viable — see sweep rows."
    )

    path2_verdict = "PASS"
    path2_note = (
        "Path 2 (HOBS-style DPC sidecar) is recommended as the next gate. "
        "DPC creates denominator variability through time-varying beam power "
        "(P_{n,m}(t) = P_{n,m}(t-T_f) + xi, with xi sign-flip on EE decrease). "
        "This is independent of interference magnitude and is directly backed "
        "by the HOBS paper formula. DPC does not require changing any channel "
        "parameter. DPC labeled as HOBS-inspired new extension, not MODQN paper-backed."
    )

    return {
        "paper_max_i_n0": paper_max_i_n0,
        "s10_max_i_n0": s10_max_i_n0,
        "sensitivity_max_i_n0": sens_max_i_n0,
        "extension_max_i_n0": ext_max_i_n0,
        "different_paper_max_i_n0": diff_max_i_n0,
        "any_paper_observable": any_paper_observable,
        "any_s10_observable": any_s10_observable,
        "any_sensitivity_observable": any_sens_observable,
        "any_extension_observable": any_ext_observable,
        "any_different_paper_observable": any_diff_observable,
        "minimum_antenna_gain_for_detectable_dbi": round(g_for_detectable, 1),
        "minimum_antenna_gain_for_observable_dbi": round(g_for_observable, 1),
        "minimum_antenna_gain_for_strong_dbi": round(g_for_strong, 1),
        "path1_verdict": path1_verdict,
        "path1_note": path1_note,
        "path2_verdict": path2_verdict,
        "path2_note": path2_note,
        "overall_recommendation": "Path 2 (DPC sidecar)",
        "i_n0_observable_threshold": I_N0_OBSERVABLE,
        "i_n0_strong_threshold": I_N0_STRONG,
        "sinr_drop_threshold": SINR_DROP_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_channel_regime_audit(output_dir: str | Path) -> dict[str, Any]:
    """Run sweep, write summary.json + sweep.csv + review.md, return summary."""
    from ._common import write_json

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = run_channel_regime_sweep()
    verdict = path1_path2_verdict(rows)

    summary: dict[str, Any] = {
        "namespace": "hobs-sinr-channel-regime-audit",
        "date": "2026-05-01",
        "purpose": "Determine whether Path 1 (channel regime) or Path 2 (DPC sidecar) is viable before Route D.",
        "thresholds": {
            "i_n0_detectable":    I_N0_DETECTABLE,
            "i_n0_observable":    I_N0_OBSERVABLE,
            "i_n0_strong":        I_N0_STRONG,
            "sinr_drop_ratio":    SINR_DROP_THRESHOLD,
        },
        "n_scenarios_swept": len(rows),
        "verdict": verdict,
        "sweep_rows": rows,
        "forbidden_claims": [
            "Do not claim HOBS SINR full reproduction.",
            "Do not claim EE-MODQN effectiveness.",
            "Do not claim physical energy saving.",
            "Do not label extension/sensitivity scenarios as paper-backed.",
            "Do not run MODQN training based on this audit.",
        ],
    }

    summary_path = write_json(out / "summary.json", summary)

    # Write sweep.csv
    csv_path = out / "sweep.csv"
    fieldnames = [
        "label", "source", "p_tx_w", "bw_mhz", "f_c_ghz", "alt_km",
        "n_neighbors", "p_neighbor_w", "antenna_gain_db",
        "snr_db", "i_intra_n0_ratio", "sinr_drop_ratio",
        "interference_status", "exceeds_observable_threshold",
        "exceeds_sinr_drop_threshold", "notes",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Write review.md
    review_path = out / "review.md"
    _write_review_md(review_path, rows, verdict)

    return {
        "summary_path": summary_path,
        "sweep_csv_path": csv_path,
        "review_md_path": review_path,
        "summary": summary,
    }


def _write_review_md(path: Path, rows: list[dict], verdict: dict) -> None:
    lines = [
        "# Route A2: HOBS-style SINR Channel-Regime Audit",
        "",
        "**Date:** 2026-05-01  ",
        "**Namespace:** hobs-sinr-channel-regime-audit  ",
        "**Status:** see verdict below",
        "",
        "## Purpose",
        "",
        "Determine whether any paper-defensible or sensitivity-defensible",
        "channel parameter regime makes intra-satellite SINR interference",
        "numerically observable (I_intra/N0 ≥ 0.1), enabling Route D",
        "(tiny learned pilot) to produce genuine EE-differentiated behavior.",
        "",
        "## Observability Thresholds",
        "",
        f"- `I_intra/N0 >= {I_N0_DETECTABLE}`: weak but detectable",
        f"- `I_intra/N0 >= {I_N0_OBSERVABLE}`: observable",
        f"- `I_intra/N0 >= {I_N0_STRONG}`: strong",
        f"- `SINR_drop_ratio <= {SINR_DROP_THRESHOLD}`: >=10% SINR degradation",
        "",
        "## Sweep Results",
        "",
        "| Label | Source | SNR_dB | I/N0 | SINR_drop | Status |",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['source']} | "
            f"{r['snr_db']:.1f} | {r['i_intra_n0_ratio']:.2e} | "
            f"{r['sinr_drop_ratio']:.4f} | {r['interference_status']} |"
        )
    lines += [
        "",
        "## Key Finding",
        "",
        "**No paper-backed or S10-backed parameter change makes interference observable.**",
        "",
        f"- Paper-backed max I/N0: `{verdict['paper_max_i_n0']:.2e}` (MODQN Table I)",
        f"- S10-backed max I/N0: `{verdict['s10_max_i_n0']:.2e}` (10W per beam)",
        f"- Sensitivity max I/N0: `{verdict['sensitivity_max_i_n0']:.2e}` (narrow BW / lower altitude)",
        f"- Extension max I/N0: `{verdict['extension_max_i_n0']:.2e}` (40dBi antenna, not in MODQN)",
        "",
        "Minimum antenna gain to reach observable threshold:",
        f"- `I/N0 >= 0.01`: **{verdict['minimum_antenna_gain_for_detectable_dbi']} dBi**",
        f"- `I/N0 >= 0.1` : **{verdict['minimum_antenna_gain_for_observable_dbi']} dBi**",
        f"- `I/N0 >= 1.0` : **{verdict['minimum_antenna_gain_for_strong_dbi']} dBi**",
        "",
        "HOBS uses 40 dBi, which just crosses the observable threshold.",
        "MODQN paper has no antenna gain term in its channel formula.",
        "Adding antenna gain is a model extension, not a paper-backed change.",
        "",
        "## Path 1 Verdict",
        "",
        f"**{verdict['path1_verdict']}**",
        "",
        verdict["path1_note"],
        "",
        "## Path 2 Recommendation",
        "",
        f"**{verdict['path2_verdict']}**",
        "",
        verdict["path2_note"],
        "",
        "## Recommended Next Step",
        "",
        "Implement HOBS-style DPC (Dynamic Power Control) sidecar:",
        "",
        "```text",
        "P_{n,m}(t) = P_{n,m}(t-T_f) + xi_{n,m}(t)",
        "if EE_beam_{n,m}(t-1) <= EE_beam_{n,m}(t-2): xi = -xi",
        "if SINR_u < threshold: xi = abs(xi)  [QoS guard]",
        "P_{n,m}(t) = clip(P_{n,m}(t), 0, P_beam_max)",
        "```",
        "",
        "DPC creates time-varying denominator (p_{s,b}^t changes each step)",
        "independent of interference strength.",
        "",
        "## Forbidden Claims",
        "",
    ]
    for claim in [
        "Do not claim HOBS SINR full reproduction.",
        "Do not claim EE-MODQN effectiveness.",
        "Do not claim physical energy saving.",
        "Do not label extension/sensitivity scenarios as paper-backed.",
    ]:
        lines.append(f"- {claim}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


__all__ = [
    "ChannelRegimeScenario",
    "SWEEP_SCENARIOS",
    "I_N0_OBSERVABLE",
    "I_N0_STRONG",
    "SINR_DROP_THRESHOLD",
    "run_channel_regime_sweep",
    "minimum_antenna_gain_for_threshold",
    "path1_path2_verdict",
    "export_channel_regime_audit",
]
