"""Route A2: Channel-regime audit tests.

Namespace: hobs-sinr-channel-regime-audit
Date: 2026-05-01

Tests prove:
  1. All paper-backed MODQN Table I parameters produce negligible interference.
  2. S10-backed power caps (10W/beam) also produce negligible interference.
  3. Sensitivity parameters (altitude, bandwidth) do not cross observable threshold.
  4. Extension (antenna gain) can cross observable threshold — but is not in MODQN.
  5. Minimum antenna gain required for each observability threshold is correct.
  6. Path 1 verdict is BLOCK (no paper/S10-backed scenario produces observable I/N0).
  7. Path 2 verdict is PASS (DPC sidecar recommended).
  8. sweep contains required parameter-source categories.
  9. export function runs without error and writes required files.
 10. config loads without error.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest

from modqn_paper_reproduction.analysis.hobs_sinr_channel_regime_audit import (
    I_N0_DETECTABLE,
    I_N0_OBSERVABLE,
    I_N0_STRONG,
    SINR_DROP_THRESHOLD,
    SWEEP_SCENARIOS,
    export_channel_regime_audit,
    minimum_antenna_gain_for_threshold,
    path1_path2_verdict,
    run_channel_regime_sweep,
)
from modqn_paper_reproduction.config_loader import build_environment, load_training_yaml

_REGIME_AUDIT_CONFIG = "configs/hobs-sinr-channel-regime-audit.resolved.yaml"
_BASELINE_CONFIG = "configs/modqn-paper-baseline.resolved-template.yaml"


# ---------------------------------------------------------------------------
# 1. MODQN paper exact — interference negligible
# ---------------------------------------------------------------------------

def test_modqn_paper_exact_interference_negligible() -> None:
    rows = run_channel_regime_sweep()
    paper_row = next(r for r in rows if r["source"] == "paper-backed")
    assert paper_row["i_intra_n0_ratio"] < I_N0_DETECTABLE, (
        f"MODQN Table I parameters must produce negligible interference "
        f"(I/N0 < {I_N0_DETECTABLE}), got {paper_row['i_intra_n0_ratio']:.2e}"
    )
    assert paper_row["interference_status"] == "negligible"
    assert paper_row["exceeds_observable_threshold"] is False


def test_modqn_paper_snr_is_very_low() -> None:
    rows = run_channel_regime_sweep()
    paper_row = next(r for r in rows if r["source"] == "paper-backed")
    # SNR at zenith should be approximately -56 dB (confirmed from Route A)
    assert paper_row["snr_db"] < -50.0, (
        f"MODQN Table I operating SNR should be below -50 dB (negligible "
        f"interference regime), got {paper_row['snr_db']:.1f} dB"
    )


# ---------------------------------------------------------------------------
# 2. S10 power caps — still negligible
# ---------------------------------------------------------------------------

def test_s10_power_caps_still_negligible() -> None:
    rows = run_channel_regime_sweep()
    s10_rows = [r for r in rows if r["source"] == "s10-backed"]
    assert len(s10_rows) >= 1, "Must have at least one S10-backed scenario"
    for r in s10_rows:
        assert r["i_intra_n0_ratio"] < I_N0_OBSERVABLE, (
            f"S10-backed scenario '{r['label']}' must remain below observable threshold "
            f"({I_N0_OBSERVABLE}), got {r['i_intra_n0_ratio']:.2e}"
        )


def test_s10_10w_per_beam_i_n0_value() -> None:
    rows = run_channel_regime_sweep()
    s10_row = next(r for r in rows if "10W" in r["label"] and "S10" in r["label"])
    # With 10W (5x MODQN paper 2W), I/N0 should be ~5x baseline = ~5e-5
    assert s10_row["i_intra_n0_ratio"] < 1e-3
    assert s10_row["i_intra_n0_ratio"] > 1e-6


# ---------------------------------------------------------------------------
# 3. Sensitivity scenarios do not cross observable threshold
# ---------------------------------------------------------------------------

def test_sensitivity_scenarios_below_observable() -> None:
    rows = run_channel_regime_sweep()
    sensitivity_rows = [r for r in rows if r["source"] == "sensitivity"]
    for r in sensitivity_rows:
        assert r["i_intra_n0_ratio"] < I_N0_OBSERVABLE, (
            f"Sensitivity scenario '{r['label']}' must remain below observable threshold, "
            f"got {r['i_intra_n0_ratio']:.2e}"
        )


# ---------------------------------------------------------------------------
# 4. Extension (antenna gain) crosses observable — but not paper-backed
# ---------------------------------------------------------------------------

def test_40dbi_antenna_gain_crosses_observable_threshold() -> None:
    """Adding 40dBi antenna gain (HOBS G_T) just crosses the observable threshold."""
    rows = run_channel_regime_sweep()
    ext_row = next(r for r in rows if r["source"] == "extension" and "40dBi" in r["label"])
    assert ext_row["i_intra_n0_ratio"] >= I_N0_OBSERVABLE, (
        f"With 40dBi antenna gain, I/N0 should be >= {I_N0_OBSERVABLE}, "
        f"got {ext_row['i_intra_n0_ratio']:.2e}"
    )
    assert ext_row["source"] == "extension", "40dBi antenna gain must be labeled 'extension'"


def test_30dbi_antenna_gain_still_below_observable() -> None:
    """30dBi antenna gain is still below observable threshold."""
    rows = run_channel_regime_sweep()
    ext_row_30 = next(
        (r for r in rows if r["source"] == "extension" and "30dBi" in r["label"]),
        None,
    )
    if ext_row_30 is not None:
        assert ext_row_30["i_intra_n0_ratio"] < I_N0_OBSERVABLE, (
            f"30dBi antenna gain should be below observable ({I_N0_OBSERVABLE}), "
            f"got {ext_row_30['i_intra_n0_ratio']:.2e}"
        )


# ---------------------------------------------------------------------------
# 5. Minimum antenna gain formulas are correct
# ---------------------------------------------------------------------------

def test_minimum_antenna_gain_for_observable_threshold() -> None:
    """Minimum G_T required: ~38.5 dBi for I/N0 >= 0.1."""
    g_for_obs = minimum_antenna_gain_for_threshold(I_N0_OBSERVABLE)
    # Should be approximately 38-40 dBi
    assert 35.0 <= g_for_obs <= 42.0, (
        f"Minimum antenna gain for observable threshold should be ~38-40 dBi, "
        f"got {g_for_obs:.1f} dBi"
    )
    # HOBS uses 40 dBi, which should be >= this threshold
    assert g_for_obs <= 40.0 + 0.5, (
        f"HOBS 40dBi should cross the observable threshold "
        f"(minimum={g_for_obs:.1f} dBi)"
    )


def test_minimum_antenna_gain_for_strong_threshold() -> None:
    g_for_strong = minimum_antenna_gain_for_threshold(I_N0_STRONG)
    assert g_for_strong > minimum_antenna_gain_for_threshold(I_N0_OBSERVABLE), (
        "Strong threshold requires more antenna gain than observable threshold"
    )


def test_minimum_antenna_gain_scales_with_threshold() -> None:
    """Higher threshold requires more antenna gain (monotone relationship)."""
    g_det = minimum_antenna_gain_for_threshold(I_N0_DETECTABLE)
    g_obs = minimum_antenna_gain_for_threshold(I_N0_OBSERVABLE)
    g_str = minimum_antenna_gain_for_threshold(I_N0_STRONG)
    assert g_det < g_obs < g_str


def test_antenna_gain_formula_exact_value() -> None:
    """Verify the formula: G_needed = 10*log10(threshold * N0 / (n * P * FSPL))."""
    import math as m
    from modqn_paper_reproduction.env.channel import dbm_to_w as d2w
    bw = 500e6; n0_psd = -174.0; P = 2.0; n = 6; f = 20e9; alt = 780e3
    n0 = d2w(n0_psd) * bw
    fspl = (3e8 / (4 * m.pi * alt * f))**2
    expected = 10 * m.log10(0.1 * n0 / (n * P * fspl))
    computed = minimum_antenna_gain_for_threshold(0.1)
    assert abs(expected - computed) < 0.01, (
        f"Antenna gain formula mismatch: expected {expected:.2f}, got {computed:.2f}"
    )


# ---------------------------------------------------------------------------
# 6. Path 1 verdict is BLOCK
# ---------------------------------------------------------------------------

def test_path1_verdict_is_block() -> None:
    rows = run_channel_regime_sweep()
    verdict = path1_path2_verdict(rows)
    assert verdict["path1_verdict"] == "BLOCK", (
        "Path 1 must be BLOCK: no paper-backed or S10-backed scenario "
        "produces observable interference"
    )


def test_path1_block_because_no_paper_s10_observable() -> None:
    rows = run_channel_regime_sweep()
    verdict = path1_path2_verdict(rows)
    assert verdict["any_paper_observable"] is False
    assert verdict["any_s10_observable"] is False


# ---------------------------------------------------------------------------
# 7. Path 2 verdict is PASS
# ---------------------------------------------------------------------------

def test_path2_verdict_is_pass() -> None:
    rows = run_channel_regime_sweep()
    verdict = path1_path2_verdict(rows)
    assert verdict["path2_verdict"] == "PASS", (
        "Path 2 (DPC sidecar) must be PASS"
    )


def test_path2_note_mentions_dpc() -> None:
    rows = run_channel_regime_sweep()
    verdict = path1_path2_verdict(rows)
    assert "DPC" in verdict["path2_note"] or "sidecar" in verdict["path2_note"]


def test_overall_recommendation_is_path2() -> None:
    rows = run_channel_regime_sweep()
    verdict = path1_path2_verdict(rows)
    assert "Path 2" in verdict["overall_recommendation"] or "DPC" in verdict["overall_recommendation"]


# ---------------------------------------------------------------------------
# 8. Sweep contains required parameter-source categories
# ---------------------------------------------------------------------------

def test_sweep_covers_all_required_source_categories() -> None:
    sources = {s.source for s in SWEEP_SCENARIOS}
    required = {"paper-backed", "s10-backed", "sensitivity", "extension", "different-paper"}
    missing = required - sources
    assert not missing, f"Sweep missing source categories: {missing}"


def test_sweep_has_at_least_one_paper_backed_row() -> None:
    rows = run_channel_regime_sweep()
    paper_rows = [r for r in rows if r["source"] == "paper-backed"]
    assert len(paper_rows) >= 1


def test_different_paper_row_shows_strong_interference() -> None:
    """HOBS full scenario should show strong interference."""
    rows = run_channel_regime_sweep()
    diff_rows = [r for r in rows if r["source"] == "different-paper"]
    assert any(r["i_intra_n0_ratio"] >= I_N0_STRONG for r in diff_rows), (
        "HOBS full parameter set should produce strong interference"
    )


# ---------------------------------------------------------------------------
# 9. Export function writes required files
# ---------------------------------------------------------------------------

def test_export_writes_required_files() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        result = export_channel_regime_audit(tmpdir)
        assert Path(result["summary_path"]).exists()
        assert Path(result["sweep_csv_path"]).exists()
        assert Path(result["review_md_path"]).exists()


def test_export_summary_contains_verdict() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        result = export_channel_regime_audit(tmpdir)
        import json
        summary = json.loads(Path(result["summary_path"]).read_text())
        assert "verdict" in summary
        assert summary["verdict"]["path1_verdict"] == "BLOCK"
        assert summary["verdict"]["path2_verdict"] == "PASS"


def test_export_csv_has_all_scenarios() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        result = export_channel_regime_audit(tmpdir)
        csv_text = Path(result["sweep_csv_path"]).read_text()
        rows_count = len(csv_text.strip().split("\n")) - 1  # minus header
        assert rows_count == len(SWEEP_SCENARIOS)


def test_review_md_mentions_path1_block_and_path2_pass() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        result = export_channel_regime_audit(tmpdir)
        review = Path(result["review_md_path"]).read_text()
        assert "BLOCK" in review
        assert "PASS" in review
        assert "DPC" in review


# ---------------------------------------------------------------------------
# 10. Config loads correctly
# ---------------------------------------------------------------------------

def test_regime_audit_config_loads() -> None:
    cfg = load_training_yaml(_REGIME_AUDIT_CONFIG)
    env = build_environment(cfg)
    assert env.power_surface_config.hobs_power_surface_mode == "active-load-concave"
    # SINR interference disabled in this audit config (interference negligible at operating point)
    assert env.power_surface_config.sinr_intra_satellite_interference is False


def test_baseline_config_unaffected() -> None:
    cfg = load_training_yaml(_BASELINE_CONFIG)
    env = build_environment(cfg)
    assert env.power_surface_config.hobs_power_surface_mode == "static-config"
    assert env.power_surface_config.sinr_intra_satellite_interference is False


# ---------------------------------------------------------------------------
# Analytical correctness
# ---------------------------------------------------------------------------

def test_interference_metric_formula_correctness() -> None:
    """Verify _compute_metrics matches known analytical result."""
    from modqn_paper_reproduction.analysis.hobs_sinr_channel_regime_audit import (
        ChannelRegimeScenario, _compute_metrics
    )
    import math as m
    from modqn_paper_reproduction.env.channel import dbm_to_w

    s = ChannelRegimeScenario(
        label="test", source="test",
        p_tx_w=2.0, bw_mhz=500.0, n0_psd_dbm_hz=-174.0,
        f_c_ghz=20.0, alt_km=780.0,
        n_neighbors=6, p_neighbor_w=1.5, antenna_gain_db=0.0,
    )
    r = _compute_metrics(s)

    # Manual computation
    n0 = dbm_to_w(-174.0) * 500e6
    fspl = (3e8 / (4 * m.pi * 780e3 * 20e9))**2
    h = fspl  # no antenna gain
    snr_db = 10 * m.log10(2.0 * h / n0)
    i_n0 = 6 * 1.5 * h / n0

    assert abs(r["snr_db"] - snr_db) < 0.1
    assert abs(r["i_intra_n0_ratio"] - i_n0) / i_n0 < 1e-6


def test_thresholds_are_ordered() -> None:
    """Observability thresholds must be strictly ordered."""
    assert I_N0_DETECTABLE < I_N0_OBSERVABLE < I_N0_STRONG
    assert 0 < SINR_DROP_THRESHOLD < 1.0
