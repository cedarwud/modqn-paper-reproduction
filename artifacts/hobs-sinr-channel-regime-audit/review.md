# Route A2: HOBS-style SINR Channel-Regime Audit

**Date:** 2026-05-01
**Namespace:** hobs-sinr-channel-regime-audit
**Status:** see verdict below

## Purpose

Determine whether any paper-defensible or sensitivity-defensible
channel parameter regime makes intra-satellite SINR interference
numerically observable (I_intra/N0 ≥ 0.1), enabling Route D
(tiny learned pilot) to produce genuine EE-differentiated behavior.

## Observability Thresholds

- `I_intra/N0 >= 0.01`: weak but detectable
- `I_intra/N0 >= 0.1`: observable
- `I_intra/N0 >= 1.0`: strong
- `SINR_drop_ratio <= 0.9`: >=10% SINR degradation

## Sweep Results

| Label | Source | SNR_dB | I/N0 | SINR_drop | Status |
|---|---|---:|---:|---:|---|
| MODQN paper exact (Table I) | paper-backed | -56.3 | 1.41e-05 | 1.0000 | negligible |
| S10 per-beam max 10W (MAAC-BHPOWER) | s10-backed | -49.3 | 7.06e-05 | 0.9999 | negligible |
| S10 sat-max 20W / 7 beams | s10-backed | -54.8 | 2.01e-05 | 1.0000 | negligible |
| LEO 550km altitude (HOBS altitude) | sensitivity | -53.2 | 2.84e-05 | 1.0000 | negligible |
| LEO 300km (Starlink-class) | sensitivity | -48.0 | 9.54e-05 | 0.9999 | negligible |
| 100MHz BW (HOBS-backed bandwidth) | sensitivity | -49.3 | 7.06e-05 | 0.9999 | negligible |
| 10MHz BW (narrow, sensitivity) | sensitivity | -39.3 | 7.06e-04 | 0.9993 | negligible |
| MODQN + 30dBi antenna gain (extension) | extension | -26.3 | 1.41e-02 | 0.9861 | detectable |
| MODQN + 40dBi antenna gain (HOBS G_T, extension) | extension | -16.3 | 1.41e-01 | 0.8763 | observable |
| HOBS paper exact (different scenario) | different-paper | 7.8 | 2.72e+01 | 0.0355 | strong |

## Key Finding

**No paper-backed or S10-backed parameter change makes interference observable.**

- Paper-backed max I/N0: `1.41e-05` (MODQN Table I)
- S10-backed max I/N0: `7.06e-05` (10W per beam)
- Sensitivity max I/N0: `7.06e-04` (narrow BW / lower altitude)
- Extension max I/N0: `1.41e-01` (40dBi antenna, not in MODQN)

Minimum antenna gain to reach observable threshold:
- `I/N0 >= 0.01`: **28.5 dBi**
- `I/N0 >= 0.1` : **38.5 dBi**
- `I/N0 >= 1.0` : **48.5 dBi**

HOBS uses 40 dBi, which just crosses the observable threshold.
MODQN paper has no antenna gain term in its channel formula.
Adding antenna gain is a model extension, not a paper-backed change.

## Path 1 Verdict

**BLOCK**

Path 1 is BLOCKED. No paper-backed or S10-backed parameter change makes I_intra/N0 >= 0.1. Only model extension (antenna gain ≥ 38.5 dBi) or a different paper's scenario crosses the observable threshold. Adding antenna gain would require a new model assumption not in the MODQN paper.

## Path 2 Recommendation

**PASS**

Path 2 (HOBS-style DPC sidecar) is recommended as the next gate. DPC creates denominator variability through time-varying beam power (P_{n,m}(t) = P_{n,m}(t-T_f) + xi, with xi sign-flip on EE decrease). This is independent of interference magnitude and is directly backed by the HOBS paper formula. DPC does not require changing any channel parameter. DPC labeled as HOBS-inspired new extension, not MODQN paper-backed.

## Recommended Next Step

Implement HOBS-style DPC (Dynamic Power Control) sidecar:

```text
P_{n,m}(t) = P_{n,m}(t-T_f) + xi_{n,m}(t)
if EE_beam_{n,m}(t-1) <= EE_beam_{n,m}(t-2): xi = -xi
if SINR_u < threshold: xi = abs(xi)  [QoS guard]
P_{n,m}(t) = clip(P_{n,m}(t), 0, P_beam_max)
```

DPC creates time-varying denominator (p_{s,b}^t changes each step)
independent of interference strength.

## Forbidden Claims

- Do not claim HOBS SINR full reproduction.
- Do not claim EE-MODQN effectiveness.
- Do not claim physical energy saving.
- Do not label extension/sensitivity scenarios as paper-backed.
