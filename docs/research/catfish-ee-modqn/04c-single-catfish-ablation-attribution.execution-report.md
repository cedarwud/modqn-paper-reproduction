# Phase 04C Execution Report: Single Catfish-MODQN Ablation Attribution

**Date:** `2026-04-29`
**Status:** `PASS for bounded ablation-attribution run completion; NEEDS MORE EVIDENCE for effectiveness`
**Scope:** bounded Phase 04C attribution over the Phase 04B single-Catfish implementation surface. No frozen baseline config/artifact mutation, EE objective, EE reward, long training, or multi-Catfish implementation was performed.

## Guardrail Boundary

Phase 04C continued to use the original MODQN reward surface:

```text
r1 = throughput
r2 = handover penalty
r3 = load balance
```

The Catfish operational details remain config-defined local assumptions, not source-backed facts:

1. intervention cadence,
2. warmup trigger and threshold,
3. high-value quantile threshold and rolling window,
4. duplicate-high-value replay partition semantics,
5. MODQN quality score adaptation `quality = 0.5*r1 + 0.3*r2 + 0.2*r3`.

## Runs

All runs used the same Phase 04B config surface, seeds, 20-episode bounded budget, evaluation cadence, batch size, and checkpoint rule. Outputs were written under new Phase 04C artifact namespaces so the Phase 04B runnable-evidence artifacts were not overwritten.

| Role | Config | Artifact |
|---|---|---|
| matched control | `configs/catfish-modqn-phase-04-b-control.resolved.yaml` | `artifacts/catfish-modqn-phase-04-c-control-20ep/` |
| primary shaping-off | `configs/catfish-modqn-phase-04-b-primary-shaping-off.resolved.yaml` | `artifacts/catfish-modqn-phase-04-c-primary-shaping-off-20ep/` |
| ablation: no intervention | `configs/catfish-modqn-phase-04-b-no-intervention.resolved.yaml` | `artifacts/catfish-modqn-phase-04-c-no-intervention-20ep/` |
| ablation: no asymmetric gamma | `configs/catfish-modqn-phase-04-b-no-asymmetric-gamma.resolved.yaml` | `artifacts/catfish-modqn-phase-04-c-no-asymmetric-gamma-20ep/` |

## Final Training Rows

| Role | Scalar | r1 | r2 | r3 | Handovers |
|---|---:|---:|---:|---:|---:|
| matched control | `607.063716` | `1224.617563` | `-4.320000` | `-19.745327` | `864` |
| primary shaping-off | `608.495986` | `1226.380362` | `-4.395000` | `-16.878477` | `879` |
| no intervention | `606.844407` | `1224.028529` | `-4.275000` | `-19.436788` | `855` |
| no asymmetric gamma | `608.495986` | `1226.380362` | `-4.395000` | `-16.878477` | `879` |

Training scalar AUC mean:

| Role | Mean Scalar | Best Train Scalar | Best Episode | Final Minus Best |
|---|---:|---:|---:|---:|
| matched control | `605.980193` | `615.204042` | `15` | `-8.140326` |
| primary shaping-off | `606.164290` | `612.700001` | `15` | `-4.204015` |
| no intervention | `606.436111` | `614.343665` | `15` | `-7.499258` |
| no asymmetric gamma | `606.164290` | `612.700001` | `15` | `-4.204015` |

## Best-Eval Checkpoint Rows

| Role | Best-Eval Episode | Mean Scalar | Mean r1 | Mean r2 | Mean r3 | Mean Handovers |
|---|---:|---:|---:|---:|---:|---:|
| matched control | `4` | `52.258708` | `174.618692` | `-0.423000` | `-174.618692` | `84.60` |
| primary shaping-off | `9` | `52.260508` | `174.618692` | `-0.417000` | `-174.618692` | `83.40` |
| no intervention | `9` | `52.260508` | `174.618692` | `-0.417000` | `-174.618692` | `83.40` |
| no asymmetric gamma | `9` | `52.260508` | `174.618692` | `-0.417000` | `-174.618692` | `83.40` |

## Catfish Diagnostics

| Role | Main Updates | Catfish Updates | Interventions | Catfish Samples Used | Actual Catfish Ratio | Catfish Replay Size | NaN |
|---|---:|---:|---:|---:|---:|---:|---|
| primary shaping-off | `200` | `194` | `196` | `3724` | `0.296875` | `3929` | `false` |
| no intervention | `200` | `194` | `0` | `0` | `null` | `3956` | `false` |
| no asymmetric gamma | `200` | `194` | `196` | `3724` | `0.296875` | `3929` | `false` |

Last-episode stability checks:

| Role | Main Q Abs Max | Catfish Q Abs Max | Catfish Empty After Warmup | Starved Training | Starved Intervention |
|---|---:|---:|---|---:|---:|
| primary shaping-off | `99.876068` | `101.294357` | `false` | `0` | `0` |
| no intervention | `98.154274` | `104.844612` | `false` | `0` | `0` |
| no asymmetric gamma | `99.876068` | `100.456932` | `false` | `0` | `0` |

## Interpretation

Phase 04C confirms that the bounded attribution grid is runnable and that the Catfish mechanism participates in training when intervention is enabled:

1. catfish replay is non-empty after warmup,
2. intervention triggers in the primary shaping-off run,
3. actual mixed replay ratio closely matches the configured `0.30`,
4. no replay starvation or NaN was observed in the bounded runs.

The evidence is not enough for an effectiveness claim:

1. this is one seed and only `20` episodes,
2. best-eval metrics are effectively tied across Catfish variants,
3. primary and no-asymmetric-gamma final rows are identical on this bounded run, so asymmetric gamma contribution is not distinguishable here,
4. no-intervention has a lower final scalar than primary but a higher mean training scalar, so intervention attribution remains weak,
5. scalar reward remains scale-dominated by `r1`, so scalar-only interpretation is disallowed.

## Decision

```text
Phase 04C bounded attribution run completion: PASS
Catfish-MODQN effectiveness promotion: FAIL / NEEDS MORE EVIDENCE
Phase 05 full multi-Catfish authorization: BLOCKED
Phase 05A multi-buffer planning: allowed only as a separate bounded planning gate
```

## Stop / Claim Rules

Do not claim:

1. Catfish-MODQN is better than MODQN from this bounded run,
2. asymmetric gamma is a proven contributor,
3. Catfish solves late-training collapse,
4. Catfish improves EE,
5. Catfish-EE-MODQN,
6. multi-Catfish readiness,
7. full paper-faithful reproduction.

## Verification

Focused Phase 04B/04C regression:

```text
.venv/bin/python -m pytest tests/test_catfish_phase04b.py -q
9 passed
```
