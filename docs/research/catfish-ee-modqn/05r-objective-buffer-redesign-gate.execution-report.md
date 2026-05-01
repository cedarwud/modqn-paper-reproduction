# Phase 05R Execution Report: Objective-Buffer Redesign Gate

**Date:** `2026-04-29`
**Status:** `PASS for later Phase 05B planning draft only`
**Scope:** offline objective-buffer admission redesign diagnostics over existing
Phase `05A` transition-level samples. No training, full Multi-Catfish agents,
`catfish-r1` / `catfish-r2` / `catfish-r3` learners, frozen baseline mutation,
reward / state / action / backbone change, EE objective, or Phase `05B`
implementation was performed.

Diagnostic artifact:

```text
artifacts/catfish-modqn-phase-05r-objective-buffer-redesign-gate/phase05r_objective_buffer_redesign_diagnostics.json
```

Source samples:

```text
artifacts/catfish-modqn-phase-05a-multi-buffer-primary-20ep/phase05a_transition_samples.jsonl
```

## 1. Current State

The fixed MODQN reward semantics remain:

```text
r1 = throughput
r2 = handover penalty
r3 = load balance
```

Phase `05B` and full objective-specialist Multi-Catfish remain blocked unless
Phase `05R` proves that objective buffers can be selective, distinct, and
interpretable without changing those semantics.

Predeclared Phase `05R` gate thresholds used here:

| Gate | Threshold |
|---|---:|
| admission-share lower bound | `0.10` |
| admission-share upper bound | `0.30` |
| scalar / high-throughput duplicate Jaccard fail threshold | `0.50` |
| pairwise objective-buffer Jaccard review threshold | `0.35` |
| minimum distinct intervention share per buffer | `0.10` |
| non-target damage tolerance | `5%` of absolute all-sample mean |

All objectives are interpreted with larger-is-better diagnostics: for `r2` and
`r3`, values closer to zero are better.

## 2. Phase 05A Disposition

Phase `05A` completed the bounded diagnostic run but failed objective-buffer
distinctness:

```text
Phase 05A bounded diagnostic completion: PASS
Objective-specific buffer distinctness: FAIL
Phase 05B full multi-agent validation planning: BLOCK
```

The failed construction had `r1` nearly duplicate scalar high-value replay,
`r2` percentile admission degenerate to all `20000` samples, and `r3` unable to
contribute distinct intervention samples after the `r2` full-set admission.

## 3. Phase 05R Redesign Candidate

Candidate name:

```text
guarded-residual-objective-admission
```

Admission rules:

| Buffer | Direction | Admission rule |
|---|---|---|
| `r1` | larger throughput is better | admit scalar-distinct high-throughput residual samples, with `r3 >= all-sample p25`; threshold `r1 >= 131.665558` |
| `r2` | handover penalty closer to zero is better | admit the complete strict best-score tie group, `r2 == 0.0`; do not expand into the next coarse tie |
| `r3` | load-balance penalty closer to zero is better | admit load-balance samples with throughput guardrail `r1 >= all-sample p25`; threshold `r3 >= -1.495561` |

Tie policy:

```text
complete threshold ties only; no random or partial tie-breaking
```

The `r1` scalar exclusion is a de-duplication guard against the known Phase
`05A` failure mode. It is not scalar reward success evidence.

## 4. Diagnostics Results

Admission, thresholds, tie mass, and unique values:

| Buffer | Size | Share | Threshold / rule | Tie mass | Selected objective unique values |
|---|---:|---:|---|---:|---:|
| `r1` | `2000` | `0.100000` | `r1 >= 131.665558`, scalar-distinct, `r3 >= -2.155818` | `1` | `2000` |
| `r2` | `2834` | `0.141700` | `r2 == 0.0` | `2834` | `1` |
| `r3` | `4061` | `0.203050` | `r3 >= -1.495561`, `r1 >= 96.775074` | `70` | `55` |

Pairwise Jaccard:

| Pair | Jaccard |
|---|---:|
| `r1` / `r2` | `0.063119` |
| `r1` / `r3` | `0.126371` |
| `r2` / `r3` | `0.087196` |

Jaccard versus scalar Phase `04` high-value replay:

| Buffer | Jaccard |
|---|---:|
| `r1` | `0.000000` |
| `r2` | `0.084847` |
| `r3` | `0.172029` |

Jaccard versus the failed Phase `05A` high-throughput `r1` top set:

| Buffer | Jaccard |
|---|---:|
| `r1` | `0.018330` |
| `r2` | `0.086486` |
| `r3` | `0.169447` |

Objective distributions, shown as means:

| Buffer | `r1` mean | `r2` mean | `r3` mean |
|---|---:|---:|---:|
| all samples | `122.228478` | `-0.429150` | `-1.821705` |
| `r1` | `139.708635` | `-0.428250` | `-1.582949` |
| `r2` | `121.566176` | `0.000000` | `-1.821950` |
| `r3` | `136.147460` | `-0.431913` | `-1.111403` |

Distinct samples versus scalar plus the other objective buffers:

| Buffer | Distinct count | Distinct share | Expected distinct slots in `10` specialist draws |
|---|---:|---:|---:|
| `r1` | `1116` | `0.558000` | `5.580000` |
| `r2` | `1703` | `0.600917` | `6.009174` |
| `r3` | `1894` | `0.466388` | `4.663876` |

Non-target degradation table:

| Buffer | Target delta vs all | Non-target deltas vs all | Significant non-target damage |
|---|---:|---|---|
| `r1` | `r1 +17.480156` | `r2 +0.000900`, `r3 +0.238756` | no |
| `r2` | `r2 +0.429150` | `r1 -0.662302`, `r3 -0.000246` | no |
| `r3` | `r3 +0.710301` | `r1 +13.918982`, `r2 -0.002763` | no |

Fixed-budget intervention-composition simulation:

```text
70 main + 10 r1 + 10 r2 + 10 r3
```

Expected mixed-batch means:

| Metric | All-sample mean | Mixed-batch expected mean | Delta |
|---|---:|---:|---:|
| `r1` | `122.228478` | `125.302162` | `+3.073684` |
| `r2` | `-0.429150` | `-0.386421` | `+0.042729` |
| `r3` | `-1.821705` | `-1.726823` | `+0.094881` |

The fixed-budget composition remains diagnostic-only. It does not update a
learner and does not establish Multi-Catfish effectiveness.

## 5. Acceptance Criteria Check

| Criterion | Result |
|---|---|
| every buffer is bounded, non-empty, and not all samples | pass |
| admission shares are within predeclared `0.10` to `0.30` bounds | pass |
| `r1` does not duplicate scalar / high-throughput replay under Jaccard threshold | pass |
| `r2` coarse/tie degeneration is resolved by a non-arbitrary rule | pass |
| every buffer contributes measurable unique samples to the simulated mix | pass |
| target-objective lift is visible without significant non-target damage | pass |
| result is explainable as objective-specialized replay, not random data mixing | pass |
| success evidence does not depend on scalar reward alone | pass |

## 6. Stop Conditions

| Stop condition | Triggered |
|---|---|
| any buffer admits all samples or no samples | no |
| `r1` remains scalar / high-throughput duplicate | no |
| `r2` still degenerates because of coarse/tied values | no |
| `r3` distinctness comes only from throughput collapse or non-target damage | no |
| distinctness requires reward / state / action / backbone change or EE | no |
| full specialist learners are needed to determine distinctness | no |
| result can only be explained by scalar reward | no |
| frozen baseline config or artifact mutation would be required | no |

## 7. Whether Phase 05B Planning Is Allowed

Later Phase `05B` planning may be drafted from this redesigned admission
candidate.

This is only planning permission. It does not authorize Phase `05B`
implementation, training, full Multi-Catfish agents, or `catfish-r1` /
`catfish-r2` / `catfish-r3` learners in this run.

## 8. Forbidden Claims

Do not claim:

1. Multi-Catfish effectiveness,
2. Catfish-r1 / r2 / r3 learner results,
3. Catfish-EE-MODQN,
4. EE-MODQN or RA-EE continuation,
5. full paper-faithful reproduction,
6. frozen baseline replacement.

## 9. PASS / FAIL

```text
Phase 05R objective-buffer admission redesign diagnostics: PASS
Phase 05B planning draft in a later run: ALLOWED
Phase 05B implementation now: FORBIDDEN
Full Multi-Catfish agents now: FORBIDDEN
```
