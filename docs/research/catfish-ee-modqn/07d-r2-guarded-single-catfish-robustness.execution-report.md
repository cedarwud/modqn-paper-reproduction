# Phase 07D Execution Report: R2-Guarded Single-Catfish Robustness

**Date:** `2026-04-30`
**Status:** `FAIL for acceptance / recovery promotion`
**Scope:** bounded implementation and pilot for r2 / handover-guarded
single-Catfish robustness. This report records the completed execution result
and does not authorize Multi-Catfish reopening, Phase `06`, EE / RA-EE /
Catfish-EE, long training, reward changes, or frozen baseline mutation.

## 1. Current State

Phase `07B` previously provided bounded positive evidence for the
single-Catfish intervention path over the original MODQN reward surface:

```text
r1 = throughput
r2 = handover penalty
r3 = load balance
```

Phase `07C` recorded the caveat that `r2` / handovers worsened and that
asymmetric gamma was not supported as an active mechanism. Phase `07D` tested
whether r2 / handover guardrails could preserve the Phase `07B` utility while
preventing handover degradation.

## 2. Implementation Summary

Implemented surfaces:

1. Phase `07D` opt-in training kind and config validation,
2. r2 admission guard rejecting r2-negative Catfish replay admission when
   enabled,
3. r2 intervention batch guard that skips Catfish injection instead of falling
   back to unguarded injection when the injected batch violates the r2-negative
   share guard,
4. guard diagnostics for pass / skip counts, skip reasons, r2 batch
   distributions, source counts, lineage, intervention windows, starvation,
   NaN / action / Q-loss stability,
5. Phase `07D` summary validator with predeclared non-inferiority margins.

Created configs:

1. `configs/catfish-modqn-phase-07d-modqn-control.resolved.yaml`,
2. `configs/catfish-modqn-phase-07d-r2-guarded-primary-shaping-off.resolved.yaml`,
3. `configs/catfish-modqn-phase-07d-no-intervention.resolved.yaml`,
4. `configs/catfish-modqn-phase-07d-random-equal-budget-injection.resolved.yaml`,
5. `configs/catfish-modqn-phase-07d-replay-only-single-learner.resolved.yaml`,
6. `configs/catfish-modqn-phase-07d-no-asymmetric-gamma.resolved.yaml`,
7. `configs/catfish-modqn-phase-07d-admission-only-guard.resolved.yaml`,
8. `configs/catfish-modqn-phase-07d-intervention-only-guard.resolved.yaml`,
9. `configs/catfish-modqn-phase-07d-full-admission-intervention-guard.resolved.yaml`,
10. `configs/catfish-modqn-phase-07d-strict-no-handover-sample-guard.resolved.yaml`.

Artifacts:

1. `27` required bounded run directories under
   `artifacts/catfish-modqn-phase-07d-*-seed{01,02,03}/`,
2. summary:
   `artifacts/catfish-modqn-phase-07d-r2-guarded-robustness-summary/phase07d_r2_guarded_robustness_summary.json`.

## 3. Bounded Protocol

Completed:

1. `9` roles x `3` matched seed triplets = `27` required runs,
2. `20` episodes per run,
3. evaluation seeds `[100, 200, 300, 400, 500]`,
4. evaluation cadence `5`,
5. final and best-eval checkpoints.

Optional strict diagnostic config was created but not run.

Tests reported:

1. compileall over `src/modqn_paper_reproduction` and the Phase `07D` test,
2. focused pytest over Phase `04B`, `05A`, `05B`, `07B`, and `07D` tests,
3. result: `45 passed`.

## 4. Diagnostics Summary

Primary final mean:

| Metric | Value |
|---|---:|
| scalar | `608.988400` |
| `r1` | `1228.125646` |
| `r2` | `-4.293333` |
| `r3` | `-18.932115` |
| handovers | `858.666667` |

Guard diagnostics:

1. actual injected ratio: `0.296875`,
2. admission pass count: `565`,
3. admission skip count: `3394`,
4. skip reason includes `r2-admission-negative-sample`,
5. injected Catfish r2-negative share mean: `0.0`,
6. matched main r2-negative share mean: `0.8617`,
7. guard violations: `0`,
8. hidden unguarded fallback absent,
9. no NaN,
10. no action collapse.

## 5. Comparator Result

Primary scalar deltas:

| Comparator | Scalar delta |
|---|---:|
| matched MODQN | `+0.629302` |
| no-intervention | `+0.693769` |
| random / equal-budget | `+0.304625` |
| replay-only | `+0.382683` |

Primary r2 / handover deltas:

| Comparator | `r2_delta` | `handover_delta` |
|---|---:|---:|
| matched MODQN | `-0.051667` | `+10.333333` |
| random / equal-budget | `+0.046667` | `-9.333333` |

Predeclared non-inferiority margins:

```text
r2_delta >= -0.02
handover_delta <= +5
```

The primary branch failed both predeclared non-inferiority margins versus
matched MODQN. It passed versus random / equal-budget on the reported deltas,
but the acceptance rule required non-inferiority versus both matched MODQN and
random / equal-budget.

## 6. Acceptance Result

Summary acceptance:

```text
Phase 07D acceptance: FAIL
Required roles complete: PASS
Scalar and component support: PASS
Not scalar-only: PASS
r2 non-inferiority: FAIL
handover non-inferiority: FAIL
starvation stop trigger absent: FAIL
```

The result is not a scalar-only failure: scalar and component support remain
visible. The failure is specifically that the r2 / handover guard did not
preserve the Phase `07B` utility while satisfying the predeclared handover
guardrails.

## 7. Protected Baseline Status

Reported protected surfaces:

1. `configs/modqn-paper-baseline.yaml`: unchanged,
2. `configs/modqn-paper-baseline.resolved-template.yaml`: unchanged,
3. frozen baseline artifact namespaces: not used or modified.

No EE / RA-EE / Catfish-EE / Phase `06` claim was made. No Multi-Catfish
reopening was performed. No asymmetric-gamma mechanism claim was made.

## 8. Disposition

Phase `07D` is a bounded negative result for r2-guarded single-Catfish
robustness.

It does not invalidate Phase `07B`'s narrower bounded intervention utility
evidence, but it blocks promotion to broader Catfish-MODQN effectiveness because
the handover / `r2` caveat remained unresolved under predeclared guardrails.

Default next action should be paper synthesis / claim-boundary writing, not
more Catfish tuning. Any future Catfish R&D would need a new design gate that
explains why it is not another guard-tuning attempt, does not require reward /
state / action / backbone changes, and can satisfy r2 / handover guardrails
without scalar-only success.

## 9. PASS / FAIL

```text
Phase 07D bounded implementation / runs / diagnostics: PASS
Phase 07D acceptance / recovery promotion: FAIL
Broader Catfish-MODQN effectiveness: NOT PROMOTED
Multi-Catfish reopening: BLOCKED
Phase 06 / Catfish-EE-MODQN: BLOCKED
Recommended next state: paper synthesis / claim-boundary writing
```
