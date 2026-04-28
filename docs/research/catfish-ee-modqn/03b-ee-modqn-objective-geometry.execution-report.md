# Phase 03B Execution Report: EE-MODQN Objective Geometry

**Date:** `2026-04-28`  
**Status:** `NEEDS MORE EVIDENCE`  
**Scope:** bounded paired reward/objective-geometry pilot only. No Catfish,
multi-Catfish, final Catfish-EE-MODQN, long 9000-episode training, action-space
change, or full paper-faithful reproduction claim was performed.

## Design

Phase 03A found that the Phase 02B denominator is variable in the environment,
but learned Phase 03 policies collapse every evaluated step to one active beam.
Phase 03B therefore tested whether an opt-in reward/objective-geometry change
could make learned policies exercise denominator variability.

Paired configs:

1. `configs/ee-modqn-phase-03b-control-objective-geometry.resolved.yaml`
2. `configs/ee-modqn-phase-03b-ee-objective-geometry.resolved.yaml`

Shared settings:

1. Phase 02B `active-load-concave` HOBS-compatible power surface.
2. `60` training episodes.
3. Seeds: train `42`, environment `1337`, mobility `7`.
4. Evaluation seed set: `[100, 200, 300, 400, 500]`.
5. Same replay capacity, target sync cadence, checkpoint rule, action space,
   state encoding, and evaluation cadence.

Phase 03B opt-in reward geometry:

1. Control r1: `throughput`.
2. EE r1: `per-user-beam-ee-credit`, defined as `R_i(t) / P_b(t)`.
3. EE r1 disclosure: this is a Phase 03B credit-assignment sensitivity, not
   system EE and not the Phase 03 allocated-power credit.
4. Reward normalization: `divide-by-fixed-scales`.
   - control scales: `[20.0, 1.0, 1.0]`
   - EE scales: `[10.0, 1.0, 1.0]`
5. Load-balance calibration: shared objective weights `[0.2, 0.2, 0.6]`.

## Artifacts

Training:

1. `artifacts/ee-modqn-phase-03b-control-objective-geometry-pilot/`
2. `artifacts/ee-modqn-phase-03b-ee-objective-geometry-pilot/`

Denominator audits:

1. `artifacts/ee-modqn-phase-03b-control-objective-geometry-pilot/denominator-audit/`
2. `artifacts/ee-modqn-phase-03b-ee-objective-geometry-pilot/denominator-audit/`

Paired comparison:

1. `artifacts/ee-modqn-phase-03b-ee-objective-geometry-pilot/paired-comparison-vs-control/`

Policy-denominator diagnostics:

1. `artifacts/ee-modqn-phase-03b-policy-diagnostics/`

## Best-Eval Metrics

Primary comparison uses best-eval checkpoints on the same evaluation seeds.

| Metric | MODQN-control | EE-MODQN | Result |
|---|---:|---:|---|
| `EE_system` aggregate bps/W | `873.0934588432312` | `873.0934588432312` | tie |
| mean raw throughput bps/user-step | `17.461869176864624` | `17.461869176864624` | tie |
| low p05 throughput bps | `13.576451826095582` | `13.576451826095582` | tie |
| served ratio | `1.0` | `1.0` | tie |
| active beam count mean | `1.0` | `1.0` | tie |
| total active beam power mean W | `2.0` | `2.0` | tie |
| handover count | `86.2` | `86.2` | tie |
| `r2` mean | `-0.43100000000000005` | `-0.43100000000000005` | tie |
| `r3` mean | `-174.61869176864755` | `-174.61869176864755` | tie |
| load-balance gap bps | `1746.1869176864625` | `1746.1869176864625` | tie |
| scalar reward | `-69.93367670745928` | `-87.39554588432391` | control higher; supporting diagnostic only |

## Denominator Variability Evidence

Config-level audits still pass:

1. denominator varies in the runtime surface: `true`
2. active beam counts sampled: `[1, 7]`
3. total active beam power sampled: `2.0 W` to `10.975454935197652 W`
4. classification: `hobs-compatible-active-load-concave-power-surface`

Learned best-eval policies still fail the denominator gate:

1. EE active beam count distribution: only `[1.0]` across `50` evaluated steps.
2. EE total active beam power distribution: only `[2.0]` across `50` evaluated
   steps.
3. `denominator_varies_in_eval = false`.
4. all evaluated EE steps one active beam: `true`.

The policy-denominator diagnostic reports:

```text
denominator_variability_exists_in_environment = true
learned_policies_exercise_denominator_variability = false
learned_policies_all_users_one_beam = true
action_mask_forces_single_beam = false
q_value_near_tie_is_primary_cause = false
r3_load_balance_insufficient_to_prevent_collapse = true
phase_02b_power_formula_is_globally_fixed = false
```

## Rescaling And Reward-Hacking Checks

Throughput-vs-EE correlation under the Phase 03B EE rescore remained effectively
identical:

```text
Pearson  = 0.9999999999999998
Spearman = 0.9999999999999999
same-policy rescore changes ranking = false
high_rescaling_risk = true
```

Same-policy rescoring did not change the method ranking:

| Rescore | MODQN-control | EE-MODQN | Winner |
|---|---:|---:|---|
| throughput-r1 scalar | `-69.93367670745928` | `-69.93367670745928` | tie |
| EE-r1 scalar | `-87.39554588432391` | `-87.39554588432391` | tie |

Reward-hacking flags were not triggered:

1. no EE increase with throughput collapse,
2. no low-percentile throughput collapse,
3. no served-ratio collapse,
4. no power drop caused by service drop,
5. no per-user EE rise without `EE_system`.

This absence is not a promotion signal because `EE_system` did not improve and
the learned denominator stayed fixed.

## Decision

`NEEDS MORE EVIDENCE`.

Phase 03B did not satisfy the promotion gate:

1. `denominator_varies_in_eval = false`,
2. learned policies still collapsed every evaluated step to one active beam,
3. `EE_system` tied control,
4. throughput and served ratio did not collapse, but also did not show an
   interpretable EE tradeoff,
5. handover / `r2` tied,
6. throughput-vs-EE ranking remained effectively identical,
7. the result is not explained by scalar reward improvement; scalar reward did
   not support EE-MODQN.

## Validation

Focused regression tests:

```text
.venv/bin/python -m pytest \
  tests/test_phase03_ee_modqn.py \
  tests/test_phase03a_diagnostics.py \
  tests/test_ee_denominator_audit.py \
  tests/test_training_hardening.py -q
```

Result:

```text
29 passed
```

## Phase 04/05/06 Implications

1. Phase 04 single-Catfish feasibility may proceed only as a separate original
   MODQN reward-mechanics branch.
2. Phase 05 remains blocked for multi-Catfish claims.
3. Phase 06 remains blocked for Catfish-EE-MODQN final-method claims.
4. Any future EE route needs a new design gate before larger training, likely
   involving a denominator-sensitive action/power design or a stronger
   credit-assignment review. More episodes alone are not the next gate.

## Remaining Blockers

1. Learned policies still do not exercise denominator variability.
2. Reward normalization plus r3 weight calibration did not prevent one-beam
   collapse in greedy evaluation.
3. Per-user EE remains unable to distinguish energy behavior from throughput
   scaling under learned collapse.
4. Phase 02B power surface remains a disclosed synthesized proxy, not a
   paper-backed HOBS optimizer.
5. Catfish, multi-Catfish, final Catfish-EE-MODQN, and full paper-faithful
   reproduction claims remain forbidden.
