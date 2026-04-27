# Phase 01G: Atmospheric-Sign Counterfactual Audit SDD

**Status:** Executed evaluation-only follow-on SDD; closed by
[`../../artifacts/phase-01g-atmospheric-sign-status-2026-04-22.md`](../../artifacts/phase-01g-atmospheric-sign-status-2026-04-22.md)

**Date:** `2026-04-22`

## 1. Depends On

1. [`phase-01f-beam-aware-eligibility-follow-on-sdd.md`](./phase-01f-beam-aware-eligibility-follow-on-sdd.md)
2. [`../../artifacts/phase-01f-bounded-pilot-status-2026-04-22.md`](../../artifacts/phase-01f-bounded-pilot-status-2026-04-22.md)
3. [`../../docs/assumptions/modqn-reproduction-assumption-register.md`](../../docs/assumptions/modqn-reproduction-assumption-register.md)

## 2. Purpose

`Phase 01F` established that beam-aware eligibility materially changes
the bounded training surface and removes the beam-collapse issue.

That still leaves one major disclosed semantic anomaly unresolved:

1. `ASSUME-MODQN-REP-009` keeps the primary run on the
   paper-published atmospheric sign,
2. that sign yields gain for typical values,
3. reward geometry remains visibly dominated by `r1`.

The next justified step is therefore not longer training. It is an
evaluation-only counterfactual that asks:

1. how much of the current bounded behavior changes when the preserved
   checkpoint is replayed under the corrected lossy sign,
2. whether the sign change still matters after the beam-aware follow-on,
3. whether any later training surface should combine beam-aware
   eligibility with corrected atmospheric loss.

## 3. Scope

### 3.1 In Scope

1. one evaluation-only replay helper that compares:
   - paper-published sign
   - corrected lossy sign
2. reward-geometry diagnostics comparison under both signs
3. policy replay comparison for:
   - `MODQN`
   - `RSS_max`
4. one closeout note stating whether a new training surface is justified

### 3.2 Out Of Scope

1. changing the frozen baseline default
2. changing trainer-side reward calibration
3. changing beam geometry
4. starting `500` or `9000` episode training
5. modifying `ntn-sim-core`

## 4. Hypothesis

If the atmospheric sign anomaly is still a first-order driver after the
beam-aware follow-on, then replaying the same checkpoint under the
corrected lossy sign should:

1. materially change the reward-geometry diagnostics,
2. materially change held-out scalar / `r1` / `r3` outcomes,
3. possibly change action selection on the same geometry trace.

Null hypothesis:

1. the corrected lossy sign only rescales magnitudes without changing
   policy-relevant behavior in a meaningful way.

## 5. Execution Slice

### 5.1 Slice 01G1: Evaluation-Only Counterfactual

Required work:

1. add one evaluation-only entrypoint that loads an existing preserved
   checkpoint
2. replay it under both atmospheric-sign modes
3. emit:
   - `atmospheric_sign_counterfactual_summary.json`
   - `atmospheric_sign_vs_baseline.csv`
   - `reward_geometry_diagnostics_comparison.csv`
   - `review.md`

Acceptance:

1. no baseline training default changes
2. no retraining required
3. outputs clearly disclose both sign modes
4. summary classifies whether the diagnostic and policy effects are
   absent, notable, or material

## 6. Stop Rules

Stop without new training if any of these becomes true:

1. the corrected lossy sign is only a cosmetic rescaling with no
   interpretable policy effect
2. the replay difference is too small to justify a new training branch
3. the observed change is too entangled with another unresolved
   semantics issue to isolate responsibly

## 7. Promotion Rule

A new training surface is allowed only if:

1. the evaluation-only counterfactual is material,
2. the effect remains interpretable after the beam-aware follow-on,
3. a closeout note explicitly recommends the next combined semantics
   surface

If those conditions are not met, the repo must stop again without
opening a larger training branch.
