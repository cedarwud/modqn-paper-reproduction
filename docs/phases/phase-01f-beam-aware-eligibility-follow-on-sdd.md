# Phase 01F: Beam-Aware Eligibility Follow-On SDD

**Status:** Proposed bounded follow-on SDD  
**Date:** `2026-04-22`

## 1. Depends On

1. [`phase-01e-beam-semantics-audit-reopen-sdd.md`](./phase-01e-beam-semantics-audit-reopen-sdd.md)
2. [`../../artifacts/phase-01e-beam-semantics-status-2026-04-22.md`](../../artifacts/phase-01e-beam-semantics-status-2026-04-22.md)
3. [`../../artifacts/codex-pilot-200ep-2026-04-22/beam-semantics-audit/review.md`](../../artifacts/codex-pilot-200ep-2026-04-22/beam-semantics-audit/review.md)
4. [`../../artifacts/codex-pilot-200ep-2026-04-22/beam-counterfactual-audit/review.md`](../../artifacts/codex-pilot-200ep-2026-04-22/beam-counterfactual-audit/review.md)

## 2. Purpose

`Phase 01E` established two things:

1. the current baseline collapses beam-level eligibility and comparator
   meaning pervasively,
2. one minimal beam-aware counterfactual materially changes downstream
   reward through beam-level load allocation, even when same-state
   satellite choice does not change.

This SDD defines the only justified next step:

1. add one opt-in beam-aware eligibility training surface,
2. keep the frozen baseline untouched and default,
3. validate the new surface on bounded runs only before any larger
   training is considered.

This is **not** authorization for an immediate `500`-episode or
`9000`-episode run.

## 3. New Authority Boundary

The new surface must be framed as a follow-on experiment, not as a silent
rewrite of the frozen baseline.

Required boundary:

1. baseline config and baseline semantics remain unchanged
2. beam-aware eligibility is introduced only through a new explicitly
   labeled follow-on config / runtime branch
3. downstream export and replay must disclose which eligibility surface
   produced the artifact

## 4. Hypothesis

If the trainer sees beam-aware eligibility during training rather than
only at evaluation time, then:

1. beam-level load allocation can become part of the learned policy
   surface,
2. comparator meaning should become less degenerate,
3. bounded training may yield a materially different reward / handover /
   load-balance tradeoff than the frozen baseline.

Null hypothesis:

1. even after training with beam-aware eligibility, the bounded runs
   remain near-equivalent to the frozen baseline or improve only in a
   way that is obviously not scientifically meaningful.

## 5. Scope

### 5.1 In Scope

1. one opt-in beam-aware eligibility mode based on the already tested
   `nearest-beam-per-visible-satellite` proxy
2. one new resolved-run follow-on config that enables that mode
3. bounded smoke and pilot runs only
4. export and metadata disclosure for the new mode
5. one closeout note deciding whether larger training is justified

### 5.2 Out Of Scope

1. changing atmospheric-sign semantics
2. changing channel gain / adding off-axis beam gain
3. reward-calibration rewrites
4. modifying `ntn-sim-core`
5. immediate `500` / `9000` training

## 6. Required Runtime Rule

The first and only beam-aware rule in this phase is:

1. for each visible satellite, keep only the nearest beam eligible for a
   given user
2. use the existing `beam_pattern.nearest_beam(...)` geometry
3. apply the rule consistently in:
   - training action masks
   - evaluation action masks
   - exported metadata / replay disclosure

The rule must be deterministic and must not become the default.

## 7. Execution Slices

### 7.1 Slice 01F1: Opt-In Runtime Surface

Required work:

1. add one explicit runtime/config switch for beam-aware eligibility
2. keep the frozen baseline default path byte-stable
3. write the active eligibility mode into metadata and exported
   artifacts

Acceptance:

1. baseline tests remain green without config changes
2. the new mode is reachable only through the new follow-on config
3. replay/export surfaces disclose the new mode clearly

### 7.2 Slice 01F2: Smoke Validation

Required run:

1. `1`-episode smoke under the new follow-on config

Acceptance:

1. training completes
2. checkpoint, metadata, training log, and export bundle all exist
3. metadata explicitly records the beam-aware eligibility surface

### 7.3 Slice 01F3: Bounded Pilot

Required runs:

1. `20`-episode bounded pilot
2. `200`-episode bounded pilot

Required comparisons:

1. compare against the preserved frozen baseline pilot surface
2. compare `MODQN` and `RSS_max`
3. inspect:
   - scalar reward
   - `r1`, `r2`, `r3`
   - handover count
   - comparator/tie behavior

Acceptance:

1. the new mode shows a clear, disclosed difference on bounded runs
2. the difference is not confined to a reporting artifact or exporter bug
3. the result is stable enough to justify or reject larger training

## 8. Stop Rules

Stop without larger training if any of these becomes true:

1. the new mode breaks baseline tests or exporter truth surfaces
2. bounded pilot runs fail to differ materially from the frozen baseline
3. the observed change collapses into noise, instability, or one-off
   seed behavior
4. the only improvement comes from a bookkeeping artifact rather than a
   real policy change

## 9. Promotion Rule

Larger training is allowed only if all of these are true:

1. `01F1` lands without destabilizing the baseline
2. `01F2` smoke passes
3. `01F3` bounded pilots show material, interpretable differences
4. the closeout note explicitly recommends the larger run

If those conditions are not met, the repo must stop again without
opening a new long-run branch.

## 10. Deliverables

Minimum deliverables for this phase:

1. one new follow-on resolved config
2. one runtime implementation slice for beam-aware eligibility
3. bounded training artifacts for smoke / `20` / `200`
4. one closeout status note stating whether larger training is justified
