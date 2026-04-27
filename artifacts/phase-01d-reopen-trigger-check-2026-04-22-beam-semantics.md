# Phase 01D Reopen Trigger Check — Beam Semantics

Date: `2026-04-22`

## Purpose

This note reapplies the gate defined in
[`docs/phases/phase-01d-reproduction-reopen-gate-sdd.md`](../docs/phases/phase-01d-reproduction-reopen-gate-sdd.md)
after a read-only audit identified a concrete semantic-validity problem in
the current beam-selection surface.

It is a trigger-assessment note, not an implementation record.

## Trigger Assessment

### Trigger 1: New Source-Backed Provenance

Status: `not satisfied`

Current evidence:

1. the audit did not recover new paper-backed provenance for beam
   footprints, off-axis gain, or STK import details,
2. the assumption register still keeps the relevant beam and atmospheric
   choices disclosed as repo-fixed assumptions.

Decision:

1. this reopen is not justified by new source-backed provenance.

### Trigger 2: New Defect That Could Invalidate Phase 01B Or Phase 01C

Status: `satisfied, narrowly`

Current evidence:

1. the read-only audit note
   [`read-only-audit-2026-04-22-baseline-contract-vs-reopen.md`](./read-only-audit-2026-04-22-baseline-contract-vs-reopen.md)
   shows that the current runtime keeps a beam-level action catalog but
   collapses the decision surface toward satellite-level semantics:
   - all beams under a visible satellite are marked valid,
   - one satellite-level `snr_linear` value is copied across the whole
     7-beam block,
   - `RSS_max` therefore degenerates toward "first valid beam" rather
     than a meaningfully beam-discriminative comparator,
2. the audit's one-step rollout trace over the preserved best-eval
   checkpoint shows a concrete example:
   - valid actions `[0, 1, 2, 3, 4, 5, 6]`
   - identical `snr_linear` for all valid actions
   - `RSS_max` selects action `0`
   - MODQN's top valid scalarized-Q values remain clustered within a
     small margin,
3. this is not evidence of artifact corruption or a hidden math bug, but
   it is concrete enough to call the current comparator/beam-semantic
   interpretation into question on the frozen baseline surface.

Decision:

1. this reopen is justified only as a **bounded semantic-validity slice**
   for beam eligibility / comparator meaning,
2. it does **not** justify broad retraining or reopening unrelated
   scenario or reward branches.

### Trigger 3: New External Comparison Requirement

Status: `not satisfied`

Current evidence:

1. the current reopen request is about semantic validity of the baseline,
   not a new downstream or external-consumer comparison requirement.

Decision:

1. this reopen is not justified under Trigger 3.

## Overall Decision

`A valid reopen trigger is now satisfied, but only under Trigger 2.`

The allowed reopen is narrow:

1. evaluation-first semantic-validity audit over preserved artifacts and
   checkpoints,
2. at most one opt-in evaluation-only counterfactual eligibility slice,
3. no new `9000`-episode long run,
4. no silent rewrite of the frozen baseline interpretation,
5. no promotion to broader retraining unless this bounded slice shows a
   material change.

## Immediate Next Step

The next allowed authority surface is:

1. [`../docs/phases/phase-01e-beam-semantics-audit-reopen-sdd.md`](../docs/phases/phase-01e-beam-semantics-audit-reopen-sdd.md)

If that bounded slice fails to show that beam-semantic collapse is both
pervasive and materially relevant to comparison outcomes, the repo should
stop again immediately.
