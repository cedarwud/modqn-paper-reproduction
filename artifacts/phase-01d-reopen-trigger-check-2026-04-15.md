# Phase 01D Reopen Trigger Check

Date: `2026-04-15`

## Purpose

This note applies the gate defined in
[`docs/phases/phase-01d-reproduction-reopen-gate-sdd.md`](../docs/phases/phase-01d-reproduction-reopen-gate-sdd.md)
to the current repo state after the Phase 01C closeout.

It is a decision check only. It does **not** reopen work.

## Trigger Assessment

### Trigger 1: New Source-Backed Provenance

Status: `not satisfied`

Current evidence:

1. [`artifacts/phase-01c-closeout-status-2026-04-15.md`](./phase-01c-closeout-status-2026-04-15.md)
   closes the comparator-protocol branch without recording any new
   source-backed paper/runtime provenance.
2. [`docs/assumptions/modqn-reproduction-assumption-register.md`](../docs/assumptions/modqn-reproduction-assumption-register.md)
   still keeps the high-impact provenance gaps visible, including the
   open STK trace/import path and accepted proxy assumptions for shell
   layout and beam geometry.

Decision:

1. No newly recovered paper/runtime source has been added that would
   justify reopening the reproduction claim.

### Trigger 2: New Defect That Could Invalidate Phase 01B Or Phase 01C

Status: `not satisfied`

Current evidence:

1. [`artifacts/phase-01c-closeout-status-2026-04-15.md`](./phase-01c-closeout-status-2026-04-15.md)
   records the bounded protocol probe and still ends as `stop`.
2. [`artifacts/phase-01c-slice-a-protocol-inventory-2026-04-14/recommendation.md`](./phase-01c-slice-a-protocol-inventory-2026-04-14/recommendation.md)
   already disclosed the artifact-preservation limitation before the
   bounded probe was run.
3. No newer repo-local status note records a bug, preservation defect,
   or evaluation error large enough to reasonably overturn the current
   interpretation.

Decision:

1. Known limitations remain disclosed, but no new invalidating defect is
   currently on record.

### Trigger 3: New External Comparison Requirement

Status: `not satisfied`

Current evidence:

1. The latest repo-local public summary,
   [`artifacts/public-summary-2026-04-15-phase-01c-closeout.md`](./public-summary-2026-04-15-phase-01c-closeout.md),
   still frames the repo as a disclosed engineering baseline rather than
   as an active reproduction branch.
2. No new repo-local authority file introduces an external benchmark or
   comparison obligation that the frozen baseline cannot already answer
   at its current claim level.

Decision:

1. There is no file-backed external comparison trigger on record today.

## Overall Decision

`No valid reopen trigger is currently satisfied.`

The correct default action remains:

1. keep the repo frozen as the disclosed engineering baseline
2. treat Phase 01B and Phase 01C as closed negative-result follow-ons
3. avoid renewed reproduction implementation unless a later file-backed
   trigger changes this assessment

## What Would Change This Decision

Any future reopen should begin only after one of these is recorded in
repo-local authority or status notes:

1. new source-backed provenance that closes a high-impact assumption gap
2. a concrete defect report that can plausibly overturn the current
   negative-result interpretation
3. a new external comparison requirement that demands one bounded reopen
   slice
