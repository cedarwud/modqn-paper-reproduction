# Phase 01D Reopen Trigger Check — Producer Diagnostics

Date: `2026-04-16`

## Purpose

This note reapplies the gate defined in
[`docs/phases/phase-01d-reproduction-reopen-gate-sdd.md`](../docs/phases/phase-01d-reproduction-reopen-gate-sdd.md)
after a new external comparison / explainability requirement was raised
for downstream bundle consumers.

It is a trigger-assessment note, not an implementation record.

## Trigger Assessment

### Trigger 1: New Source-Backed Provenance

Status: `not satisfied`

Current evidence:

1. the latest closeout authority,
   [`phase-01c-closeout-status-2026-04-15.md`](./phase-01c-closeout-status-2026-04-15.md),
   still records no newly recovered paper/runtime provenance,
2. the assumption register still keeps the highest-impact provenance
   gaps visible.

Decision:

1. this reopen is not justified by new paper-backed provenance.

### Trigger 2: New Defect That Could Invalidate Phase 01B Or Phase 01C

Status: `not satisfied`

Current evidence:

1. no new repo-local defect note overturns the Phase 01B or Phase 01C
   negative-result interpretation,
2. the known limits remain disclosed rather than newly invalidating.

Decision:

1. this reopen is not justified by a new invalidating bug.

### Trigger 3: New External Comparison Requirement

Status: `satisfied`

Current evidence:

1. the frozen Phase 03A replay bundle is already good enough for
   replay, dashboard, and replay-truth proof in downstream consumers,
   but it does not yet explain *why* the selected serving target won,
2. a new explicit downstream requirement now exists for producer-owned
   policy diagnostics / explainability rather than for more retraining
   or paper-faithful claim expansion,
3. the current frozen bundle cannot answer that requirement without one
   bounded new export slice.

Decision:

1. a single bounded reopen slice is justified for additive producer-side
   diagnostics export only.

## Overall Decision

`A valid reopen trigger is now satisfied, but only under Trigger 3.`

The allowed reopen is narrow:

1. additive exporter diagnostics for downstream explainability,
2. no broad reproduction restart,
3. no unbounded retraining or sweep expansion,
4. no silent re-interpretation of the current closeout.

## Immediate Next Step

The next allowed authority surface is:

1. [`../docs/phases/phase-03b-ntn-sim-core-producer-diagnostics-export-sdd.md`](../docs/phases/phase-03b-ntn-sim-core-producer-diagnostics-export-sdd.md)

If that bounded slice fails to define stable additive diagnostics without
changing training semantics or frozen bundle meaning, the repo should
stop again immediately.
