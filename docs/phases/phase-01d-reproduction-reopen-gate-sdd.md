# Phase 01D: Reproduction Reopen Gate SDD

**Status:** Standby reopen gate  
**Depends on:**
- [`phase-01c-comparator-protocol-experiment-sdd.md`](./phase-01c-comparator-protocol-experiment-sdd.md)
- [`../../artifacts/phase-01c-closeout-status-2026-04-15.md`](../../artifacts/phase-01c-closeout-status-2026-04-15.md)
- [`../../artifacts/public-summary-2026-04-15-phase-01c-closeout.md`](../../artifacts/public-summary-2026-04-15-phase-01c-closeout.md)
- [`../assumptions/modqn-reproduction-assumption-register.md`](../assumptions/modqn-reproduction-assumption-register.md)

## 1. Purpose

Phase 01D does **not** reopen reproduction work by itself.

Its purpose is only to define the minimum gate that must be satisfied
before any new paper-faithful reproduction development is allowed after
the Phase 01B and Phase 01C negative-result closeouts.

## 2. Current Default State

Until this gate is explicitly satisfied, the repo remains:

1. a frozen disclosed comparison baseline
2. plus a completed negative-result scenario-correction follow-on
3. plus a completed negative-result comparator-protocol follow-on

The default action remains `stop`, not `continue tuning`.

## 3. Valid Reopen Triggers

Reproduction work may reopen only if at least one of these is true:

1. new source-backed provenance becomes available for a currently
   high-impact open or accepted assumption
   - examples:
     - STK trace/import path
     - exact shell layout
     - beam geometry/footprint source
     - paper-backed protocol detail that replaces a repo-fixed
       assumption
2. a concrete implementation or artifact-preservation defect is found
   that could reasonably invalidate the current Phase 01B or Phase 01C
   conclusions
3. the user explicitly requests a new external comparison requirement
   that the frozen baseline cannot answer without a bounded new slice

## 4. Non-Triggers

These do **not** justify reopening:

1. wanting to try more episodes
2. wanting to rerun the same sweeps with more points
3. near-tie results by themselves
4. unbounded hyperparameter tuning
5. aesthetic dissatisfaction with paper-figure similarity

## 5. Allowed Scope After Reopen

If a valid trigger exists, the first reopen step must still stay narrow:

1. write one new execution SDD tied to that specific trigger
2. define exactly one bounded implementation or evaluation slice
3. preserve the frozen comparison baseline as-is
4. define explicit machine-readable outputs, review notes, and stop
   criteria before implementation starts

No reopen may silently expand back into broad Phase 01B/01C-style
exploration.

## 6. Mandatory Stop Rule

If the first bounded reopen slice does not materially change the current
interpretation, the repo must close again immediately.

`Materially change` means at least one of:

1. a previously open provenance gap is actually closed by source-backed
   evidence
2. winner identity or tie structure changes on a meaningful surface for
   reasons not confined to the already-known `r2` / handover-only effect
3. the repo can honestly upgrade its claim boundary beyond the current
   disclosed-baseline interpretation

If none of the above happens, reopen work ends as another disclosed
negative result.

## 7. First Questions To Answer Before Any Reopen Execution

Any future reopen SDD must answer these first:

1. what new evidence or defect triggered the reopen
2. which existing assumption or claim boundary it changes
3. why the new slice is bounded and auditable
4. what exact result would justify continuing past that slice
5. what exact result would force an immediate stop

## 8. Practical Interpretation

As of `2026-04-15`, this file should be read as a guardrail, not as an
active work order.

If no new trigger is present, the correct action is to keep the repo
frozen and use it as the disclosed engineering baseline already
established by the current closeout artifacts.
