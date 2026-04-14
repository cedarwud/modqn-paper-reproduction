# Phase 01B: Paper-Faithful Follow-On SDD

**Status:** Planned experimental SDD  
**Depends on:** Phase 01 comparison-baseline freeze  
**See also:**
- [`phase-01-python-baseline-reproduction-sdd.md`](./phase-01-python-baseline-reproduction-sdd.md)
- [`../baseline-acceptance-checklist.md`](../baseline-acceptance-checklist.md)
- [`../../artifacts/reproduction-status-2026-04-13.md`](../../artifacts/reproduction-status-2026-04-13.md)
- [`../assumptions/modqn-reproduction-assumption-register.md`](../assumptions/modqn-reproduction-assumption-register.md)

## 1. Purpose

Phase 01B opens a new explicitly experimental track for
`PAP-2024-MORL-MULTIBEAM` that aims to reduce the most important
paper-fidelity gaps still present after the comparison-baseline freeze.

This phase exists so the repo can pursue a stronger
`paper-faithful reproduction` claim without silently redefining the
already accepted `disclosed comparison baseline`.

## 2. Trigger

The comparison-baseline closeout is already complete, but the repo still
cannot honestly claim full paper-faithful reproduction.

The highest-signal reasons are:

1. the current result surfaces remain near-tied and handover-dominant
2. the `run-9000` long run still shows unresolved late-training collapse
3. the current executable baseline still includes scenario choices that
   diverge from paper-backed simulation details

The paper-backed simulation text states that:

1. users are randomly distributed within a `200 km × 90 km` rectangular
   area
2. that area is centered at `(40°N, 116°E)`
3. user movement adopts `random wandering`
4. the constellation is constructed in `STK`

The current frozen baseline instead uses:

1. `ground_point = (0°, 0°)` in the resolved config
2. `uniform-circular` user scatter with `radius = 50 km`
3. deterministic heading stride mobility
4. a disclosed synthetic single-plane orbit proxy rather than a
   recovered STK trace/import path

Phase 01B is the first place where those gaps should be addressed.

## 3. Claim Boundary

Entering Phase 01B does **not** change the status of the current
baseline bundle.

The following rules are mandatory:

1. the current resolved baseline config remains the frozen
   comparison-baseline surface
2. no Phase 01B change may silently upgrade the frozen baseline into a
   paper-faithful claim
3. all Phase 01B runs must use a new explicitly labeled experimental
   config surface
4. negative or non-improving outcomes are valid results and must be
   preserved rather than hidden

## 4. In Scope

Phase 01B includes:

1. aligning the user geography to the paper-backed center point and area
   shape
2. replacing the current user scatter and heading proxy with a disclosed
   `random wandering` implementation
3. surfacing the new geography and mobility fields in config, metadata,
   and export outputs
4. re-running the minimum evidence set needed to judge whether the
   scenario correction materially improves paper alignment
5. preserving raw machine-readable outputs together with the review note
   for every promoted Phase 01B artifact

Phase 01B does **not** include:

1. turning `ntn-sim-core` into the primary trainer
2. replacing the paper model with a 3GPP-accurate link budget
3. adding off-axis beam gain or interference not used by the paper
4. silently replacing the frozen comparison baseline
5. claiming full paper-faithful reproduction before the new evidence is
   reviewed

## 5. Primary Fidelity Target

The first Phase 01B target is `scenario fidelity`, not generic trainer
retuning.

The priority order is:

1. paper-backed user area center: `(40°N, 116°E)`
2. paper-backed user area extent: `200 km × 90 km`
3. paper-backed mobility family: `random wandering`
4. explicit export/runtime disclosure of those scenario settings
5. only after the above is landed: re-evaluate whether long-run behavior
   or method separation materially changes

This means Phase 01B should **not** begin by increasing episode count or
launching a new `9000`-episode run on the old scenario proxy.

## 6. Assumption Strategy

Phase 01B should treat the current assumption surface as follows:

1. `ASSUME-MODQN-REP-020` and `ASSUME-MODQN-REP-021` are no longer
   sufficient as the default story for the follow-on track because the
   paper does disclose a concrete area shape and mobility family
2. the resolved `ground_point` field should become paper-backed for the
   follow-on config rather than remain the equator default
3. `ASSUME-MODQN-REP-014` may remain open if an exact STK trace/import
   path still cannot be recovered, but its continued use must stay
   visible in run metadata and review notes
4. `ASSUME-MODQN-REP-001`, `002`, `003`, `009`, and `014` remain
   disclosed assumptions unless new source evidence closes them

Phase 01B does not require every remaining assumption to be closed.
It does require that the highest-impact paper-backed scenario mismatch no
longer be hidden under generic proxy defaults.

## 7. Execution Slices

### 7.1 Slice A: Scenario-Corrected Runtime Surface

Slice A lands the runtime and config changes needed for the follow-on
track:

1. add paper-backed area center fields to the experimental resolved config
2. replace circular scatter with a rectangular user-area model
3. implement a disclosed `random wandering` mobility rule
4. record the new scenario fields in run metadata and export bundles
5. add tests covering the new area and mobility semantics

Slice A must not modify the frozen comparison-baseline config in place.

### 7.2 Slice B: Scenario-Corrected Pilot Evidence

After Slice A lands, the minimum required evidence bundle is:

1. one scenario-corrected pilot training run with best-eval checkpointing
2. one scenario-corrected `Table II` artifact
3. one scenario-corrected `Fig. 3` artifact
4. one comparison note against the frozen baseline bundle

The goal of Slice B is not to prove parity immediately. The goal is to
measure whether correcting the scenario surface materially changes the
current near-tie structure.

### 7.3 Slice C: Expanded Sweeps And Long-Run Gate

`Fig. 4` to `Fig. 6` and any new long run are gated work.

They should only begin after Slice B is reviewed.

Allowed next steps after Slice B:

1. if the scenario-corrected pilot materially changes the objective
   surface, promote to full `Fig. 4` to `Fig. 6` sweeps
2. if the scenario-corrected pilot still collapses, document that
   negative result before opening more expensive retraining
3. only consider a staged long run after the scenario-corrected pilot is
   accepted as worth extending

Phase 01B should continue the current rule of avoiding a blind new
`9000`-episode launch by default.

## 8. Artifact Preservation Rule

Phase 01B artifacts are evidence, not just review prose.

For every promoted Phase 01B run:

1. the review note is required
2. the machine-readable outputs are also required
3. the run must preserve at least manifest/config/metadata/evaluation
   outputs together with any figures used in the conclusion
4. review-only directories are insufficient for a promoted
   paper-faithful follow-on claim

If the repo continues to ignore generated artifact leaves in Git, then
the promoted artifact bundle must still be exported and referenced
through a stable preserved path.

## 9. Acceptance Gates

### Gate 1: Scenario Fidelity Landed

Gate 1 is closed only when:

1. the follow-on config uses the paper-backed area center
2. the follow-on runtime uses a rectangular area model
3. the follow-on runtime uses a disclosed `random wandering` mobility
   rule
4. the new fields are visible in metadata/export surfaces
5. tests cover the new behavior

### Gate 2: Pilot Evidence Reviewed

Gate 2 is closed only when:

1. a scenario-corrected pilot run exists
2. a scenario-corrected `Table II` artifact exists
3. a scenario-corrected `Fig. 3` artifact exists
4. a review note states whether the scenario correction materially
   changes:
   - method separation
   - objective balance
   - long-run follow-up priority

### Gate 3: Follow-On Outcome Declared

Gate 3 is closed only when the repo records one of these conclusions:

1. scenario correction improved paper alignment enough to justify
   continuing toward a stronger reproduction claim
2. scenario correction did not recover paper-like behavior, so the repo
   remains best interpreted as a disclosed engineering baseline plus a
   negative-result follow-on experiment

## 10. Completion Boundary

Phase 01B is complete when:

1. the highest-impact paper-backed scenario mismatch is corrected
2. the corrected scenario has been exercised through the minimum pilot
   evidence bundle
3. the repo records whether that correction materially improves the
   reproduction claim
4. the frozen comparison baseline remains intact and separately
   interpretable

Phase 01B completion still does **not** automatically mean that the repo
has achieved full paper-faithful reproduction. It only means the next
most defensible claim boundary has been evaluated on a scenario surface
that is closer to the paper than the current frozen baseline.
