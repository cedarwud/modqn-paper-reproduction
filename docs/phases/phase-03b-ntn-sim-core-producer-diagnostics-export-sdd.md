# Phase 03B: `ntn-sim-core` Producer Diagnostics Export SDD

**Status:** Landed bounded reopen slice
**Date:** `2026-04-16`  
**Base producer schema:** `phase-03a-replay-bundle-v1`  
**Depends on:**

1. [`phase-01d-reproduction-reopen-gate-sdd.md`](./phase-01d-reproduction-reopen-gate-sdd.md)
2. [`../../artifacts/phase-01d-reopen-trigger-check-2026-04-16-producer-diagnostics.md`](../../artifacts/phase-01d-reopen-trigger-check-2026-04-16-producer-diagnostics.md)
3. [`../../artifacts/phase-01c-closeout-status-2026-04-15.md`](../../artifacts/phase-01c-closeout-status-2026-04-15.md)
4. [`phase-02-artifact-bridge-sdd.md`](./phase-02-artifact-bridge-sdd.md)
5. [`phase-03a-ntn-sim-core-bundle-replay-integration-sdd.md`](./phase-03a-ntn-sim-core-bundle-replay-integration-sdd.md)

## 1. Purpose

This SDD defines the only bounded reopen slice currently justified by the
`Phase 01D` gate:

1. add a small producer-side explainability surface for downstream
   bundle consumers,
2. do it without reopening broad paper-faithful reproduction work,
3. stop immediately if the required diagnostics cannot be exported as a
   clean additive surface.

This is not a license to resume baseline retraining or to revise the
current negative-result closeout.

## 2. Trigger

The trigger for this SDD is already recorded in:

1. [`../../artifacts/phase-01d-reopen-trigger-check-2026-04-16-producer-diagnostics.md`](../../artifacts/phase-01d-reopen-trigger-check-2026-04-16-producer-diagnostics.md)

The justified trigger is `Trigger 3: new external comparison
requirement`.

That requirement is narrow:

1. downstream consumers can already replay the exported bundle,
2. they still cannot explain policy choice structure well enough from
   the current exported fields alone,
3. one bounded export slice is therefore allowed.

## 3. Questions Required By Phase 01D

### 3.1 What new evidence or defect triggered the reopen?

Not new provenance and not a new invalidating defect.

The trigger is an explicit external comparison / explainability
requirement that the frozen bundle cannot answer today.

### 3.2 Which claim boundary does it change?

It changes only the downstream explainability boundary.

It does **not** change:

1. the current paper-faithful reproduction claim ceiling,
2. the current negative-result interpretation of Phase 01B/01C,
3. the training or evaluation semantics of the frozen comparison
   baseline.

### 3.3 Why is this slice bounded and auditable?

Because it is limited to:

1. exporter-only diagnostics over the already selected checkpoint and
   replay path,
2. additive bundle fields only,
3. one narrow validation/evidence set,
4. one explicit stop rule if additive export is not stable.

### 3.4 What result would justify continuing?

This slice justifies paired downstream work only if all of these are
true:

1. producer-side diagnostics can be exported without changing training
   semantics,
2. the diagnostics are clearly tied to the same replay truth already
   exported,
3. the additive fields are stable enough for an external consumer to use
   them without guessing.

### 3.5 What result would force an immediate stop?

Stop immediately if any of these becomes true:

1. the diagnostics require rewriting the training main path,
2. the diagnostics require silently changing the meaning of frozen Phase
   03A required fields,
3. the only workable export would be ambiguous or consumer-invented
   rather than producer-owned.

## 4. Scope

### 4.1 In Scope

1. one exporter-local decision-diagnostics helper over the currently
   selected replay checkpoint,
2. additive decision-time diagnostics in replay timeline rows,
3. manifest-level disclosure that the bundle contains optional producer
   diagnostics,
4. one bounded validation surface for the new diagnostics,
5. one reviewed status note that says whether downstream promotion is
   justified.

### 4.2 Explicitly Out Of Scope

1. new long training runs,
2. new sweep families,
3. changing reward equations,
4. changing checkpoint-selection rules,
5. changing the meaning of `phase-03a-replay-bundle-v1` required fields,
6. broad paper-faithful reproduction reopening,
7. direct consumer-side UI work inside this repo.

## 5. Working Hypothesis

The current exporter already has access to enough decision-time state to
compute a small explainability surface without reopening training:

1. encoded state at decision time,
2. decision mask,
3. active objective weights,
4. the currently selected checkpoint networks.

So the most defensible first step is:

1. compute diagnostics at export/replay time from the same selected
   checkpoint already used for bundle replay,
2. export only the smallest stable subset needed for downstream
   explainability,
3. avoid turning exporter diagnostics into a new training or evaluation
   rule.

## 6. Slice 03B1: Exporter-Only Diagnostics Helper

The current `select_actions()` path in `algorithms/modqn.py` should not
be repurposed into a broad new public contract.

Required rule:

1. if diagnostics are added, they should come from an exporter-only
   helper or an explicitly separate `select_actions_with_diagnostics()`
   style surface,
2. the existing training/evaluation action-selection semantics must stay
   unchanged.

The helper may expose, per user and per decision step:

1. selected action scalarized score,
2. runner-up scalarized score,
3. scalarized margin to runner-up,
4. limited top-candidate list with stable beam identity,
5. per-candidate objective-Q triplet when available from the same
   selected checkpoint.

The helper must always respect the decision mask used by the replay
timeline.

## 7. Slice 03B2: Additive Bundle Surface

The first allowed export shape is additive only.

Preferred timeline surface:

1. add an optional `policyDiagnostics` object to each
   `timeline/step-trace.jsonl` row,
2. preserve all existing Phase 03A required fields unchanged,
3. keep the new object fully ignorable by older consumers.

Preferred `policyDiagnostics` contents:

1. `diagnosticsVersion`
2. `objectiveWeights`
3. `selectedScalarizedQ`
4. `runnerUpScalarizedQ`
5. `scalarizedMarginToRunnerUp`
6. `availableActionCount`
7. `topCandidates`

The landed export keeps `objectiveWeights` as a named
object keyed by
`r1Throughput`, `r2Handover`, and `r3LoadBalance` so downstream
consumers do not need to guess positional ordering.

Each `topCandidates` entry should keep stable exported identity:

1. `beamId`
2. `beamIndex`
3. `satId`
4. `satIndex`
5. `localBeamIndex`
6. `validUnderDecisionMask`
7. `objectiveQ`
8. `scalarizedQ`

The landed export keeps `objectiveQ` as the same named
`r1Throughput` / `r2Handover` / `r3LoadBalance` object rather than as a
positional array.

Manifest disclosure should also advertise that the bundle contains
optional policy diagnostics.

The landed manifest disclosure is
`manifest.optionalPolicyDiagnostics` with:

1. `present`
2. `timelineField`
3. `diagnosticsVersion`
4. `requiredByBundleSchema`
5. `producerOwned`
6. `selectedActionSource`
7. `topCandidateLimit`
8. `rowsWithDiagnostics`
9. `rowsWithoutDiagnostics`
10. `note`

If that cannot be done without ambiguity, this slice should stop and
hand off a schema-version decision rather than silently mutating the
frozen contract.

## 8. Slice 03B3: Validation And Evidence

This slice is complete only if it lands bounded proof, not just new
fields.

Minimum required outputs:

1. one exporter test proving diagnostics are emitted only when available
   and remain consistent with the selected action,
2. one exporter test proving decision-mask-respecting candidate ordering
   and margin behavior,
3. one reviewed small artifact or fixture surface carrying the optional
   diagnostics,
4. one short status/review note saying whether downstream consumer
   promotion is justified.

The landed validation surface also extends
`validate_replay_bundle()` so that optional `policyDiagnostics` rows are
checked for:

1. selected-action alignment with `selectedServing`,
2. descending candidate ordering with stable beam-index tie breaks,
3. decision-mask-respecting candidates only,
4. manifest/timeline coverage consistency.

## 9. Promotion Rule

Promotion to paired downstream consumer work is allowed only if all of
these are true:

1. diagnostics remain additive,
2. the current replay-truth path remains the authoritative source,
3. no training/evaluation rule changed,
4. exported diagnostics are explicit enough that a consumer does not
   need to invent missing semantics.

If any of these fail, do not promote.

Current interpretation:

1. the implementation and tests described by this SDD are now landed in
   repo history,
2. the slice is the current producer-side authority for optional policy
   diagnostics export,
3. downstream consumer promotion is now justified from the producer side,
4. downstream consumer adoption still remains separate follow-on work.

## 10. Negative-Result Rule

Negative results are valid here too.

If the repo cannot export a stable, additive, producer-owned
diagnostics surface without changing the frozen meaning of the current
bundle or the trainer path, it should record:

1. the explicit external comparison requirement was evaluated,
2. the bounded export slice was not stable enough to promote,
3. the repo remains at the current frozen closeout baseline plus
   negative-result follow-ons.

## 11. Deliverables

This execution SDD is complete when the repo contains:

1. one implemented exporter-local diagnostics helper or equivalent
   additive exporter surface,
2. one additive bundle export surface with explicit disclosure,
3. one bounded validation surface,
4. one reviewed status note that says either `promote` or `stop`.

## 12. Landed Outcome

The landed slice now contains:

1. `MODQNTrainer.select_actions_with_diagnostics()` as the separate
   greedy exporter helper,
2. optional `policyDiagnostics` rows in
   `timeline/step-trace.jsonl`,
3. `manifest.optionalPolicyDiagnostics` disclosure plus replay-bundle
   validator coverage,
4. refreshed `tests/fixtures/sample-bundle-v1/` evidence,
5. reviewed execution status note at
   `artifacts/phase-03b-producer-diagnostics-export-status-2026-04-16.md`.
