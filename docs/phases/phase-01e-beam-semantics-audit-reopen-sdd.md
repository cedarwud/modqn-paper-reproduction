# Phase 01E: Beam Semantics Audit Reopen SDD

**Status:** Executed bounded reopen SDD; closed by
[`../../artifacts/phase-01e-beam-semantics-status-2026-04-22.md`](../../artifacts/phase-01e-beam-semantics-status-2026-04-22.md)

**Date:** `2026-04-22`  
**Depends on:**

1. [`phase-01d-reproduction-reopen-gate-sdd.md`](./phase-01d-reproduction-reopen-gate-sdd.md)
2. [`../../artifacts/phase-01d-reopen-trigger-check-2026-04-22-beam-semantics.md`](../../artifacts/phase-01d-reopen-trigger-check-2026-04-22-beam-semantics.md)
3. [`../../artifacts/read-only-audit-2026-04-22-baseline-contract-vs-reopen.md`](../../artifacts/read-only-audit-2026-04-22-baseline-contract-vs-reopen.md)
4. [`phase-01c-comparator-protocol-experiment-sdd.md`](./phase-01c-comparator-protocol-experiment-sdd.md)
5. [`../../artifacts/phase-01c-closeout-status-2026-04-15.md`](../../artifacts/phase-01c-closeout-status-2026-04-15.md)

## 1. Purpose

This SDD defines the only bounded reopen slice currently justified by the
`Phase 01D` gate for the newly identified beam-semantic issue:

1. quantify how strongly the current baseline collapses beam-level
   decisions toward satellite-level semantics,
2. test one minimal evaluation-only counterfactual eligibility surface
   using already accepted beam geometry,
3. decide whether any further beam-semantics work is justified,
4. stop immediately if the bounded audit does not materially change the
   current interpretation.

This is **not** a license to resume broad reproduction work or to launch
new long training runs.

## 2. Trigger

The trigger for this SDD is already recorded in:

1. [`../../artifacts/phase-01d-reopen-trigger-check-2026-04-22-beam-semantics.md`](../../artifacts/phase-01d-reopen-trigger-check-2026-04-22-beam-semantics.md)

The justified trigger is `Trigger 2: new defect that could invalidate
Phase 01B or Phase 01C interpretation`, but only in a narrow semantic
validity sense.

The defect is not artifact corruption.

The defect is that the current baseline's comparator and beam-choice
semantics may be too compressed to support the intended beam-granular
interpretation of the frozen negative-result surface.

## 3. Questions Required By Phase 01D

### 3.1 What new evidence or defect triggered the reopen?

The trigger is a concrete semantic-validity defect established by the
read-only audit:

1. beam validity is satellite-granular inside each visible satellite,
2. channel quality is copied across all beams of a visible satellite,
3. `RSS_max` degenerates toward a first-valid-beam comparator,
4. preserved checkpoint traces show that this is not just a theoretical
   concern.

### 3.2 Which claim boundary does it change?

It changes only the semantic-validity boundary for:

1. beam-level comparator interpretation,
2. beam-level decision-surface interpretation,
3. whether current near-ties should still be read as evidence about a
   meaningfully beam-discriminative handover problem.

It does **not** change:

1. artifact integrity of the frozen baseline,
2. the landed downstream bundle contract,
3. the disclosed existence of known reward-geometry anomalies,
4. the claim that the current repo is a portable disclosed baseline.

### 3.3 Why is this slice bounded and auditable?

Because it is limited to:

1. preserved artifacts and preserved checkpoints first,
2. machine-readable audit metrics over the current semantics,
3. one opt-in evaluation-only counterfactual eligibility surface,
4. one explicit stop rule before any retraining is allowed.

### 3.4 What result would justify continuing?

This slice justifies a later follow-on only if all of these are true:

1. beam-semantic collapse is shown to be pervasive on the preserved
   evaluation surface,
2. the minimal counterfactual eligibility surface materially changes
   comparator meaning or tie structure,
3. that change is not confined to a trivial beam-index tie-breaker with
   no downstream effect,
4. the evidence suggests that a later training follow-on could answer a
   real scientific question rather than simply tuning around the same
   semantics.

### 3.5 What result would force an immediate stop?

Stop immediately if any of these becomes true:

1. the audit shows the collapse is local rather than pervasive,
2. the counterfactual surface does not materially change ranking,
   decision margins, or comparator behavior,
3. the only observable change is a cosmetic tie-break difference,
4. the slice would need new long training runs just to become
   interpretable.

## 4. Scope

### 4.1 In Scope

1. one metric-producing audit surface over preserved checkpoints,
2. one opt-in evaluation-only counterfactual eligibility mode grounded in
   already accepted beam geometry,
3. one bounded comparison artifact on the default `100 users / 4
   satellites` surface,
4. one reviewed status note stating whether the repo should stop again or
   promote a later follow-on.

### 4.2 Explicitly Out Of Scope

1. a new `9000`-episode long run,
2. broad sweep-family expansion,
3. reward-geometry or atmospheric-sign changes,
4. introducing a new off-axis channel-gain formula in this slice,
5. silently changing the frozen baseline runtime semantics,
6. rewriting the landed export bundle contract,
7. changing `ntn-sim-core` consumer code.

## 5. Working Hypothesis

The current baseline likely compresses the beam-level handover problem in
two coupled ways:

1. satellite-visible validity makes all beams on a visible satellite
   legal actions,
2. channel quality then becomes identical across those beams because the
   current model omits off-axis beam gain.

If that hypothesis is right, then:

1. `RSS_max` is weaker than intended,
2. MODQN is learning mostly from load, handover cost, and weakly
   localized global signals inside a tied channel block,
3. more episodes alone are unlikely to produce a qualitatively different
   result.

The null hypothesis is explicit:

1. even under a more beam-discriminative eligibility proxy, current
   rankings and tie structure remain effectively unchanged.

If the null survives, the repo should stop again.

## 6. Slice 01E1: Beam-Degeneracy Audit Surface

This first slice is evaluation-only and uses preserved checkpoints.

Required work:

1. add one audit helper that can inspect a selected checkpoint and emit
   beam-semantic metrics over one bounded evaluation surface,
2. compute, at minimum:
   - valid action count per user-step
   - per-satellite valid beam block counts
   - distinct `channel_quality` value count inside each valid beam block
   - fraction of user-steps where all valid beams in a visible satellite
     share the same channel value
   - `RSS_max` tie / first-valid-beam rate
   - scalarized top-1 minus top-2 margin distribution on the decision
     mask
3. emit machine-readable artifacts plus one short review note.

Preferred outputs:

1. `beam_semantics_summary.json`
2. `beam_tie_metrics.csv`
3. `decision_margin_metrics.csv`
4. `review.md`

Acceptance:

1. the audit can be run on the preserved best-eval checkpoint without
   retraining,
2. the output is explicit enough to say whether beam-collapse is rare,
   common, or pervasive,
3. the note distinguishes clearly between:
   - valid-mask collapse
   - channel-value collapse
   - comparator degeneration

## 7. Slice 01E2: Evaluation-Only Counterfactual Eligibility Surface

If Slice `01E1` confirms meaningful collapse, the only allowed
counterfactual in this SDD is a **mask-only** beam-center-aware
eligibility proxy.

Rationale:

1. the repo already has an accepted beam-geometry proxy under
   `ASSUME-MODQN-REP-002`,
2. this allows one minimal semantics change without inventing a new
   off-axis channel model,
3. a mask-only counterfactual is easier to audit than a new channel-gain
   branch.

Required rule:

1. for each visible satellite, the counterfactual surface should reduce
   eligibility from "all 7 beams valid" to a beam-center-aware subset
   derived from existing beam geometry,
2. the exact proxy must be deterministic and disclosed,
3. the frozen baseline mode must remain untouched and remain the default.

The preferred first proxy is:

1. one nearest-beam-per-visible-satellite rule using existing beam-center
   geometry.

This counterfactual is evaluation-only in this slice:

1. no new training semantics,
2. no new default config role,
3. no promotion into the baseline without a separate future decision.

Required outputs:

1. `counterfactual_eval_summary.json`
2. `counterfactual_vs_baseline.csv`
3. one short note that states whether comparator meaning or tie structure
   materially changed.

## 8. Slice 01E3: Decision Gate

This slice ends with one explicit decision note.

Allowed outcomes:

1. `stop`
   - current baseline remains the disclosed authority surface
   - beam-semantic issue is recorded but does not justify further work
2. `promote one later follow-on`
   - only if Slice `01E2` shows a material semantic effect
   - any later slice must get its own execution SDD

The decision note must answer:

1. whether beam-collapse was pervasive,
2. whether the counterfactual changed comparator behavior,
3. whether any change extended beyond `r2` / handover-only effects,
4. whether a later training follow-on is scientifically justified.

## 9. Promotion Rule

Promotion beyond this SDD is allowed only if all of these are true:

1. beam-collapse is shown to be pervasive on the default surface,
2. `RSS_max` or another comparator materially changes under the
   counterfactual eligibility proxy,
3. weighted ranking or tie structure changes in a way that is not purely
   a cosmetic beam-index tie-break,
4. the result suggests a clear next bounded follow-on.

If any of these fail, do not promote.

## 10. Negative-Result Rule

Negative results are valid outcomes.

If the audit confirms the current semantics but the counterfactual does
not materially change interpretation, the repo should record that:

1. the beam-semantic concern was audited directly,
2. the concern did not justify further implementation,
3. the repo remains best interpreted as:
   - a disclosed baseline with known semantic assumptions,
   - not a branch that should keep reopening by default.

## 11. Deliverables

This execution SDD is complete when the repo contains:

1. one metric-producing beam-semantic audit surface,
2. one reviewed bounded artifact set for the baseline audit,
3. one reviewed counterfactual eligibility comparison artifact,
4. one short status note that says `stop` or `promote one later
   follow-on`.
