# Phase 01: MODQN Baseline Anchor

**Status:** Promoted as disclosed comparison baseline  
**Question:** Is the original MODQN baseline still a valid comparison anchor?

## Decision

`PROMOTE`, limited to a disclosed comparison baseline anchor.

The original MODQN baseline definition is stable enough for later Catfish / EE-MODQN comparisons when the comparison uses the same resolved config, checkpoint semantics, and metric surface. This does not promote the reproduction to a full paper-faithful reproduction, and it does not authorize reward-calibration, scenario-corrected, or beam-aware follow-on surfaces to replace the original baseline.

## Why This Is Independent

Every follow-on needs a stable reference. If the baseline cannot be reproduced, interpreted, or compared under the current artifact boundary, later EE or Catfish claims are not meaningful.

## Scope

Validate the existing MODQN baseline as the anchor:

```text
r1 = throughput
r2 = handover penalty
r3 = load balance
```

## Inputs

1. Existing MODQN reproduction docs and closeout notes.
2. Existing training / sweep / export artifacts.
3. Current resolved-run guardrails.

## Promoted Comparison Surface

Use these anchors for later phases:

1. Config anchor: `configs/modqn-paper-baseline.resolved-template.yaml`
2. Authority-only config: `configs/modqn-paper-baseline.yaml`
3. Best-eval checkpoint anchor: `artifacts/pilot-02-best-eval/`
4. Long-run limitation anchor: `artifacts/run-9000/`
5. Sweep anchors:
   - `artifacts/table-ii-200ep-01/`
   - `artifacts/fig-3-pilot-01/`
   - `artifacts/fig-4-pilot-01/`
   - `artifacts/fig-5-pilot-01/`
   - `artifacts/fig-6-pilot-01/`
6. Status notes:
   - `artifacts/reproduction-status-2026-04-13.md`
   - `docs/baseline-acceptance-checklist.md`
   - `artifacts/phase-01b-closeout-status-2026-04-14.md`
   - `artifacts/phase-01c-closeout-status-2026-04-15.md`

## Non-Goals

1. Do not change the reward.
2. Do not introduce EE.
3. Do not introduce Catfish.
4. Do not rerun long training by default.

## Checks

1. Baseline training path remains defined and reproducible.
2. Checkpoint and best-eval reporting semantics are understood.
3. Known limitations are explicitly carried forward:
   - near-tie surfaces,
   - late-training collapse,
   - reward dominance,
   - disclosed comparison-baseline claim boundary.
4. Metrics to preserve:
   - scalar reward,
   - `r1`,
   - `r2`,
   - `r3`,
   - handover count,
   - final checkpoint performance,
   - best-eval checkpoint performance.

The best-eval checkpoint is the primary reporting surface for comparable policy quality. Final checkpoint results must still be retained because `artifacts/run-9000/` documents late-training collapse and prevents final-only reporting from hiding instability.

## Allowed Claims

1. The original MODQN baseline has a rerunnable, comparable, assumption-disclosed engineering baseline surface.
2. It can anchor later comparisons if the same config, protocol, checkpoint semantics, and metric surface are used.
3. Best-eval checkpoint selection is part of baseline reporting and avoids final-only misinterpretation.

## Disallowed Claims

1. Do not claim full paper-faithful reproduction.
2. Do not claim MODQN has proven clear separation over all comparators.
3. Do not claim long-run collapse is solved.
4. Do not let reward-calibration, scenario-corrected, or beam-aware follow-on results silently replace the original baseline.

## Decision Gate

Promote only if the reviewer can state what original MODQN baseline result set should be used as the comparison anchor and what it is allowed to claim.

Stop if the reviewer finds that later phases would be comparing against an unstable or undefined baseline.

## Expected Output

A short baseline-anchor report:

1. baseline definition,
2. artifact / config anchors,
3. comparison metrics,
4. known limitations,
5. allowed and disallowed claims.

The accepted Phase 01 review is recorded in `reviews/01-modqn-baseline-anchor.review.md`.
