# Review: Phase 01 MODQN Baseline Anchor

**Date:** `2026-04-28`  
**Decision:** `PROMOTE`  
**Scope:** baseline-anchor validation only; no EE, Catfish, reward-calibration, or follow-on implementation is promoted by this review.

## Decision Summary

The original MODQN baseline definition can stand as the comparison anchor for later Catfish / EE-MODQN work, with the following reward surface:

```text
r1 = throughput
r2 = handover penalty
r3 = load balance
```

This is not a full paper-faithful reproduction. It is a disclosed comparison baseline supported by the repo's resolved-run guardrail, baseline training/checkpoint/sweep/export surfaces, Table II and Fig. 3-6 review artifacts, and closeout claim boundary.

## Promoted Anchor Surface

Use this surface for later phases:

1. Config anchor: `configs/modqn-paper-baseline.resolved-template.yaml`
2. Authority-only config: `configs/modqn-paper-baseline.yaml`; this is not a direct training config.
3. Checkpoint anchor: `artifacts/pilot-02-best-eval/`
4. Long-run limitation anchor: `artifacts/run-9000/`
5. Sweep artifact anchors:
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

Reward-calibration, scenario-corrected, and beam-aware follow-on surfaces must not replace the original baseline. They can be used only as disclosed limitations or negative-result context.

## Metrics To Preserve

Later comparisons must preserve at least:

1. scalar reward
2. `r1`
3. `r2`
4. `r3`
5. handover count
6. final checkpoint semantics
7. best-eval checkpoint semantics

The best-eval checkpoint should be the primary comparable reporting surface. Final checkpoint results still matter as stability evidence, especially because long-run collapse is already documented.

## Known Limitations

1. Near-tie behavior: Table II and Fig. 3-6 surfaces are mostly near ties, with many differences concentrated in `r2`.
2. Late-training collapse: `artifacts/run-9000/` shows that final policy quality can be worse than mid-run or best-eval regimes.
3. Reward dominance: `r1` is much larger in scale than `r2`, so scalar reward can be throughput-dominated.
4. Claim boundary: the baseline supports disclosed comparison, not full paper-faithful reproduction or clear reproduced method separation.

## Allowed Claims

1. The original MODQN baseline has a rerunnable and comparable engineering baseline surface.
2. It can anchor follow-on comparisons when the same config, protocol, metric, and checkpoint semantics are used.
3. Best-eval checkpoint selection is part of the comparison protocol and helps avoid final-only misjudgment.

## Disallowed Claims

1. Do not claim the repo fully reproduces the original paper.
2. Do not claim MODQN clearly outperforms all comparators.
3. Do not claim long-run collapse has been solved.
4. Do not silently replace the original baseline with reward-calibration, scenario-corrected, or beam-aware follow-on surfaces.

## Result

`PROMOTE`, but only as a disclosed comparison baseline anchor.

The baseline is stable enough to support later phase comparisons, but every later phase must carry the stated limitations and must keep the original reward surface fixed unless that phase explicitly declares a new method variant.
