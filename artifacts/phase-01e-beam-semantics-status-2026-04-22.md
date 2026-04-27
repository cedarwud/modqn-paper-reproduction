# Phase 01E Beam Semantics Status

Date: `2026-04-22`

## Scope

This note closes the bounded `Phase 01E` evaluation-only reopen slices:

1. `01E1` beam-degeneracy audit over the preserved best-eval checkpoint
2. `01E2` one counterfactual eligibility proxy:
   `nearest-beam-per-visible-satellite`

No trainer defaults, baseline runtime semantics, or long training runs
were changed in this phase.

## Executed Surfaces

### 01E1 Beam-Degeneracy Audit

Artifact directory:

- `artifacts/codex-pilot-200ep-2026-04-22/beam-semantics-audit/`

Key outputs:

- `beam_semantics_summary.json`
- `beam_tie_metrics.csv`
- `decision_margin_metrics.csv`
- `review.md`

Observed result:

1. valid-mask collapse was `pervasive`
2. channel-value collapse was `pervasive`
3. `RSS_max` tie / first-valid-beam degeneration was `pervasive`
4. `MODQN` top-1 / top-2 candidates stayed inside the same satellite block

Interpretation:

1. `01E1` confirms that the frozen baseline meaningfully compresses the
   beam-level decision surface
2. `01E2` was therefore justified

### 01E2 Counterfactual Eligibility

Artifact directory:

- `artifacts/codex-pilot-200ep-2026-04-22/beam-counterfactual-audit/`

Key outputs:

- `counterfactual_eval_summary.json`
- `counterfactual_vs_baseline.csv`
- `review.md`

Counterfactual rule:

1. for each visible satellite, keep exactly one eligible beam
2. choose that beam with existing `beam_pattern.nearest_beam(...)`
3. apply this only at evaluation time

Observed result:

1. same-state action-change rate was high:
   - `MODQN`: `0.954`
   - `RSS_max`: `0.936`
2. same-state satellite-change rate was `0.0` for both policies
3. tie structure changed sharply:
   - `RSS_max` tie rate: `1.0 -> 0.0`
   - `MODQN` top-1/top-2 same-satellite rate: `1.0 -> 0.0`
4. rollout-level weighted reward changed materially despite unchanged
   same-state satellite choice:
   - `MODQN` scalar reward delta: `+424.6979`
   - `RSS_max` scalar reward delta: `+424.7174`
5. throughput and handover surfaces also moved:
   - `r1` increased sharply
   - handovers increased
   - `r3` became less negative

Interpretation:

1. this is **not** a pure cosmetic beam-index tie-break
2. even without changing same-state satellite choice, the counterfactual
   beam allocation materially reshapes beam loads and downstream reward
3. the current frozen baseline therefore hides a beam-level degree of
   freedom that is scientifically relevant on the default surface

## Decision

`Phase 01E does not return to stop.`

It also does **not** authorize immediate long training.

The correct next move is:

1. create one new bounded follow-on execution SDD for training under an
   explicitly disclosed beam-aware eligibility surface
2. keep that follow-on small first:
   - smoke
   - `20` / `200` bounded pilot
3. only consider larger training if the bounded follow-on still shows
   material differences

## What This Phase Established

This phase established all of the following:

1. the frozen baseline compression is real and pervasive
2. one minimal beam-aware counterfactual produces material downstream
   effects
3. a later training follow-on can now be justified scientifically
4. the justification comes from beam-level load allocation effects, not
   from a same-state satellite-selection change

## What This Phase Did Not Establish

This phase did **not** establish:

1. that the counterfactual is the final correct semantics
2. that a new training branch will recover the paper's intended method
   separation
3. that a `500`-episode or `9000`-episode run should begin immediately
4. that atmospheric-sign or reward-geometry issues are resolved
