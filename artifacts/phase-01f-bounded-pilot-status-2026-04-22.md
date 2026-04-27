# Phase 01F Bounded Pilot Status

Date: `2026-04-22`

## Scope

This note records the bounded pilot results for the new beam-aware
eligibility follow-on surface after `Slice 01F1` landed.

Executed runs:

1. `20` episodes
2. `200` episodes

Both runs used:

1. `configs/modqn-paper-baseline.beam-aware-eligibility-follow-on.resolved.yaml`
2. `nearest-beam-per-visible-satellite`
3. the same fixed seed/evaluation policy as the frozen baseline

## Artifact Directories

### Beam-Aware 20

Training:

- `artifacts/codex-beam-aware-pilot-20ep-2026-04-22/`

Export:

- `artifacts/codex-beam-aware-pilot-20ep-2026-04-22/export-bundle/`

Audit:

- `artifacts/codex-beam-aware-pilot-20ep-2026-04-22/beam-semantics-audit/`

### Beam-Aware 200

Training:

- `artifacts/codex-beam-aware-pilot-200ep-2026-04-22/`

Export:

- `artifacts/codex-beam-aware-pilot-200ep-2026-04-22/export-bundle/`

Audit:

- `artifacts/codex-beam-aware-pilot-200ep-2026-04-22/beam-semantics-audit/`

### Frozen Baseline References

1. `artifacts/codex-smoke-20ep-2026-04-22/`
2. `artifacts/codex-pilot-200ep-2026-04-22/`

## Result Summary

### 1. Beam-Aware vs Frozen Baseline

Held-out best-eval summary changed materially and consistently.

For both `20` and `200` beam-aware pilots versus the frozen baseline:

1. `mean_scalar_reward` improved from `52.2599` to `507.0423`
2. `mean_r1` improved from `174.6187` to `1021.6402`
3. `mean_r3` improved from `-174.6187` to `-17.9846`
4. `mean_r2` became more negative from `-0.4190` to `-0.6030`
5. `mean_total_handovers` increased from `83.8` to `120.6`

Interpretation:

1. the beam-aware follow-on clearly changes the decision surface in a
   meaningful way
2. the gain is not free; it trades higher throughput and much better
   load-balance against more handovers

### 2. Beam-Semantic Structure

Beam-semantic audit on both beam-aware runs reported:

1. valid-mask collapse: `absent`
2. channel-value collapse: `absent`
3. comparator degeneration: `absent`

Interpretation:

1. the new runtime surface successfully removes the collapsed
   beam-selection structure identified in `Phase 01E`
2. this is true both after `20` episodes and after `200` episodes

### 3. 20 vs 200 Inside The Beam-Aware Branch

This is the most important bounded-pilot result.

The `20`-episode and `200`-episode beam-aware runs produced the same
held-out best-eval summary:

1. `mean_scalar_reward = 507.0422520259857`
2. `mean_r1 = 1021.6401572761536`
3. `mean_r2 = -0.603`
4. `mean_r3 = -17.984633060455426`
5. `mean_total_handovers = 120.6`

Additional context:

1. `20`-episode best eval came at episode `19`
2. `200`-episode best eval came at episode `49`
3. the `200`-episode training curve stayed in the same scalar-reward
   band rather than opening a new higher-quality surface

Interpretation:

1. the beam-aware branch improves the surface quickly
2. extending from `20` to `200` episodes did **not** add a new held-out
   signal
3. bounded evidence currently supports the existence of a better
   semantics, but not the need for larger training

## Decision

`Phase 01F bounded pilots are sufficient to prove material semantic change.`

They do **not** justify an immediate jump to:

1. `500` episodes
2. `9000` episodes

Current recommendation:

1. stop at the bounded-pilot level for now
2. keep the frozen baseline as the disclosed comparison authority
3. keep the beam-aware branch as an explicit follow-on surface
4. only consider larger training if there is a new question that the
   bounded pilots cannot answer

## Practical Bottom Line

What is now established:

1. `01F1` landed cleanly
2. beam-aware training works
3. exported artifacts disclose the new eligibility mode
4. beam collapse is removed
5. the new semantics materially outperform the frozen baseline on the
   bounded surface

What is **not** established:

1. that `200 -> 500 -> 9000` is likely to yield more useful signal
2. that the paper's intended final method separation is now reproduced
3. that the atmospheric-sign or reward-geometry issues are solved
