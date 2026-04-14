# Phase 01B Slice B Bounded Status

Date: `2026-04-14`

## Scope Anchors

- scenario-corrected pilot: `artifacts/scenario-corrected-pilot-01/`
- bounded `Table II`: `artifacts/scenario-corrected-table-ii-bounded-01/`
- bounded `Fig. 3`: `artifacts/scenario-corrected-fig-3-bounded-01/`
- frozen baseline reference:
  - `artifacts/pilot-02-best-eval/`
  - `artifacts/table-ii-pilot-01/`
  - `artifacts/table-ii-200ep-01/`
  - `artifacts/fig-3-pilot-01/`

## Executive Status

Phase 01B Gate 1 is closed.

The new paper-backed scenario surface is executable and stable enough for
real pilot work.

However, the first bounded Slice B evidence says the same thing the
frozen baseline already suggested:

- near-tie behavior is still present
- `r1` still dominates the reward scale
- visible method differences are still mostly handover-related

That means the follow-on work has **not** yet earned a stronger
`paper-faithful reproduction` claim.

## Key Findings

1. The scenario-corrected pilot is real, not just a smoke run.
   - config surface: Beijing-centered rectangular area plus
     `random-wandering`
   - `50` episodes completed in `551.30 s`
   - final scalar reward: `601.2556`
   - best observed training scalar: `613.3385` at episode `34`
   - best-eval checkpoint: episode `49`, mean scalar `51.9827`

2. Reward-scale dominance remains active on the follow-on track.
   - base follow-on diagnostics still report `|r1| / |r2| ~= 982.6x`
   - bounded `Fig. 3` diagnostics reached even larger ratios on the
     exercised low-load points:
     - `40` users: `2456.4x`
     - `60` users: `1637.6x`
     - `80` users: `1228.2x`

3. The bounded follow-on `Table II` surface is still collapsed.
   - rows exercised: `1.0/1.0/1.0`, `1.0/0.0/0.0`, `0.0/1.0/0.0`
   - exact tie rows: `2/3`
   - max scalar spread across methods: `0.0140`
   - max `r1` spread: `0.0000`
   - max `r3` spread: `0.0000`
   - only `r2` / handover shows visible method variation

4. The bounded follow-on `Fig. 3` preview also stays near-tied.
   - point set actually exercised: `40`, `60`, `80` users
   - exact tie points: `2/3`
   - max weighted-reward spread across methods: `0.026625`
   - `40` and `60` users: `MODQN`, `DQN_throughput`, and `DQN_scalar`
     tie exactly
   - `80` users: `DQN_scalar` is best, but only by `0.02325` over
     `MODQN` / `DQN_throughput`

## Comparison Against The Frozen Baseline

- The frozen baseline already showed near-tied `Table II` rows and a
  handover-heavy `Fig. 3` surface.
- The follow-on bounded results do not overturn that diagnosis.
- The stronger claim that can now be made is only this:
  - the near-tie / dominance pattern survives even after correcting the
    highest-impact paper-backed scenario mismatch

## Current Interpretation

Phase 01B has now answered an important question:

Correcting the user geography and mobility surface alone is **not**
enough to recover paper-like separation on the first bounded follow-on
evidence.

That pushes the next likely bottleneck away from pure scenario fidelity
and back toward:

1. reward geometry / scalarization scale
2. comparator training protocol
3. targeted high-signal evaluation points instead of broad expensive
   sweeps by default

## Decision

Do not treat Phase 01B as complete.

Do not upgrade the repo to a `paper-faithful baseline`.

Do not start a blind full follow-on sweep family just because the
scenario surface is now more paper-backed.

If the follow-on continues, the most defensible next step is:

1. a targeted high-load `Fig. 3` follow-up that explicitly includes the
   paper range endpoint where the frozen baseline showed its largest
   anomaly, and/or
2. a new explicit experiment on reward geometry or comparator protocol

Only after one of those surfaces shows materially different behavior
should a full follow-on `Table II` or `Fig. 3` expansion be prioritized.
