# Reproduction Status

Date: `2026-04-12`  
Scope: repo-local status summary based only on the current codebase and
artifact set.

## Scope Anchors

- baseline pilot: `artifacts/pilot-02-best-eval/`
- reward-calibration pilot: `artifacts/pilot-03-reward-calibration-200ep/`
- `Table II` analysis: `artifacts/table-ii-200ep-01/`
- `Fig. 3` pilot: `artifacts/fig-3-pilot-01/`
- reward-geometry analysis: `artifacts/reward-geometry-01/`
- long-run anomaly reference: `artifacts/run-9000/`

## Executive Status

The repo is healthy as a standalone reproduction surface, but the paper
is **not yet fully and convincingly reproduced**.

What is already true:

- repo-only authority is stable and portable
- baseline training, checkpointing, resume, best-eval checkpoint
  selection, comparator sweeps, and export surfaces all run
- `Table II` can be emitted as CSV/JSON/PNG
- first executable `Fig. 3` to `Fig. 6` sweep/export surfaces exist
- explicit analysis surfaces now exist for:
  - long-run anomaly review
  - `Table II` objective spread
  - `Fig. 3` user-count sweep review
  - reward geometry / normalization sensitivity
  - reward-calibration pilot sensitivity

What is not yet true:

- full non-smoke `Fig. 4` to `Fig. 6` pilots have not been run
- there is no strong evidence yet that the current training/evaluation
  regime reproduces the paper's intended method separation
- the long-run pathology remains unresolved

## Verified Positive Results

1. Baseline training surface is stable.
   - smoke, pilot, sweeps, export, and hardening tests are all passing
   - the repo has already validated final-checkpoint and best-eval
     checkpoint capture

2. `ASSUME-MODQN-REP-015` is implemented end to end.
   - `artifacts/pilot-02-best-eval/` carries both
     `final-episode-policy.pt` and
     `best-weighted-reward-on-eval.pt`
   - the eval-selected checkpoint is distinct from the final episode
     checkpoint

3. `Table II` is now executable rather than speculative.
   - `artifacts/table-ii-200ep-01/` contains machine-readable
     outputs and analysis

## Main Negative Findings

1. `run-9000` still shows a real long-run collapse.
   - best regime is mid-run, not final-run
   - throughput falls sharply in late training
   - load-balance term worsens sharply
   - handover cost improves while service quality degrades

2. `Table II` is near-tied across methods.
   - exact tie rows: `4`
   - max scalar spread across methods: `0.043`
   - max `r1` spread: `0.0`
   - max `r3` spread: `0.0`
   - dominant cross-method variation is in `r2` / handover

3. Reward dominance remains the simplest explanation.
   - raw diagnostic ratios remain:
     - `|r1| / |r2| ≈ 982.6x`
     - `|r1| / |r3| ≈ 28x`
   - reward-geometry re-scoring changes spread magnitudes, but it does
     not change the overall near-tied picture

4. Fixed-scale reward calibration did not improve the raw eval outcome at
   the `200`-episode pilot scale.
   - baseline pilot best-eval mean scalar: `52.2599`
   - reward-calibration pilot best-eval mean scalar: `52.2599`
   - raw eval `r1/r2/r3/handover` means are the same

5. `Fig. 3` user-count sweep does not open a new separating regime.
   - exact tie points: `5/9`
   - max scalar spread across methods: `0.1500`
   - average weighted reward across all points differs by only `0.0150`
     between best and worst method
   - `r1` and `r3` remain effectively identical across methods per point
   - the largest visible anomaly is local: at `200` users, `MODQN`
     incurs much worse `r2` / handover while keeping the same `r1` and
     `r3`

## Current Interpretation

The project has crossed the line from "training surface missing" into
"scientific reproduction bottleneck".

The main blocker is no longer infrastructure. It is that the current
policy/evaluation regime does not yet yield clear paper-like separation:

- short and pilot runs are healthy enough to trust operationally
- long runs reveal objective drift
- `Table II` mostly collapses into handover-only variation
- `Fig. 3` shows the same near-tie / handover-dominant structure over
  user-count scaling
- a first explicit calibration experiment did not fix that collapse

That means the honest current claim is:

- baseline reproduction surface: **working**
- paper-scale reproduction claim: **not yet established**

## Decision

Do **not** start a new `9000`-episode long run now.

Also do **not** silently replace the baseline with the
reward-calibration experiment.

The reward-calibration run should be retained only as a disclosed
sensitivity artifact.

## Most Reasonable Next Step

Use this status as the current repo-local conclusion and choose one of
two directions explicitly:

1. Delivery direction
   - decide whether to run `Fig. 4` to `Fig. 6` pilots even though
     `Fig. 3` already reproduces the current near-tie diagnosis
   - present the current anomaly findings as part of the reproduction
     disclosure

2. Research direction
   - design a new, clearly labeled experiment family aimed at resolving
     the near-tie / objective-drift problem
   - do not treat fixed-scale calibration as already successful

If only one next action is allowed, prefer:

- write results/disclosure around the current finding set first
- postpone further long-run retraining until a stronger experimental
  hypothesis exists
