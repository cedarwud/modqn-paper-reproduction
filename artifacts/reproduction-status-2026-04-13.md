# Reproduction Status

Date: `2026-04-13`  
Scope: repo-local status summary after closing the non-smoke `Fig. 4`
to `Fig. 6` pilot gap for the comparison-baseline bundle.

## Scope Anchors

- baseline pilot: `artifacts/pilot-02-best-eval/`
- reward-calibration pilot: `artifacts/pilot-03-reward-calibration-200ep/`
- `Table II` analysis: `artifacts/table-ii-200ep-01/`
- `Fig. 3` pilot: `artifacts/fig-3-pilot-01/`
- `Fig. 4` pilot: `artifacts/fig-4-pilot-01/`
- `Fig. 5` pilot: `artifacts/fig-5-pilot-01/`
- `Fig. 6` pilot: `artifacts/fig-6-pilot-01/`
- reward-geometry analysis: `artifacts/reward-geometry-01/`
- long-run anomaly reference: `artifacts/run-9000/`

## Executive Status

The repo is now ready to freeze as a **disclosed comparison baseline**,
but it is **not** a fully established paper-faithful reproduction.

What is now true:

- repo-only authority is stable and portable
- baseline training, checkpointing, resume, best-eval checkpoint
  selection, comparator sweeps, and export surfaces all run
- `Table II` and `Fig. 3` to `Fig. 6` all now have reviewed non-smoke
  pilot artifacts with machine-readable outputs
- the baseline-acceptance checklist is closed for comparison-baseline
  purposes

What is still not true:

- the paper's intended method separation has not been convincingly
  reproduced
- the long-run pathology remains unresolved
- the repo still cannot honestly claim full paper-faithful
  reproduction

## Verified Positive Results

1. The repo has crossed the comparison-baseline closeout boundary.
   - required figure evidence now exists for `Fig. 3` to `Fig. 6`
   - reviewed artifact notes exist for `run-9000`, baseline pilots,
     `Table II`, and all figure pilots

2. Baseline runtime plumbing remains healthy.
   - smoke, pilot, sweep, export, and hardening coverage were already in
     place
   - this round extended the figure evidence surface without changing the
     baseline definition

3. The disclosed baseline bundle is now broad enough for downstream
   comparison.
   - current bundle covers long-run reference, checkpoint-selection
     behavior, `Table II`, and all four figure families

## Main Negative Findings

1. `run-9000` still shows a real late-training collapse.
   - best regime is mid-run, not final-run
   - throughput falls sharply in late training
   - load-balance term worsens sharply
   - handover cost improves while service quality degrades

2. `Table II` remains near-tied across methods.
   - exact tie rows: `4`
   - max scalar spread across methods: `0.043`
   - dominant cross-method variation is still in `r2` / handover

3. `Fig. 3` remains near-tied and handover-dominant.
   - exact tie points: `5/9`
   - max scalar spread across methods: `0.1500`
   - the largest visible anomaly is local at `200` users

4. `Fig. 4` is even more collapsed than `Fig. 3`.
   - exact tie points: `7/7`
   - `MODQN` and `DQN_throughput` tie-best at every point
   - max scalar spread across methods: `0.0120`
   - raw objective means are invariant across the full satellite-count
     sweep under the current protocol

5. `Fig. 5` also remains near-tied.
   - exact tie points: `5/5`
   - max scalar spread across methods: `0.0120`
   - user-speed changes only produce very small drift in `r1` / `r3`
   - method differences remain local to `r2` / handover

6. `Fig. 6` keeps the same structure.
   - exact tie points: `5/5`
   - max scalar spread across methods: `0.0120`
   - satellite-speed changes shift the scalar slightly, but not the
     method ordering
   - method differences remain local to `r2` / handover

7. Reward-calibration still does not fix the raw eval surface.
   - it remains a disclosed sensitivity artifact only
   - it must not replace the baseline

## Current Interpretation

The infrastructure gap is no longer the main issue.

The repo now has enough reviewed evidence to freeze a disclosed
comparison baseline, but all major result surfaces still point to the
same scientific bottleneck:

- method separation remains weak
- throughput and load-balance remain effectively shared across methods
- visible differences are mostly handover-related
- the long-run collapse remains unresolved

That means the honest current claim is:

- comparison-baseline bundle: **ready**
- full paper-faithful reproduction: **not established**

## Decision

Freeze the current repo as a **disclosed comparison baseline**.

Do **not** start a new `9000`-episode long run by default.

Do **not** silently replace the baseline with the reward-calibration
experiment.

If work continues beyond baseline freeze, it should be opened as a new
explicitly experimental track aimed at the near-tie / objective-drift
problem.
