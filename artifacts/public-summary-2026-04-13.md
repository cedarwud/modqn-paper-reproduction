# Public Reproduction Summary

Date: `2026-04-13`

## Current Status

This repository is now a working standalone reproduction surface for
`PAP-2024-MORL-MULTIBEAM`.

What is established:

- repo-only training and evaluation work without any external paper
  workspace
- baseline training, checkpointing, resume, and best-eval checkpoint
  selection run end to end
- comparator baselines, `Table II`, and a first executable `Fig. 3`
  sweep/export surface are available with machine-readable outputs
- smoke, pilot, sweep, export, and hardening checks are passing in the
  repo-local environment

## Main Result

The project is ready to serve as a **disclosed engineering baseline**,
but it should **not yet be presented as a fully paper-faithful
reproduction**.

## Why The Full Reproduction Claim Is Still Open

- the existing `9000`-episode run shows a real late-training collapse:
  the best policy appears mid-run, while the final checkpoint is much
  worse
- `Table II` is near-tied across methods, with most visible differences
  coming from switching-cost / handover behavior rather than broad
  multi-objective separation
- the `Fig. 3` user-count pilot shows the same pattern instead of
  opening a new clearly separating regime
- a first explicit reward-calibration experiment did not improve the raw
  evaluation surface

## Recommended Interpretation

The honest current claim is:

- baseline reproduction surface: working
- artifact and analysis surface: working
- full scientific reproduction claim: not yet established

For downstream comparisons, this repo can already be used as a
comparison baseline as long as the known limitations are disclosed.

## Recommended Next Step

Do not start a new long run by default.

Either:

1. use the current repo as a disclosed baseline bundle for comparison
   work, or
2. open a new explicitly experimental track aimed at resolving the
   near-tie / objective-drift problem

## Backing Notes

- [Full Status Note](/home/sat/satellite/modqn-paper-reproduction/artifacts/reproduction-status-2026-04-12.md)
- [Long-Run Anomaly Review](/home/sat/satellite/modqn-paper-reproduction/artifacts/run-9000/anomaly-review.md)
- [Fig. 3 Pilot Review](/home/sat/satellite/modqn-paper-reproduction/artifacts/fig-3-pilot-01/review.md)
