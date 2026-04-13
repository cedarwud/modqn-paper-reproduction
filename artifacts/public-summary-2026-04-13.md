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
- comparator baselines, `Table II`, and reviewed non-smoke `Fig. 3`
  to `Fig. 6` sweep/export surfaces are available with machine-readable
  outputs
- smoke, pilot, sweep, export, and hardening checks are passing in the
  repo-local environment

## Main Result

The project is now ready to be **frozen as a disclosed engineering
baseline**, but it should **not yet be presented as a fully
paper-faithful
reproduction**.

## Why The Full Reproduction Claim Is Still Open

- the existing `9000`-episode run shows a real late-training collapse:
  the best policy appears mid-run, while the final checkpoint is much
  worse
- `Table II` is near-tied across methods, with most visible differences
  coming from switching-cost / handover behavior rather than broad
  multi-objective separation
- `Fig. 3` to `Fig. 6` all preserve the same near-tie / handover-heavy
  structure instead of opening a clearly separating regime
- a first explicit reward-calibration experiment did not improve the raw
  evaluation surface

## Recommended Interpretation

The honest current claim is:

- baseline reproduction surface: working
- artifact and analysis surface: working
- comparison-baseline freeze: ready
- full scientific reproduction claim: not yet established

For downstream comparisons, this repo can already be used as a
comparison baseline as long as the known limitations are disclosed.

## Recommended Next Step

Do not start a new long run by default.

Either:

1. freeze the current repo as the disclosed comparison-baseline bundle,
   or
2. open a new explicitly experimental track aimed at resolving the
   near-tie / objective-drift problem

## Backing Notes

- [Baseline Acceptance Checklist](../docs/baseline-acceptance-checklist.md)
- [Full Status Note](reproduction-status-2026-04-13.md)
- [Long-Run Anomaly Review](run-9000/anomaly-review.md)
- [Fig. 3 Pilot Review](fig-3-pilot-01/review.md)
- [Fig. 4 Pilot Review](fig-4-pilot-01/review.md)
- [Fig. 5 Pilot Review](fig-5-pilot-01/review.md)
- [Fig. 6 Pilot Review](fig-6-pilot-01/review.md)
