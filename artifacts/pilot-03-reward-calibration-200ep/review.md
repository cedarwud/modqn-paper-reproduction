# Reward-Calibration Pilot Review

Artifact: `artifacts/pilot-03-reward-calibration-200ep`

## Scope

This run uses the explicit experimental config
`configs/modqn-paper-baseline.reward-calibration.resolved.yaml`.
Trainer-side rewards are calibrated by fixed diagnostic scales, while
evaluation, logged `r1/r2/r3`, and checkpoint selection remain on the
raw paper-metric surface.

## Key Result

The `200`-episode reward-calibration pilot does **not** improve the
current reproduction bottleneck on the raw evaluation surface.

- baseline pilot artifact: `artifacts/pilot-02-best-eval`
- reward-calibration pilot artifact: `artifacts/pilot-03-reward-calibration-200ep`

## Comparison

- baseline final training scalar: `602.8169`
- reward-calibration final training scalar: `602.3790`
- baseline best observed training scalar: `616.1214` at episode `34`
- reward-calibration best observed training scalar: `616.3249` at episode `34`
- baseline best-eval checkpoint episode: `99`
- reward-calibration best-eval checkpoint episode: `49`
- baseline best-eval mean scalar: `52.2599`
- reward-calibration best-eval mean scalar: `52.2599`
- baseline best-eval mean raw objectives: `r1=174.6187`, `r2=-0.4190`, `r3=-174.6187`, `handover=83.8`
- reward-calibration best-eval mean raw objectives: `r1=174.6187`, `r2=-0.4190`, `r3=-174.6187`, `handover=83.8`

## Interpretation

- This experiment changes optimization targets inside the trainer, but
  it does not change the raw evaluation outcome at the `200`-episode
  pilot scale.
- The current near-tie / handover-dominated `Table II` picture remains
  unresolved.
- There is not yet evidence that fixed-scale reward calibration alone
  is enough to recover a more paper-like objective separation.

## Decision

- Do not replace the baseline pilot with this experiment run.
- Keep this artifact as an explicit sensitivity result.
- The next useful step is analysis/reporting, not immediate long-run
  retraining.
