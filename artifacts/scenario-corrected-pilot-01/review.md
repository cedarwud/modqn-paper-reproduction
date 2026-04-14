# Scenario-Corrected Pilot 01 Review

## Summary

- Run directory: `artifacts/scenario-corrected-pilot-01/`
- Config: `configs/modqn-paper-baseline.paper-faithful-follow-on.resolved.yaml`
- Episodes: `50`
- Purpose: first real follow-on pilot after landing the paper-backed
  scenario surface from Phase 01B Slice A

## Scenario Surface

- ground point: `(40.0, 116.0)`
- user area: `200 km x 90 km`
- user distribution: `uniform-rectangle`
- mobility model: `random-wandering`
- export bundle: `artifacts/scenario-corrected-pilot-01/export-bundle/`

## Observed Outputs

- `training_log.json` present
- `run_metadata.json` present
- `checkpoints/final-episode-policy.pt` present
- `checkpoints/best-weighted-reward-on-eval.pt` present
- export bundle manifest and replay surfaces present

## Key Facts

- elapsed wall time: `551.30 s`
- final episode: `49`
- final training scalar reward: `601.2556`
- best observed training scalar episode: `34`
- best observed training scalar reward: `613.3385`
- eval-selected checkpoint episode: `49`
- eval-selected mean scalar reward: `51.9827`
- eval-selected objective means:
  - `mean_r1 = 173.7448`
  - `mean_r2 = -0.4690`
  - `mean_r3 = -173.7448`
  - `mean_total_handovers = 93.8`

## Interpretation

- The follow-on scenario-corrected runtime surface is stable enough for
  real pilot work, not just smoke validation.
- Best-eval checkpointing still works, but in this `50`-episode pilot the
  selected checkpoint is simply the final episode rather than a distinct
  earlier checkpoint.
- The pilot does not show an obvious reward-scale improvement over the
  frozen baseline story. The same reward-dominance warning remains
  active, with `|r1| / |r2| ~= 982.6x`.
- This run is useful evidence that the scenario correction is executable.
  It is not evidence that scenario correction alone recovers paper-like
  method separation.

## Comparison Against The Frozen Baseline

- Compared with `artifacts/pilot-02-best-eval/`, the follow-on pilot is
  running on a more paper-backed geography and mobility surface.
- Compared with that frozen baseline pilot, the follow-on pilot remains
  in the same reward scale regime and does not by itself justify
  upgrading the repo from `disclosed comparison baseline` to
  `paper-faithful baseline`.
