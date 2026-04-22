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

## Truthful Demo Qualification Export

- reviewed producer-backed candidate:
  `artifacts/scenario-corrected-pilot-01/export-bundle/`
- replay window selection:
  `replay_start_time_s=430`, `replay_slot_count=650`
- producer-side curation:
  trimmed to `user-0` only after export so the candidate stays small while
  preserving producer-owned truth
- qualification summary:
  - `distinct visible satellites = sat-0, sat-3`
  - `visible-link rows = 43`
  - `inter-satellite-handover rows = 1`
  - `slot count = 650`
  - `user count = 1`

Root cause for the earlier blocked candidate:

- the original export always replayed from `t=0` for the default
  `10`-slot episode window,
- under the Beijing follow-on orbit surface that window only exposed
  `sat-0`, so the bundle could show visible links and same-satellite
  beam switches but not multi-satellite visible coverage or an
  inter-satellite handover,
- the fix was therefore producer-side replay-window selection, not any
  consumer-side fixture patching or JSON editing.

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
