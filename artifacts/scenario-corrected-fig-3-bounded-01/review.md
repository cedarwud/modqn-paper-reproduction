# Scenario-Corrected Fig. 3 Bounded Review

## Scope

- artifact directory: `artifacts/scenario-corrected-fig-3-bounded-01/`
- suite: `fig-3`
- training episodes per learned method run: `20`
- sweep points actually exercised: `40, 60, 80` users
- baseline weight row: `[0.5, 0.3, 0.2]`
- reference run: `artifacts/scenario-corrected-pilot-01/`

## Key Findings

- The bounded follow-on `Fig. 3` pipeline completed successfully and
  emitted JSON, CSV, PNG, and analysis outputs.
- The low-load prefix of the user-count sweep remains near-tied:
  - exact tie points: `2/3`
  - max weighted-reward spread across methods: `0.026625`
- At `40` users and `60` users, `MODQN`, `DQN_throughput`, and
  `DQN_scalar` tie exactly on weighted reward.
- At `80` users, `DQN_scalar` is best at `65.0740`, but the margin over
  `MODQN` / `DQN_throughput` is only `0.02325`.
- Across all three points, `r1` and `r3` are identical across methods.
  The only visible separation is again in `r2` / handover count.

## Interpretation

- On the exercised low-load prefix, the paper-backed scenario correction
  does not open a new method-separating regime.
- The result is consistent with the frozen baseline diagnosis:
  throughput-scale dominance still constrains the learned policies into
  near-identical objective behavior.
- This bounded run therefore does not support a stronger
  `paper-faithful reproduction` claim by itself.

## Limitation

- `--max-figure-points 3` selected the prefix of the configured point
  set, so this artifact covers `40`, `60`, and `80` users only.
- It does not exercise the higher-load points where the frozen baseline
  `Fig. 3` pilot saw the largest local anomaly near `200` users.
- That means this bounded run is useful for triage, but it is not a
  substitute for a deliberate high-load follow-on check if Phase 01B
  continues.
