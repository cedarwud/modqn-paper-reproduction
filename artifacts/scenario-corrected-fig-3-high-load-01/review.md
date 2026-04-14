# Scenario-Corrected Fig. 3 High-Load Review

## Scope

- artifact directory: `artifacts/scenario-corrected-fig-3-high-load-01/`
- config: `configs/modqn-paper-baseline.paper-faithful-follow-on.resolved.yaml`
- suite: `fig-3`
- requested sweep points: `160, 180, 200` users
- baseline weight row: `[0.5, 0.3, 0.2]`
- methods: `MODQN`, `DQN_throughput`, `DQN_scalar`, `RSS_max`
- training episodes per learned method run: `20`
- reference run: `artifacts/scenario-corrected-pilot-01/`

## Key Findings

- The targeted high-load follow-up completed successfully and emitted the
  required JSON, CSV, PNG, manifest, and analysis outputs.
- The exercised high-load end still remains near-tied:
  - exact tie points: `1/3`
  - max weighted-reward spread across methods: `0.028688`
- `MODQN` is not sole-best on any exercised high-load point:
  - `160` users: `DQN_scalar` sole-best by `0.001688`
  - `180` users: `MODQN` and `DQN_throughput` tie-best
  - `200` users: `DQN_scalar` sole-best by `0.000150`
- Across all three high-load points, `r1` and `r3` are identical across
  methods. Visible differences remain confined to `r2` / handover count.

## Comparison

- Compared with the frozen baseline `artifacts/fig-3-pilot-01/review.md`,
  this follow-up does not recover the old `200`-user anomaly as a useful
  `MODQN`-favorable regime. The earlier large local spread is gone, but it
  is replaced by an even tighter high-load near-tie rather than by stronger
  method separation.
- Compared with the bounded low-load preview
  `artifacts/scenario-corrected-fig-3-bounded-01/review.md`, the max spread
  only increases from `0.026625` to `0.028688`. That is directionally
  larger, but still far below a materially stronger signal.

## Promotion Gate Check

- Phase 01B Slice C promotion conditions are **not** met:
  - `MODQN` sole-best on at least `2/3` high-load points: `no` (`0/3`)
  - max weighted-reward spread at least `0.05`: `no` (`0.028688`)
  - non-trivial cross-method separation outside `r2` / handover: `no`

## Decision

- Do not run the optional `50`-episode confirmation pass.
- Do not promote directly to a full follow-on sweep family from this artifact.
- Treat this run as additional negative evidence that scenario correction
  alone does not recover paper-like method separation at the high-load end
  of `Fig. 3`.
