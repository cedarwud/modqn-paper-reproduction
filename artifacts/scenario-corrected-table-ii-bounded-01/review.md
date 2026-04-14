# Scenario-Corrected Table II Bounded Review

## Scope

- artifact directory: `artifacts/scenario-corrected-table-ii-bounded-01/`
- suite: `Table II`
- methods: `MODQN`, `DQN_throughput`, `DQN_scalar`, `RSS_max`
- weight rows: `3`
- rows used:
  - `1.0/1.0/1.0`
  - `1.0/0.0/0.0`
  - `0.0/1.0/0.0`
- training episodes per learned method run: `20`
- reference run: `artifacts/scenario-corrected-pilot-01/`

## Key Findings

- The bounded follow-on `Table II` pipeline completed successfully and
  emitted machine-readable outputs plus analysis.
- The surface remains near-tied even after the paper-backed scenario
  correction:
  - exact tie rows: `2/3`
  - max scalar spread across methods: `0.0140`
  - max `r1` spread across methods: `0.0000`
  - max `r3` spread across methods: `0.0000`
  - max `r2` spread across methods: `0.0140`
- On `1.0/0.0/0.0`, all four methods tie exactly on scalar reward.
- On `1.0/1.0/1.0`, `MODQN`, `DQN_throughput`, and `DQN_scalar` tie-best,
  with `RSS_max` slightly behind only through `r2` / handover.
- On `0.0/1.0/0.0`, `MODQN` is best at `-0.4690`, but the margin over
  `DQN_scalar` is only `0.0010`.

## Interpretation

- The paper-backed scenario surface does not materially change the
  frozen baseline diagnosis for these three rows.
- Learned methods still collapse to the same `r1` and `r3` values.
- The only cross-method separation remains `r2` / handover count.
- This bounded follow-on run does not justify promoting directly to a
  full expensive follow-on sweep family.

## Comparison Against The Frozen Baseline

- The frozen baseline `Table II` reviews already showed near-ties at both
  `50` and `200` episodes.
- This bounded follow-on result is consistent with that conclusion rather
  than overturning it.
- The main difference is interpretive: the collapse now persists even on
  the more paper-backed geography and mobility surface.

## Limitation

- This is a bounded preview, not the full Phase 01B Gate 2 `Table II`
  closeout artifact.
- Only three weight rows were exercised, so this run is suitable for
  triage and prioritization, not for closing the full follow-on claim.
