# Fig. 5 Pilot Review

## Scope

- suite: `fig-5`
- training episodes per learned method: `50`
- sweep points: `30, 60, 90, 120, 150` km/h
- baseline weight row: `[0.5, 0.3, 0.2]`
- reference long run: `artifacts/run-9000`

## Key Findings

- The full `Fig. 5` pilot completed successfully and emitted:
  - `evaluation/fig-5.json`
  - `evaluation/fig-5-detail.csv`
  - `evaluation/fig-5-weighted-reward.csv`
  - `evaluation/fig-5-weighted-winners.csv`
  - `figures/fig-5-objectives.png`
  - `figures/fig-5-weighted-reward.png`
  - `analysis/fig-5-analysis.md`
  - `manifest.json`
- `5/5` sweep points are exact ties on weighted reward between `MODQN` and `DQN_throughput`.
- The maximum cross-method weighted-reward spread is only `0.0120`.
- The weighted-reward trend changes only slightly with speed:
  - `MODQN` / `DQN_throughput`: `52.259007530593614 -> 52.258966605949425`
  - `DQN_scalar` / `RSS_max`: `52.2470075305936 -> 52.24696660594941`
- The raw objective surface drifts only minimally:
  - `mean_r1` and `mean_r3` move by about `1e-4`
  - `mean_r2` and handovers remain unchanged by method family
- The method split is the same at every point:
  - `MODQN` / `DQN_throughput`: `mean_r2 = -0.422`, handovers `84.4`
  - `DQN_scalar` / `RSS_max`: `mean_r2 = -0.462`, handovers `92.4`

## Interpretation

- This pilot does not show a user-speed regime that meaningfully separates the learned methods on the baseline raw reward surface.
- The sweep strengthens the existing diagnosis:
  - near-ties persist
  - the main visible difference remains switching-cost / handover behavior
  - throughput and load-balance still move almost identically across methods

## Recommendation

- Treat this pilot as valid comparison-baseline evidence, not as evidence of paper-faithful method separation.
- Use it to close the `Fig. 5` checklist gap for the disclosed comparison-baseline bundle.
- Keep any attempt to break this near-tie structure in a separately labeled experimental track.
