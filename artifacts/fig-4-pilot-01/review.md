# Fig. 4 Pilot Review

## Scope

- suite: `fig-4`
- training episodes per learned method: `50`
- sweep points: `2, 3, 4, 5, 6, 7, 8` satellites
- baseline weight row: `[0.5, 0.3, 0.2]`
- reference long run: `artifacts/run-9000`

## Key Findings

- The full `Fig. 4` pilot completed successfully and emitted:
  - `evaluation/fig-4.json`
  - `evaluation/fig-4-detail.csv`
  - `evaluation/fig-4-weighted-reward.csv`
  - `evaluation/fig-4-weighted-winners.csv`
  - `figures/fig-4-objectives.png`
  - `figures/fig-4-weighted-reward.png`
  - `analysis/fig-4-analysis.md`
  - `manifest.json`
- The weighted-reward surface is more collapsed than `Fig. 3`.
- `7/7` sweep points are exact ties on weighted reward between `MODQN` and `DQN_throughput`.
- The maximum cross-method weighted-reward spread is only `0.0120`.
- For every satellite-count point:
  - `MODQN` and `DQN_throughput` tie at `52.259007530593614`
  - `DQN_scalar` and `RSS_max` tie at `52.2470075305936`
- Raw objective means are also invariant across the full sweep:
  - `mean_r1 = 174.61869176864622`
  - `mean_r3 = -174.61869176864755`
  - `MODQN` / `DQN_throughput` `mean_r2 = -0.422`, handovers `84.4`
  - `DQN_scalar` / `RSS_max` `mean_r2 = -0.462`, handovers `92.4`

## Interpretation

- This pilot does not open a paper-like `Fig. 4` regime where changing satellite count separates the methods.
- Under the current per-topology retrain + evaluation protocol, the full satellite-count sweep collapses to the same effective comparison surface at every point.
- The only visible method difference remains `r2` / handover.
- This is stronger evidence for the current repo-local diagnosis, not a contradiction of it.

## Recommendation

- Treat this pilot as valid comparison-baseline evidence, not as evidence of paper-faithful method separation.
- Use it to close the `Fig. 4` checklist gap for the disclosed comparison-baseline bundle.
- If future work continues, it should move to a new explicitly experimental track rather than more baseline retraining by default.
