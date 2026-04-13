# Fig. 6 Pilot Review

## Scope

- suite: `fig-6`
- training episodes per learned method: `50`
- sweep points: `7.0, 7.2, 7.4, 7.6, 7.8` km/s
- baseline weight row: `[0.5, 0.3, 0.2]`
- reference long run: `artifacts/run-9000`

## Key Findings

- The full `Fig. 6` pilot completed successfully and emitted:
  - `evaluation/fig-6.json`
  - `evaluation/fig-6-detail.csv`
  - `evaluation/fig-6-weighted-reward.csv`
  - `evaluation/fig-6-weighted-winners.csv`
  - `figures/fig-6-objectives.png`
  - `figures/fig-6-weighted-reward.png`
  - `analysis/fig-6-analysis.md`
  - `manifest.json`
- `5/5` sweep points are exact ties on weighted reward between `MODQN` and `DQN_throughput`.
- The maximum cross-method weighted-reward spread is only `0.0120`.
- The weighted-reward trend shifts slightly with satellite speed:
  - `MODQN` / `DQN_throughput`: `52.27521317253105 -> 52.24190558738703`
  - `DQN_scalar` / `RSS_max`: `52.26321317253104 -> 52.22990558738703`
- The raw objective surface also changes only slightly and preserves the same ordering:
  - `mean_r1` and `mean_r3` drift smoothly with speed
  - `mean_r2` and handovers remain unchanged by method family
- The method split again stays local to `r2` / handover:
  - `MODQN` / `DQN_throughput`: `mean_r2 = -0.422`, handovers `84.4`
  - `DQN_scalar` / `RSS_max`: `mean_r2 = -0.462`, handovers `92.4`

## Interpretation

- This pilot does not reveal a satellite-speed regime where the baseline reward surface separates methods in a paper-like way.
- It extends the same repo-local diagnosis seen in `Table II`, `Fig. 3`, `Fig. 4`, and `Fig. 5`:
  - near-ties persist
  - the dominant visible difference remains handover-related
  - broader multi-objective separation is still not established

## Recommendation

- Treat this pilot as valid comparison-baseline evidence, not as evidence of paper-faithful method separation.
- Use it to close the `Fig. 6` checklist gap for the disclosed comparison-baseline bundle.
- If more work continues after baseline freeze, it should be opened as a new explicitly experimental track.
