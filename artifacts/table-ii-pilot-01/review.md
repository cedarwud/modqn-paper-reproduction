# Table II Pilot 01 Review

## Run Shape

- Artifact directory: `artifacts/table-ii-pilot-01/`
- Suite: `Table II`
- Methods: `MODQN`, `DQN_throughput`, `DQN_scalar`, `RSS_max`
- Weight rows: `11`
- Training episodes per learned method run: `50`
- Output files:
  - `figures/table-ii.csv`
  - `figures/table-ii.png`
  - `evaluation/table-ii.json`
  - `manifest.json`

## High-Signal Findings

- The first non-smoke `Table II` pipeline is working end-to-end.
- All four methods emitted machine-readable outputs and a plot.
- `MODQN` is tied or effectively tied for best on `10/11` weight rows in this pilot.
- `DQN_throughput` is best on the handover-only row `0.0/1.0/0.0` with mean scalar `-0.4220`, ahead of `MODQN` / `RSS_max` at `-0.4620` and `DQN_scalar` at `-0.9170`.
- For most mixed rows the margins are tiny, typically at the `1e-3` to `1e-2` level.
- `DQN_scalar` does not yet show a meaningful advantage from per-weight retraining in this `50`-episode pilot.

## Interpretation

- This pilot validates the comparator + sweep + export plumbing.
- It does **not** yet validate paper-style comparative claims.
- The near-ties are consistent with short training plus the already-known reward dominance issue, where throughput strongly constrains the learned behavior.
- A larger `Table II` run is needed before claiming meaningful separation between methods.

## Suggested Next Step

- Keep `Fig. 3` to `Fig. 6` pending for now.
- Run one larger `Table II` pass before expanding the sweep family:
  - same `11` weight rows
  - same four methods
  - increase learned-method training to `200` episodes
- After that, inspect whether method separation emerges. If the rows still collapse to near-ties, the next work item should be reward-calibration analysis rather than broader figure sweeps.
