# Table II 200-Episode Review

## Run Shape

- Artifact directory: `artifacts/table-ii-200ep-01/`
- Suite: `Table II`
- Methods: `MODQN`, `DQN_throughput`, `DQN_scalar`, `RSS_max`
- Weight rows: `11`
- Training episodes per learned method run: `200`

## High-Signal Findings

- The full `11`-row `Table II` run completed successfully and emitted:
  - `figures/table-ii.csv`
  - `figures/table-ii.png`
  - `evaluation/table-ii.json`
  - `manifest.json`
- Method separation remains very small even at `200` episodes.
- `MODQN` is best or tied-best on `10/11` rows.
- `DQN_scalar` is best on `0.3/0.5/0.2`, but only by `0.0010` over `MODQN`.
- On the handover-only row `0.0/1.0/0.0`, `MODQN` and `DQN_scalar` tie at `-0.4190`, while `DQN_throughput` is slightly worse at `-0.4220`.
- `RSS_max` is consistently close but slightly behind on mixed rows.

## Comparison Against `50` Episodes

- Going from `50` to `200` episodes does not create strong method separation.
- The largest meaningful change is `DQN_scalar` on `0.0/1.0/0.0`, improving from `-0.9170` to `-0.4190`.
- `MODQN` improves slightly on several rows, but mostly at the `1e-3` to `1e-2` level.
- `DQN_throughput` and `RSS_max` are effectively unchanged between the `50`-episode and `200`-episode runs.

## Interpretation

- The comparator/sweep/export pipeline is now real and stable enough for continued work.
- The current reproduction still shows a collapse of method separation under this reward geometry.
- That behavior is consistent with the already-known reward dominance finding:
  - throughput dominates scale
  - several methods collapse toward nearly identical greedy policies

## Recommended Next Step

- Do not jump to `Fig. 3` to `Fig. 6` yet.
- First do a focused analysis pass on `Table II` outputs plus existing long-run artifacts:
  - quantify method deltas row-by-row
  - connect the near-tie behavior to reward dominance
  - decide whether reward calibration / normalization must be introduced as a new explicit experiment surface
