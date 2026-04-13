# Table II vs Long-Run Analysis Note

## Core Observation

The `200`-episode `Table II` run shows almost no method separation.

Across all four methods and all `11` weight rows:

- `mean_r1` is identical at `174.61869176864622`
- `mean_r3` is identical at `-174.61869176864755`
- almost all scalar differences come from `mean_r2`
- `mean_total_handovers` is the same story in count form

For the `Table II` artifact, the spread by weight row is:

- `r1_spread = 0.0`
- `r3_spread = 0.0`
- `r2_spread ≈ 0.043`
- `handover_spread ≈ 8.6` users per evaluation episode
- `scalar_spread` stays tiny, usually `0.0x`

That means the current `Table II` comparison is effectively a handover-count comparison under a fixed throughput/load regime.

## Why This Matters

This is consistent with the older `run-9000` anomaly rather than contradicting it.

In `artifacts/run-9000/training_log.json`:

- first-500 average:
  - scalar `605.8840`
  - `r1 = 1221.7224`
  - `r2 = -4.2764`
  - `r3 = -18.4714`
- last-500 average:
  - scalar `120.0114`
  - `r1 = 313.0075`
  - `r2 = -0.5719`
  - `r3 = -181.6041`

That long-run behavior showed:

- throughput collapsing
- load-balance term worsening sharply
- handover penalty improving

By contrast, the `Table II` eval artifact is happening in a regime where:

- throughput is frozen across methods
- load balance is frozen across methods
- only handover changes enough to move the scalar score

So the current `Table II` near-ties are not evidence that the methods are truly equivalent in all regimes.
They are evidence that, on this evaluation surface, the environment/policy combination is collapsing most of the comparison onto `r2`.

## Practical Conclusion

- More `Table II` training alone is unlikely to create large comparative gaps.
- Expanding directly to `Fig. 3` to `Fig. 6` right now would multiply runs without resolving the main interpretability problem.
- The next useful work item is not a longer sweep family. It is a focused analysis or experiment surface around reward geometry.

## Recommended Next Step

Introduce one explicit analysis/report surface before broader sweeps:

- row-by-row objective decomposition for `Table II`
- method deltas in `r1`, `r2`, `r3`, and handover count
- direct linkage to the known `run-9000` reward-dominance and objective-drift findings

If that analysis confirms the same collapse pattern across more evaluation settings, the next engineering step should be a clearly labeled reward-calibration experiment surface, not a silent change to the baseline.
