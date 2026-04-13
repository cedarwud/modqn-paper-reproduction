# Fig. 3 Pilot Review

## Scope

- suite: `fig-3`
- training episodes per learned method: `50`
- sweep points: `40, 60, 80, 100, 120, 140, 160, 180, 200` users
- baseline weight row: `[0.5, 0.3, 0.2]`
- reference long run: `artifacts/run-9000`

## Key Findings

- The weighted-reward trend is monotonic with user count, but method separation remains very small across the full sweep.
- `5/9` sweep points are exact ties on weighted reward, and the maximum cross-method spread is only `0.1500`.
- The average weighted reward across all 9 points stays tightly clustered:
  - `DQN_throughput`: `56.032216`
  - `DQN_scalar`: `56.030341`
  - `RSS_max`: `56.023326`
  - `MODQN`: `56.017167`
- The figure surface reproduces the same structural result seen in `Table II`: method differences mostly come from `r2` / handover, while `r1` and `r3` are nearly identical at each sweep point.
- The most visible local anomaly is at `200` users. `MODQN` keeps the same `r1` and `r3` as the other learned methods, but its `mean_r2` drops to `-0.929` and `mean_total_handovers` rises to `371.6`, versus roughly `171.6` for `DQN_scalar` and `DQN_throughput`. That single point creates the largest spread in the sweep.

## Interpretation

- This pilot does not show a strong Fig. 3 regime where user-count scaling naturally separates methods on the baseline raw reward surface.
- Instead, it strengthens the current repo-local diagnosis:
  - throughput still dominates scalarized reward
  - near-ties persist even when moving from `Table II` to a user-count sweep
  - when a difference appears, it is primarily a switching-cost / handover effect rather than a broad multi-objective separation
- The long-run anomaly in `artifacts/run-9000` remains relevant context, not a contradicted result.

## Recommendation

- Do not treat this pilot as evidence of paper-faithful method separation.
- Do not start `Fig. 4-6` full pilots yet.
- The next useful step is to fold this `Fig. 3` result into the existing reproduction-status note and decide whether the project goal is:
  - an honest disclosed baseline bundle, or
  - a new explicitly experimental track aimed at resolving the near-tie / objective-dominance issue.
