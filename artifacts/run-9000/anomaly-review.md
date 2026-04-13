# Run 9000 Anomaly Review

Date: `2026-04-12`  
Scope: repo-only review based on:

- `artifacts/run-9000/run_metadata.json`
- `artifacts/run-9000/training_log.json`
- `artifacts/run-9000/console.log`
- comparison against `artifacts/final-smoke/` and `artifacts/pilot-01/`

## Summary

The `run-9000` artifact is usable and worth analyzing. It should not be discarded as a portability casualty of the repo-only hardening pass.

The main anomaly is a late-training policy collapse:

- early training is stable and high-performing
- the best observed scalar-reward regime appears in the mid-run, not at the end
- degradation begins around episode `5700`
- the final checkpoint at episode `8999` is materially worse than the earlier observed regime

This supports `artifact review first`, not immediate retraining.

## Key Findings

1. Smoke and pilot runs do not show the long-run failure mode.
   - `final-smoke`: scalar `599.6 -> 608.7`
   - `pilot-01`: scalar `599.6 -> 603.9`
   - `run-9000`: scalar `599.6 -> 85.6`

2. The best observed run behavior is mid-training, not final-training.
   - best scalar episode: `5147`
   - best scalar metrics:
     - scalar `618.2668`
     - `r1` `1250.4682`
     - `r2` `-2.3950`
     - `r3` `-31.2441`
     - handovers `479`

3. The final checkpoint is not representative of the best observed policy.
   - final episode: `8999`
   - final metrics:
     - scalar `85.6352`
     - `r1` `243.0597`
     - `r2` `-0.4450`
     - `r3` `-178.8056`
     - handovers `89`

4. The collapse is not just a final-episode blip.
   - first `500` episodes average:
     - scalar `605.8840`
     - `r1` `1221.7224`
     - `r2` `-4.2764`
     - `r3` `-18.4714`
     - handovers `855.2760`
   - last `500` episodes average:
     - scalar `120.0114`
     - `r1` `313.0075`
     - `r2` `-0.5719`
     - `r3` `-181.6041`
     - handovers `114.3700`

5. The degradation starts before epsilon reaches its floor.
   - 100-episode moving average scalar first drops below:
     - `550` at episode `5747`
     - `500` at episode `6095`
     - `400` at episode `6480`
     - `300` at episode `6706`
     - `200` at episode `6852`
   - epsilon reaches its floor only around episode `7000`

6. The logged reward-dominance warning remains valid.
   - `console.log` reports:
     - `|r1| / |r2| ≈ 982.6x`
     - `|r1| / |r3| ≈ 28x`
   - weighted average contributions show scalarization is still throughput-led:
     - first `500`: `0.5*r1 ≈ 610.86`, `0.3*r2 ≈ -1.28`, `0.2*r3 ≈ -3.69`
     - last `500`: `0.5*r1 ≈ 156.50`, `0.3*r2 ≈ -0.17`, `0.2*r3 ≈ -36.32`

## Windowed View

| Window | Scalar | r1 | r2 | r3 | Handovers |
|---|---:|---:|---:|---:|---:|
| first `500` | `605.8840` | `1221.7224` | `-4.2764` | `-18.4714` | `855.2760` |
| mid `4250:4750` | `600.3734` | `1216.6015` | `-2.7415` | `-35.5244` | `548.3060` |
| best window `4700:5200` | `594.1730` | `1208.1966` | `-2.4194` | `-45.9976` | `483.8720` |
| onset window `5700:6200` | `530.3933` | `1109.2552` | `-1.5802` | `-118.8013` | `316.0400` |
| late window `6500:7000` | `288.2347` | `652.5086` | `-0.8171` | `-188.8720` | `163.4220` |
| last `500` | `120.0114` | `313.0075` | `-0.5719` | `-181.6041` | `114.3700` |

Interpretation:

- handovers fall steadily
- throughput eventually collapses
- fairness/load-balance penalty becomes much more negative
- the learned policy appears to become too sticky: fewer handovers, but much worse service quality and imbalance

## Artifact-Level Implications

1. There is no evidence that repo-only hardening invalidated the old long-run conclusion.
2. There is no current need to rerun smoke, pilot, or long training just to re-establish trust in the artifacts.
3. The main missing artifact feature is checkpoint selection, not more raw training time.
4. `ASSUME-MODQN-REP-015` is still only partially implemented:
   - `primary_report = final-episode-policy`
   - `secondary_report = best-weighted-reward-on-eval`
   - `secondary_best_eval = null`

## Recommended Next Step

Do not retrain yet.

Use this artifact set to drive the next engineering change:

- add best-checkpoint capture or explicit evaluation-based checkpoint selection
- preserve final-checkpoint reporting as metadata, but stop treating final episode as the only analysis surface

Only after that change lands should a new long run be considered.
