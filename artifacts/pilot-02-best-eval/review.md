# Pilot 02 Review

## Summary

- Run directory: `artifacts/pilot-02-best-eval/`
- Config: `configs/modqn-paper-baseline.resolved-template.yaml`
- Episodes: `200`
- Purpose: validate repo-local best-eval checkpoint capture on a real pilot artifact

## Observed Outputs

- `training_log.json` present
- `run_metadata.json` present
- `checkpoints/final-episode-policy.pt` present
- `checkpoints/best-weighted-reward-on-eval.pt` present

## Key Facts

- Final training episode: `199`
- Final training scalar reward: `602.8169`
- Best observed training scalar episode: `34`
- Best observed training scalar reward: `616.1214`
- Eval-selected secondary checkpoint episode: `99`
- Eval-selected mean scalar reward: `52.2599`
- Evaluation seed set: `[100, 200, 300, 400, 500]`
- Evaluation cadence: every `50` episodes plus final episode

## Interpretation

- The new `ASSUME-MODQN-REP-015` flow is working end-to-end.
- The pilot artifact now carries both the final checkpoint and a distinct eval-selected checkpoint.
- The eval-selected checkpoint is not the final episode checkpoint, which confirms the selection rule is active rather than degenerate.
- Direct evaluation confirms the secondary checkpoint is slightly better than the final checkpoint on the configured eval seed set:
  - final checkpoint (`episode 199`): mean scalar `52.2578`
  - best-eval checkpoint (`episode 99`): mean scalar `52.2599`
- This pilot is sufficient to validate checkpoint selection plumbing.
- This pilot is not sufficient to resolve the existing long-run reward-dominance and objective-drift findings from `artifacts/run-9000/`.

## Next Step

- Do not rerun the `9000`-episode long run yet.
- Next engineering step should be comparator + sweep/export implementation, or a targeted evaluation workflow that can compare final-vs-best-eval checkpoints on the same artifact.
