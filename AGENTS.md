# modqn-paper-reproduction Agent Rules

This repository is intended to remain portable as a standalone training surface for
`PAP-2024-MORL-MULTIBEAM`.

## Scope

1. Work only inside this repository unless the user explicitly asks for external comparison.
2. Treat this repo as self-contained for training, smoke validation, and artifact review.
3. Do not assume a parent `papers/` workspace exists.

## Authority Order

When details conflict, use this order:

1. `paper-source/ref/2024_09_Handover_for_Multi-Beam_LEO_Satellite_Networks_A_Multi-Objective_Reinforcement_Learning_Method.pdf`
2. `paper-source/txt_layout/2024_09_Handover_for_Multi-Beam_LEO_Satellite_Networks_A_Multi-Objective_Reinforcement_Learning_Method.layout.txt`
3. `paper-source/catalog/PAP-2024-MORL-MULTIBEAM.json`
4. `docs/phases/phase-01-python-baseline-reproduction-sdd.md`
5. `docs/assumptions/modqn-reproduction-assumption-register.md`
6. `configs/modqn-paper-baseline.resolved-template.yaml`

## Training Rules

1. Only resolved-run configs may start training.
2. `configs/modqn-paper-baseline.yaml` is authority-only and must not be used as a training input.
3. Use the repo-local virtual environment when running tests or training: `.venv/`.
4. Keep active runtime assumptions visible in config, metadata, or both. Do not hide them in code-only defaults.

## Artifact Rules

1. Write training artifacts under `artifacts/`.
2. Keep `run_metadata.json`, `training_log.json`, and final checkpoints together per run.
3. Generated artifacts are outputs, not authority surfaces.

## Prompt Portability

1. Prompts for this repo should use repo-relative paths.
2. Do not reference `/home/u24/papers/...` paths unless the user explicitly provides that workspace.

## Read First

For a quick project-state handoff, read these before proposing new work:

1. `artifacts/public-summary-2026-04-13.md`
2. `artifacts/reproduction-status-2026-04-13.md`
3. `artifacts/run-9000/anomaly-review.md`
4. `artifacts/fig-3-pilot-01/review.md`
5. `artifacts/fig-4-pilot-01/review.md`
6. `artifacts/fig-5-pilot-01/review.md`
7. `artifacts/fig-6-pilot-01/review.md`
8. `docs/baseline-acceptance-checklist.md`

## Current State Snapshot

As of `2026-04-13`, the repo is a working standalone baseline reproduction surface,
but it is not yet a fully established paper-faithful reproduction.

What is already true:

1. repo-only authority is stable
2. baseline training, resume, best-eval checkpointing, sweeps, export, and tests run in-repo
3. `Table II` and reviewed `Fig. 3` to `Fig. 6` sweep surfaces exist with machine-readable outputs
4. the comparison-baseline checklist is complete
5. the repo can now be frozen as a disclosed engineering baseline for downstream comparison

What is not yet established:

1. the paper's intended method separation has not been convincingly reproduced
2. `run-9000` shows a real late-training collapse and objective drift
3. `Table II` and `Fig. 3` are still near-tied, with most variation coming from `r2` / handover
4. the explicit reward-calibration pilot did not improve the raw eval surface

## Current Guardrails

1. Do not start a new `9000`-episode long run by default.
2. Do not silently replace the baseline with `configs/modqn-paper-baseline.reward-calibration.resolved.yaml`.
3. Treat the reward-calibration config as an explicit experiment only.
4. Use `docs/baseline-acceptance-checklist.md` as the freeze note for comparison-baseline scope.
5. If no new user direction is given, prefer freeze/disclosure or clearly labeled experiments over more retraining.
