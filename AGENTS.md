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
