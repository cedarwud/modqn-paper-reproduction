# Phase 04B Slice B — Training Artifact Model Status

**Date:** `2026-04-19`
**SDD:** [`../docs/phases/phase-04b-refactor-training-artifact-model-sdd.md`](../docs/phases/phase-04b-refactor-training-artifact-model-sdd.md)
**Parent kickoff:** [`../docs/phases/phase-04-refactor-contract-spine-sdd.md`](../docs/phases/phase-04-refactor-contract-spine-sdd.md)
**Result:** `PROMOTE` — Slice B is landed as an internal training-artifact model seam. This does not change the external producer contract.

## 1. Summary

Slice B adds an explicit model / serialization seam for the training-side
artifact contract while preserving existing output semantics:

1. `run_metadata.json` now writes through `RunMetadataV1`,
2. `training_log.json` now writes through `TrainingLogRow`,
3. checkpoint payload build/save/load now goes through `CheckpointPayloadV1`,
4. exporter read paths now deserialize `RunMetadataV1` / training-log rows instead of reading raw JSON dicts directly,
5. `tests/test_refactor_golden.py` stays green unchanged.

## 2. Models Landed

The new package `src/modqn_paper_reproduction/artifacts/` now includes:

1. `RunMetadataV1`
2. `TrainingLogRow`
3. `CheckpointPayloadV1`
4. `CheckpointCatalog`
5. `RunArtifactPaths`

Supporting nested blocks were also typed to keep the boundary explicit:

1. `SeedsBlock`
2. `CheckpointRuleV1`
3. `CheckpointFilesV1`
4. `RewardCalibrationV1`
5. `RuntimeEnvironmentV1`
6. `TrainingSummaryV1`
7. `ResumeFromV1`

## 3. Opaque Fields Preserved

The following surfaces remain intentionally opaque pass-through data in
this slice:

1. `resolved_config_snapshot`
2. `resolved_assumptions`
3. `training_experiment`
4. `trainer_config`
5. checkpoint `q_networks` / `target_networks` / `optimizers`
6. checkpoint `evaluation_summary`
7. checkpoint `last_episode_log`

This is intentional Phase 04B scope control rather than an unfinished
field drop.

## 4. Validation

All required validation gates are green:

1. `pytest tests/test_artifacts_models.py -q` → `9 passed`
2. `pytest tests/test_refactor_golden.py -q` → `11 passed`
3. `pytest` → `224 passed`

Observed result: existing `run_metadata.json`, `training_log.json`,
checkpoint payload, and the canonical sample bundle fixture remain
contract-stable under the landed Slice A guardrails.

## 5. Deviations / Preservation Notes

No Phase 04B scope deviation was required.

One preservation detail is now explicit in the model layer:

1. `RunMetadataV1` normalizes opaque JSON-bound blocks into the
   serialized JSON shape, so trainer-config tuples are represented in
   metadata as the same JSON arrays that already existed on disk.

This preserves existing bytes rather than changing them.

## 6. Slice C Readiness

Slice C is now justified.

The training-side artifact contract is explicitly modeled and has its
own read/write seam, so later bundle-layer splitting work can consume
typed training artifacts without reopening the writer shape first.
