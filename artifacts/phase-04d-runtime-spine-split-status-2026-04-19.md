# Phase 04D Slice D — Runtime Spine Split Status

**Date:** `2026-04-19`
**SDD:** [`../docs/phases/phase-04d-refactor-runtime-spine-split-sdd.md`](../docs/phases/phase-04d-refactor-runtime-spine-split-sdd.md)
**Parent kickoff:** [`../docs/phases/phase-04-refactor-contract-spine-sdd.md`](../docs/phases/phase-04-refactor-contract-spine-sdd.md)
**Result:** `PROMOTE` — Slice D is landed as an internal runtime-spine split. This does not change the external producer contract.

## 1. Summary

Slice D extracts the trainer-adjacent runtime seam out of
`algorithms/modqn.py` while preserving the existing public façade:

1. new `runtime/` package for trainer specs, state encoding, objective
   math, replay buffer, and Q-network types,
2. `config_loader.py` now depends on
   `runtime/trainer_spec.py::TrainerConfig` instead of importing
   `algorithms.modqn.TrainerConfig`,
3. `algorithms/modqn.py` remains the stable façade for
   `TrainerConfig`, `EpisodeLog`, `EvalSummary`, `DQNNetwork`,
   `ReplayBuffer`, `encode_state()`, `state_dim_for()`,
   `scalarize_objectives()`, `apply_reward_calibration()`, and
   `MODQNTrainer`,
4. the bundle-layer serializer and sweep helpers now consume the split
   runtime seam without changing output semantics.

## 2. Runtime Modules Landed

The new package `src/modqn_paper_reproduction/runtime/` now includes:

1. `trainer_spec.py`
2. `state_encoding.py`
3. `objective_math.py`
4. `replay_buffer.py`
5. `q_network.py`
6. `__init__.py`

These landed modules are the minimum accepted Slice D runtime seam.
`MODQNTrainer` itself remains in `algorithms/modqn.py` by design.

## 3. Preservation Notes

Slice D intentionally preserves the following:

1. the Phase 03A replay-bundle contract,
2. the Phase 03B additive `policyDiagnostics` surface,
3. the Phase 04B training-artifact serialization seam,
4. checkpoint payload `format_version == 1` envelope semantics,
5. the checked-in `tests/fixtures/sample-bundle-v1/` fixture,
6. the public façade at `algorithms/modqn.py` and `algorithms/__init__.py`.

No sample fixture contract, checkpoint payload semantics, replay
timeline semantics, or producer-facing bundle fields were changed in
this slice.

## 4. Validation

All required validation gates are green:

1. `pytest tests/test_modqn_smoke.py -q` → `16 passed`
2. `pytest tests/test_sweeps_and_export.py -q` → `12 passed`
3. `pytest tests/test_refactor_golden.py -q` → `11 passed`
4. `pytest tests/test_artifacts_models.py -q` → `9 passed`
5. `pytest tests/test_replay_bundle.py -q` → `16 passed`
6. `pytest -q` → `224 passed`

Observed result: the runtime seam split preserves the existing artifact,
checkpoint, bundle, and façade behavior under the landed Phase 04A /
04B / 04C guardrails.

## 5. Deviations / Scope Control

No Phase 04D scope deviation was required.

The optional `checkpoint_payload.py`, `evaluation.py`, and
`replay_runner.py` follow-on extraction points were intentionally left
out of this landed slice to keep the change within the smallest
acceptable Slice D boundary and avoid drifting into Slice E.
