# Phase 04D: Refactor Slice D — Runtime Spine Split SDD

**Status:** Drafted execution SDD; current repo-state interpretation remains in-flight.
**Date:** `2026-04-19`
**Parent kickoff:**
[`phase-04-refactor-contract-spine-sdd.md`](./phase-04-refactor-contract-spine-sdd.md)
**Slice C status:**
[`../../artifacts/phase-04c-bundle-layer-split-status-2026-04-19.md`](../../artifacts/phase-04c-bundle-layer-split-status-2026-04-19.md)
**Current interpretation:**
[`../../artifacts/phase-04-current-state-2026-04-19.md`](../../artifacts/phase-04-current-state-2026-04-19.md)

## 1. Purpose

Slice D performs the kickoff's `Runtime Spine Split (Facade-Preserving)`
without changing the repo's external producer contract.

The goal is to reduce `algorithms/modqn.py` from a runtime utility hub
back to a trainer implementation while keeping existing imports stable:

1. move trainer-adjacent runtime types and helpers into a dedicated
   `runtime/` package,
2. reverse the `config_loader.py -> algorithms.modqn.TrainerConfig`
   dependency direction so runtime types no longer live under the
   trainer façade,
3. preserve current checkpoint bytes, replay timeline bytes, and all
   landed Phase 03A / 03B / 04A / 04B / 04C guarantees,
4. keep `algorithms/modqn.py` and `algorithms/__init__.py` as stable
   facades for the duration of this slice.

This is still internal hardening only. It does not supersede the landed
Phase 03A / 03B producer authority.

## 2. Scope

### 2.1 In Scope

1. new package `src/modqn_paper_reproduction/runtime/`
2. extracting runtime types and helpers out of
   `src/modqn_paper_reproduction/algorithms/modqn.py`
3. reversing the dependency so `config_loader.py` imports
   `TrainerConfig` from `runtime/`, not from `algorithms.modqn`
4. preserving `algorithms/modqn.py` as a compatibility façade that
   re-exports the current public surface
5. the smallest required call-site updates in:
   - `src/modqn_paper_reproduction/config_loader.py`
   - `src/modqn_paper_reproduction/sweeps.py`
   - `src/modqn_paper_reproduction/algorithms/__init__.py`
   - `src/modqn_paper_reproduction/bundle/serializers.py`
   - any tests that need import-path updates only if façade preservation
     is insufficient

### 2.2 Explicitly Out Of Scope

1. any Phase 03A / 03B bundle schema change
2. any `run_metadata.json`, `training_log.json`, or checkpoint envelope
   contract change
3. any new `policyDiagnostics` fields or validation rule changes
4. any consumer-side changes in `ntn-sim-core`
5. any analysis / plotting / sweep cleanup beyond the minimum runtime
   dependency split
6. removing the `algorithms/modqn.py` façade in this slice

## 3. Module / Type / Function Plan

### 3.1 New Modules

1. `runtime/trainer_spec.py`
   - `TrainerConfig`
   - `EpisodeLog`
   - `EvalSummary`
2. `runtime/state_encoding.py`
   - `encode_state()`
   - `state_dim_for()`
3. `runtime/objective_math.py`
   - `scalarize_objectives()`
   - `apply_reward_calibration()`
4. `runtime/replay_buffer.py`
   - `ReplayBuffer`
5. `runtime/q_network.py`
   - `DQNNetwork`
6. `runtime/checkpoint_payload.py`
   - `build_checkpoint_payload()`
   - `load_checkpoint_payload()`
7. `runtime/evaluation.py`
   - `PolicyEvaluator`
   - `evaluate_one_seed()`
   - `evaluate_policy()`
8. `runtime/replay_runner.py`
   - `GreedyReplayRunner`

### 3.2 Preserved Facade

`algorithms/modqn.py` remains the producer-facing façade for:

1. `TrainerConfig`
2. `DQNNetwork`
3. `ReplayBuffer`
4. `encode_state()`
5. `state_dim_for()`
6. `scalarize_objectives()`
7. `apply_reward_calibration()`
8. `EpisodeLog`
9. `EvalSummary`
10. `MODQNTrainer`

`algorithms/__init__.py` must continue to export the same names during
this slice.

### 3.3 Slice-D-Specific Extraction Rules

1. `MODQNTrainer` stays in `algorithms/modqn.py`; the slice is about
   shrinking its dependencies, not moving the trainer class out yet.
2. The trainer may call into `runtime/*`, but public imports should not
   need to follow those internal moves.
3. `bundle/serializers.py` may continue importing
   `scalarize_objectives()` through the façade or directly from
   `runtime/objective_math.py`, but the choice must not change output.

## 4. Byte-Stable Surfaces Slice D Must Preserve

Slice D must leave these surfaces unchanged in key names, value
semantics, and serialized content under the existing tests:

1. Phase 03A `manifest.json` required fields and identity invariants
2. Phase 03A `timeline/step-trace.jsonl` required row fields
3. Phase 03A `config-resolved.json`, `provenance-map.json`,
   `assumptions.json` presence and shape
4. Phase 03B additive `policyDiagnostics` row object
5. Phase 03B `manifest.optionalPolicyDiagnostics`
6. Phase 04B training-artifact model / serialization seam
7. checkpoint payload `format_version == 1` envelope and semantics
8. `tests/fixtures/sample-bundle-v1/` as the canonical downstream
   sample bundle fixture

## 5. Validation Set

Slice D lands only if all of the following are green:

1. `pytest tests/test_modqn_smoke.py -q`
2. `pytest tests/test_sweeps_and_export.py -q`
3. `pytest tests/test_refactor_golden.py -q`
4. `pytest tests/test_artifacts_models.py -q`
5. `pytest tests/test_replay_bundle.py -q`
6. `pytest`

Required semantic checks within that set:

1. `TrainerConfig` field coverage remains aligned with the artifact
   `trainer_config` block
2. checkpoint payload round-trip remains stable
3. exported replay timeline remains byte/semantic stable
4. bundle-layer outputs introduced in Slice C remain unchanged

## 6. Stop Conditions

Slice D must stop and roll back to the pre-split runtime layout if any
of these becomes true:

1. extracting runtime helpers changes checkpoint byte content
2. extracting runtime helpers changes replay timeline byte content
3. the dependency reversal requires changing resolved-run config
   semantics or relaxing `config_loader.py` guardrails
4. the façade can no longer preserve current public imports from
   `algorithms/modqn.py` / `algorithms/__init__.py`
5. the work starts drifting into Slice E analysis / plotting cleanup

## 7. Rollback Plan

If Slice D fails one of the stop conditions:

1. restore the affected implementation back into
   `algorithms/modqn.py`
2. keep only those `runtime/*` modules that already have stable live
   callers and do not introduce duplicate authority
3. preserve the landed Slice B artifact seam and Slice C bundle seam as-is
4. record the failure in a bounded status note instead of widening scope
   into Slice E

## 8. Expected Output Of This Slice

When landed, Slice D should leave the repo with:

1. a dedicated `runtime/` package for trainer-adjacent types and
   helpers
2. `config_loader.py` depending on `runtime/trainer_spec.py`, not on
   `algorithms/modqn.py`
3. a smaller `algorithms/modqn.py` that remains a stable façade
4. no change to the frozen Phase 03A / 03B producer contract
