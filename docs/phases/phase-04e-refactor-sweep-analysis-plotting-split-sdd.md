# Phase 04E: Refactor Slice E — Sweep / Analysis / Plotting Split SDD

**Status:** Landed execution slice; promoted via `../../artifacts/phase-04e-sweep-analysis-plotting-split-status-2026-04-19.md`.
**Date:** `2026-04-19`
**Parent kickoff:**
[`phase-04-refactor-contract-spine-sdd.md`](./phase-04-refactor-contract-spine-sdd.md)
**Slice D status:**
[`../../artifacts/phase-04d-runtime-spine-split-status-2026-04-19.md`](../../artifacts/phase-04d-runtime-spine-split-status-2026-04-19.md)
**Slice E status:**
[`../../artifacts/phase-04e-sweep-analysis-plotting-split-status-2026-04-19.md`](../../artifacts/phase-04e-sweep-analysis-plotting-split-status-2026-04-19.md)
**Current interpretation:**
[`../../artifacts/phase-04-current-state-2026-04-19.md`](../../artifacts/phase-04-current-state-2026-04-19.md)

## 1. Purpose

Slice E performs the kickoff's `Sweep / Analysis / Plotting Split`
without changing the repo's external producer contract, training
artifact contract, or already-landed internal guardrail slices.

The goal is to separate three responsibilities that are still partially
coupled today:

1. experiment execution in `sweeps.py`,
2. analysis / plotting generation in `export/pipeline.py`,
3. orchestration / CLI dispatch between train, sweep, and export flows.

This slice is still internal hardening only. It does not supersede the
landed Phase 03A / 03B producer authority.

## 2. Scope

### 2.1 In Scope

1. new `analysis/` package for landed analysis / plotting helpers now
   embedded in `export/pipeline.py`
2. splitting `sweeps.py` into experiment-running helpers versus summary
   row / suite-shape helpers where that separation is required by the
   new analysis boundary
3. introducing a minimal `orchestration/` package for explicit
   train/sweep/export dispatch surfaces
4. reducing `export/pipeline.py` to a thin export/analysis orchestrator
   or retiring it if only trivial pass-through logic remains
5. preserving current CLI and script entrypoints while moving internal
   implementation behind clearer module seams
6. the smallest required call-site updates in:
   - `src/modqn_paper_reproduction/cli.py`
   - `src/modqn_paper_reproduction/sweeps.py`
   - `src/modqn_paper_reproduction/export/pipeline.py`
   - `scripts/run_sweeps.py`
   - `scripts/export_ntn_sim_core_bundle.py`
   - `scripts/train_modqn.py`

### 2.2 Explicitly Out Of Scope

1. any Phase 03A / 03B bundle schema change
2. any `run_metadata.json`, `training_log.json`, or checkpoint envelope
   contract change
3. any new `policyDiagnostics` fields or validation rule changes
4. any change to figure/table artifact semantics, file naming, or
   output directory structure
5. any consumer-side changes in `ntn-sim-core`
6. any trainer-loop, environment, reward, or sweep-value semantics
   change
7. any new analysis family beyond the currently landed `Table II`,
   `Fig. 3` to `Fig. 6`, and `reward-geometry` surfaces

## 3. Module / Function Plan

### 3.1 New Modules

1. `analysis/table_ii.py`
   - `build_table_ii_analysis_frames()`
   - `write_table_ii_analysis_markdown()`
   - `export_table_ii_results()`
2. `analysis/reward_geometry.py`
   - `collect_reward_diagnostics()`
   - `build_reward_geometry_scale_table()`
   - `build_reward_geometry_table_ii_frames()`
   - `export_reward_geometry_analysis()`
3. `analysis/figures.py`
   - figure-suite plotting / markdown export helpers for `fig-3` to
     `fig-6`
   - `export_figure_sweep_results()`
4. `analysis/training_log.py`
   - `window_means()`
   - `summarize_training_log()`
5. `orchestration/train_main.py`
   - train-flow entry dispatch extracted from CLI glue
6. `orchestration/sweep_main.py`
   - sweep-flow entry dispatch extracted from CLI glue
7. `orchestration/export_main.py`
   - export-flow entry dispatch extracted from CLI glue

### 3.2 Intended Survivors

1. `sweeps.py`
   - remains the experiment/suite runner surface
   - should no longer own analysis markdown, plot rendering, or
     artifact-aggregation logic that belongs in `analysis/`
2. `export/pipeline.py`
   - should become a thin orchestrator over `analysis/*` and
     replay-bundle export surfaces
   - may remain as a stable façade if downstream imports already assume
     its presence
3. CLI / scripts
   - continue exposing the same entrypoints during this slice

### 3.3 Slice-E-Specific Extraction Rules

1. move analysis / plotting code first, not experiment semantics
2. preserve the current suite names, file names, and directory layout
   under `artifacts/*`
3. if a helper is used by both `sweeps.py` and `analysis/*`, prefer
   extracting a neutral helper module rather than duplicating logic
4. if `export/pipeline.py` still carries meaningful orchestration
   responsibility after the split, keep it as a façade rather than
   removing it for cleanliness

## 4. Byte-Stable / Semantically Stable Surfaces Slice E Must Preserve

Slice E must leave these surfaces unchanged in key names, value
semantics, and serialized content under the existing tests:

1. Phase 03A `manifest.json` required fields and identity invariants
2. Phase 03A `timeline/step-trace.jsonl` required row fields
3. Phase 03A `config-resolved.json`, `provenance-map.json`,
   `assumptions.json` presence and shape
4. Phase 03B additive `policyDiagnostics` row object
5. Phase 03B `manifest.optionalPolicyDiagnostics`
6. Phase 04B training-artifact model / serialization seam
7. Phase 04C bundle-layer split and `ReplaySummary` single-source
   contract
8. Phase 04D runtime seam / façade preservation
9. current landed analysis artifact semantics, including:
   - `table-ii-results.csv`
   - `table-ii-analysis.md`
   - `table-ii.png`
   - `reward-geometry` JSON / CSV / markdown surfaces
   - `fig-3` to `fig-6` summary CSV / markdown / PNG surfaces

## 5. Validation Set

Slice E lands only if all of the following are green:

1. `pytest tests/test_modqn_smoke.py -q`
2. `pytest tests/test_sweeps_and_export.py -q`
3. `pytest tests/test_refactor_golden.py -q`
4. `pytest tests/test_artifacts_models.py -q`
5. `pytest tests/test_replay_bundle.py -q`
6. `pytest`

Required semantic checks within that set:

1. `Table II` machine-readable outputs remain stable
2. `reward-geometry` re-scoring outputs remain stable
3. `Fig. 3` to `Fig. 6` suite outputs remain stable
4. replay-bundle export outputs introduced in earlier slices remain
   unchanged
5. CLI / script entrypoints still dispatch to the same effective work

## 6. Stop Conditions

Slice E must stop and roll back to the pre-split layout if any of these
becomes true:

1. splitting analysis / plotting changes the content of a landed
   analysis artifact under `artifacts/*`
2. separating orchestration requires changing CLI argument semantics or
   weakening resolved-run config guardrails
3. the split starts drifting into trainer/runtime/evaluation semantics
   rather than file-structure cleanup
4. `export/pipeline.py` removal would break a stable in-repo import
   surface that is still actively used
5. the work starts drifting into a new Phase 04F-style cleanup without
   an explicit execution SDD

## 7. Rollback Plan

If Slice E fails one of the stop conditions:

1. restore the affected analysis / plotting implementation back into
   `export/pipeline.py` or `sweeps.py`
2. keep only those extracted helpers that already have stable live
   callers and do not introduce duplicate authority
3. preserve the landed Slice B artifact seam, Slice C bundle seam, and
   Slice D runtime seam as-is
4. record the failure in a bounded status note instead of widening scope
   into deeper cleanup

## 8. Expected Output Of This Slice

When landed, Slice E should leave the repo with:

1. a dedicated `analysis/` package for Table II, figure-suite, and
   reward-geometry result generation
2. a clearer separation between experiment execution in `sweeps.py` and
   artifact aggregation / plotting in `analysis/*`
3. explicit orchestration modules for train/sweep/export dispatch
4. a thinner `export/pipeline.py` façade or a documented decision to
   keep it as a stable orchestrator
5. no change to the frozen Phase 03A / 03B producer contract
