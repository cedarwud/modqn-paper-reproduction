# Phase 04E Slice E — Sweep / Analysis / Plotting Split Status

**Date:** `2026-04-19`
**SDD:** [`../docs/phases/phase-04e-refactor-sweep-analysis-plotting-split-sdd.md`](../docs/phases/phase-04e-refactor-sweep-analysis-plotting-split-sdd.md)
**Parent kickoff:** [`../docs/phases/phase-04-refactor-contract-spine-sdd.md`](../docs/phases/phase-04-refactor-contract-spine-sdd.md)
**Result:** `PROMOTE` — Slice E is landed as an internal sweep/analysis/plotting split. This does not change the external producer contract.

## 1. Summary

Slice E lands the minimum accepted analysis-boundary split promised by
the Phase 04 kickoff:

1. a new `analysis/` package now owns the landed `Table II`,
   `reward-geometry`, `Fig. 3` to `Fig. 6`, and training-log
   analysis/plotting helpers,
2. `sweeps.py` remains the experiment runner surface but now delegates
   artifact aggregation and plotting to `analysis/*`,
3. `export/pipeline.py` is reduced to a thin façade/orchestrator over
   replay-bundle export plus the extracted analysis helpers,
4. a minimal `orchestration/` package now carries train/sweep/export
   dispatch behind the preserved `cli.py` entrypoints.

## 2. Landed Modules

The new package `src/modqn_paper_reproduction/analysis/` now includes:

1. `table_ii.py`
2. `reward_geometry.py`
3. `figures.py`
4. `training_log.py`
5. `__init__.py`

The new package `src/modqn_paper_reproduction/orchestration/` now
includes:

1. `train_main.py`
2. `sweep_main.py`
3. `export_main.py`
4. `__init__.py`

## 3. Preservation Notes

Slice E intentionally preserves the following:

1. the Phase 03A replay-bundle contract,
2. the Phase 03B additive `policyDiagnostics` surface,
3. the Phase 04B training-artifact serialization seam,
4. the Phase 04C bundle-layer split and `ReplaySummary` single-source
   rule,
5. the Phase 04D runtime façade,
6. CLI/script entrypoint names and argument semantics,
7. the checked-in `tests/fixtures/sample-bundle-v1/` fixture.

No figure/table file names, output directory layout, checkpoint payload
semantics, or bundle field semantics were changed in this slice.

## 4. Validation

All required validation gates are green:

1. `pytest tests/test_modqn_smoke.py -q` → `16 passed`
2. `pytest tests/test_sweeps_and_export.py -q` → `12 passed`
3. `pytest tests/test_refactor_golden.py -q` → `11 passed`
4. `pytest tests/test_artifacts_models.py -q` → `9 passed`
5. `pytest tests/test_replay_bundle.py -q` → `16 passed`
6. `pytest -q` → `224 passed`

Observed result: the extracted analysis/orchestration seams preserve the
existing sweep outputs, training export outputs, replay-bundle outputs,
and CLI behavior under the landed Phase 04A / 04B / 04C / 04D
guardrails.

## 5. Deviations / Scope Control

No Phase 04E scope deviation was required.

The new `orchestration/` package is intentionally minimal: it carries
dispatch only and does not reopen trainer/runtime extraction or any new
Phase 04F-style cleanup.
