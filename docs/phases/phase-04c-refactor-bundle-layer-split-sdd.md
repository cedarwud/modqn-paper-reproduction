# Phase 04C: Refactor Slice C — Bundle Contract Layer Split SDD

**Status:** Execution SDD for the landed Slice C bundle-layer split.
**Date:** `2026-04-19`
**Parent kickoff:**
[`phase-04-refactor-contract-spine-sdd.md`](./phase-04-refactor-contract-spine-sdd.md)
**Slice B status:**
[`../../artifacts/phase-04b-training-artifact-model-status-2026-04-19.md`](../../artifacts/phase-04b-training-artifact-model-status-2026-04-19.md)
**Current interpretation:**
[`../../artifacts/phase-04-current-state-2026-04-19.md`](../../artifacts/phase-04-current-state-2026-04-19.md)

## 1. Purpose

Slice C performs the smallest acceptable `Bundle Contract Layer Split`
promised by the kickoff:

1. reduce `export/replay_bundle.py` from an all-in-one exporter into a
   thin orchestration / façade surface,
2. split bundle schema, serialization, provenance, validation, fixture
   tools, and training-artifact compat lookup into independently
   evolvable modules,
3. keep the frozen Phase 03A / 03B producer contract byte-stable,
4. replace the current hand-maintained duplicate replay-summary dicts
   with one explicit `ReplaySummary` source object.

This slice is internal hardening only. It does not supersede the landed
Phase 03A / 03B producer authority.

## 2. Scope

### 2.1 In Scope

1. new package `src/modqn_paper_reproduction/bundle/` containing:
   - `schema.py`
   - `models.py`
   - `serializers.py`
   - `provenance.py`
   - `validator.py`
   - `fixture_tools.py`
2. new module `src/modqn_paper_reproduction/artifacts/compat.py`
3. keeping `src/modqn_paper_reproduction/export/replay_bundle.py` as the
   public exporter façade while moving internal responsibilities behind
   it,
4. one explicit `ReplaySummary` model used for:
   - `manifest.replaySummary`
   - `evaluation.summary.replay_timeline`
   - trimmed-fixture replay-summary synchronization
5. the smallest required call-site updates in:
   - `src/modqn_paper_reproduction/export/pipeline.py`
   - `scripts/generate_sample_bundle.py`

### 2.2 Explicitly Out Of Scope

1. moving `TrainerConfig` out of `algorithms/modqn.py`
2. runtime replay-runner extraction
3. checkpoint payload or `run_metadata.json` contract changes
4. plot / sweep / analysis cleanup beyond the replay-summary call-site
5. consumer-side changes in `ntn-sim-core`

## 3. Module / Type / Function Plan

### 3.1 New Modules

1. `bundle/schema.py`
   - `BUNDLE_SCHEMA_VERSION`
   - `TIMELINE_FORMAT_VERSION`
   - `BEAM_CATALOG_ORDER`
   - `POLICY_DIAGNOSTICS_*` constants
   - required manifest / timeline / diagnostics field definitions
2. `bundle/models.py`
   - `ReplaySummary`
   - `ReplaySampleSubset`
3. `bundle/serializers.py`
   - runtime-to-row serialization helpers
   - `serialize_policy_diagnostics()`
   - `serialize_satellite_states()`
   - `serialize_beam_states()`
   - `serialize_timeline_row()`
   - `build_replay_summary()`
   - `build_optional_policy_diagnostics_manifest()`
4. `bundle/provenance.py`
   - `build_provenance_map()`
5. `bundle/validator.py`
   - `validate_replay_bundle()`
   - `_validate_policy_diagnostics_row()`
   - `_validate_optional_policy_diagnostics_manifest()`
6. `bundle/fixture_tools.py`
   - `trim_replay_bundle_for_sample()`
   - `sync_replay_summary_in_evaluation_summary()`
7. `artifacts/compat.py`
   - `resolve_training_config_snapshot()`
   - `select_replay_checkpoint()`
   - `_select_timeline_seed()`
   - `_resolve_existing_path()`

### 3.2 Preserved Facade

`export/replay_bundle.py` stays as the producer-facing import surface
for:

1. `export_replay_bundle()`
2. `validate_replay_bundle()`
3. `trim_replay_bundle_for_sample()`
4. existing exported bundle constants used by tests / scripts

The façade may re-export helpers moved to the new modules where keeping
that import path avoids unnecessary churn.

### 3.3 ReplaySummary Single-Source Rule

Slice C introduces one `ReplaySummary` object as the source for both
cross-file surfaces:

1. `manifest.replaySummary`
2. `evaluation.summary.replay_timeline`

No Slice C code path may maintain those as separately hand-written dicts
after the split.

## 4. Byte-Stable Surfaces Slice C Must Preserve

Slice C must leave these surfaces unchanged in key names, value
semantics, and serialized content under the existing tests:

1. Phase 03A `manifest.json` required fields and identity invariants
2. Phase 03A `timeline/step-trace.jsonl` required row fields
3. Phase 03A `config-resolved.json`, `provenance-map.json`,
   `assumptions.json` presence and shape
4. Phase 03B additive `policyDiagnostics` row object
5. Phase 03B `manifest.optionalPolicyDiagnostics`
6. checkpoint payload `format_version == 1` envelope
7. `tests/fixtures/sample-bundle-v1/` as the canonical downstream sample
   bundle fixture

## 5. Validation Set

Slice C lands only if all of the following are green:

1. `pytest tests/test_refactor_golden.py -q`
2. `pytest tests/test_replay_bundle.py -q`
3. `pytest tests/test_artifacts_models.py -q`
4. `pytest`

Required semantic checks within that set:

1. `manifest.replaySummary` equals
   `evaluation.summary.replay_timeline`
2. `validate_replay_bundle()` still enforces the Phase 03A / 03B bundle
   surface
3. trimming a bundle keeps summary/manifest replay counts synchronized
4. canonical sample fixture regeneration remains deterministic

## 6. Stop Conditions

Slice C must stop and roll back to the pre-split façade if any of these
becomes true:

1. any Phase 03A / 03B required field key or value semantics would have
   to change,
2. sample fixture regeneration is not byte-stable after placeholder
   normalization,
3. `ReplaySummary` cannot represent the existing exported summary
   without dropping a field,
4. a currently green refactor-golden / bundle test must be deleted
   rather than preserved,
5. the work starts drifting into runtime spine split, `TrainerConfig`
   relocation, or checkpoint payload restructuring.

## 7. Rollback Plan

If Slice C fails one of the stop conditions:

1. restore `export/replay_bundle.py` as the single implementation file
   for the affected responsibility,
2. keep `ReplaySummary` single-sourcing only if the failing surface is
   unrelated; otherwise roll that change back together with the split,
3. remove any new module that no longer has a live caller,
4. keep the Phase 04B artifact model seam intact,
5. record the failure in a bounded status note instead of widening scope
   into Slice D.

## 8. Expected Output Of This Slice

When landed, Slice C should leave the repo with:

1. a readable bundle-contract module tree,
2. a smaller `export/replay_bundle.py` façade,
3. one `ReplaySummary` source object shared across manifest and
   evaluation summary,
4. no external bundle schema version bump,
5. no change to the frozen Phase 03A / 03B producer contract.
