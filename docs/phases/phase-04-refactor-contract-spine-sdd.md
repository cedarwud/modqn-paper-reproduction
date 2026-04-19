# Phase 04: Refactor Contract-Spine Kickoff SDD

**Status:** Drafted kickoff SDD; current repo-state interpretation remains in-flight.
**Date:** `2026-04-17`
**Current interpretation:** [`../../artifacts/phase-04-current-state-2026-04-19.md`](../../artifacts/phase-04-current-state-2026-04-19.md)
**Base producer schema:** `phase-03a-replay-bundle-v1` (plus Phase 03B
additive `policyDiagnostics` surface)
**Depends on:**

1. [`phase-01-python-baseline-reproduction-sdd.md`](./phase-01-python-baseline-reproduction-sdd.md)
2. [`phase-02-artifact-bridge-sdd.md`](./phase-02-artifact-bridge-sdd.md)
3. [`phase-03a-ntn-sim-core-bundle-replay-integration-sdd.md`](./phase-03a-ntn-sim-core-bundle-replay-integration-sdd.md)
4. [`phase-03b-ntn-sim-core-producer-diagnostics-export-sdd.md`](./phase-03b-ntn-sim-core-producer-diagnostics-export-sdd.md)
5. [`../../artifacts/phase-01c-closeout-status-2026-04-15.md`](../../artifacts/phase-01c-closeout-status-2026-04-15.md)
6. [`../../artifacts/phase-03b-producer-diagnostics-export-status-2026-04-16.md`](../../artifacts/phase-03b-producer-diagnostics-export-status-2026-04-16.md)

## 1. Purpose

This SDD is the kickoff planning surface for a future internal refactor
that would promote the repo's de facto producer contract (training
artifact shape, replay bundle surface, runtime helper surface) from
scattered dicts / string keys / large modules into an explicit model
layer.

It is **a thin kickoff SDD, not a complete refactor plan**. It exists to
pin down the things that must be decided before any module split starts:

1. which external surfaces must stay byte-stable during the refactor,
2. which non-negotiable boundary rules apply across every slice,
3. the slice ordering and stop/rollback conditions,
4. what remains explicitly out of scope.

Per-slice design detail is intentionally deferred to later execution
SDDs (`phase-04a`, `phase-04b`, ...). This is because the slice A
artifact-level semantic golden tests are expected to surface real
invariants that a pre-written full plan would have guessed at.

This kickoff document does not by itself promote any Phase 04 slice to
landed repo authority. Use the current-state note above when deciding
whether the repo should be treated as "Phase 04 landed" or merely
"Phase 04 planned / locally exercised".

## 2. Why This SDD Exists

The current repo passed its Phase 01C closeout and landed its Phase 03B
bounded reopen slice. Both the baseline comparison surface and the
additive producer-diagnostics surface are now frozen.

However, repo inspection shows that:

1. `cli.py` directly writes `training_log.json`, `run_metadata.json`,
   and checkpoint files with hand-written dict keys,
2. `export/pipeline.py` embeds the same `replay_summary` object twice,
   once into `manifest.json` (camelCase top level) and once into
   `evaluation/summary.json` (snake_case top level),
3. `scripts/generate_sample_bundle.py` has to mirror the trimmed
   `manifest.replaySummary` back into `evaluation.summary.replay_timeline`
   by deep-copy because no model layer enforces single-sourcing,
4. `export/replay_bundle.py` is currently `1441` lines and mixes schema
   constants, compat loading, replay runtime, manifest assembly,
   validation, diagnostics validation, and fixture trimming,
5. `algorithms/modqn.py` is currently `1116` lines and is imported as a
   runtime utility hub by `dqn_scalar.py`, `config_loader.py`,
   `sweeps.py`, and `export/replay_bundle.py`,
6. `contracts.py` / `settings.py` exist but are not actually imported
   by any main path or test.

None of these are bugs in the frozen bundle. They are structural debt:
the producer contract is real, but it is not a model.

## 3. Scope

### 3.1 In Scope

1. introducing explicit internal model types for:
   - run metadata,
   - training log rows,
   - checkpoint payload envelope,
   - replay summary,
   - bundle manifest,
   - evaluation summary,
2. single-sourcing cross-file data so that `manifest.replaySummary` and
   `evaluation.summary.replay_timeline` are serializations of the same
   object, not two dicts,
3. splitting `export/replay_bundle.py` along responsibility seams
   (schema, serializer, validator, provenance, fixture tools, compat),
4. splitting `algorithms/modqn.py` along responsibility seams
   (trainer config, state encoding, objective math, checkpoint payload,
   evaluation, replay runner, trainer loop),
5. reversing the `config_loader.py` → `algorithms.modqn.TrainerConfig`
   dependency direction,
6. adding artifact-level semantic golden tests that rescue the current
   implicit cross-file invariants before any module split.

### 3.2 Explicitly Out Of Scope

1. any change to `StepEnvironment` reward semantics, action ordering,
   or `slotIndex` / `timeSec` conventions,
2. any change to `Phase 03A` required manifest / timeline fields or
   their meanings,
3. any change to `Phase 03B` `policyDiagnostics` / `optionalPolicyDiagnostics`
   required keys or their validation rules,
4. any change to checkpoint `format_version = 1` payload contents or
   key names,
5. any change to `replay_seed` source priority or best-eval checkpoint
   selection rule,
6. any plot, figure-CSV, analysis-markdown cleanup,
7. any new paper-faithful reproduction claim,
8. any new sweep family, new reward equation, or new baseline,
9. any consumer-side change inside `ntn-sim-core`.

## 4. Frozen External Surfaces During Refactor

These surfaces must remain byte-stable for the lifetime of this
refactor. Any diff against a pre-refactor sample bundle must be either
zero or documented as an explicit, reviewed, one-time exception.

### 4.1 Phase 03A Required Bundle Surfaces

1. `manifest.json` required top-level fields:
   `paperId`, `runId`, `bundleSchemaVersion`, `producerVersion`,
   `exportedAt`, `sourceArtifactDir`, `checkpointRule`,
   `replayTruthMode`, `timelineFormatVersion`, `coordinateFrame`.
2. `manifest.json` identity invariants:
   `bundleSchemaVersion == "phase-03a-replay-bundle-v1"`,
   `timelineFormatVersion == "step-trace.jsonl/v1"`,
   `beamCatalogOrder == "satellite-major-beam-minor"`,
   `slotIndexSemantics.firstIndex == 1`,
   `replayTruthMode == "selected-checkpoint-greedy-replay"`.
3. `timeline/step-trace.jsonl` required row fields per
   `_timeline_row_required_fields()`:
   `slotIndex`, `timeSec`, `userId`, `userPosition`, `previousServing`,
   `selectedServing`, `handoverEvent`, `visibilityMask`,
   `actionValidityMask`, `beamLoads`, `rewardVector`, `scalarReward`,
   `satelliteStates`, `beamStates`, `kpiOverlay`.
4. `config-resolved.json`, `provenance-map.json`, `assumptions.json`
   presence and content shape.

### 4.2 Phase 03B Additive Diagnostics Surfaces

1. `timeline/step-trace.jsonl` row optional `policyDiagnostics`
   required keys:
   `diagnosticsVersion`, `objectiveWeights`,
   `selectedScalarizedQ`, `runnerUpScalarizedQ`,
   `scalarizedMarginToRunnerUp`, `availableActionCount`,
   `topCandidates`.
2. `policyDiagnostics.topCandidates` entry required keys:
   `beamId`, `beamIndex`, `satId`, `satIndex`, `localBeamIndex`,
   `validUnderDecisionMask`, `objectiveQ`, `scalarizedQ`.
3. `manifest.optionalPolicyDiagnostics` required keys:
   `present`, `timelineField`, `diagnosticsVersion`,
   `requiredByBundleSchema`, `producerOwned`, `selectedActionSource`,
   `topCandidateLimit`, `rowsWithDiagnostics`, `rowsWithoutDiagnostics`,
   `note`.
4. `objectiveWeights` and `objectiveQ` stay as named objects keyed by
   `r1Throughput` / `r2Handover` / `r3LoadBalance`, not positional
   arrays.

### 4.3 Checkpoint Payload Envelope

1. `format_version == 1`.
2. Payload top-level keys remain a superset of:
   `format_version`, `checkpoint_kind`, `episode`, `train_seed`,
   `env_seed`, `mobility_seed`, `state_dim`, `action_dim`,
   `trainer_config`, `checkpoint_rule`, `q_networks`,
   `target_networks`.
3. Best-eval checkpoints continue to carry `evaluation_summary`.
4. Primary final / secondary best-eval selection still obeys
   `ASSUME-MODQN-REP-015`.

### 4.4 Training Artifact Contract

1. `run_metadata.json` retains its current top-level key set,
   including `resolved_config_snapshot`, `checkpoint_files.primary_final`,
   `checkpoint_files.secondary_best_eval`, `seeds`, `best_eval_summary`,
   `checkpoint_rule`, `resolved_assumptions`, `training_summary`,
   `reward_calibration`, `runtime_environment`.
2. `training_log.json` remains a JSON array of per-episode rows with
   the current key set: `episode`, `epsilon`, `r1_mean`, `r2_mean`,
   `r3_mean`, `scalar_reward`, `total_handovers`, `replay_size`,
   `losses`.

### 4.5 Sample Fixture

1. `tests/fixtures/sample-bundle-v1/` must remain a valid consumer
   fixture for the entirety of the refactor.
2. Any intentional change to the fixture must be accompanied by an
   explicit status note and a regenerated fixture committed in the
   same change set.

## 5. Non-Negotiable Boundary Rules

1. No refactor slice may silently change any surface listed in section
   4 without a paired SDD amendment and a status note.
2. No refactor slice may introduce a fallback default on the main CLI
   path. `load_training_yaml()` / `require_training_config()` must stay
   strict.
3. No refactor slice may change `paper-envelope` vs `resolved-run`
   guardrail semantics.
4. No refactor slice may make `ntn-sim-core` the new authority for any
   training, handover, or bundle contract decision.
5. No refactor slice may delete `contracts.py` / `settings.py` while
   their content is still the only named definition of some paper
   parameter. They may be replaced by equivalent typed surfaces, but
   only after those replacements are landed and imported by a main
   path.
6. No refactor slice may reopen paper-faithful reproduction work,
   restart sweeps, or revise Phase 01C negative-result interpretation.
7. No refactor slice may skip tests with `--no-verify` or equivalent.
   If a hook or test fails, the fix is to resolve the underlying
   issue, not to bypass the guard.

## 6. Slice Plan (High-Level Only)

Per-slice design detail is deferred to later execution SDDs. What this
SDD commits to is only the ordering and dependency edges.

### 6.1 Slice A — Artifact-Level Semantic Golden

Purpose:

1. rescue the implicit cross-file invariants before any code moves.

Ships:

1. `run_metadata.json` key / shape golden snapshot,
2. checkpoint payload key / `format_version` / `evaluation_summary`
   golden snapshot,
3. `manifest.replaySummary` ↔ `evaluation.summary.replay_timeline`
   cross-file consistency test asserting that both serialize the same
   underlying data (tolerating top-level casing convention, not
   tolerating semantic drift),
4. timeline invariants:
   `beamStates` / `actionValidityMask` / `beamCatalogOrder` /
   `slotIndexOffset` / `coordinateFrame.groundPoint` consistency,
5. fixture round-trip determinism (regenerate sample bundle and diff).

Blocks: every later slice.

Stop rule:

1. if Slice A reveals that an existing surface in section 4 is already
   inconsistent today, that is a separate bug, not a refactor failure;
   record a status note and decide separately before starting Slice B.

### 6.2 Slice B — Training Artifact Model

Purpose:

1. make the training-side `run_metadata.json` / `training_log.json` /
   checkpoint payload contract explicit.

Ships:

1. `RunMetadataV1`, `TrainingLogRow`, `CheckpointPayloadV1`,
   `CheckpointCatalog`, `RunArtifactPaths` model types,
2. `cli.py` writes them through model serialization, not hand-built
   dicts,
3. `export/pipeline.py` and `export/replay_bundle.py` read them
   through model deserialization, not raw dict access.

Does not change file format.

Blocks: Slice C, Slice D's checkpoint extraction.

Stop rule:

1. if model types cannot express the current artifact shape without
   losing a field, preserve the field verbatim and record the gap;
   do not drop fields for cleanliness.

### 6.3 Slice C — Bundle Contract Layer Split

Purpose:

1. split `export/replay_bundle.py` into responsibility-separated modules
   so bundle fields, validation rules, and fixture tools can evolve
   independently.

Ships:

1. `bundle/schema.py` — constants and required-field definitions,
2. `bundle/serializers.py` — runtime → JSON-ready conversion,
3. `bundle/provenance.py` — provenance map builder,
4. `bundle/validator.py` — `validate_replay_bundle()` and
   `policyDiagnostics` validation,
5. `bundle/fixture_tools.py` — `trim_replay_bundle_for_sample()`,
6. `artifacts/compat.py` — `resolve_training_config_snapshot()`,
   `select_replay_checkpoint()`, `_select_timeline_seed()`,
   `_resolve_existing_path()`,
7. unified `ReplaySummary` single-source used by both manifest and
   evaluation-summary serialization.

Blocks: Slice D's replay runner extraction.

Stop rule:

1. if splitting breaks any Phase 03A / 03B required surface in
   section 4, revert the offending split and keep the combined file
   until Slice A invariants are extended.

### 6.4 Slice D — Runtime Spine Split (Facade-Preserving)

Purpose:

1. reduce `algorithms/modqn.py` from a runtime utility hub back to a
   trainer implementation, without breaking external imports.

Ships:

1. `runtime/trainer_spec.py` — `TrainerConfig`,
2. `runtime/state_encoding.py` — `encode_state()`, `state_dim_for()`,
3. `runtime/objective_math.py` — `scalarize_objectives()`,
   `apply_reward_calibration()`,
4. `runtime/replay_buffer.py` — `ReplayBuffer`,
5. `runtime/q_network.py` — `DQNNetwork`,
6. `runtime/checkpoint_payload.py` — payload builder,
7. `runtime/evaluation.py` — `PolicyEvaluator`,
8. `runtime/replay_runner.py` — deterministic greedy replay runner,
9. reversed dependency: `config_loader.py` imports
   `TrainerConfig` from `runtime/`, not from
   `algorithms/modqn`,
10. `algorithms/modqn.py` keeps its current public re-export list as a
    facade for the duration of this slice.

Blocks: Slice E's analysis split.

Stop rule:

1. if extracting any helper changes checkpoint byte content or
   replay timeline byte content, stop the extraction and treat it as
   a semantics investigation rather than a refactor.

### 6.5 Slice E — Sweep / Analysis / Plotting Split

Purpose:

1. separate experiment running from result aggregation, and separate
   analysis/plotting from bundle export.

Ships:

1. `orchestration/` module owning `train_main`, `sweep_main`,
   `export_main` dispatch,
2. `sweeps.py` split into experiment runner vs result aggregation,
3. `analysis/table_ii.py`, `analysis/reward_geometry.py`,
   `analysis/figures.py` carrying the current
   `export/pipeline.py` analysis responsibilities,
4. `export/pipeline.py` retained only as a thin orchestrator or
   removed if its remaining content is trivial.

Stop rule:

1. if splitting changes the content of any landed analysis artifact
   under `artifacts/*`, revert and narrow scope.

## 7. Slice Dependency Edges

```
Slice A
  └── blocks Slice B, C, D, E
Slice B
  └── blocks Slice C (for artifact input), Slice D (for checkpoint payload)
Slice C
  └── blocks Slice D (only for replay runner extraction)
Slice D
  └── blocks Slice E (for runtime seam reuse)
```

Slice C and Slice D schema extraction may partially parallelize after
Slice B lands, but checkpoint-extraction work inside Slice D must not
start until Slice C has a stable `ReplaySummary` type.

## 8. Stop / Rollback Rule (Any Slice)

Stop the current slice and hand off to status review if any of these
becomes true:

1. a refactor step would require changing a Phase 03A / 03B required
   field definition from section 4,
2. a refactor step would require changing `format_version` from `1` or
   restructuring the checkpoint payload envelope,
3. a refactor step would require weakening the CLI
   `load_training_yaml()` / `require_training_config()` guardrail,
4. a refactor step would require touching `StepEnvironment` reward /
   action ordering / `slotIndex` conventions,
5. sample fixture regeneration is not byte-stable (or byte-stable in
   all fields except the explicitly normalized placeholders),
6. any currently green test in `tests/` requires deletion rather than
   replacement to land the slice.

If the stop rule fires, do not keep refactoring inside the same slice.
Record a bounded status note, decide separately whether the scope needs
revision, and either re-open the slice with an amendment or close it as
a negative result.

## 9. Artifacts Committed By This SDD

This SDD, by itself, commits to:

1. section 4 as the frozen external surface list,
2. section 5 as the non-negotiable rules,
3. section 6 as the slice ordering with dependency edges,
4. section 8 as the stop rule.

It does **not** commit to:

1. detailed module-level type signatures,
2. detailed file paths beyond the slice sketches,
3. time estimates,
4. per-slice validation matrices.

Those are the job of the per-slice execution SDDs.

## 10. Per-Slice Execution SDD Rule

Each slice A–E, when it starts, must land a matching execution SDD at:

1. `docs/phases/phase-04a-refactor-semantic-golden-sdd.md`
2. `docs/phases/phase-04b-refactor-training-artifact-model-sdd.md`
3. `docs/phases/phase-04c-refactor-bundle-layer-split-sdd.md`
4. `docs/phases/phase-04d-refactor-runtime-spine-split-sdd.md`
5. `docs/phases/phase-04e-refactor-sweep-analysis-split-sdd.md`

Each per-slice SDD must, at minimum:

1. name every module / function / type it adds, moves, or removes,
2. restate which section 4 surfaces it must leave byte-stable,
3. restate which section 8 stop conditions apply,
4. define the slice's validation set,
5. define the slice's rollback plan.

No slice work may begin before its execution SDD is drafted.

## 11. Relationship To Phase 01D Reopen Gate

This SDD does **not** claim a Phase 01D reopen of reproduction work.

1. No new paper-faithful training run is scheduled.
2. No new sweep family is scheduled.
3. No reward equation is changed.
4. No checkpoint-selection rule is changed.
5. No `StepEnvironment` semantics are changed.

The refactor's goal is only to make the existing producer contract
explicit and maintainable. If any slice strays into reproduction-truth
territory, section 8 requires stopping and escalating to a Phase 01D
reassessment instead of continuing inside this SDD.

## 12. Relationship To `ntn-sim-core`

`ntn-sim-core` remains a pure consumer of the frozen Phase 03A bundle
plus the Phase 03B additive diagnostics surface. This SDD does not:

1. require any coordinated change on the consumer side,
2. require any schema version bump,
3. change the identity of `phase-03a-replay-bundle-v1` or the
   `step-trace.jsonl/v1` timeline format.

Consumer-side adoption of optional diagnostics remains separate work
tracked under Phase 03B.

## 13. Negative-Result Rule

If Slice A reveals that the current implicit contract cannot be rescued
without breaking a surface in section 4, this SDD should record:

1. the attempt was bounded,
2. the structural observation stands,
3. the repo remains at the current frozen closeout plus landed Phase
   03B additive slice,
4. no further refactor slices are justified until a new reopen trigger
   exists.

Negative results are valid outcomes for this SDD.

## 14. Status

1. This SDD is drafted and reviewed.
2. No slice has started.
3. The first slice allowed to start is Slice A — Artifact-Level
   Semantic Golden (section 6.1).
