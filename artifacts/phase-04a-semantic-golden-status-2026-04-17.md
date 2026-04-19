# Phase 04A Slice A — Artifact-Level Semantic Golden Status

**Date:** `2026-04-17`
**SDD:** [`../docs/phases/phase-04a-refactor-semantic-golden-sdd.md`](../docs/phases/phase-04a-refactor-semantic-golden-sdd.md)
**Parent kickoff:** [`../docs/phases/phase-04-refactor-contract-spine-sdd.md`](../docs/phases/phase-04-refactor-contract-spine-sdd.md)
**Current interpretation:** [`./phase-04-current-state-2026-04-19.md`](./phase-04-current-state-2026-04-19.md)
**Result:** `LOCAL-PROMOTION-CANDIDATE` — this note records a green local Slice A validation run, but should not by itself be read as the landed repo-wide promotion note for Phase 04.

## 1. Summary

This note records a local Slice A promotion-candidate run. Six test
families (F1 – F6) enforce the implicit producer contract as explicit,
cross-file semantic invariants. The observed slice shape is still the
same one described by the original draft: no file under `src/`,
`scripts/`, `configs/`, or `tests/fixtures/` was changed by the Slice A
implementation itself, and the work remains rollback-safe because it
adds only test code and one helper package.

Use `phase-04-current-state-2026-04-19.md` for the current repo-state
interpretation: this note is evidence that Slice A looks ready, not a
claim that the full repo authority chain has already promoted Phase 04
to landed status.

## 2. Test Families Observed In The Local Candidate Run

| Family | File / Test | Result |
| --- | --- | --- |
| F1  | `test_f1_run_metadata_shape` | PASS |
| F1b | `test_f1_trainer_config_shape_matches_dataclass` | PASS |
| F2  | `test_f2_training_log_row_shape` | PASS |
| F3a | `test_f3_primary_checkpoint_envelope` | PASS |
| F3b | `test_f3_best_eval_checkpoint_envelope_when_present` | PASS |
| F4  | `test_f4_manifest_summary_cross_file_consistency[fixture]` | PASS |
| F4  | `test_f4_manifest_summary_cross_file_consistency[fresh-smoke-export]` | PASS |
| F5  | `test_f5_timeline_row_geometry[fixture]` | PASS |
| F5  | `test_f5_timeline_row_geometry[fresh-smoke-export]` | PASS |
| F5  | `test_f5_slot_duration_config_matches_expectation` | PASS |
| F6  | `test_f6_sample_bundle_regeneration_is_deterministic` | PASS |

Total: `11 passed in 22.48s`.

Full-suite re-run (`pytest`) confirms no regression:
`215 passed in 250.80s`.

## 3. Local Candidate Files Expected For Slice A

1. `tests/test_refactor_golden.py`
2. `tests/refactor_golden/__init__.py`
3. `tests/refactor_golden/helpers.py`

In current interpretation, treat these as the files associated with the
local candidate Slice A result. They should not be described as landed
repo surfaces unless they are committed together with a later explicit
landed promotion note.

No existing file was modified by the Slice A implementation itself.

## 4. Surprises Discovered During Slice A

### 4.1 F6 non-determinism was a test-harness bug, not a real non-determinism

First F6 run reported two files with byte diffs
(`manifest.json`, `evaluation/summary.json`) while the timeline file
matched exactly. This ruled out training-side non-determinism.

Root cause: the test originally passed `--config <absolute path>` to
`scripts/generate_sample_bundle.py`, while the checked-in fixture had
been produced with the default **relative** path
`configs/modqn-paper-baseline.resolved-template.yaml`. The resolved
config path is embedded into `manifest.configPath` and
`evaluation/summary.json.config_path` verbatim, so the two files
diverged by string only.

Fix: F6 now omits `--config` entirely and relies on the script's own
default, with an inline comment explaining why. No source change.

### 4.2 Fragile fixture identity coupling (surface observation, not a bug)

The above finding surfaces a genuine brittleness in the producer
contract:

1. `run_metadata.json.config_path` stores whatever path string was
   passed on the command line,
2. that string flows through into `manifest.configPath` and
   `evaluation/summary.json.config_path`,
3. therefore the checked-in sample fixture implicitly depends on
   being regenerated with the script's default relative path rather
   than an absolute one.

This is not a Slice A failure — the surface behaves the way it was
built to behave. It is, however, something Slice B's training-artifact
model should consider: the eventual `RunMetadataV1` may want to
normalize `config_path` at write time (for example, to a repo-relative
form when possible) rather than preserve argv verbatim.

Recommendation: document this observation for Slice B's per-slice
SDD; do not change behavior in Slice A.

### 4.3 `test_f3_best_eval_checkpoint_envelope_when_present` not skipped

The one-episode smoke run still produces a `secondary_best_eval`
checkpoint because `MODQNTrainer.train()` evaluates at the final
episode. F3b therefore exercises the best-eval envelope on every run
rather than skipping. The `pytest.skip` branch remains in place for
safety.

## 5. Invariants Now Enforced

Slice A protects the following invariants, which future Slice B/C/D/E
work must keep:

1. `run_metadata.json` top-level key set of 16 keys.
2. `run_metadata.trainer_config` covers every field in
   `algorithms.modqn.TrainerConfig`. Slice D's facade-preserving move
   of `TrainerConfig` must leave this green.
3. `training_log.json` per-row 9-key set with correct types,
   `losses` length 3, monotonic `episode` from 0.
4. Checkpoint envelope carries `format_version == 1` with the current
   required key superset; best-eval checkpoints also carry
   `evaluation_summary` with its 13 documented keys.
5. `manifest.replaySummary` and `evaluation/summary.json.replay_timeline`
   are deep-equal both in the checked-in fixture and in any fresh
   smoke export. This is the single most important invariant protecting
   Slice B/C against silent drift when the two files stop sharing the
   same runtime dict.
6. Timeline row geometry: `beamCatalogOrder` constant,
   `len(beamStates) == len(satelliteStates) * beams_per_satellite`,
   mask lengths aligned with beam count, `selectedServing.beamIndex`
   in range and valid under `actionValidityMask`,
   `handoverEvent.kind` ∈ three allowed values, `slotIndex >= 1`
   strictly monotonic per `userId`, `timeSec == slotIndex * slot_duration_s`,
   `userPosition.localTangentKm` recomputed from `groundPoint`.
7. `scripts/generate_sample_bundle.py` is byte-reproducible against
   the checked-in fixture when invoked with its default args.

## 6. Stop / Rollback Check

None of the six stop conditions from kickoff SDD section 8 fired.
Specifically:

1. No section 4 frozen surface required redefinition.
2. No `format_version` change proposed.
3. No CLI guardrail weakened.
4. No `StepEnvironment` touched.
5. Fixture regeneration is byte-stable (after test-harness fix).
6. No existing test was deleted or weakened; the full suite still
   passes.

## 7. Slice B Readiness

Slice B (`Training Artifact Model`) is now locally justified by this
status note per kickoff SDD section 6.1 and 10, but that should still
be read through the current-state interpretation note rather than as an
already-landed repo closeout.

Before starting Slice B implementation, the next required action is to
use the existing
`docs/phases/phase-04b-refactor-training-artifact-model-sdd.md`
as the controlling planning surface and reconcile it against the final
landed interpretation of Slice A. At minimum, Slice B planning should
still confirm:

1. name every model type / module added,
2. restate section 4 frozen surfaces,
3. restate section 8 stop conditions,
4. define validation set (should include running Slice A's
   `tests/test_refactor_golden.py` as a regression gate),
5. define rollback plan.

Slice B planning should also cite section 4.2 of this status note
(config-path handling) as an explicit input into the
`RunMetadataV1` design.

## 8. Promotion Decision

**LOCAL-PROMOTION-CANDIDATE**. Slice A appears clean and useful: all 11
tests are green, there is no reported regression in the existing suite,
and section 4.2 surfaces one real design input for Slice B.

Slice B authoring may begin as planning work, but future repo-level
documentation should continue to describe Phase 04 as in-flight until
the docs/tests/status surfaces are fully reconciled into the landed
authority chain.
