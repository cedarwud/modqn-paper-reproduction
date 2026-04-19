# Phase 04A: Refactor Slice A — Artifact-Level Semantic Golden SDD

**Status:** Drafted execution SDD; current repo-state interpretation remains in-flight.
**Date:** `2026-04-17`
**Parent kickoff:**
[`phase-04-refactor-contract-spine-sdd.md`](./phase-04-refactor-contract-spine-sdd.md)
**Current interpretation:** [`../../artifacts/phase-04-current-state-2026-04-19.md`](../../artifacts/phase-04-current-state-2026-04-19.md)
**Depends on:**

1. [`phase-04-refactor-contract-spine-sdd.md`](./phase-04-refactor-contract-spine-sdd.md)
2. [`phase-03a-ntn-sim-core-bundle-replay-integration-sdd.md`](./phase-03a-ntn-sim-core-bundle-replay-integration-sdd.md)
3. [`phase-03b-ntn-sim-core-producer-diagnostics-export-sdd.md`](./phase-03b-ntn-sim-core-producer-diagnostics-export-sdd.md)

## 1. Purpose

This execution SDD defines the first slice of the Phase 04 refactor.

The only purpose of Slice A is to **rescue the current producer
contract as explicit, test-enforced invariants before any module or
type moves**. No code under `src/` is modified. No file format is
changed. No module is split.

Slice A is deliberately the smallest possible slice: it only adds test
code and one small test-helper module.

This SDD defines Slice A's intended acceptance rules. It does not by
itself claim that Slice A is already landed. For current repo-state
interpretation, read the current-state note above together with any
local status note.

## 2. Why Slice A Goes First

The repo today has a working, frozen producer bundle surface, but
several cross-file invariants are enforced only by coincidence of
call order or by hand-written dict assembly, not by an explicit
contract. In particular:

1. `manifest.replaySummary` and `evaluation.summary.replay_timeline`
   are currently the same runtime dict written to two files by
   `export/pipeline.py`, but no test asserts that.
2. `scripts/generate_sample_bundle.py` already has to deep-copy the
   trimmed `replaySummary` back into the summary file to keep them in
   sync, which confirms the implicit dependency is real.
3. `run_metadata.json` key set, `training_log.json` row shape, and
   the checkpoint payload envelope are consumed by `export/pipeline.py`
   and `export/replay_bundle.py` as if they were contracts, but no
   test enforces their key set.
4. Timeline row geometry invariants (`beamStates` cardinality,
   `actionValidityMask` length, `beamCatalogOrder` constant,
   `slotIndexOffset`, `coordinateFrame.groundPoint` consistency) are
   exercised only indirectly by replay validation.

Without golden tests around these invariants, Slice B/C/D/E can all
pass `pytest` while silently drifting the meaning of a frozen surface.

## 3. Scope

### 3.1 In Scope

1. new test code that locks current observable invariants of:
   - `run_metadata.json`,
   - `training_log.json` row shape,
   - checkpoint payload envelope,
   - `manifest.json` ↔ `evaluation/summary.json` shared-data
     consistency,
   - timeline row geometry,
2. one new small test helper that loads a canonical small run
   artifact (produced by a smoke training pass) plus the canonical
   sample bundle,
3. one regression test that regenerates the sample bundle fixture and
   diffs against the checked-in one.

### 3.2 Explicitly Out Of Scope

1. any change under `src/modqn_paper_reproduction/`,
2. any change to `scripts/`,
3. any change to `configs/`,
4. any change to the content of `tests/fixtures/sample-bundle-v1/`,
5. any new SDD for Slice B/C/D/E (those remain deferred),
6. any refactor of existing tests; Slice A adds tests, it does not
   rewrite them.

## 4. Frozen Surfaces Slice A Must Leave Byte-Stable

The same list as `phase-04` section 4 applies here, restated so this
SDD is self-contained:

1. `manifest.json` required top-level fields and identity invariants
   (section 4.1 of the kickoff SDD).
2. `timeline/step-trace.jsonl` required row fields (section 4.1).
3. Phase 03B `policyDiagnostics` and `optionalPolicyDiagnostics`
   required keys (section 4.2).
4. Checkpoint payload envelope with `format_version == 1` and current
   top-level keys (section 4.3).
5. `run_metadata.json` top-level keys and `training_log.json` row
   keys (section 4.4).
6. The checked-in `tests/fixtures/sample-bundle-v1/` (section 4.5).

Since Slice A adds only tests, the risk of changing any of these is
low. The practical rule is: if any new golden test initially runs red,
the fix is to investigate the inconsistency, not to mutate the
surface.

## 5. Stop Conditions Applying To Slice A

All six kickoff-level stop conditions (section 8 of `phase-04`) apply.
The two most likely to fire during Slice A are:

1. a golden test reveals that a Phase 03A / 03B required field is
   already missing or mis-typed in the current artifact (kickoff stop
   #1),
2. the sample fixture regeneration is not byte-stable after
   placeholder normalization (kickoff stop #5).

If either fires, Slice A must stop, record a bounded status note, and
decide separately whether the finding is a latent bug (fix outside
Slice A) or an acceptable current-state surface (the golden test is
adjusted with a comment and an explicit reference to the decision
note).

Slice A must not paper over a real inconsistency by weakening a
golden test.

## 6. Test Set

Slice A ships five test families. Each family lists its target
invariants, its representation strategy (value vs structural), and its
expected red-first behavior.

### 6.1 F1 — `run_metadata.json` Key / Shape Golden

Target:

1. every top-level key currently written by `train_main()` is
   present,
2. every nested object currently written by `train_main()` has its
   expected subkeys,
3. identity values are exact where the surface is paper-bound
   (`paper_id`, `config_role`, `checkpoint_rule.assumption_id`,
   `checkpoint_rule.primary_report`,
   `checkpoint_rule.secondary_report`),
4. environment-dependent values are checked by type only, not value
   (paths, wall-clock times, seed numbers).

Representation:

1. assert on key set (using Python `set`, not positional order),
2. assert on type per key,
3. assert on exact string value for identity fields only.

Red-first behavior:

1. likely green on day one because the key set is stable,
2. if red, that means `train_main()` is already writing an unexpected
   key and the finding must be recorded in a status note.

### 6.2 F2 — `training_log.json` Row Shape Golden

Target:

1. `training_log.json` is a JSON array of per-episode rows,
2. every row has exactly the current key set:
   `episode`, `epsilon`, `r1_mean`, `r2_mean`, `r3_mean`,
   `scalar_reward`, `total_handovers`, `replay_size`, `losses`,
3. `losses` is a 3-element list of floats,
4. `episode` values are monotonic starting from `0`.

Representation:

1. key-set assert,
2. type assert per key,
3. structural assert on `losses` length,
4. monotonic assert on `episode`.

Red-first behavior:

1. expected green; if red, surface drift already happened.

### 6.3 F3 — Checkpoint Payload Envelope Golden

Target:

1. `format_version == 1`,
2. top-level key set is a superset of:
   `format_version`, `checkpoint_kind`, `episode`, `train_seed`,
   `env_seed`, `mobility_seed`, `state_dim`, `action_dim`,
   `trainer_config`, `checkpoint_rule`, `q_networks`,
   `target_networks`,
3. when the payload is built with `include_optimizers=True`,
   `optimizers` is also present,
4. when the payload is built from the best-eval path,
   `evaluation_summary` is present and carries
   `eval_seeds`, `episode`, `evaluation_every_episodes`,
   `mean_scalar_reward`, `std_scalar_reward`.

Representation:

1. key-superset assert (do not forbid future additive keys),
2. exact value assert on `format_version`,
3. key-presence assert on `evaluation_summary` only when the
   payload's `checkpoint_kind` is the best-eval kind.

Red-first behavior:

1. expected green; any red indicates a current-state payload shape
   surprise.

### 6.4 F4 — `manifest.replaySummary` ↔ `evaluation.summary.replay_timeline` Cross-File Consistency

Target:

1. both objects, when read as JSON, are equal under deep equality,
2. this holds both for the checked-in `tests/fixtures/sample-bundle-v1/`
   and for any freshly exported smoke bundle.

Representation:

1. load both JSON files,
2. extract the two sub-objects,
3. assert equality by `==` on parsed dicts.

Red-first behavior:

1. expected green today because `export/pipeline.py` assigns the same
   dict to both,
2. if red, it means the pipeline already drifted; record and escalate
   before writing Slice B.

This is the single most important test in Slice A because Slice B and
Slice C will reorganize how both surfaces are produced, and this test
is what prevents silent drift during that work.

### 6.5 F5 — Timeline Row Geometry Invariants

Target, per row in `timeline/step-trace.jsonl`:

1. `beamCatalogOrder == "satellite-major-beam-minor"`,
2. length of `beamStates == len(satelliteStates) * beams_per_satellite`,
3. length of `actionValidityMask == len(beamStates)`,
4. length of `visibilityMask == len(beamStates)`,
5. `selectedServing.beamIndex` is in range `[0, len(beamStates))`,
6. `selectedServing.beamIndex` is `True` under
   `actionValidityMask`,
7. `handoverEvent.kind` is one of `none`,
   `intra-satellite-beam-switch`, `inter-satellite-handover`,
8. `slotIndex >= 1` and is strictly monotonic within each `userId`,
9. `timeSec` is consistent with `slotIndex * slot_duration_s`
   within a small numerical tolerance,
10. `coordinateFrame.groundPoint` (from manifest) matches the
    `localTangentKm` anchor implied by `userPosition`.

Representation:

1. iterate timeline rows,
2. assert each invariant with clear per-invariant error messages.

Red-first behavior:

1. expected green; any red finding is a latent bug, not a refactor
   failure.

### 6.6 F6 — Sample Fixture Regeneration Determinism

Target:

1. running `scripts/generate_sample_bundle.py` against a temporary
   output directory, with the default args used by the current
   fixture (`--episodes 1 --max-users 1`), produces byte-stable
   output matching `tests/fixtures/sample-bundle-v1/`,
2. byte stability is measured after the script's documented
   placeholder normalizations (`sourceArtifactDir`, `inputArtifactDir`,
   `outputDir`, `exportedAt`, checkpoint paths, `elapsed_s`) are
   applied.

Representation:

1. run the script against `tmp_path`,
2. walk both directory trees,
3. assert file-by-file equality on normalized content.

Red-first behavior:

1. this is the highest-risk-of-red test. If red, the finding is
   either:
   - a non-determinism in training that was previously hidden, or
   - a documented placeholder that is not being applied correctly.
2. both possibilities require a status note and a separate fix
   before Slice A can be considered landed.

Slice A is allowed to mark F6 as `xfail` with a bounded status note
**only if** the non-determinism is unrelated to the refactor's scope
and is recorded separately; otherwise F6 must be green.

## 7. Files Added

Slice A adds:

1. `tests/test_refactor_golden.py` — hosts F1 – F6 and their
   module-scoped fixtures (one-episode smoke run artifact directory
   for F1 / F2 / F3 reuse, exported smoke bundle for F4 / F5, plus a
   `sample_bundle_dir` fixture pointing at
   `tests/fixtures/sample-bundle-v1/` for F4 / F5).
2. `tests/refactor_golden/__init__.py` — marker file only.
3. `tests/refactor_golden/helpers.py` — loaders and small assertion
   helpers shared by F1 – F6 (pure functions, no pytest state).

Fixtures are declared inline in the top-level test file rather than
through a `conftest.py` because they are used only by this test
module. Using inline fixtures avoids introducing a subdirectory
`conftest.py` whose scope would not cover a top-level test file.

No existing test file is modified.

## 8. Files Not Modified By Slice A

Explicitly, Slice A must not modify any file under:

1. `src/modqn_paper_reproduction/`,
2. `scripts/`,
3. `configs/`,
4. `tests/fixtures/sample-bundle-v1/`,
5. the existing test files under `tests/` (except to add a reference
   in the existing `pytest` config if strictly required for test
   discovery; that reference must be the minimum possible).

If any such file appears in the Slice A diff, the slice is rejected
and must be re-opened scoped correctly.

## 9. Expected Red-First Findings From Prior Audit

The pre-refactor audit identified these likely-but-not-certain
findings. Each is recorded here so the reviewer is not surprised:

1. F4 may be subtly red if any non-deterministic field (e.g. a
   wall-clock `exportedAt`) slipped into the shared dict; it would
   not be red today, but any Slice B work that reorganizes the
   pipeline would be likely to introduce it.
2. F6 may be red today because existing `generate_sample_bundle.py`
   runs go through `train_main()` which touches seeds via
   `np.random.SeedSequence`. If determinism holds, F6 is green;
   otherwise Slice A must record the non-determinism and coordinate
   a fix before Slice B.
3. F1 – F3 are expected green; any red finding should be investigated
   as a current-state surprise.

None of these expected findings are Slice A failures by themselves.
They are pre-Slice-A observability gains; that is the whole point of
Slice A.

## 10. Validation

Slice A is considered validated when:

1. all six test families are green on a clean `pytest` run,
2. the new test file is discoverable without disabling any existing
   test,
3. the new tests run under the repo's existing smoke-test timing
   budget without requiring training-scale compute,
4. a manual review confirms that every Slice A test asserts a
   specific invariant from `phase-04` section 4, not just an
   existence check.

## 11. Rollback Plan

Because Slice A adds only test code and one test helper module:

1. rollback is a single `git revert` of the landing commit,
2. no data, fixture, or production code needs to be restored,
3. no downstream consumer (`ntn-sim-core`) is affected,
4. no schema version changes.

If Slice A lands partially (for example, F1 – F5 green but F6
non-deterministic), the acceptable state is:

1. F1 – F5 remain landed,
2. F6 is marked `xfail` with a linked status note,
3. a separate bounded investigation fixes the determinism gap,
4. F6 is flipped to `pass` in a follow-on commit before Slice B
   begins.

## 12. Deliverables

Slice A is complete when:

1. `tests/test_refactor_golden.py` exists with F1 – F6 implemented,
2. `tests/refactor_golden/` helpers exist as specified in section 7,
3. all tests are green (or F6 is the only one marked `xfail` under
   the condition in section 11),
4. a short status note is committed at
   `artifacts/phase-04a-semantic-golden-status-YYYY-MM-DD.md`
   summarizing:
   - which of F1 – F6 are green,
   - any invariant surprise discovered,
   - whether Slice B is now justified.

Slice B is allowed to start only after this status note exists and
declares Slice A green.

## 13. Non-Escalation Rule

Slice A is not allowed to:

1. invent new required fields,
2. rename any existing field,
3. change any bundle, manifest, or timeline schema,
4. touch `contracts.py` / `settings.py`,
5. touch `env/step.py` semantics,
6. refactor any module under `src/`.

If any proposed Slice A change would require any of the above, that
proposal is Slice B or later work, and must wait for the matching
execution SDD.

## 14. Status

1. This SDD is drafted.
2. No test code has been written.
3. The next allowed action is to start implementing Slice A under
   this SDD.
