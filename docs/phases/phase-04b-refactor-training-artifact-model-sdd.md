# Phase 04B: Refactor Slice B — Training Artifact Model SDD

**Status:** Landed execution slice; promoted via `phase-04b-training-artifact-model-status-2026-04-19.md`.
**Date:** `2026-04-17`
**Parent kickoff:**
[`phase-04-refactor-contract-spine-sdd.md`](./phase-04-refactor-contract-spine-sdd.md)
**Slice A status:**
[`../../artifacts/phase-04a-semantic-golden-status-2026-04-17.md`](../../artifacts/phase-04a-semantic-golden-status-2026-04-17.md)
**Current interpretation:** [`../../artifacts/phase-04-current-state-2026-04-19.md`](../../artifacts/phase-04-current-state-2026-04-19.md)
**Depends on:**

1. [`phase-04-refactor-contract-spine-sdd.md`](./phase-04-refactor-contract-spine-sdd.md)
2. [`phase-04a-refactor-semantic-golden-sdd.md`](./phase-04a-refactor-semantic-golden-sdd.md)
3. [`../../artifacts/phase-04a-semantic-golden-status-2026-04-17.md`](../../artifacts/phase-04a-semantic-golden-status-2026-04-17.md)

## 1. Purpose

Slice B lifts the implicit training-side artifact contract
(`run_metadata.json`, `training_log.json`, checkpoint payload) out of
hand-built dicts in `cli.py` / `algorithms/modqn.py` and into explicit,
named model types.

The goal is **typed boundary, unchanged output**:

1. every existing artifact field continues to be written with the
   same key, same value, same JSON shape,
2. consumer modules (`export/pipeline.py`, `export/replay_bundle.py`,
   `scripts/generate_sample_bundle.py`) read through the model layer
   instead of indexing raw dicts,
3. Slice A's golden tests remain green without modification.

This SDD now records the landed internal hardening slice for the
training-artifact model seam. It does not promote Phase 04 as a whole
to landed repo authority; read the linked current-state interpretation
note before using this slice as evidence of any external producer-contract
change.

## 2. Why Slice B Goes Second

The landed Slice A guardrail confirms that the contract is real enough
to enforce with cross-file tests. Slice B upgrades the contract from
"test-enforced" to "type-enforced". Without Slice B:

1. Slice C (bundle-layer split) has to keep indexing raw dicts from
   training output, multiplying the number of files that encode the
   same implicit shape;
2. Slice D (runtime spine split) cannot move `TrainerConfig` to
   `runtime/` without simultaneously rediscovering every call site
   that produces its `asdict` form into artifacts;
3. any drift between CLI writing and exporter reading stays silent
   until one of Slice A's cross-file invariants breaks.

Slice B is therefore the structural prerequisite for Slice C and D.

## 3. Scope

### 3.1 In Scope

1. a new package `src/modqn_paper_reproduction/artifacts/` containing:
   - `models.py` — dataclasses for every structural boundary of the
     training artifact contract,
   - `io.py` — write / read helpers that serialize and deserialize
     models against the current JSON format,
   - `paths.py` — `RunArtifactPaths` helper that centralizes path
     conventions inside a run directory,
2. `cli.py` is rewritten to build `RunMetadataV1` /
   `TrainingLogRow` collections / `CheckpointCatalog` objects and
   hand them to `artifacts/io.py` for serialization, instead of
   constructing dicts inline,
3. `algorithms/modqn.py.build_checkpoint_payload()` returns a
   `CheckpointPayloadV1` envelope (with tensor state-dict blocks
   carried as untyped fields), and the load path accepts the same
   envelope,
4. `export/pipeline.py` and `export/replay_bundle.py` read
   `RunMetadataV1` / `CheckpointPayloadV1` through `artifacts/io.py`
   instead of indexing raw dicts,
5. `scripts/generate_sample_bundle.py` unchanged except if it
   currently reads artifact dicts directly (it does not — it reads
   bundle JSON from the trimmed fixture, which Slice B does not
   reshape).

### 3.2 Explicitly Out Of Scope

1. any change to the written JSON shape, key order, or value
   semantics of `run_metadata.json` / `training_log.json` /
   checkpoint payload,
2. normalizing `config_path` (see section 4.2 of the Slice A status
   note) — Slice B records the behavior as "preserve argv verbatim"
   and does not normalize; any normalization is a separate future
   amendment,
3. splitting `export/replay_bundle.py` (Slice C),
4. moving `TrainerConfig` out of `algorithms/modqn.py` (Slice D),
5. any change to `StepEnvironment`, reward semantics, or action
   ordering,
6. any touch to `contracts.py` / `settings.py` — they remain the
   same non-live files they are today,
7. any change to `bundle/manifest.json` or `timeline/step-trace.jsonl`
   — Slice B covers the training artifact, not the bundle surface.

## 4. Frozen Surfaces Slice B Must Leave Byte-Stable

Restated from `phase-04` section 4 so this SDD is self-contained:

1. every key / subkey / value shape in `run_metadata.json`, per the
   current `cli.py` writer,
2. every key / value in `training_log.json` per-episode rows, and
   the JSON-array-of-objects outer shape,
3. checkpoint payload with `format_version == 1` and its current
   required key set plus optional `optimizers` / `last_episode_log` /
   `evaluation_summary` keys,
4. best-eval checkpoint carries `evaluation_summary` with all 13
   documented inner keys,
5. `tests/fixtures/sample-bundle-v1/` remains byte-equal after Slice B.

These surfaces are already enforced by Slice A tests F1–F6. Slice B
must not weaken or delete any of those tests.

## 5. Stop Conditions Applying To Slice B

All six kickoff stop conditions (section 8 of `phase-04`) apply. The
two most likely to fire during Slice B are:

1. a model type cannot express the current artifact shape without
   losing a field (kickoff stop #1 — a section 4 surface would have
   to change). In that case Slice B preserves the field verbatim and
   records the gap as a known "weak type" in an amendment;
2. checkpoint payload byte content changes after round-trip through
   the new envelope (kickoff stop #5). In that case Slice B stops
   and investigates before landing.

Slice B must not delete a Phase 03A / 03B required field to make the
model types cleaner. Field preservation always wins over type
aesthetics.

## 6. Model Type Design

### 6.1 New Package Layout

```text
src/modqn_paper_reproduction/artifacts/
    __init__.py
    models.py
    io.py
    paths.py
```

### 6.2 `models.py`

All models are `@dataclass(frozen=True)` unless noted. Every model
exposes `to_dict() -> dict[str, Any]` and a classmethod
`from_dict(cls, payload: dict[str, Any]) -> Self`.

```python
@dataclass(frozen=True)
class SeedsBlock:
    train_seed: int
    environment_seed: int
    mobility_seed: int
    evaluation_seed_set: tuple[int, ...]

@dataclass(frozen=True)
class CheckpointRuleV1:
    assumption_id: str
    primary_report: str
    secondary_report: str
    secondary_implemented: bool
    secondary_status: str

@dataclass(frozen=True)
class CheckpointFilesV1:
    primary_final: str
    secondary_best_eval: str | None

@dataclass(frozen=True)
class RewardCalibrationV1:
    enabled: bool
    mode: str
    source: str
    scales: tuple[float, float, float]
    training_experiment_kind: str
    training_experiment_id: str
    evaluation_metrics: str
    checkpoint_selection_metric: str

@dataclass(frozen=True)
class RuntimeEnvironmentV1:
    num_users: int
    num_satellites: int
    beams_per_satellite: int
    user_lat_deg: float
    user_lon_deg: float
    r3_gap_scope: str
    r3_empty_beam_throughput: float
    user_heading_stride_rad: float
    user_scatter_radius_km: float
    user_scatter_distribution: str
    user_area_width_km: float
    user_area_height_km: float
    mobility_model: str
    random_wandering_max_turn_rad: float

@dataclass(frozen=True)
class TrainingSummaryV1:
    episodes_requested: int
    episodes_completed: int
    elapsed_s: float
    final_episode_index: int
    final_scalar_reward: float | None

@dataclass(frozen=True)
class ResumeFromV1:
    path: str
    checkpoint_kind: str | None
    episode: int | None

@dataclass(frozen=True)
class RunMetadataV1:
    paper_id: str
    package_version: str
    config_path: str                         # preserve argv verbatim
    config_role: str | None
    resolved_config_snapshot: dict[str, Any]  # opaque pass-through
    training_experiment: dict[str, Any] | None
    seeds: SeedsBlock
    checkpoint_rule: CheckpointRuleV1
    reward_calibration: RewardCalibrationV1
    checkpoint_files: CheckpointFilesV1
    resolved_assumptions: dict[str, Any]     # opaque pass-through
    runtime_environment: RuntimeEnvironmentV1
    trainer_config: dict[str, Any]           # Slice D will model this
    best_eval_summary: dict[str, Any] | None # uses EvalSummary asdict shape
    resume_from: ResumeFromV1 | None
    training_summary: TrainingSummaryV1

@dataclass(frozen=True)
class TrainingLogRow:
    episode: int
    epsilon: float
    r1_mean: float
    r2_mean: float
    r3_mean: float
    scalar_reward: float
    total_handovers: int
    replay_size: int
    losses: tuple[float, float, float]

@dataclass(frozen=True)
class CheckpointPayloadV1:
    format_version: int = 1                  # must always be 1
    checkpoint_kind: str
    episode: int
    train_seed: int
    env_seed: int
    mobility_seed: int
    state_dim: int
    action_dim: int
    trainer_config: dict[str, Any]           # Slice D will model this
    checkpoint_rule: CheckpointRuleV1
    q_networks: list[dict[str, Any]]         # torch state dicts, opaque
    target_networks: list[dict[str, Any]]
    optimizers: list[dict[str, Any]] | None = None
    last_episode_log: dict[str, Any] | None = None
    evaluation_summary: dict[str, Any] | None = None

@dataclass(frozen=True)
class CheckpointCatalog:
    """Paths to the two checkpoint files produced by one training run."""
    primary_final: Path
    secondary_best_eval: Path | None

    def to_v1(self) -> CheckpointFilesV1: ...
```

### 6.3 `paths.py`

```python
@dataclass(frozen=True)
class RunArtifactPaths:
    run_dir: Path

    @property
    def training_log_json(self) -> Path: ...
    @property
    def run_metadata_json(self) -> Path: ...
    @property
    def checkpoints_dir(self) -> Path: ...

    def primary_checkpoint(self, rule: CheckpointRuleV1) -> Path: ...
    def secondary_checkpoint(self, rule: CheckpointRuleV1) -> Path: ...
```

### 6.4 `io.py`

Public surface:

```python
def write_run_metadata(path: Path, metadata: RunMetadataV1) -> None: ...
def read_run_metadata(path: Path) -> RunMetadataV1: ...

def write_training_log(path: Path, rows: Iterable[TrainingLogRow]) -> None: ...
def read_training_log(path: Path) -> list[TrainingLogRow]: ...

def write_checkpoint(path: Path, payload: CheckpointPayloadV1) -> Path: ...
def read_checkpoint(path: Path) -> CheckpointPayloadV1: ...
```

`write_*` helpers must produce JSON that is byte-equal (or at least
semantically-equal under the current `json.dumps(..., indent=2)`
formatting) to the current output.

## 7. Files Added / Modified / Not Modified

### 7.1 Added

1. `src/modqn_paper_reproduction/artifacts/__init__.py`
2. `src/modqn_paper_reproduction/artifacts/models.py`
3. `src/modqn_paper_reproduction/artifacts/io.py`
4. `src/modqn_paper_reproduction/artifacts/paths.py`
5. `tests/test_artifacts_models.py` — round-trip unit tests for every
   model in section 6.2, plus explicit assertions that writing-then-
   reading preserves byte-content.

### 7.2 Modified

1. `src/modqn_paper_reproduction/cli.py`
   - `train_main()` builds `RunMetadataV1` + `list[TrainingLogRow]` +
     `CheckpointCatalog` and hands them to `artifacts/io.py`,
   - hand-built dict literals for `metadata` and training-log rows
     are removed.
2. `src/modqn_paper_reproduction/algorithms/modqn.py`
   - `build_checkpoint_payload()` returns a `CheckpointPayloadV1`
     instead of `dict[str, Any]`,
   - `save_checkpoint()` / `save_best_eval_checkpoint()` call
     `artifacts.io.write_checkpoint()`,
   - `load_checkpoint()` calls `artifacts.io.read_checkpoint()` and
     converts back to the same internal state updates it does today;
     no call-site behavior changes.
3. `src/modqn_paper_reproduction/export/pipeline.py`
   - `export_training_run()` reads `RunMetadataV1` via
     `artifacts/io.py` instead of `json.loads(...read_text())`,
   - downstream dict access is replaced with typed attribute access.
4. `src/modqn_paper_reproduction/export/replay_bundle.py`
   - `resolve_training_config_snapshot()`,
     `select_replay_checkpoint()`, `_select_timeline_seed()`, and
     `export_replay_bundle()` read `RunMetadataV1` / `CheckpointPayloadV1`
     through `artifacts/io.py`,
   - the file does not change its bundle-side responsibilities in
     this slice.

### 7.3 Not Modified

1. `src/modqn_paper_reproduction/config_loader.py` — Slice B leaves
   its `algorithms.modqn.TrainerConfig` import alone; Slice D handles
   the reverse dependency.
2. `src/modqn_paper_reproduction/env/*` — out of scope.
3. `src/modqn_paper_reproduction/contracts.py` /
   `src/modqn_paper_reproduction/settings.py` — Slice B does not
   promote, rewrite, or delete these. They remain non-live.
4. `scripts/generate_sample_bundle.py` — unless a tactical change
   is strictly required; if it must change, the change is limited
   to using `artifacts/io.py` read helpers to normalize fixture
   manifests, and must be reviewed against the Slice A F6 byte-
   equality constraint.
5. `configs/*.yaml`.
6. `tests/test_refactor_golden.py` and all other existing tests.
7. `tests/fixtures/sample-bundle-v1/`.

## 8. Validation

Slice B is considered validated when all of the following are green
on a clean checkout:

1. `pytest tests/test_refactor_golden.py` — all 11 Slice A tests
   pass with zero changes to the test file (section 7.3 #6),
2. `pytest` — all 215 existing-plus-Slice-A tests still pass,
3. `pytest tests/test_artifacts_models.py` — new round-trip tests:
   - one per model that `to_dict()` then `from_dict()` returns an
     equal instance,
   - one per model that `write_*` then `read_*` against a temp path
     yields an equal instance,
   - one test that reads the checked-in
     `tests/fixtures/sample-bundle-v1/evaluation/summary.json`'s
     `best_eval_summary` block and shapes it into `EvalSummary` /
     `RunMetadataV1.best_eval_summary` without loss,
   - one test that runs `train_main()` at `--episodes 1`, reads the
     resulting `run_metadata.json` with
     `artifacts.io.read_run_metadata()`, and asserts the re-serialized
     output is byte-equal to the original file.

Slice A's F6 (sample-bundle regeneration determinism) is the load-
bearing regression test: if it still passes after Slice B lands, the
whole surface is byte-stable.

## 9. Rollback Plan

Slice B touches `src/` files, so rollback is less trivial than
Slice A's single `git revert`, but it remains a single-commit-or-PR
revert because:

1. the new `artifacts/` package is self-contained and additive,
2. `cli.py` / `export/pipeline.py` / `export/replay_bundle.py` /
   `algorithms/modqn.py` edits are localized to the dict-building and
   dict-reading call sites that Slice B converts,
3. no file format, key, or value changes, so downstream consumers
   (including the fixture and any prior run artifact) remain
   compatible before and after.

If Slice A tests are observed to regress at any point during Slice B
development, the slice must be rolled back and re-opened. Slice A
regression is the primary signal that Slice B drifted the contract.

## 10. Deliverables

Slice B is complete when:

1. the four new files in `src/modqn_paper_reproduction/artifacts/`
   are landed,
2. the four modified src files are rewritten against the model
   layer with no behavior change,
3. `tests/test_artifacts_models.py` is landed and green,
4. `tests/test_refactor_golden.py` still passes without
   modification,
5. the full suite passes,
6. a status note is committed at
   `artifacts/phase-04b-training-artifact-model-status-YYYY-MM-DD.md`
   summarizing:
   - list of fields successfully modeled,
   - list of fields left as opaque dict (should match section 6.2's
     comments: `resolved_config_snapshot`, `resolved_assumptions`,
     `training_experiment`, `trainer_config`, torch state dicts),
   - any preservation concession (see section 5 stop #1),
   - whether Slice C is now justified.

Slice C is allowed to start only after this status note exists and
declares Slice B green.

## 11. Non-Escalation Rule

Slice B is not allowed to:

1. change any JSON key name or value semantics,
2. introduce path normalization (e.g. for `config_path`) even
   though the Slice A status note flags it as a worthwhile future
   change,
3. delete `contracts.py` / `settings.py`,
4. change the `algorithms.modqn.TrainerConfig` public location
   (that is Slice D),
5. split `export/replay_bundle.py` (that is Slice C),
6. change the checkpoint `.pt` binary shape beyond rewrapping keys
   into `CheckpointPayloadV1` — torch state dicts must round-trip
   byte-equal.

If any proposed Slice B change would require any of the above, it is
Slice C or later work, and must wait for the matching execution SDD.

## 12. Input From Slice A Status Note

Slice A status note section 4.2 records that `run_metadata.config_path`
preserves argv verbatim (relative or absolute), and that this string
propagates into `manifest.configPath` and
`evaluation/summary.json.config_path`. Slice B explicitly chooses to
**preserve this behavior unchanged**, because normalizing it would:

1. change the fixture bytes and break Slice A F6,
2. change a field whose historical meaning has not been defined
   elsewhere,
3. exceed "typed boundary, unchanged output".

Slice B does record the behavior in `RunMetadataV1`'s docstring as
"`config_path` is the argv path passed to `train_main` at write time;
its relative/absolute form is intentionally preserved. Any future
normalization is a separate amendment."

## 13. Status

1. This SDD is drafted.
2. No code has been written.
3. The next allowed action is to start implementing Slice B under
   this SDD.
