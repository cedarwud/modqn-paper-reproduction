# Phase SDD Index

Current landed producer authority stops at the Phase 03A replay bundle
contract plus the Phase 03B additive diagnostics export. Under Phase 04,
Slices A / B / C / D are now landed as internal guardrail slices, while
the broader cleanup track still does not revise the frozen external
contract.

Short read before the detailed Phase 04 docs:

- `phase-04-readme-summary.md`

1. `phase-01-python-baseline-reproduction-sdd.md`
   Python-first paper baseline reproduction.
2. `phase-01b-paper-faithful-follow-on-sdd.md`
   Explicit experimental follow-on for paper-fidelity corrections after the comparison-baseline freeze.
3. `phase-01b-slice-c-targeted-high-load-follow-on-sdd.md`
   Execution SDD for targeted high-load `Fig. 3` validation plus a reward-geometry/protocol decision gate.
4. `phase-01c-comparator-protocol-experiment-sdd.md`
   Explicit bounded comparator-protocol follow-on after the Phase 01B closeout.
5. `phase-01d-reproduction-reopen-gate-sdd.md`
   Standby reopen gate that defines when reproduction work may be restarted after the Phase 01C stop.
6. `phase-02-artifact-bridge-sdd.md`
   Stable export bundle from Python reproduction to external consumers.
7. `phase-03-ntn-sim-core-visual-integration-sdd.md`
   3D visualization and UI consumption in `ntn-sim-core`.
8. `phase-03a-ntn-sim-core-bundle-replay-integration-sdd.md`
   Promoted follow-on for low-coupling bundle replay integration into `ntn-sim-core`.
9. `phase-03b-ntn-sim-core-producer-diagnostics-export-sdd.md`
   Landed bounded reopen slice for the additive producer-owned policy
   diagnostics export needed by downstream explainability consumers.
10. `phase-04-readme-summary.md`
    Short interpretation surface for Phase 04. Explains what the newer
    refactor materials are, what they are not, and when they may be
    treated as landed.
11. `phase-04-refactor-contract-spine-sdd.md`
    Draft kickoff SDD authorizing a future internal refactor that
    promotes the de facto producer contract into an explicit model
    layer. It freezes Phase 03A/03B external surfaces, fixes slice
    ordering, and defers per-slice design to later execution SDDs.
12. `phase-04a-refactor-semantic-golden-sdd.md`
    Landed execution slice for Phase 04 Slice A. Adds artifact-level
    semantic golden tests (`run_metadata`, `training_log`, checkpoint
    payload, manifest↔summary cross-file consistency, timeline
    geometry, fixture regeneration determinism) without touching `src/`.
13. `phase-04b-refactor-training-artifact-model-sdd.md`
    Landed execution slice for Phase 04 Slice B. Introduces
    `RunMetadataV1`, `TrainingLogRow`, `CheckpointPayloadV1`,
    `CheckpointCatalog`, `RunArtifactPaths` under a new `artifacts/`
    package, rewires `cli.py` / exporter / trainer to write and read
    through those models, keeps output byte-stable.
14. `phase-04c-refactor-bundle-layer-split-sdd.md`
    Landed execution slice for Phase 04 Slice C. Splits the replay-bundle
    monolith into schema / serializers / provenance / validator /
    fixture-tools modules, adds `artifacts/compat.py`, and single-sources
    `ReplaySummary` across `manifest.replaySummary` and
    `evaluation.summary.replay_timeline`.
15. `phase-04d-refactor-runtime-spine-split-sdd.md`
    Landed execution slice for Phase 04 Slice D. Introduces the
    `runtime/` package for trainer spec, state encoding, objective
    math, replay buffer, and Q-network seams while preserving
    `algorithms/modqn.py` as a façade.
