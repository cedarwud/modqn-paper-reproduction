# Phase 04 Current-State Interpretation

**Date:** `2026-04-19`
**Read with:**

1. [`../docs/phases/phase-04-refactor-contract-spine-sdd.md`](../docs/phases/phase-04-refactor-contract-spine-sdd.md)
2. [`../docs/phases/phase-04a-refactor-semantic-golden-sdd.md`](../docs/phases/phase-04a-refactor-semantic-golden-sdd.md)
3. [`../docs/phases/phase-04b-refactor-training-artifact-model-sdd.md`](../docs/phases/phase-04b-refactor-training-artifact-model-sdd.md)
4. [`./phase-04a-semantic-golden-status-2026-04-17.md`](./phase-04a-semantic-golden-status-2026-04-17.md)

## 1. Summary

This note is the current repo-state interpretation surface for Phase 04.

As of `2026-04-19`:

1. the landed producer authority still stops at the frozen comparison-baseline closeout plus the landed Phase 03A / Phase 03B producer surfaces,
2. Phase 04 is still best read as an **internal hardening track**, not as a landed external-contract change,
3. no Phase 04 refactor under `src/`, `scripts/`, or `configs/` is promoted by this note as a new external producer-contract change,
4. Phase 04A semantic golden tests are now landed as the first internal guardrail slice,
5. Phase 04B training-artifact model and serialization seam is now landed as the second internal guardrail slice,
6. later Phase 04 slices remain planning / in-flight follow-on work.

## 2. What Is Already Landed

The following remain the active landed authority surfaces:

1. [`./phase-01c-closeout-status-2026-04-15.md`](./phase-01c-closeout-status-2026-04-15.md) for comparison-baseline closeout,
2. [`../docs/baseline-acceptance-checklist.md`](../docs/baseline-acceptance-checklist.md) for freeze/claim scope,
3. [`../docs/phases/phase-03a-ntn-sim-core-bundle-replay-integration-sdd.md`](../docs/phases/phase-03a-ntn-sim-core-bundle-replay-integration-sdd.md) for the frozen replay bundle contract,
4. [`../docs/phases/phase-03b-ntn-sim-core-producer-diagnostics-export-sdd.md`](../docs/phases/phase-03b-ntn-sim-core-producer-diagnostics-export-sdd.md) plus [`./phase-03b-producer-diagnostics-export-status-2026-04-16.md`](./phase-03b-producer-diagnostics-export-status-2026-04-16.md) for the additive producer-owned diagnostics export,
5. `tests/fixtures/sample-bundle-v1/` as the current canonical downstream sample bundle fixture,
6. strict resolved-run guardrails described in `AGENTS.md` and `configs/modqn-paper-baseline.resolved-template.yaml`.

## 3. What Has Landed Inside Phase 04

The following surfaces are now landed as the first internal hardening
slice under Phase 04:

1. [`../docs/phases/phase-04a-refactor-semantic-golden-sdd.md`](../docs/phases/phase-04a-refactor-semantic-golden-sdd.md),
2. [`./phase-04a-semantic-golden-status-2026-04-17.md`](./phase-04a-semantic-golden-status-2026-04-17.md),
3. [`../tests/__init__.py`](../tests/__init__.py),
4. [`../tests/test_refactor_golden.py`](../tests/test_refactor_golden.py),
5. [`../tests/refactor_golden/__init__.py`](../tests/refactor_golden/__init__.py),
6. [`../tests/refactor_golden/helpers.py`](../tests/refactor_golden/helpers.py).

The following surfaces are now landed as the second internal hardening
slice under Phase 04:

1. [`../docs/phases/phase-04b-refactor-training-artifact-model-sdd.md`](../docs/phases/phase-04b-refactor-training-artifact-model-sdd.md),
2. [`./phase-04b-training-artifact-model-status-2026-04-19.md`](./phase-04b-training-artifact-model-status-2026-04-19.md),
3. [`../src/modqn_paper_reproduction/artifacts/__init__.py`](../src/modqn_paper_reproduction/artifacts/__init__.py),
4. [`../src/modqn_paper_reproduction/artifacts/models.py`](../src/modqn_paper_reproduction/artifacts/models.py),
5. [`../src/modqn_paper_reproduction/artifacts/io.py`](../src/modqn_paper_reproduction/artifacts/io.py),
6. [`../src/modqn_paper_reproduction/artifacts/paths.py`](../src/modqn_paper_reproduction/artifacts/paths.py),
7. [`../src/modqn_paper_reproduction/cli.py`](../src/modqn_paper_reproduction/cli.py),
8. [`../src/modqn_paper_reproduction/algorithms/modqn.py`](../src/modqn_paper_reproduction/algorithms/modqn.py),
9. [`../src/modqn_paper_reproduction/export/pipeline.py`](../src/modqn_paper_reproduction/export/pipeline.py),
10. [`../src/modqn_paper_reproduction/export/replay_bundle.py`](../src/modqn_paper_reproduction/export/replay_bundle.py),
11. [`../tests/test_artifacts_models.py`](../tests/test_artifacts_models.py).

The following checked-in planning surfaces still matter, but remain
planning / in-flight materials for later slices rather than landed
producer-boundary changes:

1. [`../docs/phases/phase-04-refactor-contract-spine-sdd.md`](../docs/phases/phase-04-refactor-contract-spine-sdd.md).

The practical interpretation is:

1. Slice A's semantic-golden guardrail is now landed as the first internal hardening seam,
2. Slice B's artifact-model seam is now landed as the second internal hardening seam,
3. no landed Phase 04 slice should be presented as a producer-contract change or as a reason to treat Phase 03A / Phase 03B as superseded,
4. the repo now has both a semantic regression gate and a typed training-artifact seam for later export / bundle refactor work.

## 4. Interpretation Rules

When Phase 04 materials are read together, use these rules:

1. Phase 03A / 03B remain the frozen external producer contract.
2. Phase 04 kickoff / slice docs are internal refactor-planning surfaces only unless a later landed slice status note says otherwise.
3. The Phase 04A and Phase 04B status notes record landed internal guardrail slices, not a new repo-landed closeout for the external producer contract.
4. Any future claim that Phase 04 as a whole is landed should be paired with:
   - committed test or code surfaces,
   - synchronized README / docs indexes,
   - and an explicit status note that supersedes this interpretation note.
5. Until then, integration planning should assume:
   - producer remains standalone and bundle-driven,
   - external bundle/fixture compatibility is frozen,
   - internal hardening may proceed later, but has not changed the published producer boundary.

## 5. Immediate Next Step

Before any later code refactor begins, the repo should keep the
following documentation state clear:

1. the standalone producer / exporter / truth-source role is still the primary architectural decision,
2. the frozen external surfaces are the replay bundle, additive diagnostics, sample fixture, and resolved-run guardrails,
3. Phase 04A and Phase 04B are now landed internal guardrail slices,
4. later Phase 04 cleanup beyond Slice B remains in-flight and should not be mistaken for a completed refactor program.
