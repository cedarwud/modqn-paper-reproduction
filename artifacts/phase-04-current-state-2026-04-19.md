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
2. the checked-in and local-working-tree Phase 04 materials are best read as **in-flight internal hardening guidance**, not as a landed external-contract change,
3. no Phase 04 refactor under `src/`, `scripts/`, or `configs/` is promoted by this note,
4. Phase 04A semantic golden tests are the preferred next guardrail, but they should be treated as an in-flight prerequisite gate until the docs/tests are fully reconciled into the landed repo authority chain.

## 2. What Is Already Landed

The following remain the active landed authority surfaces:

1. [`./phase-01c-closeout-status-2026-04-15.md`](./phase-01c-closeout-status-2026-04-15.md) for comparison-baseline closeout,
2. [`../docs/baseline-acceptance-checklist.md`](../docs/baseline-acceptance-checklist.md) for freeze/claim scope,
3. [`../docs/phases/phase-03a-ntn-sim-core-bundle-replay-integration-sdd.md`](../docs/phases/phase-03a-ntn-sim-core-bundle-replay-integration-sdd.md) for the frozen replay bundle contract,
4. [`../docs/phases/phase-03b-ntn-sim-core-producer-diagnostics-export-sdd.md`](../docs/phases/phase-03b-ntn-sim-core-producer-diagnostics-export-sdd.md) plus [`./phase-03b-producer-diagnostics-export-status-2026-04-16.md`](./phase-03b-producer-diagnostics-export-status-2026-04-16.md) for the additive producer-owned diagnostics export,
5. `tests/fixtures/sample-bundle-v1/` as the current canonical downstream sample bundle fixture,
6. strict resolved-run guardrails described in `AGENTS.md` and `configs/modqn-paper-baseline.resolved-template.yaml`.

## 3. What Exists But Is Still In-Flight

The following checked-in or doc-level surfaces now exist and matter,
but they are not yet the landed repo-state authority for
"Phase 04 completed":

1. [`../docs/phases/phase-04-refactor-contract-spine-sdd.md`](../docs/phases/phase-04-refactor-contract-spine-sdd.md),
2. [`../docs/phases/phase-04a-refactor-semantic-golden-sdd.md`](../docs/phases/phase-04a-refactor-semantic-golden-sdd.md),
3. [`../docs/phases/phase-04b-refactor-training-artifact-model-sdd.md`](../docs/phases/phase-04b-refactor-training-artifact-model-sdd.md),
4. [`./phase-04a-semantic-golden-status-2026-04-17.md`](./phase-04a-semantic-golden-status-2026-04-17.md).

The following additional local candidate surfaces may exist in the
working tree and are relevant to Phase 04A interpretation, but should
not be described as landed repo contents unless and until they are
committed together with a matching landed status note:

1. [`../tests/test_refactor_golden.py`](../tests/test_refactor_golden.py),
2. [`../tests/refactor_golden/helpers.py`](../tests/refactor_golden/helpers.py).

The practical interpretation is:

1. Slice A's semantic-golden direction is approved as the next internal hardening seam,
2. Slice B's artifact-model direction is approved as a likely follow-on if Slice A is fully reconciled,
3. neither slice should yet be presented as a landed producer-contract change or as a reason to treat Phase 03A / 03B as superseded,
4. local candidate test surfaces should be discussed as working-tree or promotion-candidate evidence, not as already-landed repo guardrails.

## 4. Interpretation Rules

When Phase 04 materials are read together, use these rules:

1. Phase 03A / 03B remain the frozen external producer contract.
2. Phase 04 kickoff / slice docs are internal refactor-planning surfaces only.
3. The Phase 04A status note records a local promotion-candidate result, not by itself a new repo-landed closeout.
4. Any future claim that Phase 04A is landed should be paired with:
   - committed test surfaces,
   - synchronized README / docs indexes,
   - and an explicit status note that supersedes this interpretation note.
5. Until then, integration planning should assume:
   - producer remains standalone and bundle-driven,
   - external bundle/fixture compatibility is frozen,
   - internal hardening may proceed later, but has not yet changed the published producer boundary.

## 5. Immediate Document-Only Next Step

Before any code refactor begins, the repo should keep the following documentation state clear:

1. the standalone producer / exporter / truth-source role is still the primary architectural decision,
2. the frozen external surfaces are the replay bundle, additive diagnostics, sample fixture, and resolved-run guardrails,
3. Phase 04 is the approved internal cleanup direction, but is still in-flight and should not be mistaken for a completed refactor.
