# Cross-Repo Integration Handoff

**Date:** `2026-04-19`
**Producer repo:** `modqn-paper-reproduction`
**Result:** `HANDOFF-READY`

## 1. Producer Boundary To Treat As Frozen

The producer-side authority that downstream repos should integrate
against is:

1. Phase 03A replay bundle contract,
2. Phase 03B additive `policyDiagnostics` surface,
3. `tests/fixtures/sample-bundle-v1/` as the canonical sample bundle,
4. baseline / closeout claim boundary,
5. strict resolved-run config guardrails.

`Phase 04A` through `Phase 04E` are landed internal hardening slices
only. They improve the producer's internal seam quality, but they do not
change the external producer contract.

## 2. Current Cross-Repo State

As of this handoff:

1. `modqn-paper-reproduction` is ready for downstream integration prep
   as a standalone producer / exporter / truth-source,
2. `ntn-sim-core` already has a landed replay-bundle consumer path,
3. `ntn-sim-core` already has a landed bundle replay UI path,
4. `ntn-sim-core` already has a landed consumer-side diagnostics /
   explainability follow-on for the producer-owned `policyDiagnostics`
   surface,
5. the next unlanded integration target is therefore not baseline
   bundle consumption inside `ntn-sim-core`, but a broader
   globe-centric / same-page consumer follow-on.

## 3. Recommended Authority Chain For The Next Planning Thread

For the next cross-repo planning thread, read in this order:

1. `modqn-paper-reproduction/artifacts/phase-04-integration-readiness-2026-04-19.md`
2. `modqn-paper-reproduction/artifacts/phase-04-current-state-2026-04-19.md`
3. `modqn-paper-reproduction/docs/phases/phase-03a-ntn-sim-core-bundle-replay-integration-sdd.md`
4. `modqn-paper-reproduction/docs/phases/phase-03b-ntn-sim-core-producer-diagnostics-export-sdd.md`
5. `ntn-sim-core/sdd/modqn-bundle-replay-consumer-sdd.md`
6. `ntn-sim-core/sdd/modqn-bundle-replay-ui-sdd.md`
7. `ntn-sim-core/sdd/modqn-producer-diagnostics-consumer-follow-on.md`
8. `scenario-globe-viewer/README.md`

## 4. What The Next Thread Should Not Reopen

The next planning thread should not reopen:

1. producer-side Phase 04 internal refactor by default,
2. Python trainer porting into `ntn-sim-core`,
3. replay-bundle or `policyDiagnostics` contract redefinition,
4. paper-faithful reproduction claims,
5. sample-bundle fixture meaning.

## 5. Recommended Next Step

Start a consumer-side planning thread that answers only this question:

1. how should the already-frozen producer artifact surfaces be consumed
   by the next globe-centric / same-page presentation path?

That planning thread should produce a new consumer-side SDD or handoff
surface in the target repo, not another producer-side refactor slice in
this repo.
