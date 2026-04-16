# Phase 03B Producer Diagnostics Export Status

Date: `2026-04-16`

## Decision

`promote`

This bounded reopen slice landed successfully and now establishes the
producer-side optional diagnostics export surface. Downstream
`ntn-sim-core` consumer promotion remains a separate step that still
depends on consumer-side adoption.

## Scope Completed

1. added a separate greedy exporter helper,
   `MODQNTrainer.select_actions_with_diagnostics()`, without changing
   the existing `select_actions()` API or checkpoint-selection
   semantics
2. added optional `policyDiagnostics` rows to
   `timeline/step-trace.jsonl`
3. added `manifest.optionalPolicyDiagnostics` disclosure so older
   consumers can ignore the new surface while newer consumers can detect
   it explicitly
4. extended replay-bundle validation so inconsistent diagnostics fail
   loudly instead of being silently fabricated
5. refreshed `tests/fixtures/sample-bundle-v1/` so the checked-in sample
   bundle now carries the additive diagnostics surface

## Landed Export Shape

Each replay row may now carry an optional `policyDiagnostics` object
with:

1. `diagnosticsVersion`
2. `objectiveWeights`
   - named fields: `r1Throughput`, `r2Handover`, `r3LoadBalance`
3. `selectedScalarizedQ`
4. `runnerUpScalarizedQ`
5. `scalarizedMarginToRunnerUp`
6. `availableActionCount`
7. `topCandidates`

Each `topCandidates` entry carries:

1. stable serving identity: `beamId`, `beamIndex`, `satId`, `satIndex`,
   `localBeamIndex`
2. `validUnderDecisionMask`
3. `objectiveQ`
   - named fields: `r1Throughput`, `r2Handover`, `r3LoadBalance`
4. `scalarizedQ`

The manifest now discloses bundle-level presence and coverage through
`optionalPolicyDiagnostics`.

## Validation Evidence

Tests run:

1. `./.venv/bin/pytest tests/test_modqn_smoke.py::test_select_actions_with_diagnostics_respects_mask_and_ordering tests/test_modqn_smoke.py::test_select_actions_with_diagnostics_only_emits_when_available`
2. `./.venv/bin/pytest tests/test_sweeps_and_export.py::test_export_cli_emits_bundle tests/test_replay_bundle.py::test_validate_replay_bundle_rejects_misaligned_policy_diagnostics`
3. `./.venv/bin/python scripts/generate_sample_bundle.py --output tests/fixtures/sample-bundle-v1 --episodes 1 --max-users 1`
4. `./.venv/bin/pytest tests/test_modqn_smoke.py::test_select_actions_with_diagnostics_respects_mask_and_ordering tests/test_modqn_smoke.py::test_select_actions_with_diagnostics_only_emits_when_available tests/test_sweeps_and_export.py::test_export_cli_emits_bundle tests/test_replay_bundle.py`
5. `python3 -m compileall src/modqn_paper_reproduction`

What those checks prove:

1. diagnostics are emitted only when a stable masked decision surface is
   available
2. selected action, runner-up, margin, and candidate ordering stay
   aligned
3. invalid actions are excluded from the exported candidate list
4. manifest disclosure matches timeline coverage
5. a checked-in fixture now exists for downstream consumer ingestion

## Landing Judgment

This landed producer-side authority is justified because:

1. the surface is additive only
2. Phase 03A required field meaning is unchanged
3. replay truth still comes from the selected checkpoint greedy replay
4. the new fields are explicit enough that consumers do not need to
   invent missing producer semantics

Post-landing downstream consumer work may be evaluated from this landed
producer surface, but that consumer-side adoption is not established by
this repo-local validation alone.

## Residual Risks

1. the optional diagnostics are still limited to checkpoint-greedy
   scalarized-Q ranking; they do not explain broader training dynamics
2. downstream consumers still need to decide how much of
   `topCandidates` to render and how to disclose nullable runner-up
   fields when only one action is available
3. future producer revisions should treat
   `phase-03b-policy-diagnostics-v1` as a compatibility surface rather
   than silently reshaping it
