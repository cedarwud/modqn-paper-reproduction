# Phase 01F Slice 01F1 Status

Date: `2026-04-22`

## Scope

This note records `Phase 01F / Slice 01F1` only:

1. add one opt-in beam-aware eligibility runtime surface
2. keep the frozen baseline default unchanged
3. verify the new surface with a real `1`-episode smoke run plus export

This slice does **not** include the bounded `20` / `200` pilots yet.

## Landed Runtime Surface

The repo now supports two explicit eligibility modes:

1. `satellite-visible-all-beams`
   - frozen baseline default
2. `nearest-beam-per-visible-satellite`
   - new `Phase 01F` follow-on mode

The new mode is wired through:

1. resolved-run config
2. environment step runtime
3. `run_metadata.json`
4. export `manifest.json`

The frozen baseline mode remains unchanged and remains the default.

## New Follow-On Config

Config:

- `configs/modqn-paper-baseline.beam-aware-eligibility-follow-on.resolved.yaml`

Track:

1. `phase: phase-01f`
2. `label: beam-aware-eligibility-follow-on`
3. `status: experimental`

## Real Smoke Artifact

Training artifact:

- `artifacts/codex-beam-aware-smoke-1ep-2026-04-22/`

Export artifact:

- `artifacts/codex-beam-aware-smoke-1ep-2026-04-22/export-bundle/`

Observed result:

1. training completed `1/1` episodes
2. final scalar reward: `487.01186931800845`
3. final checkpoint and best-eval checkpoint were both written
4. export completed successfully
5. both metadata and manifest disclose
   `nearest-beam-per-visible-satellite`

## Validation

Automated validation completed:

1. `tests/test_step.py`
2. `tests/test_modqn_smoke.py`
3. `tests/test_artifacts_models.py`
4. `tests/test_training_hardening.py`

Relevant new assertions cover:

1. runtime config selects the new eligibility mode only when requested
2. nearest-beam masks keep at most one valid beam per visible satellite
3. training metadata records the active eligibility mode
4. export manifest records the active eligibility mode

## Decision

`Slice 01F1 is complete.`

The next justified step is:

1. `Slice 01F2 / 01F3`
2. run bounded beam-aware pilots:
   - `20` episodes
   - `200` episodes
3. compare against the frozen baseline pilot surfaces before considering
   any larger training

Do **not** start `500` or `9000` episodes from this slice alone.
