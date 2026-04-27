# modqn-paper-reproduction Agent Rules

This repository is intended to remain portable as a standalone training surface for
`PAP-2024-MORL-MULTIBEAM`.

## Scope

1. Work only inside this repository unless the user explicitly asks for external comparison.
2. Treat this repo as self-contained for training, smoke validation, and artifact review.
3. Do not assume a parent `papers/` workspace exists.

## Authority Order

When details conflict, use this order:

1. `paper-source/ref/2024_09_Handover_for_Multi-Beam_LEO_Satellite_Networks_A_Multi-Objective_Reinforcement_Learning_Method.pdf`
2. `paper-source/txt_layout/2024_09_Handover_for_Multi-Beam_LEO_Satellite_Networks_A_Multi-Objective_Reinforcement_Learning_Method.layout.txt`
3. `paper-source/catalog/PAP-2024-MORL-MULTIBEAM.json`
4. `docs/phases/phase-01-python-baseline-reproduction-sdd.md`
5. `docs/assumptions/modqn-reproduction-assumption-register.md`
6. `configs/modqn-paper-baseline.resolved-template.yaml`

## Training Rules

1. Only resolved-run configs may start training.
2. `configs/modqn-paper-baseline.yaml` is authority-only and must not be used as a training input.
3. Use the repo-local virtual environment when running tests or training: `.venv/`.
4. Keep active runtime assumptions visible in config, metadata, or both. Do not hide them in code-only defaults.

## Environment Bootstrap

For a new local environment, use:

```bash
python3 -m venv .venv
env PIP_CACHE_DIR=/tmp/pip-cache .venv/bin/python -m pip install -r requirements.txt
```

If Matplotlib cannot write to the home config directory, prefix training,
sweep, export, and pytest commands with:

```bash
env MPLCONFIGDIR=/tmp/modqn-mplconfig
```

Do not install dependencies into the system Python for repo work; use
the repo-local `.venv/`.

## Artifact Rules

1. Write training artifacts under `artifacts/`.
2. Keep `run_metadata.json`, `training_log.json`, and final checkpoints together per run.
3. Generated artifacts are outputs, not authority surfaces.

## Prompt Portability

1. Prompts for this repo should use repo-relative paths.
2. Do not reference `/home/u24/papers/...` paths unless the user explicitly provides that workspace.

## Read First

For a quick project-state handoff, read these before proposing new work:

1. `artifacts/modqn-current-direction-2026-04-22.md`
2. `artifacts/public-summary-2026-04-15-phase-01c-closeout.md`
3. `artifacts/phase-01c-closeout-status-2026-04-15.md`
4. `artifacts/phase-01b-closeout-status-2026-04-14.md`
5. `artifacts/reproduction-status-2026-04-13.md`
6. `artifacts/phase-01c-protocol-bounded-03/review.md`
7. `artifacts/run-9000/anomaly-review.md`
8. `docs/baseline-acceptance-checklist.md`

For regenerating curated OriginLab-ready plotting CSVs, also read:

1. `docs/origin-plot-data-runbook.md`
2. `artifacts/origin-plot-data/README.md`

For the planned low-coupling `ntn-sim-core` presentation follow-on, also
read:

1. `docs/phases/phase-03a-ntn-sim-core-bundle-replay-integration-sdd.md`

If the user explicitly asks whether reproduction work may reopen, also
read:

1. `docs/phases/phase-01d-reproduction-reopen-gate-sdd.md`
2. `artifacts/phase-01d-reopen-trigger-check-2026-04-16-producer-diagnostics.md`
3. `artifacts/phase-01e-beam-semantics-status-2026-04-22.md`
4. `artifacts/phase-01f-bounded-pilot-status-2026-04-22.md`
5. `artifacts/phase-01g-atmospheric-sign-status-2026-04-22.md`
6. `artifacts/modqn-current-direction-2026-04-22.md`

For current interpretation of the newer Phase 04 refactor materials,
also read:

1. `artifacts/phase-04-current-state-2026-04-19.md`
2. `docs/phases/phase-04-readme-summary.md`
3. `docs/phases/phase-04-refactor-contract-spine-sdd.md`
4. `docs/phases/phase-04a-refactor-semantic-golden-sdd.md`
5. `docs/phases/phase-04b-refactor-training-artifact-model-sdd.md`
6. `artifacts/phase-04b-training-artifact-model-status-2026-04-19.md`
7. `docs/phases/phase-04c-refactor-bundle-layer-split-sdd.md`
8. `artifacts/phase-04c-bundle-layer-split-status-2026-04-19.md`
9. `docs/phases/phase-04d-refactor-runtime-spine-split-sdd.md`
10. `docs/phases/phase-04e-refactor-sweep-analysis-plotting-split-sdd.md`
11. `artifacts/phase-04e-sweep-analysis-plotting-split-status-2026-04-19.md`

## Current State Snapshot

As of `2026-04-22`, the repo is a working standalone baseline
reproduction surface.
The frozen comparison-baseline bundle remains valid, and both the
paper-faithful scenario-correction follow-on and the comparator-protocol
follow-on have now been closed as negative results rather than as
upgrades to a full paper-faithful baseline. Later bounded reopen work
established a real beam-semantics issue and a useful beam-aware
follow-on surface, but it did not authorize default long retraining or a
new paper-faithful reproduction claim.

What is already true:

1. repo-only authority is stable
2. baseline training, resume, best-eval checkpointing, sweeps, export, and tests run in-repo
3. `Table II` and reviewed `Fig. 3` to `Fig. 6` sweep surfaces exist with machine-readable outputs
4. the comparison-baseline checklist is complete
5. the repo can now be frozen as a disclosed engineering baseline for downstream comparison
6. the highest-impact paper-backed scenario mismatch has been exercised on
   an explicit follow-on track
7. that follow-on track did not recover convincing paper-like method
   separation
8. the comparator-protocol follow-on is also complete
9. held-out reporting can reshuffle two already near-tied high-load
   `Fig. 3` points, but only through `r2` / handover
10. final-vs-best checkpoint reporting is a no-op on the bounded
    `20`-episode protocol surface
11. the Phase 01D external-comparison reopen trigger was satisfied only
    for additive producer diagnostics export
12. the Phase 03B producer-owned policy diagnostics slice is now landed
    as an exporter-only additive surface over the frozen replay bundle
13. the Phase 04A semantic-golden guardrail slice is now landed as the
    first internal hardening step
14. the Phase 04B training-artifact model seam is now landed as the
    second internal hardening step
15. the Phase 04C bundle-layer split is now landed as the third
    internal hardening step
16. the Phase 04D runtime-spine split is now landed as the fourth
    internal hardening step
17. the Phase 04E sweep/analysis/plotting split is now landed as the
    fifth internal hardening step
18. later Phase 04 slices remain internal hardening direction only and
    do not supersede the landed Phase 03A / 03B producer authority
19. Phase 01E established that the frozen baseline compresses the
    beam-level decision surface and justifies one bounded beam-aware
    follow-on
20. Phase 01F established that the beam-aware eligibility follow-on
    materially improves the bounded held-out surface and removes
    beam-collapse / comparator-degeneration on the audited surface
21. Phase 01G established that the corrected atmospheric sign has a
    real but modest diagnostic effect and does not by itself justify a
    new training branch
22. curated OriginLab-ready plotting CSVs should be regenerated through
    `docs/origin-plot-data-runbook.md` and tracked only under
    `artifacts/origin-plot-data/`

What is not yet established:

1. the paper's intended method separation has not been convincingly reproduced
2. `run-9000` shows a real late-training collapse and objective drift
3. `Table II` and `Fig. 3` are still near-tied, with most variation coming from `r2` / handover
4. the explicit reward-calibration pilot did not improve the raw eval surface
5. the Phase 01B scenario correction did not overturn the near-tie /
   dominance diagnosis
6. the Phase 01C comparator-protocol check did not overturn the same
   diagnosis
7. a broader globe-centric or same-page consumer presentation follow-on
   remains separate work
8. any landed Phase 04 internal hardening slice beyond Slice E

## Current Guardrails

1. Do not start a new `9000`-episode long run by default.
2. Do not start a new `500`-episode run by default.
3. Do not silently replace the baseline with `configs/modqn-paper-baseline.reward-calibration.resolved.yaml`.
4. Treat the reward-calibration config as an explicit experiment only.
5. Use `docs/baseline-acceptance-checklist.md` as the freeze note for comparison-baseline scope.
6. Treat Phase 01B and Phase 01C as closed unless a new explicitly
   labeled reopen surface is created.
7. If no new user direction is given, prefer freeze/disclosure over more
   retraining.
8. `docs/phases/phase-01d-reproduction-reopen-gate-sdd.md` is only a
   standby gate; it does not by itself authorize renewed reproduction
   implementation.
9. Treat `artifacts/modqn-current-direction-2026-04-22.md` as the
   current stop/recommendation note after Phase 01E / 01F / 01G.
10. Treat `artifacts/phase-04-current-state-2026-04-19.md` as the
   interpretation note for Phase 04. Slice A semantic-golden tests,
   Slice B's training-artifact model seam, Slice C's bundle-layer
   split, Slice D's runtime-spine split, and Slice E's
   sweep/analysis/plotting split are now landed internal guardrails,
   but the kickoff and later slices are not landed
   producer-contract changes unless a later explicit status note says
   so.
