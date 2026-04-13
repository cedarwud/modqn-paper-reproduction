# Baseline Acceptance Checklist

Date: `2026-04-13`

This checklist defines the current closeout target for
`modqn-paper-reproduction`: freeze the repo as a **disclosed comparison
baseline** without over-claiming a full paper-faithful reproduction.

It does not replace:

1. `docs/phases/phase-01-python-baseline-reproduction-sdd.md`
2. `docs/assumptions/modqn-reproduction-assumption-register.md`
3. `artifacts/reproduction-status-2026-04-13.md`

Use this checklist when deciding whether the repo is ready to stop
baseline hardening and shift either to downstream comparison work or to a
new explicitly experimental track.

## Claim Boundary

If every required item below is closed, the repo may be presented as:

1. a repo-only standalone reproduction surface
2. a disclosed comparison baseline
3. an artifact bundle with explicit assumptions and known limitations

Even after this checklist is complete, the repo should **not** be
presented as a fully paper-faithful reproduction unless the separate
scientific gaps listed near the end of this document are also closed.

## Required For Comparison-Baseline Freeze

### 1. Authority And Guardrails

- [x] Repo-local authority is stable and portable.
- [x] Real training only starts from resolved-run configs.
- [x] `configs/modqn-paper-baseline.yaml` remains authority-only and not a legal training input.
- [x] `configs/modqn-paper-baseline.reward-calibration.resolved.yaml` remains an explicit experiment rather than a silent baseline replacement.
- [x] Current handoff docs state that the repo is a disclosed baseline, not a full paper-faithful reproduction.

### 2. Baseline Runtime Surface

- [x] Baseline training runs end to end in-repo.
- [x] Resume support runs in-repo.
- [x] Best-eval checkpoint selection is implemented and validated.
- [x] Comparator sweeps run in-repo.
- [x] Export surfaces run in-repo.
- [x] Smoke and hardening tests pass in the repo-local environment.

### 3. Required Baseline Evidence Bundle

- [x] A reviewed baseline pilot artifact exists: `artifacts/pilot-02-best-eval/`.
- [x] A reviewed long-run anomaly artifact exists: `artifacts/run-9000/`.
- [x] A reviewed `Table II` artifact exists with machine-readable outputs: `artifacts/table-ii-200ep-01/`.
- [x] A reviewed `Fig. 3` artifact exists with machine-readable outputs: `artifacts/fig-3-pilot-01/`.
- [x] A non-smoke `Fig. 4` artifact exists and has a review note.
- [x] A non-smoke `Fig. 5` artifact exists and has a review note.
- [x] A non-smoke `Fig. 6` artifact exists and has a review note.

### 4. Disclosure And Interpretation

- [x] The late-training collapse in `artifacts/run-9000/` is preserved and disclosed rather than ignored.
- [x] The current near-tie / handover-dominant diagnosis is recorded in repo-local status notes.
- [x] Reward-calibration sensitivity is disclosed as an experiment only.
- [x] Public-facing summary language says "disclosed engineering baseline" and not "full scientific reproduction".
- [x] Current next-step guidance says freeze/disclosure comes before any new long run.

## Current Candidate Comparison-Baseline Bundle

Unless replaced by a later explicit freeze note, the current comparison
bundle should be treated as this artifact set:

1. baseline pilot: `artifacts/pilot-02-best-eval/`
2. long-run anomaly reference: `artifacts/run-9000/`
3. `Table II` review artifact: `artifacts/table-ii-200ep-01/`
4. `Fig. 3` review artifact: `artifacts/fig-3-pilot-01/`
5. `Fig. 4` review artifact: `artifacts/fig-4-pilot-01/`
6. `Fig. 5` review artifact: `artifacts/fig-5-pilot-01/`
7. `Fig. 6` review artifact: `artifacts/fig-6-pilot-01/`
8. reward-calibration sensitivity artifact: `artifacts/pilot-03-reward-calibration-200ep/`
9. repo-level status summaries:
   - `artifacts/reproduction-status-2026-04-13.md`
   - `artifacts/public-summary-2026-04-13.md`

## Not Required For Comparison-Baseline Freeze

The following are **not** prerequisites for freezing the repo as a
comparison baseline:

1. proving exact numeric parity with the paper
2. resolving the reward-dominance hypothesis
3. making reward-calibration outperform the raw baseline surface
4. starting a fresh `9000`-episode long run
5. silently redefining the baseline protocol

## Still Open For A Full Paper-Faithful Reproduction Claim

These gaps remain outside the comparison-baseline closeout target:

1. convincing paper-like method separation on the raw evaluation surface
2. a resolved explanation for the late-training collapse that is strong enough for a scientific reproduction claim
3. stronger evidence that the reported baseline protocol reproduces the paper's intended objective behavior rather than only a stable engineering approximation
4. figure-level trends that do more than repeat the same near-tie / handover-dominant structure across `Table II` and `Fig. 3` to `Fig. 6`

## Current Readiness Summary

As of `2026-04-13`, the repo already passes most infrastructure and
disclosure requirements for a disclosed comparison baseline.

It is now **checklist-complete** for the comparison-baseline closeout
target.

That means the current recommended order is:

1. freeze the repo as a disclosed comparison baseline
2. preserve the current result disclosures with the bundle
3. only open new work if it is a clearly labeled experimental track for paper-faithful reproduction
