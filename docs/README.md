# Documentation Index

This directory records the implementation plan for the standalone MODQN paper reproduction project.

For standalone use, pair this directory with the repo-local `paper-source/` snapshot and root
`AGENTS.md`; do not assume the larger `papers/` workspace exists.

For current project-state interpretation, treat
`artifacts/phase-01c-closeout-status-2026-04-15.md` as the latest
repo-local closeout authority,
`artifacts/public-summary-2026-04-15-phase-01c-closeout.md` as the
matching public summary surface, and
`artifacts/phase-01d-reopen-trigger-check-2026-04-16-producer-diagnostics.md`
as the latest reopen assessment note.
For the landed bounded reopen execution status, also read
`artifacts/phase-03b-producer-diagnostics-export-status-2026-04-16.md`,
which records the producer-side prerequisite now established for
downstream consumer promotion.
For current interpretation of the newer Phase 04 materials, also read
`artifacts/phase-04-current-state-2026-04-19.md`. That note keeps
Phase 03A / 03B as the landed external producer authority, records
Phase 04A / 04B / 04C / 04D / 04E as landed internal guardrail slices, and
treats later Phase 04 planning as internal hardening follow-on work.

## Read Order

1. `baseline-acceptance-checklist.md`
2. `decisions/ADR-001-separate-python-reproduction-project.md`
3. `phases/phase-01-python-baseline-reproduction-sdd.md`
4. `phases/phase-01b-paper-faithful-follow-on-sdd.md`
5. `phases/phase-01b-slice-c-targeted-high-load-follow-on-sdd.md`
6. `phases/phase-01c-comparator-protocol-experiment-sdd.md`
7. `phases/phase-01d-reproduction-reopen-gate-sdd.md`
8. `assumptions/modqn-reproduction-assumption-register.md`
9. `phases/phase-02-artifact-bridge-sdd.md`
10. `phases/phase-03-ntn-sim-core-visual-integration-sdd.md`
11. `phases/phase-03a-ntn-sim-core-bundle-replay-integration-sdd.md`
12. `phases/phase-03b-ntn-sim-core-producer-diagnostics-export-sdd.md`
13. `../artifacts/phase-04-current-state-2026-04-19.md`
14. `phases/phase-04-readme-summary.md`
15. `phases/phase-04-refactor-contract-spine-sdd.md`
16. `phases/phase-04a-refactor-semantic-golden-sdd.md`
17. `phases/phase-04b-refactor-training-artifact-model-sdd.md`
18. `phases/phase-04c-refactor-bundle-layer-split-sdd.md`
19. `phases/phase-04d-refactor-runtime-spine-split-sdd.md`
20. `phases/phase-04e-refactor-sweep-analysis-plotting-split-sdd.md`

## Purpose Split

1. `baseline-acceptance-checklist.md`
   Records the current closeout target for freezing a disclosed comparison baseline.
2. `decisions/`
   Records architectural decisions and rationale.
3. `phases/`
   Records phase-by-phase SDDs, including the explicit follow-on tracks, the standby reopen gate, any later bounded reopen execution surfaces, and draft internal-hardening plans that do not automatically supersede landed producer authority.
4. `assumptions/`
   Tracks reproduction assumptions that are not fully fixed by the paper.
