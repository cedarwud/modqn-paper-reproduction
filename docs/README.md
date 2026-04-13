# Documentation Index

This directory records the implementation plan for the standalone MODQN paper reproduction project.

For standalone use, pair this directory with the repo-local `paper-source/` snapshot and root
`AGENTS.md`; do not assume the larger `papers/` workspace exists.

For current project-state interpretation, treat
`artifacts/reproduction-status-2026-04-13.md` as the latest repo-local
status authority and `artifacts/public-summary-2026-04-13.md` as the
matching public summary surface.

## Read Order

1. `baseline-acceptance-checklist.md`
2. `decisions/ADR-001-separate-python-reproduction-project.md`
3. `phases/phase-01-python-baseline-reproduction-sdd.md`
4. `assumptions/modqn-reproduction-assumption-register.md`
5. `phases/phase-02-artifact-bridge-sdd.md`
6. `phases/phase-03-ntn-sim-core-visual-integration-sdd.md`
7. `phases/phase-03a-ntn-sim-core-bundle-replay-integration-sdd.md`

## Purpose Split

1. `baseline-acceptance-checklist.md`
   Records the current closeout target for freezing a disclosed comparison baseline.
2. `decisions/`
   Records architectural decisions and rationale.
3. `phases/`
   Records phase-by-phase SDDs.
4. `assumptions/`
   Tracks reproduction assumptions that are not fully fixed by the paper.
