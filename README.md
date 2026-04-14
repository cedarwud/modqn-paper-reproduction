# MODQN Paper Reproduction

This project is the separate Python-first reproduction surface for `PAP-2024-MORL-MULTIBEAM`:

- paper ID: `PAP-2024-MORL-MULTIBEAM`
- title: `Handover for Multi-Beam LEO Satellite Networks: A Multi-Objective Reinforcement Learning Method`

The immediate goal is to build a paper-consistent baseline reproduction in a standalone Python project before any integration with `ntn-sim-core` or its 3D UI.

## Why This Is Separate

`ntn-sim-core` is the long-term simulator platform. It already contains a reviewed MODQN bridge, result schema, and UI-facing projection layer, but that surface also carries platform/runtime proxy assumptions that are useful for integration and visualization rather than for a first clean paper-baseline reproduction.

This project keeps the first implementation focused on:

1. paper-faithful state/action/reward
2. reproducible training and sweep execution
3. paper-style plots and tables
4. explicit separation between `paper-backed`, `recovered-from-paper`, `reproduction-assumption`, and `platform-visualization-only`

## Current Status

Training-ready baseline reproduction surface with a real environment stack, MODQN trainer,
resolved-run config enforcement, checkpointing, and smoke-tested training entrypoints.

Current closeout target:

1. the comparison-baseline checklist is complete
2. the repo is ready to freeze as a disclosed comparison baseline
3. do not over-claim a full paper-faithful reproduction

What exists now:

1. project structure
2. repo-local paper source snapshot under `paper-source/`
3. phase-based SDDs and assumption register
4. paper-envelope config plus executable resolved-run config
5. environment modules for orbit, beam, channel, and step semantics
6. MODQN trainer, replay buffer, target sync, checkpoints, and CLI entrypoints
7. final and best-eval checkpoint capture with run metadata
8. executable `Table II` sweep surface for `MODQN`, `DQN_throughput`, `DQN_scalar`, and `RSS_max`
9. training-run export to CSV/PNG bundle surfaces
10. analysis outputs for `Table II` winners, spreads, deltas-vs-MODQN, and long-run linkage
11. explicit experimental `reward-geometry` analysis surface for re-scoring existing artifacts
12. explicit experimental reward-calibrated training config for sensitivity runs
13. reviewed non-smoke pilot artifacts for `Fig. 3` to `Fig. 6`
14. first executable `Fig. 3` to `Fig. 6` sweep surfaces with figure-style CSV/JSON/PNG outputs
15. smoke and hardening tests for training flow, sweeps, export, and analysis

What does not exist yet:

1. a complete paper-faithful reproduction claim with convincing method separation
2. paper-visual parity polish for the original figure layouts

The Phase 02 export bundle freeze is now landed as the Phase 03A
`phase-03a-replay-bundle-v1` surface (see
`docs/phases/phase-03a-ntn-sim-core-bundle-replay-integration-sdd.md` and
`artifacts/reproduction-status-2026-04-13-phase-03a-slice-a.md`). The
checked-in `tests/fixtures/sample-bundle-v1/` is the canonical
sample export bundle for downstream consumers.

## Directory Layout

```text
modqn-paper-reproduction/
├── AGENTS.md
├── README.md
├── paper-source/
├── pyproject.toml
├── configs/
├── docs/
│   ├── decisions/
│   ├── assumptions/
│   └── phases/
├── artifacts/
├── scripts/
└── src/modqn_paper_reproduction/
```

## Authority Order

When details conflict, use this order:

1. `paper-source/ref/2024_09_Handover_for_Multi-Beam_LEO_Satellite_Networks_A_Multi-Objective_Reinforcement_Learning_Method.pdf`
2. `paper-source/txt_layout/2024_09_Handover_for_Multi-Beam_LEO_Satellite_Networks_A_Multi-Objective_Reinforcement_Learning_Method.layout.txt`
3. `paper-source/catalog/PAP-2024-MORL-MULTIBEAM.json`
4. `docs/phases/phase-01-python-baseline-reproduction-sdd.md`
5. `docs/assumptions/modqn-reproduction-assumption-register.md`
6. `docs/phases/phase-02-artifact-bridge-sdd.md`
7. `docs/phases/phase-03-ntn-sim-core-visual-integration-sdd.md`
8. `docs/phases/phase-03a-ntn-sim-core-bundle-replay-integration-sdd.md`

If an external historical workspace exists, use it only as a cross-check. The repo-local
`paper-source/` snapshot is the portable authority surface for standalone work.

## Quick Start

Create a virtual environment and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[train]"
```

Primary training commands:

```bash
./.venv/bin/python scripts/train_modqn.py --config configs/modqn-paper-baseline.resolved-template.yaml --episodes 5 --output-dir artifacts/final-smoke
./.venv/bin/python scripts/train_modqn.py --config configs/modqn-paper-baseline.resolved-template.yaml --episodes 200 --output-dir artifacts/pilot-01
./.venv/bin/python scripts/train_modqn.py --config configs/modqn-paper-baseline.reward-calibration.resolved.yaml --episodes 5 --output-dir artifacts/reward-calibration-smoke
```

Additional entrypoints:

```bash
./.venv/bin/python scripts/run_sweeps.py --config configs/modqn-paper-baseline.resolved-template.yaml --suite table-ii --episodes 1 --max-weight-rows 2 --output-dir artifacts/table-ii-smoke --reference-run artifacts/run-9000
./.venv/bin/python scripts/run_sweeps.py --config configs/modqn-paper-baseline.resolved-template.yaml --suite fig-3 --episodes 1 --max-figure-points 2 --output-dir artifacts/fig-3-smoke --reference-run artifacts/run-9000
./.venv/bin/python scripts/run_sweeps.py --config configs/modqn-paper-baseline.resolved-template.yaml --suite reward-geometry --input-table-ii artifacts/table-ii-200ep-01 --reference-run artifacts/run-9000 --output-dir artifacts/reward-geometry-01
./.venv/bin/python scripts/export_ntn_sim_core_bundle.py --input artifacts/pilot-02-best-eval --output-dir artifacts/pilot-02-best-eval/export-bundle
./.venv/bin/python scripts/generate_sample_bundle.py --output tests/fixtures/sample-bundle-v1 --episodes 1 --max-users 1
```

`train_modqn.py` is a real training entrypoint and writes both the final checkpoint and
the eval-selected secondary checkpoint when an output directory is requested. `run_sweeps.py`
now implements a first executable `Table II` slice plus first executable `Fig. 3`
to `Fig. 6` sweep surfaces, and can emit objective-decomposition
analysis against a reference run. It also exposes an explicit experimental
`reward-geometry` suite for re-scoring existing `Table II` artifacts under alternative
normalization scales without changing the baseline training rule. The separate
`modqn-paper-baseline.reward-calibration.resolved.yaml` surface is an opt-in
training experiment: it calibrates trainer-side rewards by fixed scales while
keeping evaluation, logged raw objectives, and checkpoint selection on the
raw paper metric surface. `export_ntn_sim_core_bundle.py` exports a completed
training run into the frozen Phase 03A `phase-03a-replay-bundle-v1` bundle
surface (manifest, config-resolved, assumptions, provenance-map,
training/evaluation summaries, figures, and a replay-complete
`timeline/step-trace.jsonl`). `generate_sample_bundle.py` runs that pipeline
end-to-end on a one-episode smoke training and trims the timeline into a
small, byte-stable `tests/fixtures/sample-bundle-v1/` fixture that downstream
consumers (e.g. `ntn-sim-core`) can ingest without re-running training.

## Config Surfaces

This project intentionally separates two config roles:

1. `configs/modqn-paper-baseline.yaml`
   Paper-envelope config. It records `paper-backed` parameters, `recovered-from-paper` Table II weights, paper ranges, and unresolved assumption references. It is not the final executable run config.
2. `configs/modqn-paper-baseline.resolved-template.yaml`
   Resolved-run template. This is the surface where concrete `reproduction-assumption` values, seeds, aggregation, and checkpoint rules are frozen before a real training run is allowed.
3. `configs/modqn-paper-baseline.reward-calibration.resolved.yaml`
   Explicit experimental resolved-run config. It inherits the baseline resolved template but opt-in enables trainer-side reward calibration for sensitivity runs. It is not the default paper-baseline training surface.

No implementation should silently promote the paper-envelope config into an executable run by inventing default values in code.

Training entrypoints now hard-reject the paper-envelope config and require a resolved-run config.

## Documentation Map

1. `docs/baseline-acceptance-checklist.md`
2. `docs/decisions/ADR-001-separate-python-reproduction-project.md`
3. `docs/phases/phase-01-python-baseline-reproduction-sdd.md`
4. `docs/phases/phase-02-artifact-bridge-sdd.md`
5. `docs/phases/phase-03-ntn-sim-core-visual-integration-sdd.md`
6. `docs/phases/phase-03a-ntn-sim-core-bundle-replay-integration-sdd.md`
7. `docs/assumptions/modqn-reproduction-assumption-register.md`

## Intended Deliverables

Phase 1 should eventually produce:

1. trained `MODQN`
2. `RSS_max`, `DQN_throughput`, `DQN_scalar`, and `MODQN` evaluation outputs
3. paper-style outputs for `Table II` and `Fig. 3` to `Fig. 6`
4. figure-ready CSV/JSON data
5. explicit assumption disclosures per run

Phase 2 should freeze an exportable artifact bundle for `ntn-sim-core` ingestion.

Phase 3 should map those artifacts into `ntn-sim-core` replay/overlay/3D presentation without moving the trainer itself into the platform repo.
