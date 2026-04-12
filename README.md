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

What exists now:

1. project structure
2. repo-local paper source snapshot under `paper-source/`
3. phase-based SDDs and assumption register
4. paper-envelope config plus executable resolved-run config
5. environment modules for orbit, beam, channel, and step semantics
6. MODQN trainer, replay buffer, target sync, checkpoints, and CLI entrypoints
7. smoke and hardening tests for training flow

What does not exist yet:

1. comparator baselines
2. sweep/figure generation logic
3. Phase 2 export bundle implementation
4. secondary best-eval checkpoint flow for `ASSUME-MODQN-REP-015`

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
```

Additional entrypoints:

```bash
./.venv/bin/python scripts/run_sweeps.py --config configs/modqn-paper-baseline.resolved-template.yaml
./.venv/bin/python scripts/export_ntn_sim_core_bundle.py --input artifacts/example-run
```

`train_modqn.py` is a real training entrypoint. Sweep and export entrypoints remain placeholders.

## Config Surfaces

This project intentionally separates two config roles:

1. `configs/modqn-paper-baseline.yaml`
   Paper-envelope config. It records `paper-backed` parameters, `recovered-from-paper` Table II weights, paper ranges, and unresolved assumption references. It is not the final executable run config.
2. `configs/modqn-paper-baseline.resolved-template.yaml`
   Resolved-run template. This is the surface where concrete `reproduction-assumption` values, seeds, aggregation, and checkpoint rules are frozen before a real training run is allowed.

No implementation should silently promote the paper-envelope config into an executable run by inventing default values in code.

Training entrypoints now hard-reject the paper-envelope config and require a resolved-run config.

## Documentation Map

1. `docs/decisions/ADR-001-separate-python-reproduction-project.md`
2. `docs/phases/phase-01-python-baseline-reproduction-sdd.md`
3. `docs/phases/phase-02-artifact-bridge-sdd.md`
4. `docs/phases/phase-03-ntn-sim-core-visual-integration-sdd.md`
5. `docs/assumptions/modqn-reproduction-assumption-register.md`

## Intended Deliverables

Phase 1 should eventually produce:

1. trained `MODQN`
2. `RSS_max`, `DQN_throughput`, `DQN_scalar`, and `MODQN` evaluation outputs
3. paper-style outputs for `Table II` and `Fig. 3` to `Fig. 6`
4. figure-ready CSV/JSON data
5. explicit assumption disclosures per run

Phase 2 should freeze an exportable artifact bundle for `ntn-sim-core` ingestion.

Phase 3 should map those artifacts into `ntn-sim-core` replay/overlay/3D presentation without moving the trainer itself into the platform repo.
