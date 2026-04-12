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

Authority/config ready for Phase 1 implementation. Runtime code is still scaffold only.

What exists now:

1. project structure
2. phase-based SDDs
3. assumption register with accepted Phase 01 blocking values
4. paper-envelope config plus filled resolved-run template
5. minimal Python package/CLI skeleton

What does not exist yet:

1. full environment implementation
2. MODQN trainer
3. comparator baselines
4. figure generation logic
5. `ntn-sim-core` artifact export adapter

## Directory Layout

```text
modqn-paper-reproduction/
├── README.md
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

1. `../paper-catalog/ref/2024_09_Handover_for_Multi-Beam_LEO_Satellite_Networks_A_Multi-Objective_Reinforcement_Learning_Method.pdf`
2. `../paper-catalog/txt_layout_all/2024_09_Handover_for_Multi-Beam_LEO_Satellite_Networks_A_Multi-Objective_Reinforcement_Learning_Method.layout.txt`
3. `../paper-catalog/catalog/PAP-2024-MORL-MULTIBEAM.json`
4. `docs/phases/phase-01-python-baseline-reproduction-sdd.md`
5. `docs/assumptions/modqn-reproduction-assumption-register.md`
6. `docs/phases/phase-02-artifact-bridge-sdd.md`
7. `docs/phases/phase-03-ntn-sim-core-visual-integration-sdd.md`

`../system-model-refs/` is reference material for cross-checking formulas and identifying where the paper is simplified; it must not silently override the paper when this project is operating in reproduction mode.

## Quick Start

Create a virtual environment and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[train]"
```

Current scaffold commands:

```bash
python scripts/train_modqn.py --config configs/modqn-paper-baseline.resolved-template.yaml
python scripts/run_sweeps.py --config configs/modqn-paper-baseline.resolved-template.yaml
python scripts/export_ntn_sim_core_bundle.py --input artifacts/example-run
```

The current scripts are placeholders that only confirm the intended entry points and configuration surface.

## Config Surfaces

This project intentionally separates two config roles:

1. `configs/modqn-paper-baseline.yaml`
   Paper-envelope config. It records `paper-backed` parameters, `recovered-from-paper` Table II weights, paper ranges, and unresolved assumption references. It is not the final executable run config.
2. `configs/modqn-paper-baseline.resolved-template.yaml`
   Resolved-run template. This is the surface where concrete `reproduction-assumption` values, seeds, aggregation, and checkpoint rules are frozen before a real training run is allowed.

No implementation should silently promote the paper-envelope config into an executable run by inventing default values in code.

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
