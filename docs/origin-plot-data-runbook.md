# Origin Plot Data Runbook

This runbook describes the repo-local process for regenerating curated
OriginLab-ready plotting CSVs from the disclosed comparison-baseline
sweep surfaces.

It does not authorize a new paper-faithful reproduction claim. It only
regenerates the already reviewed comparison-baseline plotting inputs.

## Guardrails

1. Use the repo-local virtual environment:
   `.venv/bin/python`.
2. Bootstrap the virtual environment with:
   `env PIP_CACHE_DIR=/tmp/pip-cache .venv/bin/python -m pip install -r requirements.txt`.
3. Use a resolved-run config:
   `configs/modqn-paper-baseline.resolved-template.yaml`.
4. Do not use `configs/modqn-paper-baseline.yaml` as a training input.
5. Do not start `500` or `9000` episode runs by default.
6. Track only curated CSV/README files under
   `artifacts/origin-plot-data/`; keep raw checkpoints, training logs,
   replay timelines, and full run bundles untracked.
7. In sandboxed environments, set `MPLCONFIGDIR=/tmp/modqn-mplconfig`
   when running sweeps to avoid Matplotlib writing under `$HOME`.

## Episode Choices

Use these episode counts when the goal is to reproduce the existing
Origin plot-data surface:

| Surface | Suite | Episodes per learned method | Source |
|---|---|---:|---|
| Table II | `table-ii` | `200` | `artifacts/table-ii-200ep-01/review.md` |
| Fig. 3 | `fig-3` | `50` | `artifacts/fig-3-pilot-01/review.md` |
| Fig. 4 | `fig-4` | `50` | `artifacts/fig-4-pilot-01/review.md` |
| Fig. 5 | `fig-5` | `50` | `artifacts/fig-5-pilot-01/review.md` |
| Fig. 6 | `fig-6` | `50` | `artifacts/fig-6-pilot-01/review.md` |

The later bounded follow-on work used `20` and `200` episodes for
beam-aware eligibility checks, but those are not the frozen baseline
figure-sweep defaults.

## Generate Sweep Outputs

Run from the repository root.

```bash
.venv/bin/python scripts/run_sweeps.py \
  --config configs/modqn-paper-baseline.resolved-template.yaml \
  --suite table-ii \
  --episodes 200 \
  --output-dir artifacts/table-ii-200ep-01
```

```bash
.venv/bin/python scripts/run_sweeps.py \
  --config configs/modqn-paper-baseline.resolved-template.yaml \
  --suite fig-3 \
  --episodes 50 \
  --output-dir artifacts/fig-3-pilot-01
```

```bash
.venv/bin/python scripts/run_sweeps.py \
  --config configs/modqn-paper-baseline.resolved-template.yaml \
  --suite fig-4 \
  --episodes 50 \
  --output-dir artifacts/fig-4-pilot-01
```

```bash
.venv/bin/python scripts/run_sweeps.py \
  --config configs/modqn-paper-baseline.resolved-template.yaml \
  --suite fig-5 \
  --episodes 50 \
  --output-dir artifacts/fig-5-pilot-01
```

```bash
.venv/bin/python scripts/run_sweeps.py \
  --config configs/modqn-paper-baseline.resolved-template.yaml \
  --suite fig-6 \
  --episodes 50 \
  --output-dir artifacts/fig-6-pilot-01
```

These commands emit CSV/JSON/PNG outputs under each artifact directory.
The Origin-ready CSVs are:

1. `artifacts/table-ii-200ep-01/figures/table-ii.csv`
2. `artifacts/fig-3-pilot-01/evaluation/fig-3-detail.csv`
3. `artifacts/fig-3-pilot-01/evaluation/fig-3-weighted-reward.csv`
4. `artifacts/fig-4-pilot-01/evaluation/fig-4-detail.csv`
5. `artifacts/fig-4-pilot-01/evaluation/fig-4-weighted-reward.csv`
6. `artifacts/fig-5-pilot-01/evaluation/fig-5-detail.csv`
7. `artifacts/fig-5-pilot-01/evaluation/fig-5-weighted-reward.csv`
8. `artifacts/fig-6-pilot-01/evaluation/fig-6-detail.csv`
9. `artifacts/fig-6-pilot-01/evaluation/fig-6-weighted-reward.csv`

## Copy Curated CSVs

After the sweeps complete, copy only the curated plotting inputs:

```bash
mkdir -p artifacts/origin-plot-data
cp artifacts/table-ii-200ep-01/figures/table-ii.csv artifacts/origin-plot-data/
cp artifacts/fig-3-pilot-01/evaluation/fig-3-detail.csv artifacts/origin-plot-data/
cp artifacts/fig-3-pilot-01/evaluation/fig-3-weighted-reward.csv artifacts/origin-plot-data/
cp artifacts/fig-4-pilot-01/evaluation/fig-4-detail.csv artifacts/origin-plot-data/
cp artifacts/fig-4-pilot-01/evaluation/fig-4-weighted-reward.csv artifacts/origin-plot-data/
cp artifacts/fig-5-pilot-01/evaluation/fig-5-detail.csv artifacts/origin-plot-data/
cp artifacts/fig-5-pilot-01/evaluation/fig-5-weighted-reward.csv artifacts/origin-plot-data/
cp artifacts/fig-6-pilot-01/evaluation/fig-6-detail.csv artifacts/origin-plot-data/
cp artifacts/fig-6-pilot-01/evaluation/fig-6-weighted-reward.csv artifacts/origin-plot-data/
```

Then verify that the curated CSVs are trackable:

```bash
git status --short artifacts/origin-plot-data
git add -n artifacts/origin-plot-data
```

The raw sweep outputs under `artifacts/table-ii-200ep-01/` and
`artifacts/fig-*-pilot-01/` remain generated artifacts and are ignored
by git.
