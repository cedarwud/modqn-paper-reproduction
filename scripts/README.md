# Script Entry Points

Current scripts expose this command surface:

Install the repo-local training/test environment before running these
entrypoints:

```bash
python3 -m venv .venv
env PIP_CACHE_DIR=/tmp/pip-cache .venv/bin/python -m pip install -r requirements.txt
```

1. `train_modqn.py`
   Real MODQN training entrypoint. Requires a resolved-run config and writes logs, the
   final checkpoint, and the eval-selected best checkpoint. It can also run the
   explicit experimental reward-calibration resolved config without changing the
   default baseline config.
2. `run_sweeps.py`
   Real sweep entrypoint for the first `Table II` slice plus first executable
   `Fig. 3` to `Fig. 6` surfaces. Emits machine-readable CSV/JSON plus PNG plots for
   `MODQN`, `DQN_throughput`, `DQN_scalar`, and `RSS_max`. It can
   also emit winners/spreads/deltas analysis and link the sweep to a reference training run.
   It additionally supports an experimental `reward-geometry` suite that re-scores an
   existing `Table II` artifact under alternative normalization scales.
   For curated OriginLab-ready CSV regeneration, use the documented
   commands and episode counts in `../docs/origin-plot-data-runbook.md`.
3. `export_ntn_sim_core_bundle.py`
   Real export entrypoint for completed training artifacts. Emits training CSVs,
   summary JSON, PNG plots, and the landed Phase 03A replay-bundle-v1
   contract surface, with Phase 03B optional producer diagnostics when
   available.
4. `audit_beam_semantics.py`
   Evaluation-only audit entrypoint for the bounded `Phase 01E` reopen slice.
   Replays a preserved checkpoint, quantifies beam-mask/channel collapse, and
   emits `beam_semantics_summary.json`, `beam_tie_metrics.csv`,
   `decision_margin_metrics.csv`, and `review.md`.
5. `evaluate_beam_counterfactual.py`
   Evaluation-only `Phase 01E2` entrypoint. Replays a preserved checkpoint under
   the opt-in `nearest-beam-per-visible-satellite` eligibility proxy and emits
   `counterfactual_eval_summary.json`, `counterfactual_vs_baseline.csv`, and
   `review.md`.
6. `evaluate_atmospheric_sign_counterfactual.py`
   Evaluation-only `Phase 01G` entrypoint. Replays a preserved checkpoint under
   the paper-published atmospheric sign and the corrected lossy sign, then emits
   `atmospheric_sign_counterfactual_summary.json`,
   `atmospheric_sign_vs_baseline.csv`,
   `reward_geometry_diagnostics_comparison.csv`, and `review.md`.
