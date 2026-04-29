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
7. `audit_ee_denominator.py`
   Phase 02 report-only EE denominator audit entrypoint. Loads a resolved-run
   config without training, samples runtime steps under explicit audit policies,
   and emits `ee_denominator_summary.json`, `ee_denominator_audit.csv`, and
   `review.md`. Static baseline configs are labeled as fixed-power diagnostics;
   opt-in Phase 02B configs with `hobs_power_surface.mode:
   active-load-concave` report explicit per-beam `beam_transmit_power_w`,
   active-beam masks, total active beam power, and the HOBS EE go/no-go gate.
8. `compare_phase03_ee_modqn.py`
   Phase 03 / 03B paired-validation entrypoint. Replays matched `MODQN-control`
   and `EE-MODQN` checkpoints, reports `EE_system`, raw throughput, served
   ratio, active beam count / active power distributions, handover / r2,
   load-balance / r3, same-policy rescoring, and reward-hacking flags. It does
   not run Catfish or promote final EE-MODQN effectiveness claims.
9. `diagnose_phase03a_policy_power.py`
   Policy-denominator diagnostic entrypoint. Replays paired checkpoints and
   counterfactual policies to distinguish a variable runtime denominator from
   learned policies that fail to exercise that denominator.
10. `audit_phase03c_b_power_mdp.py`
   Phase 03C-B static/counterfactual power-MDP audit entrypoint. Replays fixed
   handover trajectories under fixed, Phase 02B proxy, and Phase 03C-B codebook
   power semantics to test whether explicit power decisions can change the
   denominator before any bounded training pilot.
11. `compare_phase03c_c_power_mdp.py`
   Phase 03C-C bounded paired-pilot comparison entrypoint. Replays matched
   fixed-power-control and runtime-selector candidate checkpoints, logs selected
   power profile, active-beam transmit power, budget violations, denominator
   variability, system EE, QoS guardrails, and throughput-vs-EE ranking checks.
   It does not run Catfish, multi-Catfish, long training, or frozen baseline
   mutation.
12. `audit_ra_ee_02_oracle_power_allocation.py`
   RA-EE-02 offline oracle / heuristic upper-bound audit entrypoint. Replays
   fixed association trajectories under finite-codebook power allocations,
   applies per-beam, total-budget, inactive-beam, served-ratio, outage, and
   p05-throughput guardrails, and reports whether a budget-respecting
   power-allocation candidate can improve system EE without QoS collapse. It
   does not run RL training, Catfish, multi-Catfish, or mutate frozen baseline
   artifacts.
13. `run_ra_ee_04_bounded_power_allocator.py`
   RA-EE-04 bounded fixed-association centralized power-allocation pilot.
   Runs a 20-episode calibration surface, evaluates fixed 1 W control versus
   the safe-greedy per-active-beam power allocator on non-collapsed fixed
   trajectories, and optionally exports constrained-oracle upper-bound
   diagnostics. It does not learn association, run Catfish or multi-Catfish,
   continue old EE-MODQN, run long training, or mutate frozen baseline
   artifacts.
