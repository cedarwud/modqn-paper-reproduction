# HOBS Active-TX EE Anti-Collapse Design Gate Execution Report

**Date:** `2026-05-01`  
**Status:** `BLOCK`  
**Scope:** bounded opt-in anti-collapse design gate only. This report does not
authorize EE-MODQN effectiveness, physical energy saving, HOBS optimizer,
Catfish / Multi-Catfish repair, RA-EE association, Phase `03C` continuation, or
frozen baseline mutation.

## Design Gate Decision

`capacity-aware-greedy-assignment` was safe to test only as a new explicit,
config-gated, candidate-only anti-collapse mechanism.

The tested mechanism changes the candidate action-selection surface, so it is
not frozen MODQN semantics. It was therefore implemented only under the new
namespace:

```text
configs/hobs-active-tx-ee-anti-collapse-*.yaml
artifacts/hobs-active-tx-ee-anti-collapse-*/
```

Control and candidate both used HOBS active-TX EE as `r1` and the same
HOBS-inspired DPC sidecar. The only intended behavioral difference was that the
candidate enabled the opt-in assignment constraint.

## What Was Implemented

Implemented a candidate-only trainer action-selection hook:

```text
anti_collapse_action_constraint:
  enabled: true
  mode: capacity-aware-greedy-assignment
  max_users_per_beam: 50
  min_active_beams_target: 2
  assignment_order: user-index
```

At action-selection time, users are processed in stable user-index order and
assigned to the highest-ranked valid action whose current candidate-assignment
load is below `max_users_per_beam`. For `100` users this makes a one-active-beam
assignment infeasible when enough valid beams exist.

## Matched Boundary Proof

`matched_boundary_pass=true`.

Checks:

```text
both_r1_are_hobs_active_tx_ee=true
same_training_experiment_kind=true
same_phase=true
same_episode_budget=true
tiny_episode_budget=true
same_seed_block=true
same_objective_weights=true
same_training_hyperparameters=true
same_checkpoint_rule=true
same_dpc_sidecar=true
dpc_sidecar_enabled=true
same_environment_boundary=true
same_constraint_parameters=true
control_constraint_disabled=true
candidate_constraint_enabled=true
```

Predeclared tolerances before the pilot:

```text
p05_throughput_ratio_vs_control >= 0.95
served_ratio_delta >= 0.0
outage_ratio_delta <= 0.0
handover_delta <= +25
r2_mean_delta >= -0.05
budget_violation_count = 0
per_beam_power_violation_count = 0
inactive_beam_nonzero_power_step_count = 0
```

## Metrics

Candidate:

```text
all_evaluated_steps_one_active_beam=false
active_beam_count_distribution={"2.0": 50}
denominator_varies_in_eval=true
active_power_single_point_distribution=false
distinct_total_active_power_w_values=[1.1, 1.2, 1.3, 1.5, 1.7, 1.9, 2.2]
power_control_activity_rate=0.5777777777777777
throughput_vs_ee_pearson=-0.0461997871581038
same_policy_throughput_vs_ee_rescore_ranking_change=true
raw_throughput_mean_bps=1295.8475300574303
p05_throughput_bps=1.6895455479621888
served_ratio=1.0
outage_ratio=0.0
handover_count=918
r2=-0.9179999999999999
load_balance_metric=-100.6084526309966
scalar_reward_diagnostic_mean=434.6984225051388
budget_violation_count=0
per_beam_power_violation_count=0
inactive_beam_nonzero_power_step_count=0
```

Matched control:

```text
all_evaluated_steps_one_active_beam=true
active_beam_count_distribution={"1.0": 50}
denominator_varies_in_eval=true
active_power_single_point_distribution=false
raw_throughput_mean_bps=934.4793098068237
p05_throughput_bps=5.927901649475098
served_ratio=1.0
outage_ratio=0.0
handover_count=423
r2=-0.42300000000000004
load_balance_metric=-93.44793098068195
scalar_reward_diagnostic_mean=434.6653310778192
```

Candidate minus control:

```text
raw_throughput_mean_bps=+361.36822025060656
p05_throughput_bps=-4.23835610151291
p05_throughput_ratio_vs_control=0.28501578600107075
served_ratio_delta=0.0
outage_ratio_delta=0.0
handover_count_delta=+495
r2_delta=-0.4949999999999999
load_balance_metric_delta=-7.160521650314649
episode_scalar_reward_diagnostic_mean_delta=+0.3309142732032342
```

## Acceptance Result

`BLOCK`.

The candidate did remove one-active-beam collapse and preserved denominator
variability, but it failed the predeclared QoS and handover / `r2` guardrails:

```text
p05_throughput_ratio_vs_control=0.28501578600107075 < 0.95
handover_delta=+495 > +25
r2_delta=-0.4949999999999999 < -0.05
```

The scalar reward diagnostic increased slightly, but scalar reward is diagnostic
only and cannot override the p05 / handover / `r2` failures.

## Tests / Checks Run

```text
.venv/bin/python -m pytest \
  tests/test_hobs_active_tx_ee_anti_collapse.py \
  tests/test_hobs_dpc_denominator_check.py \
  tests/test_hobs_active_tx_ee_feasibility.py
```

Result: `29 passed`.

Pilot command:

```text
.venv/bin/python scripts/run_hobs_active_tx_ee_anti_collapse.py
```

Result: completed `5`-episode matched control and candidate runs.

## Artifacts

```text
artifacts/hobs-active-tx-ee-anti-collapse-control/
artifacts/hobs-active-tx-ee-anti-collapse-candidate/
artifacts/hobs-active-tx-ee-anti-collapse-candidate/paired-comparison-vs-control/
```

Key files:

```text
artifacts/hobs-active-tx-ee-anti-collapse-control/summary.json
artifacts/hobs-active-tx-ee-anti-collapse-candidate/summary.json
artifacts/hobs-active-tx-ee-anti-collapse-candidate/paired-comparison-vs-control/summary.json
artifacts/hobs-active-tx-ee-anti-collapse-candidate/paired-comparison-vs-control/review.md
```

## Forbidden Claims Still Active

Do not claim:

```text
EE-MODQN effectiveness
physical energy saving
HOBS optimizer reproduction
DPC as MODQN-paper-backed optimizer
Catfish / Multi-Catfish / Catfish-EE effectiveness
full RA-EE-MODQN
learned association effectiveness
RB / bandwidth allocation effectiveness
denominator variability alone proves energy-aware learning
scalar reward proves success
```

## Deviations / Blockers

No forbidden route was opened: no longer Route `D` training, no Catfish /
Multi-Catfish, no Phase `03C` selector / reward tuning, no RA-EE association,
and no frozen baseline mutation.

The blocker is that the minimal capacity-aware assignment constraint removed
collapse by forcing a two-beam target, but the result depended on unacceptable
tail-throughput and handover / `r2` regressions. Therefore this exact candidate
cannot pass.

## Final Verdict

```text
PASS / BLOCK / NEEDS MORE DESIGN: BLOCK
```
