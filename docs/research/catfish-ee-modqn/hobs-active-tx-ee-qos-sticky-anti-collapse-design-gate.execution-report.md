# HOBS Active-TX EE QoS-Sticky Anti-Collapse Design Gate Execution Report

**Date:** `2026-05-01`  
**Status:** `PASS`  
**Scope:** bounded opt-in anti-collapse design gate only. This report does not
authorize EE-MODQN effectiveness, physical energy saving, HOBS optimizer,
Catfish / Multi-Catfish repair, RA-EE association, Phase `03C` continuation, or
frozen baseline mutation.

## Design Gate Decision

`qos-sticky-overflow-reassignment` was safe to test only as a new explicit,
config-gated, candidate-only anti-collapse mechanism.

The tested mechanism changes the candidate action-selection surface, so it is
not frozen MODQN semantics. It was implemented only under the new namespace:

```text
configs/hobs-active-tx-ee-qos-sticky-anti-collapse-*.resolved.yaml
artifacts/hobs-active-tx-ee-qos-sticky-anti-collapse-*/
```

Control and candidate both used HOBS active-TX EE as `r1` and the same
HOBS-inspired DPC sidecar. The only intended behavioral difference was that the
candidate enabled the opt-in QoS-sticky overflow reassignment constraint.

## What Was Implemented

Implemented a candidate-only trainer action-selection hook:

```text
anti_collapse_action_constraint:
  enabled: true
  mode: qos-sticky-overflow-reassignment
  overload_threshold_users_per_beam: 50
  qos_ratio_min: 0.95
  allow_nonsticky_moves: false
  nonsticky_move_budget: 0
  min_active_beams_target: 0
  assignment_order: user-index
```

At action-selection time, the hook first computes normal learned policy actions,
projects per-beam load, and intervenes only for users beyond the declared
overload threshold on an overloaded source beam. The primary candidate permits
only sticky overrides back to the user's current beam when the current beam is
valid, remains under threshold after the move, and satisfies:

```text
estimated_R(uid, target, target_projected_load)
  >= 0.95 * estimated_R(uid, greedy_source, greedy_source_projected_load)

estimated_R(uid, beam, projected_load)
  = bandwidth_hz / max(projected_load, 1) * log2(1 + state.channel_quality[beam])
```

No forced `min_active_beams_target` was used. Non-sticky moves were disabled
with a hard budget of `0`.

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
active_beam_count_distribution={"7.0": 50}
denominator_varies_in_eval=true
active_power_single_point_distribution=false
distinct_total_active_power_w_values=[7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8]
power_control_activity_rate=1.0
throughput_vs_ee_pearson=0.6166801450896923
same_policy_throughput_vs_ee_rescore_ranking_change=true
raw_throughput_mean_bps=7101.951499691009
p05_throughput_bps=14.953540515899657
served_ratio=1.0
outage_ratio=0.0
handover_count=212
r2=-0.21200000000000002
load_balance_metric=-40.23427828407311
scalar_reward_diagnostic_mean=435.2582971630092
budget_violation_count=0
per_beam_power_violation_count=0
inactive_beam_nonzero_power_step_count=0
overflow_steps=50
overflow_user_count=2500
sticky_override_count=2110
nonsticky_move_count=0
qos_guard_reject_count=0
handover_guard_reject_count=390
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
raw_throughput_mean_bps=+6167.472189884185
p05_throughput_bps=+9.02563886642456
p05_throughput_ratio_vs_control=2.522568929129207
served_ratio_delta=0.0
outage_ratio_delta=0.0
handover_count_delta=-211
r2_delta=+0.21100000000000002
load_balance_metric_delta=+53.21365269660883
episode_scalar_reward_diagnostic_mean_delta=+5.929660851898916
```

## Acceptance Result

`PASS`.

The candidate removed one-active-beam collapse without a forced active-beam
target, preserved denominator variability, preserved the power guardrails, and
met the predeclared QoS / handover / `r2` tolerances:

```text
all_evaluated_steps_one_active_beam=false
p05_throughput_ratio_vs_control=2.522568929129207 >= 0.95
served_ratio_delta=0.0 >= 0.0
outage_ratio_delta=0.0 <= 0.0
handover_delta=-211 <= +25
r2_delta=+0.21100000000000002 >= -0.05
budget_violation_count=0
per_beam_power_violation_count=0
inactive_beam_nonzero_power_step_count=0
nonsticky_move_count=0
```

Scalar reward improved only as a diagnostic and is not used as success evidence.

## Tests / Checks Run

```text
.venv/bin/python -m pytest tests/test_hobs_active_tx_ee_anti_collapse.py
```

Result: `9 passed`.

Focused HOBS suite:

```text
.venv/bin/python -m pytest \
  tests/test_hobs_active_tx_ee_anti_collapse.py \
  tests/test_hobs_dpc_denominator_check.py \
  tests/test_hobs_active_tx_ee_feasibility.py
```

Result: `32 passed`.

Pilot command:

```text
.venv/bin/python scripts/run_hobs_active_tx_ee_qos_sticky_anti_collapse.py
```

Result: completed `5`-episode matched control and candidate runs.

## Artifacts

```text
artifacts/hobs-active-tx-ee-qos-sticky-anti-collapse-control/
artifacts/hobs-active-tx-ee-qos-sticky-anti-collapse-candidate/
artifacts/hobs-active-tx-ee-qos-sticky-anti-collapse-candidate/paired-comparison-vs-control/
```

Key files:

```text
artifacts/hobs-active-tx-ee-qos-sticky-anti-collapse-control/summary.json
artifacts/hobs-active-tx-ee-qos-sticky-anti-collapse-candidate/summary.json
artifacts/hobs-active-tx-ee-qos-sticky-anti-collapse-candidate/paired-comparison-vs-control/summary.json
artifacts/hobs-active-tx-ee-qos-sticky-anti-collapse-candidate/paired-comparison-vs-control/review.md
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

This is a bounded anti-collapse gate result only. It does not promote
EE-MODQN effectiveness or any physical energy-saving claim.

## Final Verdict

```text
PASS / BLOCK / NEEDS MORE DESIGN: PASS
```
