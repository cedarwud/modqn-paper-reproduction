# HOBS Active-TX EE Anti-Collapse Arm

Config: `configs/hobs-active-tx-ee-qos-sticky-anti-collapse-control.resolved.yaml`
R1 mode: `hobs-active-tx-ee`
Anti-collapse enabled: `False`
Checkpoint used for greedy eval: `best-weighted-reward-on-eval`

## Greedy Diagnostics

- `denominator_varies_in_eval`: `True`
- `all_evaluated_steps_one_active_beam`: `True`
- `active_beam_count_distribution`: `{'1.0': 50}`
- `active_power_single_point_distribution`: `False`
- `power_control_activity_rate`: `1.0`
- `throughput_vs_ee_pearson`: `0.19303453314619476`
- `raw_throughput_mean_bps`: `934.4793098068237`
- `p05_throughput_bps`: `5.927901649475098`
- `served_ratio`: `1.0`
- `outage_ratio`: `0.0`
- `handover_count`: `423`
- `r2`: `-0.42300000000000004`
- `load_balance_metric`: `-93.44793098068195`
- `budget_violation_count`: `0`
- `per_beam_power_violation_count`: `0`
- `inactive_beam_nonzero_power_step_count`: `0`
- `overflow_steps`: `0`
- `overflow_user_count`: `0`
- `sticky_override_count`: `0`
- `nonsticky_move_count`: `0`
- `qos_guard_reject_count`: `0`
- `handover_guard_reject_count`: `0`

## Claim Boundary

- Scalar reward is diagnostic only.
- This arm alone does not prove effectiveness.
- The constraint is opt-in and changes only this new gate surface.
