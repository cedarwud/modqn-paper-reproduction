# HOBS Active-TX EE Anti-Collapse Arm

Config: `configs/hobs-active-tx-ee-qos-sticky-anti-collapse-candidate.resolved.yaml`
R1 mode: `hobs-active-tx-ee`
Anti-collapse enabled: `True`
Checkpoint used for greedy eval: `best-weighted-reward-on-eval`

## Greedy Diagnostics

- `denominator_varies_in_eval`: `True`
- `all_evaluated_steps_one_active_beam`: `False`
- `active_beam_count_distribution`: `{'7.0': 50}`
- `active_power_single_point_distribution`: `False`
- `power_control_activity_rate`: `1.0`
- `throughput_vs_ee_pearson`: `0.6166801450896923`
- `raw_throughput_mean_bps`: `7101.951499691009`
- `p05_throughput_bps`: `14.953540515899657`
- `served_ratio`: `1.0`
- `outage_ratio`: `0.0`
- `handover_count`: `212`
- `r2`: `-0.21200000000000002`
- `load_balance_metric`: `-40.23427828407311`
- `budget_violation_count`: `0`
- `per_beam_power_violation_count`: `0`
- `inactive_beam_nonzero_power_step_count`: `0`
- `overflow_steps`: `50`
- `overflow_user_count`: `2500`
- `sticky_override_count`: `2110`
- `nonsticky_move_count`: `0`
- `qos_guard_reject_count`: `0`
- `handover_guard_reject_count`: `390`

## Claim Boundary

- Scalar reward is diagnostic only.
- This arm alone does not prove effectiveness.
- The constraint is opt-in and changes only this new gate surface.
