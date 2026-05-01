# HOBS Active-TX EE Anti-Collapse Arm

Config: `configs/hobs-active-tx-ee-anti-collapse-candidate.resolved.yaml`
R1 mode: `hobs-active-tx-ee`
Anti-collapse enabled: `True`
Checkpoint used for greedy eval: `best-weighted-reward-on-eval`

## Greedy Diagnostics

- `denominator_varies_in_eval`: `True`
- `all_evaluated_steps_one_active_beam`: `False`
- `active_beam_count_distribution`: `{'2.0': 50}`
- `active_power_single_point_distribution`: `False`
- `power_control_activity_rate`: `0.5777777777777777`
- `throughput_vs_ee_pearson`: `-0.0461997871581038`
- `raw_throughput_mean_bps`: `1295.8475300574303`
- `p05_throughput_bps`: `1.6895455479621888`
- `served_ratio`: `1.0`
- `outage_ratio`: `0.0`
- `handover_count`: `918`
- `r2`: `-0.9179999999999999`
- `load_balance_metric`: `-100.6084526309966`
- `budget_violation_count`: `0`
- `per_beam_power_violation_count`: `0`
- `inactive_beam_nonzero_power_step_count`: `0`

## Claim Boundary

- Scalar reward is diagnostic only.
- This arm alone does not prove effectiveness.
- The constraint is opt-in and changes only this new gate surface.
