# Route D DPC Denominator Check Arm

Config: `configs/hobs-dpc-denominator-check-control.resolved.yaml`
R1 mode: `throughput`
Checkpoint used for greedy eval: `best-weighted-reward-on-eval`

## Greedy Diagnostics

- `denominator_varies_in_eval`: `True`
- `all_evaluated_steps_one_active_beam`: `True`
- `active_power_single_point_distribution`: `False`
- `power_control_activity_rate`: `1.0`
- `dpc_sign_flip_count`: `20`
- `throughput_vs_ee_pearson`: `0.19303453314619476`
- `raw_throughput_mean_bps`: `934.4793098068237`
- `p05_throughput_bps`: `5.927901649475098`
- `served_ratio`: `1.0`
- `handover_count`: `423`

## Claim Boundary

- Scalar reward is diagnostic only.
- This arm alone does not prove effectiveness.
- DPC is opt-in and not MODQN-paper-backed.
