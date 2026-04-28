# Phase 02 EE Denominator Audit Review

This audit is report-only. It does not train EE-MODQN and does not change the MODQN reward surface.

## Runtime Mapping

- `R_i(t)`: `RewardComponents.r1_throughput`, computed from `user_throughputs` in `StepEnvironment._build_states_and_masks` and aggregated into `beam_throughputs`.
- `P_b(t)`: only `ChannelConfig.tx_power_w` is present. It is a static linear-watt scalar used by `compute_channel` for received power and SNR.
- `active_beams`: derivable from `StepResult.user_states[0].beam_loads > 0` after a step.

## Audit Result

- denominator classification: `fixed-power-active-beam-count-proxy`
- configured `tx_power_w`: `2.0` W
- distinct active-beam counts sampled: `[1, 7]`
- distinct total fixed-power denominators sampled: `[2.0, 14.0]`
- HOBS EE defensible in current runtime: `False`

## Decision

- Phase 02 status: `blocked`
- Phase 03 gate: `no-go`
- EE degenerates to throughput scaling: `True`
