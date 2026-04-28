# Phase 02B EE Power Surface Audit Review

This audit is report-only. It does not train EE-MODQN and does not change the MODQN reward surface.

## Power Surface

- mode: `active-load-concave`
- inactive beam policy: `zero-w`
- denominator classification: `hobs-compatible-active-load-concave-power-surface`

## Runtime Mapping

- `R_i(t)`: `RewardComponents.r1_throughput`, computed from `user_throughputs` in `StepEnvironment._build_states_and_masks` and aggregated into `beam_throughputs`.
- `P_b(t)`: `StepResult.beam_transmit_power_w`, an explicit linear-W per-beam runtime vector.
- `active_beams`: `StepResult.active_beam_mask`; inactive beams are `zero-w`.

## Audit Result

- active power values sampled: `[1.03262379212, 1.10732140997, 1.17601295887, 1.23994949366, 1.3, 1.35679718106, 1.41081867662, 1.4624355653, 1.51194294641, 1.55958008537, 1.60554417117, 1.65, 1.69308696897, 1.73492424049, 1.77561463024, 1.81524758425, 1.85390149323, 1.89164551594, 1.96464281995, 2.0]`
- distinct active-beam counts sampled: `[1, 7]`
- distinct total active-beam powers sampled: `[2.0, 10.760084397829, 10.80169407404, 10.819683890854, 10.824536119422, 10.888042281874, 10.93342592142, 10.953017613855, 10.956729828119, 10.963803023484, 10.963898915639, 10.975454935198]`
- denominator varies: `True`
- HOBS EE defensible in current runtime: `True`

## Decision

- Phase 02B status: `promoted-for-metric-audit`
- Phase 03 gate: `conditional-go-for-paired-phase-03-design`
- EE degenerates to throughput scaling: `False`
- reason: The opt-in active-load-concave surface emits explicit linear-W per-beam P_b(t), assigns inactive beams 0 W, uses the same P_b(t) in the SINR numerator path, and the sampled denominator varies.
