# Phase 03A Diagnostic Review

Diagnostic-only follow-up. No reward change, Catfish, multi-Catfish, or long 9000-episode training was run.

## Primary Learned Policy

- policy: `EE-MODQN/best-eval`
- active beam count distinct values: `[1.0]`
- total active beam power distinct values: `[2.0]`
- all-users-one-beam step ratio: `1.0`
- valid action count mean: `7.0`
- mean EE-credit / throughput ratio: `50.0`
- Q near-zero margin ratio: `0.0`
- denominator varies in learned eval: `False`

## Counterfactuals

- first-valid is a deliberate collapse control and keeps one active beam.
- hold-current active beam count distinct values: `[7.0]`
- random-valid total active beam power distinct values include: `[10.792375806669591, 10.840162489872052, 10.843045499989373, 10.8525758572524, 10.871629015607677]`
- spread-valid active beam count distinct values: `[7.0]`

## Root Cause

- denominator exists in environment: `True`
- learned policies exercise denominator variability: `False`
- action mask forces single beam: `False`
- Q near-tie primary cause: `False`
- r3 load balance insufficient to prevent collapse: `True`
- per-user EE degenerates under collapse: `True`
- Phase 02B power formula globally fixed: `False`

## Next Gate

- more training budget: `not the next gate by itself; first make the objective surface exercise denominator variability`
- reward normalization: `required design candidate before larger training`
- load-balance calibration: `required design candidate before larger training`
- denominator-sensitive action/power design remains required before larger training.

## Decision

- Phase 03 remains `NEEDS MORE EVIDENCE`.
- Do not claim EE-MODQN effectiveness from this diagnostic.
