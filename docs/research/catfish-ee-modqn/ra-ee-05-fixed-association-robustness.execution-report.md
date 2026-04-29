# RA-EE-05 Execution Report: Fixed-Association Robustness and Held-Out Validation

**Date:** `2026-04-29`
**Status:** `PASS as fixed-association robustness evidence only`
**Scope:** fixed-association centralized power allocation over calibration and
held-out trajectory buckets. No learned association, joint association + power
training, Catfish, multi-Catfish, RB / bandwidth allocation, old EE-MODQN
continuation, frozen baseline mutation, HOBS optimizer claim, or physical
energy-saving claim was performed.

## Protocol

Method label:

```text
RA-EE fixed-association centralized power allocator
```

Config:

```text
configs/ra-ee-05-fixed-association-robustness.resolved.yaml
```

Artifacts:

```text
artifacts/ra-ee-05-fixed-association-robustness/
```

Candidates:

1. `fixed-control`: fixed `1.0 W` per active beam, inactive beams `0 W`.
2. `safe-greedy-power-allocator`: centralized per-active-beam discrete power
   allocator from RA-EE-04, with a `0.005` step-level p05 safety margin for the
   held-out robustness gate.
3. `constrained-oracle-upper-bound`: diagnostic upper bound only.

Power contract:

```text
levels: [0.5, 0.75, 1.0, 1.5, 2.0] W
per-beam max: 2.0 W
total active-beam budget: 8.0 W
inactive beams: 0 W
```

For every step, the same `effective_power_vector_w` feeds SINR / SNR
numerator, throughput `R_i(t)`, `EE_system = sum_i R_i(t) / sum_active_beams
P_b(t)`, audit logs, and budget checks.

## Buckets

Calibration / train-like bucket:

```text
seeds: [100, 200, 300, 400, 500]
trajectories: [hold-current, random-valid, spread-valid]
```

Held-out bucket:

```text
seeds: [600, 700, 800, 900, 1000]
trajectories:
- random-valid-heldout
- spread-valid-heldout
- load-skewed-heldout
- mobility-shift-heldout
- mixed-valid-heldout
```

The held-out families shift association randomness, spread tie-break behavior,
load skew, periodic reassignment, and mixed association patterns. Association
is fixed by each generated trajectory before power evaluation; no association
policy is trained.

## Results

Calibration candidate versus fixed control:

| Trajectory | EE delta | p05 throughput ratio | Accepted |
|---|---:|---:|---|
| `hold-current` | `+3.0756706652333605` | `0.980130088896158` | `true` |
| `random-valid` | `+1.856608844948937` | `0.9691160803712308` | `true` |
| `spread-valid` | `+0.034912718586951996` | `0.995020283044902` | `true` |

Held-out candidate versus fixed control:

| Trajectory | EE delta | p05 throughput ratio | Throughput winner | EE winner | Accepted |
|---|---:|---:|---|---|---|
| `random-valid-heldout` | `+2.8552862308795284` | `0.9549524027957726` | `fixed-control` | `safe-greedy-power-allocator` | `true` |
| `spread-valid-heldout` | `+0.021591980638277164` | `0.9963484525708493` | `fixed-control` | `safe-greedy-power-allocator` | `true` |
| `load-skewed-heldout` | `+1.8625878077186826` | `1.0` | `fixed-control` | `safe-greedy-power-allocator` | `true` |
| `mobility-shift-heldout` | `+2.238167231124862` | `0.9532709399187361` | `fixed-control` | `safe-greedy-power-allocator` | `true` |
| `mixed-valid-heldout` | `+1.491732050135397` | `1.0` | `fixed-control` | `safe-greedy-power-allocator` | `true` |

Held-out summary:

```text
noncollapsed held-out trajectories: 5 / 5
positive EE delta trajectories: 5 / 5
accepted candidate trajectories: 5 / 5
p05 throughput guardrail: pass
served ratio guardrail: pass
outage ratio guardrail: pass
budget / per-beam / inactive-power violations: 0
denominator_varies_in_eval for accepted trajectories: true
selected power vectors not single-point: true
total active power not single-point: true
throughput winner and EE winner separate: true
```

The constrained oracle remains diagnostic only. Oracle gaps are exported in the
summary JSON and are not counted as the candidate claim.

## Validation

Focused tests:

```text
.venv/bin/python -m pytest tests/test_ra_ee_05_fixed_association_robustness.py -q
```

Result:

```text
7 passed
```

Artifact command:

```text
.venv/bin/python scripts/run_ra_ee_05_fixed_association_robustness.py
```

Result:

```text
decision=PASS
```

## Decision

```text
RA-EE-05: PASS
```

This PASS means only that the fixed-association centralized power allocator
passed the robustness and held-out validation gate. It does not establish
learned association, full RA-EE-MODQN effectiveness, Catfish effectiveness,
HOBS optimizer behavior, RB / bandwidth allocation, full paper-faithful
reproduction, or physical energy saving.

## Remaining Blockers

1. No learned association or full RA-EE-MODQN policy exists.
2. No joint association + power training exists.
3. No RB / bandwidth allocation exists.
4. The oracle upper bound remains diagnostic only.
5. The result remains a fixed-association power-allocation robustness result.

## Forbidden Claims Still Active

1. Do not call RA-EE-05 full RA-EE-MODQN.
2. Do not claim learned association effectiveness.
3. Do not claim joint association + power training.
4. Do not claim old EE-MODQN effectiveness.
5. Do not claim HOBS optimizer behavior.
6. Do not claim Catfish, multi-Catfish, or Catfish-EE-MODQN effectiveness.
7. Do not add or claim RB / bandwidth allocation.
8. Do not treat per-user EE credit as system EE.
9. Do not use scalar reward alone as success evidence.
10. Do not claim full paper-faithful reproduction.
11. Do not claim physical energy saving.
