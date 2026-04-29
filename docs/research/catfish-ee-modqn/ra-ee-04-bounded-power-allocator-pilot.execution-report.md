# RA-EE-04 Execution Report: Bounded Centralized Power-Allocator Pilot

**Date:** `2026-04-29`
**Status:** `PASS as bounded fixed-association power-allocation pilot only`
**Scope:** fixed-association centralized per-beam power allocation. No learned
association, Catfish, multi-Catfish, old EE-MODQN continuation, long training,
HOBS optimizer claim, physical energy-saving claim, or frozen baseline mutation
was performed.

## Protocol

Method label:

```text
RA-EE-MDP
```

Implementation sublabel:

```text
RA-EE-04 fixed-association power-allocation pilot
```

Configs:

```text
configs/ra-ee-04-bounded-power-allocator-control.resolved.yaml
configs/ra-ee-04-bounded-power-allocator-candidate.resolved.yaml
```

Artifacts:

```text
artifacts/ra-ee-04-bounded-power-allocator-control-pilot/
artifacts/ra-ee-04-bounded-power-allocator-candidate-pilot/
artifacts/ra-ee-04-bounded-power-allocator-candidate-pilot/paired-comparison-vs-control/
```

Fixed association trajectories used as primary evidence:

1. `hold-current`
2. `random-valid`
3. `spread-valid`

Control:

```text
fixed association + fixed 1.0 W per active beam, inactive beams 0 W
```

Candidate:

```text
safe-greedy-power-allocator
```

The candidate is a centralized adaptive allocator over fixed trajectories. It
starts from fixed `1.0 W` per active beam and accepts finite-codebook
per-beam demotions only when system EE improves and p05 throughput, served
ratio, outage, total budget, per-beam power, and inactive-beam-zero guardrails
remain valid versus matched fixed control.

This is not a HOBS optimizer and not a learned association policy.

## Power Contract

Action:

```text
centralized per-active-beam discrete power vector
levels: [0.5, 0.75, 1.0, 1.5, 2.0] W
inactive beams: 0 W
per-beam max power: 2.0 W
total active-beam budget: 8.0 W
```

For each step, the pilot resolves one authoritative
`effective_power_vector_w`. The same vector feeds:

1. SINR / SNR numerator,
2. throughput `R_i(t)`,
3. `EE_system = sum_i R_i(t) / sum_active_beams P_b(t)`,
4. audit logs,
5. budget and inactive-power checks.

No power repair was used. Requested and effective vectors are still exported.

## Training Budget

The only training-like surface is bounded calibration:

```text
training episodes: 20
train seed: 42
environment seed: 1337
mobility seed: 7
evaluation seeds: [100, 200, 300, 400, 500]
```

No `500`-episode or `9000`-episode run was started.

## Results

Primary candidate versus matched fixed control:

| Trajectory | Control EE | Candidate EE | EE delta | p05 ratio | QoS | Budget |
|---|---:|---:|---:|---:|---|---|
| `hold-current` | `872.6413880638145` | `875.8117405506861` | `+3.170352486871593` | `0.9762855249221921` | pass | pass |
| `random-valid` | `874.0261962030479` | `875.9733777320105` | `+1.9471815289625738` | `0.9645767924959346` | pass | pass |
| `spread-valid` | `873.055555234289` | `873.1048097977671` | `+0.04925456347802992` | `0.9932220296454757` | pass | pass |

Upper-bound diagnostic was also exported as
`constrained-oracle-upper-bound`; it remains diagnostic only and is not the
learned/adaptive comparator claim.

## Denominator Variability Result

The candidate passed the denominator gate on all primary non-collapsed fixed
trajectories:

```text
denominator_varies_in_eval = true
selected_power_vector_not_single_point = true
total_active_power_not_single_point = true
```

Candidate total-active-power distributions:

1. `hold-current`: `[5.75, 6.0, 6.25, 6.5, 6.75, 7.0] W`
2. `random-valid`: `[5.75, 6.0, 6.25, 6.5, 6.75, 7.0] W`
3. `spread-valid`: `[6.75, 7.0] W`

## QoS Guardrail Result

All primary trajectories passed:

1. p05 throughput >= `95%` of matched fixed control,
2. served ratio did not drop,
3. outage ratio did not increase,
4. zero total-budget violations,
5. zero per-beam max-power violations,
6. zero inactive-beam nonzero-power violations.

## Ranking Separation Result

For each primary trajectory, throughput and EE ranking separated between
fixed control and candidate:

| Trajectory | Throughput winner | EE winner | Ranking changed |
|---|---|---|---|
| `hold-current` | `fixed-control` | `safe-greedy-power-allocator` | `true` |
| `random-valid` | `fixed-control` | `safe-greedy-power-allocator` | `true` |
| `spread-valid` | `fixed-control` | `safe-greedy-power-allocator` | `true` |

Scalar reward was diagnostic only and was not used as success evidence.

## Validation

Focused tests:

```text
.venv/bin/python -m pytest tests/test_ra_ee_04_bounded_power_allocator.py -q
```

Result:

```text
6 passed
```

Pilot command:

```text
.venv/bin/python scripts/run_ra_ee_04_bounded_power_allocator.py
```

Result:

```text
decision=PASS
```

## Decision

```text
RA-EE-04: PASS
```

This PASS means only that the bounded fixed-association centralized
power-allocation pilot passed its implementation gate. It does not establish
full RA-EE-MODQN effectiveness.

## Remaining Blockers

1. Association is still fixed by trajectory; no learned RA-EE association
   policy exists.
2. The candidate is a bounded adaptive allocator, not a HOBS optimizer.
3. The result does not authorize Catfish, multi-Catfish, Catfish-EE-MODQN, or
   old EE-MODQN continuation.
4. A follow-on learned association or hierarchical RA-EE route requires a new
   design gate.

## Forbidden Claims Still Active

Do not claim:

1. old EE-MODQN effectiveness,
2. HOBS optimizer behavior,
3. Catfish, multi-Catfish, or Catfish-EE-MODQN effectiveness,
4. full paper-faithful reproduction,
5. physical energy saving,
6. per-user EE credit as system EE,
7. scalar reward alone as success evidence,
8. learned association or full RA-EE-MODQN effectiveness from RA-EE-04.
