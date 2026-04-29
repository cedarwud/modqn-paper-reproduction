# RA-EE-02 Execution Report: Oracle Power-Allocation Audit

**Date:** `2026-04-28`
**Status:** `PASS to RA-EE-03 design`
**Scope:** offline fixed-association oracle / heuristic upper-bound proof only.
No RL training, Catfish, multi-Catfish, old EE-MODQN promotion, HOBS optimizer
claim, or frozen baseline mutation was performed.

## Protocol

Config:

```text
configs/ra-ee-02-oracle-power-allocation-audit.resolved.yaml
```

Artifacts:

```text
artifacts/ra-ee-02-oracle-power-allocation-audit/
```

Method label:

```text
RA-EE-MDP / RA-EE-MODQN
```

The audit replays fixed association trajectories and rescores them under
finite-codebook power allocations. A unit-power replay recovers fixed-trajectory
per-user unit-SINR; each candidate power vector then feeds both the SINR
numerator and the `EE_system = sum_i R_i / sum_active_beams P_b` denominator.

Audited trajectories:

1. `hold-current`
2. `random-valid`
3. `spread-valid`
4. existing `phase03c-c-candidate-best-eval`, when available

Power candidates:

1. `fixed-control`
2. `load-concave`
3. `budget-trim`
4. `qos-tail-boost`
5. `constrained-oracle`

Constraints:

```text
per-beam max power: 2.0 W
total active-beam budget: 8.0 W
inactive beams: 0 W
p05 throughput guardrail: >= 95% matched fixed control
served ratio guardrail: no drop versus fixed control
outage guardrail: no increase versus fixed control
```

## Result

RA-EE-02 passes as an offline upper-bound proof. The proof candidates are the
`constrained-oracle` allocations on non-collapsed trajectories:

| Trajectory | EE delta vs fixed control | p05 ratio vs fixed control | QoS | Budget |
|---|---:|---:|---|---|
| `hold-current` | `+3.189055583818231` | `0.9818337745575232` | pass | pass |
| `random-valid` | `+2.214164300102084` | `0.9731482959399329` | pass | pass |
| `spread-valid` | `+1.039861022592504` | `1.0117000462649277` | pass | pass |

All proof candidates had zero total-budget violations, zero per-beam max-power
violations, zero inactive-beam nonzero-power violations, served ratio `1.0`, and
outage ratio `0.0`.

The existing Phase 03C-C learned trajectory was also audited, but it remained a
one-active-beam collapsed trajectory and did not contribute to the PASS
decision.

## Proof Flags

```text
denominator_changed_by_power_decision = true
denominator_varies_for_accepted_candidate = true
ranking_separates_under_same_policy_rescore = true
has_budget_respecting_candidate = true
oracle_or_heuristic_beats_fixed_control_on_EE = true
QoS_guardrails_pass = true
selected_profile_not_single_point_on_noncollapsed_trajectories = true
active_power_not_single_point_on_noncollapsed_trajectories = true
no_budget_violations_for_accepted_candidate = true
```

## Decision

```text
RA-EE-02: PASS to RA-EE-03 design
```

This is not a training result. It authorizes a design step for a
resource-allocation EE-MDP / RA-EE-MODQN only.

## Remaining Blockers

1. No learned RA-EE-MODQN policy exists yet.
2. The oracle is a bounded finite-codebook upper-bound surface, not a HOBS
   optimizer.
3. The Phase 03C-C learned policy trajectory remains collapsed and does not
   support old EE-MODQN effectiveness.
4. RA-EE-03 must define the actual resource-allocation state/action contract
   before any training can be considered.

## Forbidden Claims Still Active

1. Do not claim old EE-MODQN effectiveness.
2. Do not claim HOBS optimizer behavior.
3. Do not claim Catfish, multi-Catfish, or Catfish-EE-MODQN effectiveness.
4. Do not treat per-user EE credit as system EE.
5. Do not use scalar reward alone as success evidence.
6. Do not claim full paper-faithful reproduction or absolute physical energy
   saving.
