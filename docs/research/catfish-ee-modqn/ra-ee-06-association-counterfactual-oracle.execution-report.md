# RA-EE-06 Execution Report: Association Counterfactual / Oracle Design Gate

**Date:** `2026-04-29`
**Status:** `BLOCKED for learned hierarchical association training`
**Scope:** offline active-set / served-set association counterfactual and
diagnostic association + power oracle only. No learned association, joint
association + power training, Catfish, multi-Catfish, RB / bandwidth
allocation, old EE-MODQN continuation, frozen baseline mutation, HOBS optimizer
claim, or physical energy-saving claim was performed.

## Protocol

Method label:

```text
RA-EE hierarchical association + power counterfactual
```

Config:

```text
configs/ra-ee-06-association-counterfactual-oracle.resolved.yaml
```

Artifacts:

```text
artifacts/ra-ee-06-association-counterfactual-oracle/
```

Matched control:

```text
fixed-hold-current association + safe-greedy-power-allocator
```

Candidate association proposals:

```text
active-set-load-spread
active-set-quality-spread
active-set-sticky-spread
```

Diagnostic comparator:

```text
per-user-greedy-best-beam
```

Diagnostic upper bound:

```text
association-oracle+constrained-power-upper-bound
```

The candidate proposals use a centralized active-set / served-set contract.
They first choose a bounded active beam set and then assign users into that
served set. The power allocator is embedded only as a post-association
optimizer. The RA-EE-04/05 `safe-greedy-power-allocator` is reused unchanged as
the matched control and candidate allocator.

## Power and QoS Contract

```text
power levels: [0.5, 0.75, 1.0, 1.5, 2.0] W
per-beam max: 2.0 W
total active-beam budget: 8.0 W
inactive beams: 0 W
p05 throughput guardrail: >= 95% matched control
served ratio guardrail: no drop versus matched control
outage guardrail: no increase versus matched control
```

For every evaluated step, the resolved `effective_power_vector_w` feeds the
SINR / SNR numerator, throughput, `EE_system = sum_i R_i / sum_active_beams
P_b`, audit logs, and budget checks.

## Results

RA-EE-06 is blocked for learned hierarchical training.

Held-out candidate proposal results versus matched fixed association +
`safe-greedy-power-allocator`:

| Association proposal | EE delta | p05 ratio | Accepted |
|---|---:|---:|---|
| `active-set-load-spread` | `-2.919243206981264` | `0.39743071159396487` | `false` |
| `active-set-quality-spread` | `-2.919243206981264` | `0.39743071159396487` | `false` |
| `active-set-sticky-spread` | `-1.7672459194036492` | `1.2382114044314652` | `false` |

Held-out diagnostic oracle results versus the same matched control:

| Diagnostic oracle | EE delta | p05 ratio | Accepted |
|---|---:|---:|---|
| `association-oracle+constrained-power-upper-bound` | `+0.44226236820429676` | `0.9711843089200792` | `true` |

The key split is:

1. the active-set proposal contract avoids one-active-beam collapse,
2. the tested proposal rules do not beat the matched fixed-association +
   safe-greedy allocator,
3. the diagnostic oracle shows a positive held-out upper-bound path still
   exists under the same QoS and power constraints.

## Gate Flags

```text
held_out_bucket_exists_and_reported_separately = true
association_counterfactual_only = true
learned_association_disabled = true
joint_association_power_training_disabled = true
catfish_disabled = true
multi_catfish_disabled = true
rb_bandwidth_allocation_disabled = true
active_set_served_set_proposal_contract = true
matched_control_uses_same_power_allocator = true
safe_greedy_allocator_retained = true
constrained_oracle_upper_bound_diagnostic_only = true
majority_noncollapsed_held_out_positive_EE_delta = false
majority_noncollapsed_held_out_accepted = false
held_out_gains_not_concentrated_in_one_policy = false
p05_throughput_guardrail_pass_for_accepted_held_out = false
served_ratio_does_not_drop_for_accepted_held_out = false
outage_ratio_does_not_increase_for_accepted_held_out = false
zero_budget_per_beam_inactive_power_violations = true
denominator_varies_for_accepted_held_out = false
one_active_beam_collapse_avoided_for_accepted_held_out = true
scalar_reward_success_basis = false
per_user_EE_credit_success_basis = false
physical_energy_saving_claim = false
hobs_optimizer_claim = false
full_RA_EE_MODQN_claim = false
```

## Validation

Focused tests:

```text
.venv/bin/python -m pytest tests/test_ra_ee_06_association_counterfactual_oracle.py -q
```

Result:

```text
5 passed
```

Regression-focused RA-EE tests:

```text
.venv/bin/python -m pytest \
  tests/test_ra_ee_04_bounded_power_allocator.py \
  tests/test_ra_ee_05_fixed_association_robustness.py \
  tests/test_ra_ee_06_association_counterfactual_oracle.py -q
```

Result:

```text
18 passed
```

Artifact command:

```text
.venv/bin/python scripts/run_ra_ee_06_association_counterfactual_oracle.py
```

Result:

```text
decision=BLOCKED
```

## Decision

```text
RA-EE-06: BLOCKED for learned hierarchical association training
```

This result does not close the RA-EE association route entirely. It says the
current minimal active-set proposal rules are not sufficient. The constrained
association + power oracle remains diagnostic evidence that a better
association proposal layer may exist.

## Allowed Next Step

The allowed next step is a narrower RA-EE-06B design iteration:

```text
association proposal refinement / oracle distillation only
```

That next step may use the RA-EE-06 oracle traces to design a stronger
proposal rule or imitation target. It still must not start learned hierarchical
RL until a proposal-level counterfactual gate passes.

## Stop Conditions Triggered

```text
held_out_association_gains_disappear_or_concentrate = true
```

The following stop conditions did not trigger:

```text
association_proposals_collapse_to_one_active_beam = false
p05_throughput_guardrail_fails = false
budget_or_inactive_power_violations = false
learned_association_added = false
joint_training_added = false
catfish_added = false
frozen_baseline_mutated = false
oracle_used_as_candidate_claim = false
```

## Forbidden Claims Still Active

1. Do not call RA-EE-06 full RA-EE-MODQN.
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
