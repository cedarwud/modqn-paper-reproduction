# RA-EE-06B Execution Report: Association Proposal Refinement / Oracle Distillation Audit

**Date:** `2026-04-29`
**Status:** `BLOCKED for learned hierarchical association training`
**Scope:** offline oracle-trace export and deterministic proposal-rule
association refinement only. No learned hierarchical RL, joint association +
power training, Catfish, multi-Catfish, RB / bandwidth allocation, old
EE-MODQN continuation, frozen baseline mutation, HOBS optimizer claim, or
physical energy-saving claim was performed.

## Protocol

Method label:

```text
RA-EE association proposal refinement / oracle distillation audit
```

Config:

```text
configs/ra-ee-06b-association-proposal-refinement.resolved.yaml
```

Artifacts:

```text
artifacts/ra-ee-06b-association-proposal-refinement/
```

Primary matched control:

```text
fixed-hold-current association + safe-greedy-power-allocator
```

Primary candidate:

```text
deterministic proposal rule + same safe-greedy-power-allocator
```

Proposal rules:

```text
sticky-oracle-count-local-search
p05-slack-aware-active-set
power-response-aware-load-balance
bounded-move-served-set
oracle-score-topk-active-set
```

Diagnostic comparators:

```text
proposal + fixed-1W
per-user-greedy-best-beam + safe-greedy-power-allocator
association-oracle + same-safe-greedy-power-allocator
association-oracle + constrained-power-upper-bound
matched fixed association + constrained-power oracle isolation
```

Only the primary candidate versus matched fixed association with the same
safe-greedy allocator is eligible for the gate. Oracle rows are diagnostic
only.

## Trace Export

The per-step oracle trace is exported at:

```text
artifacts/ra-ee-06b-association-proposal-refinement/ra_ee_06b_oracle_trace.csv
```

The trace includes active beam count and mask, active-set size and source
policy, beam load distribution and load max/min/std, load cap slack, load
balance gap, selected quality, top-k quality, best-vs-selected margin, valid
beam count, current/control/oracle/selected beams, moved flags, rank and beam
offset distance proxies, moved-user count, handover count and r2, p05
control/candidate/oracle throughput, p05 ratio and slack to the 0.95 threshold,
selected/effective power vector, demoted beams, total active power,
denominator, safe-greedy demotion counts, active-beam throughput gap, tail-user
IDs, oracle selected association policy, oracle power profile, EE delta,
accepted flag, and rejection reason.

## Held-Out Results

Held-out candidate proposal results versus matched fixed association +
`safe-greedy-power-allocator`:

| Proposal rule | EE delta | p05 ratio | moved-user ratio | Denominator varies | Accepted |
|---|---:|---:|---:|---|---|
| `sticky-oracle-count-local-search` | `-1.7672459194036492` | `1.2382114044314652` | `0.0054` | true | false |
| `p05-slack-aware-active-set` | `-1.7672459194036492` | `1.2382114044314652` | `0.0054` | true | false |
| `power-response-aware-load-balance` | `-1.7672459194036492` | `1.2382114044314652` | `0.0054` | true | false |
| `bounded-move-served-set` | `-1.7672459194036492` | `1.2382114044314652` | `0.0054` | true | false |
| `oracle-score-topk-active-set` | `-2.8533857556310522` | `1.3748199249811` | `0.0854` | true | false |

All five proposal candidates remained noncollapsed on the held-out bucket and
kept active beam count at `7`. The accepted-candidate count is still `0`
because every primary proposal has negative EE delta versus matched fixed
association using the same safe-greedy allocator.

Diagnostic oracle results:

| Oracle diagnostic | EE delta | p05 ratio | Role |
|---|---:|---:|---|
| `association-oracle+same-safe-greedy-diagnostic` | `-1.476841467806139` | `1.248435499914564` | diagnostic only |
| `association-oracle+constrained-power-upper-bound` | `+0.9583236804177204` | `1.0204597070289083` | upper bound only |

This split is the blocking result: the positive held-out upper bound requires
constrained-oracle power, not the same safe-greedy allocator used by the
matched candidate/control comparison.

## Gate Flags

```text
held_out_bucket_exists_and_reported_separately = true
offline_trace_export_only = true
deterministic_proposal_refinement_only = true
learned_hierarchical_RL_disabled = true
joint_association_power_training_disabled = true
catfish_disabled = true
multi_catfish_disabled = true
rb_bandwidth_allocation_disabled = true
matched_control_uses_same_power_allocator = true
safe_greedy_allocator_retained = true
oracle_diagnostic_only = true
majority_noncollapsed_held_out_positive_EE_delta = false
majority_noncollapsed_held_out_accepted = false
held_out_gains_not_concentrated_in_one_policy = false
zero_budget_per_beam_inactive_power_violations = true
one_active_beam_or_two_beam_overload_collapse_avoided = true
candidate_closes_meaningful_oracle_gap = false
scalar_reward_success_basis = false
physical_energy_saving_claim = false
hobs_optimizer_claim = false
```

Stop conditions triggered:

```text
proposal_gains_require_constrained_oracle_power = true
held_out_EE_delta_negative_or_concentrated = true
```

## Validation

Focused tests:

```text
.venv/bin/python -m pytest tests/test_ra_ee_06b_association_proposal_refinement.py -q
```

Result:

```text
8 passed
```

Regression-focused RA-EE tests:

```text
.venv/bin/python -m pytest \
  tests/test_ra_ee_04_bounded_power_allocator.py \
  tests/test_ra_ee_05_fixed_association_robustness.py \
  tests/test_ra_ee_06_association_counterfactual_oracle.py \
  tests/test_ra_ee_06b_association_proposal_refinement.py -q
```

Result:

```text
26 passed
```

Artifact command:

```text
.venv/bin/python scripts/run_ra_ee_06b_association_proposal_refinement.py
```

Result:

```text
decision=BLOCKED
```

## Decision

```text
RA-EE-06B: BLOCKED
```

RA-EE-06B does not authorize learned hierarchical RA-EE training. The refined
proposal rules improved trace visibility and avoided one-active/two-beam
collapse, but they did not beat matched fixed association under the same
safe-greedy allocator. The remaining positive path is still diagnostic-only
and depends on constrained-oracle power.

## Remaining Blockers

1. No deterministic proposal rule has positive held-out EE delta versus
   matched fixed association + same safe-greedy allocator.
2. No candidate closes a meaningful portion of the diagnostic oracle gap.
3. The positive upper-bound path requires constrained-oracle power.
4. No learned hierarchical association or full RA-EE-MODQN policy exists.
5. No joint association + power training or RB / bandwidth allocation exists.

## Forbidden Claims Still Active

1. Do not call RA-EE-06B full RA-EE-MODQN.
2. Do not claim learned hierarchical RL or learned association effectiveness.
3. Do not claim joint association + power training.
4. Do not claim old EE-MODQN effectiveness.
5. Do not claim HOBS optimizer behavior.
6. Do not claim Catfish, multi-Catfish, or Catfish-EE-MODQN effectiveness.
7. Do not add or claim RB / bandwidth allocation.
8. Do not treat per-user EE credit as system EE.
9. Do not use scalar reward alone as success evidence.
10. Do not claim full paper-faithful reproduction.
11. Do not claim physical energy saving.
