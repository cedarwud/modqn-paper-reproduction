# RA-EE-08 Execution Report: Offline Association Re-Evaluation

**Date:** `2026-04-29`
**Status:** `BLOCKED`
**Scope:** offline deterministic association proposal replay only. No training,
learned association, learned hierarchical RL, joint association + power
training, Catfish, multi-Catfish, RB / bandwidth allocation, old EE-MODQN
continuation, frozen baseline mutation, oracle runtime method, HOBS optimizer
claim, physical energy-saving claim, or full RA-EE-MODQN claim was performed.

## Protocol

Method label:

```text
RA-EE offline association re-evaluation gate
```

Config:

```text
configs/ra-ee-08-offline-association-reevaluation.resolved.yaml
```

Artifacts:

```text
artifacts/ra-ee-08-offline-association-reevaluation/
```

Primary matched comparison:

```text
matched fixed association + deployable-stronger-power-allocator
vs
proposal association + same deployable-stronger-power-allocator
```

The deployable allocator candidates are the RA-EE-07 candidates:

1. `p05-slack-aware-trim-tail-protect-boost`
2. `bounded-local-search-codebook`
3. `finite-codebook-dp-knapsack`
4. `deterministic-hybrid-runtime`

The primary deployable allocator is `deterministic-hybrid-runtime`. The
candidate and control use the same seeds, held-out bucket, power codebook,
budget, p05 / served / outage guardrails, and `effective_power_vector_w` audit
path. The primary comparison has no step-cap mismatch.

Association proposal families replayed:

1. `active-set-load-spread`
2. `active-set-quality-spread`
3. `active-set-sticky-spread`
4. `sticky-oracle-count-local-search`
5. `p05-slack-aware-active-set`
6. `power-response-aware-load-balance`
7. `bounded-move-served-set`
8. `oracle-score-topk-active-set`

The predeclared primary proposal family is
`power-response-aware-load-balance`.

Diagnostic-only rows:

1. `association-proposal+safe-greedy-diagnostic`
2. `matched-fixed-association+safe-greedy-diagnostic`
3. `matched-fixed-association+constrained-power-oracle-diagnostic`
4. `association-oracle+constrained-power-oracle-diagnostic`
5. `association-oracle+deployable-stronger-power-allocator-diagnostic`

Only the primary candidate versus matched fixed + same deployable allocator is
eligible for acceptance. Safe-greedy and oracle rows are not deployable methods
and do not count for the RA-EE-08 decision.

## Held-Out Results

Held-out candidate proposal results versus matched fixed association +
`deployable-stronger-power-allocator`:

| Proposal rule | EE delta | p05 ratio | moved-user ratio | Denominator varies | Accepted |
|---|---:|---:|---:|---|---|
| `active-set-load-spread` | `-8.283605568482926` | `0.42255785095605874` | `0.9876` | false | false |
| `active-set-quality-spread` | `-8.283605568482926` | `0.42255785095605874` | `0.0846` | false | false |
| `active-set-sticky-spread` | `-3.3928346793936726` | `1.2387973192040846` | `0.0054` | true | false |
| `sticky-oracle-count-local-search` | `-3.3928346793936726` | `1.2387973192040846` | `0.0054` | true | false |
| `p05-slack-aware-active-set` | `-3.3928346793936726` | `1.2387973192040846` | `0.0054` | true | false |
| `power-response-aware-load-balance` | `-3.3928346793936726` | `1.2387973192040846` | `0.0054` | true | false |
| `bounded-move-served-set` | `-3.3928346793936726` | `1.2387973192040846` | `0.0054` | true | false |
| `oracle-score-topk-active-set` | `-6.070621808780743` | `1.3889638194877596` | `0.0854` | true | false |

All eight proposal families avoid one-active-beam collapse on held-out replay,
but none has positive `EE_system` delta versus matched fixed association with
the same deployable allocator. The predeclared primary family
`power-response-aware-load-balance` is also negative.

Held-out seed-level aggregate:

```text
positive seeds: 0 / 5
seed 600 EE delta: -4.071519962830848
seed 700 EE delta: -6.413819030206355
seed 800 EE delta: -3.11617509602695
seed 900 EE delta: -2.843222154472869
seed 1000 EE delta: -5.302497167680485
```

Because no proposal is positive, gain concentration checks cannot pass.

## QoS / Handover Burden

No budget, per-beam, or inactive-power violations were observed for the primary
candidate rows.

The QoS guardrails are not the main blocker for the six higher-active-set
families; their p05 ratios are above `0.95`, served ratio does not drop, and
outage ratio does not increase. The two two-active-beam active-set variants
also fail p05 with ratio `0.42255785095605874`, and
`active-set-load-spread` has unbounded movement burden with moved-user ratio
`0.9876`.

## Oracle Gap Result

RA-EE-08 does not close a meaningful deployable/oracle gap. In this matched
comparison, the diagnostic association-oracle rows are also below the fixed +
deployable control:

```text
association-oracle + deployable allocator EE delta: -2.7592661903154294
association-oracle + constrained-power oracle EE delta: -7.442920209643489
aggregate oracle gap closure: not applicable
```

This means the stronger fixed-association deployable allocator from RA-EE-07
erases the earlier positive association-oracle path under this RA-EE-08
matched comparison.

## Gate Flags

```text
held_out_bucket_exists_and_reported_separately = true
offline_replay_only = true
deterministic_association_proposals_only = true
learned_association_disabled = true
learned_hierarchical_RL_disabled = true
association_training_disabled = true
joint_association_power_training_disabled = true
catfish_disabled = true
multi_catfish_disabled = true
rb_bandwidth_allocation_disabled = true
old_EE_MODQN_continuation_disabled = true
frozen_baseline_mutation = false
matched_control_uses_same_deployable_allocator = true
primary_comparison_uses_same_deployable_allocator = true
primary_comparison_no_step_cap_mismatch = true
same_power_codebook_and_budget = true
same_effective_power_vector_feeds_numerator_denominator_audit = true
oracle_diagnostic_only = true
oracle_rows_excluded_from_acceptance = true
candidate_does_not_use_oracle_labels_or_future_or_heldout_answers = true
majority_noncollapsed_held_out_positive_EE_delta = false
predeclared_primary_held_out_positive_EE_delta = false
majority_or_predeclared_primary_held_out_positive_EE_delta = false
majority_or_predeclared_primary_held_out_accepted = false
held_out_gains_not_concentrated_in_one_policy = false
held_out_gains_not_concentrated_in_one_seed = false
p05_throughput_guardrail_pass_for_accepted_held_out = false
served_ratio_does_not_drop_for_accepted_held_out = false
outage_ratio_does_not_increase_for_accepted_held_out = false
zero_budget_per_beam_inactive_power_violations = true
denominator_varies_for_accepted_held_out = false
active_beam_behavior_noncollapsed_for_accepted_held_out = false
handover_burden_bounded_for_accepted_held_out = false
candidate_closes_meaningful_oracle_gap = false
ranking_separates_or_oracle_gap_reduction_clear = false
scalar_reward_success_basis = false
per_user_EE_credit_success_basis = false
physical_energy_saving_claim = false
hobs_optimizer_claim = false
full_RA_EE_MODQN_claim = false
```

## Validation

Focused tests:

```text
.venv/bin/python -m pytest tests/test_ra_ee_08_offline_association_reevaluation.py -q
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
  tests/test_ra_ee_06b_association_proposal_refinement.py \
  tests/test_ra_ee_07_constrained_power_allocator_distillation.py \
  tests/test_ra_ee_08_offline_association_reevaluation.py -q
```

Result:

```text
43 passed
```

Artifact command:

```text
.venv/bin/python scripts/run_ra_ee_08_offline_association_reevaluation.py
```

Result:

```text
decision=BLOCKED
```

## Decision

```text
RA-EE-08: BLOCKED
```

RA-EE-08 does not authorize learned association, hierarchical RL, joint
association + power training, Catfish, RB / bandwidth allocation, or a full
RA-EE-MODQN claim. The association proposal gains vanish when paired fairly
against fixed association with the same deployable stronger power allocator.

## Remaining Blockers

1. No held-out proposal family has positive `EE_system` delta versus matched
   fixed association + same deployable allocator.
2. The predeclared primary proposal family is also negative.
3. No accepted candidate exists, so QoS-for-accepted, denominator-for-accepted,
   handover-for-accepted, ranking separation, and gain concentration gates
   cannot pass.
4. No meaningful deployable/oracle gap closure remains under the RA-EE-08
   matched comparison.
5. No learned association, joint association + power training, or RB /
   bandwidth allocation exists.

## Forbidden Claims Still Active

1. Do not call RA-EE-08 full RA-EE-MODQN.
2. Do not claim learned hierarchical RL or learned association effectiveness.
3. Do not claim joint association + power training.
4. Do not claim old EE-MODQN effectiveness.
5. Do not claim HOBS optimizer behavior.
6. Do not claim Catfish, multi-Catfish, or Catfish-EE-MODQN effectiveness.
7. Do not add or claim RB / bandwidth allocation.
8. Do not treat per-user EE credit as system EE.
9. Do not use scalar reward alone as success evidence.
10. Do not use oracle rows as deployable runtime methods.
11. Do not claim full paper-faithful reproduction.
12. Do not claim physical energy saving.
