# RA-EE-08 Offline Association Re-Evaluation Review

Offline deterministic association proposal replay only. Primary rows compare fixed association + deployable stronger power allocator against proposal association + the same deployable stronger power allocator. Safe-greedy and oracle rows are diagnostic-only. No training, learned association, hierarchical RL, joint association + power training, Catfish, multi-Catfish, RB / bandwidth allocation, oracle runtime method, or frozen baseline mutation was performed.

## Protocol

- method label: `RA-EE offline association re-evaluation gate`
- implementation sublabel: `RA-EE-08 offline association re-evaluation gate`
- config: `configs/ra-ee-08-offline-association-reevaluation.resolved.yaml`
- artifact namespace: `artifacts/ra-ee-08-offline-association-reevaluation`
- primary control: `matched-fixed-association+deployable-stronger-power-allocator`
- primary candidate: `association-proposal+same-deployable-stronger-power-allocator`
- proposal families: `['active-set-load-spread', 'active-set-quality-spread', 'active-set-sticky-spread', 'sticky-oracle-count-local-search', 'p05-slack-aware-active-set', 'power-response-aware-load-balance', 'bounded-move-served-set', 'oracle-score-topk-active-set']`
- deployable allocator candidates: `['p05-slack-aware-trim-tail-protect-boost', 'bounded-local-search-codebook', 'finite-codebook-dp-knapsack', 'deterministic-hybrid-runtime']`
- primary deployable allocator: `deterministic-hybrid-runtime`
- diagnostics: `['association-proposal+safe-greedy-diagnostic', 'matched-fixed-association+safe-greedy-diagnostic', 'matched-fixed-association+constrained-power-oracle-diagnostic', 'association-oracle+constrained-power-oracle-diagnostic', 'association-oracle+deployable-stronger-power-allocator-diagnostic']`

## Held-Out Gate

- noncollapsed candidates: `['held-out:active-set-load-spread', 'held-out:active-set-quality-spread', 'held-out:active-set-sticky-spread', 'held-out:sticky-oracle-count-local-search', 'held-out:p05-slack-aware-active-set', 'held-out:power-response-aware-load-balance', 'held-out:bounded-move-served-set', 'held-out:oracle-score-topk-active-set']`
- positive EE delta candidates: `[]`
- accepted candidates: `[]`
- rejection reasons: `{'held-out:active-set-load-spread': 'nonpositive-ee-delta-vs-fixed-deployable;p05-ratio-below-threshold;denominator-fixed;handover-burden;oracle-gap-not-meaningfully-closed', 'held-out:active-set-quality-spread': 'nonpositive-ee-delta-vs-fixed-deployable;p05-ratio-below-threshold;denominator-fixed;oracle-gap-not-meaningfully-closed', 'held-out:active-set-sticky-spread': 'nonpositive-ee-delta-vs-fixed-deployable;oracle-gap-not-meaningfully-closed', 'held-out:sticky-oracle-count-local-search': 'nonpositive-ee-delta-vs-fixed-deployable;oracle-gap-not-meaningfully-closed', 'held-out:p05-slack-aware-active-set': 'nonpositive-ee-delta-vs-fixed-deployable;oracle-gap-not-meaningfully-closed', 'held-out:power-response-aware-load-balance': 'nonpositive-ee-delta-vs-fixed-deployable;oracle-gap-not-meaningfully-closed', 'held-out:bounded-move-served-set': 'nonpositive-ee-delta-vs-fixed-deployable;oracle-gap-not-meaningfully-closed', 'held-out:oracle-score-topk-active-set': 'nonpositive-ee-delta-vs-fixed-deployable;oracle-gap-not-meaningfully-closed'}`
- majority or predeclared primary positive EE delta: `False`
- gains not concentrated in one policy: `False`
- gains not concentrated in one seed: `False`
- QoS guardrails pass for accepted: `False`
- handover burden bounded: `False`
- aggregate oracle gap closure: `None`

## Gate Flags

- held_out_bucket_exists_and_reported_separately: `True`
- offline_replay_only: `True`
- deterministic_association_proposals_only: `True`
- learned_association_disabled: `True`
- learned_hierarchical_RL_disabled: `True`
- association_training_disabled: `True`
- joint_association_power_training_disabled: `True`
- catfish_disabled: `True`
- multi_catfish_disabled: `True`
- rb_bandwidth_allocation_disabled: `True`
- old_EE_MODQN_continuation_disabled: `True`
- frozen_baseline_mutation: `False`
- matched_control_uses_same_deployable_allocator: `True`
- primary_comparison_uses_same_deployable_allocator: `True`
- primary_comparison_no_step_cap_mismatch: `True`
- same_power_codebook_and_budget: `True`
- same_effective_power_vector_feeds_numerator_denominator_audit: `True`
- oracle_diagnostic_only: `True`
- oracle_rows_excluded_from_acceptance: `True`
- candidate_does_not_use_oracle_labels_or_future_or_heldout_answers: `True`
- majority_noncollapsed_held_out_positive_EE_delta: `False`
- predeclared_primary_held_out_positive_EE_delta: `False`
- majority_or_predeclared_primary_held_out_positive_EE_delta: `False`
- majority_or_predeclared_primary_held_out_accepted: `False`
- held_out_gains_not_concentrated_in_one_policy: `False`
- held_out_gains_not_concentrated_in_one_seed: `False`
- p05_throughput_guardrail_pass_for_accepted_held_out: `False`
- served_ratio_does_not_drop_for_accepted_held_out: `False`
- outage_ratio_does_not_increase_for_accepted_held_out: `False`
- zero_budget_per_beam_inactive_power_violations: `True`
- denominator_varies_for_accepted_held_out: `False`
- active_beam_behavior_noncollapsed_for_accepted_held_out: `False`
- handover_burden_bounded_for_accepted_held_out: `False`
- candidate_closes_meaningful_oracle_gap: `False`
- ranking_separates_or_oracle_gap_reduction_clear: `False`
- scalar_reward_success_basis: `False`
- per_user_EE_credit_success_basis: `False`
- physical_energy_saving_claim: `False`
- hobs_optimizer_claim: `False`
- full_RA_EE_MODQN_claim: `False`

## Decision

- RA-EE-08 decision: `BLOCKED`
- allowed claim: Do not promote RA-EE-08 beyond a blocked or inconclusive offline replay gate.

## Forbidden Claims

- Do not call RA-EE-08 full RA-EE-MODQN.
- Do not claim learned hierarchical RL or learned association effectiveness.
- Do not claim joint association + power training.
- Do not claim old EE-MODQN effectiveness.
- Do not claim HOBS optimizer behavior.
- Do not claim Catfish, multi-Catfish, or Catfish-EE-MODQN effectiveness.
- Do not add or claim RB / bandwidth allocation.
- Do not treat per-user EE credit as system EE.
- Do not use scalar reward alone as success evidence.
- Do not use oracle rows as deployable runtime methods.
- Do not claim full paper-faithful reproduction.
- Do not claim physical energy saving.
