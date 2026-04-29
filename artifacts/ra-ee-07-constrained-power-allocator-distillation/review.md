# RA-EE-07 Constrained-Power Allocator Distillation Review

Offline fixed-association deployable power-allocator comparison only. Association proposal buckets and oracle rows are diagnostic-only. No learned association, hierarchical RL, joint association + power training, Catfish, multi-Catfish, RB / bandwidth allocation, HOBS optimizer claim, physical energy-saving claim, or frozen baseline mutation was performed.

## Protocol

- method label: `RA-EE constrained-power allocator distillation gate`
- implementation sublabel: `RA-EE-07 constrained-power allocator distillation gate`
- config: `configs/ra-ee-07-constrained-power-allocator-distillation.resolved.yaml`
- artifact namespace: `artifacts/ra-ee-07-constrained-power-allocator-distillation`
- primary control: `matched-fixed-association+safe-greedy-power-allocator`
- primary candidate: `matched-fixed-association+deployable-stronger-power-allocator`
- deployable allocator candidates: `['p05-slack-aware-trim-tail-protect-boost', 'bounded-local-search-codebook', 'finite-codebook-dp-knapsack', 'deterministic-hybrid-runtime']`
- diagnostic fixed 1W: `matched-fixed-association+fixed-1w-diagnostic`
- diagnostic constrained oracle: `matched-fixed-association+constrained-power-oracle-isolation`
- diagnostic association proposal: `association-proposal+deployable-stronger-power-allocator-diagnostic`
- diagnostic association oracle upper bound: `association-oracle+constrained-power-oracle-upper-bound`

## Held-Out Gate

- noncollapsed held-out trajectories: `['random-valid-heldout', 'spread-valid-heldout', 'load-skewed-heldout', 'mobility-shift-heldout', 'mixed-valid-heldout']`
- positive EE delta trajectories: `['random-valid-heldout', 'spread-valid-heldout', 'load-skewed-heldout', 'mobility-shift-heldout', 'mixed-valid-heldout']`
- accepted candidate trajectories: `['random-valid-heldout', 'spread-valid-heldout', 'load-skewed-heldout', 'mobility-shift-heldout', 'mixed-valid-heldout']`
- rejection reasons: `{'random-valid-heldout': 'accepted', 'spread-valid-heldout': 'accepted', 'load-skewed-heldout': 'accepted', 'mobility-shift-heldout': 'accepted', 'mixed-valid-heldout': 'accepted'}`
- aggregate oracle gap closure: `1.0`
- gains not concentrated in one trajectory: `True`
- gains not concentrated in one seed: `True`
- QoS guardrails pass for accepted: `True`
- zero budget / per-beam / inactive-power violations: `True`
- ranking separates or oracle gap reduction clear: `True`

## Gate Flags

- held_out_bucket_exists_and_reported_separately: `True`
- offline_replay_only: `True`
- fixed_association_primary_only: `True`
- diagnostic_association_buckets_reported_separately: `True`
- deployable_non_oracle_power_allocator_comparison_only: `True`
- learned_association_disabled: `True`
- learned_hierarchical_RL_disabled: `True`
- joint_association_power_training_disabled: `True`
- catfish_disabled: `True`
- multi_catfish_disabled: `True`
- rb_bandwidth_allocation_disabled: `True`
- old_EE_MODQN_continuation_disabled: `True`
- frozen_baseline_mutation: `False`
- oracle_diagnostic_only: `True`
- candidate_does_not_use_oracle_labels_or_future_or_heldout_answers: `True`
- same_effective_power_vector_feeds_numerator_denominator_audit: `True`
- majority_noncollapsed_held_out_positive_EE_delta: `True`
- majority_noncollapsed_held_out_accepted: `True`
- held_out_gains_not_concentrated_in_one_trajectory: `True`
- held_out_gains_not_concentrated_in_one_seed: `True`
- candidate_closes_meaningful_oracle_gap: `True`
- p05_throughput_guardrail_pass_for_accepted_held_out: `True`
- served_ratio_does_not_drop_for_accepted_held_out: `True`
- outage_ratio_does_not_increase_for_accepted_held_out: `True`
- zero_budget_per_beam_inactive_power_violations: `True`
- denominator_varies_for_accepted_held_out: `True`
- selected_profiles_not_single_point_for_accepted_held_out: `True`
- total_active_power_not_single_point_for_accepted_held_out: `True`
- ranking_separates_or_oracle_gap_reduction_clear: `True`
- scalar_reward_success_basis: `False`
- per_user_EE_credit_success_basis: `False`
- physical_energy_saving_claim: `False`
- hobs_optimizer_claim: `False`
- full_RA_EE_MODQN_claim: `False`

## Decision

- RA-EE-07 decision: `PASS`
- allowed claim: PASS only means a deployable non-oracle power allocator beat the matched fixed-association safe-greedy allocator on the RA-EE-07 offline held-out gate. It is not learned association or full RA-EE-MODQN.

## Forbidden Claims

- Do not call RA-EE-07 full RA-EE-MODQN.
- Do not claim learned association or hierarchical RL effectiveness.
- Do not claim joint association + power training.
- Do not claim old EE-MODQN effectiveness.
- Do not claim HOBS optimizer behavior.
- Do not claim Catfish, multi-Catfish, or Catfish-EE-MODQN effectiveness.
- Do not add or claim RB / bandwidth allocation.
- Do not treat per-user EE credit as system EE.
- Do not use scalar reward alone as success evidence.
- Do not claim full paper-faithful reproduction.
- Do not claim physical energy saving.
