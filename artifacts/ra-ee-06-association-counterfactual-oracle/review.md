# RA-EE-06 Association Counterfactual / Oracle Review

Offline active-set / served-set association counterfactual gate only. No learned association, joint association + power training, Catfish, multi-Catfish, RB / bandwidth allocation, HOBS optimizer claim, physical energy-saving claim, or frozen baseline mutation was performed.

## Protocol

- method label: `RA-EE hierarchical association + power counterfactual`
- implementation sublabel: `RA-EE-06 association counterfactual / oracle design gate`
- config: `configs/ra-ee-06-association-counterfactual-oracle.resolved.yaml`
- artifact namespace: `artifacts/ra-ee-06-association-counterfactual-oracle`
- matched control: `matched-fixed-association+safe-greedy-power-allocator`
- candidate: `association-proposal+safe-greedy-power-allocator`
- oracle: `association-oracle+constrained-power-upper-bound`
- candidate association policies: `['active-set-load-spread', 'active-set-quality-spread', 'active-set-sticky-spread']`
- diagnostic association policies: `['per-user-greedy-best-beam']`

## Held-Out Gate

- noncollapsed candidates: `['held-out:active-set-load-spread', 'held-out:active-set-quality-spread', 'held-out:active-set-sticky-spread']`
- positive EE delta candidates: `[]`
- accepted candidates: `[]`
- majority noncollapsed positive EE delta: `False`
- majority noncollapsed accepted: `False`
- gains not concentrated in one policy: `False`
- QoS guardrails pass for accepted: `False`
- zero budget / per-beam / inactive-power violations: `True`
- denominator varies for accepted: `False`

## Gate Flags

- held_out_bucket_exists_and_reported_separately: `True`
- association_counterfactual_only: `True`
- learned_association_disabled: `True`
- joint_association_power_training_disabled: `True`
- catfish_disabled: `True`
- multi_catfish_disabled: `True`
- rb_bandwidth_allocation_disabled: `True`
- old_EE_MODQN_continuation_disabled: `True`
- frozen_baseline_mutation: `False`
- active_set_served_set_proposal_contract: `True`
- matched_control_uses_same_power_allocator: `True`
- safe_greedy_allocator_retained: `True`
- constrained_oracle_upper_bound_diagnostic_only: `True`
- majority_noncollapsed_held_out_positive_EE_delta: `False`
- majority_noncollapsed_held_out_accepted: `False`
- held_out_gains_not_concentrated_in_one_policy: `False`
- p05_throughput_guardrail_pass_for_accepted_held_out: `False`
- served_ratio_does_not_drop_for_accepted_held_out: `False`
- outage_ratio_does_not_increase_for_accepted_held_out: `False`
- zero_budget_per_beam_inactive_power_violations: `True`
- denominator_varies_for_accepted_held_out: `False`
- one_active_beam_collapse_avoided_for_accepted_held_out: `True`
- scalar_reward_success_basis: `False`
- per_user_EE_credit_success_basis: `False`
- physical_energy_saving_claim: `False`
- hobs_optimizer_claim: `False`
- full_RA_EE_MODQN_claim: `False`

## Decision

- RA-EE-06 decision: `BLOCKED`
- allowed claim: Do not proceed to learned hierarchical RA-EE training without resolving blockers.

## Forbidden Claims

- Do not call RA-EE-06 full RA-EE-MODQN.
- Do not claim learned association effectiveness.
- Do not claim joint association + power training.
- Do not claim old EE-MODQN effectiveness.
- Do not claim HOBS optimizer behavior.
- Do not claim Catfish, multi-Catfish, or Catfish-EE-MODQN effectiveness.
- Do not add or claim RB / bandwidth allocation.
- Do not treat per-user EE credit as system EE.
- Do not use scalar reward alone as success evidence.
- Do not claim full paper-faithful reproduction.
- Do not claim physical energy saving.
