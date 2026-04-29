# RA-EE-06B Association Proposal Refinement Review

Offline oracle-trace export and deterministic association proposal refinement only. No learned hierarchical RL, joint association + power training, Catfish, multi-Catfish, RB / bandwidth allocation, HOBS optimizer claim, physical energy-saving claim, or frozen baseline mutation was performed.

## Protocol

- method label: `RA-EE association proposal refinement / oracle distillation audit`
- implementation sublabel: `RA-EE-06B association proposal refinement / oracle distillation audit`
- config: `configs/ra-ee-06b-association-proposal-refinement.resolved.yaml`
- artifact namespace: `artifacts/ra-ee-06b-association-proposal-refinement`
- matched control: `matched-fixed-association+safe-greedy-power-allocator`
- primary candidate: `association-proposal-rule+safe-greedy-power-allocator`
- diagnostic oracle: `association-oracle+same-safe-greedy-diagnostic`
- upper bound oracle: `association-oracle+constrained-power-upper-bound`
- proposal rules: `['sticky-oracle-count-local-search', 'p05-slack-aware-active-set', 'power-response-aware-load-balance', 'bounded-move-served-set', 'oracle-score-topk-active-set']`

## Held-Out Gate

- noncollapsed candidates: `['held-out:sticky-oracle-count-local-search', 'held-out:p05-slack-aware-active-set', 'held-out:power-response-aware-load-balance', 'held-out:bounded-move-served-set', 'held-out:oracle-score-topk-active-set']`
- positive EE delta candidates: `[]`
- accepted candidates: `[]`
- rejection reasons: `{'held-out:sticky-oracle-count-local-search': 'nonpositive-ee-delta;oracle-gap-not-meaningfully-closed', 'held-out:p05-slack-aware-active-set': 'nonpositive-ee-delta;oracle-gap-not-meaningfully-closed', 'held-out:power-response-aware-load-balance': 'nonpositive-ee-delta;oracle-gap-not-meaningfully-closed', 'held-out:bounded-move-served-set': 'nonpositive-ee-delta;oracle-gap-not-meaningfully-closed', 'held-out:oracle-score-topk-active-set': 'nonpositive-ee-delta;oracle-gap-not-meaningfully-closed'}`
- majority noncollapsed positive EE delta: `False`
- majority noncollapsed accepted: `False`
- gains not concentrated in one policy: `False`
- QoS guardrails pass for accepted: `False`
- denominator varies for accepted: `False`
- handover burden bounded: `False`
- oracle gap closed: `False`

## Gate Flags

- held_out_bucket_exists_and_reported_separately: `True`
- offline_trace_export_only: `True`
- deterministic_proposal_refinement_only: `True`
- learned_hierarchical_RL_disabled: `True`
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
- oracle_diagnostic_only: `True`
- constrained_oracle_upper_bound_diagnostic_only: `True`
- majority_noncollapsed_held_out_positive_EE_delta: `False`
- majority_noncollapsed_held_out_accepted: `False`
- held_out_gains_not_concentrated_in_one_policy: `False`
- p05_throughput_guardrail_pass_for_accepted_held_out: `False`
- served_ratio_does_not_drop_for_accepted_held_out: `False`
- outage_ratio_does_not_increase_for_accepted_held_out: `False`
- zero_budget_per_beam_inactive_power_violations: `True`
- denominator_varies_for_accepted_held_out: `False`
- one_active_beam_or_two_beam_overload_collapse_avoided: `True`
- handover_burden_bounded_for_accepted_held_out: `False`
- candidate_closes_meaningful_oracle_gap: `False`
- scalar_reward_success_basis: `False`
- per_user_EE_credit_success_basis: `False`
- physical_energy_saving_claim: `False`
- hobs_optimizer_claim: `False`
- full_RA_EE_MODQN_claim: `False`

## Decision

- RA-EE-06B decision: `BLOCKED`
- allowed claim: Do not proceed to learned hierarchical RA-EE training without resolving blockers.

## Forbidden Claims

- Do not call RA-EE-06B full RA-EE-MODQN.
- Do not claim learned hierarchical RL or learned association effectiveness.
- Do not claim joint association + power training.
- Do not claim old EE-MODQN effectiveness.
- Do not claim HOBS optimizer behavior.
- Do not claim Catfish, multi-Catfish, or Catfish-EE-MODQN effectiveness.
- Do not add or claim RB / bandwidth allocation.
- Do not treat per-user EE credit as system EE.
- Do not use scalar reward alone as success evidence.
- Do not claim full paper-faithful reproduction.
- Do not claim physical energy saving.
