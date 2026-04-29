# RA-EE-05 Fixed-Association Robustness Review

Fixed-association centralized power-allocation robustness and held-out validation only. No learned association, joint association + power training, Catfish, multi-Catfish, RB / bandwidth allocation, old EE-MODQN continuation, HOBS optimizer claim, physical energy-saving claim, or frozen baseline mutation was performed.

## Protocol

- method label: `RA-EE fixed-association centralized power allocator`
- implementation sublabel: `RA-EE-05 fixed-association robustness and held-out validation`
- config: `configs/ra-ee-05-fixed-association-robustness.resolved.yaml`
- artifact namespace: `artifacts/ra-ee-05-fixed-association-robustness`
- calibration seeds: `[100, 200, 300, 400, 500]`
- held-out seeds: `[600, 700, 800, 900, 1000]`
- calibration trajectories: `['hold-current', 'random-valid', 'spread-valid']`
- held-out trajectories: `['random-valid-heldout', 'spread-valid-heldout', 'load-skewed-heldout', 'mobility-shift-heldout', 'mixed-valid-heldout']`
- candidate: `safe-greedy-power-allocator`
- control: `fixed-control-1w-per-active-beam`
- oracle: `constrained-oracle-upper-bound`

## Held-Out Gate

- noncollapsed held-out trajectories: `['random-valid-heldout', 'spread-valid-heldout', 'load-skewed-heldout', 'mobility-shift-heldout', 'mixed-valid-heldout']`
- positive EE delta trajectories: `['random-valid-heldout', 'spread-valid-heldout', 'load-skewed-heldout', 'mobility-shift-heldout', 'mixed-valid-heldout']`
- accepted candidate trajectories: `['load-skewed-heldout', 'mixed-valid-heldout', 'mobility-shift-heldout', 'random-valid-heldout', 'spread-valid-heldout']`
- majority noncollapsed positive EE delta: `True`
- majority noncollapsed accepted: `True`
- gains not concentrated in one trajectory: `True`
- QoS guardrails pass for accepted: `True`
- zero budget / per-beam / inactive-power violations: `True`
- ranking separates for accepted: `True`

## Gate Flags

- held_out_bucket_exists_and_reported_separately: `True`
- fixed_association_only: `True`
- learned_association_disabled: `True`
- catfish_disabled: `True`
- multi_catfish_disabled: `True`
- joint_association_power_training_disabled: `True`
- old_EE_MODQN_continuation_disabled: `True`
- frozen_baseline_mutation: `False`
- majority_noncollapsed_held_out_positive_EE_delta: `True`
- majority_noncollapsed_held_out_accepted: `True`
- held_out_gains_not_concentrated_in_one_trajectory: `True`
- p05_throughput_guardrail_pass_for_accepted_held_out: `True`
- served_ratio_does_not_drop_for_accepted_held_out: `True`
- outage_ratio_does_not_increase_for_accepted_held_out: `True`
- zero_budget_per_beam_inactive_power_violations: `True`
- denominator_varies_for_accepted_held_out: `True`
- selected_power_vectors_not_single_point_for_accepted_held_out: `True`
- selected_profiles_not_single_point_for_accepted_held_out: `True`
- total_active_power_not_single_point_for_accepted_held_out: `True`
- throughput_winner_vs_EE_winner_separate_for_accepted_held_out: `True`
- oracle_upper_bound_diagnostic_only: `True`
- scalar_reward_success_basis: `False`
- per_user_EE_credit_success_basis: `False`
- physical_energy_saving_claim: `False`
- hobs_optimizer_claim: `False`

## Decision

- RA-EE-05 decision: `PASS`
- allowed claim: PASS only means fixed-association centralized power-allocation robustness passed the held-out gate. It is not full RA-EE-MODQN.

## Forbidden Claims

- Do not call RA-EE-05 full RA-EE-MODQN.
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
