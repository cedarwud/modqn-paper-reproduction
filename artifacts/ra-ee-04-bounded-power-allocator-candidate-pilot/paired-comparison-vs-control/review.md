# RA-EE-04 Bounded Power-Allocator Pilot Review

Fixed-association centralized power-allocation pilot only. No learned association, Catfish, multi-Catfish, old EE-MODQN continuation, long training, HOBS optimizer claim, physical energy-saving claim, or frozen baseline mutation was performed.

## Protocol

- method label: `RA-EE-MDP`
- implementation sublabel: `RA-EE-04 fixed-association power-allocation pilot`
- training episodes: `20`
- evaluation seeds: `[100, 200, 300, 400, 500]`
- fixed trajectories: `['hold-current', 'random-valid', 'spread-valid']`
- action: `centralized per-active-beam discrete power vector; inactive beams 0 W`
- levels W: `[0.5, 0.75, 1.0, 1.5, 2.0]`
- total active budget W: `8.0`

## Gate Flags

- fixed_association_only: `True`
- learned_association_disabled: `True`
- catfish_disabled: `True`
- multi_catfish_disabled: `True`
- training_episodes_is_20: `True`
- all_primary_noncollapsed_trajectories_present: `True`
- denominator_varies_in_eval: `True`
- selected_power_vector_not_single_point: `True`
- total_active_power_not_single_point: `True`
- not_all_one_active_beam: `True`
- EE_system_improves_vs_fixed_control: `True`
- QoS_guardrails_pass: `True`
- zero_budget_per_beam_inactive_power_violations: `True`
- ranking_separates_or_rescore_changes: `True`
- scalar_reward_success_basis: `False`
- per_user_EE_credit_success_basis: `False`

## Decision

- RA-EE-04 decision: `PASS`
- allowed claim: PASS only means this bounded fixed-association centralized power-allocation pilot passed its implementation gate.

## Forbidden Claims

- Do not claim old EE-MODQN effectiveness.
- Do not claim HOBS optimizer behavior.
- Do not claim Catfish, multi-Catfish, or Catfish-EE-MODQN effectiveness.
- Do not treat per-user EE credit as system EE.
- Do not use scalar reward alone as success evidence.
- Do not claim full paper-faithful reproduction.
- Do not claim physical energy saving.
