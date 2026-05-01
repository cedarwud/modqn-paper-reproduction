# HOBS Active-TX EE Non-Codebook Continuous-Power Bounded Pilot Execution Report

**Date:** `2026-05-02`
**Status:** `BLOCK`
**Method label:** `CP-base-EE-MODQN bounded matched pilot`
**Scope:** bounded matched pilot only; not Catfish-EE, not physical energy saving, not HOBS optimizer reproduction, not full RA-EE-MODQN, and not a general EE-MODQN effectiveness claim.

## Changed Files

```text
configs/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot-throughput-control.resolved.yaml
configs/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot-ee-candidate.resolved.yaml
scripts/run_hobs_active_tx_ee_non_codebook_continuous_power_bounded_pilot.py
src/modqn_paper_reproduction/analysis/hobs_active_tx_ee_non_codebook_continuous_power_bounded_pilot.py
src/modqn_paper_reproduction/config_loader.py
src/modqn_paper_reproduction/runtime/trainer_spec.py
src/modqn_paper_reproduction/analysis/hobs_active_tx_ee_anti_collapse.py
tests/test_hobs_active_tx_ee_non_codebook_continuous_power_bounded_pilot.py
docs/research/catfish-ee-modqn/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot.execution-report.md
```

## What Was Implemented

- Added the bounded-pilot config namespace and two matched resolved configs.
- Added a bounded-pilot runner that executes exactly the authorized two roles and three seed triplets.
- Added matched-boundary proof, paired comparison, summary CSV/JSON, and execution-report emission.
- Extended the CP-base continuous-power diagnostics with profile absence, continuous-power activity, and Spearman correlation fields.
- Added focused tests for config loading, matched metadata, acceptance, and scalar-only BLOCK enforcement.

## Protocol / Roles

- Roles: `throughput-control`, `ee-candidate`.
- Seed triplets: `[42, 1337, 7]`, `[43, 1338, 8]`, `[44, 1339, 9]`.
- Eval seeds: `[100, 200, 300, 400, 500]`.
- Episode budget: `5` per role / seed triplet.
- Scalar reward is diagnostic only.

## Matched Boundary Proof

`matched_boundary_pass=True`

```text
required_roles_present = True
same_training_experiment_kind = True
same_phase = True
same_episode_budget = True
tiny_episode_budget = True
same_eval_seeds = True
same_seed_triplets = True
exact_required_seed_triplets = True
exact_required_eval_seeds = True
same_objective_weights = True
same_trainer_hyperparameters = True
same_checkpoint_protocol = True
same_environment_boundary = True
same_continuous_power_surface = True
continuous_power_mode = True
same_qos_sticky_guard = True
same_nonsticky_handover_protections = True
throughput_control_r1 = True
ee_candidate_r1 = True
only_intended_difference_is_r1_reward_mode = True
finite_codebook_levels_absent = True
selected_power_profile_absent = True
forbidden_modes_disabled = True
```

## Metrics

| Role | EE_system | EE step mean | throughput | p05 | served | outage | handovers | r2 | r3 | one-beam | denom varies | selected profile absent |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `throughput-control` | `873.9817347495251` | `873.9684693879441` | `1033.2210319805145` | `2.883108186721802` | `1.0` | `0.0` | `221.0` | `-0.221` | `-8.8861994247437` | `False` | `True` | `True` |
| `ee-candidate` | `872.9239944442857` | `872.9155535916389` | `1021.5656559681893` | `2.789693629741669` | `1.0` | `0.0` | `298.3333333333333` | `-0.29833333333333334` | `-9.243529318730028` | `False` | `True` | `True` |

Aggregate candidate vs throughput-control:

```text
candidate_vs_control_EE_system_delta = -1.057740305239463
p05_throughput_ratio_vs_control = 0.9675993577312307
served_ratio_delta_vs_control = 0.0
outage_ratio_delta_vs_control = 0.0
handover_delta_vs_control = 77.33333333333331
r2_delta_vs_control = -0.07733333333333334
scalar_reward_diagnostic_delta_vs_control = 4312.822050380359
```

Per-seed EE deltas:

```text
[42, 1337, 7]: EE_delta=-1.1009355380937222, p05_ratio=0.928688683730214, handover_delta=-17.0, r2_delta=0.016999999999999987
[43, 1338, 8]: EE_delta=-1.1279326781493637, p05_ratio=1.0013261735160452, handover_delta=250.0, r2_delta=-0.24999999999999997
[44, 1339, 9]: EE_delta=-0.9443526994751892, p05_ratio=0.9763654144256759, handover_delta=-1.0, r2_delta=0.0009999999999999731
```

Per-seed role metrics:

| Role | Seed triplet | EE_system | throughput | p05 | served | outage | handovers | r2 | r3 | scalar diagnostic | denom varies | profile absent | power violations |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `throughput-control` | `[42, 1337, 7]` | `873.5019484003252` | `1046.6147098636627` | `3.054789733886719` | `1.0` | `0.0` | `229` | `-0.229` | `-8.087506837844865` | `50.64453412561416` | `True` | `True` | `(0, 0, 0)` |
| `throughput-control` | `[43, 1338, 8]` | `874.221627924125` | `1026.5241930389404` | `2.7972674131393434` | `1.0` | `0.0` | `217` | `-0.217` | `-9.285545718193116` | `49.40400050830839` | `True` | `True` | `(0, 0, 0)` |
| `throughput-control` | `[44, 1339, 9]` | `874.221627924125` | `1026.5241930389404` | `2.7972674131393434` | `1.0` | `0.0` | `217` | `-0.217` | `-9.285545718193116` | `49.40400050830839` | `True` | `True` | `(0, 0, 0)` |
| `ee-candidate` | `[42, 1337, 7]` | `872.4010128622315` | `1020.4657095479965` | `2.8369486570358275` | `1.0` | `0.0` | `212` | `-0.21200000000000002` | `-9.360568029880563` | `4360.017514575871` | `True` | `True` | `(0, 0, 0)` |
| `ee-candidate` | `[43, 1338, 8]` | `873.0936952459756` | `1029.3809158182144` | `2.800977075099945` | `1.0` | `0.0` | `467` | `-0.46699999999999997` | `-8.971151807308157` | `4363.536060723395` | `True` | `True` | `(0, 0, 0)` |
| `ee-candidate` | `[44, 1339, 9]` | `873.2772752246498` | `1014.8503425383568` | `2.7311551570892334` | `1.0` | `0.0` | `216` | `-0.21600000000000003` | `-9.39886811900136` | `4364.365110984041` | `True` | `True` | `(0, 0, 0)` |

Power and correlation diagnostics:

```text
throughput-control.active_beam_count_distribution = {'7.0': 150}
throughput-control.selected_power_profile_distribution = {'': 150}
throughput-control.distinct_total_active_power_w_value_count = 57
throughput-control.distinct_active_power_w_value_count = 56
throughput-control.power_control_activity_rate = 0.9851851851851853
throughput-control.continuous_power_activity_rate = 0.9851851851851853
throughput-control.budget_violation_count = 0
throughput-control.per_beam_power_violation_count = 0
throughput-control.inactive_beam_nonzero_power_step_count = 0
throughput-control.throughput_vs_ee_pearson = 0.8680945575163926
throughput-control.throughput_vs_ee_spearman = 0.834141656662665
throughput-control.same_policy_throughput_vs_ee_rescore_ranking_change = True

ee-candidate.active_beam_count_distribution = {'7.0': 150}
ee-candidate.selected_power_profile_distribution = {'': 150}
ee-candidate.distinct_total_active_power_w_value_count = 82
ee-candidate.distinct_active_power_w_value_count = 61
ee-candidate.power_control_activity_rate = 0.9777777777777779
ee-candidate.continuous_power_activity_rate = 0.9777777777777779
ee-candidate.budget_violation_count = 0
ee-candidate.per_beam_power_violation_count = 0
ee-candidate.inactive_beam_nonzero_power_step_count = 0
ee-candidate.throughput_vs_ee_pearson = 0.8260922940633209
ee-candidate.throughput_vs_ee_spearman = 0.8216566626650659
ee-candidate.same_policy_throughput_vs_ee_rescore_ranking_change = True

Full active-power value lists and per-seed distributions are in artifacts/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot-summary/summary.json.
```

## Acceptance Result

`PASS / BLOCK / NEEDS MORE DESIGN: BLOCK`

```text
matched_boundary_pass = True
candidate_all_evaluated_steps_one_active_beam = False
candidate_denominator_varies_in_eval = True
candidate_active_power_single_point_distribution = False
candidate_selected_power_profile_absent = True
candidate_vs_throughput_same_guard_same_power_control_EE_system_delta = -1.057740305239463
candidate_vs_throughput_same_guard_same_power_control_p05_ratio = 0.9675993577312307
candidate_vs_throughput_same_guard_same_power_control_served_ratio_delta = 0.0
candidate_vs_throughput_same_guard_same_power_control_outage_ratio_delta = 0.0
candidate_vs_throughput_same_guard_same_power_control_handover_delta = 77.33333333333331
candidate_vs_throughput_same_guard_same_power_control_r2_delta = -0.07733333333333334
budget_per_beam_inactive_power_violations = [0, 0, 0]
scalar_reward_success_basis = True
positive_EE_not_concentrated_in_single_seed_triplet = False
```

## Stop Conditions Triggered

- candidate loses EE_system to throughput same-guard same-power control
- candidate wins only scalar reward
- candidate violates protected guardrails

## Tests / Checks Run

```text
.venv/bin/python -m pytest tests/test_hobs_active_tx_ee_non_codebook_continuous_power_bounded_pilot.py tests/test_hobs_active_tx_ee_non_codebook_continuous_power.py -q
.venv/bin/python scripts/run_hobs_active_tx_ee_non_codebook_continuous_power_bounded_pilot.py
git diff --check
```

## Artifacts

- `artifacts/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot-throughput-control/`
- `artifacts/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot-ee-candidate/`
- `artifacts/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot-ee-candidate/paired-comparison-vs-throughput-control/`
- `artifacts/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot-summary/`

## Forbidden Claims Still Active

- general EE-MODQN effectiveness
- Catfish-EE readiness
- Catfish / Multi-Catfish effectiveness
- physical energy saving
- HOBS optimizer reproduction
- full RA-EE-MODQN
- learned association effectiveness
- RB / bandwidth allocation effectiveness
- Phase 03D failure is overturned
- Phase 03C selector route is reopened
- scalar reward success
- QoS-sticky anti-collapse as EE objective contribution
- denominator variability alone proves energy-aware learning
- same-throughput-less-physical-power

## Deviations / Blockers

- candidate loses EE_system to throughput same-guard same-power control
- candidate handover delta exceeds +25
- candidate r2 delta below -0.05
- candidate wins only scalar reward

## PASS / BLOCK / NEEDS MORE DESIGN

`BLOCK`
