# Phase 03C-C Execution Report: Bounded Paired Power-MDP Pilot

**Date:** `2026-04-28`
**Status:** `BLOCKED`
**Scope:** bounded paired pilot only. No long training, Catfish, multi-Catfish,
frozen baseline mutation, HOBS optimizer claim, or scalar-reward-only success
claim was performed.

## Design

Phase 03C-C tests whether the Phase 03C-B static power-MDP audit survives a
runtime/eval paired pilot.

Matched configs:

1. control:
   `configs/ee-modqn-phase-03c-c-power-mdp-control.resolved.yaml`
2. candidate:
   `configs/ee-modqn-phase-03c-c-power-mdp-candidate.resolved.yaml`

Matched artifacts:

1. control:
   `artifacts/ee-modqn-phase-03c-c-power-mdp-control-pilot/`
2. candidate:
   `artifacts/ee-modqn-phase-03c-c-power-mdp-candidate-pilot/`
3. comparison:
   `artifacts/ee-modqn-phase-03c-c-power-mdp-candidate-pilot/paired-comparison-vs-control/`

The control uses the same handover scaffold, seeds, evaluation seeds, budget,
checkpoint rule, and Phase 03C codebook surface, but fixes the power decision
to the non-adaptive `fixed-mid` profile.

The candidate uses the explicit runtime `runtime-ee-selector`, which maps
post-handover active-beam load state to a concrete codebook profile. This is a
new-extension / HOBS-inspired controller surface. It is not a HOBS optimizer.

Both runs used the bounded `20`-episode budget and the inherited
best-weighted-reward-on-eval checkpoint rule. No Catfish path was used.

## Implementation

New / changed surfaces:

1. `PowerSurfaceConfig` now supports the opt-in profile
   `runtime-ee-selector` for the existing `phase-03c-b-power-codebook` mode.
2. `StepResult` now exposes runtime power-decision diagnostics:
   - `selected_power_profile`
   - `total_active_beam_power_w`
   - `power_budget_violation`
   - `power_budget_excess_w`
3. `scripts/compare_phase03c_c_power_mdp.py` exports the paired comparison.
4. `analysis.phase03c_c_power_mdp_pilot` exports:
   - `phase03c_c_power_mdp_summary.json`
   - `phase03c_c_power_mdp_step_metrics.csv`
   - `phase03c_c_power_mdp_episode_metrics.csv`
   - `phase03c_c_power_mdp_checkpoint_summary.csv`
   - `review.md`

Frozen baseline defaults remain unchanged: baseline configs still use
`static-config`, raw throughput `r1`, and baseline objective weights.

## Control vs Candidate Protocol

| Surface | Control | Candidate |
|---|---|---|
| Phase | `phase-03c-c` | `phase-03c-c` |
| Training budget | `20` episodes | `20` episodes |
| Eval seeds | `[100, 200, 300, 400, 500]` | `[100, 200, 300, 400, 500]` |
| Budget | `8.0 W` | `8.0 W` |
| Power profile | `fixed-mid` | `runtime-ee-selector` |
| Adaptive power decision | disabled | enabled |
| r1 | throughput | per-user beam-power EE credit |
| r2 / r3 | unchanged / calibrated as Phase 03B | unchanged / calibrated as Phase 03B |
| Catfish | disabled | disabled |

## Results

Primary comparison used `best-eval`; final and best-eval were identical on this
bounded run.

| Metric | Control | Candidate | Delta |
|---|---:|---:|---:|
| `EE_system` aggregate | `873.0939975452424` | `873.0942677307129` | `0.000270185470526485` |
| `EE_system` step mean | `873.0939975452424` | `873.0942677307129` | `0.000270185470526485` |
| throughput mean | `8.730939975452424` | `4.365471338653564` | `-4.36546863679886` |
| throughput p05 | `6.782802700996399` | `3.3914018511772155` | `-3.3914008498191834` |
| served ratio | `1.0` | `1.0` | `0.0` |
| outage ratio | `0.0` | `0.0` | `0.0` |
| handover count | `917` | `423` | `-494` |
| `r2` mean | `-0.0917` | `-0.0423` | `0.0494` |
| `r3` mean | `-8.730939975452424` | `-4.365471338653564` | `4.36546863679886` |
| scalar reward mean | `-35.10715990180982` | `-8.815542677307121` | `26.2916172245027` |

The candidate's apparent `EE_system` gain is not acceptable evidence: it is
numerically tiny and coincides with a `50%` low-p05 throughput collapse.

## Denominator Variability Result

Phase 03C-C fails the denominator gate:

```text
candidate_denominator_varies_in_eval = false
candidate_selected_profile_distinct_count = 1
candidate_selected_power_profile_distribution = {fixed-low: 50}
candidate_one_active_beam_step_ratio = 1.0
candidate_total_active_beam_power_w_distribution = {0.5 W: 50}
```

The runtime selector was available, but the evaluated candidate policy
collapsed every evaluated step to one active beam. The selector therefore chose
`fixed-low` on every step, and active power remained a single-point `0.5 W`
distribution.

## Ranking Separation Result

Phase 03C-C also fails ranking separation:

```text
throughput_rescore_winner = candidate
EE_rescore_winner = candidate
same_policy_throughput_rescore_vs_EE_rescore_ranking_changes = false
candidate_throughput_EE_pearson = 1.0
candidate_throughput_EE_spearman = 0.9999999999999999
```

The comparison still behaves like throughput/EE rescaling on the evaluated
policy surface.

## Acceptance Gate

| Gate | Result |
|---|---|
| denominator varies in eval | `false` |
| selected profile not single-point | `false` |
| active power not single-point | `false` |
| `EE_system` improves over control | `true`, but tiny |
| low-p05 throughput guardrail | `false` |
| served ratio guardrail | `true` |
| handover guardrail | `true` |
| not all one active beam | `false` |
| ranking separates or rescore changes | `false` |
| no scalar-reward-only success claim | `true` |

Stop conditions triggered:

1. denominator remains fixed in eval,
2. selected profile collapses to one choice,
3. evaluated policy remains all one active beam,
4. apparent EE gain comes with throughput-tail collapse.

## Validation

Focused tests:

```text
.venv/bin/python -m pytest tests/test_phase03c_c_power_mdp_pilot.py -q
```

Result:

```text
5 passed
```

Related guardrail tests:

```text
.venv/bin/python -m pytest \
  tests/test_phase03c_b_power_mdp_audit.py \
  tests/test_phase03_ee_modqn.py -q
```

Result:

```text
9 passed
```

Run commands:

```text
.venv/bin/python scripts/train_modqn.py \
  --config configs/ee-modqn-phase-03c-c-power-mdp-control.resolved.yaml \
  --progress-every 10 \
  --output-dir artifacts/ee-modqn-phase-03c-c-power-mdp-control-pilot

.venv/bin/python scripts/train_modqn.py \
  --config configs/ee-modqn-phase-03c-c-power-mdp-candidate.resolved.yaml \
  --progress-every 10 \
  --output-dir artifacts/ee-modqn-phase-03c-c-power-mdp-candidate-pilot

.venv/bin/python scripts/compare_phase03c_c_power_mdp.py \
  --control-run-dir artifacts/ee-modqn-phase-03c-c-power-mdp-control-pilot \
  --candidate-run-dir artifacts/ee-modqn-phase-03c-c-power-mdp-candidate-pilot \
  --output-dir artifacts/ee-modqn-phase-03c-c-power-mdp-candidate-pilot/paired-comparison-vs-control
```

Comparison result:

```text
decision=BLOCKED
primary=best-eval
```

## Phase 03C-C Decision

`BLOCKED`.

Do not promote EE-MODQN effectiveness from Phase 03C-C. The static/counterfactual
precondition from Phase 03C-B did not survive the runtime/eval bounded pilot.

## Remaining Blockers

1. Evaluated policy still collapses to one active beam.
2. Runtime selector collapses to one selected profile.
3. Active power remains fixed in eval.
4. Throughput-vs-EE ranking remains effectively identical.
5. Apparent EE gain is tiny and paired with p05 throughput collapse.
6. The controller remains a new-extension / HOBS-inspired surface, not a HOBS
   optimizer.

## Forbidden Claims Still Active

1. Do not claim EE-MODQN effectiveness.
2. Do not claim HOBS optimizer behavior.
3. Do not claim full paper-faithful reproduction.
4. Do not claim Catfish, multi-Catfish, or final Catfish-EE-MODQN.
5. Do not treat per-user EE credit as system EE.
6. Do not use scalar reward alone as success evidence.
