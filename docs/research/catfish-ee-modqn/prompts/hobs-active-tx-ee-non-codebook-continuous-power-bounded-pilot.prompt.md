# Prompt: HOBS Active-TX EE Non-Codebook Continuous-Power Bounded Pilot

你是 `modqn-paper-reproduction` 的 execution worker。請只執行本 prompt
授權的 **CP-base bounded matched pilot**。不要重新規劃整條研究線。

這不是 Catfish-EE，不是 full EE-MODQN effectiveness claim，也不是長訓練。
如果任何 matched-boundary 或 stop condition 失敗，直接回報 `BLOCK`，不要自行
調參續跑。

## Read First

請先讀：

```text
AGENTS.md
docs/research/catfish-ee-modqn/00-validation-master-plan.md
docs/research/catfish-ee-modqn/execution-handoff.md
docs/research/catfish-ee-modqn/development-guardrails.md
docs/research/catfish-ee-modqn/ee-modqn-anti-collapse-controller-plan-2026-05-01.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-modqn-feasibility.execution-report.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-anti-collapse-design-gate.execution-report.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-qos-sticky-anti-collapse-design-gate.execution-report.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-qos-sticky-robustness-gate.execution-report.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-qos-sticky-broader-effectiveness-gate.execution-report.md
docs/research/catfish-ee-modqn/03c-b-power-mdp-audit.execution-report.md
docs/research/catfish-ee-modqn/03c-c-power-mdp-pilot.execution-report.md
docs/research/catfish-ee-modqn/03d-ee-route-disposition.execution-report.md
docs/research/catfish-ee-modqn/ra-ee-07-constrained-power-allocator-distillation.execution-report.md
docs/research/catfish-ee-modqn/ra-ee-08-offline-association-reevaluation.execution-report.md
docs/research/catfish-ee-modqn/ra-ee-09-fixed-association-rb-bandwidth.execution-report.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-non-codebook-continuous-power-design-gate.execution-report.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-non-codebook-continuous-power-implementation-readiness.execution-report.md
docs/research/catfish-ee-modqn/energy-efficient/README.md
docs/research/catfish-ee-modqn/energy-efficient/ee-formula-final-review-with-codex-2026-05-01.md
docs/ee-report.md
```

## Current Boundary

沿用以下結論：

```text
current QoS-sticky EE objective route: BLOCK / stop-loss
CP-base design gate: PASS, design-only
CP-base implementation-readiness: PASS, readiness-only
bounded pilot: AUTHORIZED BY THIS PROMPT ONLY
Catfish-EE: BLOCKED until base EE method beats matched controls
```

## Goal

執行一個最小、可稽核、matched 的 CP-base pilot，回答：

```text
在同一個 non-codebook continuous p_b(t) rollout surface、同一個 QoS-sticky
structural guard、同一組 seeds / budget / eval protocol 下，
r1 = hobs-active-tx-ee 是否能 beat r1 = throughput？
```

這個 pilot 只比較 base EE objective contribution。它不能被解讀為 Catfish-EE、
physical energy saving、HOBS optimizer、full RA-EE-MODQN、或 general
EE-MODQN effectiveness。

## Required Method / Namespace

Method label:

```text
CP-base-EE-MODQN bounded matched pilot
```

Use new config and artifact namespaces:

```text
configs/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot-throughput-control.resolved.yaml
configs/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot-ee-candidate.resolved.yaml

artifacts/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot-throughput-control/
artifacts/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot-ee-candidate/
artifacts/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot-ee-candidate/paired-comparison-vs-throughput-control/
artifacts/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot-summary/

docs/research/catfish-ee-modqn/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot.execution-report.md
```

Do not write pilot artifacts into frozen baseline, Phase `03C`, QoS-sticky prior
gate, RA-EE, Catfish, or implementation-readiness artifact directories.

## Required Roles

Primary decisive roles:

```text
throughput-control:
  r1_reward_mode = throughput
  same non-codebook continuous power surface
  same QoS-sticky overflow structural guard

ee-candidate:
  r1_reward_mode = hobs-active-tx-ee
  same non-codebook continuous power surface
  same QoS-sticky overflow structural guard
```

The only intended candidate/control difference is:

```text
r1_reward_mode
```

Both roles must share:

```text
same continuous p_b(t) formula and constants
same QoS-sticky anti-collapse guard
same non-sticky / handover protections
same seeds / seed triplets
same episode budget
same eval seeds
same eval schedule
same checkpoint protocol
same objective weights except r1 mode
same trainer hyperparameters
same artifact / diagnostic schema
```

Do not add no-anti-collapse, Catfish, RA-EE, Phase `03C`, oracle, or profile
selector roles to the primary acceptance gate. If you add a diagnostic-only
boundary audit, keep it separate from acceptance.

## Bounded Protocol

Use the readiness seed protocol for acceptance. If a blocking implementation
detail requires a smaller smoke first, keep that smoke diagnostic-only and
separate from acceptance; it cannot replace the protocol below:

```text
roles: throughput-control, ee-candidate
seed triplets:
  [42, 1337, 7]
  [43, 1338, 8]
  [44, 1339, 9]
eval seeds: [100, 200, 300, 400, 500]
episode budget: 5 per role / seed triplet
expected required runs: 2 roles x 3 seed triplets = 6 runs
```

Use the existing best-eval / checkpoint protocol unless the config explicitly
requires the same already-authorized bounded checkpoint protocol for both roles.

No long runs. Do not increase episode budget, add seeds, tune constants, or run
a second pilot after seeing results.

## Allowed Implementation Work

Allowed:

1. Add pilot-specific resolved configs under the required namespace.
2. Extend config namespace gating only enough to permit this bounded pilot kind.
3. Add a pilot runner and paired-comparison / summary helper if needed.
4. Add focused tests for pilot config boundary, matched role metadata, and
   no-forbidden-mode enforcement.
5. Write the bounded pilot execution report.

Not allowed:

1. Do not mutate frozen baseline configs / artifacts / semantics.
2. Do not change the CP-base power formula or constants after seeing pilot
   results.
3. Do not add Catfish / Multi-Catfish.
4. Do not reopen Phase `03C` finite-codebook selector.
5. Do not reopen RA-EE learned association or proposal replay.
6. Do not introduce oracle, future information, offline replay oracle, HOBS
   optimizer behavior, or post-hoc EE rescore.
7. Do not use scalar reward as success evidence.

## Required Boundary Proof

Before interpreting any metrics, prove:

```text
matched_boundary_pass = true
only_intended_difference_is_r1_reward_mode = true
same_continuous_power_surface = true
same_qos_sticky_guard = true
same_seed_triplets = true
same_eval_seeds = true
same_episode_budget = true
same_checkpoint_protocol = true
same_trainer_hyperparameters = true
finite_codebook_levels_absent = true
selected_power_profile_absent = true
forbidden_modes_disabled = true
```

If this proof fails, stop and report `BLOCK`.

## Required Metrics

Report aggregate and per-seed-triplet metrics for both roles:

```text
EE_system aggregate / step mean
candidate_vs_control_EE_system_delta
raw throughput mean
p05 throughput
p05 throughput ratio vs control
served ratio and served ratio delta
outage ratio and outage ratio delta
handover count and handover delta
r2 mean and r2 delta
load-balance metric / r3
scalar reward diagnostic
all_evaluated_steps_one_active_beam
active_beam_count_distribution
denominator_varies_in_eval
active_power_single_point_distribution
distinct_total_active_power_w_values
selected_power_profile distribution
power-control / continuous-power activity diagnostics
budget violation count
per-beam power violation count
inactive-beam nonzero-power count
throughput-vs-EE Pearson / Spearman
same-policy throughput-vs-EE rescore ranking change, if available
```

Scalar reward is diagnostic only and must not decide acceptance.

## Acceptance Criteria

The pilot can pass only if all are true:

```text
matched_boundary_pass = true
candidate_all_evaluated_steps_one_active_beam = false
candidate_denominator_varies_in_eval = true
candidate_active_power_single_point_distribution = false
candidate_selected_power_profile_absent = true
candidate_vs_throughput_same_guard_same_power_control_EE_system_delta > 0
candidate_vs_throughput_same_guard_same_power_control_p05_ratio >= 0.95
candidate_vs_throughput_same_guard_same_power_control_served_ratio_delta >= 0
candidate_vs_throughput_same_guard_same_power_control_outage_ratio_delta <= 0
candidate_vs_throughput_same_guard_same_power_control_handover_delta <= +25
candidate_vs_throughput_same_guard_same_power_control_r2_delta >= -0.05
budget/per-beam/inactive-power violations = 0
scalar_reward_success_basis = false
```

Also require the positive EE result not to be concentrated in a single seed
triplet. If aggregate is positive but most seed triplets are negative, report
`NEEDS MORE DESIGN` or `BLOCK` rather than promoting.

## Stop Conditions

Stop and report `BLOCK` if:

1. Candidate/control differ by anything other than `r1_reward_mode`.
2. Throughput + same guard + same continuous power control is missing.
3. Continuous power is not shared between roles.
4. Power becomes constant, codebook/profile-selected, rounded, or post-hoc.
5. Candidate still has all evaluated steps with one active beam.
6. Candidate denominator does not vary in eval.
7. Candidate loses `EE_system` to throughput + same guard + same power control.
8. Candidate wins only scalar reward.
9. Candidate violates p05, served/outage, handover, `r2`, or power guardrails.
10. Any forbidden mode is enabled.

If a stop condition triggers, do not tune and rerun.

## Forbidden Claims

Do not claim:

```text
general EE-MODQN effectiveness
Catfish-EE readiness
Catfish / Multi-Catfish effectiveness
physical energy saving
HOBS optimizer reproduction
full RA-EE-MODQN
learned association effectiveness
RB / bandwidth allocation effectiveness
Phase 03D failure is overturned
Phase 03C selector route is reopened
scalar reward success
QoS-sticky anti-collapse as EE objective contribution
denominator variability alone proves energy-aware learning
same-throughput-less-physical-power
```

Allowed claim if the pilot passes:

```text
Under the bounded CP-base matched pilot protocol, the hobs-active-tx-ee
objective candidate beat the throughput objective control while sharing the same
non-codebook continuous active-TX power surface and the same structural
anti-collapse guard, with protected QoS / handover / r2 / power guardrails
preserved.
```

Even if this passes, do not start Catfish-EE. Return to the controller for
claim-boundary update and next-gate authorization.

## Output Format

請輸出：

```text
Changed Files
What Was Implemented
Protocol / Roles
Matched Boundary Proof
Metrics
Acceptance Result
Tests / Checks Run
Artifacts
Forbidden Claims Still Active
Deviations / Blockers
PASS / BLOCK / NEEDS MORE DESIGN
```

If `PASS`, do not launch any follow-up pilot or Catfish work. Return to the
controller.
