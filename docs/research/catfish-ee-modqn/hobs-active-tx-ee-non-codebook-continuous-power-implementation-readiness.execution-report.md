# HOBS Active-TX EE Non-Codebook Continuous-Power Implementation-Readiness Execution Report

**Date:** `2026-05-01`
**Status:** `PASS`
**Method label:** `CP-base-EE-MODQN implementation-readiness`
**Scope:** opt-in config / wiring / metadata readiness only. No pilot training,
matched pilot, Catfish / Multi-Catfish, Phase `03C` continuation, RA-EE learned
association, oracle, future-information, offline replay oracle, HOBS optimizer,
physical energy-saving, or EE-MODQN effectiveness claim was run or authorized.

## Changed Files

```text
configs/hobs-active-tx-ee-non-codebook-continuous-power-boundary-audit.resolved.yaml
configs/hobs-active-tx-ee-non-codebook-continuous-power-ee-candidate.resolved.yaml
configs/hobs-active-tx-ee-non-codebook-continuous-power-throughput-control.resolved.yaml
scripts/run_hobs_active_tx_ee_non_codebook_continuous_power_boundary_audit.py
src/modqn_paper_reproduction/analysis/hobs_active_tx_ee_non_codebook_continuous_power_boundary_audit.py
src/modqn_paper_reproduction/config_loader.py
src/modqn_paper_reproduction/env/step.py
src/modqn_paper_reproduction/runtime/trainer_spec.py
tests/test_hobs_active_tx_ee_non_codebook_continuous_power.py
artifacts/hobs-active-tx-ee-non-codebook-continuous-power-boundary-audit/
docs/research/catfish-ee-modqn/hobs-active-tx-ee-non-codebook-continuous-power-implementation-readiness.execution-report.md
```

## What Was Implemented

Implemented a new opt-in power-surface mode:

```text
hobs_power_surface.mode: non-codebook-continuous-power
```

The mode computes active-beam transmit power analytically from post-action
beam load, assigned user channel-pressure inputs, overflow pressure, and
explicit config constants:

```text
p_active_lo_w = 0.05
p_active_hi_w = 0.25
alpha = 0.85
beta = 0.35
kappa = 0.6
bias = -2.0
q_ref = 0.0
n_qos = 50.0
total_power_budget_w = 8.0
```

The implemented channel-quality feature is `log1p(unit-power-snr-linear)`.
That keeps the sidecar numeric range stable while preserving the design
contract that assigned unit-power link quality contributes to
`channel_pressure_b(t)`.

## Config / Namespace Boundary

The new mode is namespace-gated. Config loading rejects
`non-codebook-continuous-power` unless:

```text
track.phase starts with hobs-active-tx-ee-non-codebook-continuous-power
track.label starts with hobs-active-tx-ee-non-codebook-continuous-power
training_experiment.kind =
  hobs-active-tx-ee-non-codebook-continuous-power-implementation-readiness
```

No frozen baseline config, Phase `03C`, QoS-sticky prior gate, RA-EE, or
Catfish artifact namespace was used for the new readiness surface.

## Rollout Power Wiring

For the new mode, `StepEnvironment.step()` applies the policy action first.
Any opt-in structural guard remains trainer-side and shared by both future
roles. The environment then computes post-action beam loads and unit-power
channel-quality features, evaluates continuous `p_b(t)`, and only then
computes SNR / optional SINR, throughput, rewards, and active-TX EE.

The same `StepResult.beam_transmit_power_w` vector feeds:

```text
throughput SNR numerator
total_active_beam_power_w
r1_hobs_active_tx_ee denominator
diagnostic / guardrail checks
```

The readiness tests and artifact recompute `sum_u R_u(t) /
sum_active_beams p_b(t)` from the exported step vector and match
`r1_hobs_active_tx_ee`.

## Candidate / Control Boundary Proof

The future matched pair is represented by:

```text
control:
  configs/hobs-active-tx-ee-non-codebook-continuous-power-throughput-control.resolved.yaml
  r1_reward_mode = throughput

candidate:
  configs/hobs-active-tx-ee-non-codebook-continuous-power-ee-candidate.resolved.yaml
  r1_reward_mode = hobs-active-tx-ee
```

The boundary audit proves both roles share the same continuous power surface,
same QoS-sticky overflow guard, same seeds / seed triplets, same episode budget,
same eval seeds, same objective weights, same trainer hyperparameters, same
checkpoint protocol, and same artifact / diagnostic schema. The only
boundary-critical intended difference is `r1_reward_mode`.

The QoS-sticky overflow guard is retained only as a shared structural guard.
It is not counted as EE objective evidence.

## Boundary-Audit Artifact

Generated:

```text
artifacts/hobs-active-tx-ee-non-codebook-continuous-power-boundary-audit/summary.json
artifacts/hobs-active-tx-ee-non-codebook-continuous-power-boundary-audit/boundary_audit_step_samples.csv
artifacts/hobs-active-tx-ee-non-codebook-continuous-power-boundary-audit/review.md
```

Key artifact flags:

```text
acceptance_result = PASS
matched_boundary_pass = true
same_power_vector_for_candidate_and_control = true
same_throughput_for_candidate_and_control = true
ee_denominator_reuses_step_power_vector = true
policy_action_consequences_change_power = true
active_power_nonconstant = true
active_power_non_codebook = true
selected_power_profile_absent = true
inactive_beams_zero_w = true
power_budget_violations = 0
pilot_training_authorized = false
```

## Tests / Checks Run

```text
.venv/bin/python -m pytest \
  tests/test_hobs_active_tx_ee_non_codebook_continuous_power.py -q
```

Result: `9 passed`.

```text
.venv/bin/python -m pytest \
  tests/test_step.py \
  tests/test_hobs_active_tx_ee_feasibility.py \
  tests/test_hobs_dpc_denominator_check.py \
  tests/test_hobs_active_tx_ee_anti_collapse.py \
  tests/test_hobs_active_tx_ee_qos_sticky_broader_effectiveness.py \
  tests/test_hobs_active_tx_ee_non_codebook_continuous_power.py -q
```

Result: `97 passed`.

```text
.venv/bin/python -m pytest \
  tests/test_phase03c_b_power_mdp_audit.py \
  tests/test_phase03c_c_power_mdp_pilot.py \
  tests/test_ra_ee_07_constrained_power_allocator_distillation.py \
  tests/test_ra_ee_08_offline_association_reevaluation.py \
  tests/test_modqn_smoke.py -q
```

Result: `43 passed`.

```text
.venv/bin/python scripts/run_hobs_active_tx_ee_non_codebook_continuous_power_boundary_audit.py
```

Result: `decision=PASS matched_boundary=True`.

```text
git diff --check
```

Result: no whitespace errors.

## Acceptance Result

```text
PASS
```

All implementation-readiness criteria passed for this slice:

```text
frozen baseline unchanged
new behavior opt-in and namespace-gated
active p_b(t) analytic / continuous
p_b(t) nonconstant, non-codebook, and not selected profile
p_b(t) computed before throughput / reward / EE
same power vector feeds throughput and EE denominator
candidate/control boundary differs only by r1_reward_mode
throughput + same-guard + same-power control is present
forbidden modes absent
```

## Forbidden Claims Still Active

Do not claim:

```text
EE-MODQN effectiveness
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

## Deviations / Blockers

No blocker was hit.

Implementation-specific details:

```text
q_u,b(t) feature = log1p(unit-power-snr-linear)
p_active_hi_w * total_beam_count <= total_power_budget_w is enforced at
environment construction for this mode
selected_power_profile is exported as an empty string for continuous mode
```

These details preserve the required properties: continuous active-beam power,
non-codebook behavior, inactive-beam `0 W`, rollout-time computation, policy
sensitivity through assignment/load/channel inputs, and declared per-beam /
total-power guardrails.

## PASS / BLOCK / NEEDS MORE DESIGN

```text
PASS
```

This PASS is limited to implementation-readiness / boundary-audit. It does not
authorize pilot training. Controller approval is still required before any
bounded pilot.
