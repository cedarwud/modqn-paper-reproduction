# HOBS Active-TX EE Non-Codebook Continuous-Power Design Gate Execution Report

**Date:** `2026-05-01`
**Status:** `PASS` for later controller review only
**Scope:** design gate only. No pilot implementation, training, Catfish,
Multi-Catfish, Phase `03C` continuation, RA-EE learned association, frozen
baseline mutation, HOBS optimizer claim, physical energy-saving claim, or
EE-MODQN effectiveness claim is authorized by this report.

## Changed Files

```text
docs/research/catfish-ee-modqn/hobs-active-tx-ee-non-codebook-continuous-power-design-gate.execution-report.md
```

## Design Summary

New method family / label:

```text
HOBS-active-TX non-codebook continuous-power base EE-MODQN
short label: CP-base-EE-MODQN
```

This is a base EE-MODQN design gate, not full EE-MODQN, not Catfish-EE, and not
a pilot implementation. "Catfish-ready" means only that the method boundary,
artifact namespace, per-step power diagnostics, objective components, and
matched-control protocol are explicit enough for a later Catfish design to use
after, and only after, the base method is promoted. No Catfish replay,
intervention, buffer routing, specialist learner, or Catfish claim is part of
this design.

The design replaces finite-codebook / selected-profile power behavior with a
deterministic analytic continuous-power sidecar that is evaluated during
rollout after the policy action and before SINR, throughput, reward, and EE
metrics are computed.

The primary candidate/control comparison is deliberately narrow:

```text
candidate:
  r1 = hobs-active-tx-ee
  same opt-in structural anti-collapse guard
  same continuous power surface

control:
  r1 = throughput
  same opt-in structural anti-collapse guard
  same continuous power surface
```

The QoS-sticky overflow guard, if retained, is a structural guard only. It is
not EE objective evidence and cannot be counted as EE contribution.

## New Namespace

Future configs, if separately authorized, must use new names such as:

```text
configs/hobs-active-tx-ee-non-codebook-continuous-power-throughput-control.resolved.yaml
configs/hobs-active-tx-ee-non-codebook-continuous-power-ee-candidate.resolved.yaml
configs/hobs-active-tx-ee-non-codebook-continuous-power-boundary-audit.resolved.yaml
```

Future artifacts, if separately authorized, must use new roots such as:

```text
artifacts/hobs-active-tx-ee-non-codebook-continuous-power-throughput-control/
artifacts/hobs-active-tx-ee-non-codebook-continuous-power-ee-candidate/
artifacts/hobs-active-tx-ee-non-codebook-continuous-power-ee-candidate/paired-comparison-vs-throughput-control/
artifacts/hobs-active-tx-ee-non-codebook-continuous-power-boundary-audit/
```

These namespaces must not write into frozen MODQN, Phase `03C`, QoS-sticky,
RA-EE, or Catfish artifact directories.

## `p_b(t)` Definition

Let the policy action plus the same opt-in structural guard produce the
post-action user set for beam `b`:

```text
U_b(t) = users assigned to beam b after action/guard
z_b(t) = 1 if |U_b(t)| > 0 else 0
n_b(t) = |U_b(t)|
q_u,b(t) = current observed channel-quality / unit-power link-quality feature
```

For active beams, define a continuous pressure score:

```text
softplus(x) = log(1 + exp(x))
sigma(x) = 1 / (1 + exp(-x))

channel_pressure_b(t)
  = mean_{u in U_b(t)} softplus(q_ref - q_u,b(t))

load_pressure_b(t)
  = log(1 + n_b(t))

overflow_pressure_b(t)
  = softplus(n_b(t) / n_qos - 1)

x_b(t)
  = alpha * load_pressure_b(t)
  + beta * channel_pressure_b(t)
  + kappa * overflow_pressure_b(t)
  + bias
```

with `alpha >= 0`, `beta >= 0`, and `kappa >= 0`. The active-beam continuous
power is:

```text
p_active_hi_w = min(per_beam_cap_w, total_power_cap_w / max_active_beam_count)

p_b(t)
  = 0, if z_b(t) = 0

p_b(t)
  = p_active_lo_w
    + (p_active_hi_w - p_active_lo_w) * sigma(x_b(t)),
    if z_b(t) = 1
```

`p_active_lo_w`, `p_active_hi_w`, `alpha`, `beta`, `kappa`, `bias`, `q_ref`,
and `n_qos` are config constants. `p_active_hi_w` is a config-time safety cap,
not a runtime optimizer. The runtime formula performs no search, no finite
profile selection, no codebook lookup, no held-out replay lookup, no future
information access, and no HOBS optimizer emulation.

For active beams, `p_b(t)` is analytic and differentiable in the continuous
pressure inputs. It is monotone increasing in load pressure, channel difficulty
pressure, and overflow pressure. Inactive beams use the existing discrete
active/inactive closure and contribute `0 W`.

This is not constant, not a finite codebook, and not a selected fixed profile:

```text
no selected_power_profile
no fixed-low / fixed-mid / fixed-high label
no runtime selector over profile IDs
no rounding to declared power levels
no finite codebook levels
no post-eval replacement of beam_transmit_power_w
```

Denominator variability is necessary for eligibility, but it is not sufficient
for success.

## Rollout / Action Coupling

`p_b(t)` must be computed inside the environment rollout after the policy action
and any same structural guard, before physical step metrics are computed:

```text
state_t
  -> policy action a_t
  -> same opt-in structural guard, if enabled
  -> U_b(t), z_b(t), n_b(t), assigned channel-pressure inputs
  -> continuous p_b(t)
  -> SINR / SNR using p_b(t)
  -> throughput R_u(t)
  -> r1 / r2 / r3 and EE metrics
  -> state_{t+1}
```

The same `beam_transmit_power_w` vector must feed:

```text
SINR numerator / interference semantics
throughput computation
total_active_beam_power_w
EE_system = sum_u R_u(t) / sum_active_beams p_b(t)
r1_hobs_active_tx_ee for the candidate
diagnostics and guardrail checks
```

This is not eval post-processing and not post-hoc rescoring. If a future
implementation computes throughput under one power vector and later rescales
EE under another vector, the design must be blocked.

The policy can affect `p_b(t)` because its association / handover action
changes `U_b(t)`, `z_b(t)`, `n_b(t)`, the assigned users' channel-pressure
aggregate, and overflow pressure before the power sidecar is evaluated. If an
implementation makes `p_b(t)` depend only on exogenous channel/load state that
the policy cannot affect through association/resource/action consequences, the
result is `NEEDS MORE DESIGN`.

## Matched Control Boundary

The primary matched boundary for any future pilot is:

```text
same environment boundary
same continuous power surface and constants
same opt-in structural anti-collapse guard
same non-sticky / handover protections
same seeds and seed triplets
same episode budget
same evaluation seeds
same evaluation schedule
same checkpoint protocol
same objective weights except the r1 mode
same trainer hyperparameters
same artifact and diagnostic schema
no Catfish
no Multi-Catfish
no RA-EE learned association
no Phase 03C selector route
no frozen baseline mutation
```

The required primary control is:

```text
throughput + same anti-collapse guard + same continuous power surface
```

It is a DPC / continuous-power matched throughput control, not the frozen MODQN
baseline. A no-anti-collapse role may be reported only as diagnostic context;
it cannot replace the throughput + same-guard control.

## Candidate vs Control Only-Difference Statement

The only intended candidate/control difference is:

```text
candidate r1_reward_mode = hobs-active-tx-ee
control   r1_reward_mode = throughput
```

Both sides must use:

```text
same continuous p_b(t) surface
same QoS-sticky guard if retained
same seeds
same episode budget
same eval schedule
same checkpoint protocol
same scalar-reward diagnostic treatment
```

The QoS-sticky guard is an opt-in structural guard shared by both roles. It is
not EE objective evidence. A candidate win is not promotable unless it beats
the throughput + same-guard + same-continuous-power control on the predeclared
EE objective boundary while preserving protected QoS / handover / r2 guardrails.

## Failure-Mode Avoidance

Phase `03` / `03B` r1-substitution collapse is avoided by making active
transmit power part of rollout dynamics and requiring the throughput control to
share the same power surface. A reward-only substitution is a stop condition.

Phase `03C` finite-codebook / runtime selector collapse is avoided by removing
profile IDs, selected fixed profiles, finite power levels, and runtime profile
selection. Continuous `p_b(t)` is generated directly from analytic pressure
inputs.

Route `D` one-active-beam learned-policy blocker is addressed only structurally,
through the same opt-in anti-collapse guard on both arms. The guard is not EE
evidence. Future pilot eligibility still requires non-collapsed active-beam
diagnostics and denominator variability.

The capacity-aware forced-split p05 / handover / r2 failure is avoided by not
using a forced `min_active_beams_target` as the success mechanism. If
QoS-sticky overflow is retained, it must remain sticky, QoS-preserving,
non-sticky disabled, and shared by both candidate and control.

The QoS-sticky broader-effectiveness block is handled by making the
anti-collapse-throughput control the decisive control. If the candidate's gains
are explained by throughput + same anti-collapse + same power surface, the
future pilot must block.

RA-EE learned association / proposal association failure is avoided by not
reopening learned association, association proposal refinement, hierarchical
RL, or joint association + power training. The design uses only the existing
policy action consequences plus a shared structural guard.

RA-EE-09 RB / bandwidth non-promotion is respected by not claiming RB /
bandwidth allocation effectiveness and not adding resource-share scheduling as
a hidden success mechanism.

RA-EE-07 finite-codebook power evidence is not reused as this method's power
surface. RA-EE-07 remains a scoped fixed-association finite-codebook offline
result; this design is a new non-codebook continuous-power base gate.

## Proposed Pilot Eligibility Criteria

A future pilot is eligible to be proposed only if the controller first approves
all of the following on paper and in config metadata:

```text
method label is CP-base-EE-MODQN, not full EE-MODQN or Catfish-EE
new config and artifact namespaces are used
p_b(t) formula has no codebook, profile selector, fixed selected profile, or rounding
p_b(t) is computed during rollout before SINR / throughput / reward
candidate and control share the same p_b(t) surface
candidate and control share the same opt-in structural anti-collapse guard
throughput + same anti-collapse + same continuous-power control is primary
the only intended candidate/control difference is r1
policy action consequences affect p_b(t)
no oracle, future information, offline replay oracle, or HOBS optimizer is used
scalar reward is diagnostic only
denominator variability is necessary but not sufficient
```

Future pilot acceptance, if separately authorized, must require at minimum:

```text
matched_boundary_pass = true
candidate_all_evaluated_steps_one_active_beam = false
candidate_denominator_varies_in_eval = true
candidate_active_power_single_point_distribution = false
candidate_power_profile_selector_absent = true
candidate_vs_throughput_same_guard_control_EE_system_delta > 0
candidate_vs_throughput_same_guard_control_p05_ratio >= 0.95
candidate_vs_throughput_same_guard_control_served_ratio_delta >= 0
candidate_vs_throughput_same_guard_control_outage_ratio_delta <= 0
candidate_vs_throughput_same_guard_control_handover_delta <= predeclared tolerance
candidate_vs_throughput_same_guard_control_r2_delta >= predeclared tolerance
budget/per-beam/inactive-power violations = 0
```

If the future candidate loses to throughput + same anti-collapse + same
continuous power, the pilot must be `BLOCK` with no tuning rerun.

## Stop Conditions

Stop immediately if any of these are true:

```text
design is only r1 substitution
power is constant in rollout
power is finite-codebook / selected fixed profile
power is a runtime profile selector
power is post-hoc rescore or eval-only replacement
p_b(t) varies but policy cannot influence it
candidate and control do not share the same power surface
throughput + same anti-collapse + same power control is missing
candidate/control differ by anything other than r1
QoS-sticky guard is counted as EE objective evidence
gain can be explained by anti-collapse-only / throughput same-guard control
design needs Catfish or Multi-Catfish
design needs RA-EE learned association or proposal refinement
design needs oracle, future information, offline replay oracle, or HOBS optimizer behavior
denominator variability is used as sufficient evidence
scalar reward is used as success evidence
future pilot candidate loses to throughput + same anti-collapse control
```

## Forbidden Claims

Do not claim:

```text
EE-MODQN effectiveness
Catfish-EE readiness
Catfish or Multi-Catfish effectiveness
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

## PASS / BLOCK / NEEDS MORE DESIGN

```text
PASS / BLOCK / NEEDS MORE DESIGN: PASS
```

This is a design-gate `PASS` only. It passes because the proposed method family
is materially different from r1-only substitution, finite-codebook selectors,
post-hoc rescoring, Catfish repair, and RA-EE learned association; because
`p_b(t)` is a continuous analytic rollout sidecar; because the policy can affect
`p_b(t)` through association/action consequences; and because the decisive
future control is throughput + the same anti-collapse guard + the same
continuous power surface.

It does not authorize a pilot by itself. A later controller must still approve
implementation scope. Any future pilot that is explained by the throughput +
same anti-collapse + same continuous-power control must be blocked rather than
tuned.
