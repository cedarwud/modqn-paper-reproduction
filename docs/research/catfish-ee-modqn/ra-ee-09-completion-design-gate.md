# RA-EE-09 Completion Design Gate

**Date:** `2026-04-29`
**Status:** `DESIGN PASS; 09E TESTED CANDIDATE NOT PROMOTED`
**Scope:** fixed-association RB / bandwidth allocation design gate. This file
records the design authority. The later execution outcome is recorded in
`ra-ee-09-fixed-association-rb-bandwidth.execution-report.md`.

## Current State

Use this state without reopening earlier gates:

```text
old EE-MODQN r1-substitution route: BLOCKED / STOP
RA-EE fixed-association deployable power allocation: PASS, scoped
RA-EE learned association / hierarchical RL / full RA-EE-MODQN: BLOCKED
Catfish for EE repair: BLOCKED
```

The only positive RA-EE result is fixed association plus the RA-EE-07
deployable stronger power allocator on held-out replay. RA-EE does not yet have
learned association, hierarchical RL, joint association + power training, RB /
bandwidth allocation effectiveness evidence, or full RA-EE-MODQN evidence.

## Execution Outcome

RA-EE-09 was executed through Slice `09E` after this design gate. The
equal-share control, resource accounting, deterministic bounded QoS-slack
candidate, and matched held-out replay were implemented as offline fixed-
association analysis only.

Result:

```text
RA-EE-09 boundary/accounting: PASS
tested bounded-qos-slack-resource-share-allocator: NOT PROMOTED
RA-EE-09 RB / bandwidth effectiveness: NEEDS MORE EVIDENCE
```

The matched replay used the same held-out seeds, fixed association trajectories,
evaluation schedule, and RA-EE-07 deployable stronger power allocator for
control and candidate. The candidate kept association, handover, effective
power vectors, and resource accounting matched, but failed the effectiveness
gate:

```text
EE_system delta: -46.64859074452477
predeclared resource-efficiency delta: -0.1944519284254136
p05 throughput ratio: 0.9016412169223311
```

Use `ra-ee-09-fixed-association-rb-bandwidth.execution-report.md` as the
execution-result authority. This design document must not be read as a
promotion of RB / bandwidth allocation effectiveness.

## RA-EE-09 Goal

RA-EE-09 answers one question:

```text
With association fixed and the power comparison boundary fixed to the RA-EE-07
deployable stronger power allocator, does adding a bounded RB / bandwidth
allocator provide auditable resource-allocation value beyond power allocation?
```

This is not a training gate and not an effectiveness claim. The first approved
output is a design plan for a bounded offline replay pilot.

## Design Boundary

1. Association is fixed by existing trajectories; the candidate must not change
   beam assignment.
2. Control and candidate must use the same seeds, held-out bucket, association
   trajectories, and evaluation schedule.
3. The power allocator is fixed to the RA-EE-07
   `deployable-stronger-power-allocator`; control and candidate
   `effective_power_vector_w` must be stepwise identical or hash-proven
   identical.
4. The RB / bandwidth allocator can only recompute throughput after the power
   vector is resolved; it must not feed back into or alter the power decision.
5. Do not add learned association, hierarchical RL, joint association + power
   training, Catfish, Phase `03C` continuation, or RA-EE-06 / 06B / 08 proposal
   tuning.

## RB / Bandwidth Contract

Primary resource unit:

```text
normalized per-beam bandwidth / resource fraction
```

This is not a physical 3GPP RB claim.

Current throughput:

```text
R_i(t) = B / N_b(t) * log2(1 + gamma_i,b(t))
```

RA-EE-09 generalized throughput:

```text
R_i(t) = B * rho_i,b(t) * log2(1 + gamma_i,b(t))
```

Where:

1. `rho_i,b(t)` is user `i`'s normalized bandwidth share on fixed associated
   beam `b`.
2. Control uses equal/default allocation:
   `rho_i,b(t) = 1 / N_b(t)`, which must reproduce the existing formula.
3. Candidate uses a bounded allocator to choose `rho_i,b(t)`.
4. `gamma_i,b(t)` comes from the same RA-EE-07 power vector and the same
   channel / noise path; RA-EE-09 does not change noise semantics.
5. Inactive beams require `sum_i rho_i,b(t) = 0` and `resource_usage_b = 0`;
   any nonzero value is a violation.
6. Per-active-beam budget is `sum_i rho_i,b(t) = 1.0`; the primary gate must
   not win by leaving bandwidth unused.
7. Total normalized budget is `sum_active_beams 1.0`.
8. Per-user bounds are centered on equal share. Initial recommended config:
   `rho_i,b >= 0.25 / N_b` and `rho_i,b <= min(4.0 / N_b, 1.0)`.
9. Optional RB count is reporting-only, for example
   `rb_count_i = rho_i,b * total_rb_per_active_beam`. If integer RBs are used,
   deterministic rounding audit is required and per-beam sum must remain exact.

Allocator granularity is centralized per-step allocation that outputs per-user
resource shares under per-beam and total budget constraints. It is not a
per-user independent agent and not a learned scheduler.

## Control / Candidate

Control:

```text
fixed association
+ RA-EE-07 deployable stronger power allocator
+ equal/default bandwidth allocation, rho_i,b = 1 / N_b
```

Candidate:

```text
fixed association
+ same RA-EE-07 deployable stronger power allocator
+ bounded centralized per-user resource-share allocator
```

Initial candidate name:

```text
bounded-qos-slack-resource-share-allocator
```

The candidate starts from equal share and performs bounded redistribution under
per-user min/max, per-beam sum, and QoS guardrails. It must be deterministic and
deployable: no oracle labels, future outcomes, or held-out answers.

## Metrics

Report at least:

1. simulated `EE_system = sum_i R_i(t) / sum_active_beams P_b(t)`,
2. throughput mean, p05, and sum throughput,
3. served ratio,
4. outage ratio,
5. handover count, which should remain unchanged under fixed association,
6. load-balance metrics, separated into association load balance and
   throughput / resource load balance,
7. active beam count,
8. total active power,
9. RB / bandwidth usage distribution: per-user share, per-beam sum, min/max,
   entropy, and unused resource,
10. RB / bandwidth budget violations,
11. per-beam and per-user resource violations,
12. gain concentration by seed and trajectory,
13. throughput-vs-EE ranking separation / correlation diagnostics,
14. scalar reward as diagnostic only.

## Artifact / Metadata Plan

Future implementation namespaces:

```text
configs/ra-ee-09-fixed-association-rb-bandwidth-control.resolved.yaml
configs/ra-ee-09-fixed-association-rb-bandwidth-candidate.resolved.yaml
artifacts/ra-ee-09-fixed-association-rb-bandwidth-control-pilot/
artifacts/ra-ee-09-fixed-association-rb-bandwidth-candidate-pilot/
artifacts/ra-ee-09-fixed-association-rb-bandwidth-candidate-pilot/paired-comparison-vs-control/
```

Required metadata:

```text
ra_ee_gate_id = RA-EE-09
method_label = fixed-association RB / bandwidth allocation design gate
association_mode = fixed-replay
association_trajectory_id/hash
held_out_bucket_id
eval_seeds
power_allocator_id = RA-EE-07 deployable-stronger-power-allocator
same_power_vector_as_control = true
resource_unit = normalized_per_beam_bandwidth_fraction
resource_allocator_id
per_beam_budget
total_resource_budget
per_user_min/max
inactive_beam_resource_policy = zero
throughput_formula_version
noise_policy = unchanged
learned_association_disabled = true
hierarchical_RL_disabled = true
catfish_disabled = true
phase03c_continuation_disabled = true
scalar_reward_success_basis = false
```

Review outputs:

```text
summary.json
paired_comparison.json
resource_budget_report.json
step_resource_trace.csv
review.md
```

## Focused Tests

1. RA-EE-09 disabled: baseline and existing RA-EE outputs unchanged.
2. Config namespace gating: only explicit `ra-ee-09-*` configs can enable this
   surface.
3. Fixed-association enforcement: candidate cannot change assignment or
   handover trajectory.
4. Same RA-EE-07 deployable power allocator and same `effective_power_vector_w`
   for control and candidate.
5. RB / bandwidth accounting: per-beam sum, total budget, inactive beam zero.
6. Throughput recomputation: equal-share control is equivalent to
   `B / N_b * log2(1 + gamma)`.
7. QoS guardrail computation: p05, served ratio, and outage match existing
   RA-EE guardrail semantics.
8. Budget violation reporting catches overuse, underuse, per-user min/max, and
   inactive nonzero resource.
9. Artifact metadata and review output contain matched-boundary proof.
10. Forbidden mode tests: no learned association, no Catfish, no EE-MODQN
    r1-substitution, no Phase `03C` mode.

## Acceptance Criteria

RA-EE-09 bounded pilot could pass only if all are true:

1. Candidate has positive held-out simulated `EE_system` delta versus matched
   control, or a predeclared and justified resource-efficiency metric delta.
2. p05 throughput ratio is at least `0.95` versus matched control.
3. Served ratio does not drop.
4. Outage ratio does not increase.
5. Handover count is unchanged; load-balance regression is within a predeclared
   tolerance.
6. Power budget, RB / bandwidth budget, per-beam, per-user, and inactive-resource
   violations are all `0`.
7. Gains are not concentrated in one seed or one trajectory. Initial recommended
   gate: multiple positive seeds / trajectories and max positive share `< 0.80`.
8. Scalar reward is not the success basis.
9. Metadata proves same association, same RA-EE-07 power boundary, and same
   evaluation schedule.

## Stop Conditions

Stop RA-EE-09 immediately if:

1. frozen baseline semantics must change,
2. the result depends on learned association,
3. the candidate wins only by reducing service quality,
4. RB / bandwidth accounting cannot be audited,
5. gains appear only in scalar reward,
6. gains concentrate in one seed or one trajectory,
7. the work is framed as Catfish, HOBS optimizer, physical energy saving, or
   full RA-EE-MODQN,
8. the candidate must feed RB allocation back into the power allocator, breaking
   the matched power boundary,
9. equal/default control cannot reproduce the existing throughput formula.

## Forbidden Claims

Do not claim:

1. full RA-EE-MODQN,
2. learned association effectiveness,
3. old EE-MODQN effectiveness,
4. HOBS optimizer behavior,
5. physical energy saving,
6. Catfish-EE or Catfish repair,
7. RB / bandwidth allocation effectiveness from the tested RA-EE-09 candidate,
8. full paper-faithful reproduction.

## Implementation Slices

1. `09A`: config / schema / design metadata only; no trainer change. Complete.
2. `09B`: disabled-by-default resource accounting and equal-share parity tests.
   Complete.
3. `09C`: offline fixed-association control replay; prove equivalence to the
   current throughput path. Complete.
4. `09D`: deterministic bounded resource-share candidate. Complete.
5. `09E`: matched held-out replay, budget / QoS review, and concentration
   review. Complete; tested candidate not promoted.
6. `09F`: execution report closeout. Complete in
   `ra-ee-09-fixed-association-rb-bandwidth.execution-report.md`.

## Questions / Assumptions

1. Primary unit is normalized bandwidth fraction. Physical RB / noise scaling is
   a separate gate.
2. `ChannelConfig.bandwidth_hz` remains the per-beam throughput pool; RA-EE-09
   does not redefine RF noise.
3. RA-EE-07 deterministic hybrid allocator is the only matched power boundary.
4. First pilot is offline deterministic replay, not RL training.
5. Integer RB count requires `total_rb_per_active_beam` and deterministic
   rounding audit before use.

## PASS / FAIL / NEEDS MORE DESIGN

```text
RA-EE-09 design gate: PASS to bounded offline pilot design
RA-EE-09 boundary/accounting: PASS
RA-EE-09 tested RB / bandwidth candidate: NOT PROMOTED
RA-EE-09 RB / bandwidth effectiveness: NEEDS MORE EVIDENCE
full RA-EE-MODQN: FAIL / BLOCKED
learned association: FAIL / BLOCKED
Catfish-EE: FAIL / BLOCKED
```
