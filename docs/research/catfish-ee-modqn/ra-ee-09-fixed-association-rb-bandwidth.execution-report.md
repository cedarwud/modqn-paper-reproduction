# RA-EE-09 Execution Report: Fixed-Association RB / Bandwidth Allocation

**Date:** `2026-04-29`
**Status:** `NEEDS MORE EVIDENCE / TESTED CANDIDATE NOT PROMOTED`
**Scope:** offline fixed-association matched replay only. No training, learned
association, learned hierarchical RL, joint association + power/resource
training, Catfish, Phase `03C` continuation, frozen baseline mutation, HOBS
optimizer claim, physical energy-saving claim, or full RA-EE-MODQN claim was
performed.

## Protocol

Method label:

```text
fixed-association RB / bandwidth allocation design gate
```

Config:

```text
configs/ra-ee-09-fixed-association-rb-bandwidth-control.resolved.yaml
```

Implementation and entrypoints:

```text
src/modqn_paper_reproduction/analysis/ra_ee_09_fixed_association_rb_bandwidth.py
src/modqn_paper_reproduction/analysis/_ra_ee_09_common.py
src/modqn_paper_reproduction/analysis/_ra_ee_09_resource.py
src/modqn_paper_reproduction/analysis/_ra_ee_09_replay.py
src/modqn_paper_reproduction/analysis/_ra_ee_09_compare.py
scripts/run_ra_ee_09_fixed_association_rb_bandwidth_control.py
scripts/run_ra_ee_09_fixed_association_rb_bandwidth_matched_comparison.py
tests/test_ra_ee_09_fixed_association_rb_bandwidth.py
```

Artifacts:

```text
artifacts/ra-ee-09-fixed-association-rb-bandwidth-control-pilot/
artifacts/ra-ee-09-fixed-association-rb-bandwidth-candidate-pilot/
artifacts/ra-ee-09-fixed-association-rb-bandwidth-candidate-pilot/paired-comparison-vs-control/
```

Primary matched comparison:

```text
fixed association
+ RA-EE-07 deployable stronger power allocator
+ equal-share resource control

vs

fixed association
+ same RA-EE-07 deployable stronger power allocator
+ bounded-qos-slack-resource-share-allocator
```

The resource unit is `normalized_per_beam_bandwidth_fraction`, not a physical
3GPP RB claim. The control uses `rho_i,b(t) = 1 / N_b(t)` and reproduces the
existing throughput formula:

```text
R_i(t) = B / N_b(t) * log2(1 + gamma_i,b(t))
```

The candidate uses:

```text
R_i(t) = B * rho_i,b(t) * log2(1 + gamma_i,b(t))
```

with per-user bounds:

```text
rho_i,b >= 0.25 / N_b
rho_i,b <= min(4.0 / N_b, 1.0)
```

## Slice Results

| Slice | Result | Meaning |
|---|---|---|
| `09A-09C` | `PASS_TO_09D` | Explicit RA-EE-09 config / metadata namespace, resource accounting, equal-share parity, and fixed-association control replay were established. |
| `09D` | `PASS_TO_09E` | Deterministic bounded resource-share candidate was implemented with fixed association, unchanged handover, unchanged power vector, exact resource accounting, and forbidden-mode metadata. |
| `09E` | `NEEDS MORE EVIDENCE / NOT PROMOTED` | Matched held-out replay was auditable, but the tested candidate failed the effectiveness gate. |

## Matched Boundary Proof

Held-out replay:

```text
held-out seeds: 600, 700, 800, 900, 1000
held-out trajectories:
  random-valid-heldout
  spread-valid-heldout
  load-skewed-heldout
  mobility-shift-heldout
  mixed-valid-heldout
matched steps: 250
```

Boundary proof:

```text
same evaluation schedule: true
same association hash per step: true
same association schedule hash: true
same effective power vector hash per step: true
same effective power schedule hash: true
same RA-EE-07 power boundary: true
resource allocation feedback to power decision: false
```

Schedule hashes:

```text
evaluation_schedule_hash:
  653657a30d10bc09b5553db32f43e8e9c72f054266f676f837260e3a95e6323d
control_association_schedule_hash:
  00d4d5c602fdd6109ea3f56502e2a52579424bf0cd370a84c51a810b2ee286bf
candidate_association_schedule_hash:
  00d4d5c602fdd6109ea3f56502e2a52579424bf0cd370a84c51a810b2ee286bf
control_effective_power_schedule_hash:
  78a618eeea88a31c416f809dbd5b4fdbe36c361ad0636ef7fddb669c48a264cc
candidate_effective_power_schedule_hash:
  78a618eeea88a31c416f809dbd5b4fdbe36c361ad0636ef7fddb669c48a264cc
```

## Held-Out Results

Overall matched comparison:

| Metric | Control | Candidate | Delta / ratio |
|---|---:|---:|---:|
| `EE_system` aggregate bps/W | `882.2348210629095` | `835.5862303183848` | `-46.64859074452477` |
| Sum throughput bps | `1442674.4911431228` | `1366392.3831281387` | `-76282.10801498406` |
| Mean throughput bps | `57.706979645724914` | `54.65569532512555` | `-3.0512843205993647` |
| p05 throughput bps | `11.268703793840453` | `10.160327801815596` | ratio `0.9016412169223311` |
| Served ratio | `1.0` | `1.0` | `0.0` |
| Outage ratio | `0.0` | `0.0` | `0.0` |
| Handover count | `10818` | `10818` | `0` |
| Total active power W sum | `1635.25` | `1635.25` | `0.0` |
| Active resource budget sum | `1425.0` | `1425.0` | `0.0` |
| Predeclared resource-efficiency delta | | | `-0.1944519284254136` |

Trajectory-level held-out results:

| Trajectory | EE delta | Sum throughput delta | p05 ratio | Resource-efficiency delta |
|---|---:|---:|---:|---:|
| `load-skewed-heldout` | `-47.70228283619406` | `-6654.468455649083` | `0.6358085778121443` | `-1.7618282505836635` |
| `mixed-valid-heldout` | `-47.15969716417112` | `-14984.993773915397` | `0.8782412211068865` | `-0.24696691286714523` |
| `mobility-shift-heldout` | `-46.32038378418463` | `-18088.109867724066` | `0.5514947165489249` | `-2.1594369918889407` |
| `random-valid-heldout` | `-46.54744875191284` | `-18083.683840118116` | `0.5402930478090904` | `-2.2978704864208606` |
| `spread-valid-heldout` | `-46.29286234981828` | `-18470.852077577496` | `0.46782270641083995` | `-3.5802038565295677` |

The control ranks above the candidate for both throughput and `EE_system`.
Throughput-vs-EE ranking separation is false overall and false for every
held-out trajectory.

## Resource Accounting

Control resource accounting:

```text
active_beam_resource_sum_exact = true
inactive_beam_zero_resource = true
resource_budget_violation_count = 0
resource_budget_violation_step_count = 0
max_active_beam_resource_sum_abs_error = 1.3322676295501878e-15
```

Candidate resource accounting:

```text
active_beam_resource_sum_exact = true
inactive_beam_zero_resource = true
resource_budget_violation_count = 0
resource_budget_violation_step_count = 0
max_active_beam_resource_sum_abs_error = 5.551115123125783e-16
```

The candidate did not fail because of budget or accounting violations. It failed
because matched held-out `EE_system`, sum throughput, p05 throughput ratio, and
the predeclared resource-efficiency metric were negative versus equal-share
control.

## Acceptance Criteria

| Criterion | Result |
|---|---|
| Positive held-out `EE_system` delta or positive predeclared resource-efficiency delta | `false` |
| p05 throughput ratio at least `0.95` | `false` |
| Served ratio does not decrease | `true` |
| Outage ratio does not increase | `true` |
| Handover count unchanged | `true` |
| All power / resource / per-beam / per-user / inactive violations zero | `true` |
| Gains not concentrated | `false` because no positive gain basis exists |
| Metadata proves same association and same power schedule | `true` |
| Scalar reward is not the success basis | `true` |

## Stop / Continue Decision

```text
RA-EE-09 Slice 09E decision: NEEDS MORE EVIDENCE
Tested candidate: NOT PROMOTED
Default next action: STOP THIS CANDIDATE AND WRITE CLOSEOUT
```

No boundary stop condition fired: association, handover, power, and accounting
were auditable and matched. The effectiveness gate failed, so the tested
bounded QoS-slack resource-share allocator must not be promoted and should not
be tuned in place as the next default step.

Any further RB / bandwidth work requires a new explicitly scoped design gate,
not a claim extension from RA-EE-09. Physical RB semantics, integer RB rounding,
noise scaling, learned scheduling, and feedback from resource allocation into
power allocation remain out of scope.

## Validation

Focused RA-EE-09 tests:

```text
.venv/bin/python -m pytest tests/test_ra_ee_09_fixed_association_rb_bandwidth.py -q
```

Result:

```text
11 passed
```

Regression-focused RA-EE and baseline smoke tests:

```text
.venv/bin/python -m pytest \
  tests/test_ra_ee_09_fixed_association_rb_bandwidth.py \
  tests/test_ra_ee_07_constrained_power_allocator_distillation.py \
  tests/test_ra_ee_08_offline_association_reevaluation.py \
  tests/test_modqn_smoke.py
```

Result:

```text
44 passed
```

Artifact command:

```text
.venv/bin/python scripts/run_ra_ee_09_fixed_association_rb_bandwidth_matched_comparison.py
```

Primary artifact:

```text
artifacts/ra-ee-09-fixed-association-rb-bandwidth-candidate-pilot/paired-comparison-vs-control/paired_comparison.json
```

## Allowed Claim

Allowed wording:

```text
Under fixed association and matched RA-EE-07 deployable power allocation, the
tested bounded normalized bandwidth/resource-share allocator was auditable but
did not improve held-out simulated EE or the predeclared QoS-preserving
resource-efficiency metric versus equal-share control.
```

This is a negative / non-promotion result for the tested RA-EE-09 candidate.

## Forbidden Claims Still Active

1. Do not call RA-EE-09 full RA-EE-MODQN.
2. Do not claim learned association effectiveness.
3. Do not claim old EE-MODQN effectiveness.
4. Do not claim RB / bandwidth allocation effectiveness from RA-EE-09.
5. Do not claim HOBS optimizer behavior.
6. Do not claim physical energy saving.
7. Do not claim Catfish-EE or Catfish repair.
8. Do not claim full paper-faithful reproduction.
