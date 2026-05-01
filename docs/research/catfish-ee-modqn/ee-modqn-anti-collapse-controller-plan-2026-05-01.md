# EE-MODQN Anti-Collapse Controller Plan

**Date:** `2026-05-02`
**Status:** controller handoff plan after QoS-sticky broader-effectiveness block and CP-base bounded matched pilot BLOCK
**Scope:** planning, delegation, result intake, and claim-boundary control only.
The controller does not implement code, run training, or tune experiments.

## Why This Exists

The latest HOBS active-TX EE route resolved one old concern but exposed the next
hard blocker:

```text
HOBS active-TX EE formula / reward wiring: PASS, scoped
HOBS-inspired DPC sidecar denominator gate: PASS
Route D tiny learned-policy denominator check: BLOCK
hard stop: all_evaluated_steps_one_active_beam = true
capacity-aware anti-collapse assignment gate: BLOCK
QoS-sticky overflow anti-collapse gate: PASS, scoped
QoS-sticky robustness / attribution gate: PASS, scoped
QoS-sticky broader-effectiveness gate: BLOCK
CP-base bounded matched pilot: BLOCK
EE-MODQN effectiveness: NOT PROMOTED / STOP CURRENT QOS-STICKY AND CP-BASE EE OBJECTIVE ROUTES
```

The denominator can vary and same-policy throughput-vs-EE ranking can separate.
The current failure is learned beam-selection collapse: the greedy policy still
uses exactly one active beam across evaluated steps.

The first bounded anti-collapse / capacity / assignment design gate has now
run. The tested `capacity-aware-greedy-assignment` candidate removed
one-active-beam collapse, but failed the p05 throughput and handover / `r2`
guardrails. The current useful work is therefore not more Route `D` training,
not Catfish, and not tuning the same assignment candidate by default. Any
continuation needed a new design gate that preserves anti-collapse without
tail-throughput or handover collapse.

That follow-up gate has now passed in bounded form. The tested
`qos-sticky-overflow-reassignment` candidate removed one-active-beam collapse
without forced active-beam targets, preserved p05 throughput and handover /
`r2` guardrails, and kept non-sticky moves disabled. This is still scoped
anti-collapse evidence only, not EE-MODQN effectiveness.

The robustness / mechanism-attribution gate also passed in bounded form across
`3` matched seed triplets and nearby threshold ablations. The QoS-ratio guard
was not binding on this surface; attribution should be to sticky overflow
reassignment under non-sticky / handover protections.

The broader-effectiveness gate has now blocked the current QoS-sticky
HOBS-active-TX EE objective-contribution route. The candidate still removed
one-active-beam collapse and preserved the DPC denominator boundary, but it did
not improve active-TX EE beyond the matched anti-collapse-throughput control and
failed handover / `r2` guardrails versus that control. The remaining positive
evidence is scoped anti-collapse robustness only, not EE-MODQN effectiveness.

The failure-informed non-codebook continuous-power design gate has now passed as
design-only. It defines `CP-base-EE-MODQN` as a materially different future base
EE boundary: the EE candidate and throughput + same-anti-collapse control must
share the same analytic continuous rollout-time `p_b(t)` power surface, with
`r1_reward_mode` as the only intended difference. This does not authorize code
implementation, pilot training, Catfish-EE, or EE-MODQN effectiveness.

That implementation-readiness / boundary-audit slice has now completed and
passed. It implemented the opt-in `non-codebook-continuous-power` mode, proved
namespace gating and frozen-baseline preservation through focused tests, and
generated a boundary-audit artifact showing the future candidate/control pair
can share the same continuous power vector with only `r1_reward_mode` differing.
The artifact explicitly records `pilot_training_authorized=false`; readiness
alone did not authorize effectiveness.

The separately authorized bounded matched pilot has now completed and blocked.
The matched boundary passed, but the EE candidate lost against throughput +
same QoS-sticky guard + same continuous power, violated the handover / `r2`
guardrails, and only won scalar reward. This blocks CP-base effectiveness and
does not authorize tuning reruns.

## Controller Role

The controller dialogue is responsible for:

1. keeping the evidence boundary current,
2. deciding which bounded gate should be opened next,
3. writing or approving the execution prompt for a separate implementation
   dialogue,
4. reading the execution report returned by that dialogue,
5. deciding `PASS`, `BLOCK`, or `NEEDS MORE DESIGN`,
6. updating handoff / master-plan docs after each result.

The controller must not:

1. write code,
2. run training,
3. tune hyperparameters interactively,
4. reopen Phase `03C`,
5. use Catfish as EE repair,
6. claim EE-MODQN effectiveness from denominator variability alone.

## Current Authority Files

Read in this order:

```text
AGENTS.md
docs/research/catfish-ee-modqn/00-validation-master-plan.md
docs/research/catfish-ee-modqn/execution-handoff.md
docs/research/catfish-ee-modqn/development-guardrails.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-modqn-feasibility.execution-report.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-anti-collapse-design-gate.execution-report.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-qos-sticky-anti-collapse-design-gate.execution-report.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-qos-sticky-robustness-gate.execution-report.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-qos-sticky-broader-effectiveness-gate.execution-report.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-non-codebook-continuous-power-design-gate.execution-report.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-non-codebook-continuous-power-implementation-readiness.execution-report.md
docs/research/catfish-ee-modqn/prompts/hobs-active-tx-ee-non-codebook-continuous-power-implementation-readiness.prompt.md
docs/research/catfish-ee-modqn/prompts/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot.prompt.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot.execution-report.md
artifacts/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot-summary/summary.json
docs/research/catfish-ee-modqn/repository-curation-2026-05-01.md
docs/research/catfish-ee-modqn/energy-efficient/README.md
docs/research/catfish-ee-modqn/energy-efficient/ee-formula-final-review-with-codex-2026-05-01.md
docs/research/catfish-ee-modqn/energy-efficient/modqn-r1-to-hobs-active-tx-ee-design-2026-05-01.md
docs/ee-report.md
```

Supporting historical evidence:

```text
docs/research/catfish-ee-modqn/03d-ee-route-disposition.execution-report.md
docs/research/catfish-ee-modqn/03c-c-power-mdp-pilot.execution-report.md
docs/research/catfish-ee-modqn/ra-ee-07-constrained-power-allocator-distillation.execution-report.md
docs/research/catfish-ee-modqn/ra-ee-09-fixed-association-rb-bandwidth.execution-report.md
```

## Latest Gate Results

The first bounded anti-collapse gate tested:

```text
capacity-aware-greedy-assignment
```

Result:

```text
PASS / BLOCK / NEEDS MORE DESIGN: BLOCK
```

Key facts:

```text
matched_boundary_pass=true
all_evaluated_steps_one_active_beam=false
active_beam_count_distribution={"2.0": 50}
denominator_varies_in_eval=true
active_power_single_point_distribution=false
p05_throughput_ratio_vs_control=0.28501578600107075
handover_delta=+495
r2_delta=-0.4949999999999999
```

The candidate proved that a minimal explicit assignment constraint can remove
one-active-beam collapse under the same HOBS active-TX EE / DPC boundary. It did
not pass because the anti-collapse behavior came with unacceptable p05
throughput and handover / `r2` regressions. Scalar reward improvement was
diagnostic only.

The follow-up bounded anti-collapse gate tested:

```text
qos-sticky-overflow-reassignment
```

Result:

```text
PASS / BLOCK / NEEDS MORE DESIGN: PASS, scoped
```

Key facts:

```text
matched_boundary_pass=true
all_evaluated_steps_one_active_beam=false
active_beam_count_distribution={"7.0": 50}
denominator_varies_in_eval=true
active_power_single_point_distribution=false
p05_throughput_ratio_vs_control=2.522568929129207
handover_delta=-211
r2_delta=+0.21100000000000002
nonsticky_move_count=0
```

The candidate passes the bounded anti-collapse acceptance criteria and does not
trigger the anti-collapse-specific stop-loss point. It does not by itself
promote EE-MODQN effectiveness.

The robustness / mechanism-attribution gate tested:

```text
primary-qos-sticky
no-qos-guard-ablation
stricter-qos-ablation
threshold-sensitivity-45
threshold-sensitivity-55
```

Result:

```text
PASS / BLOCK / NEEDS MORE DESIGN: PASS, scoped
```

Key facts:

```text
matched_boundary_pass=true
primary_all_seed_statuses_pass=true
active_beam_count_distribution={"7.0": 150}
p05_throughput_ratio_vs_control=3.105617320531727
handover_delta=-292.00000000000006
r2_delta=+0.29200000000000004
threshold_fragility_detected=false
nonsticky_move_count=0
```

Mechanism attribution:

```text
QoS ratio guard was not binding in this bounded run.
Attribution: sticky overflow reassignment under non-sticky / handover protections.
```

The broader-effectiveness gate tested:

```text
matched-throughput-control
hobs-ee-control-no-anti-collapse
qos-sticky-ee-candidate
anti-collapse-throughput-control
```

Result:

```text
PASS / BLOCK / NEEDS MORE DESIGN: BLOCK
```

Key facts:

```text
candidate_all_evaluated_steps_one_active_beam=false
candidate_active_beam_count_distribution={"7.0": 150}
candidate_denominator_varies_in_eval=true
candidate_active_power_single_point_distribution=false
candidate_vs_anti_collapse_throughput_control_EE_delta=-0.32138238121922313
candidate_vs_anti_collapse_throughput_control_p05_ratio=0.9606352634550617
candidate_vs_anti_collapse_throughput_control_handover_delta=+83.33333333333331
candidate_vs_anti_collapse_throughput_control_r2_delta=-0.08333333333333334
```

Controller disposition:

```text
Anti-collapse mechanism remains PASS, scoped.
EE objective contribution is BLOCK.
Current QoS-sticky EE objective-contribution route has reached stop-loss.
```

The CP-base non-codebook continuous-power design gate tested:

```text
HOBS-active-TX non-codebook continuous-power base EE-MODQN
```

Result:

```text
PASS / BLOCK / NEEDS MORE DESIGN: PASS, design-only
```

Controller disposition:

```text
Current QoS-sticky EE objective route remains BLOCK / stop-loss.
CP-base design gate is the accepted design-only continuation.
Implementation / pilot training is not authorized by the design report.
The next allowed worker step was implementation-readiness / boundary audit.
```

The CP-base implementation-readiness / boundary-audit slice tested:

```text
non-codebook-continuous-power wiring and matched-boundary metadata
```

Result:

```text
PASS / BLOCK / NEEDS MORE DESIGN: PASS, readiness-only
```

Controller disposition:

```text
CP-base implementation-readiness passes.
The future matched pilot boundary is representable.
Pilot training was not authorized by readiness alone.
A separate bounded pilot was later authorized and is recorded below.
```

The CP-base bounded matched pilot tested:

```text
CP-base-EE-MODQN bounded matched pilot
```

Result:

```text
PASS / BLOCK / NEEDS MORE DESIGN: BLOCK
```

Key metrics:

```text
matched_boundary_pass=true
only_intended_difference_is_r1_reward_mode=true
candidate_vs_control_EE_system_delta=-1.057740305239463
candidate_vs_control_p05_ratio=0.9675993577312307
candidate_vs_control_handover_delta=+77.33333333333331
candidate_vs_control_r2_delta=-0.07733333333333334
per_seed_EE_deltas=[-1.1009355380937222, -1.1279326781493637, -0.9443526994751892]
scalar_reward_diagnostic_delta=+4312.822050380359
power_violations=0/0/0
```

Controller disposition:

```text
CP-base matched boundary passes.
CP-base EE objective contribution is BLOCK.
Candidate loses to throughput + same guard + same continuous power.
Candidate violates handover / r2 guardrails and only wins scalar reward.
No tuning rerun or Catfish-EE authorization follows from this result.
```

## Possible Future Gate Families

Default future work remains paper synthesis / claim-boundary writing unless the
user explicitly continues EE research. The CP-base bounded matched pilot is now
blocked, so there is no active implementation prompt to run by default. Any EE
continuation requires a new design gate with a materially different base-EE
mechanism. It must not rerun the failed capacity-aware candidate, tune the
current QoS-sticky candidate, tune the blocked CP-base candidate, or keep
broadening the same route by default.

Candidate families, if a separate design gate is explicitly reopened, remain:

1. capacity-aware action masking,
2. overload penalty tied to per-beam user count or served ratio,
3. active-beam diversity / load-spread constraint,
4. centralized assignment constraint,
5. renamed resource-allocation MDP with explicit resource actions,
6. non-codebook analytic continuous-power sidecar shared by the EE candidate
   and throughput + same-anti-collapse control.

The controller should pick the smallest auditable candidate only after a new
design rationale. Do not combine several mechanisms unless the design review
justifies the coupling.

## Required Matched Boundary

Any execution agent must preserve:

1. frozen original MODQN baseline semantics,
2. new opt-in config / artifact namespace,
3. same seeds for control and candidate,
4. same episode budget,
5. same evaluation schedule,
6. same checkpoint protocol,
7. same DPC sidecar parameters unless the gate explicitly tests DPC sensitivity,
8. same HOBS active-TX EE formula,
9. no Catfish,
10. no RA-EE association route,
11. no Phase `03C` continuation.

For any CP-base bounded pilot prompt, preserve:

1. rollout-time continuous `p_b(t)`, not post-hoc rescore,
2. same continuous power surface for candidate and throughput control,
3. same opt-in structural anti-collapse guard for candidate and throughput
   control,
4. only `r1_reward_mode` differs between future roles,
5. no oracle, future information, offline replay oracle, or HOBS optimizer
   behavior.

## Primary Acceptance Criteria

An anti-collapse pilot can only pass if all are true:

1. `all_evaluated_steps_one_active_beam = false`,
2. active-beam count has more than one value or has a declared non-degenerate
   multi-beam target distribution,
3. `denominator_varies_in_eval = true`,
4. `active_power_single_point_distribution = false`,
5. throughput-vs-EE ranking separation remains present or is explicitly
   explained,
6. p05 throughput does not collapse relative to matched control,
7. served ratio does not decrease,
8. outage ratio does not increase,
9. handover / `r2` regression is within a predeclared tolerance,
10. scalar reward alone is not used as success evidence.

## Stop Conditions

Stop the route if any of these occur:

1. candidate still has `all_evaluated_steps_one_active_beam = true`,
2. anti-collapse only works by forcing no service or severe p05 throughput
   collapse,
3. broader EE objective contribution is explained away by a matched
   anti-collapse-throughput control,
4. candidate harms protected handover / `r2` guardrails versus the relevant
   matched control,
5. gains appear only in scalar reward,
6. implementation mutates frozen baseline behavior,
7. mechanism requires Catfish,
8. mechanism reopens RA-EE learned association or Phase `03C`,
9. diagnostics cannot prove matched boundary,
10. CP-base boundary-audit output is framed as effectiveness evidence,
11. CP-base candidate/control metadata cannot prove the shared continuous power
    surface and only-`r1` intended difference,
12. a CP-base pilot candidate loses to throughput + same-guard + same-power
    control,
13. the CP-base pilot prompt is expanded into longer training, extra tuning
    rounds, Catfish, Phase `03C`, RA-EE learned association, or scalar-only
    success.

## Result Intake Protocol

When an execution dialogue returns:

1. read changed files and artifact paths,
2. verify tests / checks reported,
3. compare reported metrics against the acceptance criteria,
4. classify the result as `PASS`, `BLOCK`, or `NEEDS MORE DESIGN`,
5. update:
   - `00-validation-master-plan.md`,
   - `execution-handoff.md`,
   - `docs/ee-report.md`,
   - a new execution report if one was not already produced,
6. write the next execution prompt only if the previous gate justifies it.

## Claim Boundary

Allowed current claim:

```text
HOBS active-TX EE can be wired into MODQN and the active transmit-power
denominator can vary under a HOBS-inspired DPC sidecar. Route `D` remains
blocked by one-active-beam collapse. The first capacity-aware anti-collapse
candidate removed that collapse, but is also blocked because it failed p05
throughput and handover / `r2` guardrails. The QoS-sticky overflow candidate
passes bounded anti-collapse and robustness gates by avoiding one-active-beam
collapse while preserving the declared p05, served/outage, handover / `r2`, and
power/accounting guardrails on the matched protocols. The broader-effectiveness
gate blocks EE objective-contribution claims because the anti-collapse-throughput
control explains the useful gains and the QoS-sticky EE candidate fails
handover / `r2` guardrails versus that control. A separate CP-base design gate
and implementation-readiness slice passed for a future non-codebook
continuous-power base method, but only as readiness evidence: it requires a
shared rollout-time continuous power surface and a throughput +
same-anti-collapse control for any separately authorized pilot.
That separately authorized CP-base pilot has now blocked: the EE candidate lost
to throughput + same guard + same power and violated handover / `r2` guardrails.
```

Forbidden:

1. EE-MODQN effectiveness,
2. physical energy saving,
3. HOBS optimizer reproduction,
4. Catfish-EE repair,
5. full RA-EE-MODQN,
6. learned association effectiveness,
7. scalar reward as success evidence,
8. "more training will fix Route D" without a new anti-collapse mechanism,
9. capacity-aware assignment anti-collapse as EE-MODQN effectiveness,
10. QoS-sticky anti-collapse as general EE-MODQN effectiveness,
11. QoS-sticky robustness as physical energy saving or HOBS optimizer behavior,
12. QoS-sticky broader-effectiveness as EE objective contribution,
13. another tuning pass on the current QoS-sticky EE objective route by default,
14. CP-base design or implementation-readiness as pilot authorization or
    EE-MODQN effectiveness,
15. blocked CP-base bounded pilot as EE-MODQN effectiveness,
16. another tuning rerun of the blocked CP-base candidate by default,
17. Catfish-EE before the base EE method beats matched controls.
