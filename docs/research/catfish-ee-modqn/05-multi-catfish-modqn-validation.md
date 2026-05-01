# Phase 05: Multi-Catfish-MODQN Validation

**Status:** Phase 07C records bounded single-Catfish utility; Multi-Catfish remains closed
**Question:** Does objective-specialized Catfish add value beyond single Catfish?

## Decision

`CLOSED AS BOUNDED NEGATIVE RESULT`.

Do not start full multi-agent Catfish. Phase `05A` multi-buffer validation completed and blocked escalation from the original buffer construction because objective-specific buffers did not prove meaningfully distinct sample types on the bounded surface.

Phase `05R` then passed an analysis-first objective-buffer-redesign gate. Phase
`05B` planning passed and the bounded pilot has now run. The implementation
surface is runnable, but the result does not promote Multi-Catfish-MODQN:
primary Multi-Catfish did not beat single Catfish, controls explained away the
result, and replay starvation diagnostics triggered. Phase `05C` closes this
route. Phase `07A` then opened a single-Catfish-first recovery path, and Phase
`07B` produced bounded single-Catfish intervention utility evidence. Phase
`07C` records that this does not reopen Multi-Catfish or Phase `06`.

Execution result:

```text
Phase 05A bounded diagnostic completion: PASS
Objective-specific buffer distinctness: FAIL
Phase 05B full multi-agent validation planning: BLOCK
Phase 05R objective-buffer admission redesign diagnostics: PASS
Phase 05B planning draft in a later run: ALLOWED
Phase 05B implementation now: FORBIDDEN
Phase 05B planning boundary: PASS
Future implementation prompt draft: ALLOWED AFTER REVIEW
Phase 05B bounded runnable evidence: PASS
Phase 05B acceptance / effectiveness: FAIL
Multi-Catfish-MODQN promotion: BLOCK
Phase 05C read-only disposition and claim synthesis: PASS
Phase 07A read-only recovery gate: PASS
Phase 07B bounded single-Catfish utility pilot: PASS
Phase 07C post-07B disposition: PASS
Direct Phase 05B continuation: BLOCK
Phase 06 / Catfish-EE-MODQN final claim: FAIL / BLOCKED
```

See `05a-multi-buffer-validation.execution-report.md` and
`05r-objective-buffer-redesign-gate.execution-report.md`. The Phase `05B`
planning boundary is recorded in
`05b-multi-catfish-planning.execution-report.md`; the bounded pilot result is
recorded in `05b-multi-catfish-bounded-pilot.execution-report.md`; the route
closeout is recorded in
`05c-multi-catfish-route-disposition.execution-report.md`; the recovery gate is
recorded in `07a-catfish-recovery-gate.execution-report.md`; the post-07B
disposition is recorded in
`07c-catfish-intervention-utility-disposition.execution-report.md`.

## Why This Is Independent

Multi-catfish is the main novelty candidate. It must be separated from both EE objective substitution and basic Catfish feasibility.

## Candidate Definition

```text
Multi-Catfish-MODQN
= one main MODQN agent
+ objective-specialized Catfish challengers
```

Possible full form:

```text
catfish-r1: throughput-specialized
catfish-r2: handover-specialized
catfish-r3: load-balance-specialized
```

## Recommended Sub-Phases

### 05A: Multi-Buffer Validation

Execution status: complete, failed for escalation.

Before adding three full agents, validate objective-specific replay buffers:

```text
r1 high-value buffer
r2 high-value buffer
r3 high-value buffer
```

This was the first valid Phase 05 target. It tested whether the three objectives produce meaningfully different high-value experience before adding three specialist learners.

Use objective-wise percentile / rank criteria, not raw scalar reward:

1. `r1` buffer: high throughput. Risk: duplicates scalar high-value replay because `r1` dominates scale.
2. `r2` buffer: low handover penalty, meaning `r2` closer to zero is better. Risk: captures sticky policies with fewer handovers but worse throughput or load balance.
3. `r3` buffer: load-balance penalty closer to zero is better. Risk: sacrifices throughput for balance.

Required 05A diagnostics:

1. top-sample Jaccard overlap across `r1` / `r2` / `r3` buffers,
2. each buffer's `r1` / `r2` / `r3` distribution,
3. objective-wise percentile thresholds,
4. intervention batch composition,
5. whether main-agent updates actually receive different sample types.

Phase `05A` result:

1. `r1` high-value replay nearly duplicated scalar/high-throughput replay,
2. `r2` percentile admission degenerated to all samples,
3. `r3` did not provide distinct intervention samples after the `r2` degeneration,
4. the diagnostic triggered `diagnostics-cannot-prove-distinct-sample-types` and `objective-percentile-degenerated`.

Therefore this buffer construction has no clear research value for full multi-Catfish escalation.

### 05R: Objective-Buffer Redesign Gate

Execution status: complete, passed for later Phase `05B` planning draft only.

Phase `05R` may answer only this question:

```text
Can objective-buffer admission be redesigned so r1/r2/r3 buffers are selective,
distinct, and interpretable without changing original MODQN semantics?
```

Allowed scope:

1. use Phase `05A` transition samples / diagnostics where sufficient,
2. redesign only buffer admission, ranking, threshold, and tie handling,
3. keep `r1 = throughput`, `r2 = handover penalty`, and `r3 = load balance`,
4. simulate fixed-budget intervention composition only,
5. report a redesign-gate result, not a training result.

Forbidden scope:

1. no long training or new policy training,
2. no `catfish-r1`, `catfish-r2`, or `catfish-r3` learners,
3. no reward, state, action, or backbone change,
4. no EE reward / objective / Catfish-EE framing,
5. no random tie-breaking used to manufacture distinctness,
6. no scalar reward alone as success evidence.

Required `05R` diagnostics:

1. admission share, threshold, tie mass, and unique value count per buffer,
2. pairwise Jaccard for `r1` / `r2` / `r3`,
3. Jaccard versus scalar Phase `04` high-value replay,
4. each buffer's `r1` / `r2` / `r3` distribution,
5. distinct samples versus the union of scalar and other objective buffers,
6. non-target degradation table,
7. fixed total intervention-budget composition simulation, such as
   `70 main + 10 r1 + 10 r2 + 10 r3`, without training.

Minimum acceptance criteria:

1. every objective buffer is a bounded subset, not all samples and not empty,
2. `r1` no longer duplicates scalar/high-throughput replay under a predeclared
   threshold,
3. `r2` coarse/tie degeneration is solved by a non-arbitrary rule,
4. every buffer has measurable unique contribution to the simulated
   intervention composition,
5. target-objective lift is visible without significant non-target damage,
6. the report explains why the result is objective-specialized replay rather
   than random extra data mixing.

Stop and close or redesign the route if any buffer again admits all or no
samples, `r1` still duplicates scalar replay, `r3` distinctness comes only from
throughput collapse, distinctness requires changing original MODQN semantics or
adding EE, or full specialist learners are needed just to determine whether the
buffers are distinct.

Phase `05R` result:

1. candidate: `guarded-residual-objective-admission`,
2. buffer shares: `r1 = 0.10`, `r2 = 0.1417`, `r3 = 0.20305`,
3. pairwise Jaccards were below the review threshold,
4. Jaccard versus scalar Phase `04` high-value replay stayed low,
5. fixed `70 main + 10 r1 + 10 r2 + 10 r3` composition improved expected
   `r1`, `r2`, and `r3` means without significant non-target damage.

This result is planning permission only. It is not a Multi-Catfish
effectiveness claim.

### 05B: Multi-Agent Validation

Execution status: complete for bounded runnable evidence; failed for
acceptance / effectiveness.

The accepted planning boundary is:

1. bounded `20`-episode pilot,
2. `3` matched seed triplets minimum,
3. evaluation seeds `[100, 200, 300, 400, 500]`,
4. eval cadence every `5` episodes and final,
5. total Catfish share fixed at `0.30`,
6. primary intervention mix `70 main + 10 r1 + 10 r2 + 10 r3`,
7. matched MODQN, single-Catfish, multi-buffer / single-learner, and random /
   uniform equal-budget controls,
8. primary run shaping off,
9. guarded-residual admission with complete threshold ties only.

The draft implementation prompt is
`prompts/05b-multi-catfish-implementation-draft.prompt.md`. It is not
executable unless the user explicitly authorizes Phase `05B` implementation /
bounded training in a later request.

Phase `05B` bounded pilot result:

1. all `5` comparators × `3` matched seed triplets completed,
2. actual Catfish ratio was `0.296875`, close to configured `0.30`,
3. primary Multi-Catfish final mean scalar was `608.037110`,
4. single Catfish final mean scalar was `609.209311`,
5. multi-buffer / single-learner final mean scalar was `608.293825`,
6. random / uniform buffer control final mean scalar was `609.128592`,
7. primary improved `r2` versus single Catfish but did not improve `r1` or
   `r3`,
8. no NaN or action collapse was detected,
9. replay starvation counters were nonzero,
10. multi-buffer / single-learner and random-buffer controls matched or
    explained away the primary result.

Decision:

```text
Phase 05B bounded runnable evidence: PASS
Phase 05B acceptance / effectiveness: FAIL
Multi-Catfish-MODQN promotion: BLOCK
```

### 05C: Route Disposition

Execution status: complete.

Phase `05C` formally closes the current Multi-Catfish route:

1. Phase `05B` is a bounded negative result,
2. Multi-Catfish-MODQN is not promoted,
3. longer Phase `05B` training by default is blocked,
4. shaping-on primary, ratio tuning, specialist tweaks, and more seeds are
   blocked by default,
5. Phase `06` / Catfish-EE-MODQN final claims remain blocked.

Safe current claim:

1. Phase `05B` proved the implementation surface is runnable and
   diagnostically useful under the bounded original-MODQN reward pilot.

Forbidden claim:

1. objective-specialized Multi-Catfish adds value beyond single Catfish.

Default next action is paper-section synthesis: claim boundary, limitations,
and negative-results narrative. Any future Multi-Catfish reopening requires a
new explicit design gate explaining why matched controls and replay-starvation
diagnostics would not explain the result.

### 07A: Read-Only Recovery Gate

Execution status: complete.

Phase `07A` locks the recovery conclusion:

```text
Phase 07A read-only recovery gate: PASS
Phase 07B implementation/training from Phase 07A alone: NOT AUTHORIZED
Direct Phase 05B continuation: BLOCK
Multi-Catfish promotion: BLOCK
Phase 06 / Catfish-EE-MODQN: BLOCK
```

The Phase `05B` failure model is:

1. current evidence does not support implementation failure as the primary
   explanation,
2. current evidence does not justify simple budget insufficiency / longer
   training,
3. the strongest working explanation is intervention utility not proven beyond
   equal-budget controls,
4. replay starvation is an observed confounder / stop trigger.

The next R&D route is single-Catfish intervention utility / causal diagnostics,
not Multi-Catfish tuning. Future Phase `07B`, if separately authorized, must be
single-Catfish-first and compare against no-intervention, random/equal-budget
replay injection, replay-only single learner, no-asymmetric-gamma, and matched
MODQN control.

Future Phase `07B` must keep original MODQN reward semantics: `r1 = throughput`,
`r2 = handover penalty`, `r3 = load balance`, no EE, no Catfish-EE, no frozen
baseline mutation, primary shaping-off, and no scalar-reward-only success
claim.

### 07B / 07C: Single-Catfish Intervention Utility Disposition

Execution status: Phase `07B` bounded pilot complete; Phase `07C` disposition
complete.

Phase `07B` tested the single-Catfish-first recovery hypothesis instead of
continuing Phase `05B`. It compared primary shaping-off single Catfish against
matched MODQN, no-intervention, random / equal-budget injection, replay-only
single learner, and no-asymmetric-gamma controls.

Bounded result:

```text
Phase 07B bounded implementation / diagnostics / pilot: PASS
Bounded single-Catfish intervention utility evidence: PASS
Broader Catfish-MODQN effectiveness promotion: NOT PROMOTED
Multi-Catfish reopening: BLOCKED / DEFERRED
Phase 06 / Catfish-EE-MODQN: BLOCKED
```

Final mean scalar values:

1. primary shaping-off single Catfish: `609.209311`,
2. matched MODQN control: `608.359098`,
3. no-intervention: `608.294631`,
4. random / equal-budget injection: `608.683775`,
5. replay-only single learner: `608.486224`.

Phase `07C` interprets this as bounded utility evidence for the single-Catfish
intervention path under the original MODQN reward surface. It is not scalar-only
because `r1` and `r3` component deltas support the primary branch.

Caveats:

1. `r2` / handovers worsened,
2. aggregate metrics did not support asymmetric gamma as an active mechanism,
3. broader Catfish-MODQN effectiveness is not promoted,
4. Multi-Catfish and Phase `06` remain blocked.

At Phase `07C`, the next gate was Phase `07D` r2-guarded single-Catfish
robustness planning, not Multi-Catfish reopening. Phase `07D` has now run and
failed its acceptance gate.

### 07D: R2-Guarded Single-Catfish Robustness

Execution status: bounded pilot complete; acceptance failed.

Phase `07D` tested whether r2 / handover guardrails could preserve Phase `07B`
single-Catfish intervention utility without handover degradation. The guard was
limited to Catfish replay admission, injection eligibility, batch composition,
and skip policy. Original MODQN reward semantics remained fixed.

Bounded result:

```text
Phase 07D bounded implementation / runs / diagnostics: PASS
Phase 07D acceptance / recovery promotion: FAIL
Broader Catfish-MODQN effectiveness: NOT PROMOTED
Multi-Catfish reopening: BLOCKED
Phase 06 / Catfish-EE-MODQN: BLOCKED
```

Primary guarded final mean:

1. scalar: `608.988400`,
2. `r1`: `1228.125646`,
3. `r2`: `-4.293333`,
4. `r3`: `-18.932115`,
5. handovers: `858.666667`.

Primary scalar deltas remained positive versus matched MODQN
(`+0.629302`), no-intervention (`+0.693769`), random / equal-budget
(`+0.304625`), and replay-only (`+0.382683`). However, the predeclared
non-inferiority margins failed versus matched MODQN:

```text
r2_delta = -0.051667 < -0.02
handover_delta = +10.333333 > +5
```

The Phase `07D` summary also reports `starvation_stop_trigger_absent=false`.

Disposition:

1. Phase `07D` does not invalidate Phase `07B`'s bounded utility evidence,
2. Phase `07D` blocks broader Catfish-MODQN effectiveness promotion,
3. Phase `07D` does not reopen Multi-Catfish,
4. Phase `07D` does not authorize Phase `06` / Catfish-EE-MODQN,
5. default next action is paper synthesis / claim-boundary writing, not guard
   tuning.

Candidate specialists:

```text
catfish-r1 agent
catfish-r2 agent
catfish-r3 agent
```

Specialists must not optimize only their own target objective. Admission to intervention should use non-target guardrails:

1. `r1` specialist: high `r1`, with `r2` / `r3` not below the single-Catfish tolerance band.
2. `r2` specialist: high `r2`, with `r1` / `r3` not significantly degraded.
3. `r3` specialist: high `r3`, with `r1` / `r2` not significantly degraded.

Without non-target constraints, the design becomes single-objective replay injection rather than multi-objective improvement.

## Inputs

1. Phase 04 single Catfish feasibility.
2. Current MODQN objective definitions.
3. Catfish high-value replay concept.

## Non-Goals

1. Do not introduce EE unless Phase 05 is explicitly repeated under EE later.
2. Do not claim final method value without comparison to single Catfish.
3. Do not allow an objective-specialist to ignore all other objectives.
4. Do not treat Phase `05R` diagnostics as a Multi-Catfish effectiveness claim.
5. Do not use Phase `05A` diagnostic completion as Phase `05B` escalation evidence.
6. Do not treat Phase `05B` planning permission as Phase `05B` implementation permission.
7. Do not execute the Phase `05B` implementation draft without explicit user authorization.
8. Do not treat Phase `07A` as authorization for Phase `07B` implementation or
   training.
9. Do not recover by direct Phase `05B` continuation or Multi-Catfish tuning.
10. Do not treat Phase `07B` as broad Catfish-MODQN effectiveness.
11. Do not claim asymmetric gamma as the active Phase `07B` mechanism.
12. Do not claim handover / `r2` improvement from Phase `07B`.
13. Do not treat Phase `07D` as recovery promotion after the failed r2 /
    handover non-inferiority gate.
14. Do not continue Phase `07D` guard tuning by default.

## Checks

1. Can Phase `05R` produce r1/r2/r3 high-value buffers that are meaningfully different?
2. Does objective-specialized replay improve more than scalar high-value replay?
3. Do objective specialists conflict?
4. Does main intervention become biased toward one objective?
5. Is multi-catfish better than single Catfish-MODQN?
6. What intervention ratio is interpretable?
7. What is the minimum bounded pilot that would separate multi-buffer replay
   value from extra learners, intervention budget, and compute?

## Required Metrics

Compare against single Catfish-MODQN with equal budget, same seeds, and same checkpoint protocol:

1. scalar reward: train, eval, best, and final,
2. individual `r1`, `r2`, and `r3`,
3. handover count,
4. convergence speed: AUC, episode to threshold, and best-eval timing,
5. final-vs-best gap,
6. cross-seed variance,
7. TD loss, Q-value spikes, replay starvation, and action collapse,
8. replay diagnostics: each buffer size, threshold, sample distribution, and actual intervention proportion,
9. complexity cost: memory, training time, and agent count versus interpretable benefit.

## Intervention Mixing

Multi-catfish must not win by using a larger total Catfish intervention budget than single Catfish. If single Catfish uses `70 main / 30 catfish`, multi-catfish should start with the same total Catfish share.

Candidate schedules:

```text
70 main + 10 r1 + 10 r2 + 10 r3
70 main + 15 r1 + 9 r2 + 6 r3
70 main + adaptive(r1,r2,r3), total catfish = 30
warm-up: 100 main -> 90/10 -> 80/20 -> 70/30
gated: only inject specialist samples that pass non-target guardrails
```

The clean first version is `70 main + 10/10/10`. Adaptive mixing should be an ablation, not the first claim-bearing design.

## Promotion Conditions

Multi-catfish can claim added value only if all of these hold:

1. Phase `05R` proves objective-specific buffers have meaningfully different distributions after the failed Phase `05A` construction.
2. Multi-catfish beats single Catfish under equal budget.
3. Improvement is not only scalar reward; it is explainable through `r1` / `r2` / `r3` or stability.
4. At least one non-target objective is not significantly damaged.
5. final-vs-best gap, collapse risk, and cross-seed variance are not worse than single Catfish.
6. Ablation shows gains come from objective-specialized replay / intervention, not random extra data mixing.

## Stop Conditions

Stop Phase 05 escalation if:

1. `r1` / `r2` / `r3` buffers highly overlap,
2. multi-buffer matches full multi-agent performance, making three agents unnecessary,
3. multi-agent improves only one objective while degrading others,
4. the advantage is within variance or comes from more intervention / compute budget,
5. training becomes less stable, with replay starvation, Q divergence, or action collapse,
6. intervention ratio sensitivity causes results to fail across seeds or ratios.

## Decision Gate

Promote only if multi-catfish shows distinct value over single Catfish and can be explained as objective-specialized replay/intervention.

Stop if it only increases complexity without a clear metric or interpretability benefit.

For now, Phase `05` is closed as a bounded negative result. Do not continue with
longer training, shaping-on primary, ratio tuning, specialist tweaks, or more
seeds by default. Any reopening requires a new explicit design gate that
starts with single-Catfish intervention utility / causal diagnostics and
explains why matched controls and replay-starvation diagnostics would not
explain the result.

Phase `07B` provided bounded single-Catfish intervention utility evidence, but
it did not reopen Phase `05` or promote Multi-Catfish. Phase `07D` then tested
the remaining handover / `r2` caveat with guardrails.

Phase `07D` has now tested that guardrail path and failed the acceptance gate.
Do not reopen Phase `05`, promote Multi-Catfish, or start Phase `06` from Phase
`07D`. Default next action is paper synthesis / claim-boundary writing.

## Expected Output

A Phase `05B` bounded-pilot execution report:

1. configs and artifacts created under the Phase `05B` namespace,
2. bounded pilot comparators and ablations completed,
3. diagnostics and metrics summary,
4. deviations from the implementation prompt,
5. explicit `PASS / FAIL` for bounded runnable evidence and acceptance,
6. explicit statement that no Catfish-EE / EE / full paper-faithful claim was
   made,
7. recommendation to treat the result as not promoted unless a new design gate
   is opened.

The accepted Phase 05 review is recorded in `reviews/05-multi-catfish-modqn-validation.review.md`.
