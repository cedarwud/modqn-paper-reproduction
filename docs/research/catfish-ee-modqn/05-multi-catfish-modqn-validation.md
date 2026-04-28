# Phase 05: Multi-Catfish-MODQN Validation

**Status:** Multi-buffer path defined; full multi-agent needs evidence  
**Question:** Does objective-specialized Catfish add value beyond single Catfish?

## Decision

`NEEDS MORE EVIDENCE`.

Do not start with full multi-agent Catfish. Phase 04 single Catfish-MODQN is still `NEEDS MORE EVIDENCE`, so three Catfish agents would add more replay buffers, learners, intervention scheduling, and failure modes before the base mechanism is proven stable. Phase 05 should first run `05A` multi-buffer validation.

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

Before adding three full agents, validate objective-specific replay buffers:

```text
r1 high-value buffer
r2 high-value buffer
r3 high-value buffer
```

This is the first valid Phase 05 target. It tests whether the three objectives produce meaningfully different high-value experience before adding three specialist learners.

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

If the three buffers strongly overlap, multi-catfish has no clear research value.

### 05B: Multi-Agent Validation

Only if 05A is useful, add objective-specialized Catfish agents.

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

## Checks

1. Are r1/r2/r3 high-value buffers meaningfully different?
2. Does objective-specialized replay improve more than scalar high-value replay?
3. Do objective specialists conflict?
4. Does main intervention become biased toward one objective?
5. Is multi-catfish better than single Catfish-MODQN?
6. What intervention ratio is interpretable?

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

1. `05A` proves objective-specific buffers have meaningfully different distributions.
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

For now, Phase 05 remains `NEEDS MORE EVIDENCE`. Only `05A` multi-buffer validation is ready to run as the next bounded step.

## Expected Output

A multi-catfish validation report:

1. multi-buffer result,
2. multi-agent result if applicable,
3. objective conflict analysis,
4. intervention-mix analysis,
5. comparison to single Catfish,
6. recommendation for final EE integration.

The accepted Phase 05 review is recorded in `reviews/05-multi-catfish-modqn-validation.review.md`.
