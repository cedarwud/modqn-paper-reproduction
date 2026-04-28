# Review: Phase 05 Multi-Catfish-MODQN Validation

**Date:** `2026-04-28`  
**Decision:** `NEEDS MORE EVIDENCE`  
**Scope:** multi-catfish validation design only; no full multi-agent implementation or effectiveness claim is promoted by this review.

## Decision Summary

Phase 05 should not start with full multi-agent Catfish. Phase 04 single Catfish-MODQN is still `NEEDS MORE EVIDENCE`, so adding three Catfish agents now would mix several new variables:

1. more replay buffers,
2. more learners,
3. more intervention scheduling,
4. possible specialist conflicts,
5. more compute and memory budget.

If it fails, the cause would be unclear. If it improves, attribution would also be unclear. The next valid step is `05A` multi-buffer validation.

## 05A: Multi-Buffer Validation

First establish three objective-specific high-value buffers:

```text
r1 high-value buffer
r2 high-value buffer
r3 high-value buffer
```

This is cleaner than three agents because it tests whether the objectives actually select different high-value experience.

Use objective-wise percentile / rank instead of raw scalar reward:

1. `r1` buffer: high throughput. Risk: duplicates scalar high-value replay because `r1` dominates reward scale.
2. `r2` buffer: low handover penalty, with `r2` closer to zero treated as better. Risk: captures sticky policies that reduce handovers while hurting throughput or load balance.
3. `r3` buffer: load-balance penalty closer to zero treated as better. Risk: sacrifices throughput to balance load.

Required diagnostics:

1. top-sample Jaccard overlap across buffers,
2. each buffer's `r1` / `r2` / `r3` distribution,
3. objective-wise percentile thresholds,
4. intervention batch composition,
5. whether main-agent updates receive distinct sample types after intervention.

If the buffers strongly overlap, multi-catfish has no clear research value.

## 05B: Multi-Agent Validation

Only proceed to three agents if `05A` shows distinct buffer distributions:

```text
catfish-r1 agent
catfish-r2 agent
catfish-r3 agent
```

Each specialist needs non-target guardrails before its samples enter intervention:

1. `r1` specialist: high `r1`, with `r2` / `r3` inside the single-Catfish tolerance band.
2. `r2` specialist: high `r2`, with `r1` / `r3` not significantly degraded.
3. `r3` specialist: high `r3`, with `r1` / `r2` not significantly degraded.

Without these guardrails, the method becomes single-objective replay injection, not multi-objective improvement.

## Required Comparison

Multi-catfish must compare against single Catfish-MODQN using equal budget, same seeds, and the same checkpoint protocol.

Required metrics:

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

Multi-catfish must not use a larger total Catfish intervention share than single Catfish. If single Catfish uses:

```text
70 main + 30 catfish
```

the clean first multi-catfish schedule is:

```text
70 main + 10 r1 + 10 r2 + 10 r3
```

Other schedules can be tested later:

```text
70 main + 15 r1 + 9 r2 + 6 r3
70 main + adaptive(r1,r2,r3), total catfish = 30
warm-up: 100 main -> 90/10 -> 80/20 -> 70/30
gated: only inject specialist samples that pass non-target guardrails
```

Adaptive mixing should be an ablation, not the first claim-bearing setup.

## Conditions For Added-Value Claim

Multi-catfish can claim added value only if:

1. `05A` proves objective-specific buffers have meaningfully different distributions.
2. Multi-catfish beats single Catfish under equal budget.
3. Improvement is explainable through `r1` / `r2` / `r3` or stability, not scalar reward alone.
4. At least one non-target objective is not significantly damaged.
5. final-vs-best gap, collapse risk, and cross-seed variance are not worse than single Catfish.
6. Ablation shows gains come from objective-specialized replay / intervention, not random extra data mixing.

## Stop Conditions

Stop escalation if:

1. `r1` / `r2` / `r3` buffers highly overlap,
2. multi-buffer already matches full multi-agent performance,
3. multi-agent improves only one objective while degrading others,
4. the advantage is within variance or comes from larger intervention / compute budget,
5. training becomes less stable, with replay starvation, Q divergence, or action collapse,
6. results are highly sensitive to intervention ratio or seed.

## Result

`NEEDS MORE EVIDENCE`.

Phase 05 has research value, but only `05A` multi-buffer validation is ready as the next bounded step. Full three-agent multi-catfish should remain blocked until objective-specific buffers are distinct and equal-budget comparison against single Catfish-MODQN supports the added complexity.
