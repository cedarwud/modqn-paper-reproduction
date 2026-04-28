# Review: Phase 04 Single Catfish-MODQN Feasibility

**Date:** `2026-04-28`  
**Decision:** `NEEDS MORE EVIDENCE`  
**Scope:** Single Catfish-MODQN feasibility design only; no EE objective, multi-catfish design, or effectiveness claim is promoted by this review.

## Decision Summary

Catfish-style dual-agent replay / intervention training can reasonably be attached to the original MODQN backbone for feasibility validation, provided Phase 04 strictly preserves the original reward surface:

```text
r1 = throughput
r2 = handover penalty
r3 = load balance
```

The correct positioning is to test Catfish as a training strategy, not as a replacement objective, state/action redesign, or original Catfish / CDRL actor-critic backbone transplant.

## Method Boundary

The primary Phase 04 method should keep:

1. same MODQN state encoding,
2. same discrete action space,
3. same original reward surface,
4. same MODQN backbone,
5. same baseline comparison discipline.

It may add:

1. main MODQN agent,
2. catfish MODQN agent,
3. main replay buffer,
4. high-value catfish replay buffer,
5. asymmetric discount factors,
6. periodic mixed replay intervention.

Competitive reward shaping should be disabled in the primary feasibility run or isolated as an ablation.

## Allowed Claims

1. Whether Catfish replay / intervention can be connected to MODQN.
2. Whether dual-agent, dual-replay, asymmetric gamma, and mixed replay train stably.
3. Whether high-value replay receives enough samples and enters main-agent updates during intervention.
4. Under the same MODQN objective, Catfish-MODQN's effect on scalar reward, `r1` / `r2` / `r3`, handover count, convergence speed, and best-eval checkpoint.
5. Mechanism-specific contribution only when supported by ablation.

## Disallowed Claims

1. Do not claim the task objective or reward changed.
2. Do not claim the original Catfish / CDRL actor-critic backbone is better suited to this task.
3. Do not declare Catfish-MODQN better using scalar reward alone.
4. Do not claim improvement without multi-seed, equal-budget, same-checkpoint-protocol comparison.
5. Do not attribute gains to replay stratification, gamma asymmetry, intervention, or shaping without ablation.
6. Do not claim late-training collapse is solved; only report whether it is reduced or worsened.

## Required Metrics

1. Scalar reward: train, eval, best, and final.
2. `r1` / `r2` / `r3` separately.
3. Handover count: at least total and per episode.
4. Convergence speed: episode to fixed threshold, AUC, and best-eval timing.
5. Best-eval checkpoint under Phase 01 semantics, with final checkpoint retained.
6. Replay composition: main / catfish buffer size, high-value threshold, quality percentiles, and each replay's `r1` / `r2` / `r3` distribution.
7. Intervention count: trigger count, batch ratio, and actual catfish sample count mixed into main updates.
8. Stability / collapse indicators: final-vs-best gap, cross-seed variance, TD loss / Q-value spikes, NaN, action collapse, handover collapse, and replay starvation.

## High-Value Criterion

The first feasibility pass may use:

```text
quality = 0.5*r1 + 0.3*r2 + 0.2*r3
```

This is acceptable as an initial high-value replay criterion, but it must be audited for reward-scale dominance. Reports should include quality percentiles and component contributions so the high-value buffer does not silently become only throughput-heavy replay.

## Competitive Reward Shaping

Competitive shaping should be off in the primary Phase 04 run.

The core claim is that Catfish can attach to MODQN without changing the original reward. Even catfish-side auxiliary shaping can blur that claim. It should be introduced only as an ablation after the shaping-off design has been evaluated.

## Promotion Conditions

Promote only if bounded evidence shows:

1. dual-agent training is rerunnable, with no replay starvation or clear Q divergence,
2. catfish replay reliably provides high-value samples after warm-up,
3. intervention actually triggers and batch composition matches settings,
4. best-eval scalar reward is not lower than baseline, or convergence speed / stability clearly improves,
5. `r1` / `r2` / `r3` do not show single-objective cheating,
6. handover-count improvement is not caused by policy rigidity,
7. final-vs-best gap is not worse than baseline,
8. initial ablation shows replay / intervention has a real signal.

## Result

`NEEDS MORE EVIDENCE`.

The design direction is methodologically reasonable and has no current blocker, but it only supports a clean feasibility-validation design. It does not yet support claiming that Catfish-MODQN is feasible, effective, or worth expanding to a more complex method family.
