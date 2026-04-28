# Phase 04: Single Catfish-MODQN Feasibility

**Status:** Feasibility design defined; needs pilot evidence  
**Question:** Can Catfish training mechanics attach to the original MODQN backbone?

## Decision

`NEEDS MORE EVIDENCE`.

Catfish-style dual-agent replay / intervention training can reasonably be attached to the original MODQN backbone for feasibility validation. This phase is not allowed to change the state space, action space, reward surface, or MODQN backbone. It can only test whether Catfish works as a sibling training strategy under the original objective.

## Why This Is Independent

This phase validates Catfish as a training strategy over discrete MODQN before mixing in EE or multi-catfish design.

## Method Boundary

Keep the original MODQN objective:

```text
r1 = throughput
r2 = handover penalty
r3 = load balance
```

Add only Catfish-style training mechanics:

1. main MODQN agent,
2. catfish MODQN agent,
3. main replay,
4. high-value catfish replay,
5. asymmetric discount factors,
6. periodic mixed replay intervention,
7. optional competitive reward shaping only in ablation.

The primary feasibility run should use shaping-off:

```text
dual agent + high-value replay + asymmetric gamma + periodic mixed replay
```

Competitive reward shaping should be disabled in the first claim-bearing run or reported only as an ablation because it can blur the claim that Phase 04 attaches Catfish without changing the original reward surface.

## High-Value Criterion

First feasibility pass may use the original scalarized reward:

```text
quality = 0.5*r1 + 0.3*r2 + 0.2*r3
```

This is for Catfish mechanism validation only. It is not an EE claim.

Record the quality percentile and each `r1` / `r2` / `r3` contribution. Otherwise, the high-value buffer may silently become a throughput-heavy replay buffer due to reward-scale dominance.

## Inputs

1. Phase 01 baseline anchor.
2. Catfish notes / report.
3. Current MODQN training-flow explanation.

## Non-Goals

1. Do not modify the objective to EE.
2. Do not introduce three catfish agents.
3. Do not claim energy-efficiency improvement.

## Checks

1. Does dual-agent training remain stable?
2. Does catfish replay receive enough high-value samples?
3. Does intervention degrade or improve main-agent learning?
4. Is best-eval performance better, worse, or neutral?
5. Is any improvement concentrated only in `r2` handover?
6. Does Catfish reduce or worsen collapse risk?

## Required Metrics

1. Scalar reward: train, eval, best, and final.
2. `r1` / `r2` / `r3` separately, not only weighted sum.
3. Handover count: at least total and per episode.
4. Convergence speed: episode to fixed threshold, AUC, and best-eval timing.
5. Best-eval checkpoint under Phase 01 semantics, with final checkpoint retained.
6. Replay composition: main / catfish buffer size, high-value threshold, quality percentiles, and each replay's `r1` / `r2` / `r3` distribution.
7. Intervention count: trigger count, batch ratio, and actual catfish sample count mixed into main updates.
8. Stability / collapse indicators: final-vs-best gap, cross-seed variance, TD loss / Q-value spikes, NaN, action collapse, handover collapse, and replay starvation.

## Allowed Claims

1. Whether Catfish replay / intervention can be connected to MODQN engineering-wise.
2. Whether dual-agent, dual-replay, asymmetric gamma, and mixed replay can train stably.
3. Whether high-value replay receives enough samples and participates in main updates.
4. Under the same MODQN objective, Catfish-MODQN's effect on scalar reward, `r1` / `r2` / `r3`, handover count, convergence speed, and best-eval checkpoint.
5. Mechanism-specific contribution only if ablations support it.

## Disallowed Claims

1. Do not claim a changed task objective or reward.
2. Do not claim the original Catfish / CDRL actor-critic backbone is better suited to this discrete handover task.
3. Do not declare Catfish-MODQN better using scalar reward alone.
4. Do not claim improvement without multi-seed, equal-budget, same-checkpoint-protocol comparison.
5. Do not attribute improvement to replay stratification, gamma asymmetry, intervention, or shaping without ablation.
6. Do not claim late-training collapse is solved; only report whether it is reduced or worsened.

## Promotion Conditions

Promote only after at least bounded-pilot evidence shows:

1. dual-agent training is rerunnable, with no replay starvation or clear Q divergence,
2. catfish replay reliably provides high-value samples after warm-up,
3. intervention actually triggers and batch composition matches settings,
4. best-eval scalar reward is not lower than baseline, or convergence speed / stability clearly improves,
5. `r1` / `r2` / `r3` do not show single-objective cheating,
6. handover-count improvement is not caused by policy rigidity,
7. final-vs-best gap is not worse than baseline,
8. initial ablation shows replay / intervention has a real signal.

## Decision Gate

Promote only if Catfish can be described as a feasible sibling training strategy over MODQN.

Stop if dual replay / intervention is unstable or cannot be evaluated without changing the baseline objective.

For now, Phase 04 remains `NEEDS MORE EVIDENCE` until bounded pilot, same-protocol baseline comparison, replay / intervention diagnostics, and shaping-off primary results exist.

## Expected Output

A feasibility report:

1. Catfish-MODQN method definition,
2. replay routing definition,
3. intervention definition,
4. metrics,
5. stability risks,
6. recommendation for multi-catfish or EE phases.

The accepted Phase 04 review is recorded in `reviews/04-single-catfish-modqn-feasibility.review.md`.
