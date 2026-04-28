# Review: Phase 06 Final Catfish-EE-MODQN Validation

**Date:** `2026-04-28`  
**Decision:** `NEEDS MORE EVIDENCE`  
**Scope:** final-method validation design only; no final Catfish-EE-MODQN claim is promoted by this review.

## Decision Summary

The Phase 06 validation direction is reasonable, but final Catfish-EE-MODQN cannot be promoted yet.

Phase 03, Phase 04, and Phase 05 are all still `NEEDS MORE EVIDENCE`:

1. Phase 03 has not shown that EE-MODQN is more than reward rescaling or service reduction.
2. Phase 04 has not shown that single Catfish replay / intervention is stable and effective on the original MODQN backbone.
3. Phase 05 has not shown that multi-catfish is necessary or better than single Catfish.

Therefore Phase 06 can only define the controlled final validation design. It cannot support final-method claims yet.

## Primary Comparison

Do not use:

```text
original MODQN vs Catfish-EE-MODQN
```

as the sole proof. Original MODQN uses `r1 = throughput`, while the final method uses `r1 = EE`. A direct-only comparison mixes objective substitution, HOBS power denominator behavior, Catfish replay / intervention, and multi-catfish complexity.

The correct primary comparison is:

```text
EE-MODQN
vs
Catfish-EE-MODQN
```

If multi-catfish is part of the claim:

```text
EE-MODQN
vs
Single-Catfish-EE-MODQN
vs
Multi-Catfish-EE-MODQN
```

Original MODQN remains a context anchor only.

## Required Metrics

Final reports must include:

1. `EE_system = sum_i R_i(t) / sum_active_beams P_b(t)`, where `P_b` is linear W, variable, and consistent with SINR power semantics.
2. Per-user EE as credit-assignment diagnostics only.
3. Raw throughput: total, mean per user, and low-percentile throughput.
4. Total active beam power.
5. Handover count / `r2`.
6. Load-balance gap / `r3`.
7. Service outage / below-threshold ratio / served-unserved ratio.
8. Scalar reward as training / checkpoint diagnostics only.
9. Convergence speed: AUC, episode-to-threshold, and best-eval timing.
10. Best-eval checkpoint: selection criterion, selection episode, final-vs-best, and full metric panel on the same eval seeds.
11. Replay composition: main / catfish buffer size, thresholds, sample distribution, and actual mixed-in ratio.
12. Intervention count: trigger count, batch ratio, and actual catfish sample count.
13. Stability / collapse: final-vs-best gap, cross-seed variance, TD loss / Q spikes, NaN, action collapse, handover collapse, and replay starvation.

## Added-Value Conditions

Catfish improves EE-MODQN only if, under the same EE formula, state / action, seeds, training budget, evaluation cadence, target sync, replay settings, and checkpoint rule:

1. `EE_system` improves over EE-MODQN,
2. throughput and outage remain within the pre-declared QoS guardrail,
3. handover count and load-balance gap do not degrade beyond tolerance,
4. convergence speed or stability improves,
5. replay / intervention diagnostics show Catfish actually participated in training,
6. ablations show the gain is not only shaping, extra mixed data, more compute, or checkpoint luck.

## Non-Improvements

Do not count as real improvement:

1. `EE_system` rises while raw throughput, low-percentile throughput, or served ratio drops sharply.
2. Outage or below-threshold ratio increases.
3. Total beam power falls only because fewer users are served.
4. Scalar reward rises without support from EE, throughput, handover, and load-balance metrics.
5. Throughput reward and EE reward rankings are nearly identical.
6. Handover count falls because of sticky policy while throughput or load balance collapses.
7. Per-user EE rises but system-level EE does not.

## Allowed Claims

If evidence is complete, the final method may claim:

1. Catfish replay / intervention adds EE or stability improvement under fixed EE-MODQN objective.
2. The improvement passes throughput and service-quality guardrails.
3. Specific Catfish mechanisms contribute, if ablation supports them.
4. Objective-specialized replay / intervention improves beyond single Catfish, if multi-catfish passes equal-budget single-vs-multi comparison.

## Disallowed Claims

Do not claim:

1. Catfish is effective because it only beats original MODQN.
2. Higher scalar reward alone means the final method is better.
3. Per-user EE equals system EE.
4. Absolute physical energy saving without full power-denominator assumptions, sensitivity, and hardware power model.
5. Multi-catfish beats single Catfish without equal-budget ablation.
6. The original Catfish actor-critic backbone is better suited to LEO handover.
7. Collapse is solved; at most report reduced collapse risk.

## Required Ablations

A final claim requires at least:

1. `EE-MODQN`,
2. `Single-Catfish-EE-MODQN`,
3. `Multi-Catfish-EE-MODQN` if multi-catfish is claimed,
4. no Catfish intervention,
5. no replay stratification,
6. no asymmetric gamma,
7. no competitive shaping, with shaping-off as primary result,
8. single scalar high-value replay vs objective-specific buffers,
9. multi-buffer only vs full multi-agent,
10. fixed equal total intervention share, such as single `70/30` vs multi `70/10/10/10`,
11. adaptive mixing as ablation only,
12. objective-buffer overlap / Jaccard diagnostics,
13. non-target guardrail ablation.

## Result

`NEEDS MORE EVIDENCE`.

Phase 06 can enter controlled validation design, but final Catfish-EE-MODQN cannot enter final claim until the unresolved evidence gaps from Phases 03-05 are closed.
