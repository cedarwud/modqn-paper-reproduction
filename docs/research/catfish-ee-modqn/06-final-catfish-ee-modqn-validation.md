# Phase 06: Final Catfish-EE-MODQN Validation

**Status:** Final validation design defined; final claim blocked  
**Question:** Does Catfish improve EE-MODQN under the final claim boundary?

## Decision

`NEEDS MORE EVIDENCE`.

The Phase 06 comparison design is valid, but final Catfish-EE-MODQN cannot be promoted yet. Phase 03, Phase 04, and Phase 05 are all still `NEEDS MORE EVIDENCE`, so the final method currently has no promoted EE-MODQN effect, no promoted single-Catfish mechanism, and no promoted multi-catfish necessity.

## Why This Is Independent

This is the final-method validation. It should only run after EE formula, EE-MODQN, and Catfish mechanism questions are clear enough to interpret.

## Primary Comparisons

Minimum:

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

Original MODQN remains useful as context, but it is not the sole fair baseline because it uses a different `r1`.

Do not use:

```text
original MODQN vs Catfish-EE-MODQN
```

as the sole proof. That comparison mixes objective substitution, HOBS power denominator behavior, Catfish replay / intervention, and possible multi-catfish complexity. It cannot identify what caused an improvement or regression.

## Inputs

1. Phase 02 EE formula validation.
2. Phase 03 EE-MODQN validation.
3. Phase 04 single Catfish feasibility.
4. Phase 05 multi-catfish validation if used.

## Non-Goals

1. Do not claim improvement over original MODQN alone as the main proof.
2. Do not hide EE formula assumptions.
3. Do not omit throughput / service-quality reporting.

## Metrics

1. `EE_system = sum_i R_i(t) / sum_active_beams P_b(t)`, where `P_b` is linear W, variable, and consistent with SINR power semantics.
2. Per-user EE as credit-assignment diagnostics only, not as system EE.
3. Raw throughput: total, mean per user, and low-percentile throughput.
4. Total active beam power.
5. Handover count / `r2`.
6. Load-balance gap / `r3`.
7. Service outage / below-threshold ratio / served-unserved ratio.
8. Scalar reward as training / checkpoint diagnostics only.
9. Convergence speed: AUC, episode-to-threshold, and best-eval timing.
10. Best-eval checkpoint performance: selection criterion, selection episode, final-vs-best, and complete metric panel on the same eval seeds.
11. Replay composition: main / catfish buffer size, thresholds, sample distribution, and actual mixed-in ratio.
12. Intervention count: trigger count, batch ratio, and actual catfish sample count.
13. Stability / collapse indicators: final-vs-best gap, cross-seed variance, TD loss / Q spikes, NaN, action collapse, handover collapse, and replay starvation.

## Added-Value Conditions

Catfish adds value over EE-MODQN only if, under the same EE formula, state / action, seeds, training budget, evaluation cadence, target sync, replay settings, and checkpoint rule:

1. `Catfish-EE-MODQN` improves `EE_system` over `EE-MODQN`,
2. raw throughput and outage remain within the pre-declared QoS guardrail,
3. handover count and load-balance gap do not degrade beyond the allowed tolerance,
4. convergence speed or stability improves,
5. replay / intervention diagnostics prove Catfish actually participated in training,
6. ablations show the gain is not caused only by shaping, extra mixed data, more compute, or checkpoint luck.

## Non-Improvements

Do not count the result as a final-method improvement if:

1. `EE_system` rises while raw throughput, low-percentile throughput, or served ratio drops sharply,
2. outage or below-threshold ratio increases,
3. total beam power falls only because fewer users are served,
4. scalar reward rises without support from `EE_system`, throughput, handover, and load-balance metrics,
5. throughput reward and EE reward produce nearly identical ranking, indicating throughput divided by a constant,
6. handover count falls because of sticky policy while throughput or load balance collapses,
7. per-user EE rises but system-level EE does not.

## Allowed Claims

If evidence exists, final reports may claim:

1. Catfish replay / intervention adds EE or stability improvement under a fixed EE-MODQN objective.
2. The improvement passes throughput and service-quality guardrails.
3. Specific Catfish mechanisms contribute, if supported by ablation.
4. Objective-specialized replay / intervention adds value beyond single Catfish, if multi-catfish passes equal-budget single-vs-multi comparison.

## Disallowed Claims

Do not claim:

1. Catfish is effective because final method beats original MODQN alone.
2. Higher scalar reward alone means the final method is better.
3. Per-user EE is system EE.
4. Absolute physical energy saving without complete power-denominator assumptions, sensitivity, and hardware power model.
5. Multi-catfish is better than single Catfish without equal-budget single-vs-multi ablation.
6. The original Catfish actor-critic backbone is better suited to LEO handover.
7. Collapse is solved; at most claim reduced collapse risk if evidence supports it.

## Required Ablations

A final claim needs at least:

1. `EE-MODQN`,
2. `Single-Catfish-EE-MODQN`,
3. `Multi-Catfish-EE-MODQN` if multi-catfish is claimed,
4. no Catfish intervention,
5. no replay stratification,
6. no asymmetric gamma,
7. no competitive shaping, with shaping-off as the primary result,
8. single scalar high-value replay vs objective-specific buffers,
9. multi-buffer only vs full multi-agent,
10. fixed equal total intervention share, such as single `70/30` vs multi `70/10/10/10`,
11. adaptive mixing only as ablation,
12. objective-buffer overlap / Jaccard diagnostics,
13. non-target guardrail ablation.

## Decision Gate

Promote only if Catfish improves or meaningfully stabilizes EE-MODQN without unacceptable service-quality, handover, or load-balance regression.

Stop if gains only come from reduced service, reduced throughput below guardrail, or unexplained reward scaling.

For now, Phase 06 remains `NEEDS MORE EVIDENCE`. It can enter controlled validation design, but it cannot enter final claim until Phases 03-05 produce supporting evidence.

## Expected Output

A final validation report:

1. method comparison,
2. EE improvement summary,
3. service-quality guardrail result,
4. Catfish contribution summary,
5. ablation summary,
6. final allowed claims,
7. disallowed claims.

The accepted Phase 06 review is recorded in `reviews/06-final-catfish-ee-modqn-validation.review.md`.
