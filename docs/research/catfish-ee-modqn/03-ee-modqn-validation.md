# Phase 03: EE-MODQN Validation

**Status:** Bounded paired pilot complete; needs more evidence  
**Question:** What changes when only the first MODQN objective becomes energy efficiency?

## Decision

`NEEDS MORE EVIDENCE`.

Phase `02B` provides a defensible opt-in EE metric surface for paired testing,
and a bounded Phase `03` pilot has now run. The EE-MODQN effect itself is not
promoted: best-eval `EE_system`, throughput, served ratio, active power, and
load balance tied `MODQN-control`; EE-MODQN worsened handover / `r2`; and
throughput-vs-EE correlation plus same-policy rescoring showed high rescaling
risk.

The execution report is:

```text
docs/research/catfish-ee-modqn/03-ee-modqn-validation.execution-report.md
```

## Why This Is Independent

`EE-MODQN` isolates the objective substitution. Without this phase, a later `Catfish-EE-MODQN` improvement cannot be attributed to Catfish versus the EE objective itself.

## Method Boundary

Original MODQN:

```text
R = (throughput, handover penalty, load balance)
```

EE-MODQN:

```text
R = (energy efficiency, handover penalty, load balance)
```

Everything else should remain as close as possible to the original MODQN backbone:

1. same discrete beam-selection action semantics,
2. same three-objective MODQN shape,
3. same non-Catfish replay / target-network concept,
4. same baseline comparison discipline.

## Primary Comparator

The fair comparison for this phase is not old MODQN artifacts versus a new EE-MODQN run. It is a paired experiment:

```text
MODQN-control: R = (throughput, handover penalty, load balance)
EE-MODQN:     R = (energy efficiency, handover penalty, load balance)
```

Both sides must use the same HOBS-linked SINR / power surface, action space, state encoding, seeds, training episodes, evaluation cadence, target-network sync, replay settings, and checkpoint rule. Otherwise, observed differences may come from radio / power-profile changes rather than the EE objective substitution.

## Inputs

1. Phase 01 baseline anchor.
2. Phase 02 EE formula validation.
3. Current MODQN training / evaluation metric set.

## Non-Goals

1. Do not add Catfish.
2. Do not add multi-catfish agents.
3. Do not claim final method improvement.

## Checks

Compare `MODQN` vs `EE-MODQN`:

1. `EE_system`
2. per-user EE reward behavior as credit-assignment diagnostics only
3. raw throughput: total, mean per user, and low-percentile throughput
4. total active beam power in linear W
5. handover count / `r2`, preferably total, per user / episode, and intra / inter split if available
6. load-balance gap / `r3`, including raw gap and reward term
7. outage or below-threshold service ratio, including served / unserved ratio
8. scalar reward behavior under new reward scale
9. best-eval checkpoint behavior, including selection episode, selection criterion, final-vs-best, and full metric panel on the same eval seed set

Scalar reward is only a training / selection diagnostic across different reward definitions. It is not a standalone cross-method victory metric.

## Rescaling Checks

EE-MODQN must be checked for throughput rescaling:

1. Denominator audit: confirm `sum_active_beams P_b(t)` varies with action, active beam state, or power allocation.
2. Correlation / rank check: compare throughput and EE reward by Pearson / Spearman correlation, action ranking, and checkpoint ranking.
3. Replay counterfactual: rescore the same policy or replay set under throughput reward and EE reward; if action choice, best checkpoint, and method ranking do not change, the method cannot claim energy-aware behavior.

If the denominator is fixed or action-independent, EE becomes throughput divided by a constant.

## Reward-Hacking Checks

EE-MODQN must reject apparent improvements caused by service reduction:

1. If `EE_system` rises while raw throughput, low-percentile throughput, or served ratio falls sharply, treat it as suspected reward hacking.
2. If outage or below-threshold ratio rises, do not claim overall improvement.
3. If handover count decreases while throughput collapses or load gap worsens, treat the result as a sticky-policy risk.
4. If per-user EE improves but `EE_system` does not, treat the per-user credit assignment as distorted.
5. If denominator approaches zero, power units mix dBm / dBW / W, or inactive beams are handled inconsistently, discard the result.

## Throughput Guardrail

Throughput should not become a fourth objective. It should be a claim gate:

```text
EE-MODQN is considered improved only if EE_system increases while raw throughput
and outage remain within the pre-declared QoS guardrail relative to MODQN-control.
```

If EE rises only because throughput or service quality drops too much, the result is an EE-throughput tradeoff, not an overall improvement.

## Decision Gate

Promote only if EE-MODQN has a coherent metric surface and does not simply collapse into throughput rescaling or low-service reward hacking.

Stop if the EE objective produces uninterpretable behavior or violates the QoS guardrail.

For now, Phase 03 remains `NEEDS MORE EVIDENCE` until paired runs and checks above exist.

## Expected Output

An EE-MODQN validation report:

1. comparison to original MODQN,
2. effect of objective substitution,
3. reward-scale notes,
4. QoS / service-risk notes,
5. recommendation for Catfish-EE work.

The accepted Phase 03 review is recorded in `reviews/03-ee-modqn-validation.review.md`.
