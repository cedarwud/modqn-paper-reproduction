# Review: Phase 03 EE-MODQN Validation

**Date:** `2026-04-28`  
**Decision:** `NEEDS MORE EVIDENCE`  
**Scope:** EE-MODQN validation design only; this review does not promote EE-MODQN effectiveness.

## Decision Summary

Phase 03 should isolate one change:

```text
r1: throughput -> energy efficiency
```

The original MODQN backbone should otherwise stay fixed. Because Phase 02 promoted the HOBS-linked active-transmit-power EE formula, EE-MODQN can be tested. However, no EE-MODQN improvement claim should be made until paired evidence exists.

## Primary Comparator

The fair comparator is:

```text
MODQN-control: R = (throughput, handover penalty, load balance)
EE-MODQN:     R = (energy efficiency, handover penalty, load balance)
```

Both runs must share the same HOBS-linked SINR / power surface, action space, state encoding, seeds, training episodes, evaluation cadence, target-network sync, replay settings, and checkpoint rule.

Old MODQN artifacts can provide context and anchor expectations, but they cannot by themselves establish the effect of the EE objective substitution.

## Required Metrics

Later reports must include:

1. `EE_system = sum throughput / sum active beam power`
2. per-user EE, only as credit-assignment diagnostics
3. raw throughput: total, mean per user, and low-percentile throughput
4. total active beam power in linear W, with active-TX denominator separated from any broader total-power proxy
5. handover count: total, per user / episode, and intra / inter split if available
6. load-balance gap: raw gap and reward term
7. outage / below-threshold ratio, including served / unserved ratio
8. scalar reward, only as training / selection diagnostics
9. best-eval checkpoint: selection episode, selection criterion, final-vs-best comparison, and a full metric panel on the same eval seed set

## Rescaling Checks

EE-MODQN must prove it is not just throughput rescaling:

1. Denominator audit: verify `sum_active_beams P_b(t)` changes with action, active beam state, or power allocation.
2. Correlation / rank check: compare throughput and EE reward by Pearson / Spearman correlation, action ranking, and checkpoint ranking.
3. Replay counterfactual: rescore the same policies or replay data under throughput reward and EE reward. If action choice, best checkpoint, and method ranking stay the same, do not claim energy-aware behavior.

## Reward-Hacking Checks

Reject or qualify results where:

1. EE rises because fewer users are served.
2. raw throughput, low-percentile throughput, or served ratio falls sharply.
3. outage or below-threshold ratio rises.
4. handover count falls only because the policy becomes sticky while throughput or load balance worsens.
5. per-user EE improves but `EE_system` does not.
6. denominator values approach zero, use mixed dBm / dBW / W units, or handle inactive beams inconsistently.

## Throughput Guardrail

Throughput should be a claim gate, not a fourth objective:

```text
EE-MODQN is considered improved only if EE_system increases while raw throughput
and outage remain within the pre-declared QoS guardrail relative to MODQN-control.
```

If EE improves while throughput drops too much, the correct claim is an EE-throughput tradeoff, not an overall improvement.

## Allowed Claims

1. The phase can claim the effect of replacing the first MODQN objective under a fixed MODQN backbone, once paired runs exist.
2. If the guardrail passes, it can claim `EE_system` improvement under throughput / QoS constraints.
3. It can claim objective-substitution tradeoffs.

## Disallowed Claims

1. Do not claim full paper-faithful reproduction.
2. Do not claim absolute physical energy saving without denominator assumptions and sensitivity analysis.
3. Do not use scalar reward alone to declare a winner.
4. Do not treat per-user EE as system EE.
5. Do not claim energy-aware learning if the denominator is fixed or action-independent.

## Evidence Required Before Promotion

1. Paired `MODQN-control` vs `EE-MODQN` runs.
2. Denominator variability audit.
3. Throughput-vs-EE rescaling checks.
4. Replay counterfactual results.
5. Reward-hacking and QoS guardrail results.

## Result

`NEEDS MORE EVIDENCE`.

The validation protocol is usable, but EE-MODQN effectiveness cannot be promoted yet.
