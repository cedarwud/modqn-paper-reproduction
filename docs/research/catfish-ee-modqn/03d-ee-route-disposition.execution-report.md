# Phase 03D Execution Report: EE Route Disposition

**Date:** `2026-04-28`
**Status:** `BLOCKED / STOP current EE-MODQN r1-substitution route`
**Scope:** read-only disposition. No code changes, document changes, training,
Catfish, multi-Catfish, or frozen baseline mutation were performed during the
disposition review.

## Current Evidence

Phase `02B` made the system EE formula auditable:

```text
EE_system(t) = sum_i R_i(t) / sum_active_beams P_b(t)
```

The Phase `02B` `P_b(t)` surface is an opt-in `active-load-concave`
synthesized proxy. It is useful for audit and bounded follow-on experiments,
but it is not a HOBS optimizer and must not be labeled as paper-backed power
allocation.

Phase `03`, `03A`, and `03B` did not promote EE-MODQN. The repeated failure
mode was consistent: evaluated policies collapsed to one active beam,
`denominator_varies_in_eval=false`, throughput-vs-EE correlation stayed near
`1.0`, and same-policy rescoring did not change ranking.

Phase `03C-B` passed only as a static / counterfactual precondition. It showed
that an explicit power-decision surface can change the denominator and separate
same-policy throughput-vs-EE ranking. This was not learned runtime evidence.

Phase `03C-C` then ran the bounded paired runtime pilot and blocked the route.
The candidate runtime selector collapsed to one selected profile (`fixed-low`),
kept `candidate_denominator_varies_in_eval=false`, kept active power at a
single-point `0.5 W` distribution, and had one active beam on every evaluated
step. The tiny `EE_system` increase was paired with a large p05-throughput
collapse, so it is not acceptable energy-aware learning evidence.

## Disposition Verdict

The current EE-MODQN route is blocked:

```text
current EE-MODQN r1-substitution route: BLOCKED / STOP
Phase 03C continuation by small tuning: FAIL
Catfish as an EE repair path: BLOCKED
new resource-allocation EE-MDP design gate: PASS as a new route only
```

This does not invalidate the EE formula. It invalidates the current method
boundary where the original MODQN handover MDP is kept and only `r1` is
replaced with an EE-oriented reward or a small power-codebook selector.

## Why Phase 03C-C Blocks The Current Route

Phase `03C-C` demonstrated that the Phase `03C-B` static ranking separation did
not survive runtime evaluation. The selector was available, but the evaluated
policy still collapsed the environment into a one-active-beam state. Once the
policy did that, the selector had no meaningful denominator-sensitive decision
left to make and chose a single low-power profile on every evaluated step.

That means the evaluated surface still behaves like throughput divided by an
effectively fixed denominator. The apparent scalar-reward and tiny EE gains
are not promotable because the denominator gate, ranking-separation gate, and
throughput-tail guardrail all failed.

## Options Considered

Continuing Phase `03C` with more episodes, another selector tweak, or another
reward scale is rejected. Those changes would repeat already triggered stop
conditions without changing the structural action / denominator coupling.

Terminating all EE research is not required. Phase `02B` and Phase `03C-B`
showed that the metric and explicit power-decision surface can be scientifically
useful when treated as a resource-allocation problem.

Continuing EE as a new resource-allocation MDP is the only acceptable route.
That route must be renamed and scoped as a new method family, not as a small
Phase `03` patch over the current MODQN handover MDP.

## Recommended Next Research Route

If EE remains a project goal, open a new design gate for a resource-allocation
EE-MDP family such as:

```text
Resource-Allocation EE-MDP
RA-EE-MODQN
Hierarchical EE Resource Allocation
```

The minimum defensible design should include:

1. a centralized scheduler or allocator for beam/user/resource conflict,
2. multi-beam association that prevents evaluated one-active-beam collapse,
3. explicit power allocation where `P_b(t)` is an action or controller output,
4. at least coarse bandwidth / RB allocation or an equivalent bounded
   resource allocator,
5. constrained-MDP style guardrails for total power, per-beam power, served
   ratio, outage, p05 throughput, handover, and load balance,
6. a hierarchical boundary if MODQN handover is retained as only one layer.

Any such design must be labeled as a new extension unless the source papers
explicitly support the exact action, state, and allocator semantics.

## Catfish Status

Catfish should not enter the EE repair route. Catfish is a replay /
intervention training strategy; it does not by itself supply the missing
resource-allocation action surface or make `P_b(t)` policy-sensitive.

If Catfish work proceeds next, it should be a separate Phase `04`
single-Catfish feasibility branch using the original MODQN objective:

```text
r1 = throughput
r2 = handover penalty
r3 = load balance
```

No EE claim should be attached to Phase `04`.

## Phase 05 / Phase 06 Status

Phase `05` remains blocked. It must not jump directly to three Catfish agents;
at most, a later `05A` multi-buffer validation may be considered after Phase
`04` produces evidence.

Phase `06` remains blocked. There is no promoted EE-MODQN, no promoted
Catfish-MODQN, and no promoted multi-Catfish evidence. If the EE route is
renamed into a new resource-allocation MDP, the final comparison design must be
rewritten for that method family.

## Stop Conditions

The following stop conditions are already triggered for the current route:

1. `denominator_varies_in_eval=false`,
2. evaluated one-active-beam ratio is `1.0`,
3. selected power profile is a single-point distribution,
4. active power is a single-point distribution,
5. throughput-vs-EE ranking does not separate,
6. apparent EE gain is paired with p05-throughput collapse,
7. scalar reward improves but cannot serve as success evidence.

Any future EE design that reproduces these failures should stop rather than
extend training duration.

## Forbidden Claims

Do not claim:

1. EE-MODQN effectiveness,
2. Phase `03C-C` solved denominator collapse,
3. Phase `03C-B` is learned EE evidence,
4. Phase `02B` `P_b(t)` is a HOBS optimizer,
5. per-user EE credit is system EE,
6. scalar reward alone proves success,
7. Catfish solves EE collapse,
8. Catfish / Multi-Catfish / Catfish-EE-MODQN final effectiveness,
9. full paper-faithful reproduction,
10. absolute physical energy saving,
11. final EE-method success based only on comparison with original MODQN.

## Final Decision

```text
FAIL / BLOCKED for the current EE-MODQN r1-substitution route.
PASS only to reopen EE as a new resource-allocation MDP design gate.
PASS to Phase 04 Catfish feasibility only as a separate original-MODQN branch.
```
