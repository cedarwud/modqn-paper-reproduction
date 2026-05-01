# EE-MODQN Anti-Collapse Controller Plan

**Date:** `2026-05-01`
**Status:** controller handoff plan for the next EE-MODQN research gate
**Scope:** planning, delegation, result intake, and claim-boundary control only.
The controller does not implement code, run training, or tune experiments.

## Why This Exists

The latest HOBS active-TX EE route resolved one old concern but exposed the next
hard blocker:

```text
HOBS active-TX EE formula / reward wiring: PASS, scoped
HOBS-inspired DPC sidecar denominator gate: PASS
Route D tiny learned-policy denominator check: BLOCK
hard stop: all_evaluated_steps_one_active_beam = true
```

The denominator can vary and same-policy throughput-vs-EE ranking can separate.
The current failure is learned beam-selection collapse: the greedy policy still
uses exactly one active beam across evaluated steps.

The next useful work is therefore not more Route `D` training and not Catfish.
It is a bounded anti-collapse / capacity / assignment design gate.

## Controller Role

The controller dialogue is responsible for:

1. keeping the evidence boundary current,
2. deciding which bounded gate should be opened next,
3. writing or approving the execution prompt for a separate implementation
   dialogue,
4. reading the execution report returned by that dialogue,
5. deciding `PASS`, `BLOCK`, or `NEEDS MORE DESIGN`,
6. updating handoff / master-plan docs after each result.

The controller must not:

1. write code,
2. run training,
3. tune hyperparameters interactively,
4. reopen Phase `03C`,
5. use Catfish as EE repair,
6. claim EE-MODQN effectiveness from denominator variability alone.

## Current Authority Files

Read in this order:

```text
AGENTS.md
docs/research/catfish-ee-modqn/00-validation-master-plan.md
docs/research/catfish-ee-modqn/execution-handoff.md
docs/research/catfish-ee-modqn/development-guardrails.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-modqn-feasibility.execution-report.md
docs/research/catfish-ee-modqn/repository-curation-2026-05-01.md
docs/research/catfish-ee-modqn/energy-efficient/README.md
docs/research/catfish-ee-modqn/energy-efficient/ee-formula-final-review-with-codex-2026-05-01.md
docs/research/catfish-ee-modqn/energy-efficient/modqn-r1-to-hobs-active-tx-ee-design-2026-05-01.md
docs/ee-report.md
```

Supporting historical evidence:

```text
docs/research/catfish-ee-modqn/03d-ee-route-disposition.execution-report.md
docs/research/catfish-ee-modqn/03c-c-power-mdp-pilot.execution-report.md
docs/research/catfish-ee-modqn/ra-ee-07-constrained-power-allocator-distillation.execution-report.md
docs/research/catfish-ee-modqn/ra-ee-09-fixed-association-rb-bandwidth.execution-report.md
```

## Next Gate Candidate

Open one bounded design gate:

```text
EE-MODQN anti-collapse / capacity / assignment gate
```

The gate must answer:

```text
Can a minimal explicit anti-collapse mechanism prevent learned greedy
one-active-beam collapse while preserving the HOBS active-TX EE / DPC boundary?
```

Acceptable mechanism families for the first design review:

1. capacity-aware action masking,
2. overload penalty tied to per-beam user count or served ratio,
3. active-beam diversity / load-spread constraint,
4. centralized assignment constraint,
5. renamed resource-allocation MDP with explicit resource actions.

The controller should pick the smallest auditable candidate. Do not combine
several mechanisms in the first execution prompt unless the design review
justifies the coupling.

## Required Matched Boundary

Any execution agent must preserve:

1. frozen original MODQN baseline semantics,
2. new opt-in config / artifact namespace,
3. same seeds for control and candidate,
4. same episode budget,
5. same evaluation schedule,
6. same checkpoint protocol,
7. same DPC sidecar parameters unless the gate explicitly tests DPC sensitivity,
8. same HOBS active-TX EE formula,
9. no Catfish,
10. no RA-EE association route,
11. no Phase `03C` continuation.

## Primary Acceptance Criteria

The first anti-collapse pilot can only pass if all are true:

1. `all_evaluated_steps_one_active_beam = false`,
2. active-beam count has more than one value or has a declared non-degenerate
   multi-beam target distribution,
3. `denominator_varies_in_eval = true`,
4. `active_power_single_point_distribution = false`,
5. throughput-vs-EE ranking separation remains present or is explicitly
   explained,
6. p05 throughput does not collapse relative to matched control,
7. served ratio does not decrease,
8. outage ratio does not increase,
9. handover / `r2` regression is within a predeclared tolerance,
10. scalar reward alone is not used as success evidence.

## Stop Conditions

Stop the route if any of these occur:

1. candidate still has `all_evaluated_steps_one_active_beam = true`,
2. anti-collapse only works by forcing no service or severe p05 throughput
   collapse,
3. gains appear only in scalar reward,
4. implementation mutates frozen baseline behavior,
5. mechanism requires Catfish,
6. mechanism reopens RA-EE learned association or Phase `03C`,
7. diagnostics cannot prove matched boundary.

## Result Intake Protocol

When an execution dialogue returns:

1. read changed files and artifact paths,
2. verify tests / checks reported,
3. compare reported metrics against the acceptance criteria,
4. classify the result as `PASS`, `BLOCK`, or `NEEDS MORE DESIGN`,
5. update:
   - `00-validation-master-plan.md`,
   - `execution-handoff.md`,
   - `docs/ee-report.md`,
   - a new execution report if one was not already produced,
6. write the next execution prompt only if the previous gate justifies it.

## Claim Boundary

Allowed current claim:

```text
HOBS active-TX EE can be wired into MODQN and the active transmit-power
denominator can vary under a HOBS-inspired DPC sidecar, but the current learned
policy route remains blocked by one-active-beam collapse.
```

Forbidden:

1. EE-MODQN effectiveness,
2. physical energy saving,
3. HOBS optimizer reproduction,
4. Catfish-EE repair,
5. full RA-EE-MODQN,
6. learned association effectiveness,
7. scalar reward as success evidence,
8. "more training will fix Route D" without a new anti-collapse mechanism.
