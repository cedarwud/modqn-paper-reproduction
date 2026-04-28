# Catfish / EE-MODQN Development Guardrails

**Date:** `2026-04-28`  
**Status:** active guardrails for follow-on development  
**Scope:** how to develop EE-MODQN, Catfish-MODQN, Multi-Catfish-MODQN, and Catfish-EE-MODQN inside `modqn-paper-reproduction` without mutating the frozen MODQN baseline.

## Core Rule

Follow-on work may be implemented inside `modqn-paper-reproduction`, but the original MODQN baseline must remain frozen.

The rule is:

```text
baseline frozen; follow-on methods use new configs, new method IDs, and new artifact namespaces
```

This keeps the original comparison surface available while still allowing the project to reuse the same environment, trainer, evaluator, and artifact contracts.

## Frozen Baseline Surface

Do not redefine these as EE-MODQN, Catfish-MODQN, or any follow-on method:

1. `configs/modqn-paper-baseline.yaml`
2. `configs/modqn-paper-baseline.resolved-template.yaml`
3. `artifacts/pilot-02-best-eval/`
4. `artifacts/run-9000/`
5. `artifacts/table-ii-200ep-01/`
6. `artifacts/fig-3-pilot-01/`
7. `artifacts/fig-4-pilot-01/`
8. `artifacts/fig-5-pilot-01/`
9. `artifacts/fig-6-pilot-01/`

`configs/modqn-paper-baseline.yaml` remains authority-only and must not become a direct training config.

## Follow-On Namespaces

Use new config names for follow-on methods, for example:

```text
configs/ee-modqn-*.yaml
configs/catfish-modqn-*.yaml
configs/multi-catfish-modqn-*.yaml
configs/catfish-ee-modqn-*.yaml
```

Use new artifact directories for follow-on runs, for example:

```text
artifacts/ee-modqn-*/
artifacts/catfish-modqn-*/
artifacts/multi-catfish-modqn-*/
artifacts/catfish-ee-modqn-*/
```

Do not write follow-on outputs into frozen baseline artifact directories.

## Shared-Code Changes

Shared code may be extended only if baseline behavior remains selectable and unchanged.

When changing shared trainer, environment, evaluator, export, or artifact code:

1. gate new behavior behind explicit config fields or method IDs,
2. keep baseline defaults equivalent to the frozen MODQN baseline,
3. keep `r1 = throughput`, `r2 = handover penalty`, and `r3 = load balance` for baseline,
4. preserve best-eval and final-checkpoint reporting semantics,
5. preserve baseline metrics: scalar reward, `r1`, `r2`, `r3`, handover count, best-eval, and final-vs-best.

If a change cannot preserve baseline behavior, stop and create a new design review before implementation.

## Method Family Labels

Use explicit labels in configs, metadata, reports, and artifacts:

1. `MODQN-baseline`
2. `EE-MODQN`
3. `Catfish-MODQN`
4. `Multi-Catfish-MODQN`
5. `Catfish-EE-MODQN`

Do not relabel follow-on methods as the original MODQN baseline.

## Phase-Specific Guardrails

Phase `02`:

1. add or verify EE metric / audit surfaces only,
2. do not change the reward,
3. do not train EE-MODQN,
4. do not introduce Catfish.

Phase `03`:

1. change only `r1` from throughput to EE,
2. compare paired `MODQN-control` vs `EE-MODQN`,
3. use the same HOBS-linked SINR / power surface.

Phase `04`:

1. keep original MODQN reward,
2. add only single Catfish replay / intervention mechanics,
3. keep competitive shaping off in the primary run or isolate it as ablation.

Phase `05`:

1. start with `05A` multi-buffer validation,
2. do not jump directly to three Catfish agents,
3. keep total intervention budget equal when comparing against single Catfish.

Phase `06`:

1. compare primarily against `EE-MODQN`,
2. use original MODQN only as context,
3. do not make final claims until Phases `03`, `04`, and `05` produce evidence.

## Forbidden Operations

Do not:

1. edit baseline configs to mean EE or Catfish,
2. overwrite baseline artifacts,
3. make EE or Catfish behavior the implicit default path,
4. use reward-calibration, scenario-corrected, or beam-aware follow-on surfaces as silent baseline replacements,
5. claim full paper-faithful reproduction from this follow-on work,
6. use scalar reward alone as a victory metric,
7. claim energy-aware learning if EE reduces to throughput divided by a fixed constant.

## Minimum Delivery Expectation

Every follow-on implementation handoff should state:

1. which method family it touched,
2. which config namespace it used,
3. which artifact namespace it used,
4. whether baseline behavior remains unchanged,
5. which baseline guardrail or smoke check was run,
6. which phase claim is allowed and which claims remain disallowed.
