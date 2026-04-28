# Catfish / EE-MODQN Execution Handoff

**Date:** `2026-04-28`  
**Status:** handoff for a new plan-controller dialogue  
**Scope:** planning and execution control for the next Catfish / EE-MODQN research step. This document does not authorize code changes by itself.

## Current Conclusion

Phase `01` is complete for the current research track.

The repo can use the existing MODQN surface as a frozen disclosed comparison baseline:

```text
r1 = throughput
r2 = handover penalty
r3 = load balance
```

Phase `02` and `02B` execution are complete for this track. Phase `02`
blocked the static-power runtime as a fixed-power trap; Phase `02B` added the
explicit opt-in `active-load-concave` power surface and promoted it for metric
audit / paired Phase `03` design only.

Phase `03` now has a bounded paired pilot:

1. `MODQN-control`: `r1 = throughput`
2. `EE-MODQN`: `r1 = per-user EE credit-assignment reward`
3. both used the same Phase `02B` HOBS-compatible power surface
4. no Catfish or multi-Catfish was run

The Phase `03` pilot is **not promoted**. Best-eval `EE_system`, throughput,
served ratio, active power, and load-balance metrics tied control; EE-MODQN
worsened handover / `r2`; throughput-vs-EE correlation was approximately `1.0`;
same-policy rescoring did not change method ranking.

Phase `03B` is also **not promoted**. The bounded reward/objective-geometry
follow-up added opt-in fixed-scale reward normalization, shared r3-heavy
load-balance calibration, and an EE r1 credit `R_i(t) / P_b(t)` on the same
Phase `02B` power surface. The config-level denominator still varied, but both
learned best-eval policies collapsed every evaluated step to one active beam,
`denominator_varies_in_eval=false`, `EE_system` tied control, and same-policy
throughput-vs-EE rescoring remained effectively identical. See
`03b-ee-modqn-objective-geometry.execution-report.md`.

Phases `04` to `06` remain evidence-gated:

1. Phase `04` is a separate single-Catfish feasibility branch and should not be
   mixed into Phase `03`.
2. Phase `05` must not jump directly to three Catfish agents.
3. Phase `06` cannot make final Catfish-EE-MODQN claims until Phases `03` to
   `05` produce evidence.

## Authority Files

Read in this order when taking over the plan:

1. `AGENTS.md`
2. `docs/research/catfish-ee-modqn/00-validation-master-plan.md`
3. `docs/research/catfish-ee-modqn/execution-handoff.md`
4. `docs/research/catfish-ee-modqn/development-guardrails.md`
5. `docs/research/catfish-ee-modqn/reviews/01-modqn-baseline-anchor.review.md`
6. `docs/research/catfish-ee-modqn/reviews/02-hobs-ee-formula-validation.review.md`
7. `docs/research/catfish-ee-modqn/reviews/03-ee-modqn-validation.review.md`
8. `docs/research/catfish-ee-modqn/02-hobs-ee-formula-validation.md`
9. `docs/research/catfish-ee-modqn/02-hobs-ee-formula-validation.execution-report.md`
10. `docs/research/catfish-ee-modqn/02b-hobs-power-surface.execution-report.md`
11. `docs/research/catfish-ee-modqn/03-ee-modqn-validation.execution-report.md`
12. `docs/research/catfish-ee-modqn/03b-ee-modqn-objective-geometry.execution-report.md`

Use later phase reviews only as constraints unless the user explicitly asks to plan those phases.

## Phase 01 Status

Phase `01` is complete as a disclosed comparison baseline, not as a full paper-faithful reproduction.

The promoted baseline anchors are:

1. `configs/modqn-paper-baseline.resolved-template.yaml`
2. `configs/modqn-paper-baseline.yaml` as authority-only, not as a direct training config
3. `artifacts/pilot-02-best-eval/`
4. `artifacts/run-9000/`
5. `artifacts/table-ii-200ep-01/`
6. `artifacts/fig-3-pilot-01/`
7. `artifacts/fig-4-pilot-01/`
8. `artifacts/fig-5-pilot-01/`
9. `artifacts/fig-6-pilot-01/`

Do not silently replace this baseline with reward-calibration, scenario-corrected, or beam-aware follow-on surfaces.

## Repository Development Guardrails

Follow-on work should stay in `modqn-paper-reproduction` unless the user explicitly asks for a separate repo, but the frozen baseline must remain intact.

Required rules:

1. do not edit baseline configs to carry EE or Catfish semantics,
2. do not overwrite baseline artifacts,
3. give every follow-on method a distinct config namespace,
4. give every follow-on run a distinct artifact namespace,
5. keep shared-code changes behind explicit method / config selection,
6. preserve baseline rerun behavior and baseline reporting semantics.

The detailed version is `development-guardrails.md`.

## Phase 02 / 02B Status

Phase `02` established the EE formula target:

```text
EE_system(t) = sum_i R_i(t) / sum_active_beams P_b(t)
```

Required semantics:

1. `P_b(t)` is the same downlink beam transmit power used by the SINR numerator semantics.
2. `P_b(t)` is a controlled or allocated beam power variable.
3. `P_b(t)` is in linear W.
4. Inactive beams have zero power or are excluded from `active_beams`.
5. `P_b(t)` must not be replaced by a path-loss closed form.
6. `P_b(t)` must not be a fixed config-power constant.

Per-user EE, if used later, is only credit assignment:

```text
r1_i(t) = R_i(t) / (P_b(t) / N_b(t))
```

It is not the system-level EE formula and must be labeled as a modeling assumption.

Phase `02B` added a disclosed opt-in power surface:

```text
P_b(t) = 0 W, if N_b(t) = 0
P_b(t) = min(max_power_w,
             active_base_power_w + load_scale_power_w * N_b(t)^load_exponent),
         if N_b(t) > 0
```

Use only `configs/ee-modqn-power-surface-phase-02b.resolved.yaml` or configs
that inherit it for Phase `03` paired EE work.

## Phase 03 Status

Phase `03` execution artifacts:

1. `configs/ee-modqn-phase-03-control-pilot.resolved.yaml`
2. `configs/ee-modqn-phase-03-ee-pilot.resolved.yaml`
3. `artifacts/ee-modqn-phase-03-control-pilot/`
4. `artifacts/ee-modqn-phase-03-ee-pilot/`
5. `artifacts/ee-modqn-phase-03-ee-pilot/paired-comparison-vs-control/`

Phase `03B` execution artifacts:

1. `configs/ee-modqn-phase-03b-control-objective-geometry.resolved.yaml`
2. `configs/ee-modqn-phase-03b-ee-objective-geometry.resolved.yaml`
3. `artifacts/ee-modqn-phase-03b-control-objective-geometry-pilot/`
4. `artifacts/ee-modqn-phase-03b-ee-objective-geometry-pilot/`
5. `artifacts/ee-modqn-phase-03b-ee-objective-geometry-pilot/paired-comparison-vs-control/`
6. `artifacts/ee-modqn-phase-03b-policy-diagnostics/`

Decision:

```text
NEEDS MORE EVIDENCE
```

Do not claim EE-MODQN is effective from this pilot.
Do not claim that Phase `03B` reward geometry solved denominator collapse.

## What Not To Do Next

Do not:

1. claim EE-MODQN effectiveness from the bounded Phase `03` pilot,
2. start Catfish-MODQN while interpreting Phase `03`,
3. create three Catfish agents,
4. compare final Catfish-EE-MODQN directly against original MODQN as the sole proof,
5. claim full paper-faithful reproduction,
6. use scalar reward alone as the comparison result,
7. write follow-on outputs into frozen baseline artifact directories,
8. change baseline configs to mean EE-MODQN or Catfish-MODQN.

## Recommended Next Prompt

Use `prompts/00-new-dialogue-controller.prompt.md` to start a new controller dialogue.
