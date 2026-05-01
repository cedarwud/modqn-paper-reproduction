# Catfish / EE-MODQN Execution Handoff

**Date:** `2026-05-01`
**Status:** handoff after HOBS active-TX EE Route D learned-policy denominator-check block
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

Phase `03C-B` **passes only to a bounded paired pilot**. It added an opt-in
hierarchical handover plus centralized discrete power-codebook audit surface
under `configs/ee-modqn-phase-03c-b-power-mdp-*` and
`artifacts/ee-modqn-phase-03c-b-power-mdp-*`. The static/counterfactual audit
fixed the same beam trajectories and replayed `fixed-2w`, Phase `02B` proxy,
and Phase `03C-B` codebook candidates. It proved that power decisions can
change the denominator and that same-policy throughput-vs-EE ranking can
separate. This is not learned EE-MODQN effectiveness evidence, and the
controller is a new-extension / HOBS-inspired surface, not a HOBS optimizer.
See `03c-b-power-mdp-audit.execution-report.md`.

Phase `03C-C` **is blocked**. It ran the bounded paired pilot authorized by
Phase `03C-B` under `configs/ee-modqn-phase-03c-c-power-mdp-*` and
`artifacts/ee-modqn-phase-03c-c-power-mdp-*`. The candidate used the explicit
runtime `runtime-ee-selector`, but best-eval still collapsed every evaluated
step to one active beam. The selected power profile distribution was a single
point (`fixed-low`), active power was fixed at `0.5 W`,
`denominator_varies_in_eval=false`, throughput-vs-EE ranking did not separate,
and the tiny `EE_system` increase coincided with a `50%` p05 throughput drop.
This is runtime/eval evidence against promoting the current EE route. See
`03c-c-power-mdp-pilot.execution-report.md`.

Phase `03D` **stops the current EE-MODQN route**. The disposition concludes
that the current MODQN handover MDP plus `r1` substitution is blocked and
should not continue with more episodes, selector tweaks, reward tuning, or
Catfish. The EE formula remains valid as a metric, but any renewed EE research
must be opened as a new resource-allocation MDP design gate with a renamed
method family. See `03d-ee-route-disposition.execution-report.md`.

The post-`03D` HOBS active-TX EE feasibility chain has now also run. It should
be treated as a new scoped feasibility check, not as a reversal of Phase `03D`.

```text
HOBS active-TX EE formula / reward wiring: PASS, scoped
SINR structural audit: PASS, but negligible at current MODQN operating point
channel-regime / antenna-gain path: BLOCK as a paper-backed MODQN continuation
HOBS-inspired DPC sidecar denominator gate: PASS
Route D tiny learned-policy denominator check: BLOCK
EE-MODQN effectiveness: NOT PROMOTED / BLOCKED
```

The key update is precise: the denominator can now vary under the opt-in
HOBS-inspired DPC sidecar, and same-policy throughput-vs-EE ranking can
separate. The hard stop is now learned beam-selection collapse. In Route `D`,
both matched control and candidate evaluated all `50` greedy steps with one
active beam (`all_evaluated_steps_one_active_beam=true`). Candidate scalar
reward increased only because `r1` was rescored as HOBS active-TX EE; throughput
metrics were identical under the tiny matched check. Do not scale this route
with more episodes by default. See
`hobs-active-tx-ee-modqn-feasibility.execution-report.md`.

The RA-EE follow-on is now also closed at the current evidence boundary:

```text
old EE-MODQN r1-substitution route: BLOCKED / STOP
RA-EE fixed-association deployable power allocation: PASS, scoped
RA-EE learned association / hierarchical RL / full RA-EE-MODQN: BLOCKED
Catfish for EE repair: BLOCKED
```

RA-EE-04 and RA-EE-05 established a bounded fixed-association centralized power
allocator and held-out robustness result. RA-EE-07 then established a stronger
deployable non-oracle power allocator under fixed association. That is the only
positive RA-EE claim boundary.

RA-EE-06, RA-EE-06B, and RA-EE-08 block the association path. The decisive
RA-EE-08 comparison used matched fixed association + deployable stronger power
allocator versus proposal association + the same deployable stronger power
allocator. No proposal family had positive held-out `EE_system` delta, the
predeclared primary proposal was also negative, and diagnostic oracle rows did
not leave a meaningful deployable association gap. Do not reopen Phase `03C` or
continue association proposal refinement as the next step.

Phases `04` to `06` remain evidence-gated:

1. Phase `04` is parked unless the explicit goal is Catfish feasibility. It is
   a separate single-Catfish branch over the original MODQN reward and should
   not be mixed into Phase `03` or RA-EE. Phase `04-B` runnable evidence and
   Phase `04C` bounded attribution are complete, but Catfish-MODQN effectiveness
   is not promoted.
2. Phase `05` must not jump directly to three Catfish agents. Phase `05A`
   multi-buffer validation is complete and blocks escalation: objective-specific
   buffers were not meaningfully distinct, so Phase `05B` and full
   multi-Catfish agents remain blocked. The only allowed continuation is
   Phase `05R`, a planning / diagnostic-only objective-buffer-redesign gate.
   Phase `05R` has since passed, Phase `05B` planning passed, and the bounded
   Phase `05B` pilot has now completed. Runnable evidence passed, but
   acceptance / effectiveness failed. Phase `05C` closes the current
   Multi-Catfish route as not promoted.
3. Phase `06` cannot make final Catfish-EE-MODQN claims. The current evidence
   has no promoted EE-MODQN route and no promoted Multi-Catfish route.

RA-EE-09 is recorded in `ra-ee-09-completion-design-gate.md` as the design
history for a concrete fixed-association RB / bandwidth allocation pilot. That
design did not reopen association proposal refinement and did not by itself
authorize an effectiveness claim.

RA-EE-09 Slice `09A` through `09E` have since been implemented and evaluated as
offline fixed-association replay only. The equal-share control reproduced the
existing throughput formula, the bounded QoS-slack candidate kept association,
handover, the RA-EE-07 effective power schedule, and resource accounting
matched. The held-out effectiveness gate failed: candidate `EE_system` delta
was `-46.64859074452477`, the predeclared resource-efficiency delta was
`-0.1944519284254136`, and p05 throughput ratio was
`0.9016412169223311 < 0.95`. RA-EE-09 is therefore an auditable negative result
for the tested RB / bandwidth candidate, not a promoted resource-allocation
method.

Phase `07A` read-only recovery gate is complete:

```text
Phase 07A read-only recovery gate: PASS
Phase 07B implementation/training from Phase 07A alone: NOT AUTHORIZED
Direct Phase 05B continuation: BLOCK
Multi-Catfish promotion: BLOCK
Phase 06 / Catfish-EE-MODQN: BLOCK
```

The Phase `05B` failure model is not implementation failure as the primary
explanation and not simple budget insufficiency / longer training. The strongest
working explanation is that intervention utility was not proven beyond
equal-budget controls; replay starvation is an observed confounder and stop
trigger. The next R&D route is single-Catfish intervention utility / causal
diagnostics, not Multi-Catfish tuning. See
`07a-catfish-recovery-gate.execution-report.md`.

Phase `07B` has since completed as a separately authorized bounded
single-Catfish intervention-utility pilot, and Phase `07C` has recorded the
post-07B disposition:

```text
Phase 07B bounded single-Catfish utility evidence: PASS
Broader Catfish-MODQN effectiveness promotion: NOT PROMOTED
Multi-Catfish reopening: BLOCKED / DEFERRED
Phase 06 / Catfish-EE-MODQN: BLOCK
```

The primary shaping-off single-Catfish branch beat matched MODQN,
no-intervention, random / equal-budget injection, and replay-only single learner
under the bounded protocol, but `r2` / handovers worsened and asymmetric gamma
was not supported as an active mechanism. See
`07c-catfish-intervention-utility-disposition.execution-report.md` and
`artifacts/catfish-modqn-phase-07b-bounded-pilot-summary/phase07b_bounded_pilot_summary.json`.

Phase `07D` then completed the r2 / handover-guarded single-Catfish robustness
pilot:

```text
Phase 07D bounded implementation / runs / diagnostics: PASS
Phase 07D acceptance / recovery promotion: FAIL
Broader Catfish-MODQN effectiveness: NOT PROMOTED
Multi-Catfish reopening: BLOCKED
Phase 06 / Catfish-EE-MODQN: BLOCKED
```

The primary guarded branch still beat matched MODQN, no-intervention,
random / equal-budget, and replay-only on scalar, but the predeclared
non-inferiority gate failed versus matched MODQN: `r2_delta=-0.051667 < -0.02`
and `handover_delta=+10.333333 > +5`. The summary also reports
`starvation_stop_trigger_absent=false`. See
`07d-r2-guarded-single-catfish-robustness.execution-report.md` and
`artifacts/catfish-modqn-phase-07d-r2-guarded-robustness-summary/phase07d_r2_guarded_robustness_summary.json`.

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
13. `docs/research/catfish-ee-modqn/03c-b-power-mdp-audit.execution-report.md`
14. `docs/research/catfish-ee-modqn/03c-c-power-mdp-pilot.execution-report.md`
15. `docs/research/catfish-ee-modqn/03d-ee-route-disposition.execution-report.md`
16. `docs/research/catfish-ee-modqn/04c-single-catfish-ablation-attribution.execution-report.md`
17. `docs/research/catfish-ee-modqn/05a-multi-buffer-validation.execution-report.md`
18. `docs/research/catfish-ee-modqn/05r-objective-buffer-redesign-gate.execution-report.md`
19. `docs/research/catfish-ee-modqn/prompts/05b-multi-catfish-planning.prompt.md`
20. `docs/research/catfish-ee-modqn/05b-multi-catfish-planning.execution-report.md`
21. `docs/research/catfish-ee-modqn/prompts/05b-multi-catfish-implementation-draft.prompt.md`
22. `docs/research/catfish-ee-modqn/05b-multi-catfish-bounded-pilot.execution-report.md`
23. `docs/research/catfish-ee-modqn/05c-multi-catfish-route-disposition.execution-report.md`
24. `docs/research/catfish-ee-modqn/07a-catfish-recovery-gate.execution-report.md`
25. `docs/research/catfish-ee-modqn/07c-catfish-intervention-utility-disposition.execution-report.md`
26. `artifacts/catfish-modqn-phase-07b-bounded-pilot-summary/phase07b_bounded_pilot_summary.json`
27. `docs/research/catfish-ee-modqn/07d-r2-guarded-single-catfish-robustness.execution-report.md`
28. `artifacts/catfish-modqn-phase-07d-r2-guarded-robustness-summary/phase07d_r2_guarded_robustness_summary.json`
29. `docs/research/catfish-ee-modqn/ra-ee-02-oracle-power-allocation-audit.execution-report.md`
30. `docs/research/catfish-ee-modqn/ra-ee-04-bounded-power-allocator-pilot.execution-report.md`
31. `docs/research/catfish-ee-modqn/ra-ee-05-fixed-association-robustness.execution-report.md`
32. `docs/research/catfish-ee-modqn/ra-ee-06-association-counterfactual-oracle.execution-report.md`
33. `docs/research/catfish-ee-modqn/ra-ee-06b-association-proposal-refinement.execution-report.md`
34. `docs/research/catfish-ee-modqn/ra-ee-07-constrained-power-allocator-distillation.execution-report.md`
35. `docs/research/catfish-ee-modqn/ra-ee-08-offline-association-reevaluation.execution-report.md`
36. `docs/research/catfish-ee-modqn/ra-ee-09-completion-design-gate.md`
37. `docs/research/catfish-ee-modqn/ra-ee-09-fixed-association-rb-bandwidth.execution-report.md`
38. `docs/research/catfish-ee-modqn/hobs-active-tx-ee-modqn-feasibility.execution-report.md`
39. `docs/research/catfish-ee-modqn/ee-modqn-anti-collapse-controller-plan-2026-05-01.md`
40. `docs/research/catfish-ee-modqn/prompts/ee-modqn-anti-collapse-controller.prompt.md`
41. `docs/research/catfish-ee-modqn/prompts/ee-modqn-anti-collapse-worker.prompt.md`
42. `docs/research/catfish-ee-modqn/energy-efficient/README.md`
43. `docs/research/catfish-ee-modqn/energy-efficient/ee-formula-final-review-with-codex-2026-05-01.md`
44. `docs/research/catfish-ee-modqn/energy-efficient/modqn-r1-to-hobs-active-tx-ee-design-2026-05-01.md`
45. `docs/research/catfish-ee-modqn/repository-curation-2026-05-01.md`
46. `docs/ee-report.md`

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

Phase `03C-B` static power-MDP audit artifacts:

1. `configs/ee-modqn-phase-03c-b-power-mdp-audit.resolved.yaml`
2. `artifacts/ee-modqn-phase-03c-b-power-mdp-audit/`
3. `docs/research/catfish-ee-modqn/03c-b-power-mdp-audit.execution-report.md`

Phase `03C-C` bounded paired pilot artifacts:

1. `configs/ee-modqn-phase-03c-c-power-mdp-control.resolved.yaml`
2. `configs/ee-modqn-phase-03c-c-power-mdp-candidate.resolved.yaml`
3. `artifacts/ee-modqn-phase-03c-c-power-mdp-control-pilot/`
4. `artifacts/ee-modqn-phase-03c-c-power-mdp-candidate-pilot/`
5. `artifacts/ee-modqn-phase-03c-c-power-mdp-candidate-pilot/paired-comparison-vs-control/`
6. `docs/research/catfish-ee-modqn/03c-c-power-mdp-pilot.execution-report.md`

Decision:

```text
BLOCKED for Phase 03C-C; Phase 03D stops the current EE-MODQN route
```

Do not claim EE-MODQN is effective from this pilot.
Do not claim that Phase `03B` reward geometry solved denominator collapse.
Do not claim that the Phase `03C-C` runtime selector solved denominator
collapse. The current route remains blocked until a design correction can make
evaluation exercise a non-single-point active-power distribution without
throughput/service collapse. After Phase `03D`, that design correction is no
longer treated as a Phase `03C` patch; it must be opened as a new
resource-allocation MDP method family.

## RA-EE Closeout Status

RA-EE execution artifacts:

1. `configs/ra-ee-02-oracle-power-allocation-audit.resolved.yaml`
2. `configs/ra-ee-04-bounded-power-allocator-control.resolved.yaml`
3. `configs/ra-ee-04-bounded-power-allocator-candidate.resolved.yaml`
4. `configs/ra-ee-05-fixed-association-robustness.resolved.yaml`
5. `configs/ra-ee-06-association-counterfactual-oracle.resolved.yaml`
6. `configs/ra-ee-06b-association-proposal-refinement.resolved.yaml`
7. `configs/ra-ee-07-constrained-power-allocator-distillation.resolved.yaml`
8. `configs/ra-ee-08-offline-association-reevaluation.resolved.yaml`
9. `configs/ra-ee-09-fixed-association-rb-bandwidth-control.resolved.yaml`

RA-EE artifact roots:

1. `artifacts/ra-ee-02-oracle-power-allocation-audit/`
2. `artifacts/ra-ee-04-bounded-power-allocator-control-pilot/`
3. `artifacts/ra-ee-04-bounded-power-allocator-candidate-pilot/`
4. `artifacts/ra-ee-05-fixed-association-robustness/`
5. `artifacts/ra-ee-06-association-counterfactual-oracle/`
6. `artifacts/ra-ee-06b-association-proposal-refinement/`
7. `artifacts/ra-ee-07-constrained-power-allocator-distillation/`
8. `artifacts/ra-ee-08-offline-association-reevaluation/`
9. `artifacts/ra-ee-09-fixed-association-rb-bandwidth-control-pilot/`
10. `artifacts/ra-ee-09-fixed-association-rb-bandwidth-candidate-pilot/`
11. `artifacts/ra-ee-09-fixed-association-rb-bandwidth-candidate-pilot/paired-comparison-vs-control/`

Scoped allowed RA-EE claim:

```text
Under the disclosed simulation setting and fixed-association held-out replay,
the RA-EE-07 deployable non-oracle finite-codebook power allocator improves
simulated system EE over the matched fixed-association RA-EE-04/05 safe-greedy
power allocator while preserving the declared QoS and power guardrails.
```

This is not a learned policy claim, a physical energy-saving claim, a HOBS
optimizer claim, or a full RA-EE-MODQN claim. It should not be rewritten as
`same throughput with less physical power` unless the local table separately
reports the exact throughput parity / delta.

RA-EE-07 paper-facing details:

```text
finite codebook levels: {0.5, 0.75, 1.0, 1.5, 2.0} W
per-beam cap: 2.0 W
total active-beam budget: 8.0 W
inactive-beam policy: 0 W
held-out trajectory positives: 5 / 5
held-out seed positives: 4 / 5
max positive trajectory delta share: 0.2560081286323895
max positive seed delta share: 0.28553406270120996
```

The deterministic hybrid selected `bounded-local-search-codebook` on every
held-out step in the RA-EE-07 evaluation. Treat that as an observed held-out
behavior, not as permission to collapse the method label to generic
bounded-local-search.

Allowed RA-EE-09 negative result claim:

```text
Under fixed association and matched RA-EE-07 deployable power allocation, the
tested bounded normalized bandwidth/resource-share allocator was auditable but
did not improve held-out simulated EE or the predeclared QoS-preserving
resource-efficiency metric versus equal-share control.
```

This is not a positive RB / bandwidth allocation effectiveness claim.

Forbidden RA-EE claims:

1. do not claim full RA-EE-MODQN,
2. do not claim learned association or hierarchical RL effectiveness,
3. do not claim joint association + power training,
4. do not claim RB / bandwidth allocation effectiveness,
5. do not claim HOBS optimizer behavior,
6. do not claim physical energy saving,
7. do not claim old EE-MODQN effectiveness,
8. do not use oracle rows as deployable runtime methods,
9. do not use scalar reward alone or per-user EE credit as success evidence,
10. do not use Catfish as an EE repair mechanism,
11. do not rewrite the scoped simulated-EE result as physical power saving or
    same-throughput-less-power unless separately tabulated.

## Phase 04-B / 04C Scope

Phase `04` remains `NEEDS MORE EVIDENCE` for effectiveness. Phase `04-B`
implemented the bounded Catfish-MODQN engineering slice under the original
MODQN reward only. Phase `04C` then ran the bounded attribution grid.

This section remains parked after RA-EE closeout unless the user explicitly
switches to Catfish feasibility. Do not start or continue Catfish work to repair
EE.

Scope:

1. new method family: `Catfish-MODQN`,
2. new config / artifact namespaces such as `configs/catfish-modqn-*` and
   `artifacts/catfish-modqn-*`,
3. main replay remains baseline-complete,
4. catfish replay receives the high-value subset,
5. high-value ranking uses `quality = 0.5*r1 + 0.3*r2 + 0.2*r3` initially,
6. quality percentile and `r1` / `r2` / `r3` component distributions are logged,
7. catfish agent may use a larger gamma while main agent keeps baseline gamma,
8. intervention mixes catfish samples into main update batches on a fixed period
   or fixed update interval,
9. competitive shaping is off in the primary run and ablation-only afterward.

Comparator:

```text
matched original MODQN-control
same seeds
same episode budget
same evaluation schedule
same final / best-eval checkpoint protocol
```

Tests should cover baseline unchanged behavior, config namespace gating,
high-value routing, intervention batch composition, metadata / log fields, and
no EE reward mode in Phase `04` configs.

Phase `04C` outputs:

1. `artifacts/catfish-modqn-phase-04-c-control-20ep/`,
2. `artifacts/catfish-modqn-phase-04-c-primary-shaping-off-20ep/`,
3. `artifacts/catfish-modqn-phase-04-c-no-intervention-20ep/`,
4. `artifacts/catfish-modqn-phase-04-c-no-asymmetric-gamma-20ep/`,
5. `docs/research/catfish-ee-modqn/04c-single-catfish-ablation-attribution.execution-report.md`.

The Phase `04C` result proves runnable attribution instrumentation and Catfish
participation in bounded runs. It does not prove effectiveness: the run is one
seed / `20` episodes, best-eval rows are effectively tied, primary and
no-asymmetric-gamma are identical on the final training row, and intervention
attribution remains weak.

## Phase 05A Scope And Result

Phase `05A` bounded multi-buffer validation is complete:

1. diagnostic report: `docs/research/catfish-ee-modqn/05a-multi-buffer-validation.execution-report.md`,
2. config: `configs/catfish-modqn-phase-05a-multi-buffer-primary.resolved.yaml`,
3. artifact: `artifacts/catfish-modqn-phase-05a-multi-buffer-primary-20ep/`.

Result:

```text
Phase 05A bounded diagnostic completion: PASS
Objective-specific buffer distinctness: FAIL
Phase 05B full multi-agent validation planning: BLOCK
```

Reasons:

1. `r1` high-value replay almost duplicated scalar/high-throughput replay,
2. `r2` percentile admission degenerated to all samples on the bounded surface,
3. diagnostics could not prove distinct intervention sample types,
4. no EE reward/objective or full multi-Catfish agents were introduced.

Do not start `catfish-r1`, `catfish-r2`, or `catfish-r3` learners from this
buffer construction. Any future Phase `05` work needs a new bounded design gate
that fixes objective admission/selectivity before full multi-agent validation.

## Phase 05R Result And Phase 05B Boundary

Phase `05R` has completed as an objective-buffer-redesign gate:

```text
Phase 05R: objective-buffer-redesign gate
Result: PASS for later Phase 05B planning draft only
Forbidden result: Multi-Catfish effectiveness claim
```

Report and artifact:

1. `docs/research/catfish-ee-modqn/05r-objective-buffer-redesign-gate.execution-report.md`,
2. `artifacts/catfish-modqn-phase-05r-objective-buffer-redesign-gate/phase05r_objective_buffer_redesign_diagnostics.json`.

The `guarded-residual-objective-admission` candidate produced bounded distinct
buffers:

1. admission shares: `r1 = 0.10`, `r2 = 0.1417`, `r3 = 0.20305`,
2. pairwise objective-buffer Jaccards stayed low,
3. scalar-replay duplicate risk was reduced,
4. fixed `70 main + 10 r1 + 10 r2 + 10 r3` composition improved expected
   `r1`, `r2`, and `r3` means without significant non-target damage.

This only allows drafting a Phase `05B` plan.

Forbidden:

1. no long training,
2. no full multi-Catfish agents,
3. no `catfish-r1`, `catfish-r2`, or `catfish-r3` learners,
4. no frozen baseline mutation,
5. no original MODQN reward / state / action / backbone change,
6. no EE reward, EE objective, EE-MODQN, or Catfish-EE-MODQN claim,
7. no scalar reward alone as success evidence,
8. no random tie-breaking to manufacture objective-buffer distinctness.

Phase `05B` planning must still define the bounded pilot, matched comparator,
critical ablations, diagnostics, acceptance criteria, stop conditions, and
forbidden claims before any implementation can be authorized.

## Phase 05B Planning Result

Phase `05B` planning is complete:

```text
Phase 05B planning boundary: PASS
Future implementation prompt draft: ALLOWED AFTER REVIEW
Phase 05B implementation now: FORBIDDEN
Training now: FORBIDDEN
Artifact generation now: FORBIDDEN
Full Multi-Catfish agents now: FORBIDDEN
```

Report:

1. `docs/research/catfish-ee-modqn/05b-multi-catfish-planning.execution-report.md`.

The accepted bounded implementation plan, if later authorized, is:

1. `20` episodes,
2. `3` matched seed triplets minimum,
3. evaluation seeds `[100, 200, 300, 400, 500]`,
4. eval cadence every `5` episodes and final,
5. total Catfish share fixed at `0.30`,
6. primary mix `70 main + 10 r1 + 10 r2 + 10 r3`,
7. guarded-residual admission rules from Phase `05R`,
8. matched MODQN, single-Catfish, multi-buffer / single-learner, and random /
   uniform equal-budget controls,
9. primary run shaping off.

The draft implementation prompt is:

1. `docs/research/catfish-ee-modqn/prompts/05b-multi-catfish-implementation-draft.prompt.md`.

Do not execute that draft unless the user explicitly authorizes Phase `05B`
implementation / bounded training.

## Phase 05B Bounded Pilot Result

Phase `05B` bounded pilot is complete:

```text
Phase 05B bounded runnable evidence: PASS
Phase 05B acceptance / effectiveness: FAIL
Multi-Catfish-MODQN promotion: BLOCK
```

Report and summary:

1. `docs/research/catfish-ee-modqn/05b-multi-catfish-bounded-pilot.execution-report.md`,
2. `artifacts/catfish-modqn-phase-05b-bounded-pilot-summary/phase05b_bounded_pilot_summary.json`.

Completed protocol:

1. `5` comparators × `3` matched seed triplets = `15` runs,
2. `20` episodes per run,
3. evaluation seeds `[100, 200, 300, 400, 500]`,
4. actual Catfish ratio `0.296875`.

Core outcome:

1. single Catfish final mean scalar: `609.209311`,
2. primary Multi-Catfish final mean scalar: `608.037110`,
3. multi-buffer / single-learner final mean scalar: `608.293825`,
4. random / uniform buffer control final mean scalar: `609.128592`,
5. primary improved `r2` only; `r1` and `r3` did not improve,
6. no NaN or action collapse was detected,
7. replay starvation counters were nonzero,
8. multi-buffer / single-learner and random-buffer controls matched or
   explained away primary.

Do not continue Phase `05B` by default with longer training, shaping-on
primary, ratio tuning, or specialist tweaks. Treat this as a bounded negative
result unless a new explicit design gate is opened.

## Phase 05C Disposition

Phase `05C` read-only route disposition and claim synthesis is complete:

```text
Phase 05C read-only disposition and claim synthesis: PASS
Phase 05B runnable evidence: PASS
Phase 05B effectiveness: FAIL
Multi-Catfish promotion: FAIL / BLOCK
Phase 06 / Catfish-EE-MODQN final claim: FAIL / BLOCKED
```

Report:

1. `docs/research/catfish-ee-modqn/05c-multi-catfish-route-disposition.execution-report.md`.

Default next action is paper-section synthesis only: claim boundary,
limitations, and negative-results narrative from the current evidence. No new
run, tuning, implementation, or Phase `06` execution should be opened without a
new explicit design gate.

## Phase 07A Recovery Gate

Phase `07A` read-only recovery gate is complete:

```text
Phase 07A read-only recovery gate: PASS
Phase 07B implementation/training from Phase 07A alone: NOT AUTHORIZED
Direct Phase 05B continuation: BLOCK
Multi-Catfish promotion: BLOCK
Phase 06 / Catfish-EE-MODQN: BLOCK
```

Report:

1. `docs/research/catfish-ee-modqn/07a-catfish-recovery-gate.execution-report.md`.

The gate preserves the Phase `05C` negative boundary and narrows any future
recovery route. Current evidence does not support implementation failure as the
primary explanation and does not justify simple budget insufficiency / longer
training. The strongest working explanation is that intervention utility was
not proven beyond equal-budget controls. Replay starvation is an observed
confounder and stop trigger.

Future Phase `07B`, if separately authorized, must be single-Catfish-first and
must compare against no-intervention, random/equal-budget replay injection,
replay-only single learner, no-asymmetric-gamma, and matched MODQN control.

Future Phase `07B` must keep original MODQN reward semantics:

1. `r1 = throughput`,
2. `r2 = handover penalty`,
3. `r3 = load balance`,
4. no EE,
5. no Catfish-EE,
6. no frozen baseline mutation,
7. primary shaping-off,
8. scalar reward is not enough.

## Phase 07B / 07C Single-Catfish Utility Disposition

Phase `07B` bounded pilot is complete:

```text
Phase 07B bounded implementation / diagnostics / pilot: PASS
Bounded single-Catfish intervention utility evidence: PASS
Broader Catfish-MODQN effectiveness promotion: NOT PROMOTED
Multi-Catfish reopening: BLOCKED / DEFERRED
Phase 06 / Catfish-EE-MODQN: BLOCKED
```

Report and summary:

1. `docs/research/catfish-ee-modqn/07c-catfish-intervention-utility-disposition.execution-report.md`,
2. `artifacts/catfish-modqn-phase-07b-bounded-pilot-summary/phase07b_bounded_pilot_summary.json`.

Completed protocol:

1. `18 / 18` mandatory bounded runs completed,
2. `20` episodes per run,
3. `3` matched seed triplets,
4. evaluation seeds `[100, 200, 300, 400, 500]`,
5. actual primary injected ratio `0.296875`,
6. no NaN, action collapse, or starvation stop trigger.

Core outcome:

1. primary shaping-off single Catfish final mean scalar: `609.209311`,
2. matched MODQN control: `608.359098`,
3. no-intervention: `608.294631`,
4. random / equal-budget injection: `608.683775`,
5. replay-only single learner: `608.486224`,
6. primary improved `r1` and `r3` versus key controls,
7. primary worsened `r2` / handovers,
8. primary and no-asymmetric-gamma were identical on aggregate metrics.

Use this as bounded single-Catfish intervention utility evidence only. Do not
promote broader Catfish-MODQN effectiveness. Do not claim asymmetric gamma as
the supported active mechanism. Do not claim handover / `r2` improvement.

At Phase `07C`, the next valid gate was Phase `07D` r2-guarded
single-Catfish robustness planning. Phase `07D` has now run and failed its
acceptance gate, so it must not reopen Multi-Catfish or Phase `06`.

## Phase 07D R2-Guarded Robustness Result

Phase `07D` bounded pilot is complete:

```text
Phase 07D bounded implementation / runs / diagnostics: PASS
Phase 07D acceptance / recovery promotion: FAIL
Broader Catfish-MODQN effectiveness: NOT PROMOTED
Multi-Catfish reopening: BLOCKED
Phase 06 / Catfish-EE-MODQN: BLOCKED
```

Report and summary:

1. `docs/research/catfish-ee-modqn/07d-r2-guarded-single-catfish-robustness.execution-report.md`,
2. `artifacts/catfish-modqn-phase-07d-r2-guarded-robustness-summary/phase07d_r2_guarded_robustness_summary.json`.

Core outcome:

1. required roles complete: `9` roles x `3` matched seed triplets = `27` runs,
2. primary guarded final mean scalar: `608.988400`,
3. primary scalar deltas remained positive versus matched MODQN
   (`+0.629302`), no-intervention (`+0.693769`), random / equal-budget
   (`+0.304625`), and replay-only (`+0.382683`),
4. r2 / handover non-inferiority failed versus matched MODQN:
   `r2_delta=-0.051667 < -0.02` and `handover_delta=+10.333333 > +5`,
5. summary reports `starvation_stop_trigger_absent=false`,
6. no EE / RA-EE / Catfish-EE / Phase `06` claim was made,
7. no Multi-Catfish reopening was performed.

Treat Phase `07D` as a bounded negative result for Catfish recovery promotion.
Phase `07B` remains bounded intervention utility evidence only. Do not continue
guard tuning by default; default next action is paper synthesis /
claim-boundary writing.

Stop if implementation requires changing the original reward, state, action, or
backbone; if shaping is needed for the primary result; if catfish replay
starves; if intervention does not affect main updates; if Q / loss instability
dominates; if gains appear only in scalar reward; or if results are framed as
EE / RA-EE / Catfish-EE evidence.

## What Not To Do Next

Do not:

1. claim EE-MODQN effectiveness from the bounded Phase `03` pilot,
2. start Catfish-MODQN while interpreting Phase `03`,
3. create three Catfish agents or objective-specialist learners,
4. compare final Catfish-EE-MODQN directly against original MODQN as the sole proof,
5. claim full paper-faithful reproduction,
6. use scalar reward alone as the comparison result,
7. write follow-on outputs into frozen baseline artifact directories,
8. change baseline configs to mean EE-MODQN or Catfish-MODQN,
9. treat the Phase `03C-C` tiny `EE_system` delta as effectiveness evidence,
10. rerun the same one-beam-collapsed route with more episodes as the next gate,
11. continue Phase `03C` with selector tweaks or reward retuning,
12. use Catfish as an EE repair mechanism,
13. reopen the RA-EE association proposal route by tuning RA-EE-06 / 06B / 08,
14. claim learned association, hierarchical RL, joint association + power training, or full RA-EE-MODQN,
15. claim HOBS optimizer behavior or physical energy saving,
16. claim RB / bandwidth allocation effectiveness from RA-EE-09,
17. continue tuning the tested RA-EE-09 bounded QoS-slack allocator by default,
18. treat Phase `05R` as Phase `05B`,
19. use Phase `05A` diagnostic completion as evidence that full multi-Catfish is ready,
20. use Phase `05R` PASS as permission to implement Phase `05B`,
21. use Phase `05B` planning PASS as permission to implement Phase `05B`,
22. claim Multi-Catfish effectiveness from the bounded Phase `05B` pilot,
23. continue Phase `05B` with longer training by default after the failed
    acceptance gate,
24. start Phase `06` validation from the current evidence,
25. attach Catfish as an EE repair mechanism after the Phase `05C` closeout,
26. treat Phase `07A` as authorization to implement or train Phase `07B`,
27. create a Phase `07B` implementation prompt without separate authorization,
28. continue Phase `05B` directly instead of opening single-Catfish causal
    diagnostics,
29. treat Phase `07B` as broad Catfish-MODQN effectiveness,
30. claim asymmetric gamma as the supported active Phase `07B` mechanism,
31. claim handover / `r2` improvement from Phase `07B`,
32. reopen Multi-Catfish or Phase `06` from Phase `07B` alone,
33. claim Phase `07D` recovered broader Catfish-MODQN effectiveness,
34. continue Phase `07D` guard tuning by default after the failed
    non-inferiority gate,
35. use Phase `07D` scalar gains to override the r2 / handover failure,
36. claim HOBS active-TX EE Route `D` recovered EE-MODQN effectiveness,
37. treat DPC denominator variability alone as energy-aware learning,
38. treat Route `D` scalar reward gain as success evidence,
39. scale Route `D` training before the one-active-beam collapse is addressed,
40. use Catfish or Multi-Catfish as the immediate repair for the HOBS active-TX
    EE one-beam-collapse blocker.

## Recommended Next Prompt

Default next research action is paper-section synthesis / claim-boundary
writing. Treat Phase `05B` as a bounded Multi-Catfish negative result, Phase
`07B` as bounded single-Catfish intervention utility evidence, Phase `07D` as a
bounded negative result for recovery promotion, RA-EE-09 as a negative tested
RB / bandwidth result, and HOBS active-TX EE Route `D` as a formula / DPC
denominator feasibility pass but learned-policy block.

If the explicit goal is to continue EE-MODQN, the next prompt must be a new
anti-collapse / capacity / assignment design gate. It must not ask for more
Route `D` training, Catfish repair, reward retuning, or Phase `03C`
continuation. The design gate must first specify how one-active-beam collapse
will be prevented, penalized, or made infeasible under learned greedy
evaluation.

Do not use `prompts/05-multi-catfish-modqn-validation.prompt.md` to authorize
full multi-agent implementation from the current Phase `05A` / `05R` evidence.
Do not use `prompts/04-single-catfish-modqn-feasibility.prompt.md` for EE
repair, RA-EE continuation, Multi-Catfish escalation, or Catfish-EE validation.
Do not create a Multi-Catfish or Phase `06` implementation prompt from Phase
`07B` evidence alone.

For RA-EE synthesis, use
`ra-ee-09-fixed-association-rb-bandwidth.execution-report.md` as the RA-EE-09
result authority and `ra-ee-09-completion-design-gate.md` as design history.
Any new RB / bandwidth design must be opened as a new explicit gate with a new
claim boundary; do not tune the failed RA-EE-09 candidate in place.
