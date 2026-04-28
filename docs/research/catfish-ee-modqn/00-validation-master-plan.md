# Catfish / EE-MODQN Validation Master Plan

**Date:** `2026-04-28`  
**Status:** Phase-gated validation plan with execution handoff  
**Scope:** research-design validation only; no trainer, reward, config, or artifact contract changes are authorized by this note.

## Purpose

This plan separates the proposed Catfish / EE-MODQN research direction into independently reviewable validation questions.

The goal is to avoid a single uncontrolled jump from:

```text
MODQN
→ multi-catfish + EE objective + HOBS power semantics + new replay routing
```

because that would make any improvement or regression impossible to attribute.

## Current Method Boundary

The existing MODQN baseline remains the comparison anchor:

```text
r1 = throughput
r2 = handover penalty
r3 = load balance
```

Proposed follow-on method family:

```text
EE-MODQN:
  r1 = HOBS-linked energy efficiency
  r2 = handover penalty
  r3 = load balance

Catfish-MODQN:
  original MODQN objective
  plus Catfish-style replay/intervention training

Multi-Catfish-MODQN:
  original MODQN objective
  plus objective-specialized Catfish replay/intervention

Catfish-EE-MODQN:
  EE-MODQN objective
  plus Catfish-style replay/intervention training
```

No follow-on may silently replace the original MODQN baseline.

## Validation Parts

| Phase | Validation question | Required before |
|---|---|---|
| `01` | Is the original MODQN baseline still a valid comparison anchor? | every follow-on comparison |
| `02` | Is the HOBS-linked EE formula coherent and defensible? | EE-MODQN |
| `03` | What changes when only `r1` becomes EE? | Catfish-EE-MODQN |
| `04` | Can Catfish training mechanics attach to original MODQN? | multi-catfish and final method |
| `05` | Does objective-specialized multi-catfish add value beyond single Catfish? | multi-catfish claim |
| `06` | Does Catfish improve EE-MODQN under the final claim boundary? | final method claim |

## Current Phase Decisions

| Phase | Decision | Meaning |
|---|---|---|
| `01` | `PROMOTE` | The original MODQN surface can be used as a disclosed comparison baseline, not as a full paper-faithful reproduction. See `reviews/01-modqn-baseline-anchor.review.md`. |
| `02` | `PROMOTE` | The HOBS-linked active-transmit-power EE formula can enter EE-MODQN, with provenance labels and guardrails. See `reviews/02-hobs-ee-formula-validation.review.md`. |
| `03` | `NEEDS MORE EVIDENCE` | A bounded paired pilot exists, and Phase `03B` reward/objective-geometry follow-up also did not promote EE-MODQN. Phase `03B` kept the Phase `02B` power surface, added opt-in reward normalization plus load-balance calibration, and changed EE r1 to a denominator-sensitive beam-power credit. Learned policies still collapsed every evaluated step to one active beam, `denominator_varies_in_eval=false`, `EE_system` tied control, and throughput-vs-EE ranking remained effectively identical. See `reviews/03-ee-modqn-validation.review.md`, `03-ee-modqn-validation.execution-report.md`, and `03b-ee-modqn-objective-geometry.execution-report.md`. |
| `04` | `NEEDS MORE EVIDENCE` | Single Catfish-MODQN is a reasonable feasibility design, but cannot be promoted until bounded pilots, replay/intervention diagnostics, and shaping-off primary results exist. See `reviews/04-single-catfish-modqn-feasibility.review.md`. |
| `05` | `NEEDS MORE EVIDENCE` | Multi-catfish should start with 05A objective-specific multi-buffer validation; full three-agent validation is blocked until 05A and Phase 04 produce evidence. See `reviews/05-multi-catfish-modqn-validation.review.md`. |
| `06` | `NEEDS MORE EVIDENCE` | Final Catfish-EE-MODQN comparison design is valid, but final-method claims are blocked until Phases 03-05 produce evidence. See `reviews/06-final-catfish-ee-modqn-validation.review.md`. |

## Current Execution Status

For implementation and experiment planning, distinguish review decisions from executable completion:

| Phase | Execution status | Next action |
|---|---|---|
| `01` | Complete for this research track | Use as the frozen disclosed comparison baseline. Do not reopen unless the goal changes to full paper-faithful reproduction. |
| `02` | Complete for formula plus Phase `02B` opt-in power-surface audit | Use only the disclosed Phase `02B` `active-load-concave` HOBS-compatible proxy for Phase `03` paired experiments. It remains a synthesized proxy, not a paper-backed power optimizer. |
| `03` | Bounded paired pilot plus Phase `03B` objective-geometry follow-up complete; not promoted | Do not claim EE-MODQN effectiveness. Current evidence shows high rescaling risk, no `EE_system` gain, and learned-policy one-beam collapse even after Phase `03B` reward normalization and load-balance calibration. Any further Phase `03` work needs a separately approved design change that makes evaluated policies exercise denominator variability; more episodes alone are not the next gate. |
| `04` | Separate bounded branch | Can be planned after Phase `01`, but it is not the default next step for the EE route. Keep original MODQN reward. |
| `05` | Blocked | Start only with `05A` multi-buffer validation after Phase `04` evidence exists. |
| `06` | Blocked | Final-method validation waits for evidence from Phases `03`, `04`, and `05`. |

The current handoff for a new planning dialogue is `execution-handoff.md`.

## Development Guardrail

Follow-on work may live inside `modqn-paper-reproduction`, but it must not mutate or overwrite the frozen baseline surface. The operational rules are recorded in `development-guardrails.md`.

In short:

1. keep baseline configs and artifacts frozen,
2. put every follow-on method under a new config and artifact namespace,
3. preserve baseline rerun behavior when changing shared code,
4. never relabel EE / Catfish follow-on results as the original MODQN baseline.

## Recommended Order

```text
Phase 01: MODQN baseline anchor
Phase 02: HOBS-linked EE formula validation
Phase 03: EE-MODQN validation
Phase 04: Single Catfish-MODQN feasibility
Phase 05: Multi-Catfish-MODQN validation
Phase 06: Final Catfish-EE-MODQN validation
```

Phase `04` can run before Phase `03` if the immediate goal is only to prove that Catfish can attach to the MODQN backbone. It must then remain scoped as Catfish mechanism validation and must not make any EE claim.

## Source Anchors

Use these local sources before proposing changes:

1. `modqn-paper-reproduction/docs/presentation/dqn-development-report.md`
2. `modqn-paper-reproduction/docs/presentation/deep-research-report.md`
3. `modqn-paper-reproduction/docs/research/catfish-ee-modqn/development-guardrails.md`
4. `catfish/README.md`
5. `catfish/notes/algorithm-definition.md`
6. `catfish/notes/modqn-hobs-project-route.md`
7. `catfish/notes/teacher-student-comparison-and-risks.md`
8. `catfish/notes/advisor-progress-summary.md`
9. `catfish/notes/catfish-report.md`
10. `system-model-refs/system-model-formulas.md`
11. `system-model-refs/system-model-derivation.md`
12. `system-model-refs/power-formula-taxonomy-and-hobs-downlink-note.md`
13. `system-model-refs/paper-catalog-power-formula-report-2026-04-21.md`
14. `system-model-refs/paper-catalog-strict-closed-form-power-audit-2026-04-21.md`
15. `paper-catalog/catalog/PAP-2024-HOBS.json`

## Common Claim Rules

1. `Catfish-MODQN` may claim only Catfish mechanism feasibility over the original MODQN objective.
2. `EE-MODQN` may claim only objective substitution effects, not Catfish improvement.
3. `Multi-Catfish-MODQN` may claim objective-specialized Catfish value only if compared against single Catfish.
4. `Catfish-EE-MODQN` must be compared primarily against `EE-MODQN`, not only against original MODQN.
5. Any per-user EE reward adapted from system-level HOBS EE must be disclosed as a modeling assumption.
6. Throughput should be handled as an EE numerator plus QoS guardrail / reporting metric, not automatically as a fourth objective.
7. Phase `01` fixes the original reward surface as `r1 = throughput`, `r2 = handover penalty`, and `r3 = load balance`; reward-calibration, scenario-corrected, and beam-aware follow-on surfaces must not silently replace that baseline.
8. Phase `02` fixes the EE denominator as controlled / allocated beam transmit power in linear W; it must not be replaced by a path-loss closed form or a fixed config-power constant.
9. Phase `03` requires a paired `MODQN-control` vs `EE-MODQN` experiment on the same HOBS-linked SINR / power surface; old MODQN artifacts alone cannot establish the effect of objective substitution.
10. Phase `04` treats Catfish as a training strategy only; it must keep the original MODQN state, action, reward, and backbone fixed, with competitive shaping disabled in the primary run or isolated as an ablation.
11. Phase `05` must not jump directly to full multi-agent training; it must first prove that objective-specific buffers capture meaningfully different experience under equal-budget comparison.
12. Phase `06` must compare primarily against `EE-MODQN`; original MODQN is context only because it uses a different first objective.

## Agent / Review Workflow

Each phase should be reviewed independently:

```text
1. Open one phase brief.
2. Use the matching prompt in prompts/.
3. Ask the reviewer / agent to answer only that phase.
4. Do not let a phase reviewer redesign later phases.
5. Collect phase reports.
6. Write a final synthesis only after all relevant phase reports are complete.
```

This isolation is deliberate. Formula review, objective review, Catfish feasibility, multi-agent design, and final-method comparison are different questions.
