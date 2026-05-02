# Multi-Catfish Redesign Plan

**Date:** `2026-05-02`
**Status:** current Catfish design authority for the reopened planning line
**Scope:** documentation / design only. This note does not authorize training,
pilot execution, simulator behavior changes, or thesis-claim promotion.

## Purpose

This note resets the Catfish development plan after the HEA-MODQN evidence
chain and the Catfish-over-HEA bounded pilot result.

The previous Catfish route used a narrow definition:

```text
Catfish = one replay-batch intervention helper
```

That route was useful as an engineering and evidence boundary, but it is no
longer the right planning surface for the thesis algorithm. The thesis
algorithmic direction is now:

```text
Multi-Catfish as the main Catfish method family,
designed first around the fixed HEA / EE_HO baseline.
```

Single-Catfish is no longer the required first implementation step. It is now a
collapsed ablation of the Multi-Catfish design.

## Fixed Surfaces

These surfaces are fixed and must not be changed by the Catfish redesign:

1. original MODQN remains the disclosed comparison baseline:

   ```text
   r1 = throughput
   r2 = handover penalty
   r3 = load balance
   ```

2. HEA-MODQN remains the only positive EE thesis baseline:

   ```text
   J = throughputUtilityScale * throughput
       - lambda_HO * E_HO * handover_count
   ```

3. ratio-form handover-aware EE remains:

   ```text
   EE_HO =
     total_bits
     /
     (communication_energy_joules + E_HO_total_joules)

   E_HO_total_joules = handover_count * E_HO
   ```

4. Catfish must adapt to the HEA / `EE_HO` formula and evidence boundary; it
   must not redefine the denominator, remove `E_HO_total`, use active-TX EE as
   `EE_HO`, or convert utility `J` into an EE ratio.
5. simulator behavior, state/action surface, reward semantics, evaluation,
   policy selection, seeds, budgets, checkpoint rule, and artifact accounting
   are frozen unless a later controller explicitly opens a new method family.

## Current Evidence State

Positive but scoped:

1. HEA-MODQN utility-form `J`: `PASS`, scoped.
2. HEA-MODQN ratio-form `EE_HO`: `PASS`, scoped.
3. HEA robustness / attribution: `PASS`.

Blocked or not promoted:

1. communication-only / general EE relaxation:
   `EE_general = total_bits / communication_energy_joules` gives `0/30`
   candidate wins in the clean HEA artifact.
2. Active-TX EE-MODQN: `BLOCK / NOT PROMOTED`.
3. CF-RA-CP active-TX EE: `BLOCK`.
4. CP-base active-TX / continuous-power: `BLOCK`.
5. old Phase `05` Multi-Catfish route: `CLOSED / NOT PROMOTED`.
6. Catfish-over-HEA bounded matched pilot: `BLOCK`.
7. Catfish-EE / Phase `06`: `BLOCKED`.

The Catfish-over-HEA pilot is especially important for the redesign. It proved
the default-off hook and matched-boundary machinery, but the enabled candidate
failed acceptance:

```text
mean J_delta = -1136.0548587210574
mean EE_HO_delta = +225.88343883804046
J non-worse cells = 7 / 15
EE_HO non-worse cells = 13 / 15
max handover delta = +7
max r2 worsening = 7
seed 5103/6103/7103 J_delta_sum = -37775.76469904102
```

The failed surface should be treated as evidence that a generic single
training-batch intervention is too blunt. It is not evidence that the broader
Catfish concept is impossible.

## Redefined Catfish Concept

For this thesis, Catfish is a conceptual training intervention, not a fixed
algorithm copied from a previous implementation.

The minimum concept to preserve is:

1. an external learning stimulus perturbs the main learner;
2. high-quality experiences are selected, distilled, or emphasized;
3. intervention happens during training, not as a teacher-student action label
   at inference;
4. the main MODQN / HEA policy remains the final policy integrator;
5. the intervention must be auditable against matched no-Catfish controls.

Allowed design freedom:

1. admission score,
2. specialist roles,
3. quota schedule,
4. intervention timing,
5. replay replacement / resampling / prioritization,
6. guardrail-aware rejection,
7. coordinator policy,
8. collapsed single-specialist ablation.

Forbidden design freedom:

1. changing the HEA objective or `EE_HO` formula,
2. changing simulator dynamics,
3. changing inference-time policy selection unless a later gate explicitly
   opens an ensemble variant,
4. hiding low-cost failures,
5. reporting scalar reward as success,
6. using Catfish to reopen active-TX EE, CP-base, Phase `06`, or old Phase
   `05` claims.

## Multi-Catfish-First Architecture

Design the full Multi-Catfish structure first. Single-Catfish is then the
collapsed ablation that shares the same coordinator and scoring surfaces.

Recommended roles:

| Role | Purpose | Admission emphasis | Must guard against |
|---|---|---|---|
| `catfish-ee` | Improve or preserve scoped `EE_HO` / `J` under the HEA setting | positive `EE_HO` contribution, positive `J`, reduced handover-energy denominator burden | communication-only EE loss hidden by handover denominator |
| `catfish-ho` | Protect service continuity and handover / `r2` stability | fewer harmful handovers, lower ping-pong, non-worse `r2`, lower guardrail risk | sticky policies that reduce handover while hurting throughput or served ratio |
| `catfish-qos` | Protect throughput / served / outage guardrails | high p05 support, served-ratio preservation, outage avoidance, load stability | throughput-only dominance that erases the HEA tradeoff |
| coordinator | Mix specialist pressure without starving the main learner | quotas, diversity, anti-starvation, guardrail rejection, equal-budget accounting | one specialist dominating, random-extra-data explanation, scalar-only success |

The coordinator is part of the algorithm. It must report:

1. per-specialist admitted sample counts,
2. per-specialist affected update counts,
3. quota utilization,
4. rejected samples and rejection reasons,
5. starvation counters,
6. objective / metric distribution of admitted samples,
7. overlap between specialist selections,
8. random/equal-budget control comparability.

## Single-Catfish As Collapsed Ablation

Single-Catfish should not be implemented as an unrelated older algorithm. It
should be a collapsed version of the same Multi-Catfish system:

```text
single-collapsed-catfish =
  one admission stream
  + one quota
  + the same coordinator
  + a weighted composite of catfish-ee, catfish-ho, and catfish-qos scores
```

This makes the ablation meaningful:

```text
no-Catfish HEA
vs single-collapsed Catfish-over-HEA
vs full Multi-Catfish-over-HEA
```

The question becomes whether role specialization and coordination add value
beyond a single high-quality replay view, not whether two unrelated Catfish
implementations happen to differ.

## Redesign Gate Ladder

### Gate 0: Documentation Sync

Status: complete.

Acceptance:

1. old single-first / Phase `05` / Phase `06` documents are marked historical
   or superseded;
2. HEA / `EE_HO` formula surfaces stay fixed;
3. `ntn-showcase-stack`, `catfish/`, and presentation-pack documents point to
   the current redesign boundary.

### Gate 1: Offline Multi-Catfish Design Diagnostics

No training. No pilot.

Use existing HEA artifacts and training-transition diagnostics if available.
If the required transition-level fields are missing, return `NEEDS MORE
DESIGN` rather than inventing them.

Questions:

1. Can `catfish-ee`, `catfish-ho`, and `catfish-qos` scores be computed from
   existing or explicitly planned fields?
2. Are the specialist sample sets distinct enough to justify separate roles?
3. Does the coordinator prevent replay starvation and specialist domination?
4. Would the proposed intervention have rejected the known bad
   Catfish-over-HEA pilot patterns: `J` loss, handover increase, `r2`
   worsening, and seed-5103 collapse?
5. Are random/equal-budget controls defined before training?

PASS requires a concrete artifact schema and simulated intervention plan. A
score idea without field provenance is `NEEDS MORE DESIGN`.

Status: completed as a read-only diagnostic; decision is `NEEDS MORE DESIGN`.

The diagnostic found enough cell / seed / aggregate evidence to keep the
Catfish-over-HEA bounded pilot blocked, but not enough transition-level
provenance for Multi-Catfish scoring or coordinator diagnostics. Existing HEA
and Catfish-over-HEA artifacts do not expose training transition rows, stable
sample IDs, per-transition reward components, per-transition `EE_HO` inputs,
role scores, rejection reasons, quotas, overlap diagnostics, anti-starvation
counters, or random / equal-budget sample identity comparability.

The next gate is therefore Gate 1A, not Gate 2.

### Gate 1A: Transition Provenance Schema And Coordinator Diagnostics SDD

No implementation. No training. No pilot.

Define the artifact schema required before Multi-Catfish offline diagnostics
can become concrete. Gate 1A must specify:

1. transition identity and matched-boundary provenance,
2. state / action / next-state or compact auditable feature provenance,
3. per-transition or declared-window reward and metric components,
4. per-transition or declared-window `EE_HO` source components,
5. `catfish-ee`, `catfish-ho`, and `catfish-qos` score source fields,
6. coordinator quotas, anti-starvation counters, overlap diagnostics,
   rejection reasons, and random / equal-budget controls,
7. validator fail-closed rules for missing units, missing provenance, aggregate
   back-solving, scalar reward substitution, or active-TX EE substitution.

The current Gate 1A SDD is
`2026-05-02-multi-catfish-gate1a-transition-provenance-sdd.md`.

### Gate 2: Collapsed Single-Catfish Pilot

Only after Gate 1A passes and a later implementation-readiness gate proves the
required transition provenance can be emitted without changing HEA / `EE_HO`,
simulator behavior, reward semantics, state/action surface, evaluation, or
policy selection.

Control:

```text
scoped HEA-MODQN, no Catfish
```

Candidate:

```text
same scoped HEA-MODQN
+ single-collapsed Catfish training intervention
```

PASS requires non-scalar improvement or non-inferiority on `J` and `EE_HO`,
positive seed support, p05 / served / outage / handover / `r2` guardrails, and
no low-cost or communication-only EE overclaim.

### Gate 3: Full Multi-Catfish Pilot

Only after Gate 2 passes or after the controller explicitly accepts a
multi-first skip with a stronger offline Gate 1.

Control set:

1. no-Catfish HEA,
2. single-collapsed Catfish-over-HEA,
3. equal-budget random intervention,
4. replay-only / no-specialist ablation if feasible.

Candidate:

```text
full Multi-Catfish-over-HEA
```

PASS requires full Multi-Catfish to add value beyond the collapsed single
ablation, not merely beyond no-Catfish.

### Gate 4: Thesis Claim Review

Only after Gate 3 passes.

Allowed claim shape, if supported:

```text
Under the same scoped HEA-MODQN high handover-cost /
service-continuity-sensitive setting, Multi-Catfish training intervention
improves or stabilizes HEA-MODQN learning relative to matched no-Catfish and
collapsed single-Catfish controls while preserving the HEA guardrails.
```

This still is not general EE-MODQN, active-TX EE recovery, physical energy
saving, or Phase `06`.

## Stop Conditions

Stop the redesign path if any of these hold:

1. the proposal needs a different `EE_HO` formula;
2. the proposal removes `E_HO_total` and still needs a positive EE claim;
3. the proposal changes simulator behavior, reward semantics, state/action
   surface, evaluation, or policy selection without a new method-family gate;
4. specialist sample sets are indistinct or dominated by one objective;
5. coordinator quotas cause replay starvation;
6. random/equal-budget controls explain the result;
7. `J` worsens beyond the predeclared margin;
8. `EE_HO` worsens beyond the predeclared margin;
9. p05 throughput, served ratio, outage, handover, or `r2` guardrails fail;
10. positive result is single-seed only;
11. positive result is scalar-reward-only;
12. the method needs a forbidden claim to sound successful.

## Files To Treat As Historical Evidence

Do not delete these files. They record why the old route failed:

1. `04-single-catfish-modqn-feasibility.md`
2. `05-multi-catfish-modqn-validation.md`
3. `06-final-catfish-ee-modqn-validation.md`
4. `05a-multi-buffer-validation.execution-report.md`
5. `05b-multi-catfish-bounded-pilot.execution-report.md`
6. `05c-multi-catfish-route-disposition.execution-report.md`
7. `07c-catfish-intervention-utility-disposition.execution-report.md`
8. `07d-r2-guarded-single-catfish-robustness.execution-report.md`
9. `ntn-sim-core/artifacts/catfish-over-hea-bounded-matched-pilot/summary.json`

They are constraints and failure evidence, not the current execution plan.

## Current Decision

```text
GATE 1 READ-ONLY DIAGNOSTICS: NEEDS MORE DESIGN
NEXT GATE: GATE 1A TRANSITION PROVENANCE SDD
```

No implementation, training, pilot, or promotion follows from this note. The
next action is schema / readiness design for transition-level provenance and
offline coordinator diagnostics. Gate 2 is blocked until Gate 1A and a later
implementation-readiness slice pass.
