# Catfish / EE-MODQN Validation Master Plan

**Date:** `2026-05-02`
**Status:** Phase-gated validation plan with RA-EE closeout, active-TX EE / CP-base routes blocked, HEA-MODQN scoped thesis PASS, and Multi-Catfish reopened only as a redesign-planning line
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

Redesigned Multi-Catfish-over-HEA:
  scoped HEA-MODQN objective and ratio-form EE_HO fixed
  plus a Multi-Catfish-first training intervention layer
  with Single-Catfish represented only as a collapsed ablation

Catfish-EE-MODQN:
  EE-MODQN objective
  plus Catfish-style replay/intervention training

Handover-Energy-Aware MODQN / HEA-MODQN:
  service-continuity-sensitive MODQN extension
  declared handover-aware utility plus ratio-form EE_HO under frozen
  high handover-cost sensitivity
```

No follow-on may silently replace the original MODQN baseline.

## RA-EE Closeout Boundary

The RA-EE follow-on is now closed at the current evidence boundary:

```text
old EE-MODQN r1-substitution route: BLOCKED / STOP
RA-EE fixed-association deployable power allocation: PASS, scoped
RA-EE fixed-association RB / bandwidth candidate: AUDITABLE, NOT PROMOTED
RA-EE learned association / hierarchical RL / full RA-EE-MODQN: BLOCKED
Catfish for EE repair: BLOCKED
```

This closeout must not be read as a full `RA-EE-MODQN` result. The positive
evidence is limited to fixed-association offline replay with a deployable
non-oracle finite-codebook power allocator under the disclosed simulation
environment and matched held-out replay. Association proposal replay, learned
association, hierarchical RL, joint association + power training, and final
Catfish-EE claims remain blocked. RA-EE-09 added an auditable normalized
bandwidth / resource-share candidate under the fixed-association RA-EE-07 power
boundary, but the tested candidate failed the held-out effectiveness gate and
does not authorize an RB / bandwidth allocation effectiveness claim.

Paper-safe claim boundary:

```text
Under the disclosed simulation setting and fixed-association held-out replay,
the RA-EE-07 deployable non-oracle finite-codebook power allocator improves
simulated system EE over the matched fixed-association RA-EE-04/05 safe-greedy
power allocator while preserving the declared QoS and power guardrails.
```

Do not shorten this to physical energy saving, HOBS optimizer behavior, learned
EE-MODQN, or full RA-EE-MODQN.

## HOBS Active-TX EE Reopen Boundary

The post-Phase `03D` HOBS active-TX EE feasibility chain is now recorded in
`hobs-active-tx-ee-modqn-feasibility.execution-report.md`.

Current boundary:

```text
HOBS active-TX EE formula / reward wiring: PASS, scoped
SINR structural audit: PASS, but negligible at current MODQN operating point
channel-regime / antenna-gain path: BLOCK as a paper-backed MODQN continuation
HOBS-inspired DPC sidecar denominator gate: PASS
tiny learned-policy Route D denominator check: BLOCK
capacity-aware anti-collapse assignment gate: BLOCK
QoS-sticky overflow anti-collapse gate: PASS, scoped
QoS-sticky robustness / attribution gate: PASS, scoped
QoS-sticky broader-effectiveness gate: BLOCK
CP-base bounded matched pilot: BLOCK
EE-MODQN effectiveness: NOT PROMOTED / STOP CURRENT QOS-STICKY AND CP-BASE EE OBJECTIVE ROUTES
```

This route answers one important concern: the EE denominator does not need to
remain constant. With the opt-in HOBS-inspired DPC sidecar, active transmit
power varied under greedy evaluation and same-policy throughput-vs-EE ranking
separated. It does not solve learned policy collapse: Route `D` still evaluated
every step with one active beam (`all_evaluated_steps_one_active_beam=true`
across `50` evaluated steps).

Therefore the blocker has moved from "the denominator cannot vary" to "the
learned beam-selection policy still collapses to one active beam." Do not scale
Route `D` training by default. Any continuation must first open a new
anti-collapse / capacity / assignment design gate. Catfish remains a training
strategy and must not be used as an EE repair mechanism for this structural
blocker.

The first anti-collapse design gate then tested a minimal
`capacity-aware-greedy-assignment` constraint under
`configs/hobs-active-tx-ee-anti-collapse-*` and
`artifacts/hobs-active-tx-ee-anti-collapse-*`. The matched boundary passed and
the candidate removed one-active-beam collapse
(`active_beam_count_distribution={"2.0": 50}`) while preserving denominator
variability, but acceptance failed: `p05_throughput_ratio_vs_control=0.285`,
`handover_delta=+495`, and `r2_delta=-0.495` violated the predeclared
guardrails. This gate is a bounded negative result, not an EE-MODQN
effectiveness promotion. See
`hobs-active-tx-ee-anti-collapse-design-gate.execution-report.md`.

The follow-up QoS- and stickiness-preserving gate tested
`qos-sticky-overflow-reassignment` under
`configs/hobs-active-tx-ee-qos-sticky-anti-collapse-*` and
`artifacts/hobs-active-tx-ee-qos-sticky-anti-collapse-*`. It first computed the
normal learned greedy actions, intervened only on overloaded beams, allowed only
QoS-safe sticky overrides, and kept non-sticky moves disabled. This bounded gate
passed: `all_evaluated_steps_one_active_beam=false`,
`active_beam_count_distribution={"7.0": 50}`,
`p05_throughput_ratio_vs_control=2.522568929129207`,
`handover_delta=-211`, `r2_delta=+0.21100000000000002`, and all power/accounting
violations were `0`. This is a scoped anti-collapse result only; it does not
promote EE-MODQN effectiveness or physical energy-saving claims. See
`hobs-active-tx-ee-qos-sticky-anti-collapse-design-gate.execution-report.md`.

The bounded robustness / mechanism-attribution gate then tested the same
QoS-sticky overflow mechanism across `3` matched training seed triplets and
nearby ablations. The primary role passed in aggregate and per seed:
`active_beam_count_distribution={"7.0": 150}`,
`p05_throughput_ratio_vs_control=3.105617320531727`,
`handover_delta=-292.00000000000006`, and
`r2_delta=+0.29200000000000004`. Threshold sensitivity at `45` and `55` also
passed, and no non-sticky moves were used. The QoS-ratio guard was not the
active mechanism on this bounded surface because relaxed-QoS and stricter-QoS
ablations produced the same aggregate result; attribution should be to sticky
overflow reassignment under non-sticky / handover protections. This is scoped
robustness evidence only, not a general EE-MODQN effectiveness claim. See
`hobs-active-tx-ee-qos-sticky-robustness-gate.execution-report.md`.

The bounded broader-effectiveness gate then compared four matched roles:
DPC-matched throughput control, HOBS-EE no-anti-collapse control, QoS-sticky EE
candidate, and anti-collapse-throughput control. The candidate kept the scoped
anti-collapse basics (`all_evaluated_steps_one_active_beam=false`,
`active_beam_count_distribution={"7.0": 150}`,
`denominator_varies_in_eval=true`, zero power/accounting violations), but it
failed the decisive comparison against `anti-collapse-throughput-control`:
`EE_system_delta=-0.32138238121922313`, `handover_delta=+83.33333333333331`,
and `r2_delta=-0.08333333333333334`. The throughput-control arm with the same
anti-collapse hook explains the useful gain boundary, so the current
QoS-sticky HOBS-active-TX EE objective-contribution route is stopped. The
remaining positive result is scoped anti-collapse evidence only. See
`hobs-active-tx-ee-qos-sticky-broader-effectiveness-gate.execution-report.md`.

The next failure-informed design gate then proposed
`HOBS-active-TX non-codebook continuous-power base EE-MODQN`
(`CP-base-EE-MODQN`) under a new non-codebook continuous-power namespace. This
gate passed for later controller review only: it defines an analytic continuous
rollout-time `p_b(t)` sidecar shared by the EE candidate and the decisive
throughput + same-anti-collapse control, and the only intended future role
difference is `r1 = hobs-active-tx-ee` versus `r1 = throughput`. The gate does
not authorize implementation, pilot training, Catfish-EE, or EE-MODQN
effectiveness claims. See
`hobs-active-tx-ee-non-codebook-continuous-power-design-gate.execution-report.md`.

The follow-up implementation-readiness / boundary-audit slice then passed. It
implemented the opt-in `non-codebook-continuous-power` power-surface mode,
namespace-gated configs, rollout-time continuous `p_b(t)` wiring, focused tests,
and a deterministic boundary-audit artifact. The audit proves that the future
candidate/control pair can share the same continuous power surface and
QoS-sticky structural guard, with `r1_reward_mode` as the only
boundary-critical intended difference. This is readiness evidence only: no
pilot training was run or authorized, and no EE-MODQN effectiveness or
Catfish-EE claim is allowed. See
`hobs-active-tx-ee-non-codebook-continuous-power-implementation-readiness.execution-report.md`.

The bounded matched pilot then executed under the required CP-base namespace.
The matched boundary passed: the EE candidate and throughput control shared the
same non-codebook continuous power surface, QoS-sticky structural guard, seed
triplets, eval seeds, episode budget, checkpoint protocol, and trainer
hyperparameters; only `r1_reward_mode` differed. Acceptance blocked. The EE
candidate lost against `throughput + same guard + same continuous power`
(`EE_system_delta=-1.057740305239463`), exceeded the handover guardrail
(`+77.33333333333331 > +25`), failed the `r2` guardrail
(`-0.07733333333333334 < -0.05`), and only won scalar reward. All three
per-seed EE deltas were negative. This is a bounded negative result, not
`NEEDS MORE DESIGN`, and it does not authorize tuning reruns. See
`hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot.execution-report.md`
and
`artifacts/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot-summary/summary.json`.

## HEA-MODQN Thesis Claim Boundary

The current positive thesis result is separate from active-TX EE, RA-EE,
Catfish, Multi-Catfish, and Phase `06`. Method naming must use
`Handover-Energy-Aware MODQN`, `HEA-MODQN`, or
`service-continuity-sensitive MODQN extension`; do not use HOBS in the method
name.

Current controller state:

```text
Active-TX EE-MODQN: BLOCK / NOT PROMOTED
CF-RA-CP active-TX EE: BLOCK
HEA-MODQN high handover-cost / service-continuity-sensitive scoped pilot: PASS
HEA-MODQN ratio-form EE_HO clean artifact gate: PASS, scoped
HEA-MODQN robustness / attribution gate: PASS
Catfish / Multi-Catfish for EE repair: BLOCKED / NOT PROMOTED
Catfish-over-HEA baseline parity gate: PASS readiness / parity only
Catfish-over-HEA bounded matched pilot: BLOCK
Multi-Catfish redesign plan: PASS for documentation / redesign planning only
Multi-Catfish Gate 1 read-only diagnostics: NEEDS MORE DESIGN
Multi-Catfish Gate 1A transition provenance SDD: current next design gate
Catfish-EE / Phase 06: BLOCKED
Scalar reward: diagnostic only
```

Paper-safe claim boundary:

```text
Under a frozen high handover-cost / service-continuity-sensitive sensitivity
setting, HEA-MODQN improves both the declared handover-aware utility and the
ratio-form handover-aware EE_HO relative to matched throughput-objective MODQN
while preserving p05 throughput, served ratio, and outage guardrails.
```

Formula evidence boundary:

```text
eta_EE,HO =
  (sum_t sum_u R_u^t Delta_t)
  /
  (sum_t sum_s P_tot,s^t Delta_t + E_HO,total)

E_HO,total = sum_t sum_u delta_HO,u^t * E_u,HO
```

This `handover-aware EE` formula is allowed only as a thesis-level sensitivity
definition with a declared assumption set. It is supported by the local source
chain below, but it is not a direct replacement for the blocked active-TX EE
route:

1. `system-model-refs/system-model-formulas.md` §3.14-3.16 defines
   handover-aware EE and separates it from active-TX EE and utility-form
   fallback.
2. `system-model-refs/simulator-parameter-spec.md` §6 marks
   `hoEnergyJoules` as a scenario parameter with no universal default and
   `lambdaHo=0.2` as the EAQL-derived utility anchor.
3. `paper-catalog/catalog/PAP-2025-EAQL.json` provides the closest
   energy-aware handover utility / reward precedent using
   `lambda * E_handover`.
4. `paper-catalog/catalog/PAP-2025-BIPARTITE-HM.json` provides a supporting
   handover strategy-cost precedent with per-handover energy, setup cost, RTT,
   and sensitivity to energy weight.

Controller boundary: the completed HEA-MODQN utility gate and the later clean
ratio-form artifact gate both pass, but only as scoped thesis evidence under
the declared high handover-cost / service-continuity-sensitive sensitivity
setting. Do not silently promote utility `J` success into ratio EE success in
other runs; ratio-form `EE_HO` must remain artifact-backed by explicit
`total_bits`, `communication_energy_joules`, `handover_count`,
`E_HO_total_joules`, and `EE_HO_bits_per_joule` fields.

Scenario realism boundary:

1. The high-cost condition is literature-defensible because LEO handover papers
   include explicit handover energy, operational handover cost, setup cost,
   RTT, strategy cost, and energy weighting in decision / cost models.
2. It is not a universal physical deployment default. `E_u,HO` remains a
   scenario / sensitivity parameter with no corpus-wide LEO default.
3. High-cost rows (`E_HO=130/150/200`) should be described as a
   service-continuity-sensitive regime capturing signaling, setup/recovery,
   interruption, QoS-continuity, or terminal-side overhead pressure.
4. Do not state that real LEO handovers generally consume `130 J`, `150 J`, or
   `200 J`, and do not convert this into a physical satellite energy-saving
   claim.

Final relaxation / stop-loss boundary:

1. The relaxed general formula
   `EE_general = total_bits / communication_energy_joules` is directly
   computable from the clean HEA artifact.
2. It blocks the relaxation. Candidate wins are `0/30` across the full grid,
   `0/15` in the primary high-cost subset, and `0/6` in the secondary subset.
   Primary mean communication-only `EE_delta` is about `-82980 bits/J`;
   secondary mean communication-only `EE_delta` is about `-74659 bits/J`.
3. Removing `E_HO,total` therefore removes the positive result. The accepted
   HEA-MODQN claim is handover-aware / service-continuity-sensitive only; it is
   not a general communication-energy-only EE claim.
4. No new EE design gate should be opened from this evidence. Reopening
   active-TX, CP-base, RA-EE association, Catfish, Multi-Catfish, or Phase `06`
   would repeat already blocked routes or require a new simulator / action /
   accounting family outside this thesis stop-loss boundary.

Metrics to carry into thesis tables:

```text
primary subset positive cells = 15/15
secondary subset positive cells = 6/6
primary mean J_delta = 13288.26934925599
candidate vs ablation mean primary J_delta = 13288.26934925599
break-even-near E_HO=130, lambda_HO=0.2 J_delta = 217.11219999700552
low-cost E_HO=3, lambda_HO=0.2 J_delta = 0
primary candidate/control EE_HO = 414946.480518 / 412914.217690 bits/J
primary EE_HO_delta = +2032.262828 bits/J
primary EE_HO_ratio = 1.0049217555
secondary candidate/control EE_HO = 1278023.040718 / 1272319.566835 bits/J
secondary EE_HO_delta = +5703.473883 bits/J
low-cost E_HO=3, lambda_HO=0.2 EE_HO_delta = 0
communication-only EE candidate wins = 0/30
primary communication-only EE wins = 0/15
secondary communication-only EE wins = 0/6
primary mean communication-only EE_delta ~= -82980 bits/J
secondary mean communication-only EE_delta ~= -74659 bits/J
lambda-zero ablation primary wins = 0/15
primary min p05 throughput ratio = 0.9665274269295018
served_delta = 0
outage_delta = 0
primary mean handover_delta = -43.733333333333334
```

Boundary rules:

1. active-TX EE is reported separately / diagnostic,
2. scalar reward is diagnostic only,
3. high-cost values are sensitivity / utility-accounting penalties, not
   physical constants,
4. no low-cost success claim is authorized,
5. no physical energy-saving result is authorized,
6. no active-TX EE recovery, Catfish-EE readiness, Phase `06` readiness, or
   Multi-Catfish repair claim follows from this result.

Next step for the thesis claim remains tables / figures / narrative polish. A
bounded `Catfish-over-HEA` readiness slice passed in `ntn-sim-core`, but the
subsequent bounded matched pilot is blocked: the candidate improved mean
`EE_HO` only weakly, worsened mean utility `J`, worsened some `EE_HO` cells,
and violated handover / `r2` guardrails. That negative result blocks the
generic single training-batch intervention route as a claim-bearing method.

Catfish may now be reconsidered only through the new Multi-Catfish-first
redesign plan recorded in
`2026-05-02-multi-catfish-redesign-plan.md`. This is a design-planning reset,
not Catfish effectiveness, not Catfish-EE, not Phase `06`, not Active-TX EE
recovery, and not a change to the HEA thesis claim. Single-Catfish is no
longer the required development path; it is a collapsed ablation of the
redesigned Multi-Catfish system.

The first read-only Multi-Catfish diagnostics gate completed with
`NEEDS MORE DESIGN`. Existing HEA and Catfish-over-HEA artifacts are sufficient
for aggregate gate-level failure confirmation, but not for transition-level
`catfish-ee`, `catfish-ho`, `catfish-qos`, or coordinator diagnostics. The
current next design surface is
`2026-05-02-multi-catfish-gate1a-transition-provenance-sdd.md`.

## Validation Parts

| Phase | Validation question | Required before |
|---|---|---|
| `01` | Is the original MODQN baseline still a valid comparison anchor? | every follow-on comparison |
| `02` | Is the HOBS-linked EE formula coherent and defensible? | EE-MODQN |
| `03` | What changes when only `r1` becomes EE? | Catfish-EE-MODQN |
| `04` | Can Catfish training mechanics attach to original MODQN? | multi-catfish and final method |
| `05` | Does objective-specialized multi-catfish add value beyond single Catfish? | multi-catfish claim |
| `06` | Does Catfish improve EE-MODQN under the final claim boundary? | final method claim |
| `07A` | Does the post-05B evidence authorize recovery work, direct continuation, or promotion? | any future Catfish recovery implementation |
| `07B` | Does single-Catfish intervention have bounded utility beyond no-intervention and equal-budget replay controls? | any future Catfish recovery claim |
| `07C` | How should Phase `07B` be interpreted, and what remains blocked? | post-07B claim boundary |
| `07D` | Does an r2 / handover-guarded single-Catfish variant preserve utility without handover degradation? | broader Catfish-MODQN recovery claim |
| `RA-EE` | Does a resource-allocation EE route establish deployable power and association evidence? | any RA-EE-MODQN or Catfish-EE claim |
| `HOBS active-TX EE` | Can the HOBS-style active transmit-power EE formula and denominator dynamics be attached to MODQN without reproducing Phase `03` collapse? | any renewed EE-MODQN objective-substitution claim |
| `HEA-MODQN` | Does the service-continuity-sensitive extension improve declared handover-aware utility and ratio-form handover-aware `EE_HO` under the frozen high handover-cost sensitivity setting while preserving p05 / served / outage guardrails? | thesis scoped HEA-MODQN claim |
| `Catfish-over-HEA` | Can a default-off Catfish training-time intervention attach to the scoped HEA-MODQN baseline without changing reward semantics, formulas, state/action surfaces, evaluation, policy selection, or disabled baseline behavior? | any future Catfish-over-HEA training-enhancement pilot |
| `Multi-Catfish redesign` | Can the Catfish concept be redesigned as a Multi-Catfish-first training intervention around fixed HEA / `EE_HO`, with Single-Catfish as a collapsed ablation and without reopening active-TX EE, Phase `06`, or simulator changes? | any future Multi-Catfish implementation gate |
| `Multi-Catfish Gate 1A` | What transition-provenance schema and offline coordinator diagnostics are required before Multi-Catfish scoring can be implemented or piloted? | any future Multi-Catfish implementation-readiness slice |

## Current Phase Decisions

| Phase | Decision | Meaning |
|---|---|---|
| `01` | `PROMOTE` | The original MODQN surface can be used as a disclosed comparison baseline, not as a full paper-faithful reproduction. See `reviews/01-modqn-baseline-anchor.review.md`. |
| `02` | `PROMOTE` | The HOBS-linked active-transmit-power EE formula can enter EE-MODQN, with provenance labels and guardrails. See `reviews/02-hobs-ee-formula-validation.review.md`. |
| `03` | `BLOCKED / STOP CURRENT ROUTE` | A bounded paired pilot exists, and Phase `03B` reward/objective-geometry follow-up also did not promote EE-MODQN. Phase `03B` kept the Phase `02B` power surface, added opt-in reward normalization plus load-balance calibration, and changed EE r1 to a denominator-sensitive beam-power credit. Learned policies still collapsed every evaluated step to one active beam, `denominator_varies_in_eval=false`, `EE_system` tied control, and throughput-vs-EE ranking remained effectively identical. Phase `03C-B` added only a static/counterfactual power-MDP codebook audit and passed to a bounded paired pilot by proving power decisions can change the denominator and separate same-policy throughput-vs-EE ranking. Phase `03C-C` ran that bounded paired pilot with an explicit runtime selector, but it is `BLOCKED`: the evaluated candidate still collapsed to one active beam, selected only `fixed-low`, kept `denominator_varies_in_eval=false`, lost `50%` p05 throughput, and did not separate throughput-vs-EE ranking. Phase `03D` disposition stops the current EE-MODQN r1-substitution route; EE may reopen only as a new resource-allocation MDP design gate. See `reviews/03-ee-modqn-validation.review.md`, `03-ee-modqn-validation.execution-report.md`, `03b-ee-modqn-objective-geometry.execution-report.md`, `03c-b-power-mdp-audit.execution-report.md`, `03c-c-power-mdp-pilot.execution-report.md`, and `03d-ee-route-disposition.execution-report.md`. |
| `HOBS active-TX EE` | `PASS FOR WIRING / DPC DENOMINATOR / BOUNDED QOS-STICKY ANTI-COLLAPSE ROBUSTNESS; BLOCKED FOR BROADER EE OBJECTIVE CONTRIBUTION; CP-BASE BOUNDED PILOT BLOCK` | The post-03D HOBS active-TX EE chain proves the scoped formula can be computed, the reward mode can be opt-in, and the HOBS-inspired DPC sidecar can make total active transmit power vary. Route `D` remains blocked as a plain learned-policy check because candidate and control both evaluated with one active beam on all `50` evaluated steps. The first capacity-aware anti-collapse gate removed one-active-beam collapse but failed p05 throughput and handover / `r2` guardrails. The QoS-sticky overflow gate then passed the bounded anti-collapse criteria without non-sticky moves, and the robustness / attribution gate reproduced that pass across `3` matched seed triplets and nearby threshold ablations. The broader-effectiveness gate blocked EE objective-contribution claims because the anti-collapse-throughput control explains the useful gain boundary and the QoS-sticky EE candidate harms handover / `r2` versus that control. The CP-base non-codebook continuous-power design and implementation-readiness gates passed only as boundary evidence, then the bounded matched pilot blocked effectiveness: the EE candidate lost `EE_system` to throughput + same guard + same continuous power, violated handover / `r2` guardrails, and only won scalar reward. See `hobs-active-tx-ee-modqn-feasibility.execution-report.md`, `hobs-active-tx-ee-anti-collapse-design-gate.execution-report.md`, `hobs-active-tx-ee-qos-sticky-anti-collapse-design-gate.execution-report.md`, `hobs-active-tx-ee-qos-sticky-robustness-gate.execution-report.md`, `hobs-active-tx-ee-qos-sticky-broader-effectiveness-gate.execution-report.md`, `hobs-active-tx-ee-non-codebook-continuous-power-design-gate.execution-report.md`, `hobs-active-tx-ee-non-codebook-continuous-power-implementation-readiness.execution-report.md`, and `hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot.execution-report.md`. |
| `04` | `NEEDS MORE EVIDENCE` | Single Catfish-MODQN is a reasonable feasibility design. Phase `04-B` produced runnable evidence for the opt-in implementation surface, and Phase `04C` completed a bounded attribution grid, but effectiveness is not promoted: the evidence is one seed / `20` episodes, best-eval rows are effectively tied, and asymmetric-gamma contribution is not distinguishable. See `reviews/04-single-catfish-modqn-feasibility.review.md` and `04c-single-catfish-ablation-attribution.execution-report.md`. |
| `05` | `CLOSED / NOT PROMOTED` | Phase `05A` failed objective-buffer distinctness, Phase `05R` passed guarded-residual buffer redesign, and Phase `05B` completed the bounded Multi-Catfish pilot. Runnable evidence passed, but acceptance / effectiveness failed: primary Multi-Catfish was worse than single Catfish on scalar reward, improved only `r2`, did not improve `r1` or `r3`, replay starvation counters were nonzero, and multi-buffer / single-learner plus random-buffer controls matched or explained away the result. Phase `05C` closes the current Multi-Catfish route as a bounded negative result / paper boundary finding. Do not promote Multi-Catfish-MODQN and do not continue with longer Phase `05B` training by default. See `05a-multi-buffer-validation.execution-report.md`, `05r-objective-buffer-redesign-gate.execution-report.md`, `05b-multi-catfish-planning.execution-report.md`, `05b-multi-catfish-bounded-pilot.execution-report.md`, `05c-multi-catfish-route-disposition.execution-report.md`, and `reviews/05-multi-catfish-modqn-validation.review.md`. |
| `06` | `BLOCKED FOR FINAL CLAIMS` | Final Catfish-EE-MODQN claims remain blocked because there is no promoted EE-MODQN route, no promoted Multi-Catfish route, and no valid bridge from Phase `05B` into Catfish-EE. Do not start Phase `06` validation from the current evidence. See `reviews/06-final-catfish-ee-modqn-validation.review.md` and `05c-multi-catfish-route-disposition.execution-report.md`. |
| `07A` | `PASS / READ-ONLY RECOVERY GATE` | Phase `07A` passed as a read-only recovery gate only. It did not by itself authorize Phase `07B` implementation/training, direct Phase `05B` continuation, Multi-Catfish promotion, or Phase `06` / Catfish-EE-MODQN. It narrowed the next R&D route to single-Catfish intervention utility / causal diagnostics, not Multi-Catfish tuning. See `07a-catfish-recovery-gate.execution-report.md`. |
| `07B` | `PASS / BOUNDED SINGLE-CATFISH UTILITY EVIDENCE` | Phase `07B` completed the bounded single-Catfish intervention-utility pilot. Primary shaping-off single Catfish beat matched MODQN, no-intervention, random / equal-budget injection, and replay-only single learner under the bounded protocol, with `r1` / `r3` support and no scalar-only success claim. The result is not a broader Catfish-MODQN effectiveness promotion because `r2` / handovers worsened and the primary branch was identical to no-asymmetric-gamma on aggregate metrics. See `07c-catfish-intervention-utility-disposition.execution-report.md` and `artifacts/catfish-modqn-phase-07b-bounded-pilot-summary/phase07b_bounded_pilot_summary.json`. |
| `07C` | `PASS / DISPOSITION COMPLETE` | Phase `07C` records the post-07B claim boundary: bounded single-Catfish intervention utility is supported, broader Catfish-MODQN effectiveness is not promoted, old Phase `05` Multi-Catfish reopening is blocked / deferred, and Phase `06` / Catfish-EE-MODQN remains blocked. It recommended Phase `07D` r2-guarded single-Catfish robustness planning, which has since completed and failed acceptance. |
| `07D` | `FAIL / R2-GUARDED RECOVERY NOT PROMOTED` | Phase `07D` completed the bounded r2-guarded single-Catfish robustness pilot. Implementation, configs, artifacts, diagnostics, and `27` required bounded runs completed, but acceptance failed: primary preserved scalar/component support, yet missed the predeclared r2 / handover non-inferiority margins versus matched MODQN (`r2_delta=-0.051667 < -0.02`, `handover_delta=+10.333333 > +5`) and the summary reports `starvation_stop_trigger_absent=false`. Do not promote broader Catfish-MODQN effectiveness, reopen old Phase `05` Multi-Catfish, or start Phase `06` from this evidence. See `07d-r2-guarded-single-catfish-robustness.execution-report.md` and `artifacts/catfish-modqn-phase-07d-r2-guarded-robustness-summary/phase07d_r2_guarded_robustness_summary.json`. |
| `RA-EE` | `PASS, SCOPED / BLOCKED FOR ASSOCIATION / RA-EE-09 NOT PROMOTED` | RA-EE-04, RA-EE-05, and RA-EE-07 support only fixed-association deployable non-oracle finite-codebook power allocation under the disclosed simulation envelope. RA-EE-06, RA-EE-06B, and RA-EE-08 block association proposal continuation because no proposal beats matched fixed association under the same deployable allocator. Do not reopen Phase `03C` or the RA-EE association proposal route. RA-EE-09 proved an auditable normalized resource-share comparison boundary, but the tested bounded QoS-slack resource-share candidate had negative held-out `EE_system` delta, negative predeclared resource-efficiency delta, and p05 ratio below `0.95`; RB / bandwidth effectiveness is not promoted. |
| `HEA-MODQN` | `PASS, SCOPED / THESIS UTILITY + RATIO-FORM EE_HO; GENERAL EE RELAXATION BLOCKED` | The high handover-cost / service-continuity-sensitive scoped pilot, robustness / attribution gate, and clean ratio-form artifact gate pass for the declared handover-aware utility `J` and ratio-form `EE_HO`. Primary subset positives are `15/15`, secondary subset positives are `6/6`, primary mean `J_delta=13288.26934925599`, primary candidate/control `EE_HO=414946.480518 / 412914.217690 bits/J`, `EE_HO_delta=+2032.262828 bits/J`, lambda-zero ablation primary wins are `0/15`, primary min p05 ratio is `0.9665274269295018`, served / outage deltas are `0`, and primary mean handover delta is `-43.733333333333334`. Low-cost `E_HO=3`, `lambda_HO=0.2` has `J_delta=0` and `EE_HO_delta=0`, so no low-cost success is claimed. Removing `E_HO,total` blocks relaxation to general EE: communication-only candidate wins are `0/30` overall, `0/15` primary, and `0/6` secondary. |
| `Catfish-over-HEA` | `PASS FOR BOUNDED READINESS + BASELINE PARITY; BOUNDED MATCHED PILOT BLOCKED` | The `ntn-sim-core` readiness slice adds a default-off Catfish training-batch intervention hook over the existing MODQN trainer. Disabled defaults are `catfish_enabled=false`, `intervention_policy_id=disabled`, and `intervention_ratio=0`; the disabled path returns the original sampled batch, records zero interventions, and the baseline parity gate passed with matching trainStep updates, epsilon, metrics, checkpoint protocol, replay checkpoint, DQN objective snapshots, update summaries, post-update greedy action, and q-values between default and explicit disabled trainers. The subsequent bounded matched pilot is blocked: mean `J_delta=-1136.0548587210574`, mean `EE_HO_delta=+225.88343883804046`, `J` non-worse cells were `7/15`, `EE_HO` non-worse cells were `13/15`, handover and `r2` worsened, and one seed triplet carried a large negative `J` result. This blocks the generic single training-batch intervention route as a claim-bearing Catfish-over-HEA method. |
| `Multi-Catfish redesign` | `GATE 1 NEEDS MORE DESIGN; GATE 1A CURRENT NEXT DESIGN GATE` | The current Catfish concept is reopened only as a flexible Multi-Catfish-first design line. Catfish is treated as external training stimulus plus high-quality experience distillation, not a fixed replay-injection implementation and not teacher-student imitation. The fixed HEA / `EE_HO` formula, simulator behavior, reward semantics, state/action surface, evaluation, and policy selection cannot change. Full Multi-Catfish should be designed first, with Single-Catfish represented as a collapsed ablation using the same coordinator and scoring surfaces. Gate 1 read-only diagnostics found only aggregate / seed / cell provenance, so transition-level scoring is not yet supported. See `2026-05-02-multi-catfish-redesign-plan.md` and `2026-05-02-multi-catfish-gate1a-transition-provenance-sdd.md`. |

## Current Execution Status

For implementation and experiment planning, distinguish review decisions from executable completion:

| Phase | Execution status | Next action |
|---|---|---|
| `01` | Complete for this research track | Use as the frozen disclosed comparison baseline. Do not reopen unless the goal changes to full paper-faithful reproduction. |
| `02` | Complete for formula plus Phase `02B` opt-in power-surface audit | Use only the disclosed Phase `02B` `active-load-concave` HOBS-compatible proxy for Phase `03` paired experiments. It remains a synthesized proxy, not a paper-backed power optimizer. |
| `03` | Bounded paired pilot plus Phase `03B` objective-geometry follow-up complete; Phase `03C-B` static power-MDP audit passed to bounded paired pilot; Phase `03C-C` bounded paired pilot complete and blocked; Phase `03D` disposition complete; current EE-MODQN route stopped | Do not claim EE-MODQN effectiveness. Current runtime/eval evidence still shows high rescaling risk, one-beam collapse, fixed denominator, single selected power profile, no throughput-vs-EE ranking separation, and p05-throughput collapse under the candidate. Do not continue this same route with more episodes, selector tweaks, reward tuning, or Catfish. EE can reopen only as a new resource-allocation MDP design gate with a renamed method family. |
| `HOBS active-TX EE` | Formula wiring, SINR audit, channel-regime audit, DPC denominator gate, tiny Route `D` learned-policy check, first capacity-aware anti-collapse gate, QoS-sticky overflow anti-collapse gate, QoS-sticky robustness / attribution gate, QoS-sticky broader-effectiveness gate, CP-base non-codebook continuous-power design gate, CP-base implementation-readiness slice, and CP-base bounded matched pilot complete; CP-base bounded pilot blocked | Keep the formula / DPC denominator work and QoS-sticky anti-collapse robustness result as scoped feasibility evidence only. Do not claim general EE-MODQN effectiveness. The broader-effectiveness gate is blocked: the QoS-sticky EE candidate does not improve active-TX EE beyond the anti-collapse-throughput control and violates the handover / `r2` guardrails versus that control. The current QoS-sticky EE objective-contribution route has reached its stop-loss point. The CP-base bounded pilot is also blocked: the EE candidate loses to throughput + same guard + same power on `EE_system`, violates handover / `r2`, and wins only scalar reward. Do not tune or rerun this CP-base candidate by default. |
| `04` | Separate bounded branch; Phase `04-B` runnable surface and Phase `04C` bounded attribution complete, not promoted for effectiveness | Do not rerun the same bounded grid by default. Keep original MODQN reward and use new Catfish-MODQN config / artifact namespaces. Any further Phase `04` work should target multi-seed / bounded robustness or clearer mechanism attribution, not EE repair. |
| `05` | Phase `05A`, `05R`, planning, bounded `05B` pilot, and Phase `05C` disposition complete | Treat Multi-Catfish as a bounded negative result. Do not claim Multi-Catfish effectiveness. Do not continue with longer training, shaping-on primary, ratio tuning, more seeds, or specialist tweaks by default. Any reopening requires a new explicit design gate that explains why controls and replay starvation would not explain the result. |
| `06` | Blocked for final Catfish-EE-MODQN claim | Do not start Phase `06` from current evidence. There is no promoted EE-MODQN route or promoted Multi-Catfish route to combine. |
| `07A` | Read-only recovery gate complete | Phase `07A` did not itself authorize Phase `07B` implementation/training. Direct Phase `05B` continuation, Multi-Catfish promotion, and Phase `06` remained blocked. It required any separately authorized recovery to be single-Catfish-first and compare against no-intervention, random/equal-budget replay injection, replay-only single learner, no-asymmetric-gamma, and matched MODQN control. |
| `07B` | Bounded single-Catfish intervention-utility pilot complete | Treat as bounded positive evidence for the single-Catfish intervention path only. Do not promote broader Catfish-MODQN effectiveness, because `r2` / handovers worsened and asymmetric gamma was not supported as an active mechanism. |
| `07C` | Read-only post-07B disposition complete | Phase `07C` led to Phase `07D` as an `r2`-guarded single-Catfish robustness gate. Do not reopen Multi-Catfish or Phase `06` from Phase `07B` alone. |
| `07D` | Bounded r2-guarded single-Catfish robustness pilot complete; acceptance failed | Treat as a bounded negative result for Catfish recovery promotion. Phase `07B` remains bounded utility evidence only; Phase `07D` blocks broader Catfish-MODQN effectiveness because r2 / handover guardrails did not hold versus matched MODQN. Default next action is paper synthesis / claim-boundary writing, not further guard tuning. |
| `RA-EE` | RA-EE-09 Slice `09E` matched replay complete; tested RB / bandwidth candidate not promoted | Preserve only the scoped simulated-EE fixed-association power-allocation claim. Stop learned association, hierarchical RL, joint association + power training, full RA-EE-MODQN, physical energy-saving, HOBS optimizer, and RB / bandwidth effectiveness claims. Do not continue the tested RA-EE-09 candidate by default; any new RB / bandwidth work requires a new explicit design gate. |
| `HEA-MODQN` | Scoped thesis pilot, attribution gate, and ratio-form artifact gate complete | Use only for thesis tables / figures / narrative polish under the frozen high handover-cost / service-continuity-sensitive sensitivity setting. Do not run new training, open Catfish, open Multi-Catfish, open Phase `06`, or claim active-TX EE recovery from this result. |
| `Catfish-over-HEA` | Readiness implementation slice and baseline parity gate complete in `ntn-sim-core`; bounded matched pilot complete and blocked | Keep the readiness / parity machinery as a useful precondition and negative evidence surface. Do not claim Catfish-over-HEA effectiveness. The generic enabled single intervention path worsened `J`, violated handover / `r2` guardrails, and is not the next development plan. |
| `Multi-Catfish redesign` | Documentation / design reset complete; Gate 1 read-only diagnostics returned `NEEDS MORE DESIGN`; no code, training, pilot, or artifact promotion | Use `2026-05-02-multi-catfish-redesign-plan.md` and `2026-05-02-multi-catfish-gate1a-transition-provenance-sdd.md` as the current design authorities. The next step is transition-provenance schema / coordinator diagnostics design, not Catfish implementation or a pilot. |

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
HOBS active-TX EE: post-03D formula / DPC denominator feasibility and QoS-sticky anti-collapse robustness (PASS scoped); broader EE objective contribution BLOCKED / stop current route
HEA-MODQN: scoped high handover-cost / service-continuity-sensitive thesis result (PASS scoped; thesis utility + ratio-form EE_HO only)
Catfish-over-HEA: default-off training-enhancement readiness and bounded pilot bookkeeping complete; generic single intervention pilot BLOCKED
Multi-Catfish redesign: current Catfish planning line; Gate 1 read-only diagnostics NEEDS MORE DESIGN; Gate 1A transition-provenance SDD is the next design step
Phase 04: Single Catfish-MODQN feasibility
Phase 05: Multi-Catfish-MODQN validation (closed; bounded 05B complete; not promoted)
Phase 06: Final Catfish-EE-MODQN validation (blocked)
Phase 07A: Catfish recovery gate (read-only complete; narrowed recovery route)
Phase 07B: Single-Catfish intervention utility pilot (bounded utility evidence; not promoted broadly)
Phase 07C: Post-07B disposition (complete; led to 07D)
Phase 07D: R2-guarded single-Catfish robustness (complete; acceptance failed)
```

Phase `04` can run after the Phase `03D` disposition if the immediate goal is only to prove that Catfish can attach to the MODQN backbone. It must then remain scoped as Catfish mechanism validation over the original MODQN reward and must not make any EE claim.

After the RA-EE closeout, Phase `04` is parked unless the explicit goal is
Catfish feasibility. It must not be framed as an EE repair path, a continuation
of Phase `03C`, or a way to recover learned association.

The Phase `04-B` minimum scope is `Catfish-MODQN` with new
`configs/catfish-modqn-*` and `artifacts/catfish-modqn-*` namespaces, baseline
main replay, high-value catfish replay, asymmetric gamma, periodic mixed replay
intervention, and shaping off in the primary run.

Phase `04C` bounded attribution has now run matched control, primary
shaping-off, no-intervention, and no-asymmetric-gamma branches under a
`20`-episode budget. Treat `04c-single-catfish-ablation-attribution.execution-report.md`
as the authority for this bounded result. It proves runnable attribution
instrumentation, not Catfish-MODQN effectiveness.

Phase `05A` bounded multi-buffer validation blocked escalation from the original
buffer construction. Phase `05R` then passed a guarded-residual objective-buffer
redesign gate using offline diagnostics only. Phase `05B` planning passed and
the bounded pilot has now run. It produced runnable evidence, but did not pass
the effectiveness gate: primary Multi-Catfish did not beat single Catfish, only
`r2` improved, `r1` / `r3` did not, replay starvation counters were nonzero, and
multi-buffer / single-learner plus random-buffer controls explained away the
result. Phase `05C` closes the current route. Treat this as a negative boundary
finding, not a promoted algorithm.

Phase `07A` read-only recovery gate passed first. It did not by itself authorize
Phase `07B` implementation/training, direct Phase `05B` continuation,
Multi-Catfish promotion, or Phase `06`. The Phase `05B` failure model was not
implementation failure as the primary explanation and not simple budget
insufficiency / longer training. The strongest working explanation is that
intervention utility was not proven beyond equal-budget controls; replay
starvation was an observed confounder and stop trigger. Phase `07B` was later
separately authorized and followed the required single-Catfish-first causal
diagnostics against no-intervention, random/equal-budget replay injection,
replay-only single learner, no-asymmetric-gamma, and matched MODQN controls
while keeping original MODQN reward semantics, `r1 = throughput`, `r2 =
handover penalty`, `r3 = load balance`, no EE, no Catfish-EE, no frozen
baseline mutation, primary shaping-off, and no scalar-reward-only success
claim.

Phase `07B` has since completed as a bounded single-Catfish intervention-utility
pilot. The primary shaping-off single-Catfish branch beat matched MODQN,
no-intervention, random / equal-budget injection, and replay-only single learner
under the bounded protocol, with `r1` / `r3` component support and no scalar-only
success claim. Phase `07C` records the disposition: this is bounded
single-Catfish intervention utility evidence, not broader Catfish-MODQN
effectiveness. `r2` / handovers worsened, and aggregate metrics did not support
asymmetric gamma as an active mechanism. Do not reopen the old Phase `05`
Multi-Catfish route or Phase `06` from this evidence alone. At Phase `07C`, the
next gate was Phase `07D` r2-guarded single-Catfish robustness planning.

Phase `07D` then tested that r2 / handover-guarded robustness path. The bounded
implementation and `27` required runs completed, but acceptance failed. Primary
still beat matched MODQN, no-intervention, random / equal-budget, and
replay-only on scalar, and component support was not scalar-only, but the
predeclared r2 / handover non-inferiority margins failed versus matched MODQN:
`r2_delta=-0.051667 < -0.02` and `handover_delta=+10.333333 > +5`. The summary
also reports `starvation_stop_trigger_absent=false`. Treat Phase `07D` as a
bounded negative result for Catfish recovery promotion. Default next action is
paper synthesis / claim-boundary writing, not more Catfish guard tuning.

RA-EE-09 is now closed for the tested candidate. Use
`ra-ee-09-fixed-association-rb-bandwidth.execution-report.md` as the result
authority and `ra-ee-09-completion-design-gate.md` only as the design-history
authority. Do not use the Phase `04` prompt for EE repair.

The HOBS active-TX EE route has a scoped robustness pass for the QoS-sticky
overflow candidate and a blocked broader-effectiveness result for the current
QoS-sticky EE objective-contribution route. Use
`hobs-active-tx-ee-modqn-feasibility.execution-report.md` for the formula / DPC
denominator boundary, `hobs-active-tx-ee-anti-collapse-design-gate.execution-report.md`
for the failed capacity-aware candidate,
`hobs-active-tx-ee-qos-sticky-anti-collapse-design-gate.execution-report.md` for
the scoped QoS-sticky anti-collapse pass,
`hobs-active-tx-ee-qos-sticky-robustness-gate.execution-report.md` for bounded
robustness / mechanism attribution, and
`hobs-active-tx-ee-qos-sticky-broader-effectiveness-gate.execution-report.md`
for the blocked broader-effectiveness / EE objective-contribution gate. This
does not authorize general EE-MODQN effectiveness claims. The current
QoS-sticky EE objective route remains stopped. The CP-base non-codebook
continuous-power design and implementation-readiness gates have passed as
readiness evidence only. The CP-base bounded matched pilot has now been
executed and blocked under its own stop conditions. If EE research continues,
it requires a new design gate with a materially different base-EE mechanism; do
not open another tuning pass on the current QoS-sticky route or rerun the
blocked CP-base candidate by default.

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
16. `docs/research/catfish-ee-modqn/04c-single-catfish-ablation-attribution.execution-report.md`
17. `docs/research/catfish-ee-modqn/05a-multi-buffer-validation.execution-report.md`
18. `docs/research/catfish-ee-modqn/05r-objective-buffer-redesign-gate.execution-report.md`
19. `docs/research/catfish-ee-modqn/prompts/05r-objective-buffer-redesign-gate.prompt.md`
20. `docs/research/catfish-ee-modqn/prompts/05b-multi-catfish-planning.prompt.md`
21. `docs/research/catfish-ee-modqn/05b-multi-catfish-planning.execution-report.md`
22. `docs/research/catfish-ee-modqn/prompts/05b-multi-catfish-implementation-draft.prompt.md`
23. `docs/research/catfish-ee-modqn/05b-multi-catfish-bounded-pilot.execution-report.md`
24. `docs/research/catfish-ee-modqn/05c-multi-catfish-route-disposition.execution-report.md`
25. `docs/research/catfish-ee-modqn/07a-catfish-recovery-gate.execution-report.md`
26. `docs/research/catfish-ee-modqn/07c-catfish-intervention-utility-disposition.execution-report.md`
27. `artifacts/catfish-modqn-phase-07b-bounded-pilot-summary/phase07b_bounded_pilot_summary.json`
28. `docs/research/catfish-ee-modqn/07d-r2-guarded-single-catfish-robustness.execution-report.md`
29. `artifacts/catfish-modqn-phase-07d-r2-guarded-robustness-summary/phase07d_r2_guarded_robustness_summary.json`
30. `docs/research/catfish-ee-modqn/ra-ee-02-oracle-power-allocation-audit.execution-report.md`
31. `docs/research/catfish-ee-modqn/ra-ee-04-bounded-power-allocator-pilot.execution-report.md`
32. `docs/research/catfish-ee-modqn/ra-ee-05-fixed-association-robustness.execution-report.md`
33. `docs/research/catfish-ee-modqn/ra-ee-06-association-counterfactual-oracle.execution-report.md`
34. `docs/research/catfish-ee-modqn/ra-ee-06b-association-proposal-refinement.execution-report.md`
35. `docs/research/catfish-ee-modqn/ra-ee-07-constrained-power-allocator-distillation.execution-report.md`
36. `docs/research/catfish-ee-modqn/ra-ee-08-offline-association-reevaluation.execution-report.md`
37. `docs/research/catfish-ee-modqn/ra-ee-09-completion-design-gate.md`
38. `docs/research/catfish-ee-modqn/ra-ee-09-fixed-association-rb-bandwidth.execution-report.md`
39. `docs/research/catfish-ee-modqn/prompts/ra-ee-09-completion-design-gate.prompt.md`
40. `docs/research/catfish-ee-modqn/hobs-active-tx-ee-modqn-feasibility.execution-report.md`
41. `docs/research/catfish-ee-modqn/hobs-active-tx-ee-anti-collapse-design-gate.execution-report.md`
42. `docs/research/catfish-ee-modqn/hobs-active-tx-ee-qos-sticky-anti-collapse-design-gate.execution-report.md`
43. `docs/research/catfish-ee-modqn/hobs-active-tx-ee-qos-sticky-robustness-gate.execution-report.md`
44. `docs/research/catfish-ee-modqn/hobs-active-tx-ee-qos-sticky-broader-effectiveness-gate.execution-report.md`
45. `docs/research/catfish-ee-modqn/hobs-active-tx-ee-non-codebook-continuous-power-design-gate.execution-report.md`
46. `docs/research/catfish-ee-modqn/hobs-active-tx-ee-non-codebook-continuous-power-implementation-readiness.execution-report.md`
47. `docs/research/catfish-ee-modqn/prompts/hobs-active-tx-ee-non-codebook-continuous-power-implementation-readiness.prompt.md`
48. `docs/research/catfish-ee-modqn/prompts/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot.prompt.md`
49. `docs/research/catfish-ee-modqn/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot.execution-report.md`
50. `artifacts/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot-summary/summary.json`
51. `ntn-sim-core/artifacts/handover-energy-aware-modqn-bounded-matched-pilot/summary.json`
52. `ntn-sim-core/artifacts/handover-energy-aware-modqn-bounded-matched-pilot/full-sensitivity-table.csv`
53. `internal/ntn-sim-core/devlogs/2026-05-02-handover-energy-aware-modqn-bounded-pilot.md`
54. `ntn-sim-core/src/core/experiments/catfish-over-hea-intervention.ts`
55. `ntn-sim-core/scripts/validate-catfish-over-hea-readiness.ts`
56. `internal/ntn-sim-core/devlogs/2026-05-02-catfish-over-hea-readiness-documentation-sync.md`
57. `docs/research/catfish-ee-modqn/ee-modqn-anti-collapse-controller-plan-2026-05-01.md`
58. `docs/research/catfish-ee-modqn/prompts/ee-modqn-anti-collapse-controller.prompt.md`
59. `docs/research/catfish-ee-modqn/prompts/ee-modqn-anti-collapse-worker.prompt.md`
60. `docs/research/catfish-ee-modqn/energy-efficient/README.md`
61. `docs/research/catfish-ee-modqn/energy-efficient/ee-formula-final-review-with-codex-2026-05-01.md`
62. `docs/research/catfish-ee-modqn/energy-efficient/modqn-r1-to-hobs-active-tx-ee-design-2026-05-01.md`
63. `docs/research/catfish-ee-modqn/repository-curation-2026-05-01.md`
64. `docs/ee-report.md`
65. `docs/research/catfish-ee-modqn/2026-05-02-multi-catfish-redesign-plan.md`
66. `docs/research/catfish-ee-modqn/2026-05-02-multi-catfish-gate1a-transition-provenance-sdd.md`

## Common Claim Rules

1. `Catfish-MODQN` may claim only Catfish mechanism feasibility over the original MODQN objective.
2. The current `EE-MODQN` r1-substitution route may not claim effectiveness; any future EE method must use a new explicit design gate and may claim only its own scoped objective-substitution or resource-allocation evidence.
3. Any future `Multi-Catfish` claim may claim specialist value only if compared against no-Catfish HEA and a collapsed Single-Catfish ablation under matched controls.
4. `Catfish-EE-MODQN` must be compared primarily against `EE-MODQN`, not only against original MODQN.
5. Any per-user EE reward adapted from system-level HOBS EE must be disclosed as a modeling assumption.
6. Throughput should be handled as an EE numerator plus QoS guardrail / reporting metric, not automatically as a fourth objective.
7. Phase `01` fixes the original reward surface as `r1 = throughput`, `r2 = handover penalty`, and `r3 = load balance`; reward-calibration, scenario-corrected, and beam-aware follow-on surfaces must not silently replace that baseline.
8. Phase `02` fixes the EE denominator as controlled / allocated beam transmit power in linear W; it must not be replaced by a path-loss closed form or a fixed config-power constant.
9. Phase `03` requires a paired `MODQN-control` vs `EE-MODQN` experiment on the same HOBS-linked SINR / power surface; old MODQN artifacts alone cannot establish the effect of objective substitution.
10. Phase `04` treats Catfish as a training strategy only; it must keep the original MODQN state, action, reward, and backbone fixed, with competitive shaping disabled in the primary run or isolated as an ablation.
11. Phase `05C` closes the old Multi-Catfish route as not promoted. Do not claim effectiveness, continue with longer training by default, or use scalar reward alone. The current Multi-Catfish redesign is a separate design-planning line, not a continuation of Phase `05B`.
12. Phase `06` / Catfish-EE-MODQN final claims are blocked under the current evidence because neither EE-MODQN nor Multi-Catfish has been promoted.
13. RA-EE fixed-association power allocation may be claimed only as a scoped offline replay result under the declared finite-codebook power contract.
14. Do not claim full RA-EE-MODQN, learned association effectiveness, hierarchical RL effectiveness, joint association + power training, RB / bandwidth allocation effectiveness, HOBS optimizer behavior, physical energy saving, or Catfish-EE effectiveness from RA-EE-02 through RA-EE-09.
15. Do not use scalar reward alone, per-user EE credit, oracle rows, or diagnostic association rows as success evidence for an EE or RA-EE method claim.
16. Phase `07A` blocks direct Phase `05B` continuation, old Multi-Catfish promotion, and Phase `06`; it does not authorize Phase `07B` implementation or training.
17. Phase `07B` was a historical single-Catfish-first intervention utility / causal diagnostics gate under original MODQN reward semantics and matched controls; it does not define the current Multi-Catfish-first redesign path.
18. Phase `07B` supports only bounded single-Catfish intervention utility evidence. Do not promote broader Catfish-MODQN effectiveness until handover / `r2` degradation is addressed and robustness is shown.
19. Do not claim asymmetric gamma as the active Phase `07B` mechanism; the bounded pilot reported primary and no-asymmetric-gamma as identical on aggregate metrics.
20. Phase `07D` failed the r2 / handover-guarded robustness gate. Do not promote broader Catfish-MODQN effectiveness, reopen Multi-Catfish, or start Phase `06` from Phase `07D`; default to paper synthesis / claim-boundary writing unless a new, materially different design gate is opened.
21. HOBS active-TX EE / DPC denominator feasibility does not promote EE-MODQN effectiveness. Denominator variability and same-policy throughput-vs-EE ranking separation are necessary diagnostics, but Route `D` is still blocked by `all_evaluated_steps_one_active_beam=true`.
22. The first capacity-aware anti-collapse candidate does not promote EE-MODQN effectiveness. It removed one-active-beam collapse, but failed p05 throughput and handover / `r2` guardrails; scalar reward improvement cannot override those failures.
23. The QoS-sticky overflow anti-collapse candidate passes only a bounded tiny anti-collapse gate. It may be described as preserving the HOBS active-TX EE / DPC denominator boundary while avoiding one-active-beam collapse on the matched pilot, but it does not establish general EE-MODQN effectiveness.
24. The QoS-sticky robustness gate strengthens the scoped anti-collapse evidence across the declared seed triplets and threshold ablations. Its mechanism attribution is sticky overflow reassignment under non-sticky / handover protections; the QoS ratio guard was not binding on this bounded surface. This still does not establish general EE-MODQN effectiveness.
25. The QoS-sticky broader-effectiveness gate blocks EE objective-contribution claims for the current route. The anti-collapse-throughput control explains the useful gain boundary, and the QoS-sticky EE candidate fails handover / `r2` guardrails versus that control.
26. The CP-base non-codebook continuous-power design gate and implementation-readiness slice pass only as a boundary. They require a rollout-time continuous `p_b(t)` surface shared by the EE candidate and throughput + same-anti-collapse control, with `r1` as the only intended difference. They do not authorize pilot training, EE-MODQN effectiveness, or Catfish-EE.
27. The CP-base bounded matched pilot blocks CP-base effectiveness. The matched boundary passed, but the EE candidate lost `EE_system` to throughput + same guard + same continuous power, violated handover / `r2` guardrails, had negative EE delta on all three seed triplets, and only won scalar reward.
28. Do not tune or rerun the blocked CP-base candidate by default. A future EE continuation requires a new design gate with a materially different base-EE mechanism and the same matched-control discipline.
29. Do not use Catfish or Multi-Catfish as the next repair for any HOBS active-TX EE limitation. Catfish may be reconsidered only around the scoped HEA-MODQN surface and only through a separately reviewed design gate.
30. HEA-MODQN may be claimed only under the frozen high handover-cost / service-continuity-sensitive sensitivity setting. Do not rewrite it as general EE-MODQN superiority, active-TX EE recovery, low-cost success, physical energy-saving, Catfish-EE readiness, Phase `06` readiness, Multi-Catfish repair, or scalar-reward success.
31. Final relaxation review blocks the communication-energy-only formula. If `E_HO,total` is removed from the denominator, the HEA candidate has `0/30` wins against matched throughput-control; do not reopen a general EE or active-TX claim from this evidence.
32. Catfish-over-HEA baseline parity is not Catfish effectiveness. The passed disabled train-step parity gate permits only discussion of pilot readiness, not a metric or sample-efficiency claim.
33. The Catfish-over-HEA bounded pilot is blocked. Do not continue the same generic single training-batch intervention path by tuning ratios, seeds, or longer training.
34. Future Catfish work must start from the Multi-Catfish-first redesign plan. Single-Catfish is a collapsed ablation, not the main path.
35. A redesigned Catfish method may adjust admission, specialist roles, quotas, coordinator logic, and intervention scheduling, but it must not alter HEA utility `J`, ratio-form `EE_HO`, reward semantics, simulator behavior, state/action surface, evaluation, policy selection, or active-TX EE claims.
36. Multi-Catfish cannot be claimed effective unless it beats both no-Catfish HEA and the collapsed single-Catfish ablation under matched seeds, budget, guardrails, and non-scalar evidence.
37. Gate 1 read-only Multi-Catfish diagnostics returned `NEEDS MORE DESIGN`. Do not implement Catfish scoring or run a pilot until a transition-provenance schema and coordinator diagnostics surface are accepted.
38. Do not infer transition-level Catfish scores from aggregate `J_delta`, aggregate `EE_HO_delta`, handover delta, scalar reward, active-TX EE, or seed-level summaries.

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
