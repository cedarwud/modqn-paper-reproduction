# HOBS Active-TX EE MODQN Feasibility Execution Report

**Date:** `2026-05-01`
**Status:** formula / wiring / denominator gates passed; learned-policy pilot blocked
**Scope:** opt-in HOBS-style active transmit-power EE reward and HOBS-inspired DPC denominator diagnostics for MODQN. This report does not authorize Catfish, RA-EE association, HOBS optimizer, physical energy-saving, or EE-MODQN effectiveness claims.

## Current State

The HOBS active-TX EE formula can be wired into MODQN as an opt-in `r1` mode:

```text
r1_hobs_active_tx_ee(t) =
  sum_u R_u(t) / (sum_active_beams z_b(t) * p_b(t) + epsilon)
```

The numerator is the current simulator throughput sum. The denominator is the
active-beam transmit-power sum in linear W, with inactive beams contributing
`0 W`. This is a transmission-side active-beam EE metric, not total spacecraft
energy efficiency.

The latest route no longer has the old fixed-denominator blocker in the narrow
sense: with the HOBS-inspired DPC sidecar, total active transmit power varies
under greedy evaluation. However, the learned MODQN policy still collapses to
one active beam on every evaluated step in the tiny matched pilot. Therefore the
current route remains blocked for EE-MODQN effectiveness.

## Evidence Summary

| Gate | Result | Evidence boundary |
|---|---|---|
| Formula policy | `PASS, SCOPED` | Use active-beam transmit-power EE as the main metric. Composite / circuit / handover-aware EE may be sensitivity only unless separately sourced and validated. |
| Wiring feasibility | `PASS` | `StepResult.active_beam_mask`, `StepResult.beam_transmit_power_w`, `StepResult.total_active_beam_power_w`, and throughput are available at reward-computation time. New `hobs-active-tx-ee` mode is opt-in; baseline `r1=throughput` remains unchanged. |
| Slice B structural gate | `BLOCK current-config Route D` | Under SNR-only + active-load-concave power, HOBS active-TX EE is numerically equivalent to the old Phase `03B` per-user EE credit in the relevant uniform / proportional cases. A tiny learned pilot on that surface would likely repeat Phase `03B` collapse. |
| Route A SINR audit | `PASS structural / weak practical signal` | Opt-in intra-satellite SINR interference is structurally correct: one active beam matches SNR; multiple active beams reduce SINR; inactive beams do not interfere. At the MODQN operating point, `I_intra/N0` is about `1.1e-5`, so the effect is numerically negligible. |
| Route A2 channel-regime audit | `Path 1 BLOCK / Path 2 PASS` | No MODQN-paper-backed or S10-backed channel parameter makes interference observable. Adding antenna gain would be a larger model extension. The recommended denominator route is the HOBS-inspired DPC sidecar. |
| Route B DPC sidecar denominator gate | `PASS` | Under heuristic hold-current diagnostics, DPC produced `denominator_varies_in_eval=true`, `active_power_single_point_distribution=false`, `power_control_activity_rate=1.0`, `10` distinct total active-power values, `dpc_sign_flip_count=27`, and zero power guardrail violations. |
| Route D tiny learned-policy check | `BLOCK` | Matched `5`-episode control/candidate training completed. Boundary matched. Candidate had `denominator_varies_in_eval=true`, `active_power_single_point_distribution=false`, throughput-vs-EE Pearson `0.19303453314619476`, and same-policy ranking separation, but `all_evaluated_steps_one_active_beam=true` across `50` evaluated steps. |

## Route D Matched Boundary

The tiny learned-policy denominator check compared:

```text
control:
  r1 = throughput
  hobs_power_surface_mode = hobs-dpc-sidecar

candidate:
  r1 = hobs-active-tx-ee
  hobs_power_surface_mode = hobs-dpc-sidecar
```

Matched boundary proof passed:

1. same DPC sidecar parameters,
2. same seeds `100, 200, 300, 400, 500`,
3. same `5`-episode budget,
4. same evaluation schedule,
5. same checkpoint rule,
6. same environment, objective weights, and trainer hyperparameters,
7. only intended difference was `r1_reward_mode`.

Artifacts:

```text
artifacts/hobs-dpc-denominator-check-control/
artifacts/hobs-dpc-denominator-check-candidate/
artifacts/hobs-dpc-denominator-check-candidate/paired-comparison-vs-control/
```

## Route D Diagnostics

Candidate learned greedy evaluation:

```text
denominator_varies_in_eval: true
all_evaluated_steps_one_active_beam: true
active_beam_count_distribution: {"1.0": 50}
active_power_single_point_distribution: false
distinct_total_active_power_w_values: [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
power_control_activity_rate: 1.0
dpc_sign_flip_count: 20
throughput_vs_ee_pearson: 0.19303453314619476
same_policy_throughput_vs_ee_rescore_ranking_change: true
raw_throughput_mean_bps: 934.4793098068237
p05_throughput_bps: 5.927901649475098
served_ratio: 1.0
handover_count: 423
```

The denominator is no longer constant, and EE is no longer merely a fixed scalar
rescale of throughput under this diagnostic. The remaining hard blocker is the
learned policy's one-active-beam collapse.

## Decision

```text
HOBS active-TX EE formula: PASS, scoped
MODQN reward wiring: PASS, opt-in only
SINR structural audit: PASS, but numerically weak at current MODQN operating point
HOBS-inspired DPC denominator gate: PASS
Tiny learned-policy Route D: BLOCK
EE-MODQN effectiveness: NOT PROMOTED / BLOCKED
```

This route improves the evidence chain relative to old Phase `03`: the
denominator can vary, and same-policy throughput-vs-EE ranking can separate.
It does not solve the learned association / beam-selection collapse. The
current MODQN beam-only action surface still permits a degenerate learned
policy that serves every evaluated step through one active beam.

## Recommended Next Step

Do not scale Route D training by default. More episodes do not address the
observed hard stop.

If EE-MODQN is continued, open a new explicit anti-collapse design gate before
additional training. That gate must introduce and test a mechanism that prevents
or penalizes one-active-beam collapse, such as capacity-aware action masking,
active-beam diversity / load-spread constraints, overload penalties,
centralized assignment constraints, or a renamed resource-allocation MDP with
explicit resource actions.

Catfish / Multi-Catfish should not be used as the next repair step for this
blocker. Catfish changes replay / intervention dynamics; it does not add a
missing beam-capacity, assignment, or power-control action surface. It can only
be reconsidered after a base EE method has passed the anti-collapse gate.

## Allowed Claims

Allowed:

1. HOBS active-TX EE is computable inside the MODQN environment as an opt-in
   reward / metric mode.
2. The active transmit-power denominator can be made variable through the
   HOBS-inspired DPC sidecar.
3. Route D showed denominator variability and throughput-vs-EE ranking
   separation under a matched tiny learned-policy check.
4. Route D also showed the learned policy still collapses to one active beam on
   every evaluated step.

## Forbidden Claims

Do not claim:

1. EE-MODQN effectiveness,
2. old Phase `03` failures are overturned,
3. physical spacecraft energy saving,
4. HOBS optimizer reproduction,
5. DPC is MODQN-paper-backed,
6. Catfish, Multi-Catfish, or Catfish-EE effectiveness,
7. RA-EE learned association or full RA-EE-MODQN,
8. scalar reward as success evidence,
9. that denominator variability alone proves energy-aware learning,
10. that more Route D training is the default next gate.

## Tests / Checks Reported

The execution reports feeding this closeout reported:

```text
HOBS active-TX EE feasibility: 16 new + 109 existing tests passed
Slice B structural gate: 131 passed
Route A SINR audit: 242 passed
Route A2 channel-regime audit: 80 key tests passed; full suite had 411 passed and 1 pre-existing refactor-golden failure
Route B DPC denominator gate: 177 passed
Route D denominator check focused suite: 62 passed
```

This document did not rerun tests; it records the latest reported evidence and
artifact state.
