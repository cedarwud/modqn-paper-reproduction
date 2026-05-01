# HOBS Active-TX EE QoS-Sticky Broader-Effectiveness Gate Execution Report

**Date:** `2026-05-01`
**Status:** `BLOCK`
**Method label:** `QoS-sticky HOBS-active-TX EE-MODQN`
**Scope:** new extension / method variant; bounded active-TX EE
validation only; not full EE-MODQN; not physical energy saving; not
HOBS optimizer reproduction; not RA-EE association; not Catfish-EE.

## Protocol

- Roles: `matched-throughput-control`, `hobs-ee-control-no-anti-collapse`, `qos-sticky-ee-candidate`, `anti-collapse-throughput-control`.
- Seed triplets: `[42, 1337, 7]`, `[43, 1338, 8]`, `[44, 1339, 9]`.
- Evaluation seeds: `[100, 200, 300, 400, 500]`.
- Episode budget: `5` per role / seed triplet.
- Scalar reward is diagnostic only.

## Matched Boundary Proof

`matched_boundary_pass=True`.
All roles use the same environment boundary, HOBS-inspired DPC sidecar,
seed triplets, eval seeds, bounded training budget, eval schedule,
checkpoint protocol, objective weights, and hyperparameters. The only
intended differences are `r1_reward_mode` and whether the QoS-sticky
anti-collapse hook is enabled. The matched-throughput-control is a
DPC-matched throughput control, not the frozen MODQN baseline.

## Role Table

| Role | r1 | anti-collapse | EE_system | throughput | p05 | served | outage | handovers | r2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `matched-throughput-control` | `throughput` | `False` | `873.3451493521717` | `934.4793098068236` | `5.927901649475099` | `1.0` | `0.0` | `423.6666666666667` | `-0.4236666666666667` |
| `hobs-ee-control-no-anti-collapse` | `hobs-active-tx-ee` | `False` | `872.981750872254` | `729.9828142178058` | `4.214356013139089` | `1.0` | `0.0` | `590.3333333333334` | `-0.5903333333333334` |
| `qos-sticky-ee-candidate` | `hobs-active-tx-ee` | `True` | `873.0012922050197` | `7048.5868065198265` | `13.08817702929179` | `1.0` | `0.0` | `298.3333333333333` | `-0.29833333333333334` |
| `anti-collapse-throughput-control` | `throughput` | `True` | `873.3226745862389` | `7044.184608853658` | `13.624501959482828` | `1.0` | `0.0` | `215.0` | `-0.215` |

## Control Comparisons

| Control | EE delta | throughput delta | p05 ratio | served delta | outage delta | handover delta | r2 delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| `matched-throughput-control` | `-0.34385714715199356` | `6114.107496713003` | `2.2078937545211663` | `0.0` | `0.0` | `-125.33333333333337` | `0.12533333333333335` |
| `hobs-ee-control-no-anti-collapse` | `0.019541332765697916` | `6318.603992302021` | `3.105617320531727` | `0.0` | `0.0` | `-292.00000000000006` | `0.29200000000000004` |
| `anti-collapse-throughput-control` | `-0.32138238121922313` | `4.402197666168831` | `0.9606352634550617` | `0.0` | `0.0` | `83.33333333333331` | `-0.08333333333333334` |

## Verdicts

- Anti-collapse mechanism: `BLOCK`
- EE objective contribution: `BLOCK`

## Acceptance Result

`PASS / BLOCK / NEEDS MORE DESIGN: BLOCK`

## Stop Conditions Triggered

- candidate harms protected QoS / handover / r2 guardrails
- candidate wins only scalar reward
- anti-collapse-throughput-control explains all gains

## Allowed Claim Boundary

- QoS-sticky overflow reassignment can be discussed only as a bounded anti-collapse mechanism if that verdict passes.
- Any EE statement is limited to active-TX EE under the disclosed HOBS-inspired DPC sidecar and this bounded protocol.
- The matched-throughput-control is a DPC-matched throughput-objective control, not the frozen MODQN baseline.
- Scalar reward remains diagnostic only.

## Forbidden Claims Still Active

- general EE-MODQN effectiveness
- physical energy saving
- HOBS optimizer reproduction
- full RA-EE-MODQN
- learned association effectiveness
- Catfish / Multi-Catfish / Catfish-EE repair
- scalar reward success
- denominator variability alone proves energy-aware learning
- QoS-sticky robustness PASS means general EE-MODQN effectiveness
- Phase 03D failure has been overturned

## Artifact Paths

- `artifacts/hobs-active-tx-ee-qos-sticky-broader-effectiveness-*/`
- `artifacts/hobs-active-tx-ee-qos-sticky-broader-effectiveness-summary/summary.json`
- `artifacts/hobs-active-tx-ee-qos-sticky-broader-effectiveness-summary/role_table.csv`
- `artifacts/hobs-active-tx-ee-qos-sticky-broader-effectiveness-summary/control_comparisons.csv`

## Reasons

- Anti-collapse: handover guard failed vs anti-collapse-throughput-control
- Anti-collapse: r2 guard failed vs anti-collapse-throughput-control
- EE objective: anti-collapse / guardrail verdict is not PASS
- EE objective: anti-collapse-throughput-control explains the EE/ranking gain boundary
- EE objective: candidate wins only scalar reward vs anti-collapse-throughput-control
