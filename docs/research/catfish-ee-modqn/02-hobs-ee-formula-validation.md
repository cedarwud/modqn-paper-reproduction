# Phase 02: HOBS-Linked EE Formula Validation

**Status:** Promoted as HOBS-linked active-TX EE formula  
**Question:** Is the proposed EE formula coherent with HOBS SINR and power semantics?

## Decision

`PROMOTE`, limited to a HOBS-linked active-transmit-power EE definition.

The system-level formula can enter the EE-MODQN phase:

```text
EE_system(t) = sum_i R_i(t) / sum_active_beams P_b(t)
```

This is a HOBS-equivalent notation normalization because HOBS uses the same `P_{n,m}(t)` power semantics in both the SINR numerator and the EE denominator. The per-user training reward form can be used only as a disclosed MODQN credit-assignment adaptation, not as a directly adopted HOBS formula.

## Why This Is Independent

If the EE formula is weak, every later EE-MODQN and Catfish-EE-MODQN result will be scientifically fragile. This phase validates the formula before any training-method changes.

## Core Position To Review

HOBS uses `P_{n,m}(t)` as a downlink per-beam transmit-power control / allocation variable:

```text
SINR numerator: P_{n,m}(t)
EE denominator: sum_n sum_m P_{n,m}(t)
```

The first EE definition should therefore keep the same power semantics:

```text
EE_system(t) = sum_i R_i(t) / sum_active_beams P_b(t)
```

For per-user MODQN training, a possible adaptation is:

```text
r1_i(t) = R_i(t) / (P_{b_i}(t) / N_{b_i}(t))
```

This per-user form is not directly HOBS-paper-backed. It is a credit-assignment assumption that must be disclosed.

`P_{n,m}(t)` must not be rewritten as a distance / path-loss / antenna-gain closed form. Path loss may affect the channel term or a synthesized required-power inversion, but the HOBS power variable itself remains the controlled or allocated downlink beam transmit power.

## Inputs

1. HOBS formula / catalog entries.
2. `system-model-refs` power taxonomy.
3. Paper-catalog EE and power formula reports.
4. Current MODQN per-user reward structure.

## Non-Goals

1. Do not design Catfish.
2. Do not tune reward weights.
3. Do not add circuit / processing / handover energy unless marked as future sensitivity.
4. Do not introduce throughput as a fourth objective.

## Checks

1. Does `P_{n,m}(t)` in SINR match the EE denominator?
2. Is HOBS beam power correctly treated as a control / allocation variable, not a path-loss closed form?
3. Does `EE_system = sum R / sum P` follow HOBS semantics?
4. Is the per-user reward adaptation defensible?
5. Does throughput need a QoS guardrail rather than becoming a fourth objective?
6. Does the formula avoid the fixed-power trap:

```text
EE = throughput / constant
```

## Provenance Classification

1. `P_{n,m}(t)` in HOBS SINR: paper-backed / directly adopted.
2. HOBS EE denominator `sum_n sum_m P_{n,m}(t)`: paper-backed / directly adopted.
3. `EE_system(t) = sum_i R_i(t) / sum_active_beams P_b(t)`: HOBS-equivalent notation normalization / synthesized closure.
4. `r1_i(t) = R_i(t) / (P_{b_i}(t) / N_{b_i}(t))`: modeling / credit-assignment assumption.
5. Throughput as QoS guardrail / reporting metric: method-design guardrail, not a direct HOBS formula.

## Assumptions

1. `P_b(t)` is the same beam transmit power used by the SINR numerator.
2. `P_b(t)` uses linear W, not dBm or dBW.
3. Inactive beams have `P_b(t) = 0` or are excluded from `active_beams`.
4. `N_b(t)` is the number of users served by beam `b`.
5. Per-user EE reward uses equal power attribution `P_b(t) / N_b(t)` for reward credit assignment only.

## Guardrails

1. Do not define `P_b(t)` as a path-loss closed form.
2. Do not use fixed max power or static config power as the EE denominator.
3. Report `EE_system`, per-user EE, average throughput, low-percentile throughput, served / unserved ratio, and total active beam power.
4. Keep throughput as an EE numerator plus QoS guardrail / reporting metric, not a fourth objective.
5. If circuit power, PA efficiency, idle power, or handover energy is added later, put it under a separate assumption set and sensitivity analysis.

## Decision Gate

Promote only if the reviewer can defend:

1. system-level EE as HOBS-backed,
2. per-user EE as a disclosed adaptation,
3. throughput as numerator plus QoS guardrail / metric.

Stop if the power denominator is fixed, unrelated to HOBS `P`, or derived from an incompatible uplink / received-power formula.

The accepted Phase 02 review is recorded in `reviews/02-hobs-ee-formula-validation.review.md`.

## Expected Output

A formula-validation report:

1. proposed formula,
2. provenance classification,
3. assumptions,
4. risks,
5. minimum reporting metrics,
6. recommendation for EE-MODQN.
