# Review: Phase 02 HOBS-Linked EE Formula Validation

**Date:** `2026-04-28`  
**Decision:** `PROMOTE`  
**Scope:** EE formula validation only; no trainer, Catfish strategy, reward weighting, or implementation change is promoted by this review.

## Formula Recommendation

Adopt the following system-level EE formula with explicit provenance labels:

```text
EE_system(t) = sum_i R_i(t) / sum_active_beams P_b(t)
```

This is coherent with HOBS because HOBS uses `P_{n,m}(t)` in both:

```text
SINR numerator: P_{n,m}(t) H G^T G^R a
EE denominator: E_eff(t) = R_tot(t) / sum_n sum_m P_{n,m}(t)
```

`P_{n,m}(t)` is a downlink per-beam transmit-power control / allocation variable. It is not a closed-form quantity directly derived from path loss, distance, or antenna gain. Path loss can affect the channel term, or a separately disclosed synthesized required-power inversion, but it must not replace the HOBS power variable itself.

## Provenance Classification

1. HOBS SINR `P_{n,m}(t)`: paper-backed / directly adopted.
2. HOBS EE denominator `sum_n sum_m P_{n,m}(t)`: paper-backed / directly adopted.
3. `EE_system(t) = sum_i R_i(t) / sum_active_beams P_b(t)`: HOBS-equivalent notation normalization / synthesized closure.
4. `r1_i(t) = R_i(t) / (P_{b_i}(t) / N_{b_i}(t))`: modeling / credit-assignment assumption.
5. Throughput as QoS guardrail / reporting metric: method-design guardrail, not a direct HOBS formula.

## Assumptions

1. `P_b(t)` is the same beam transmit power used in the SINR numerator.
2. `P_b(t)` uses linear W.
3. Inactive beams either have `P_b(t) = 0` or are excluded from `active_beams`.
4. `N_b(t)` is the number of users served by beam `b`.
5. Per-user reward uses equal power attribution `P_b(t) / N_b(t)`.

The per-user form is not a claim that each user has a separate physical beam power. It is only a reward credit-assignment assumption.

## Risks

The main risk is reducing EE to:

```text
EE = throughput / fixed_power
```

If `P_b(t)` is a fixed constant, maximum power value, static profile default, or otherwise unrelated to action / allocation / active beam state, then the EE reward becomes only a scaled throughput reward and loses energy-aware meaning.

A second risk is that summing per-user rewards may not equal system EE. Per-user EE may guide agent-level credit, but final evaluation must report `EE_system`.

## Guardrails

1. Do not define `P_b(t)` as a path-loss closed form.
2. `P_b(t)` must come from controlled / allocated beam power, not static config power.
3. EE denominators must use linear W, not mixed dBm / dBW units.
4. `r1_i(t)` must be labeled as a modeling / credit-assignment assumption.
5. Reports must include `EE_system`, average throughput, low-percentile throughput, served / unserved ratio, and total active beam power.
6. Throughput remains a QoS guardrail / reporting metric, not a fourth objective.
7. Circuit power, PA efficiency, idle / off power, and handover energy require a separate assumption set and sensitivity analysis if added later.

## Recommendation For EE-MODQN

Proceed to EE-MODQN using the HOBS-linked active-transmit-power EE formula.

The system-level EE formula is defensible as HOBS-linked. The per-user formula may be used for MODQN adaptation only when explicitly disclosed as credit assignment and checked against system-level EE reporting.

## Result

`PROMOTE`.

The formula is coherent enough to support Phase 03, provided all provenance labels and guardrails above are carried forward.
