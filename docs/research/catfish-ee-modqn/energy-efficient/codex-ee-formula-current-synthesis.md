# EE Formula Current Synthesis

Date: 2026-05-01

This note summarizes the current working understanding around the energy-efficiency
formula. It is a synthesis note for review, not a final paper claim and not an
implementation instruction.

## Purpose

The immediate question is how to define an energy-efficiency (EE) metric for a
LEO multi-beam system whose signal model is anchored on the HOBS-style SINR.

The formula should be decided first as a system/evaluation metric. Whether it
should later become a training reward, objective component, or `r1` replacement
is a separate downstream decision.

## Source Evidence Used So Far

### HOBS

HOBS is the primary source for the signal model and the simplest system EE
structure.

HOBS SINR uses per-beam transmit power:

```text
gamma_{n,m,k}(t)
= P_{n,m}(t) H_{n,m,k}(t) G^T_{n,m,k}(t) G^R_{n,m,k}(t) a_{n,m,k}(t)
  / (I^a_{n,m,k} + I^b_{n,m,k} + sigma^2)
```

HOBS throughput is latency-aware:

```text
R_{n,m,k}(t)
= (1 - C_n(t) / T_f) * W_{n,m} / U_{n,m}(t) * log2(1 + gamma_{n,m,k}(t))
```

HOBS system EE is:

```text
E_eff(t) = R_tot(t) / sum_n sum_m P_{n,m}(t)
```

HOBS also makes `P_{n,m}(t)` a dynamic power-control variable:

```text
P_{n,m}(t) = P_{n,m}(t - T_f) + xi^P_{n,m}(t)
```

The key point is that HOBS does not define EE as throughput divided by a fixed
constant. The denominator is a time-varying beam transmit-power sum.

### Other EE Papers

The other useful EE papers do not all share the same denominator, but they
support the broader rate-over-power / bits-per-Joule structure.

`PAP-2017-BEAMCOORD-EE` defines a richer total power consumption model:

```text
P_tot,b = (1 / eta) * sum_k ||w_k||^2 + P_CP,b + P_RD * delta(r_b)
P_CP,b = P_FIX + P_TC,b + P_CE + P_LP,b
```

This supports adding PA efficiency, circuit/static power, and rate-dependent
processing power. Its numeric parameters are terrestrial BS parameters and
should not be silently promoted to satellite-backed constants.

`PAP-2022-EESAT-RELIABLE` defines EE as correctly decoded bits over expected
HARQ energy. This supports the bits/Joule concept, but its HARQ-specific energy
model is not directly transferable to multi-beam LEO handover.

`PAP-2025-EEBH-UPLINK` defines per-terminal EE using:

```text
E_{j,l,t} = C_{j,l,t} / (P_l^t + P_sys)
```

This supports a maintenance-power term, but it is uplink terminal-focused, not
downlink active-beam transmit power.

The original Catfish thesis also uses a total-power denominator for RIS/BS
energy efficiency, including static, BS, and RIS-related power. It supports the
general rate-over-total-power pattern but should not be used as the direct
formula for a LEO multi-beam downlink metric.

## Current Project-Specific Understanding

### HOBS-Style EE Adaptation

The current HOBS-style adapted system metric has been:

```text
EE_system(t) = sum_i R_i(t) / sum_{b in active beams} P_b(t)
```

with:

```text
R_i(t) = B / N_b(t) * log2(1 + gamma_{i,b}(t))
```

This is not a complete copy of HOBS:

- HOBS sums `R_{n,m,k}` with beam-training latency in the numerator.
- The adapted metric sums served-user throughput available in the current
  simulator surface.
- HOBS sums all `P_{n,m}(t)`.
- The adapted metric sums active-beam power and sets inactive beams to `0 W`.
- HOBS uses dynamic power control.
- The current bounded positive evidence used a finite-codebook power allocator.

### Finite-Codebook Power

The finite-codebook contract used in the current bounded power-allocation work
is:

```text
P_b(t) in {0.5, 0.75, 1.0, 1.5, 2.0} W
inactive beam = 0 W
per-beam cap = 2.0 W
total active-beam budget = 8.0 W
```

This is not a physical satellite power table and not HOBS DPC. It is a bounded,
auditable, non-oracle discrete power-control approximation.

It is stronger than a constant denominator because the denominator can vary step
by step, but it may look weak as a final formula because other EE papers often
define power through richer formulas or continuous control variables.

### Current Evidence Boundary

The current positive evidence supports only a scoped statement:

```text
fixed association
+ deployable non-oracle finite-codebook power allocation
+ held-out matched replay
+ QoS / budget / per-beam / inactive-power guardrails
=> positive simulated EE_system result
```

This does not prove full learned EE control, physical energy saving, HOBS
optimizer behavior, or a complete paper-faithful reproduction.

### Reward / r1 Replacement Is Not The Immediate Question

The formula should first be defined as a system metric. Whether to use it as a
training reward is a later question.

The recommended sequence is:

```text
1. define SINR / throughput / power model
2. define system-level EE metric
3. decide whether any reward proxy should use the EE metric
```

Directly replacing a throughput objective with EE is unsafe if the policy cannot
control the denominator. A reward can become a throughput proxy when the
denominator is fixed or effectively fixed during evaluation.

## Candidate Formula Directions

### Version A: Strict HOBS-Style Transmit-Power EE

```text
EE_tx(t) = sum_i R_i(t) / sum_{b in active beams} P_b^tx(t)
```

Role:

- Best aligned with HOBS.
- Cleanest paper-backed denominator.
- Suitable as a reference metric or main metric if the paper wants the narrowest
  defensible claim.

Risk:

- Looks very close to HOBS.
- Does not model circuit/static/processing power.
- If `P_b^tx(t)` is not truly controlled, it can behave like throughput
  rescaling.

### Version B: HOBS-Inspired Bounded DPC EE

```text
EE_dpc(t) = sum_i R_i(t) / sum_{b in active beams} P_b^dpc(t)
P_b^dpc(t) = clip(P_b(t - T_f) + xi_b(t), 0, P_beam_max)
```

Role:

- More directly inspired by HOBS power-control logic than a finite codebook.
- Reduces the concern that the denominator is just a small enumerated set.
- Keeps the same HOBS-style transmit-power denominator.

Risk:

- Still not a full HOBS optimizer unless HOBS beam training / association /
  overhead logic is also reproduced.
- Requires a new design gate and matched evaluation.

### Version C: Composite Simulated System EE

```text
EE_comp(t)
= sum_i R_i(t)
  /
  [ sum_{b in active} P_b^tx(t) / eta_PA
    + |B_active(t)| * P_beam_on
    + P_RD * sum_b delta(R_b(t)) ]
```

Possible variants:

```text
P_total(t)
= P_static
  + sum_{b in active} (P_b^tx(t) / eta_PA + P_beam_on)
  + P_RD * sum_b delta(R_b(t))
```

Role:

- Most different from HOBS while remaining literature-grounded.
- Uses HOBS for the signal/transmit-power surface.
- Uses BeamCoord-style power-model structure for PA efficiency and
  rate-dependent processing power.
- Adds active-beam overhead, which is natural for beam-configuration research.

Risk:

- Satellite-specific values for `eta_PA`, `P_beam_on`, `P_static`, `P_RD`, and
  `delta(.)` may not be paper-backed in the current corpus.
- Should be called normalized or simulated composite EE unless parameter sources
  are fully defended.
- If fixed terms dominate, the metric may again track throughput too closely.

### Version D: System-Maintenance Power EE

```text
EE_sys(t) = sum_i R_i(t) / (sum_{b in active} P_b^tx(t) + P_sys)
```

Role:

- Simpler than Version C.
- Inspired by papers that include a maintenance or static system power term.

Risk:

- If `P_sys` is identical for all methods and too large, improvements are
  diluted and rankings may become throughput-like.
- If `P_sys` is not source-backed, it must be treated as sensitivity.

### Version E: Handover-Aware Energy Metric

Windowed form:

```text
EE_ho = total_bits_over_window
        / (total_power_energy_over_window + sum_handover_events E_HO)
```

Role:

- Useful if the target story explicitly includes handover signaling energy.
- Better as sensitivity or auxiliary reporting than the first main EE metric.

Risk:

- Handover event energy values are often scenario-specific.
- Mixing instantaneous W and event-level J requires careful time-window
  normalization.

### Version F: Utility Form, Not EE

```text
J = total_bits - lambda_HO * E_HO - other_costs
```

Role:

- Useful as a training objective or utility if a strict bits/Joule denominator
  becomes too assumption-heavy.

Risk:

- It is not EE and should not be presented as bits/Joule.

## Current Recommendation

Do not lock the final paper metric to the finite-codebook-only version without
further review. The finite-codebook result is useful evidence, but it may be
hard to defend as the final EE model if the committee expects a formula-driven
power denominator.

Recommended review path:

1. Keep HOBS-style transmit-power EE as the clean reference metric.
2. Ask a clean reviewer to compare whether the final main metric should be:
   - strict HOBS-style transmit-power EE, or
   - composite simulated EE with clearly separated paper-backed and assumption
     parameters.
3. If advisor acceptance is the concern, prioritize reviewing Version C
   composite EE, but require parameter disclosure and sensitivity sweeps.
4. If Version C lacks enough paper-backed satellite parameters, use it as
   sensitivity and keep Version A as the main metric.
5. Consider a HOBS-inspired DPC gate if the finite-codebook denominator looks too
   discrete or too implementation-specific.

## Unsafe Claims To Avoid

Do not claim:

- physical spacecraft energy saving,
- complete satellite power consumption modeling,
- HOBS optimizer reproduction,
- full learned EE control,
- full paper-faithful reproduction,
- that finite-codebook power levels are hardware-backed satellite constants,
- that a scalar reward increase alone proves EE improvement,
- that per-user EE credit is the same as system EE.

Safe phrasing:

```text
simulated system EE under a declared transmit-power or composite-power contract
```

or:

```text
HOBS-style rate-over-beam-transmit-power EE, extended with explicitly disclosed
composite power terms for sensitivity.
```

## Questions For The Next Review

1. Should the final main metric be strict HOBS-style transmit-power EE or
   composite simulated EE?
2. If composite EE is recommended, which terms are paper-backed and which must
   remain sensitivity assumptions?
3. Is active-beam overhead justified by the current literature, or should it be
   introduced only as a simulation assumption?
4. Should rate-dependent processing power be included in the main denominator,
   or only in sensitivity?
5. Is a HOBS-inspired DPC denominator needed to replace or supplement finite
   codebook power allocation?
6. Which formula best balances defensibility, advisor acceptance, and avoiding a
   near-copy of HOBS Eq. (13)?

