# EE Formula Final Review With Codex.md

Date: 2026-05-01

Scope: final review of all current `energy-efficient/` notes after adding
`codex.md`. This is a review/conclusion document, not an implementation
instruction.

No code was changed, no experiments were run, and no source documents outside
this review file were modified.

## Source Files Reviewed

- `AGENTS.md`
- `energy-efficient/gemini.md`
- `energy-efficient/claude.md`
- `energy-efficient/codex.md`
- `energy-efficient/codex-ee-formula-current-synthesis.md`
- `system-model-refs/system-model-formulas.md`
- `system-model-refs/system-model-derivation.md`
- `system-model-refs/simulator-parameter-spec.md`
- `system-model-refs/simulator-parameter-provenance-inventory.md`
- `system-model-refs/hobs-vs-closed-form-power-thesis-summary-2026-04-21.md`
- `system-model-refs/hobs-vs-closed-form-power-report-2026-04-21.md`
- `paper-catalog/paper-audit-report.md`
- EE-related catalog and text/layout extracts for HOBS, BEAMCOORD-EE,
  EEBH-UPLINK, EESAT-RELIABLE, EAQL, EAQL-DRL, SMASH-MADQL,
  HO-OVERHEAD, D2AOF-HOPOWER, and MAAC-BHPOWER.

## Agreement Across Files

The four `energy-efficient/` review files now converge on these points:

1. HOBS is the correct signal-model anchor.
2. HOBS SINR uses the same beam transmit power variable in desired signal and
   interference terms.
3. HOBS EE is a transmit-power EE:
   throughput divided by the sum of beam transmit powers.
4. HOBS does not define total consumed spacecraft power, circuit/static power,
   processing power, active-beam hardware overhead, or handover event energy.
5. BEAMCOORD-EE supports a richer denominator structure, but its numeric values
   are terrestrial multicell MISO values, not LEO satellite constants.
6. EEBH-UPLINK supports a `TX power + P_sys` pattern, but it is uplink
   terminal-side, not downlink satellite-beam-side.
7. EESAT-RELIABLE supports bits-per-Joule framing for HARQ/reliability, but its
   expected-energy denominator is not a direct HOBS multi-beam denominator.
8. Handover energy is useful for sensitivity or utility objectives, but the
   current corpus does not support one universal LEO per-HO energy default.

## Conflicts / Disagreements

The main disagreement is the role of composite EE.

- `gemini.md` recommends composite simulated system EE as the main metric.
- `claude.md` recommends HOBS-style transmit-power EE as the main metric and
  composite EE only as sensitivity.
- `codex.md` also recommends HOBS-style active transmit-power EE as the main
  metric, with composite EE as ablation/sensitivity.
- `codex-ee-formula-current-synthesis.md` leaves the decision open, but says
  that if composite EE lacks satellite-specific parameters it should remain
  sensitivity while HOBS-style EE remains the main metric.

After reading `codex.md`, the final judgment shifts more firmly away from
`gemini.md`: composite EE is technically useful, but it is not safe as the
headline physical EE metric under the current evidence boundary.

There is also a local documentation tension: `system-model-derivation.md`
foregrounds a HO-aware EE form, while `simulator-parameter-spec.md` quarantines
`E_HO`, `P_c`, `rho`, and active/idle/off beam powers as open gaps. For final
claims, the gap governance in `simulator-parameter-spec.md` should control.

## Paper-Backed Formula Components

The following components are safe to use in the main formula:

- HOBS-style SINR:
  `gamma = signal / (intra-interference + inter-interference + noise)`.
- Beam transmit power `p_{s,b}^t` as the variable that drives both SINR and EE
  denominator.
- HOBS-style throughput, including beam-training overhead if the model actually
  includes `C_n(t) / T_f`.
- Time-aggregated bits-per-Joule:
  total delivered bits over total transmit energy.
- Inactive beam closure through `z_{s,b}^t` and `p_{s,b}^t = 0` when inactive.
- Per-beam and per-satellite transmit-power caps, with the current default caps
  sourced from S10 / MAAC-BHPOWER rather than HOBS.

## Assumption / Sensitivity Components

The following components must not be promoted to paper-backed main-formula
parameters without new LEO-specific sources:

- PA efficiency `rho_s`.
- Satellite circuit/static power `P_c,s` or `P_static`.
- Active-beam hardware overhead `P_beam,on`.
- Idle/off beam power.
- Rate-dependent satellite processing power `P_RD delta(R)` or `P_RD R^m`.
- Per-handover event energy `E_HO`.
- Any fixed active/idle/off beam values such as `20 / 5 / 0.1 W`.
- Finite-codebook power levels, if described as hardware-backed constants.

## Candidate Final Formula Options

### Option A: HOBS-Style Transmit-Power EE

```text
eta_EE,active-TX =
  (sum_t sum_u R_u^t Delta_t)
  /
  (sum_t sum_s sum_b z_{s,b}^t p_{s,b}^t Delta_t)
```

Role: main metric.

This is the cleanest and safest final EE formula. It uses the same power
variable as the HOBS SINR and does not introduce unsupported denominator terms.
It should be named `transmission-side EE`, `active-beam transmit-power EE`, or
`HOBS-style transmit-power EE`, not total spacecraft EE.

The explicit `z_{s,b}^t` term makes active-beam accounting visible in the paper
formula. This is equivalent to the inactive-beam closure `p_{s,b}^t = 0` when
inactive, but is clearer for readers and reviewers. It is a reader-facing
explicitization of the active-beam accounting, not an additional semantic
constraint beyond the model's existing activity / inactive-power closure.

### Option B: Composite Simulated Communication EE

```text
P_comm,s^t =
  P_c,s
  + (1 / rho_s) * sum_b z_{s,b}^t p_{s,b}^t
  + P_beam,on * sum_b z_{s,b}^t

eta_comm =
  (sum_t sum_u R_u^t Delta_t)
  /
  (sum_t sum_s P_comm,s^t Delta_t)
```

Role: ablation / sensitivity.

This is structurally defensible, but `P_c,s`, `rho_s`, and `P_beam,on` remain
assumption-backed. It is useful to test whether the algorithm ranking survives a
broader communication-power model.

### Option C: Rich Composite + Handover-Aware EE

```text
eta_rich =
  total_bits
  /
  (total_composite_energy + sum_t sum_u delta_HO,u^t * E_HO)
```

Role: sensitivity only.

This can be reported only with explicit `E_HO` sweep values and assumption
disclosure. EAQL's `3 J` may be one scenario point, not a universal default.

### Option D: Utility Objective, Not EE

```text
J =
  total_bits
  - lambda_HO * E_HO,total
```

Role: optional training or decision objective.

This is not bits/Joule EE. It is useful if the paper needs a handover-cost-aware
objective without putting assumption-heavy event energy into the EE denominator.

## Recommended Final Formula

Use Option A as the final main EE metric:

```text
eta_EE,active-TX =
  (sum_t sum_u R_u^t Delta_t)
  /
  (sum_t sum_s sum_b z_{s,b}^t p_{s,b}^t Delta_t)
```

Definitions:

- `R_u^t` is the delivered user throughput generated by the HOBS-style SINR.
- If beam-training latency is modeled, use the HOBS latency-aware throughput
  factor `(1 - C_s(t) / T_f)`.
- If beam-training latency is not modeled, state that the numerator is a
  HOBS-style SINR plus Shannon-rate adaptation, not a full HOBS throughput copy.
- `p_{s,b}^t` is the linear-Watt beam transmit power used in SINR.
- `z_{s,b}^t` is the active-beam indicator.
- Inactive beams contribute zero because either `z_{s,b}^t = 0` explicitly or
  `p_{s,b}^t = 0` under inactive-beam closure.
- DAPS or dual-active intervals should count every active RF transmission in
  the denominator, while delivered duplicate bits must not be double-counted in
  the numerator.

Composite EE should be a named secondary analysis:

```text
eta_EE,comp(rho, P_c, P_beam,on) =
  total_bits
  /
  total_declared_composite_energy
```

Do not call the composite result paper-backed unless all denominator parameters
have LEO-specific citations.

Composite-denominator reporting should be framed as metric robustness testing,
not as evidence of physical spacecraft energy savings.

## Recommended Ablation / Sensitivity Metrics

Report these alongside the main metric:

- `eta_EE,active-TX`: HOBS-style active transmit-power EE.
- Total throughput / delivered bits.
- Total active transmit energy.
- Active beam count and active-beam time.
- Power-control activity / denominator-variability diagnostics: report temporal
  variation of `p_{s,b}^t`, distinct power levels or continuous power range,
  and the share of evaluation steps where active transmit power changes. If
  `p_{s,b}^t` is effectively fixed, `eta_EE,active-TX` can degenerate into a
  throughput-proxy metric.
- Handover count.
- Per-beam EE versus system EE.
- Beam-training overhead on/off, if `C_s(t)` is implemented.
- Power-cap sweep, especially because HOBS and S10 use different power contexts.
- PA efficiency sweep, for example `rho_s in {0.3, 0.5, 0.7, 0.9}`.
- Circuit/static power sweep, explicitly marked scenario assumption.
- Active-beam overhead sweep, explicitly marked scenario assumption.
- Handover energy sweep, with `3 J` labeled as EAQL paper-specific if used.
- Composite-ranking stability: whether algorithm ordering changes under
  composite denominator assumptions.

## Parameters Needing Citation Or Disclosure

The following must be disclosed in any table or manifest if used:

- `rho_s` / PA efficiency.
- `P_c,s` / satellite circuit power.
- `P_static`.
- `P_beam,on`.
- Idle and off beam power.
- `P_RD` and any processing exponent.
- `E_HO`.
- Finite-codebook power values.
- Whether HOBS `50 dBm` or S10 `10 dBW / 13 dBW` caps are being used.
- Whether the HOBS beam-training latency factor is included or omitted.

## Unsafe Claims To Avoid

Do not claim:

- HOBS defines total consumed spacecraft EE.
- HOBS includes PA, circuit, static, processing, active-beam, or HO-event energy.
- Composite EE is fully paper-backed for LEO under the current corpus.
- BEAMCOORD-EE numeric constants are LEO satellite constants.
- EEBH-UPLINK terminal-side `P_sys` directly applies to downlink satellite beams.
- EAQL's `3 J` is a universal per-handover energy value.
- `totalPowerW` is the denominator of `systemEeBitsPerJoule`.
- A utility reward is bits/Joule EE.
- Finite-codebook simulated gains prove physical spacecraft energy savings.

## Final Verdict

PASS for this final layered formula decision:

1. Main metric: HOBS-style active transmit-power EE.
2. Secondary analysis: composite simulated communication EE as ablation /
   sensitivity.
3. Optional analysis: HO-aware EE or utility objective only with explicit
   assumption disclosure.

NEEDS MORE EVIDENCE if the intended final paper claim is instead:

1. composite EE as the headline physical EE metric,
2. total spacecraft energy savings,
3. LEO-specific PA/circuit/processing/active-beam constants, or
4. fixed universal handover energy.
