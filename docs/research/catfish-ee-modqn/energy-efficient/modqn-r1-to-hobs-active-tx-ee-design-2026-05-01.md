# MODQN r1 Replacement With HOBS-Style Active-TX EE

Date: 2026-05-01

Purpose: define the concrete research and implementation plan for replacing
MODQN's original throughput reward `r1` with the final HOBS-style active
transmit-power energy-efficiency formula.

This document began as a design note. As of the 2026-05-01 feasibility gates,
it also records the current implementation-status boundary. The result
authority for the MODQN implementation side is:

```text
modqn-paper-reproduction/docs/research/catfish-ee-modqn/hobs-active-tx-ee-modqn-feasibility.execution-report.md
```

## 2026-05-01 Implementation Status Boundary

The formula and denominator-design parts have passed scoped feasibility, but
the learned MODQN policy route is still blocked:

```text
HOBS active-TX EE formula / reward wiring: PASS, scoped
SINR structural audit: PASS, but negligible at current MODQN operating point
channel-regime / antenna-gain path: BLOCK as paper-backed MODQN continuation
HOBS-inspired DPC sidecar denominator gate: PASS
tiny learned-policy Route D denominator check: BLOCK
EE-MODQN effectiveness: NOT PROMOTED / BLOCKED
```

Important interpretation:

1. The denominator no longer has to be constant. With the opt-in
   HOBS-inspired DPC sidecar, total active transmit power can vary in
   evaluation.
2. The formula is not merely `throughput / constant` under the DPC diagnostic;
   same-policy throughput-vs-EE ranking can separate.
3. The current learned-policy blocker is one-active-beam collapse. Route `D`
   still evaluated all `50` greedy steps with one active beam.
4. Therefore this design is not yet an EE-MODQN effectiveness result.

Any continuation must first add an anti-collapse / capacity / assignment design
gate. More Route `D` episodes, Catfish repair, reward retuning, or Phase `03C`
continuation are not the default next step.

## Source Basis

### MODQN Baseline

MODQN's original reward vector is:

```text
R_i^t = (r1,t, r2,t, r3,t)
```

where:

```text
r1,t = throughput reward
r2,t = handover cost reward
r3,t = load-balancing reward
```

The local catalog records MODQN as throughput / handover-cost / load-balancing
MORL. It also records that MODQN uses SNR, has no interference model, and uses a
fixed transmit power:

```text
p_{i,l,v} = 2 W
powerControl = no dynamic power control
```

Therefore, simply dividing MODQN throughput by the original fixed `2 W` power
would mostly rescale throughput and would be weak as an EE research claim.

### HOBS Anchor

HOBS supplies the evidence-backed signal and EE direction:

```text
gamma = P * H * G_T * G_R * association / (I_intra + I_inter + noise)
```

HOBS system EE is:

```text
E_eff(t) = R_tot(t) / sum_n sum_m P_{n,m}(t)
```

HOBS also treats beam transmit power as a dynamic control variable:

```text
P_{n,m}(t) = P_{n,m}(t - T_f) + xi^P_{n,m}(t)
```

with the update direction changed when per-beam EE decreases.

## Research Claim

The defensible thesis claim is:

```text
Replace MODQN's throughput-oriented r1 with a HOBS-style active-transmit-power
EE objective, while retaining MODQN's multi-objective structure for handover cost
and load balancing.
```

The claim should not be:

```text
full spacecraft energy saving
```

or:

```text
total consumed satellite EE
```

The safe name is:

```text
HOBS-style active transmit-power EE reward for MODQN-based multi-beam LEO handover
```

## Final r1 Formula

Use this as the replacement for MODQN's original `r1 = throughput`:

```text
r1,t^EE =
  R_active,t
  /
  (P_active,t + epsilon_P)
```

where:

```text
R_active,t = sum_u R_u^t
P_active,t = sum_s sum_b z_{s,b}^t p_{s,b}^t
epsilon_P > 0
```

For time-window reporting, use:

```text
eta_EE,active-TX =
  (sum_t sum_u R_u^t Delta_t)
  /
  (sum_t sum_s sum_b z_{s,b}^t p_{s,b}^t Delta_t)
```

Use the per-epoch ratio for reward and the time-aggregated ratio for final
evaluation.

## Required Model Changes

### Replace MODQN SNR With HOBS-Style SINR

MODQN's original SNR-only model is not enough for a HOBS-style EE claim because
it ignores inter-beam and inter-satellite interference.

The upgraded signal model should use:

```text
gamma_u^t =
  x_{u,s,b}^t p_{s,b}^t h_{u<-s,b}^t
  /
  (
    I_intra,u^t
    + I_inter,u^t
    + N_0 W_u^t
  )
```

with:

```text
I_intra,u^t =
  sum_{b' != b} z_{s,b'}^t p_{s,b'}^t h_{u<-s,b'}^t

I_inter,u^t =
  sum_{s' != s} sum_{b''} z_{s',b''}^t p_{s',b''}^t h_{u<-s',b''}^t
```

This is the main formula bridge from HOBS into MODQN.

### Add Dynamic Transmit-Power Control

The EE reward is meaningful only if the denominator can vary. The implementation
must not keep every active link at a fixed `2 W`.

At minimum, each active beam needs:

```text
p_{s,b}^t
```

as an actual state-dependent transmit-power value.

Valid implementation choices:

1. HOBS-style DPC sidecar.
2. SINR-target required-power rule.
3. Finite power codebook.
4. Learned power-allocation head.

The first implementation should prefer option 1 or 3 because they are easier to
audit than a new learned continuous power policy.

## Recommended First Implementation: MODQN-EE With HOBS-Style DPC Sidecar

This is the most defensible first version because it keeps MODQN's action space
focused on handover / beam selection and adds a HOBS-inspired power update layer.

### State

Keep MODQN's original state concepts, but include or derive:

```text
candidate beams / satellites
current association x_{u,s,b}^t
beam active state z_{s,b}^t
geometry-derived path loss
off-axis antenna gain
SINR gamma_u^t
throughput R_u^t
current beam power p_{s,b}^t
previous per-beam EE
beam load / user count
handover status
```

### Action

Keep MODQN's action as the user or agent selecting a target beam / satellite:

```text
a_u^t = selected beam or selected satellite-beam pair
```

The action updates association:

```text
x_{u,s,b}^t
```

Then derive active beams:

```text
z_{s,b}^t = 1 if any user is assigned to beam (s,b), else 0
```

### Power Update

After association and active-beam update, run a HOBS-style DPC sidecar:

```text
if z_{s,b}^t = 0:
  p_{s,b}^t = 0
else:
  candidate_p = p_{s,b}^{t-1} + xi_{s,b}^{t-1}

  if EE_beam,s,b^{t-1} <= EE_beam,s,b^{t-2}:
    xi_{s,b}^t = -xi_{s,b}^{t-1}
  else:
    xi_{s,b}^t = xi_{s,b}^{t-1}

  if any served user on beam has gamma_u^t < gamma_thr:
    xi_{s,b}^t = abs(xi_{s,b}^t)

  p_{s,b}^t = clip(candidate_p, 0, P_beam,max)
```

Then enforce satellite aggregate power:

```text
if sum_b p_{s,b}^t > P_sat,max:
  p_{s,b}^t = p_{s,b}^t * P_sat,max / sum_b p_{s,b}^t
```

### Throughput

Compute throughput after SINR:

```text
R_u^t = W_u^t log2(1 + gamma_u^t)
```

If beam-training latency is explicitly implemented:

```text
R_u^t =
  (1 - C_s(t) / T_f)
  * W_u^t
  * log2(1 + gamma_u^t)
```

If beam-training latency is not implemented, disclose that the numerator is a
HOBS-style SINR plus Shannon-rate adaptation, not a full HOBS throughput copy.

### r1 Replacement

Original MODQN:

```text
r1,t = throughput_i,t
```

Proposed MODQN-EE:

```text
r1,t^EE =
  normalize(
    R_scope,t / (P_scope,t + epsilon_P)
  )
```

Recommended reward scope:

```text
R_scope,t = sum of throughput for users served by the agent's selected beam
P_scope,t = transmit power of the selected active beam
```

or, for a centralized/global reward variant:

```text
R_scope,t = sum_u R_u^t
P_scope,t = sum_s sum_b z_{s,b}^t p_{s,b}^t
```

For the first paper version, use the global system scope for evaluation and make
the reward scope explicit. If training becomes unstable with a global reward,
use selected-beam local EE for `r1` and report global EE as the KPI.

## Reward Vector After Replacement

The modified vector reward becomes:

```text
R_i^t = (r1,t^EE, r2,t, r3,t)
```

where:

```text
r1,t^EE = normalized active-TX EE
r2,t = original MODQN handover cost reward
r3,t = original MODQN load-balancing reward
```

Do not remove `r2` and `r3`. The research contribution is not replacing MODQN
with a scalar EE-only DQN; it is changing the first objective while preserving
the MODQN multi-objective structure.

## Normalization Requirements

Raw EE can have large and unstable scale, especially when active power is small.
Normalize before feeding it to MODQN.

Recommended first version:

```text
r1,t^EE_norm =
  clip(
    eta_t / eta_ref,
    0,
    r1_max
  )
```

where:

```text
eta_t = R_scope,t / (P_scope,t + epsilon_P)
eta_ref = moving average or baseline EE from throughput-MODQN
r1_max = 1 or 10, chosen to match the scale of r2 and r3
```

Alternative stable transform:

```text
r1,t^EE_norm = log(1 + eta_t / eta_ref)
```

Guardrail:

```text
if R_scope,t < R_min or QoS is violated:
  r1,t^EE_norm = r1,t^EE_norm - qos_penalty
```

This prevents the policy from gaming EE by serving almost no traffic with very
low power.

## Key Algorithm

Per epoch:

```text
for each epoch t:
  update satellite / user geometry
  derive path loss, antenna gain, channel gain

  for each agent/user:
    observe state_t
    choose target satellite-beam action with MODQN

  update association x
  update active beams z from x

  run HOBS-style DPC sidecar to update p
  enforce per-beam and per-satellite power caps

  compute HOBS-style SINR with intra/inter interference
  compute throughput R
  compute handover events
  compute beam load distribution

  compute r1_EE from active-TX EE
  compute r2 handover cost
  compute r3 load-balancing reward

  update MODQN objective-specific Q networks
  select future actions by scalarized Q value
```

The critical ordering is:

```text
association -> active beams -> power update -> SINR -> throughput -> EE reward
```

Do not compute `r1_EE` before updating `p_{s,b}^t`; otherwise the denominator is
stale.

## Key Implementation Data Structures

Minimum runtime fields:

```text
beamPowerW[s][b]
beamPowerStepW[s][b]
beamActive[s][b]
association[u] = {satellite, beam}
previousAssociation[u]
sinr[u]
throughputBps[u]
beamThroughputBps[s][b]
beamEeBitsPerJoule[s][b]
previousBeamEeBitsPerJoule[s][b]
systemEeActiveTxBitsPerJoule
activeTxPowerW
activeTxEnergyJ
handoverCount
```

Minimum config fields:

```text
P_beam_max_W
P_sat_max_W
powerStepW
gamma_thr_linear_or_dB
epsilon_P
eta_ref_mode
r1_norm_max
qos_min_rate_bps
```

Do not use these in the main denominator unless running sensitivity:

```text
P_c,s
rho_s
P_beam,on
P_idle
P_off
E_HO
P_RD
```

## Baselines Needed

To make the paper claim defensible, compare at least:

1. Original MODQN:
   `r1 = throughput`, fixed power.
2. MODQN with HOBS-style SINR but throughput `r1`:
   isolates the signal-model change.
3. MODQN-EE:
   `r1 = active-TX EE`, dynamic power sidecar.
4. HOBS-style DPC / heuristic baseline if available:
   non-RL EE-oriented reference.
5. DQN-throughput baseline:
   original paper's single-objective throughput comparator.

This separation is important because replacing SNR with SINR and adding dynamic
power control are both major changes. The ablation must show which change drives
the result.

## Metrics To Report

Primary:

```text
eta_EE,active-TX
total delivered bits / throughput
total active transmit energy
handover count
load-balancing metric
```

Diagnostics:

```text
active beam count
power variance over time
number of distinct power levels used
share of epochs where p changes
per-beam EE distribution
QoS violation rate
blocked / unserved user count
```

Sensitivity:

```text
power caps
power step size
epsilon_P
eta_ref normalization
gamma_thr
composite EE with declared rho / P_c / P_beam,on assumptions
```

## Unsafe Shortcuts

Do not:

1. Replace `r1` with `throughput / 2 W` while leaving all powers fixed.
2. Claim total spacecraft energy saving.
3. Use composite denominator terms in the main reward without source disclosure.
4. Double-count duplicate DAPS bits in the numerator.
5. Let a near-zero denominator produce artificially large reward.
6. Compare against original MODQN without disclosing that the channel changed
   from SNR-only to HOBS-style SINR.

## Recommended Research Title

Strong title:

```text
Energy-Efficiency-Aware MODQN for Multi-Beam LEO Handover Using
HOBS-Style Active Transmit-Power Reward
```

Shorter title:

```text
HOBS-Style Active-TX EE Reward for MODQN-Based Multi-Beam LEO Handover
```

## Final Recommendation

This remains a viable design direction only if the next method gate directly
solves one-active-beam collapse. The intended future method shape is still:

```text
MODQN multi-objective handover
+ HOBS-style SINR
+ dynamic beam transmit-power control
+ r1 replaced by normalized active-TX EE
+ original r2/r3 retained
+ active-TX EE reported as the main metric
+ composite EE reported only as sensitivity
```

The minimum condition for the topic to be credible is that `p_{s,b}^t` must be a
state-dependent dynamic transmit-power variable. If `p_{s,b}^t` remains fixed,
the EE reward collapses into a throughput proxy and the research contribution is
not strong enough.

The 2026-05-01 feasibility gates satisfy the denominator-variability condition
through the HOBS-inspired DPC sidecar, but they do not satisfy the learned-policy
effectiveness condition. The next credible gate is anti-collapse / capacity /
assignment design, not a larger training run.
