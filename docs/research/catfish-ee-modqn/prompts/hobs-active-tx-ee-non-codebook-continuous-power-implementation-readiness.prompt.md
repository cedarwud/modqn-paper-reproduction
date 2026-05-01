# Prompt: HOBS Active-TX EE Non-Codebook Continuous-Power Implementation Readiness

你是 `modqn-paper-reproduction` 的 execution worker。請只執行
**implementation-readiness / boundary-audit slice**。

這不是 pilot，不是 training，不是 Catfish-EE，也不是 EE-MODQN
effectiveness validation。不要重新規劃整條研究線。

## Read First

請先讀：

```text
AGENTS.md
docs/research/catfish-ee-modqn/00-validation-master-plan.md
docs/research/catfish-ee-modqn/execution-handoff.md
docs/research/catfish-ee-modqn/development-guardrails.md
docs/research/catfish-ee-modqn/ee-modqn-anti-collapse-controller-plan-2026-05-01.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-modqn-feasibility.execution-report.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-anti-collapse-design-gate.execution-report.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-qos-sticky-anti-collapse-design-gate.execution-report.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-qos-sticky-robustness-gate.execution-report.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-qos-sticky-broader-effectiveness-gate.execution-report.md
docs/research/catfish-ee-modqn/03c-b-power-mdp-audit.execution-report.md
docs/research/catfish-ee-modqn/03c-c-power-mdp-pilot.execution-report.md
docs/research/catfish-ee-modqn/03d-ee-route-disposition.execution-report.md
docs/research/catfish-ee-modqn/ra-ee-07-constrained-power-allocator-distillation.execution-report.md
docs/research/catfish-ee-modqn/ra-ee-08-offline-association-reevaluation.execution-report.md
docs/research/catfish-ee-modqn/ra-ee-09-fixed-association-rb-bandwidth.execution-report.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-non-codebook-continuous-power-design-gate.execution-report.md
docs/research/catfish-ee-modqn/energy-efficient/README.md
docs/research/catfish-ee-modqn/energy-efficient/ee-formula-final-review-with-codex-2026-05-01.md
docs/ee-report.md
```

## Current Boundary

沿用以下結論：

```text
current QoS-sticky EE objective route: BLOCK / stop-loss
CP-base continuous-power design gate: PASS, design-only
pilot / training: NOT AUTHORIZED
Catfish-EE: BLOCKED until base EE method beats matched controls
```

## Goal

實作並驗證一個最小 opt-in boundary-readiness surface：

```text
method label: CP-base-EE-MODQN implementation-readiness
power surface: non-codebook analytic continuous-power sidecar
claim boundary: config / wiring / metadata readiness only
```

此 slice 的目標是證明未來 bounded pilot 可以被公平設定，不是證明
EE-MODQN 有效。

## Required Namespace

使用新的 config / artifact / report namespace：

```text
configs/hobs-active-tx-ee-non-codebook-continuous-power-*.resolved.yaml
artifacts/hobs-active-tx-ee-non-codebook-continuous-power-*/
docs/research/catfish-ee-modqn/hobs-active-tx-ee-non-codebook-continuous-power-implementation-readiness.execution-report.md
```

不要寫入 frozen baseline、Phase `03C`、QoS-sticky prior gates、RA-EE、或 Catfish
artifact directories。

## Allowed Work

允許：

1. 新增 config-gated continuous power mode / config schema。
2. 新增或調整最小 runtime wiring，確保 baseline default 不變。
3. 新增 boundary-audit script 或 analysis helper，前提是它不跑 training。
4. 新增 focused tests。
5. 新增 implementation-readiness execution report。
6. 產生 boundary-audit artifact，只能作 wiring / metadata / deterministic
   boundary proof，不得作 effectiveness evidence。

不允許：

1. 不要跑 `scripts/train_modqn.py`。
2. 不要跑 matched pilot。
3. 不要增加 episode training。
4. 不要引入 Catfish / Multi-Catfish。
5. 不要重開 Phase `03C` finite-codebook selector route。
6. 不要重開 RA-EE learned association route。
7. 不要改 frozen baseline configs / artifacts / semantics。
8. 不要 claim EE-MODQN effectiveness。

## Continuous `p_b(t)` Contract

實作的 active-beam power 必須是 analytic continuous sidecar。Design-gate
reference formula 是：

```text
softplus(x) = log(1 + exp(x))
sigma(x) = 1 / (1 + exp(-x))

channel_pressure_b(t)
  = mean_{u in U_b(t)} softplus(q_ref - q_u,b(t))

load_pressure_b(t)
  = log(1 + n_b(t))

overflow_pressure_b(t)
  = softplus(n_b(t) / n_qos - 1)

x_b(t)
  = alpha * load_pressure_b(t)
  + beta * channel_pressure_b(t)
  + kappa * overflow_pressure_b(t)
  + bias

p_b(t)
  = 0, if z_b(t) = 0

p_b(t)
  = p_active_lo_w
    + (p_active_hi_w - p_active_lo_w) * sigma(x_b(t)),
    if z_b(t) = 1
```

`p_active_lo_w`, `p_active_hi_w`, `alpha`, `beta`, `kappa`, `bias`, `q_ref`,
and `n_qos` must be explicit config / metadata fields. If a safer equivalent
analytic formula is needed because of existing code contracts, explain the
deviation and keep these properties:

```text
non-constant
continuous for active beams
non-codebook
not selected profile
rollout-time, not post-hoc
policy-sensitive through association/action consequences
bounded by declared per-beam / total-power guardrails
```

## Rollout Wiring Requirements

Power must be computed in rollout after policy action and shared structural
guard, before SINR / throughput / reward / EE metrics:

```text
state_t
  -> policy action a_t
  -> same opt-in structural guard, if enabled
  -> U_b(t), z_b(t), n_b(t), assigned channel-pressure inputs
  -> continuous p_b(t)
  -> SINR / throughput
  -> r1 / r2 / r3 and EE metrics
  -> state_{t+1}
```

The same `beam_transmit_power_w` vector must feed:

```text
SINR numerator / interference semantics
throughput computation
total_active_beam_power_w
EE_system = sum_u R_u(t) / sum_active_beams p_b(t)
r1_hobs_active_tx_ee for the future candidate
diagnostics and guardrail checks
```

If implementation computes throughput with one power vector and later rescales
EE using another vector, stop and report `BLOCK`.

## Matched Boundary To Prove

Create config metadata and tests showing a future primary comparison can be:

```text
candidate:
  r1 = hobs-active-tx-ee
  same continuous power surface
  same opt-in anti-collapse guard

control:
  r1 = throughput
  same continuous power surface
  same opt-in anti-collapse guard
```

The only intended candidate/control difference must be:

```text
candidate r1_reward_mode = hobs-active-tx-ee
control   r1_reward_mode = throughput
```

Both roles must share:

```text
same continuous p_b(t) formula and constants
same anti-collapse guard
same seeds / seed triplets
same episode budget
same eval seeds
same eval schedule
same checkpoint protocol
same objective weights except r1 mode
same trainer hyperparameters
same artifact / diagnostic schema
```

This slice may define config templates or resolved configs for boundary audit,
but it must not run pilot training.

## Required Tests / Checks

Focused tests must cover at least:

1. Baseline unchanged when the continuous-power mode is disabled.
2. Config namespace gating.
3. Continuous power formula emits non-codebook non-profile power values.
4. Inactive beams emit `0 W`.
5. Active beams stay within configured bounds.
6. No finite codebook, selected profile, profile label, or rounding path is used.
7. Power is computed before SINR / throughput / reward metrics.
8. The same power vector feeds throughput and EE denominator.
9. Policy/action consequences can change `p_b(t)` through association/load/channel
   assignment inputs.
10. Candidate/control boundary metadata proves only `r1_reward_mode` differs.
11. QoS-sticky anti-collapse, if retained, is shared by both roles and not counted
    as EE objective evidence.
12. No Catfish, Multi-Catfish, Phase `03C`, RA-EE learned association, oracle,
    future information, offline replay oracle, or HOBS optimizer mode is enabled.

## Boundary-Audit Artifact

If you generate an artifact, it must be under:

```text
artifacts/hobs-active-tx-ee-non-codebook-continuous-power-boundary-audit/
```

It may report:

```text
config hashes / metadata
candidate-control intended-difference table
power-surface constants
proof that selected_power_profile is absent
proof that finite codebook levels are absent
proof that candidate/control share the same power surface
deterministic step-level wiring samples, if no training is run
```

It must not report pilot effectiveness or treat scalar reward as success.

## Acceptance Criteria

This implementation-readiness slice can pass only if all are true:

1. Frozen baseline behavior remains unchanged.
2. New behavior is opt-in and namespace-gated.
3. `p_b(t)` is continuous / analytic for active beams.
4. `p_b(t)` is not constant, not finite-codebook, not selected fixed profile.
5. `p_b(t)` is not post-hoc rescore.
6. `p_b(t)` is computed before SINR / throughput / reward / EE metrics.
7. The same power vector feeds throughput and EE denominator.
8. Candidate/control boundary can be expressed with only `r1_reward_mode` changed.
9. Throughput + same anti-collapse + same continuous power control is present as
   the decisive future control.
10. No oracle, future information, offline replay oracle, HOBS optimizer, Catfish,
    Phase `03C`, or RA-EE learned association dependency exists.

## Stop Conditions

Stop and report `BLOCK` or `NEEDS MORE DESIGN` if:

1. Implementation requires mutating frozen baseline semantics.
2. Power can only be added as eval post-processing or rescore.
3. Power is codebook/profile-selected or rounded to finite levels.
4. `p_b(t)` varies but policy cannot influence it through association/action
   consequences.
5. Candidate/control must differ by anything besides `r1_reward_mode`.
6. The throughput + same-guard + same-power control cannot be represented.
7. Any design needs Catfish, RA-EE learned association, oracle, future
   information, offline replay oracle, or HOBS optimizer behavior.
8. The slice starts running pilot training or framing results as effectiveness.

## Forbidden Claims

Do not claim:

```text
EE-MODQN effectiveness
Catfish-EE readiness
Catfish / Multi-Catfish effectiveness
physical energy saving
HOBS optimizer reproduction
full RA-EE-MODQN
learned association effectiveness
RB / bandwidth allocation effectiveness
Phase 03D failure is overturned
Phase 03C selector route is reopened
scalar reward success
QoS-sticky anti-collapse as EE objective contribution
denominator variability alone proves energy-aware learning
same-throughput-less-physical-power
```

## Output Format

請輸出：

```text
Changed Files
What Was Implemented
Config / Namespace Boundary
Rollout Power Wiring
Candidate / Control Boundary Proof
Boundary-Audit Artifact
Tests / Checks Run
Acceptance Result
Forbidden Claims Still Active
Deviations / Blockers
PASS / BLOCK / NEEDS MORE DESIGN
```

若本 slice `PASS`，請不要自行啟動 pilot。回報總控等待下一個 bounded pilot
authorization。
