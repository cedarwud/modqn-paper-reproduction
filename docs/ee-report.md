# `modqn-paper-reproduction` EE 路線下一步評估報告

## 2026-05-02 HEA-MODQN scoped thesis claim boundary

本節同步 `Handover-Energy-Aware MODQN` (`HEA-MODQN`) 的 thesis/report
claim boundary。這個 method 也可寫成
`service-continuity-sensitive MODQN extension`。不要把 method name 寫成
HOBS；multi-beam SINR 與 active transmit-power accounting 只能作為
system-model source 的背景引用，不是此 HEA-MODQN claim 的方法名稱或成功證據。

目前 controller state：

```text
Active-TX EE-MODQN: BLOCK / NOT PROMOTED
CF-RA-CP active-TX EE: BLOCK
HEA-MODQN high handover-cost / service-continuity-sensitive utility gate: PASS, scoped
HEA-MODQN ratio-form EE_HO clean artifact gate: PASS, scoped
HEA-MODQN robustness / attribution gate: PASS
Catfish / Multi-Catfish for EE repair: BLOCKED / NOT PROMOTED
Catfish-over-HEA baseline parity gate: PASS readiness / parity only
Catfish-over-HEA bounded matched pilot: BLOCK
Multi-Catfish redesign plan: current design authority
Multi-Catfish Gate 1 read-only diagnostics: NEEDS MORE DESIGN
Multi-Catfish Gate 1A transition provenance SDD: current next design gate
Catfish-EE / Phase 06: BLOCKED
Scalar reward: diagnostic only
```

正面 claim 只限於 frozen high handover-cost /
service-continuity-sensitive sensitivity setting。可直接放進論文的文字是：

> Under a frozen high handover-cost / service-continuity-sensitive sensitivity
> setting, HEA-MODQN improves both the declared handover-aware utility and the
> ratio-form handover-aware EE_HO relative to matched throughput-objective
> MODQN while preserving p05 throughput, served ratio, and outage guardrails.

這個 `J` 與 `EE_HO` 都是 thesis sensitivity accounting，不是 physical
energy saving result，也不是 active-TX EE recovery。High-cost `E_HO` 與
`lambda_HO` 值是 sensitivity / utility-accounting penalties，不是 physical
constants；`lambda_HO` 只影響 utility-form objective / policy selection，不進
ratio-form `EE_HO` denominator。

Formula / literature evidence for presentation:

```text
eta_EE,HO =
  (sum_t sum_u R_u^t Delta_t)
  /
  (sum_t sum_s P_tot,s^t Delta_t + E_HO,total)

E_HO,total = sum_t sum_u delta_HO,u^t * E_u,HO
```

This ratio-form `handover-aware energy efficiency` is a defensible thesis-level
extension for sensitivity analysis, but it is not a single-paper copied
standard metric. The evidence chain to cite in slides is:

| Evidence source | What it supports | Claim limit |
|---|---|---|
| `system-model-refs/system-model-formulas.md` §3.14-3.16 | Defines `eta_EE,HO` and separates active-TX EE, total communication-power proxy, handover-aware EE, and utility fallback | `eta_EE,HO` requires explicit `E_u,HO` assumption set and sensitivity path |
| `system-model-refs/simulator-parameter-spec.md` §6 | Treats `hoEnergyJoules` as a scenario parameter and `lambdaHo=0.2` as an EAQL-derived utility anchor | No universal LEO physical default for per-handover energy |
| `paper-catalog/catalog/PAP-2025-EAQL.json` | Uses an energy-aware handover utility / reward with `lambda * E_handover` and reports handover-energy sensitivity | Supports handover energy cost in decision objective, not a copied ratio denominator |
| `paper-catalog/catalog/PAP-2025-BIPARTITE-HM.json` | Includes per-handover energy, RTT, and setup cost in strategy cost and analyzes energy-weight sensitivity | Supports operational / energy handover cost modeling, not direct MODQN evidence |

Therefore, it is acceptable to present `EE_HO` as a scenario-declared,
handover-aware EE metric in a high handover-cost / service-continuity-sensitive
section. It must not be presented as the original active-TX EE metric, a
physical satellite energy-saving measurement, or a universal EE definition.
The clean ratio-form artifact gate has also passed for the same scoped high-cost
setting, but the result remains scenario-declared and sensitivity-bound.

Scenario realism boundary for thesis writing:

1. The condition is academically reasonable because prior LEO handover work
   models handover energy, operational handover cost, setup cost, RTT, or energy
   weighting inside handover decisions.
2. The condition is not a universal physical baseline. `E_u,HO` has no
   corpus-wide LEO default in `system-model-refs/simulator-parameter-spec.md`;
   every numeric use must be declared as a scenario / sensitivity assumption.
3. The high-cost rows (`E_HO=130/150/200`) should be described as a
   service-continuity-sensitive regime where handover events stand in for
   signaling, setup, interruption, recovery, QoS continuity, or terminal-side
   overhead pressure. Do not write that real LEO handovers are generally
   `130 J`, `150 J`, or `200 J`.
4. The low-cost reference (`E_HO=3`, `lambda_HO=0.2`) remains visible and neutral;
   it supports no low-cost success claim.
5. The thesis-safe interpretation is "reasonable sensitivity setting", not
   "measured spacecraft energy saving" or "general NTN deployment default".

Final relaxation / stop-loss review:

1. The clean artifact can directly evaluate the relaxed communication-only
   formula:

   ```text
   EE_general = total_bits / communication_energy_joules
   ```

2. Removing `E_HO,total` from the denominator blocks the relaxed claim. The
   candidate has `0/30` wins over matched throughput-control across the full
   sensitivity grid, `0/15` wins in the primary high-cost subset, and `0/6`
   wins in the secondary subset. Primary mean communication-only
   `EE_delta` is about `-82980 bits/J`; secondary mean communication-only
   `EE_delta` is about `-74659 bits/J`. The break-even-near
   `E_HO=130`, `lambda_HO=0.2` row has communication-only
   `EE_delta` about `-9735 bits/J`, and the low-cost
   `E_HO=3`, `lambda_HO=0.2` row remains `0`.
3. Therefore the HEA-MODQN win is specifically a handover-aware /
   service-continuity-sensitive result. It comes from reducing the
   handover-event denominator burden under the declared high-cost setting, not
   from improving general communication-energy-only EE.
4. This fires the stop-loss condition for further relaxation: do not remove
   the handover denominator and still claim EE superiority; do not reopen
   active-TX, CP-base, RA-EE association, or Catfish as a last-minute repair
   path.

Metrics to record：

| Metric | Value |
|---|---:|
| primary subset positive cells | `15/15` |
| secondary subset positive cells | `6/6` |
| primary mean `J_delta` | `13288.26934925599` |
| candidate vs ablation mean primary `J_delta` | `13288.26934925599` |
| break-even-near `E_HO=130`, `lambda_HO=0.2` `J_delta` | `217.11219999700552` |
| low-cost `E_HO=3`, `lambda_HO=0.2` `J_delta` | `0` |
| primary candidate mean `EE_HO` | `414946.480518 bits/J` |
| primary control mean `EE_HO` | `412914.217690 bits/J` |
| primary `EE_HO_delta` | `+2032.262828 bits/J` |
| primary `EE_HO_ratio` | `1.0049217555` |
| secondary candidate mean `EE_HO` | `1278023.040718 bits/J` |
| secondary control mean `EE_HO` | `1272319.566835 bits/J` |
| secondary `EE_HO_delta` | `+5703.473883 bits/J` |
| secondary `EE_HO_ratio` | `1.0044827369` |
| low-cost `E_HO=3`, `lambda_HO=0.2` `EE_HO_delta` | `0` |
| communication-only EE candidate wins | `0/30` |
| primary communication-only EE wins | `0/15` |
| secondary communication-only EE wins | `0/6` |
| primary mean communication-only `EE_delta` | `about -82980 bits/J` |
| secondary mean communication-only `EE_delta` | `about -74659 bits/J` |
| lambda-zero ablation primary wins | `0/15` |
| primary min p05 throughput ratio | `0.9665274269295018` |
| served delta | `0` |
| outage delta | `0` |
| primary mean handover delta | `-43.733333333333334` |

The low-cost reference stays visible: `E_HO=3`, `lambda_HO=0.2` has
`J_delta=0`, so no low-cost success is claimed.

Claim boundaries:

1. active-TX EE results remain separate and diagnostic for this thesis claim,
2. scalar reward is diagnostic only,
3. high-cost values are sensitivity / utility-accounting penalties,
4. no low-cost success claim is authorized,
5. no physical energy-saving result is authorized,
6. no Catfish, Multi-Catfish effectiveness, Catfish-EE, or Phase `06` readiness follows from
   this result,
7. Catfish-over-HEA readiness is separate and default-off; its enabled bounded
   pilot is blocked and is not evidence of Catfish effectiveness or HEA metric
   improvement,
8. the current Catfish work, if reopened, is Multi-Catfish-first redesign
   planning with Single-Catfish as a collapsed ablation.

Forbidden rewrites:

1. do not generalize any EE variant as superior to MODQN,
2. do not describe active-TX EE as recovered or promoted,
3. do not write that an optimizer from the HOBS line was reproduced,
4. do not write that Catfish-EE or Phase `06` is ready,
5. do not describe Multi-Catfish as an EE repair,
6. do not treat scalar reward as proof of success.

Next step for the thesis result is tables / figures / narrative polish and
status sync. Catfish-over-HEA may be discussed as a separate default-off
training-enhancement readiness surface over HEA-MODQN, but the enabled bounded
pilot is blocked. Current Catfish planning must use
`docs/research/catfish-ee-modqn/2026-05-02-multi-catfish-redesign-plan.md`
and
`docs/research/catfish-ee-modqn/2026-05-02-multi-catfish-gate1a-transition-provenance-sdd.md`.
Gate 1 read-only diagnostics found only aggregate / seed / cell provenance, so
transition-level Catfish scoring remains `NEEDS MORE DESIGN`. Multi-Catfish is
designed first, Single-Catfish is a collapsed ablation, and no Catfish pilot
may be framed as EE repair, Active-TX EE recovery, Catfish-EE, Phase `06`,
current Multi-Catfish effectiveness, or HEA thesis-claim dependency.

Authority evidence:

```text
ntn-sim-core/artifacts/handover-energy-aware-modqn-bounded-matched-pilot/summary.json
ntn-sim-core/artifacts/handover-energy-aware-modqn-bounded-matched-pilot/full-sensitivity-table.csv
internal/ntn-sim-core/devlogs/2026-05-02-handover-energy-aware-modqn-bounded-pilot.md
```

## 2026-05-01 CP-base non-codebook continuous-power design gate update

本節同步 `HOBS-active-TX non-codebook continuous-power base EE-MODQN`
design gate。短 label 是 `CP-base-EE-MODQN`。這是 **design-gate PASS only**：
不是 pilot implementation、不是 training、不是 full EE-MODQN effectiveness、
不是 Catfish-EE readiness，也不是 physical energy saving 或 HOBS optimizer
reproduction。

此 gate 保留 current QoS-sticky EE objective route 的停損結論：

```text
current QoS-sticky EE objective route: BLOCK / stop-loss
CP-base continuous-power design gate: PASS, design-only
pilot / training: NOT AUTHORIZED
Catfish-EE: BLOCKED until base EE method beats matched controls
```

CP-base 的核心差異是把 `p_b(t)` 定義成 rollout 內的 analytic continuous
sidecar，而不是 post-hoc rescore、finite codebook、runtime profile selector、
或 selected fixed profile。未來若 implementation-readiness slice 被授權，power
必須在 policy action 與 shared structural guard 之後、SINR / throughput /
reward / EE metrics 之前計算：

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

未來 decisive matched control 也必須使用同一個 continuous power surface：

```text
candidate:
  r1 = hobs-active-tx-ee
  same continuous power surface
  same anti-collapse guard

control:
  r1 = throughput
  same continuous power surface
  same anti-collapse guard
```

也就是說，candidate / control 的唯一 intended difference 只能是 `r1`。這個
要求是為了隔離 `EE objective`，避免再次被 `throughput + same anti-collapse`
或 power sidecar 本身解釋掉。

此 design gate 當時只允許 **implementation-readiness / boundary-audit slice**。
該 slice 已在下一節同步為 readiness-only PASS。它仍未授權 pilot training、未
claim effectiveness，也未引入 Catfish / Multi-Catfish、Phase `03C`、RA-EE
learned association、oracle、future information、offline replay oracle、或
HOBS optimizer 假設。

Authority reports / prompt：

```text
docs/research/catfish-ee-modqn/hobs-active-tx-ee-non-codebook-continuous-power-design-gate.execution-report.md
docs/research/catfish-ee-modqn/prompts/hobs-active-tx-ee-non-codebook-continuous-power-implementation-readiness.prompt.md
```

## 2026-05-01 CP-base implementation-readiness / boundary-audit update

CP-base implementation-readiness / boundary-audit slice 已完成，controller
判定為 **PASS, readiness-only**。這一輪不是 pilot training，也不是
effectiveness validation。

Worker 實作了新的 opt-in power mode：

```text
hobs_power_surface.mode: non-codebook-continuous-power
```

並新增：

```text
configs/hobs-active-tx-ee-non-codebook-continuous-power-throughput-control.resolved.yaml
configs/hobs-active-tx-ee-non-codebook-continuous-power-ee-candidate.resolved.yaml
configs/hobs-active-tx-ee-non-codebook-continuous-power-boundary-audit.resolved.yaml
artifacts/hobs-active-tx-ee-non-codebook-continuous-power-boundary-audit/
docs/research/catfish-ee-modqn/hobs-active-tx-ee-non-codebook-continuous-power-implementation-readiness.execution-report.md
```

重點結果：

```text
acceptance_result = PASS
matched_boundary_pass = true
same_power_vector_for_candidate_and_control = true
same_throughput_for_candidate_and_control = true
ee_denominator_reuses_step_power_vector = true
policy_action_consequences_change_power = true
active_power_nonconstant = true
active_power_non_codebook = true
selected_power_profile_absent = true
inactive_beams_zero_w = true
power_budget_violations = 0
pilot_training_authorized = false
```

Candidate / control 邊界已能表示為：

```text
candidate:
  r1_reward_mode = hobs-active-tx-ee
  same continuous power surface
  same QoS-sticky structural guard

control:
  r1_reward_mode = throughput
  same continuous power surface
  same QoS-sticky structural guard
```

唯一 boundary-critical intended difference 是 `r1_reward_mode`。Power 在
rollout 內於 action / shared guard 後、SNR/SINR / throughput / reward / EE
metrics 前計算；同一個 `beam_transmit_power_w` 同時餵給 throughput numerator
與 `r1_hobs_active_tx_ee` denominator。`q_u,b(t)` 的實作 feature 是
`log1p(unit-power-snr-linear)`。

這個 PASS 只表示未來 bounded matched pilot 的 config / wiring / metadata
邊界已可稽核。它仍然不可 claim：

```text
EE-MODQN effectiveness
Catfish-EE readiness
physical energy saving
HOBS optimizer reproduction
Phase 03C reopening
RA-EE learned association
scalar reward success
denominator variability alone proves energy-aware learning
```

Controller 後續另行草擬 bounded matched pilot prompt，且 execution worker 已在
2026-05-02 執行該 pilot。結果是 `BLOCK`，詳見下一節；不可調參續跑。

Authority report：

```text
docs/research/catfish-ee-modqn/hobs-active-tx-ee-non-codebook-continuous-power-implementation-readiness.execution-report.md
```

## 2026-05-01 CP-base bounded matched pilot prompt update

Controller 已草擬下一個可交給 execution worker 的 bounded matched pilot
prompt：

```text
docs/research/catfish-ee-modqn/prompts/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot.prompt.md
```

這個 prompt 後續已由 execution worker 執行。此段保留 prompt 邊界；pilot
result 見下一節。

Pilot prompt 的 decisive comparison 是：

```text
ee-candidate:
  r1_reward_mode = hobs-active-tx-ee
  same non-codebook continuous power surface
  same QoS-sticky structural guard

throughput-control:
  r1_reward_mode = throughput
  same non-codebook continuous power surface
  same QoS-sticky structural guard
```

唯一 intended difference 仍只能是 `r1_reward_mode`。Protocol 被限制為
`2` roles x `3` seed triplets x `5` episodes，並要求相同 eval seeds、相同
checkpoint protocol、相同 trainer hyperparameters、相同 artifact schema。若需要
任何 smoke run，只能作為診斷，不能取代 acceptance protocol。

若 pilot candidate 輸給 `throughput + same guard + same continuous power`
control，或只贏 scalar reward，或違反 p05 / served / outage / handover / `r2`
/ power guardrails，必須 `BLOCK`，不可調參續跑。即使未來 pilot PASS，也只能
回到 controller 更新 claim boundary；不得直接啟動 Catfish-EE。

## 2026-05-02 CP-base bounded matched pilot execution update

CP-base bounded matched pilot 已完成，controller 判定為 **BLOCK**。這是
bounded negative result，不是 `PASS`，也不是 `NEEDS MORE DESIGN`。

Matched boundary 通過：

```text
matched_boundary_pass = true
only_intended_difference_is_r1_reward_mode = true
same_continuous_power_surface = true
same_qos_sticky_guard = true
same_seed_triplets / eval_seeds / budget / checkpoint_protocol = true
finite_codebook_levels_absent = true
selected_power_profile_absent = true
forbidden_modes_disabled = true
```

但 decisive comparison 失敗：

```text
throughput-control EE_system = 873.9817347495251
ee-candidate EE_system = 872.9239944442857
candidate_vs_control_EE_system_delta = -1.057740305239463
p05_throughput_ratio_vs_control = 0.9675993577312307
handover_delta_vs_control = +77.33333333333331
r2_delta_vs_control = -0.07733333333333334
per_seed_EE_deltas = [-1.1009355380937222, -1.1279326781493637, -0.9443526994751892]
scalar_reward_diagnostic_delta_vs_control = +4312.822050380359
power_violations = 0/0/0
```

Stop conditions triggered:

1. candidate lost `EE_system` to throughput + same guard + same continuous power
   control,
2. candidate exceeded the handover guardrail,
3. candidate failed the `r2` guardrail,
4. candidate only won scalar reward.

因此 CP-base non-codebook continuous-power boundary 仍可當作 auditable negative
evidence，但不可 claim EE-MODQN effectiveness、Catfish-EE readiness、physical
energy saving、HOBS optimizer reproduction、或 scalar reward success。下一步不
是 tune / rerun 這個 CP-base candidate；若繼續 EE research，必須先回到新的
design gate。

Authority reports / artifacts：

```text
docs/research/catfish-ee-modqn/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot.execution-report.md
artifacts/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot-summary/summary.json
```

## 2026-05-01 QoS-sticky HOBS active-TX EE broader-effectiveness gate update

本節同步 `QoS-sticky HOBS-active-TX EE-MODQN` bounded
broader-effectiveness gate。這個 method 只能標成 **new extension / method
variant** 與 **bounded active-TX EE validation only**；它不是 full
EE-MODQN、不是 physical energy saving、不是 HOBS optimizer reproduction、
不是 RA-EE association、也不是 Catfish-EE。

本 gate 使用四個 matched roles：

```text
matched-throughput-control:
  r1=throughput, same DPC sidecar, anti-collapse disabled
  label = DPC-matched throughput control, not frozen MODQN baseline

hobs-ee-control-no-anti-collapse:
  r1=hobs-active-tx-ee, same DPC sidecar, anti-collapse disabled

qos-sticky-ee-candidate:
  r1=hobs-active-tx-ee, same DPC sidecar,
  qos-sticky-overflow-reassignment enabled, nonsticky moves disabled

anti-collapse-throughput-control:
  r1=throughput, same DPC sidecar,
  qos-sticky-overflow-reassignment enabled, nonsticky moves disabled
```

Matched boundary passed across the same environment boundary, same DPC sidecar,
same three robustness seed triplets, same eval seeds, same `5`-episode budget,
same eval/checkpoint protocol, and same objective weights / hyperparameters.
The only intended differences were `r1` objective and anti-collapse enablement.

Result:

```text
anti-collapse mechanism under broader multi-control acceptance: BLOCK
EE objective contribution: BLOCK
overall: BLOCK
```

Candidate removed one-active-beam collapse in aggregate
(`active_beam_count_distribution={"7.0": 150}`), kept
`denominator_varies_in_eval=true`, kept
`active_power_single_point_distribution=false`, kept `nonsticky_move_count=0`,
and had zero budget / per-beam / inactive-beam power violations. However,
broader guardrails failed against `anti-collapse-throughput-control`:

```text
candidate EE_system delta vs anti-collapse-throughput-control: -0.32138238121922313
candidate throughput delta vs anti-collapse-throughput-control: +4.402197666168831 bps
p05 ratio vs anti-collapse-throughput-control: 0.9606352634550617
handover delta vs anti-collapse-throughput-control: +83.33333333333331
r2 delta vs anti-collapse-throughput-control: -0.08333333333333334
```

Therefore the candidate does not provide EE objective contribution evidence
beyond anti-collapse alone. Scalar reward improved diagnostically, but scalar
reward remains forbidden as success evidence.

The detailed execution report is:

```text
docs/research/catfish-ee-modqn/hobs-active-tx-ee-qos-sticky-broader-effectiveness-gate.execution-report.md
```

## 2026-05-01 HOBS active-TX EE / DPC feasibility update

本節同步最新的 HOBS active-TX EE objective-substitution feasibility chain。
它不是 RA-EE，也不是 Catfish，也不是 full EE-MODQN effectiveness claim。

最新結論是：

```text
HOBS active-TX EE formula / reward wiring: PASS, scoped
SINR structural audit: PASS, but negligible at current MODQN operating point
channel-regime / antenna-gain path: BLOCK as paper-backed MODQN continuation
HOBS-inspired DPC sidecar denominator gate: PASS
tiny learned-policy Route D denominator check: BLOCK
capacity-aware anti-collapse assignment gate: BLOCK
QoS-sticky overflow anti-collapse gate: PASS, scoped
QoS-sticky robustness / attribution gate: PASS, scoped
QoS-sticky broader-effectiveness gate: BLOCK
CP-base bounded matched pilot: BLOCK
EE-MODQN effectiveness: NOT PROMOTED / STOP CURRENT QOS-STICKY AND CP-BASE EE OBJECTIVE ROUTES
```

### Formula Boundary

目前最安全的主指標是 active-beam transmit-power EE，而不是 total spacecraft
EE：

```text
r1_hobs_active_tx_ee(t)
  = sum_u R_u(t) / (sum_active_beams z_b(t) * p_b(t) + epsilon)
```

其中 `R_u(t)` 是 simulator 內的 user throughput，`z_b(t)` 是 active-beam
indicator，`p_b(t)` 是 active beam 的 transmit power in linear W。Inactive
beam 必須貢獻 `0 W`。這個 metric 可以稱為 transmission-side EE / active-TX
EE，不可寫成 physical spacecraft energy saving。

Composite denominator、circuit/static/processing overhead、handover-aware EE、
utility-form EE 都只能當 sensitivity / ablation，除非後續有新的
LEO-specific parameter evidence 和明確 assumption disclosure。

### What Passed

HOBS active-TX EE 公式可以乾淨接進 MODQN：

1. `active_beam_mask`、`beam_transmit_power_w`、`total_active_beam_power_w`
   與 throughput 都已在 reward-computation 時可用。
2. `hobs-active-tx-ee` 是 opt-in reward mode，baseline default 仍是
   `r1=throughput`。
3. Formula wiring 不需要改 frozen baseline。
4. HOBS-inspired DPC sidecar 可以讓 denominator 在 eval 中變動；Route `B`
   的 denominator gate 有 `denominator_varies_in_eval=true`、
   `active_power_single_point_distribution=false`、`power_control_activity_rate=1.0`
   與 zero power-guardrail violations。

這表示先前最擔心的「EE 分母只能是常數」已經被縮小：在 opt-in DPC surface
下，分母可以變動。

### What Still Failed

Route `D` 的 tiny matched learned-policy denominator check 仍是 **BLOCK**。
Matched boundary 是 PASS：control / candidate 使用相同 DPC sidecar、相同
seeds `100, 200, 300, 400, 500`、相同 `5` episodes、相同 evaluation
schedule、相同 checkpoint rule、相同 environment / weights / hyperparameters；
唯一差異是：

```text
control:   r1 = throughput
candidate: r1 = hobs-active-tx-ee
```

Candidate learned greedy eval 的關鍵結果：

```text
denominator_varies_in_eval: true
all_evaluated_steps_one_active_beam: true
active_beam_count_distribution: {"1.0": 50}
active_power_single_point_distribution: false
throughput_vs_ee_pearson: 0.19303453314619476
same-policy throughput-vs-EE rescore ranking change: true
served_ratio: 1.0
handover_count: 423
```

所以最新狀態不是「分母固定導致 EE 沒意義」。新的 blocker 是：即使 DPC 讓
分母變動、且 throughput-vs-EE ranking 可以分離，learned MODQN beam-selection
policy 仍然在所有 evaluated steps collapse 到 one active beam。

### 2026-05-01 Anti-Collapse Gate Result

第一個 anti-collapse / capacity / assignment gate 已完成，結果是 **BLOCK**，
不是 EE-MODQN effectiveness promotion。Worker 使用新的
`configs/hobs-active-tx-ee-anti-collapse-*` 與
`artifacts/hobs-active-tx-ee-anti-collapse-*` namespace 測試
`capacity-aware-greedy-assignment`。Control / candidate 都使用
`r1=hobs-active-tx-ee` 與相同 DPC sidecar；唯一 intended difference 是
candidate 啟用 opt-in assignment constraint。

Matched boundary 是 PASS，candidate 也確實移除 one-active-beam collapse：

```text
all_evaluated_steps_one_active_beam=false
active_beam_count_distribution={"2.0": 50}
denominator_varies_in_eval=true
active_power_single_point_distribution=false
power_control_activity_rate=0.5777777777777777
throughput_vs_ee_pearson=-0.0461997871581038
same-policy throughput-vs-EE rescore ranking change=true
```

但是 predeclared guardrails 失敗：

```text
p05_throughput_ratio_vs_control=0.28501578600107075 < 0.95
handover_delta=+495 > +25
r2_delta=-0.4949999999999999 < -0.05
```

因此這個 gate 只證明「最小 capacity-aware constraint 可以強迫 greedy eval
不再全程 one active beam」，但同時也證明此 candidate 以不可接受的 tail
throughput 與 handover / `r2` regression 達成 anti-collapse。Scalar reward
diagnostic 有小幅增加，但不能作成功依據。

Authority report:

```text
docs/research/catfish-ee-modqn/hobs-active-tx-ee-anti-collapse-design-gate.execution-report.md
```

### 2026-05-01 QoS-Sticky Anti-Collapse Gate Result

第二個 anti-collapse gate 測試 `qos-sticky-overflow-reassignment`，結果是
**PASS, scoped**。它不是 EE-MODQN effectiveness promotion，而是 bounded
anti-collapse evidence。

這個 candidate 使用新的
`configs/hobs-active-tx-ee-qos-sticky-anti-collapse-*` 與
`artifacts/hobs-active-tx-ee-qos-sticky-anti-collapse-*` namespace。Control /
candidate 都使用 `r1=hobs-active-tx-ee` 與相同 DPC sidecar；candidate 的唯一
intended difference 是 opt-in QoS-sticky overflow reassignment。

機制重點：

1. 先計算 normal learned greedy actions。
2. 只在 projected overload 時介入。
3. 不使用 forced `min_active_beams_target`。
4. 只允許 QoS-safe sticky override。
5. `nonsticky_move_count=0`。

Matched boundary 是 PASS，candidate 結果：

```text
all_evaluated_steps_one_active_beam=false
active_beam_count_distribution={"7.0": 50}
denominator_varies_in_eval=true
active_power_single_point_distribution=false
power_control_activity_rate=1.0
throughput_vs_ee_pearson=0.6166801450896923
same-policy throughput-vs-EE rescore ranking change=true
p05_throughput_ratio_vs_control=2.522568929129207
served_ratio_delta=0.0
outage_ratio_delta=0.0
handover_delta=-211
r2_delta=+0.21100000000000002
budget/per-beam/inactive-beam violations=0/0/0
```

因此此 gate 當時沒有觸發 anti-collapse-specific 停損點。可說 QoS-sticky
overflow hook 在 tiny matched protocol 中通過 bounded anti-collapse gate；
不可說 EE-MODQN effectiveness 已成立。

Authority report:

```text
docs/research/catfish-ee-modqn/hobs-active-tx-ee-qos-sticky-anti-collapse-design-gate.execution-report.md
```

### 2026-05-01 QoS-Sticky Robustness / Attribution Gate Result

QoS-sticky overflow 的 robustness / mechanism-attribution gate 也已完成，結果是
**PASS, scoped**。Protocol 使用 `3` 個 matched training seed triplets：
`[42,1337,7]`、`[43,1338,8]`、`[44,1339,9]`，同樣保持 `5` episode bounded
budget、相同 eval seeds、相同 HOBS active-TX EE `r1`、相同 DPC sidecar、相同
checkpoint protocol。

Primary aggregate：

```text
all_evaluated_steps_one_active_beam=false
active_beam_count_distribution={"7.0": 150}
denominator_varies_in_eval=true
active_power_single_point_distribution=false
power_control_activity_rate=1.0
throughput_vs_ee_pearson=0.5022146200296446
same-policy throughput-vs-EE rescore ranking change=true
p05_throughput_ratio_vs_control=3.105617320531727
served_ratio_delta=0.0
outage_ratio_delta=0.0
handover_delta=-292.00000000000006
r2_delta=+0.29200000000000004
budget/per-beam/inactive-beam violations=0/0/0
```

Per-seed primary 全部 PASS，且 threshold `45` / `55` sensitivity 也 PASS，沒有
偵測到 threshold fragility。Mechanism attribution 的重點是：QoS ratio guard
在這個 bounded surface 沒有 binding，因為 relaxed-QoS 與 stricter-QoS ablation
產生相同 aggregate result；目前應歸因於 sticky overflow reassignment 與
non-sticky / handover protections，而不是特定 `qos_ratio_min=0.95`。

Authority report:

```text
docs/research/catfish-ee-modqn/hobs-active-tx-ee-qos-sticky-robustness-gate.execution-report.md
```

### Decision

HOBS active-TX EE 公式本身與 MODQN wiring 是可行的。DPC sidecar 也證明分母
可以變動。Route `D` 仍有 one-active-beam collapse hard stop；第一個
capacity-aware anti-collapse candidate 雖然移除 collapse，卻 failed p05
throughput 與 handover / `r2` guardrails。第二個 QoS-sticky overflow
candidate 通過 bounded anti-collapse gate，robustness / attribution gate 也
是 scoped PASS。

但 broader-effectiveness gate 已經 BLOCK：`anti-collapse-throughput-control`
解釋了主要 gain boundary，`qos-sticky-ee-candidate` 沒有比該 control 改善
active-TX EE，且 handover / `r2` guardrails 失敗。因此目前已到 current
QoS-sticky EE objective-contribution route 的停損點。若繼續 EE research，現在
只能沿用 CP-base non-codebook continuous-power boundary；design gate 與
implementation-readiness / boundary-audit slice 已經 PASS，但 bounded matched
pilot 已經 BLOCK：candidate 輸給 throughput + same guard + same continuous
power control，且 handover / `r2` guardrails 失敗。下一步不是加長 Route `D`
training、Catfish repair、Phase `03C`、RA-EE association、繼續 tuning 這條
QoS-sticky EE objective route，或 rerun / tune 這個 CP-base candidate。

### Allowed Claims

目前可說：

1. HOBS active-TX EE 可以作為 opt-in MODQN reward / metric 被計算。
2. DPC sidecar 可以讓 active transmit-power denominator 變動。
3. Route `D` 顯示 denominator variability 與 throughput-vs-EE ranking separation
   可以同時出現。
4. Route `D` 也顯示 learned policy 仍 one-active-beam collapse，因此 current
   EE-MODQN effectiveness 未成立。
5. 第一個 capacity-aware anti-collapse gate 移除 one-active-beam collapse，但因
   p05 throughput 與 handover / `r2` guardrails 失敗而 BLOCK。
6. QoS-sticky overflow anti-collapse gate 在 tiny matched protocol 中移除
   one-active-beam collapse，且通過 p05、served/outage、handover / `r2`、power
   accounting guardrails。
7. QoS-sticky robustness gate 在 `3` 個 matched seed triplets 與附近 threshold
   ablation 上重現 scoped PASS；目前 attribution 是 sticky overflow reassignment
   under non-sticky / handover protections。
8. QoS-sticky broader-effectiveness gate 是 BLOCK；正面 claim 必須縮成 scoped
   anti-collapse robustness，不能 claim EE objective contribution。
9. CP-base design gate 與 implementation-readiness / boundary-audit slice 是
   readiness evidence。
10. CP-base bounded matched pilot 是 bounded negative evidence：matched boundary
   passed，但 EE candidate 輸給 throughput + same guard + same continuous power
   control，且 handover / `r2` guardrails 失敗。

### Forbidden Claims

不可 claim：

1. EE-MODQN effectiveness,
2. Phase `03D` failure 已被推翻,
3. physical spacecraft energy saving,
4. HOBS optimizer reproduction,
5. DPC 是 MODQN paper-backed,
6. Catfish / Multi-Catfish / Catfish-EE effectiveness,
7. scalar reward as success evidence,
8. denominator variability alone proves energy-aware learning,
9. more Route `D` training is the default next gate,
10. capacity-aware assignment anti-collapse success proves EE-MODQN effectiveness,
11. QoS-sticky anti-collapse scoped PASS proves general EE-MODQN effectiveness,
12. QoS-sticky robustness scoped PASS proves physical energy saving or HOBS optimizer behavior,
13. QoS-sticky broader-effectiveness gate proves EE objective contribution,
14. current QoS-sticky EE objective route should continue tuning by default,
15. CP-base prompt draft proves EE-MODQN effectiveness,
16. CP-base bounded matched pilot proves EE-MODQN effectiveness,
17. CP-base scalar reward gain rescues the failed EE / handover / r2 gate,
18. CP-base should be tuned or rerun by default after the stop conditions fired.

Authority reports:

```text
docs/research/catfish-ee-modqn/hobs-active-tx-ee-modqn-feasibility.execution-report.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-anti-collapse-design-gate.execution-report.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-qos-sticky-anti-collapse-design-gate.execution-report.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-qos-sticky-robustness-gate.execution-report.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-qos-sticky-broader-effectiveness-gate.execution-report.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-non-codebook-continuous-power-design-gate.execution-report.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-non-codebook-continuous-power-implementation-readiness.execution-report.md
docs/research/catfish-ee-modqn/prompts/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot.prompt.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot.execution-report.md
artifacts/hobs-active-tx-ee-non-codebook-continuous-power-bounded-pilot-summary/summary.json
docs/research/catfish-ee-modqn/energy-efficient/ee-formula-final-review-with-codex-2026-05-01.md
docs/research/catfish-ee-modqn/energy-efficient/modqn-r1-to-hobs-active-tx-ee-design-2026-05-01.md
```

## 2026-04-29 RA-EE closeout report section

本節是 RA-EE-02 到 RA-EE-09 後的正式 closeout。後續 agent 應沿用這個
claim boundary，不要重新打開 Phase 03C 或 RA-EE association proposal route。

```text
old EE-MODQN r1-substitution route: BLOCKED / STOP
RA-EE fixed-association deployable power allocation: PASS, scoped
RA-EE fixed-association RB / bandwidth candidate: AUDITABLE, NOT PROMOTED
RA-EE learned association / hierarchical RL / full RA-EE-MODQN: BLOCKED
Catfish for EE repair: BLOCKED
```

### Current State

目前可成立的 RA-EE 正面結論只有一個：在已揭露的 simulation channel /
mobility / traffic 假設、固定 association 軌跡、明確 finite-codebook power
contract、以及 matched held-out replay 下，RA-EE-07 的 deployable non-oracle
power allocator 可以提升 simulated `EE_system`，同時維持既定 p05 throughput、
served ratio、outage、total budget、per-beam power 與 inactive-beam-zero
guardrails。這是 scoped simulated-EE result，不是 physical energy saving。

RA-EE-09 也已完成 fixed-association normalized bandwidth/resource-share
pilot。它證明 comparison boundary 和 accounting 可稽核，但 tested
`bounded-qos-slack-resource-share-allocator` 在 held-out replay 上沒有通過
effectiveness gate：`EE_system` delta 是負值，predeclared resource-efficiency
delta 是負值，p05 throughput ratio 低於 `0.95`。

因此目前仍不是 learned association，不是 hierarchical RL，不是 joint
association + power training，不是 positive RB / bandwidth allocation
effectiveness，也不是 full RA-EE-MODQN。

### Evidence Summary

| Gate | Result | Claim boundary |
|---|---|---|
| `03D` | `BLOCKED / STOP` | Current EE-MODQN `r1` substitution route stops; do not continue with more episodes, selector tweaks, reward tuning, or Catfish. |
| `RA-EE-02` | `PASS to design` | Offline oracle / heuristic power-allocation upper-bound proof only. |
| `RA-EE-04` | `PASS, scoped` | Fixed-association centralized safe-greedy power-allocation bounded pilot. |
| `RA-EE-05` | `PASS, scoped` | Fixed-association robustness and held-out validation for the power allocator. |
| `RA-EE-06` | `BLOCKED` | Association counterfactual does not authorize learned hierarchical association training. |
| `RA-EE-06B` | `BLOCKED` | Proposal refinement / oracle distillation does not beat matched fixed association with the same safe-greedy allocator. |
| `RA-EE-07` | `PASS, scoped` | Fixed-association deployable non-oracle finite-codebook stronger power allocator beats the matched fixed-association RA-EE-04/05 safe-greedy-power-allocator on held-out simulated `EE_system` while preserving QoS and power guardrails. |
| `RA-EE-08` | `BLOCKED` | Fair association re-evaluation with the same stronger allocator shows no positive association proposal path. |
| `RA-EE-09` | `NEEDS MORE EVIDENCE / not promoted` | Fixed-association normalized resource-share comparison is auditable, but the tested bounded QoS-slack bandwidth candidate loses to equal-share control on held-out `EE_system`, predeclared resource-efficiency, and p05 guardrail. |

The decisive split is RA-EE-07 versus RA-EE-08. RA-EE-07 supports a stronger
fixed-association power allocator. RA-EE-08 then uses that same stronger
allocator for fixed association and proposal association, and all proposal
families remain negative on held-out `EE_system`. Therefore the association
route should stop for now.

### Allowed Claims

Allowed wording:

```text
Under the disclosed simulation setting and fixed-association held-out replay,
the RA-EE-07 deployable non-oracle finite-codebook power allocator improves
simulated system EE over the matched fixed-association RA-EE-04/05 safe-greedy
power allocator while preserving the declared QoS and power guardrails.
```

Also allowed:

1. `EE_system = sum_i R_i(t) / sum_active_beams P_b(t)` is the claim metric.
2. RA-EE-04 / 05 / 07 are fixed-association deployable power-allocation
   evidence.
3. RA-EE-06 / 06B / 08 are negative association-route evidence.
4. RA-EE-09 is negative RB / bandwidth candidate evidence under fixed
   association and matched RA-EE-07 power allocation.
5. Phase 04 Catfish, if started, is only an independent original-MODQN reward
   feasibility branch.

Paper-safe Chinese wording:

```text
在已揭露的模擬環境與固定 association held-out replay 下，RA-EE-07 的
deployable non-oracle finite-codebook power allocator 相較於 matched
fixed-association RA-EE-04/05 safe-greedy power allocator 提升了 simulated
system EE，且維持既定 QoS 與 power guardrails。
```

Avoid wording that says `same throughput with less power` unless the local table
also reports the exact throughput parity / delta. The safer claim is
`simulated EE_system improves while QoS guardrails are preserved`.

### Forbidden Claims

Do not claim:

1. full RA-EE-MODQN,
2. learned association effectiveness,
3. hierarchical RL effectiveness,
4. joint association + power training,
5. old EE-MODQN effectiveness,
6. HOBS optimizer behavior,
7. physical energy saving,
8. Catfish, Multi-Catfish, or Catfish-EE-MODQN effectiveness,
9. RB / bandwidth allocation effectiveness,
10. per-user EE credit as system EE,
11. scalar reward alone as success evidence,
12. oracle rows as deployable runtime methods,
13. full paper-faithful reproduction,
14. Catfish as an EE repair mechanism,
15. `same throughput with less physical power` unless separately proven by the
    reported simulation table.

### Route Decision

RA-EE is now closeout-ready at the current evidence boundary, and Catfish
remains parked unless the explicit goal changes. RA-EE-09 has completed the
bounded fixed-association pilot:

```text
RA-EE-09 fixed-association RB / bandwidth allocation matched replay
```

RA-EE-09 must not carry a full RA-EE-MODQN claim. It kept association fixed and
used RA-EE-07's deployable power allocator as the matched control boundary. The
tested bounded RB / bandwidth allocator did not add held-out
resource-allocation value over equal-share control. Phase 04 Catfish remains
parked unless the explicit goal changes to original-MODQN Catfish feasibility.

Default next action is RA-EE closeout / paper-section synthesis, not more
RA-EE-09 tuning. Any new RB / bandwidth route must be opened as a new explicit
design gate with a new candidate and acceptance boundary.

The RA-EE-09 design and execution reports are recorded at:

```text
docs/research/catfish-ee-modqn/ra-ee-09-completion-design-gate.md
docs/research/catfish-ee-modqn/ra-ee-09-fixed-association-rb-bandwidth.execution-report.md
```

## 2026-04-29 RA-EE-09 implementation update

RA-EE-09 已實作為 **fixed-association normalized bandwidth/resource-share
allocation gate**。它仍是 offline replay，不是 training、learned association、
hierarchical RL、joint association + power/resource training、Catfish、Phase 03C
continuation，也不是 full RA-EE-MODQN。新的 config 是
`configs/ra-ee-09-fixed-association-rb-bandwidth-control.resolved.yaml`；
artifact 寫在
`artifacts/ra-ee-09-fixed-association-rb-bandwidth-candidate-pilot/paired-comparison-vs-control/`。

Primary matched comparison 是 fixed association +
`RA-EE-07 deployable-stronger-power-allocator` + equal-share resource control
對 fixed association + **same** RA-EE-07 power allocator +
`bounded-qos-slack-resource-share-allocator`。兩者使用相同 held-out seeds
`600, 700, 800, 900, 1000`、相同五條 held-out fixed-association trajectories、
相同 evaluation schedule、相同 effective power schedule。matched step count 是
`250`。

Boundary proof 是 **PASS**：same association schedule hash、same effective power
schedule hash、same RA-EE-07 power boundary、resource allocation after power
selection、no resource-to-power feedback 全部成立。Resource accounting 也是
**PASS**：control 與 candidate 都有 exact active per-beam resource sum、
inactive beam zero resource、resource budget violation count `0`。

Effectiveness gate 是 **not promoted**。overall control `EE_system` 是
`882.2348210629095`，candidate `EE_system` 是 `835.5862303183848`，delta
`-46.64859074452477`。sum throughput delta 是 `-76282.10801498406`，p05
throughput ratio 是 `0.9016412169223311`，低於 `0.95` guardrail。served ratio
delta 是 `0.0`、outage ratio delta 是 `0.0`、handover delta 是 `0`，但
predeclared `p05_throughput_per_active_resource_budget` delta 也是負值
`-0.1944519284254136`。所有五條 held-out trajectory 的 `EE_system` delta 都是
負值，throughput-vs-EE ranking separation 也沒有發生；control 同時是 throughput
和 `EE_system` winner。

RA-EE-09 結論是：

```text
boundary/accounting PASS
tested bounded resource-share candidate NOT PROMOTED
RB / bandwidth allocation effectiveness remains forbidden
```

論文中可以使用這個 negative result 來說明：在固定 association 且 power allocator
已 matched 的情況下，單純 deterministic bounded bandwidth redistribution 沒有提供
可接受的 held-out EE / QoS-preserving resource-efficiency improvement。

## 2026-04-29 RA-EE-08 implementation update

RA-EE-08 已實作為 **offline association re-evaluation gate**，仍是 offline
deterministic proposal replay，不是 training、learned association、hierarchical
RL、joint association + power training、Catfish、RB / bandwidth allocation，也不是
full RA-EE-MODQN。新的 config 是
`configs/ra-ee-08-offline-association-reevaluation.resolved.yaml`；artifact 寫在
`artifacts/ra-ee-08-offline-association-reevaluation/`。

Primary matched comparison 是 matched fixed association +
`deployable-stronger-power-allocator` 對 proposal association + **same**
`deployable-stronger-power-allocator`。重放的 association proposal families 是
`active-set-load-spread`、`active-set-quality-spread`、
`active-set-sticky-spread`、`sticky-oracle-count-local-search`、
`p05-slack-aware-active-set`、`power-response-aware-load-balance`、
`bounded-move-served-set` 與 `oracle-score-topk-active-set`。`proposal +
safe-greedy`、`fixed + safe-greedy`、`fixed + constrained-power oracle`、
`association-oracle + constrained-power oracle`、以及 `association-oracle +
deployable allocator` 都只作 diagnostic，不進 acceptance。

RA-EE-08 結果是 **BLOCKED**。八個 held-out proposal 都避免 one-active-beam
collapse，但沒有任何 proposal 對 matched fixed + same deployable allocator 有正
`EE_system` delta；predeclared primary family
`power-response-aware-load-balance` 也是負 delta。held-out EE delta 分別為
`active-set-load-spread -8.283605568482926`、
`active-set-quality-spread -8.283605568482926`、
`active-set-sticky-spread -3.3928346793936726`、
`sticky-oracle-count-local-search -3.3928346793936726`、
`p05-slack-aware-active-set -3.3928346793936726`、
`power-response-aware-load-balance -3.3928346793936726`、
`bounded-move-served-set -3.3928346793936726`、以及
`oracle-score-topk-active-set -6.070621808780743`。

QoS / burden split 是：六個 7-active-beam proposal 的 p05 ratio 都 >= `0.95`
且 served/outage 沒壞，但仍是 negative EE delta；兩個 2-active-beam proposal 的
p05 ratio 只有 `0.42255785095605874`，其中 `active-set-load-spread` 的
moved-user ratio 是 `0.9876`。primary rows 沒有 budget、per-beam 或
inactive-power violations。

RA-EE-08 也沒有 meaningful oracle gap closure。`association-oracle +
deployable allocator` 對 fixed + deployable control 的 held-out EE delta 是
`-2.7592661903154294`，`association-oracle + constrained-power oracle` 是
`-7.442920209643489`，所以 aggregate oracle gap closure 不適用。這表示
RA-EE-07 的 fixed-association deployable allocator 成為更強比較基準後，RA-EE-06 /
06B 可見的 association oracle path 沒有在這個公平 pairing 下保留正 gap。

## 2026-04-29 RA-EE-07 implementation update

RA-EE-07 已實作為 **constrained-power allocator distillation gate**，仍是
offline replay，不是 learned association、hierarchical RL、joint association
+ power training，也不是 full RA-EE-MODQN。新的 config 是
`configs/ra-ee-07-constrained-power-allocator-distillation.resolved.yaml`；
artifact 寫在
`artifacts/ra-ee-07-constrained-power-allocator-distillation/`。

Primary comparison 是 matched fixed association + RA-EE-04/05
`safe-greedy-power-allocator` 對 matched fixed association +
deployable non-oracle stronger allocator。實作的 deployable candidates 是
`p05-slack-aware-trim-tail-protect-boost`、
`bounded-local-search-codebook`、`finite-codebook-dp-knapsack` 與
`deterministic-hybrid-runtime`；completed held-out replay 裡 hybrid 每一步選到
`bounded-local-search-codebook`。這個選擇只使用 current runtime channel/load/QoS
slack 與 finite codebook evaluation，不使用 oracle labels、future outcomes 或
held-out answers。`fixed-1W`、matched fixed + constrained-power oracle
isolation、association proposal + same deployable allocator、association oracle
+ constrained-power oracle upper bound 都只作 diagnostic。RA-EE-06B association
proposal buckets 有輸出為 bounded diagnostics；primary fixed-association replay
沒有 step cap。

RA-EE-07 結果是 **PASS as offline fixed-association deployable
power-allocator gate only**。held-out 五條 fixed-association trajectories 全部對
matched safe-greedy 有正 `EE_system` delta 且全部 accepted：
`random-valid-heldout` `+7.095305632511327`、`spread-valid-heldout`
`+5.155057818148293`、`load-skewed-heldout` `+3.5780032540779985`、
`mobility-shift-heldout` `+7.483104400382672`、`mixed-valid-heldout`
`+5.918477335805619`。p05 throughput ratio 分別是 `0.975101299563657`、
`0.9995533609188448`、`1.0421330635721502`、`0.986358235603686`、
`1.0743825126983608`；served ratio 沒有下降、outage ratio 沒有上升，budget /
per-beam / inactive-power violations 為 `0`。

Held-out aggregate oracle gap closure 是 `1.0` under the RA-EE-07 feasible-codebook
oracle diagnostic；positive seeds 是 `4 / 5`，max positive trajectory delta share
是 `0.2560081286323895`，max positive seed delta share 是
`0.28553406270120996`，所以 gain 沒有集中在單一 trajectory 或 seed。這個 PASS
只支持「在此 simulation envelope、fixed association、finite-codebook power
contract 下，一個 deployable non-oracle power allocator 可以 beat matched
RA-EE-04/05 safe-greedy 並關閉本 gate 的 constrained-power oracle gap」。它仍不能
claim learned association、hierarchical RA-EE、joint association + power
training、RB / bandwidth allocation effectiveness、HOBS optimizer、physical
energy saving、Catfish-EE，或 full paper-faithful reproduction。

Paper-facing precision:

```text
finite codebook levels: {0.5, 0.75, 1.0, 1.5, 2.0} W
per-beam cap: 2.0 W
total active-beam budget: 8.0 W
inactive-beam policy: 0 W
held-out trajectory positives: 5 / 5
held-out seed positives: 4 / 5
max positive trajectory delta share: 0.2560081286323895
max positive seed delta share: 0.28553406270120996
```

The deterministic hybrid selected `bounded-local-search-codebook` on every
held-out step in this evaluation. This is an observed held-out behavior, not a
license to relabel the whole method as a generic bounded-local-search method.

## 2026-04-29 RA-EE-06B implementation update

RA-EE-06B 已實作為 **association proposal refinement / oracle distillation
audit**，仍是 offline trace export + deterministic proposal rule gate，不是
learned hierarchical RL，也不是 full RA-EE-MODQN。新的 config 是
`configs/ra-ee-06b-association-proposal-refinement.resolved.yaml`；artifact
寫在 `artifacts/ra-ee-06b-association-proposal-refinement/`。

這輪新增完整 per-step oracle trace：
`ra_ee_06b_oracle_trace.csv`，並輸出 candidate summary、guardrail checks、
summary JSON 與 `review.md`。trace 包含 active beam mask/count、active-set
source、load distribution/slack/gap、per-user selected/top-k quality、
best-vs-selected margin、valid beam count、current/control/oracle/selected
beam、moved flags、rank/offset distance proxy、handover/r2、p05
control/candidate/oracle throughput、p05 ratio/slack、selected power vector、
demoted beams、denominator、safe-greedy demotion counts、tail-user IDs、oracle
selected policy/profile、EE delta、accepted flag 與 rejection reason。

實作的 deterministic proposal rules 是
`sticky-oracle-count-local-search`、`p05-slack-aware-active-set`、
`power-response-aware-load-balance`、`bounded-move-served-set`、以及
`oracle-score-topk-active-set`。primary comparison 仍是 proposal rule + 同一個
RA-EE-04/05 `safe-greedy-power-allocator` 對 matched `fixed-hold-current` +
同一 allocator；`proposal + fixed-1W`、`per-user-greedy-best-beam`、
`association-oracle+same-safe-greedy`、`association-oracle+constrained-power`
與 `matched fixed + constrained-power oracle` 都只作 diagnostic / upper-bound
使用。

RA-EE-06B 結果是 **BLOCKED**。五個 held-out proposal 全部避免 one-active-beam
或 pathological two-beam overload collapse，且 denominator 都有 variability，
但沒有任何 primary candidate 對 matched fixed association + same safe-greedy
allocator 有正 EE delta。四個 sticky/slack/load-balance/bounded-move proposal 的
held-out EE delta 都是 `-1.7672459194036492`，p05 ratio
`1.2382114044314652`，moved-user ratio `0.0054`；`oracle-score-topk-active-set`
的 EE delta 是 `-2.8533857556310522`，p05 ratio `1.3748199249811`，
moved-user ratio `0.0854`。accepted candidate count 是 `0`。

Diagnostic oracle split 仍然存在：`association-oracle+same-safe-greedy` 在
held-out 是負 EE delta `-1.476841467806139`，但
`association-oracle+constrained-power-upper-bound` 是正 EE delta
`+0.9583236804177204`、p05 ratio `1.0204597070289083`。所以目前可見正路徑仍
需要 constrained-oracle power，而不是 proposal rule + same safe-greedy
allocator。這觸發 `proposal_gains_require_constrained_oracle_power=true` 與
`held_out_EE_delta_negative_or_concentrated=true`；不能進 learned hierarchical
training。

## 2026-04-29 RA-EE-06 implementation update

RA-EE-06 已實作為 **association counterfactual / oracle design gate**，method
label 是 `RA-EE hierarchical association + power counterfactual`，不是 full
RA-EE-MODQN。新的 config 是
`configs/ra-ee-06-association-counterfactual-oracle.resolved.yaml`；artifact
寫在 `artifacts/ra-ee-06-association-counterfactual-oracle/`。

這輪只做離線 active-set / served-set association counterfactual。matched
control 是 `fixed-hold-current` association + RA-EE-04/05
`safe-greedy-power-allocator`；candidate 是
`active-set-load-spread`、`active-set-quality-spread`、`active-set-sticky-spread`
再接同一個 `safe-greedy-power-allocator`；另有
`association-oracle+constrained-power-upper-bound` 作 diagnostic upper bound。
沒有 learned association、joint association + power training、Catfish、
multi-Catfish、RB / bandwidth allocation，也沒有修改 frozen baseline。

RA-EE-06 結果是 **BLOCKED for learned hierarchical association training**。
三個 held-out active-set proposal 都避免了 one-active-beam collapse，但都沒有
對 matched fixed-association + same allocator 產生正 `EE_system` delta：
`active-set-load-spread` 與 `active-set-quality-spread` 的 EE delta 都是
`-2.919243206981264`，p05 ratio 都是 `0.39743071159396487`；
`active-set-sticky-spread` 的 EE delta 是 `-1.7672459194036492`，p05 ratio 是
`1.2382114044314652`。因此沒有 accepted candidate，不能進 learned
hierarchical RL。

重要的是，diagnostic oracle 在 held-out 上對同一 matched control 有正 EE delta
`+0.44226236820429676`，p05 ratio `0.9711843089200792`，且通過 QoS / budget
guardrails。這表示 association + power 的可行上界仍存在，但目前 minimal
active-set proposal rule 不足。下一步若要繼續，應是 **RA-EE-06B association
proposal refinement / oracle distillation**，仍不得直接做 joint association +
power training。

## 2026-04-29 RA-EE-05 implementation update

RA-EE-05 已實作為 **fixed-association robustness and held-out validation**，
method label 是 `RA-EE fixed-association centralized power allocator`，不是
full RA-EE-MODQN。新的 config 是
`configs/ra-ee-05-fixed-association-robustness.resolved.yaml`；artifact 寫在
`artifacts/ra-ee-05-fixed-association-robustness/`。

這輪使用 calibration / train-like bucket
`[hold-current, random-valid, spread-valid]` 搭配 eval seeds
`[100, 200, 300, 400, 500]`，以及 held-out bucket
`[random-valid-heldout, spread-valid-heldout, load-skewed-heldout,
mobility-shift-heldout, mixed-valid-heldout]` 搭配 eval seeds
`[600, 700, 800, 900, 1000]`。association 仍由固定 trajectory 事先給定；
沒有 learned association、joint association + power training、Catfish、
multi-Catfish、RB / bandwidth allocation，且沒有修改 frozen baseline。

RA-EE-05 結果是 **PASS as fixed-association robustness evidence only**。
held-out 五條 non-collapsed trajectories 全部對 matched fixed-control 有正
`EE_system` delta，且全部通過 p05 throughput / served ratio / outage /
budget / per-beam / inactive-power guardrails。held-out EE delta 分別為
`+2.8552862308795284`、`+0.021591980638277164`、`+1.8625878077186826`、
`+2.238167231124862`、`+1.491732050135397`；p05 throughput ratio 分別為
`0.9549524027957726`、`0.9963484525708493`、`1.0`、
`0.9532709399187361`、`1.0`。accepted held-out trajectories 全部
`denominator_varies_in_eval=true`，selected power vectors 與 total active
power 皆非單點分布，且 throughput winner 是 `fixed-control`、EE winner 是
`safe-greedy-power-allocator`，ranking 分離成立。這只能支持「固定
association 的集中式 power allocator 通過 held-out robustness gate」；仍不能
claim learned association、full RA-EE-MODQN effectiveness、HOBS optimizer、
physical energy saving、Catfish-EE，或 full paper-faithful reproduction。

## 2026-04-29 RA-EE-04 implementation update

RA-EE-04 已實作為 **fixed-association centralized power-allocation bounded
pilot**，不是 old EE-MODQN continuation。新的 config 是
`configs/ra-ee-04-bounded-power-allocator-control.resolved.yaml` 與
`configs/ra-ee-04-bounded-power-allocator-candidate.resolved.yaml`；artifact
寫在 `artifacts/ra-ee-04-bounded-power-allocator-control-pilot/`、
`artifacts/ra-ee-04-bounded-power-allocator-candidate-pilot/` 與
`artifacts/ra-ee-04-bounded-power-allocator-candidate-pilot/paired-comparison-vs-control/`。

這輪結果是 **PASS as bounded fixed-association power-allocation pilot only**。
candidate `safe-greedy-power-allocator` 在 `hold-current`、`random-valid`、
`spread-valid` 三個 non-collapsed fixed trajectories 上都讓
`denominator_varies_in_eval=true`，且 `EE_system` 對 matched fixed 1 W
control 分別提升 `+3.170352486871593`、`+1.9471815289625738`、
`+0.04925456347802992`，同時 p05 throughput ratio 分別為
`0.9762855249221921`、`0.9645767924959346`、`0.9932220296454757`，通過
QoS / budget / inactive-power guardrails。這只能支持「固定 association 的
集中式 power allocation pilot gate 已通過」；仍不能 claim learned
association、完整 RA-EE-MODQN effectiveness、HOBS optimizer、physical energy
saving、Catfish-EE，或 full paper-faithful reproduction。

## 2026-04-28 implementation update

Phase 03C-C 已把 Phase 03C-B 的 static / counterfactual power-MDP precondition
推到 bounded paired runtime pilot，但結果是 **BLOCKED**，不是 promote。新的
control / candidate config 分別是
`configs/ee-modqn-phase-03c-c-power-mdp-control.resolved.yaml` 與
`configs/ee-modqn-phase-03c-c-power-mdp-candidate.resolved.yaml`；candidate
啟用了 `runtime-ee-selector`，comparison artifact 寫在
`artifacts/ee-modqn-phase-03c-c-power-mdp-candidate-pilot/paired-comparison-vs-control/`。

最重要的結果是：candidate best-eval 仍然每一步都是 one active beam，
`selected_power_profile` 也完全 collapse 成 `fixed-low`，所以
`denominator_varies_in_eval=false`、active power 是單點 `0.5 W` 分布。
雖然 `EE_system` aggregate 對 control 有極小正 delta
(`+0.000270185470526485`)，但 p05 throughput 掉約 `50%`，且
throughput-vs-EE Pearson / Spearman 仍接近 `1.0`、same-policy rescore ranking
沒有改變。因此這輪實作強化了本報告原本的結論：不能 claim
EE-MODQN effectiveness，不能把 scalar reward 或 per-user EE credit 當主證據；
下一步若要重開，必須先解 one-beam collapse 和真正 denominator-sensitive
action / power coupling，而不是加長同一路線的 training。

## Short verdict

我的短結論是：**應暫停目前這條 EE-MODQN training 路線的擴大訓練與任何 effectiveness claim，先把問題重定義成「顯式 power-coupling 的新方法」再往下走。** 不是因為 EE 指標本身錯，而是因為目前專案裡的學習行為仍然是 **beam-only handover policy**，而評估時 learned policy 幾乎每一步都收斂成 **one active beam / total active power = 2.0 W**，導致 `EE_system` 在實際 evaluation 裡沒有被 policy 真正驅動；Phase 03A/03B 已經明確顯示 `denominator_varies_in_eval=false`、`EE_system` 與 control 打平、throughput–EE correlation 幾乎等於 1，且 repo 內部 gate 也已寫明「更多 episode 不是下一個 gate」。在這種狀態下，繼續做 reward normalization、r3 calibration、或直接把 Catfish 疊上去，都不太可能把「EE = throughput / 幾乎固定常數」這個核心問題變成可辯護的 EE learning。fileciteturn11file0L1-L1 fileciteturn12file0L1-L1 fileciteturn15file0L1-L1

更精準地說，**beam-only action 對 handover optimization 是足夠的，但對 HOBS-style energy efficiency optimization 通常不夠。** HOBS 論文把 EE 明確寫成總吞吐除以總 beam transmit power，且把優化變數直接定義成 power、association indicator 與 beam-training set，之後再把問題拆成「beam training / handover」與「power allocation」兩個子問題，並另外給出 dynamic power control。相對地，MODQN 論文可確認支持的是 throughput / handover / load-balance 的多目標 handover MDP，而不是顯式 power-control MDP。這表示如果你要在這個專案裡追求真正的 EE，不該再把它當成「只換 r1」的 Phase 03 延伸，而要把它視為 **new extension / method-design surface**。citeturn4view1turn4view2turn5search0turn5search1 fileciteturn9file0L1-L1 fileciteturn13file0L1-L1

## HOBS / MODQN papers actually support what

**HOBS paper 實際支持的，是「EE 必須與顯式 beam transmit power 綁在一起」。** 在 HOBS 的系統模型裡，SINR 式子直接包含 `P_{n,m}(t)`，文中也明確說 `P_{n,m}(t)` 是 beam `m` 在 satellite `n` 的 transmit power；其 EE 定義則是 `E_eff(t) = R_tot(t) / Σ_n Σ_m P_{n,m}(t)`。更重要的是，它的總優化問題把決策變數直接寫成 `P, a, Φ`，並施加 per-beam power constraint 與 per-satellite total power constraint；進入方法設計後，又把原問題拆成「beam training association」與「power allocation」兩個子問題，接著在第二個子問題裡給了 dynamic power control 更新規則。也就是說，HOBS 不是在做「只靠 beam selection 間接碰到 EE」，而是在做 **beam / handover + explicit power allocation** 的聯合設計。citeturn1view0turn4view1turn4view2

**MODQN paper 實際支持的，則是「multi-objective beam handover」，不是 energy-aware power control。** 可取得的 paper metadata / abstract 明確表示，它把 multi-beam LEO handover 問題建成 multi-objective optimization，再轉成 MOMDP，用 MODQN 來 jointly maximize throughput、minimize handover frequency、keep load balanced。repo 內被 freeze 的 baseline surface 也完全對齊這個邏輯：`r1 = throughput`、`r2 = handover penalty`、`r3 = load balance`，action 是 across all beams 的 one-hot beam selection。repo guardrails 也明說 original MODQN baseline 只能作為 disclosed comparison baseline，不可被靜默改寫成 EE or Catfish follow-on。換句話說，**MODQN paper 支持 beam-handover MORL；它不支持你把 HOBS 的 EE 直接視為原始 MODQN 的內生 objective，也不支持把 power-control 說成原 paper action space 的一部分。**citeturn5search0turn5search1 fileciteturn8file0L1-L1 fileciteturn9file0L1-L1 fileciteturn13file0L1-L1

放到你的八個問題裡，第一題的答案就很清楚了：**在 LEO / NTN / multi-beam literature 裡，handover 本身常可用 beam-only action 建模；但一旦目標是 EE 或更完整的 resource optimization，常見做法就會把 power、bandwidth、time-slot、beam profile 等資源顯式納入決策。** 這不只在 HOBS 如此；其他 LEO / multibeam 資源配置文獻也把 joint power/channel allocation、joint power/bandwidth allocation、time/power/frequency 三維資源配置，視為主要設計面。citeturn8view0turn10view0turn10view1turn6search0turn9view0

## Current implementation gap

目前專案的核心 gap，不是在「EE 公式有沒有 audit 到」，而是在 **policy 與 denominator 沒有真正耦合**。repo 的 environment 目前仍以 one-hot beam action 為主：state 由 access vector、channel quality、beam offsets、beam loads 構成；action 是 across all beams 的 one-hot beam selection；baseline reward surface 仍是 throughput / handover / load balance。雖然 `step.py` 已額外實作 `beam_transmit_power_w`、`active_beam_mask`，也在 reward component 裡加入 `r1_energy_efficiency_credit` 與 `r1_beam_power_efficiency_credit`，但那是 follow-on credit surface，不等於 paper-backed 的 system EE MDP。fileciteturn13file0L1-L1

Phase 02B 的確把 `EE_system(t) = sum_i R_i(t) / sum_active_beams P_b(t)` 做成了可 audit 的 runtime surface，而且這個 surface 在 repo 規格裡被明確標示為 `active-load-concave`、inactive beam `0 W`、units = linear W；它讓 `beam_transmit_power_w[b]` 也能進入 SINR numerator path。可是同一份 execution report 與 config 都反覆強調：這只是 **Phase 02B synthesized allocation proxy**，目的是讓 EE denominator 可被 defensibly audit，**不是 HOBS paper-backed optimizer，也不是已被論文支持的 power-control policy**。fileciteturn10file0L1-L1 fileciteturn16file0L1-L1

真正讓 EE route 卡死的，是 Phase 03A/03B 的 learned behavior。repo 的 Phase 03A diagnostic review 已經指出：environment 裡 `denominator exists in environment = True`，但 `learned policies exercise denominator variability = False`；active beam count distinct values 只有 `[1.0]`，total active beam power distinct values 只有 `[2.0]`，而且 action mask 並沒有強迫 single-beam，主要診斷反而是 `r3 load balance insufficient to prevent collapse` 與 `per-user EE degenerates under collapse = True`。Phase 03B 再加入 reward normalization、r3 calibration、以及 `per-user-beam-ee-credit = R_i(t)/P_b(t)` 之後，best-eval 仍然在 `EE_system`、throughput、served ratio、handover count、active beam count mean、total active beam power mean 全數與 control 打平，並保持 `denominator_varies_in_eval = false`；同時計算出的 throughput-vs-EE correlation 幾乎是 1，同 policy rescoring 也不改排名。這代表目前的 EE-MODQN 不像是在學「energy-aware tradeoff」，而像是在學「throughput 的重標度版本」。fileciteturn12file0L1-L1 fileciteturn11file0L1-L1

所以第三題「如何避免 EE 退化成 throughput / constant？」在這個專案脈絡下，答案非常具體：**不是先改 reward，而是先讓 evaluation 時的 denominator 成為 policy-sensitive quantity。** 如果 learned policy 在 greedy eval 裡永遠只使用單一 active beam、總 active power 永遠 2 W，那你把 r1 換成 `R_i/P_b`、`R_i/(P_b/N_b)`、甚至再做 normalization，最後都只是在同一個 collapse surface 上重寫 scoring。repo 自己的 validation gate 已經把這件事形式化：若 `ee_denominator_varies_in_eval` 不成立、one-beam collapse 未解除、`EE_system` 未改善、throughput–EE ranking 未分離，就不能 promote。fileciteturn15file0L1-L1 fileciteturn11file0L1-L1

## Candidate design routes

如果你現在要重設方法，我認為有四條路，但只有其中兩條值得當下一個正式 gate。

**第一條路是 hierarchical handover + power-control design，我認為這是最可辯護的主路線。** 這條路與 HOBS 最相容，因為 HOBS 本來就把問題拆成 association / beam training 與 power allocation 兩個子問題；對你這個 repo 來說，則可以保留 MODQN-style handover backbone，同時額外加入一個顯式 beam-power controller，在 frame-level 或 slot-level 更新 `P_b(t)`。這樣做的好處是：一方面保留 MODQN 的可比較性，另一方面又能讓 `EE_system` 的 denominator 真的由 controller / policy 控制；壞處是它已經超出原始 MODQN paper，必須明確標成 **new extension / method-design surface**。citeturn4view0turn4view2 fileciteturn9file0L1-L1

**第二條路是離散化 joint action，例如 `(beam_id, power_level)`、`(beam_id, Δpower sign)`、或 `(beam_id, power-codebook index)`。** 這條路的優點是可以維持 DQN / MODQN 家族的離散 action machinery，不必直接跳進 continuous control；缺點是 action space 會放大，而且 per-user action 與 per-beam power 的耦合要很小心設計，否則容易造成多 user 同步動作對同一 beam power 的 conflict。這類 joint discrete resource action 在 LEO / multibeam RL 文獻裡並不罕見，例如 joint channel-power allocation 的 DQN 類方法直接把 action 定義成 `{m, p}`，其中 `m` 是 channel、`p` 是 power，state 則包含 beam / traffic / resource occupancy。若你堅持留在 MODQN 家族，這是最實務的 power-control MDP 版本，但同樣必須標成 **new extension**，不能說是 HOBS 或 MODQN 原論文已支持的原生 action。citeturn10view0turn10view1turn10view2

**第三條路是 profile / codebook action。** 也就是不讓 agent 直接輸出所有 beam 的連續 power，而是從一小組預先定義的 beam-power profile、active-beam budget、spread / concentrate 模式、或 time–power–frequency template 中擇一。這條路在樣本效率上通常比 fully continuous 好，而且與 multibeam / beam-hopping literature 裡的三維資源配置思維一致；但它抽象層更高，跟原始 MODQN 的距離也更大，因此更適合作為後續 method-design surface，而不是你現在的最小下一步。citeturn6search0turn9view0

**第四條路是維持 beam-only action，只重做 reward 或 replay。這條路我不建議當主路線。** 因為 repo 已經做過最接近這個方向的 Phase 03A/03B：reward normalization、r3 calibration、per-user beam-power EE credit 都進過場，但 one-beam collapse 仍然存在。這表示問題不是簡單的 reward scale mismatch，而是 action / state / denominator coupling 缺位。就第五題而言，**Catfish replay / intervention 單獨不太可能解這個 collapse**。Catfish 是 training strategy，不是 environment / action semantics；它也許可以改 exploration 或 sample efficiency，但不能憑空創造 `P_b(t)` 的可控性。repo 的 master plan 已明講：Phase 04 Catfish feasibility 可以作為 original MODQN reward branch 的獨立驗證，但不是 EE route 的 default next step；而 Phase 03B 也已明講，未來 EE route 需要的是 denominator-sensitive action/power design 或更強的 credit-assignment review，不是更多 episode，更不是直接把 Catfish 疊上去。fileciteturn11file0L1-L1 fileciteturn9file0L1-L1 fileciteturn12file0L1-L1

## Recommended next gate

我建議的下一個 gate 很明確：**暫停 EE-MODQN training promotion，先寫一份新的 power-coupled MDP / controller spec，通過 design review 後再做新的 bounded pilot。** 這其實就是第四題的答案：是，應先停，不要再把資源投入同一條 beam-only EE training 線上。repo 的 guardrails 與 master plan 本身就要求：baseline frozen、follow-on 用新 config / method family / artifact namespace，並且當 Phase 03 已證明「更多 episodes alone 不是 next gate」時，不能把持續訓練當成方法進展。fileciteturn8file0L1-L1 fileciteturn9file0L1-L1

這個設計 gate 至少要先把四件事情寫死。第一，**agent granularity**：你到底要做 per-user handover agent + frame-level power controller，還是 central controller；我建議前者，因為它最接近 HOBS 的分解。第二，**state contract**：不只要看 beam quality，還要顯式看 `beam_loads`、`active_beam_mask`、`beam_transmit_power_w`、剩餘 power budget、以及 QoS slack。第三，**action contract**：若你走 hierarchy，power controller 至少要能選 `ΔP_b` 或 codebook profile；若你走 joint discrete，則每個 handover action 都要帶 power token。第四，**evaluation contract**：主指標只能是 `EE_system = sum_i R_i / sum_active_beams P_b`；per-user EE credit 只能當 training signal 或 diagnostic，不能當 claim 主體。這四項沒有明寫好，就不應該重新進入 training。citeturn4view1turn4view2turn10view0turn10view1turn9view0 fileciteturn13file0L1-L1

對 Catfish 的定位也應一起寫清楚。我的建議是：**把 Catfish 暫時從 EE 修復路線中拿掉，僅保留為原始 MODQN objective 的獨立 Phase 04 feasibility branch。** 等到 power-coupled EE extension 已證明 `denominator_varies_in_eval=true`、且 `EE_system` 與 throughput ranking 已分離之後，再考慮 Catfish 對新 EE extension 是否有增益。否則 Catfish 的任何正向結果，都很容易只是對 throughput-like objective surface 的 optimization 改善，而不是對 energy-aware learning 的改善。fileciteturn9file0L1-L1 fileciteturn11file0L1-L1

## Minimal experiment

最小可辯護的下一個實驗，我不建議一開始就做「完整 joint RL」。我建議做一個**兩段式最小實驗**，先證明 denominator 真的可被控制，再證明學習是必要的。

第一段是 **power-sensitivity proof**。做法是先把 beam policy 固定住，用同一組 beam decisions 分別配上三種 power semantics：`fixed-2W baseline`、`Phase 02B active-load-concave proxy`、以及一個**明確可控的 power controller**。這個第三個 controller 可以非常小：例如每個 active beam 從 `{0.5, 1.0, 1.5, 2.0} W` 四個 level 中選一個，並遵守 per-satellite total power budget；或者仿照 HOBS 的 sign-flip power update 做一個 rule-based DPC。這一步的目標不是證明 RL，而是先看：**在 beam decisions 相同時，system EE ranking 會不會因 power control 而改變？活躍 beam 數 / total active power 會不會在 eval 中出現多樣性？throughput–EE correlation 會不會下降？** 如果連這一步都無法讓 EE 與 throughput 分離，那就不應該進入第二段。citeturn4view2turn9view0turn10view0 fileciteturn10file0L1-L1 fileciteturn11file0L1-L1

第二段才是 **bounded RL pilot**。若第一段通過，再用一個最小的 power-coupled agent 做 paired control vs candidate：兩邊用相同 seed set、相同 environment surface、相同 checkpoint rule、相同 eval cadence；差別只在 candidate 是否擁有 power action。若你想最小變動、又不離開 Q-learning family，我建議先做 **discrete joint action** 或 **hierarchical DQN + rule-based DPC**，不要馬上跳 continuous actor-critic。因為你眼前最需要回答的不是「哪種 RL 最強」，而是「一旦 action 對 power 有控制權，EE 是否不再退化成 throughput 的重標度？」citeturn10view0turn10view1turn9view0

這個最小實驗的觀測指標也要先固定。除了 scalar reward，你至少要記錄：`EE_system aggregate`、`EE_system step mean`、`active_beam_count_distribution`、`total_active_beam_power_w_distribution`、`denominator_varies_in_eval`、`all_evaluated_steps_one_active_beam`、`raw_throughput_mean`、`raw_throughput_low_p05`、`served_ratio`、`handover_count`、以及 throughput-vs-EE correlation 與 same-policy rescore ranking 是否改變。repo 現在的 phase03 validator 已經幫你把這些檢查項大致定義好了；下一個實驗不需要再發明新的成功條件，只要把它們移到新的 power-coupled design 上即可。fileciteturn15file0L1-L1

## Acceptance criteria

在任何新設計下，要 claim「EE-MODQN effective」之前，我認為至少要同時滿足以下 acceptance criteria，而且主體必須是 **system EE**，不是 per-user EE credit。

第一，**learned best-eval policy 必須真的讓 denominator 動起來。** 最低要求是 `denominator_varies_in_eval=true`，且 `all_evaluated_steps_one_active_beam=false`；更實際地說，你要看到 active beam count 與 total active beam power 在 evaluation seeds 上都不是單點分布。這是目前 Phase 03B 最明確沒過去的一條，也是所有後續 claim 的前置條件。fileciteturn11file0L1-L1 fileciteturn15file0L1-L1

第二，**`EE_system` 必須在 matched control 上有正向提升，而且不是以明顯 QoS 崩壞換來的。** repo 現有 gate 對 throughput / service / handover 已有 guardrail：low-p05 throughput 不應明顯惡化、served ratio 不應顯著下降、handover count 不應無理由暴增。若 EE 變好但 throughput tail、served ratio 或 handover 成本明顯變壞，那只代表你找到了一個差的能耗 tradeoff，不代表方法有效。fileciteturn15file0L1-L1

第三，**throughput 與 EE 的 ranking 必須產生可解釋的分離。** 也就是說，至少要滿足以下其中之一：throughput-vs-EE correlation 不再幾乎等於 1；或 same-policy rescoring 會改變方法排名；或在一些 seed / scenario 下，control 與 candidate 在 throughput 差不多時 EE_system 會分出高下。若排名永遠不變，那你頂多證明新 reward 是舊 reward 的 rescaling。fileciteturn11file0L1-L1 fileciteturn15file0L1-L1

第四，**成功依據不能只靠 scalar reward。** 這不只是你的限制，也是 repo guardrails 的硬規則。即便 candidate 在 scalarized reward 上贏，只要 `EE_system` 沒贏、或 denominator 沒變、或 per-user EE 增加但 system EE 沒變，就不能 promote。fileciteturn8file0L1-L1 fileciteturn15file0L1-L1

第五，**所有 power semantics 都必須有 provenance label。** 若仍沿用 Phase 02B power surface，claim 必須限制為「在 disclosed synthesized proxy 下成立」；若換成新的 power-control MDP 或 controller，則必須標為 **new extension / method-design surface**。只有當設計真的被 HOBS 或 MODQN 原論文明確支持時，才可以把它說成 paper-backed；目前這一點尚未成立。fileciteturn10file0L1-L1 fileciteturn16file0L1-L1

## Stop conditions and forbidden claims

接下來我會把 stop conditions 與 forbidden claims 一起寫，因為它們本質上是在回答「什麼情況下必須停、哪些話現在絕對不能說」。

應當立刻 stop 的情況有幾個。**只要新設計跑完後，best-eval 仍然是 `denominator_varies_in_eval=false`、仍然所有 evaluated steps 只有一個 active beam、或 throughput-vs-EE correlation 仍然近乎 1、或排名在 throughput-rescore / EE-rescore 下完全不變，就應停止 EE effectiveness 敘事，而不是再加 episode。** 同樣地，若看見的只是 scalar reward 上升、per-user EE credit 上升、或 Catfish 讓 sample efficiency 變好，但 `EE_system` 沒改善，那也應停。這些都不是 energy-aware learning 的證據。fileciteturn11file0L1-L1 fileciteturn12file0L1-L1 fileciteturn15file0L1-L1

目前仍必須禁止的 claims，我建議你明文列成專案 guardrail。**不能 claim full paper-faithful reproduction；不能把 per-user EE credit 當成 system EE；不能用 scalar reward alone 當成功依據；不能把 Phase 02B synthetic `P_b(t)` 說成 HOBS paper-backed optimizer；不能 claim current Phase 03 / 03A / 03B 已證明 EE-MODQN effective；也不能 claim Catfish replay / intervention 單獨解掉了 EE collapse。** 若你接下來採用任何顯式 power-control MDP、hierarchical controller、joint discrete action、或 profile codebook，都應標成 **new extension / method-design surface**，除非未來能從原始 HOBS / MODQN text 找到更直接的支持。fileciteturn8file0L1-L1 fileciteturn9file0L1-L1 fileciteturn10file0L1-L1 fileciteturn17file0L1-L1

最後補一個必要的限制說明：**我對 HOBS 的 EE formula、`P_b(t)` 與 dynamic power control 有直接文本依據；但對 MODQN 論文的 state / action / reward 細節，我能高信心確認的是 abstract-level 的三目標定義，而更細的 MDP surface 主要依賴 repo 內的 paper-linked implementation / SDD mapping。** 這不影響本報告的主結論，因為你眼前要解的不是「原 MODQN 到底有沒有 power action」，而是「目前 repo 的 EE route 是否已足夠可辯護」；答案仍然是否定的，且下一步最合理的是先做顯式 power-coupled redesign。citeturn5search0turn5search1 fileciteturn13file0L1-L1
