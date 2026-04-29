# Prompt: RA-EE-09 Completion Design Gate

請你接手 `modqn-paper-reproduction` 的 RA-EE continuation 工作。

第一輪只做 **RA-EE-09 design plan**，不要寫程式、不要改 trainer、不要跑訓練、
不要產生 artifacts。

請先讀：

```text
AGENTS.md
docs/research/catfish-ee-modqn/development-guardrails.md
docs/research/catfish-ee-modqn/00-validation-master-plan.md
docs/research/catfish-ee-modqn/execution-handoff.md
docs/research/catfish-ee-modqn/03d-ee-route-disposition.execution-report.md
docs/research/catfish-ee-modqn/ra-ee-02-oracle-power-allocation-audit.execution-report.md
docs/research/catfish-ee-modqn/ra-ee-04-bounded-power-allocator-pilot.execution-report.md
docs/research/catfish-ee-modqn/ra-ee-05-fixed-association-robustness.execution-report.md
docs/research/catfish-ee-modqn/ra-ee-06-association-counterfactual-oracle.execution-report.md
docs/research/catfish-ee-modqn/ra-ee-06b-association-proposal-refinement.execution-report.md
docs/research/catfish-ee-modqn/ra-ee-07-constrained-power-allocator-distillation.execution-report.md
docs/research/catfish-ee-modqn/ra-ee-08-offline-association-reevaluation.execution-report.md
docs/research/catfish-ee-modqn/ra-ee-09-completion-design-gate.md
docs/ee-report.md
```

請沿用目前守門結論：

```text
old EE-MODQN r1-substitution route: BLOCKED / STOP
RA-EE fixed-association deployable power allocation: PASS, scoped
RA-EE learned association / hierarchical RL / full RA-EE-MODQN: BLOCKED
Catfish for EE repair: BLOCKED
```

## Goal

請產出 RA-EE-09 的具體 design plan：

```text
RA-EE-09 fixed-association RB / bandwidth allocation design gate
```

核心問題：

在 association 固定、且使用 RA-EE-07 deployable stronger power allocator 作為
matched control boundary 的前提下，是否值得加入 bounded RB / bandwidth
allocator，來測試 power allocation 之外的 resource-allocation 增益？

## Required Boundaries

必須保持：

1. association fixed,
2. no learned association,
3. no hierarchical RL,
4. no joint association + power training,
5. no Catfish,
6. no Phase 03C continuation,
7. no tuning RA-EE-06 / 06B / 08 association proposal rules,
8. no frozen baseline mutation,
9. no HOBS optimizer claim,
10. no physical energy saving claim,
11. no full RA-EE-MODQN claim.

## Design Questions To Answer

請明確回答：

1. RB / bandwidth resource unit 要怎麼定義？
   - RB count
   - bandwidth share
   - normalized resource fraction
   - 或其他 bounded equivalent
2. allocator granularity 是 per-user、per-beam、還是 centralized？
3. RB / bandwidth allocation 如何進入 throughput `R_i(t)`？
4. 它如何和 RA-EE-07 deployable power allocator 共存？
5. inactive beam 的 resource 行為是什麼？
6. total RB / bandwidth budget、per-beam cap、per-user min/max 是否需要？
7. matched control 是什麼？
8. candidate 是什麼？
9. artifact namespace 與 metadata schema 是什麼？
10. focused tests 應該有哪些？
11. acceptance criteria 和 stop conditions 是什麼？

## Required Comparator

Comparator 必須是 matched comparison：

```text
control:
  fixed association
  + RA-EE-07 deployable stronger power allocator
  + fixed/equal/default RB or bandwidth allocation

candidate:
  fixed association
  + same RA-EE-07 deployable stronger power allocator
  + bounded RB / bandwidth allocator
```

兩者必須使用相同 seeds、held-out bucket、association trajectories、power
contract、QoS guardrails、budget checks、evaluation schedule、artifact schema。

## Required Metrics

至少包含：

1. simulated `EE_system`,
2. throughput mean and p05,
3. served ratio,
4. outage ratio,
5. handover count,
6. load-balance metrics,
7. active beam count,
8. total active power,
9. RB / bandwidth usage distribution,
10. RB / bandwidth budget violations,
11. per-beam and per-user resource violations,
12. gain concentration by seed and trajectory,
13. scalar reward only as diagnostic.

## Required Tests

設計 plan 必須列出 focused tests，至少涵蓋：

1. baseline unchanged when RA-EE-09 is disabled,
2. config namespace gating,
3. fixed-association enforcement,
4. same deployable power allocator used by control and candidate,
5. RB / bandwidth accounting,
6. throughput recomputation path,
7. QoS guardrail computation,
8. budget violation reporting,
9. artifact metadata and review output,
10. no learned association / no Catfish / no EE-MODQN r1-substitution mode.

## Acceptance Criteria

RA-EE-09 只有在 bounded fixed-association pilot 顯示以下條件時才能 PASS：

1. candidate 對 matched control 有 positive held-out simulated `EE_system`
   delta，或有明確合理的 resource-efficiency metric delta,
2. p05 throughput guardrail pass,
3. served ratio 不下降,
4. outage ratio 不上升,
5. handover and load-balance regression bounded,
6. power and RB / bandwidth budgets zero violations,
7. gains not concentrated in one seed or one trajectory,
8. scalar reward is not used as success basis,
9. metadata proves the same association and same power comparison boundary.

## Stop Conditions

請明確列入：

1. 需要改 frozen baseline semantics,
2. result depends on learned association,
3. candidate only wins by reducing service quality,
4. RB / bandwidth accounting cannot be audited,
5. gains appear only in scalar reward,
6. gains concentrate in one seed or one trajectory,
7. work starts being framed as Catfish, HOBS optimizer, physical energy saving,
   or full RA-EE-MODQN.

## Forbidden Claims

不可 claim：

1. full RA-EE-MODQN,
2. learned association effectiveness,
3. old EE-MODQN effectiveness,
4. HOBS optimizer behavior,
5. physical energy saving,
6. Catfish-EE or Catfish repair,
7. RB / bandwidth allocation effectiveness before RA-EE-09 evidence exists,
8. full paper-faithful reproduction.

## Output Format

請輸出繁體中文，格式如下：

```text
Current State
RA-EE-09 Goal
Design Boundary
RB / Bandwidth Contract
Control / Candidate
Metrics
Artifact / Metadata Plan
Focused Tests
Acceptance Criteria
Stop Conditions
Forbidden Claims
Implementation Slices
Questions / Assumptions
PASS / FAIL / NEEDS MORE DESIGN
```

最後的 verdict 應該是設計 gate 狀態，不是 method effectiveness claim。
