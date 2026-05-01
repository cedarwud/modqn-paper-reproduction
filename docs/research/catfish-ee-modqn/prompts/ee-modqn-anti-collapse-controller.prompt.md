# Prompt: EE-MODQN Anti-Collapse Controller

請你接手成為 `modqn-paper-reproduction` 專案的 EE-MODQN anti-collapse
研究總控 agent。

重要：你是總控，不是執行 agent。第一輪不要寫程式、不要改文件、不要跑訓練。
你的工作是讀取狀態、鎖定下一個 bounded gate、產出給另一個新對話使用的
execution prompt。後續執行 agent 回報後，你再負責判定 PASS / BLOCK /
NEEDS MORE DESIGN 並同步文件。

請先讀：

```text
AGENTS.md
docs/research/catfish-ee-modqn/00-validation-master-plan.md
docs/research/catfish-ee-modqn/execution-handoff.md
docs/research/catfish-ee-modqn/development-guardrails.md
docs/research/catfish-ee-modqn/ee-modqn-anti-collapse-controller-plan-2026-05-01.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-modqn-feasibility.execution-report.md
docs/research/catfish-ee-modqn/repository-curation-2026-05-01.md
docs/research/catfish-ee-modqn/energy-efficient/README.md
docs/research/catfish-ee-modqn/energy-efficient/ee-formula-final-review-with-codex-2026-05-01.md
docs/research/catfish-ee-modqn/energy-efficient/modqn-r1-to-hobs-active-tx-ee-design-2026-05-01.md
docs/ee-report.md
```

目前必須沿用的狀態：

```text
HOBS active-TX EE formula / reward wiring: PASS, scoped
HOBS-inspired DPC sidecar denominator gate: PASS
Route D tiny learned-policy denominator check: BLOCK
current hard blocker: all_evaluated_steps_one_active_beam = true
EE-MODQN effectiveness: NOT PROMOTED / BLOCKED
Catfish as EE repair: BLOCKED
RA-EE learned association / full RA-EE-MODQN: BLOCKED
RA-EE-07 fixed-association power allocator: PASS, scoped, separate evidence
```

第一輪請只輸出：

```text
Current State
Controller Role
Next Gate Decision
Execution Prompt For Worker
Acceptance Criteria
Stop Conditions
Forbidden Claims
Questions / Assumptions
PASS / BLOCK / NEEDS MORE DESIGN
```

下一個 gate 的預設方向：

```text
EE-MODQN anti-collapse / capacity / assignment design gate
```

這個 gate 的目標不是證明完整 EE-MODQN，而是先回答：

```text
能否用最小、可稽核、opt-in 的 anti-collapse 機制，讓 learned greedy eval
不再出現 all_evaluated_steps_one_active_beam=true，同時保留 HOBS active-TX EE
與 DPC denominator boundary？
```

你可以考慮的候選機制：

1. capacity-aware action masking,
2. overload penalty,
3. active-beam diversity / load-spread constraint,
4. centralized assignment constraint,
5. renamed resource-allocation MDP with explicit resource actions.

限制：

1. 不要要求執行 agent 加長 Route D training。
2. 不要要求執行 agent 引入 Catfish 或 Multi-Catfish。
3. 不要重新打開 Phase 03C selector / reward tuning。
4. 不要重新打開 RA-EE association route。
5. 不要改 frozen baseline。
6. 不要用 scalar reward alone 作成功依據。
7. 不要 claim physical energy saving、HOBS optimizer、full RA-EE-MODQN、learned association effectiveness。

你要產出的 worker prompt 必須要求 worker：

1. 先做設計 gate，不要直接大訓練。
2. 使用新 config / artifact namespace。
3. 保持 matched control / candidate boundary。
4. 報告 `all_evaluated_steps_one_active_beam`、active-beam distribution、
   `denominator_varies_in_eval`、`active_power_single_point_distribution`、
   throughput-vs-EE correlation / ranking separation、p05 throughput、served
   ratio、outage、handover / `r2`、scalar reward diagnostic。
5. 若 candidate 仍 one-active-beam collapse，立即 BLOCK，不要調參續跑。

請保持繁體中文。不要執行 worker prompt；只產出給下一個新對話使用的 prompt。
