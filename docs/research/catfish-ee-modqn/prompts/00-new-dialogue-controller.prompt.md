# Prompt: New Dialogue Controller For Catfish / EE-MODQN

請你接手成為 `modqn-paper-reproduction` 專案中 Catfish / EE-MODQN 研究計畫的總控 agent。

請先不要寫程式、不要改文件、不要開始訓練。你的第一個任務是讀取文件並確認目前計畫狀態，然後只針對 Phase 02 execution 做可執行計畫。

請先讀這些文件，順序如下：

```text
AGENTS.md
docs/research/catfish-ee-modqn/00-validation-master-plan.md
docs/research/catfish-ee-modqn/execution-handoff.md
docs/research/catfish-ee-modqn/development-guardrails.md
docs/research/catfish-ee-modqn/reviews/01-modqn-baseline-anchor.review.md
docs/research/catfish-ee-modqn/reviews/02-hobs-ee-formula-validation.review.md
docs/research/catfish-ee-modqn/02-hobs-ee-formula-validation.md
```

讀完後請先確認以下狀態：

```text
Phase 01: completed as disclosed comparison baseline
Phase 02: next execution target
Phase 03: blocked until Phase 02 EE metric evidence exists
Phase 04: separate single-Catfish feasibility branch, not part of Phase 02
Phase 05: blocked; only 05A multi-buffer may be considered after Phase 04 evidence
Phase 06: blocked until Phases 03-05 produce evidence
```

Phase 02 的目標不是訓練 EE-MODQN，而是先驗證或實作 EE metric / formula surface：

```text
EE_system(t) = sum_i R_i(t) / sum_active_beams P_b(t)
```

請特別檢查：

1. `R_i(t)` throughput 目前在哪裡計算或輸出？
2. `P_b(t)` beam transmit power 目前是否存在？
3. `P_b(t)` 是否是 linear W？
4. `P_b(t)` 是否會隨 action、active beam state 或 power allocation 改變？
5. `active_beams` 是否能明確辨識？
6. 如果 denominator 是固定常數，請明確指出 Phase 02 blocked，因為 EE 會退化成 throughput scaling。

請不要做：

1. 不要直接開始 Phase 03 EE-MODQN training。
2. 不要開始 Catfish-MODQN。
3. 不要開始 multi-catfish。
4. 不要把 original MODQN 當成 final Catfish-EE-MODQN 的唯一比較基準。
5. 不要 claim full paper-faithful reproduction。
6. 不要改 baseline config 來承載 EE 或 Catfish。
7. 不要把 follow-on artifact 寫進 frozen baseline artifact 目錄。

請輸出繁體中文，格式如下：

```text
Current State
Phase 02 Scope
Files To Inspect
Execution Plan
Expected Deliverables
Stop Conditions
Questions / Assumptions
```

請保持總控角色：先把 Phase 02 execution plan 鎖清楚，再等使用者確認是否進入實作。
