# Prompt: Phase 04 Single Catfish-MODQN Feasibility Review

請根據 Catfish 原始文件、Catfish notes、`catfish-report.md`、MODQN baseline 文件，只檢查 **Phase 04: Single Catfish-MODQN Feasibility**。

前置 closeout 結論：

```text
old EE-MODQN r1-substitution route: BLOCKED / STOP
RA-EE fixed-association deployable power allocation: PASS, scoped
RA-EE learned association / hierarchical RL / full RA-EE-MODQN: BLOCKED
Catfish for EE repair: BLOCKED
```

請直接沿用上述結論，不要重新打開 Phase 03C 或 RA-EE association proposal route。

目標：

判斷 Catfish-style dual-agent replay/intervention training 是否能在不改 reward 的情況下接到原始 MODQN backbone。

這是 original-MODQN reward 的獨立 feasibility branch，不是 EE repair，也不是
RA-EE continuation。

請不要討論：

1. EE objective
2. EE formula
3. multi-catfish
4. final Catfish-EE method
5. Phase 03C rerun / selector tweak / reward retuning
6. RA-EE learned association / hierarchical RL / association proposal refinement
7. HOBS optimizer or physical energy saving claim

保持原始 MODQN objective：

```text
r1 = throughput
r2 = handover penalty
r3 = load balance
```

Catfish-MODQN 只新增：

1. main MODQN agent
2. catfish MODQN agent
3. main replay
4. high-value catfish replay
5. asymmetric discount factors
6. optional competitive reward shaping, ablation only
7. periodic mixed replay intervention

Primary feasibility run must keep competitive reward shaping off. If shaping is
considered, isolate it as an ablation after the shaping-off path is evaluated.

第一階段 high-value criterion 可使用：

```text
quality = 0.5*r1 + 0.3*r2 + 0.2*r3
```

請檢查：

1. 這樣是否能合理驗證 Catfish feasibility？
2. 它能 claim 什麼？
3. 它不能 claim 什麼？
4. 需要哪些 metrics？
   - scalar reward
   - r1/r2/r3
   - handover count
   - convergence speed
   - best-eval checkpoint
   - replay composition
   - intervention count
   - stability / collapse indicators
5. competitive reward shaping 是否應先關掉或當 ablation？
6. 什麼結果才值得進入 multi-catfish 或 EE phase？
7. 如何確保任何正向結果都不是 EE / RA-EE claim？

Allowed claims:

1. Catfish replay / intervention 是否能工程上接到 MODQN。
2. dual-agent / dual-replay / asymmetric gamma / mixed replay 是否能穩定訓練。
3. high-value replay 是否真的收到樣本並進入 main-agent update。
4. 在相同 original MODQN objective 下，Catfish-MODQN 對 scalar reward、`r1` /
   `r2` / `r3`、handover count、convergence speed、best-eval checkpoint、
   stability / collapse indicators 的影響。

Forbidden claims:

1. 不 claim full RA-EE-MODQN。
2. 不 claim learned association effectiveness。
3. 不 claim old EE-MODQN effectiveness。
4. 不 claim HOBS optimizer or physical energy saving。
5. 不把 Catfish 當 EE repair。
6. 不用 scalar reward alone 宣稱 Catfish-MODQN 更好。
7. 不宣稱 Catfish-EE-MODQN 或 Multi-Catfish-MODQN effectiveness。

請輸出繁體中文報告。

最後請給出：

```text
PROMOTE / BLOCK / NEEDS MORE EVIDENCE
```

並說明理由。
