# Prompt: Phase 04 Single Catfish-MODQN Feasibility Review

請根據 Catfish 原始文件、Catfish notes、`catfish-report.md`、MODQN baseline 文件，只檢查 **Phase 04: Single Catfish-MODQN Feasibility**。

目標：

判斷 Catfish-style dual-agent replay/intervention training 是否能在不改 reward 的情況下接到原始 MODQN backbone。

請不要討論：

1. EE objective
2. EE formula
3. multi-catfish
4. final Catfish-EE method

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
6. optional competitive reward shaping
7. periodic mixed replay intervention

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

請輸出繁體中文報告。

最後請給出：

```text
PROMOTE / BLOCK / NEEDS MORE EVIDENCE
```

並說明理由。

