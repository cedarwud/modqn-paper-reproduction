# Prompt: Phase 01 MODQN Baseline Anchor Review

請根據我提供的 MODQN repo 文件、artifact status notes、`dqn-development-report.md`、`deep-research-report.md`，只檢查 **Phase 01: MODQN Baseline Anchor**。

目標：

判斷原始 MODQN baseline 是否可以作為後續 Catfish / EE-MODQN / Catfish-EE-MODQN 的比較基準。

請不要討論：

1. EE 公式設計
2. Catfish 訓練策略
3. multi-catfish
4. final method

請輸出繁體中文報告，包含：

1. 原始 MODQN baseline 的定義：
   - `r1 = throughput`
   - `r2 = handover penalty`
   - `r3 = load balance`
2. 可作為 baseline anchor 的 config / artifact / status notes。
3. 必須保留的比較 metrics：
   - scalar reward
   - r1/r2/r3
   - handover count
   - best-eval checkpoint
4. 已知限制：
   - near-tie
   - late-training collapse
   - reward dominance
   - disclosed comparison-baseline claim boundary
5. 建議後續 phase 使用哪個 baseline comparison surface。
6. 可以 claim 什麼，不能 claim 什麼。

最後請給出：

```text
PROMOTE / BLOCK / NEEDS MORE EVIDENCE
```

並說明理由。

