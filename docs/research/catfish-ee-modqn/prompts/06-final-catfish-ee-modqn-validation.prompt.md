# Prompt: Phase 06 Final Catfish-EE-MODQN Review

請根據 Phase 02 EE formula、Phase 03 EE-MODQN、Phase 04 Catfish-MODQN、Phase 05 multi-catfish 設計，只檢查 **Phase 06: Final Catfish-EE-MODQN Validation**。

目標：

判斷最終方法應如何公平驗證，避免把 EE objective、Catfish training、multi-catfish 設計混成不可解釋的單一結果。

主要比較至少包含：

```text
EE-MODQN
vs
Catfish-EE-MODQN
```

如果 multi-catfish 是主張的一部分，還要比較：

```text
EE-MODQN
vs
Single-Catfish-EE-MODQN
vs
Multi-Catfish-EE-MODQN
```

請檢查：

1. 為什麼 final method 不能只和 original MODQN 比？
2. 哪些 metrics 必須報告？
   - EE_system
   - per-user EE
   - raw throughput
   - total beam power
   - handover count
   - load-balance gap
   - service outage / below-threshold ratio
   - scalar reward
   - convergence speed
   - best-eval checkpoint
   - replay composition
   - intervention count
   - stability / collapse indicators
3. 什麼結果算 Catfish 對 EE-MODQN 有額外提升？
4. 什麼結果只是 reward scaling 或 service reduction？
5. final method 可以 claim 什麼？
6. final method 不能 claim 什麼？
7. 需要哪些 ablation 才能支持 multi-catfish claim？

請輸出繁體中文報告。

最後請給出：

```text
PROMOTE / BLOCK / NEEDS MORE EVIDENCE
```

並說明理由。

