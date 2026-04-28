# Prompt: Phase 03 EE-MODQN Review

請根據 MODQN baseline 文件、Phase 02 EE formula 結論，以及目前 HOBS-linked EE 設計，只檢查 **Phase 03: EE-MODQN Validation**。

目標：

判斷只把 MODQN 的第一個 objective 從 throughput 換成 EE 後，應該怎麼驗證其效果。

請不要討論：

1. Catfish
2. single Catfish
3. multi-catfish
4. final Catfish-EE method

請檢查：

原始 MODQN：

```text
R = (throughput, handover penalty, load balance)
```

EE-MODQN：

```text
R = (energy efficiency, handover penalty, load balance)
```

請回答：

1. EE-MODQN 和 original MODQN 應該怎麼公平比較？
2. 哪些 metrics 必須同時報告？
   - EE_system
   - per-user EE
   - raw throughput
   - total beam power
   - handover count
   - load-balance gap
   - outage / below-threshold ratio
   - scalar reward
   - best-eval checkpoint
3. 如何檢查 EE 是否只是 throughput rescaling？
4. 如何檢查是否出現 reward hacking？
5. throughput guardrail 應該怎麼放在 claim 裡？
6. EE-MODQN 可以 claim 什麼，不能 claim 什麼？

請輸出繁體中文報告。

最後請給出：

```text
PROMOTE / BLOCK / NEEDS MORE EVIDENCE
```

並說明理由。

