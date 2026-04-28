# Prompt: Phase 02 HOBS-Linked EE Formula Review

請根據我提供的 HOBS 公式文件、`system-model-refs`、`paper-catalog` power/EE 分類報告，只檢查 **Phase 02: HOBS-Linked EE Formula Validation**。

目標：

判斷 proposed EE formula 是否和 HOBS SINR / power semantics 一致。

請不要討論：

1. Catfish
2. multi-catfish
3. training implementation
4. final method comparison

請重點檢查：

1. HOBS 中 `P_{n,m}(t)` 是否同時進入：
   - SINR numerator
   - EE denominator
2. `P_{n,m}(t)` 是否應被視為 downlink per-beam transmit-power control / allocation variable。
3. 是否不應把 `P_{n,m}(t)` 寫成 path-loss closed-form。
4. system-level EE 是否合理：

```text
EE_system(t) = sum_i R_i(t) / sum_active_beams P_b(t)
```

5. per-user MODQN reward adaptation 是否合理：

```text
r1_i(t) = R_i(t) / (P_{b_i}(t) / N_{b_i}(t))
```

6. per-user form 是否必須標成 modeling / credit-assignment assumption。
7. throughput 是否應作為 QoS guardrail / reporting metric，而不是第四個 objective。
8. 是否存在 `EE = throughput / fixed_power` 的退化風險。

請輸出繁體中文報告，包含：

1. formula recommendation
2. provenance classification
3. assumptions
4. risks
5. guardrails
6. 是否可以進入 EE-MODQN phase

最後請給出：

```text
PROMOTE / BLOCK / NEEDS MORE EVIDENCE
```

並說明理由。

