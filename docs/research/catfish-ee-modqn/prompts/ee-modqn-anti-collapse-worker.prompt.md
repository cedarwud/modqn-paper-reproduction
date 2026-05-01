# Prompt: EE-MODQN Anti-Collapse Worker Gate

你是 `modqn-paper-reproduction` 的執行 agent。請只執行本 prompt 指定的
bounded anti-collapse gate。不要重新規劃整條研究線。

請先讀：

```text
AGENTS.md
docs/research/catfish-ee-modqn/00-validation-master-plan.md
docs/research/catfish-ee-modqn/execution-handoff.md
docs/research/catfish-ee-modqn/development-guardrails.md
docs/research/catfish-ee-modqn/ee-modqn-anti-collapse-controller-plan-2026-05-01.md
docs/research/catfish-ee-modqn/hobs-active-tx-ee-modqn-feasibility.execution-report.md
docs/research/catfish-ee-modqn/energy-efficient/README.md
docs/ee-report.md
```

任務：

```text
設計並實作一個最小 opt-in anti-collapse / capacity / assignment gate，
用來測試 learned greedy eval 是否能擺脫 all_evaluated_steps_one_active_beam=true。
```

嚴格限制：

1. 不要跑長訓練；若需要 pilot，必須是 tiny bounded pilot。
2. 不要引入 Catfish / Multi-Catfish。
3. 不要重新打開 Phase 03C selector / reward tuning。
4. 不要重新打開 RA-EE association proposal route。
5. 不要改 frozen baseline configs / artifacts / semantics。
6. 不要把 scalar reward 當成功依據。
7. 不要 claim EE-MODQN effectiveness；這只是 anti-collapse gate。

設計要求：

1. 新 config namespace，例如 `hobs-active-tx-ee-anti-collapse-*`。
2. 新 artifact namespace，例如 `artifacts/hobs-active-tx-ee-anti-collapse-*`。
3. Matched control / candidate boundary：
   - same seeds,
   - same episode budget,
   - same evaluation schedule,
   - same checkpoint protocol,
   - same HOBS active-TX EE formula,
   - same DPC sidecar parameters unless explicitly justified,
   - only intended anti-collapse mechanism differs.
4. Candidate mechanism 必須是最小且可稽核，優先考慮：
   - capacity-aware action masking,
   - overload penalty,
   - active-beam diversity / load-spread constraint,
   - centralized assignment constraint.

必須報告 metrics：

```text
all_evaluated_steps_one_active_beam
active_beam_count_distribution
denominator_varies_in_eval
active_power_single_point_distribution
distinct_total_active_power_w_values
power_control_activity_rate
throughput_vs_ee_pearson
same_policy_throughput_vs_ee_rescore_ranking_change
raw_throughput_mean_bps
p05_throughput_bps
served_ratio
outage_ratio
handover_count
r2 / handover regression vs matched control
load-balance metric
scalar reward diagnostic
budget / power / inactive-beam violations
```

Acceptance criteria：

1. `all_evaluated_steps_one_active_beam = false`。
2. Active-beam distribution 非退化，或符合預先宣告的 multi-beam target。
3. `denominator_varies_in_eval = true`。
4. `active_power_single_point_distribution = false`。
5. p05 throughput 不可 collapse relative to matched control。
6. served ratio 不下降。
7. outage ratio 不上升。
8. handover / `r2` regression 在預先宣告 tolerance 內。
9. 無 power / inactive-beam / budget accounting violation。
10. scalar reward 不作成功依據。

Stop conditions：

1. candidate 仍 `all_evaluated_steps_one_active_beam=true`。
2. anti-collapse 靠嚴重 p05 throughput collapse 或 service collapse 達成。
3. gains 只出現在 scalar reward。
4. matched boundary 無法證明。
5. 需要 Catfish、RA-EE association、Phase 03C continuation 或 frozen baseline mutation。

請輸出：

```text
Changed Files
What Was Implemented
Matched Boundary Proof
Metrics
Acceptance Result
Tests / Checks Run
Artifacts
Forbidden Claims Still Active
Deviations / Blockers
PASS / BLOCK / NEEDS MORE DESIGN
```

若 gate BLOCK，請不要自行調參續跑；直接回報總控。
