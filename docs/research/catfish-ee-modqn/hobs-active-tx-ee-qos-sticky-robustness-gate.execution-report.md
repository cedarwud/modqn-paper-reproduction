# HOBS Active-TX EE QoS-Sticky Robustness Gate Execution Report

**Date:** `2026-05-01`
**Status:** `PASS`
**Scope:** bounded robustness / mechanism-attribution gate only. This
report does not authorize EE-MODQN effectiveness, physical energy
saving, HOBS optimizer, Catfish repair, RA-EE association, Phase `03C`
continuation, or frozen baseline mutation.

## Protocol

- Roles: `matched-control`, `primary-qos-sticky`, `no-qos-guard-ablation`, `stricter-qos-ablation`, `threshold-sensitivity-45`, `threshold-sensitivity-55`.
- Seed triplets: `[42, 1337, 7]`, `[43, 1338, 8]`, `[44, 1339, 9]`.
- Evaluation seeds: `[100, 200, 300, 400, 500]`.
- Episode budget: `5` per role / seed triplet.
- Scalar reward is diagnostic only.

## Matched Boundary Proof

`matched_boundary_pass=True`.
All roles use HOBS active-TX EE `r1`, the same HOBS-inspired DPC sidecar,
the same training seed triplets, the same evaluation seeds, and the
same bounded trainer / checkpoint protocol. Only the opt-in
anti-collapse role or ablation knob differs.

## Primary Aggregate Metrics

- `all_evaluated_steps_one_active_beam`: `False`
- `active_beam_count_distribution`: `{'7.0': 150}`
- `denominator_varies_in_eval`: `True`
- `active_power_single_point_distribution`: `False`
- `distinct_total_active_power_w_values`: `[6.8, 6.9, 7.0, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9]`
- `power_control_activity_rate`: `1.0`
- `throughput_vs_ee_pearson`: `0.5022146200296446`
- `same_policy_throughput_vs_ee_rescore_ranking_change`: `True`
- `raw_throughput_mean_bps`: `7048.5868065198265`
- `p05_throughput_bps`: `13.08817702929179`
- `p05_throughput_ratio_vs_control`: `3.105617320531727`
- `served_ratio_delta`: `0.0`
- `outage_ratio_delta`: `0.0`
- `handover_delta`: `-292.00000000000006`
- `r2_delta`: `0.29200000000000004`
- `budget/per-beam/inactive-power violations`: `0/0/0`
- `overflow/sticky/nonsticky/qos-reject/handover-reject`: `150/6269/0/0/1231`

## Aggregate Pass / Fail

| Role | Status | p05 ratio | handover delta | r2 delta | one-beam seeds |
|---|---:|---:|---:|---:|---:|
| `matched-control` | `REFERENCE` | `None` | `None` | `None` | `3` |
| `primary-qos-sticky` | `PASS` | `3.105617320531727` | `-292.00000000000006` | `0.29200000000000004` | `0` |
| `no-qos-guard-ablation` | `PASS` | `3.105617320531727` | `-292.00000000000006` | `0.29200000000000004` | `0` |
| `stricter-qos-ablation` | `PASS` | `3.105617320531727` | `-292.00000000000006` | `0.29200000000000004` | `0` |
| `threshold-sensitivity-45` | `PASS` | `3.226914506326887` | `-323.33333333333337` | `0.3233333333333334` | `0` |
| `threshold-sensitivity-55` | `PASS` | `2.860440310506819` | `-262.6666666666667` | `0.2626666666666667` | `0` |

## Mechanism Attribution

- Relaxed-QoS ablation also passes; QoS guard may not be the active mechanism on this bounded surface.
- Stricter-QoS preserves pass with lower or equal sticky intervention.
- Both nearby threshold sensitivity arms preserve aggregate pass.

## Acceptance Result

`PASS / BLOCK / NEEDS MORE DESIGN: PASS`

## Artifacts

- `artifacts/hobs-active-tx-ee-qos-sticky-robustness-*/`
- `artifacts/hobs-active-tx-ee-qos-sticky-robustness-summary/summary.json`
- `artifacts/hobs-active-tx-ee-qos-sticky-robustness-summary/per_seed_pass_fail_table.csv`
- `artifacts/hobs-active-tx-ee-qos-sticky-robustness-summary/aggregate_pass_fail_table.csv`

## Forbidden Claims Still Active

- EE-MODQN effectiveness
- physical energy saving
- HOBS optimizer reproduction
- full RA-EE-MODQN
- learned association effectiveness
- Catfish / Multi-Catfish / Catfish-EE repair
- Phase 03C selector or reward tuning
- scalar reward as success evidence

## Deviations / Blockers

- None within the bounded robustness protocol.
