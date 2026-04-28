# Phase 02 Execution Report: EE Metric Surface and Denominator Audit

**Date:** `2026-04-28`  
**Status:** `BLOCKED after audit`  
**Scope:** EE metric / formula surface and denominator audit only. No EE-MODQN
training, reward replacement, Catfish, or multi-Catfish work was performed.

## Execution Decision

Phase 02 produced the required runtime audit evidence, but the current runtime
does not expose a defensible HOBS `P_b(t)`.

`EE_system(t) = sum_i R_i(t) / sum_active_beams P_b(t)` is therefore not
defensibly computable from the current runtime as a HOBS-linked EE metric.
The only available denominator is a fixed-power active-beam-count proxy:

```text
sum_active_beams tx_power_w
```

This is a fixed-power-trap diagnostic, not a valid allocated-power denominator.

## Runtime Mapping

### `R_i(t)` throughput

`R_i(t)` is computed in `StepEnvironment._build_states_and_masks` as:

```text
R_i(t) = B / N_b * log2(1 + gamma)
```

The single-truth throughput vector is `user_throughputs`; it is copied into
`RewardComponents.r1_throughput` by `_compute_rewards`.

Current outputs:

1. `StepResult.rewards[*].r1_throughput`
2. `StepResult.beam_throughputs`
3. trainer `EpisodeLog.r1_mean` / `training_log.json`
4. replay bundle `rewardVector.r1Throughput` and `beamThroughputs`

### `P_b(t)` beam transmit power

The only beam-power surface found is `ChannelConfig.tx_power_w`.

It is used in `compute_channel` as:

```text
rx_power_w = tx_power_w * channel_gain
snr_linear = rx_power_w / noise_power_w
```

This means transmit power exists for the channel/SINR path, but only as a
static scalar config value.

### Unit audit

`tx_power_w` is linear watts. Evidence:

1. the field name is `tx_power_w`
2. default config value is `2.0`
3. `compute_channel` multiplies it directly into `rx_power_w`
4. dB conversions are separate helpers and are not used for `tx_power_w`

No mixed dBm/dBW transmit-power path was found for `P_b`.

### Power variability audit

Artifact evidence:

1. `artifacts/ee-denominator-audit-phase-02/ee_denominator_summary.json`
2. `artifacts/ee-denominator-audit-phase-02/ee_denominator_audit.csv`
3. `artifacts/ee-denominator-audit-phase-02/review.md`

Audit summary:

```text
rows audited: 30
policies: hold-current, random-valid, first-valid
tx_power_w distinct values: [2.0]
active_beam_count distinct values: [1, 7]
total fixed-power denominator distinct values: [2.0, 14.0]
denominator classification: fixed-power-active-beam-count-proxy
throughput sum max abs delta: 0.0 bps
```

The total denominator can vary only because active beam count changes. Per-beam
`P_b(t)` itself does not vary with action, active-beam state, or power
allocation.

### `active_beams`

`active_beams` is not a named runtime field, but it is explicitly derivable
after a step:

```text
active_beams = flatnonzero(StepResult.user_states[0].beam_loads > 0)
```

This is clear enough for diagnostics and reporting. It does not solve the
missing allocated-power variable.

## Metric Reporting Patch

Added an audit-only reporting surface:

1. `scripts/audit_ee_denominator.py`
2. `modqn_paper_reproduction.cli:ee_denominator_audit_main`
3. `analysis.export_ee_denominator_audit`

It emits:

1. denominator variability audit
2. unit audit
3. runtime mapping for `R_i(t)`, `P_b(t)`, and `active_beams`
4. fixed-power proxy `ee_fixed_power_proxy_bps_per_w`
5. explicit blocked/no-go decision

No training metric, reward, config, checkpoint, or frozen baseline artifact was
changed. Because `P_b(t)` is only static `tx_power_w`, the patch intentionally
does not add HOBS EE as a trainer/evaluator metric.

## Validation

Focused tests:

```text
./.venv/bin/python -m pytest tests/test_ee_denominator_audit.py tests/test_step.py -q
56 passed
```

Runtime audit:

```text
./.venv/bin/python scripts/audit_ee_denominator.py \
  --config configs/modqn-paper-baseline.resolved-template.yaml \
  --output-dir artifacts/ee-denominator-audit-phase-02 \
  --evaluation-seed 20260428
```

Result:

```text
decision=blocked
phase03=no-go
classification=fixed-power-active-beam-count-proxy
```

## Phase 02 Decision

Phase 02 is `BLOCKED` for HOBS-valid EE adoption.

The audit/reporting deliverable is complete, but the current runtime denominator
is not a controlled or allocated HOBS beam-power variable. It is a static
`tx_power_w` multiplied by active-beam count.

## Phase 03 Gate

Phase 03 is `NO-GO`.

Training EE-MODQN now would reduce EE to throughput scaling under a fixed-power
active-beam-count proxy. That would not support an energy-aware learning claim.

## Claims Still Forbidden

1. Do not claim HOBS-valid `EE_system(t)` from the current runtime.
2. Do not claim Phase 03 EE-MODQN is unblocked.
3. Do not claim energy-aware learning from static `tx_power_w`.
4. Do not relabel fixed active-beam-count power as allocated `P_b(t)`.
5. Do not modify the frozen MODQN baseline artifacts or configs for EE.
