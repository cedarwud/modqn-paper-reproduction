# Phase 02B Execution Report: HOBS Power Surface and Denominator Audit

**Date:** `2026-04-28`  
**Status:** `PROMOTED for metric audit only`  
**Scope:** per-beam transmit-power surface and denominator audit. No
EE-MODQN training, reward replacement, Catfish, multi-Catfish, or full
paper-faithful reproduction claim was performed.

## Execution Decision

Phase 02B adds an opt-in runtime power surface that makes
`EE_system(t) = sum_i R_i(t) / sum_active_beams P_b(t)` defensibly
computable for audit and later paired Phase 03 design.

The surface is not enabled by default. The frozen MODQN baseline still uses
`ChannelConfig.tx_power_w` exactly as before.

## Chosen Power Model

Config namespace:

```text
configs/ee-modqn-power-surface-phase-02b.resolved.yaml
```

Runtime mode:

```text
hobs_power_surface_mode = active-load-concave
inactive_beam_policy = zero-w
```

Power allocation:

```text
P_b(t) = 0 W, if N_b(t) = 0
P_b(t) = min(max_power_w,
             active_base_power_w + load_scale_power_w * N_b(t)^load_exponent),
         if N_b(t) > 0
```

Phase 02B parameters:

```text
active_base_power_w = 0.25
load_scale_power_w = 0.35
load_exponent = 0.5
max_power_w = 2.0
units = linear W
```

This is a synthesized allocation proxy. It is HOBS-compatible because it
creates explicit downlink per-beam transmit power in linear W, uses inactive
beam `0 W`, and feeds the same `beam_transmit_power_w[b]` into the SINR
numerator path. It is not a paper-faithful power-control policy.

## Implementation Surface

Changed runtime surfaces:

1. `PowerSurfaceConfig` gates power behavior behind explicit mode selection.
2. `StepResult.beam_transmit_power_w` reports per-beam `P_b(t)` in linear W.
3. `StepResult.active_beam_mask` reports active beams explicitly.
4. `StepEnvironment._build_states_and_masks` uses `beam_transmit_power_w[b]`
   in the opt-in SINR path and preserves the static baseline path by default.
5. `compute_channel(..., tx_power_w=...)` accepts a linear-W override for
   explicit power-surface use.
6. `export_ee_denominator_audit` reports per-beam power vectors, active masks,
   total active beam power, denominator variability, unit audit, and go/no-go.

## Audit Artifacts

Generated under:

```text
artifacts/ee-power-surface-phase-02b/
```

Key result:

```text
rows audited: 30
mode: active-load-concave
inactive beam policy: zero-w
active beam counts sampled: [1, 7]
total active beam power sampled: 2.0 W to 10.975454935197652 W
beam power vector distinct count: 12
denominator varies: true
classification: hobs-compatible-active-load-concave-power-surface
HOBS EE defensible: true
```

## Validation

Focused tests:

```text
./.venv/bin/python -m pytest \
  tests/test_ee_denominator_audit.py tests/test_step.py tests/test_channel.py -q
```

Result:

```text
110 passed
```

Audit command:

```text
./.venv/bin/python scripts/audit_ee_denominator.py \
  --config configs/ee-modqn-power-surface-phase-02b.resolved.yaml \
  --output-dir artifacts/ee-power-surface-phase-02b \
  --evaluation-seed 20260428
```

Result:

```text
decision=promoted-for-metric-audit
phase03=conditional-go-for-paired-phase-03-design
classification=hobs-compatible-active-load-concave-power-surface
```

## Phase 02B Decision

`PROMOTED for metric audit only`.

`EE_system(t)` is defensible on the opt-in Phase 02B power surface. This does
not authorize replacing the reward, training EE-MODQN in this phase, adding
Catfish, or claiming full paper-faithful reproduction.

## Phase 03 Gate

`CONDITIONAL-GO for paired Phase 03 design`.

Phase 03 may proceed only as a paired `MODQN-control` vs `EE-MODQN` validation
on the same opt-in HOBS-linked SINR / power surface. Old baseline artifacts do
not establish the effect of objective substitution.

## Remaining Blockers

1. The power surface is a disclosed proxy, not a paper-backed HOBS optimizer.
2. There is still no EE-MODQN reward implementation or training evidence.
3. Phase 03 still needs paired control vs EE-MODQN runs, rescaling checks, and
   reward-hacking diagnostics before any EE-MODQN effect claim.
4. Catfish, multi-Catfish, and final Catfish-EE-MODQN claims remain blocked.
