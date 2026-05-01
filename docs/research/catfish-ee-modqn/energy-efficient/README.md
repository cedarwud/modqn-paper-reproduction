# Energy-Efficient Formula Review Pack

**Date moved into repo:** `2026-05-01`
**Scope:** formula-review evidence and design notes for HOBS-style active
transmit-power EE in the MODQN research track.

This directory was moved from the workspace-level
`/home/u24/papers/energy-efficient/` folder so that
`modqn-paper-reproduction` remains self-contained for the next EE-MODQN design
gate.

## Authority Boundary

These files are supporting review material, not a replacement for the current
execution authority.

Use this order:

1. `../hobs-active-tx-ee-modqn-feasibility.execution-report.md` for the latest
   implementation and gate result.
2. `ee-formula-final-review-with-codex-2026-05-01.md` for the final formula
   policy and claim boundary.
3. `modqn-r1-to-hobs-active-tx-ee-design-2026-05-01.md` for the design direction
   and the post-Route-D implementation-status note.
4. `codex-ee-formula-current-synthesis.md`, `codex.md`, `claude.md`, and
   `gemini.md` as comparison reviews only.

## Current Formula Policy

Main metric:

```text
eta_EE,active-TX =
  (sum_t sum_u R_u^t Delta_t)
  /
  (sum_t sum_s sum_b z_{s,b}^t p_{s,b}^t Delta_t)
```

Use it as transmission-side / active-beam transmit-power EE. Do not present it
as total spacecraft EE or physical energy saving.

Composite, circuit/static, processing, handover-aware, and utility-form EE
variants are sensitivity / ablation material only unless future work supplies
new LEO-specific parameter evidence and a separate gate.

## Current Development Boundary

The formula and DPC denominator feasibility have passed, but Route `D` remains
blocked because the learned MODQN policy still collapses to one active beam.

Next EE-MODQN work must be an anti-collapse / capacity / assignment design gate,
not more Route `D` training and not Catfish repair.
