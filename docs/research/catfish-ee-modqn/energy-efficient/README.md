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

## Handover-Aware EE Evidence Map

For later thesis slides, `handover-aware EE` may be discussed as a
sensitivity-defined ratio-form metric:

```text
eta_EE,HO =
  total_bits
  /
  (total_communication_energy + E_HO,total)

E_HO,total = sum_t sum_u delta_HO,u^t * E_u,HO
```

This is not the same as the active-TX EE main metric above. It is a thesis-level
sensitivity extension. Use the following evidence chain:

1. `system-model-refs/system-model-formulas.md` §3.14-3.16 defines the local
   `handover-aware EE` formula and keeps it separate from active-TX EE.
2. `system-model-refs/simulator-parameter-spec.md` §6 marks `E_u,HO` as a
   scenario parameter with no universal default.
3. `paper-catalog/catalog/PAP-2025-EAQL.json` supports explicit handover energy
   cost in an energy-aware handover utility / reward.
4. `paper-catalog/catalog/PAP-2025-BIPARTITE-HM.json` supports handover
   strategy-cost modeling with per-handover energy and sensitivity to energy
   weighting.

Allowed wording: handover-aware EE under a declared high handover-cost /
service-continuity-sensitive sensitivity setting.

Forbidden wording: universal EE-MODQN superiority, active-TX EE recovery,
physical satellite energy saving, or adoption of any single `E_HO` value as a
universal LEO constant.

## Scenario Realism Boundary

The high handover-cost / service-continuity-sensitive condition is a reasonable
thesis sensitivity setting, not a universal deployment baseline. The supporting
literature does show that LEO handover studies can model handover energy,
operational handover cost, setup cost, RTT, strategy cost, and energy weighting
inside handover decisions. The local system model also allows
`E_{u,HO}` to enter both utility-form and ratio-form handover-aware EE analysis.

However, `E_{u,HO}` remains a scenario parameter. The low-cost `E_HO=3 J`
value is paper-specific to the EAQL evidence chain and is not adopted as a
universal LEO constant. The high-cost rows used by the HEA-MODQN thesis result
(`E_HO=130/150/200`) should be described as a declared
service-continuity-sensitive sensitivity regime that represents signaling,
setup/recovery, interruption, QoS-continuity, or terminal-side overhead
pressure. Do not write that real LEO handovers generally consume `130 J`,
`150 J`, or `200 J`.

As of the clean ratio-form artifact gate, the scoped HEA-MODQN claim may include
both utility-form `J` and ratio-form `EE_HO` improvement under that high-cost
setting. It still must not be shortened to physical satellite energy saving,
active-TX EE recovery, general EE-MODQN superiority, low-cost success, or
Catfish-EE / Phase `06` readiness.

## Final Relaxation Stop-Loss

Before thesis writing, the relaxed communication-only formula was checked from
the clean HEA artifact:

```text
EE_general = total_bits / communication_energy_joules
```

This check blocks any attempt to remove the handover-event term from the
denominator. The HEA candidate has `0/30` wins across the full sensitivity grid,
`0/15` wins in the primary high-cost subset, and `0/6` wins in the secondary
subset. Primary mean communication-only `EE_delta` is about `-82980 bits/J`;
secondary mean communication-only `EE_delta` is about `-74659 bits/J`.

Therefore, the positive result exists only for the handover-aware denominator:

```text
total_bits / (communication_energy_joules + E_HO,total)
```

Do not claim general EE, active-TX EE recovery, or communication-energy-only EE
superiority from the HEA-MODQN result. The correct next step is thesis writing,
not another EE design gate.

## Catfish / Multi-Catfish Boundary

Catfish may now be discussed only as future training-time intervention research
over the scoped HEA-MODQN baseline. It is not an EE repair mechanism and it is
not part of the accepted HEA-MODQN thesis claim.

The old Catfish-over-HEA generic single-intervention route has two separate
statuses:

```text
readiness / baseline parity = PASS
bounded matched pilot = BLOCK
```

The readiness slice in `ntn-sim-core` has passed only for the default-off hook
surface:

```text
catfish_enabled=false
intervention_policy_id=disabled
intervention_ratio=0
disabled path returns the original sampled batch
disabled path records zero interventions
```

This is readiness / baseline-parity evidence, not effectiveness evidence. The
baseline parity gate passed for default versus explicit `catfish_enabled=false`
trainStep behavior, but the enabled bounded pilot worsened utility `J`, failed
some `EE_HO` cells, and violated handover / `r2` guardrails. Do not tune or
rerun that same generic single-intervention route as the next step.

The current Catfish planning line is:

```text
../2026-05-02-multi-catfish-redesign-plan.md
../2026-05-02-multi-catfish-gate1a-transition-provenance-sdd.md
```

It designs Multi-Catfish first and treats Single-Catfish as a collapsed
ablation. Gate 1 read-only diagnostics returned `NEEDS MORE DESIGN`: existing
artifacts are aggregate / seed / cell level and cannot support transition-level
`catfish-ee`, `catfish-ho`, `catfish-qos`, or coordinator diagnostics. The next
step is transition-provenance schema design, not implementation or pilot work.
Any future Catfish pilot must be separately controller-authorized and must not
change utility `J`, ratio-form `EE_HO`, active-TX EE accounting, reward
semantics, state/action surfaces, evaluation, policy selection, seeds, budget,
or checkpoint protocol. It must not claim Catfish-EE, Phase `06`, Active-TX EE
recovery, current Multi-Catfish effectiveness, physical energy saving, or
low-cost EE success.

## Current Active-TX Development Boundary

The formula and DPC denominator feasibility have passed, but Route `D` remains
blocked because the learned MODQN policy still collapses to one active beam.

Next EE-MODQN work must be an anti-collapse / capacity / assignment design gate,
not more Route `D` training and not Catfish repair.
