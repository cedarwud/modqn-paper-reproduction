# RA-EE-07 Execution Report: Constrained-Power Allocator Distillation Gate

**Date:** `2026-04-29`
**Status:** `PASS as offline fixed-association deployable power-allocator gate only`
**Scope:** offline replay over fixed association as the primary comparison, with
RA-EE-06B association proposal buckets exported as diagnostics only. No learned
association, hierarchical RL, joint association + power training, Catfish,
multi-Catfish, RB / bandwidth allocation, old EE-MODQN continuation, frozen
baseline mutation, HOBS optimizer claim, or physical energy-saving claim was
performed.

## Protocol

Method label:

```text
RA-EE constrained-power allocator distillation gate
```

Config:

```text
configs/ra-ee-07-constrained-power-allocator-distillation.resolved.yaml
```

Artifacts:

```text
artifacts/ra-ee-07-constrained-power-allocator-distillation/
```

Primary comparison:

```text
matched fixed association + safe-greedy-power-allocator
vs
matched fixed association + deployable-stronger-power-allocator
```

Deployable candidates:

1. `p05-slack-aware-trim-tail-protect-boost`
2. `bounded-local-search-codebook`
3. `finite-codebook-dp-knapsack`
4. `deterministic-hybrid-runtime`

The primary deployable row is the deterministic hybrid. In the completed
held-out replay it selected `bounded-local-search-codebook` on all held-out
steps. This selection used only current runtime channel/load/QoS slack and
finite codebook evaluation. It did not use oracle labels, future outcomes, or
held-out answers.

Diagnostic-only rows:

1. `matched-fixed-association+fixed-1w-diagnostic`
2. `matched-fixed-association+constrained-power-oracle-isolation`
3. `association-proposal+deployable-stronger-power-allocator-diagnostic`
4. `association-oracle+constrained-power-oracle-upper-bound`

The association diagnostic replay uses the RA-EE-06B proposal buckets with
`diagnostic_max_steps=5`; the primary fixed-association replay is not step
capped.

## Power Contract

```text
levels: [0.5, 0.75, 1.0, 1.5, 2.0] W
per-beam max: 2.0 W
total active-beam budget: 8.0 W
inactive beams: 0 W
```

For every row, the same `effective_power_vector_w` feeds SINR / SNR numerator,
throughput, `EE_system = sum_i R_i / sum_active_beams P_b`, audit logs, and
budget checks. No power repair is used; requested and effective vectors are
exported.

## Held-Out Results

Primary deployable allocator versus matched fixed association +
`safe-greedy-power-allocator`:

| Held-out trajectory | EE delta | p05 ratio | Oracle gap closure | Accepted |
|---|---:|---:|---:|---|
| `random-valid-heldout` | `+7.095305632511327` | `0.975101299563657` | `1.0` | `true` |
| `spread-valid-heldout` | `+5.155057818148293` | `0.9995533609188448` | `1.0` | `true` |
| `load-skewed-heldout` | `+3.5780032540779985` | `1.0421330635721502` | `1.0` | `true` |
| `mobility-shift-heldout` | `+7.483104400382672` | `0.986358235603686` | `1.0` | `true` |
| `mixed-valid-heldout` | `+5.918477335805619` | `1.0743825126983608` | `1.0` | `true` |

Held-out gate summary:

```text
noncollapsed held-out trajectories: 5 / 5
positive EE delta trajectories: 5 / 5
accepted candidate trajectories: 5 / 5
aggregate oracle gap closure: 1.0
positive seeds: 4 / 5
max positive trajectory delta share: 0.2560081286323895
max positive seed delta share: 0.28553406270120996
p05 throughput guardrail: pass
served ratio guardrail: pass
outage ratio guardrail: pass
budget / per-beam / inactive-power violations: 0
denominator/profile/power distribution: non-single-point
```

The constrained-power oracle remains diagnostic-only. RA-EE-07's oracle
isolation is a feasible-codebook upper-bound diagnostic for the RA-EE-07
power contract; it is not a runtime method and is not a HOBS optimizer.

## Diagnostic Association Buckets

RA-EE-06B association proposal buckets are exported separately as diagnostics.
They do not enter the RA-EE-07 PASS decision. Their purpose is to keep the
blocked association-proposal route visible while testing whether the deployable
power allocator changes the fixed-association conclusion. It does not authorize
learned association or hierarchical training.

## Validation

Focused tests:

```text
.venv/bin/python -m pytest tests/test_ra_ee_07_constrained_power_allocator_distillation.py -q
```

Result:

```text
9 passed
```

Artifact command:

```text
.venv/bin/python scripts/run_ra_ee_07_constrained_power_allocator_distillation.py
```

Result:

```text
decision=PASS
```

## Decision

```text
RA-EE-07: PASS
```

This PASS means only that a deployable non-oracle power allocator beat the
matched fixed-association safe-greedy allocator on the offline held-out gate.
It does not establish learned association, hierarchical RL, joint association +
power training, RB / bandwidth allocation, full RA-EE-MODQN effectiveness, HOBS
optimizer behavior, or physical energy saving.

## Remaining Blockers

1. No learned association or full RA-EE-MODQN policy exists.
2. No joint association + power training exists.
3. No RB / bandwidth allocation exists.
4. Association proposal and constrained-oracle rows remain diagnostic-only.
5. The deployable allocator is a bounded finite-codebook controller, not a
   HOBS optimizer.

## Forbidden Claims Still Active

1. Do not call RA-EE-07 full RA-EE-MODQN.
2. Do not claim learned association or hierarchical RL effectiveness.
3. Do not claim joint association + power training.
4. Do not claim old EE-MODQN effectiveness.
5. Do not claim HOBS optimizer behavior.
6. Do not claim Catfish, multi-Catfish, or Catfish-EE-MODQN effectiveness.
7. Do not add or claim RB / bandwidth allocation.
8. Do not treat per-user EE credit as system EE.
9. Do not use scalar reward alone as success evidence.
10. Do not claim full paper-faithful reproduction.
11. Do not claim physical energy saving.
