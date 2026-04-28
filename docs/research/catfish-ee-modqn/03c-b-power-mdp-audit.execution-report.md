# Phase 03C-B Execution Report: Power-MDP Codebook Audit

**Date:** `2026-04-28`  
**Status:** `PASS to bounded paired pilot`  
**Scope:** static / counterfactual denominator audit only. No long training,
Catfish, multi-Catfish, frozen baseline mutation, or HOBS optimizer claim was
performed.

## Design

Phase 03C-B adds an opt-in hierarchical handover plus centralized
beam-power-codebook audit surface:

1. Handover trajectories are fixed first.
2. The same beam decisions are replayed under multiple power semantics.
3. The primary metric is system EE:

```text
EE_system(t) = sum_i R_i(t) / sum_active_beams P_b(t)
```

The new controller is a **new-extension / HOBS-inspired** codebook surface. It
is not a HOBS optimizer and does not relabel Phase 02B as paper-backed power
control.

## Implementation

New config namespace:

```text
configs/ee-modqn-phase-03c-b-power-mdp-audit.resolved.yaml
```

New artifact namespace:

```text
artifacts/ee-modqn-phase-03c-b-power-mdp-audit/
```

Changed / added surfaces:

1. `PowerSurfaceConfig` now supports the opt-in mode
   `phase-03c-b-power-codebook`.
2. Codebook profiles:
   - `fixed-low`
   - `fixed-mid`
   - `fixed-high`
   - `load-concave`
   - `qos-tail-boost`
   - `budget-trim`
3. In codebook mode, inactive beams are always `0 W`.
4. Active beams receive one configured linear-W level with
   `0 < P_b(t) <= max_power_w`.
5. `scripts/audit_phase03c_b_power_mdp.py` exports the static audit.

Phase 02B remains labeled as a synthesized proxy. Phase 03C-B is labeled as
new-extension / HOBS-inspired.

## Audit Protocol

Fixed handover trajectories:

1. `phase03b-ee-best-eval`, loaded from
   `artifacts/ee-modqn-phase-03b-ee-objective-geometry-pilot/` checkpoint
   `best-weighted-reward-on-eval`, episode `49`
2. `hold-current`
3. `random-valid`
4. `spread-valid`

Power semantics replayed over each trajectory:

1. `fixed-2w`
2. `phase-02b-proxy`
3. `fixed-low`
4. `fixed-mid`
5. `fixed-high`
6. `load-concave`
7. `qos-tail-boost`
8. `budget-trim`

Evaluation seeds: `[100, 200, 300, 400, 500]`.

## Key Results

Generated artifacts:

1. `phase03c_b_power_mdp_audit_summary.json`
2. `phase03c_b_power_mdp_step_metrics.csv`
3. `phase03c_b_power_mdp_candidate_summary.csv`
4. `review.md`

Decision flags:

```text
denominator_changed_by_power_decision = true
ranking_separates_under_same_policy_rescore = true
has_budget_respecting_codebook_candidate = true
fixed_denominator_audit_catches_fixed_denominator = true
```

Ranking top changed under same-policy throughput vs EE rescoring for all four
fixed trajectories:

```text
hold-current: fixed-2w -> load-concave
phase03b-ee-best-eval: budget-trim -> fixed-low
random-valid: fixed-2w -> fixed-low
spread-valid: fixed-2w -> qos-tail-boost
```

Representative summaries:

| Trajectory | Power semantics | EE_system aggregate | Throughput mean | Throughput p05 | Mean active power | Denominator varies | Budget violations |
|---|---:|---:|---:|---:|---:|---|---:|
| `phase03b-ee-best-eval` | `fixed-2w` | `873.093459` | `17.461869` | `13.565598` | `2.0 W` | `false` | `0` |
| `phase03b-ee-best-eval` | `fixed-low` | `873.094268` | `4.365471` | `3.391402` | `0.5 W` | `false` | `0` |
| `spread-valid` | `fixed-2w` | `873.055017` | `122.227702` | `94.478677` | `14.0 W` | `false` | `50` |
| `spread-valid` | `budget-trim` | `873.203403` | `69.856272` | `48.455051` | `8.0 W` | `false` | `0` |
| `spread-valid` | `qos-tail-boost` | `873.941909` | `56.806224` | `25.057351` | `6.5 W` | `false` | `0` |
| `random-valid` | `load-concave` | `872.875725` | `102.824760` | `67.428473` | `11.78 W` | `true` | `50` |
| `random-valid` | `qos-tail-boost` | `872.193826` | `70.037164` | `33.378709` | `8.03 W` | `true` | `10` |

All audited rows kept served ratio at `1.0`; outage ratio was `0.0`.

## Denominator Variability Result

The audit proves the minimum requested denominator-coupling property:

1. For the same fixed beam trajectory, switching power semantics changes total
   active beam power.
2. Fixed-denominator candidates are detected as fixed.
3. Variable-load candidates show within-evaluation denominator variability on
   non-collapsed trajectories such as `hold-current` and `random-valid`.

The Phase 03B learned trajectory still remains a one-active-beam trajectory.
Its per-candidate denominator is fixed inside each replay, though cross-candidate
power decisions change the denominator from `0.5 W` to `2.0 W`.

## Ranking Separation Result

Throughput and EE rescoring no longer produce identical candidate rankings
under fixed trajectories. This is the required static precondition for a
bounded paired pilot.

This is not learned EE evidence. It only shows that an explicit power decision
surface can change the denominator and separate throughput ranking from
system-EE ranking.

## Validation

Focused tests:

```text
.venv/bin/python -m pytest tests/test_phase03c_b_power_mdp_audit.py -q
```

Result:

```text
5 passed
```

Guardrail tests:

```text
.venv/bin/python -m pytest \
  tests/test_ee_denominator_audit.py \
  tests/test_phase03_ee_modqn.py \
  tests/test_phase03a_diagnostics.py \
  tests/test_phase03c_b_power_mdp_audit.py \
  tests/test_step.py -q
```

Result:

```text
74 passed
```

Audit command:

```text
.venv/bin/python scripts/audit_phase03c_b_power_mdp.py \
  --config configs/ee-modqn-phase-03c-b-power-mdp-audit.resolved.yaml \
  --output-dir artifacts/ee-modqn-phase-03c-b-power-mdp-audit
```

Result:

```text
decision=PASS to bounded paired pilot
denominator_changed=true
ranking_separates=true
```

## Phase 03C-B Decision

`PASS to bounded paired pilot`.

Allowed next step:

```text
bounded paired pilot only; no EE-MODQN effectiveness claim
```

## Remaining Blockers

1. This is static / counterfactual evidence, not learned policy evidence.
2. The Phase 03B learned trajectory still collapses to one active beam.
3. Phase 02B remains a synthesized proxy comparator.
4. Phase 03C-B is a new-extension / HOBS-inspired controller, not a HOBS
   optimizer.
5. Any training pilot must be separately bounded and paired.

## Forbidden Claims Still Active

1. Do not claim EE-MODQN effectiveness.
2. Do not claim full paper-faithful reproduction.
3. Do not claim Catfish, multi-Catfish, or final Catfish-EE-MODQN.
4. Do not treat per-user EE credit as system EE.
5. Do not use scalar reward alone as success evidence.
6. Do not label Phase 03C-B as a HOBS optimizer.
