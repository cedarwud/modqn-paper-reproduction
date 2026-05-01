# Phase 05A Execution Report: Multi-Buffer Validation

**Date:** `2026-04-29`
**Status:** `PASS for bounded diagnostic completion; FAIL/BLOCK for Phase 05B planning`
**Scope:** objective-specific replay-buffer diagnostics only. No frozen baseline config/artifact mutation, EE reward/objective, full multi-Catfish agents, `catfish-r1` / `catfish-r2` / `catfish-r3` learners, long training, or effectiveness claim was introduced.

## Guardrail Boundary

Phase 05A kept the original MODQN reward surface fixed:

```text
r1 = throughput
r2 = handover penalty
r3 = load balance
```

This run used objective-wise percentile/rank admission for `r1`, `r2`, and `r3`
buffers. Scalar Phase 04 high-value replay was used only as a comparison
reference:

```text
quality = 0.5*r1 + 0.3*r2 + 0.2*r3
```

No scalar-only success evidence is used in this report.

## Existing Artifact Check

The existing Phase 04C artifacts were checked first:

1. `artifacts/catfish-modqn-phase-04-c-primary-shaping-off-20ep/`
2. `artifacts/catfish-modqn-phase-04-c-no-intervention-20ep/`
3. `artifacts/catfish-modqn-phase-04-c-control-20ep/`

They were not sufficient for Phase 05A because they contain aggregate training
rows and Catfish replay distribution summaries, but not transition-level sample
ids and raw `r1` / `r2` / `r3` vectors needed to compute objective top-set
Jaccard overlap.

The smallest additional instrumentation added here records transition-level
reward vectors during the existing Phase 04-B single-Catfish trainer surface and
then runs analysis-only buffer selection.

## New Bounded Diagnostic

| Item | Value |
|---|---|
| Config | `configs/catfish-modqn-phase-05a-multi-buffer-primary.resolved.yaml` |
| Artifact root | `artifacts/catfish-modqn-phase-05a-multi-buffer-primary-20ep/` |
| Source trainer surface | Phase 04-B single `Catfish-MODQN` primary shaping-off |
| Episode budget | `20` |
| Transition samples | `20000` |
| Objective top percentile | `0.80` |
| Full multi-Catfish agents | not started |
| EE reward/objective | not introduced |

Artifact files:

1. `phase05a_multi_buffer_diagnostics.json`
2. `phase05a_transition_samples.jsonl`
3. `phase05a_run_metadata.json`
4. `training_log.json`

## Buffer Diagnostics

| Buffer | Size | Admission share | Threshold | r1 mean | r2 mean | r3 mean | Distinct intervention samples |
|---|---:|---:|---:|---:|---:|---:|---|
| `r1` high throughput | `4000` | `0.20` | `r1 >= 147.809079` | `178.192927` | `-0.432000` | `-1.799970` | no |
| `r2` handover penalty closer to zero | `20000` | `1.00` | `abs(r2) <= 0.5` | `122.228478` | `-0.429150` | `-1.821705` | yes, but degenerate |
| `r3` load-balance penalty closer to zero | `4000` | `0.20` | `abs(r3) <= 1.325692` | `122.193760` | `-0.432625` | `-1.000462` | no |
| scalar Phase 04 high-value | `3917` | `0.19585` | rolling scalar high-value admission | `178.447253` | `-0.432601` | `-1.786527` | comparison only |

Important interpretation:

1. `r1` almost duplicates the scalar Phase 04 high-value buffer.
2. `r2` does not form a bounded high-value subset in this run. The percentile
   criterion degenerates because the handover penalty is too coarse on the
   sampled surface, so `abs(r2) <= 0.5` admits every transition.
3. `r3` improves load-balance penalty distribution, but it contributes no
   samples distinct from the union of other objective buffers plus scalar
   high-value replay because `r2` admitted the full sample set.

## Overlap

Objective-buffer Jaccard:

| Pair | Jaccard |
|---|---:|
| `r1` vs `r2` | `0.200000` |
| `r1` vs `r3` | `0.117631` |
| `r2` vs `r3` | `0.200000` |

Overlap against scalar Phase 04 high-value replay:

| Buffer | Jaccard vs scalar |
|---|---:|
| `r1` | `0.936644` |
| `r2` | `0.195850` |
| `r3` | `0.119327` |

## Warnings

1. `r1` duplicates the scalar high-value buffer (`Jaccard = 0.936644`).
2. `r1` has no distinct samples versus all other buffers plus scalar replay.
3. `r2` percentile admission is degenerate (`admission_share = 1.0`).
4. `r3` has no distinct samples versus all other buffers plus scalar replay.

## Stop Conditions

Triggered:

1. `diagnostics-cannot-prove-distinct-sample-types`
2. `objective-percentile-degenerated`

Not triggered:

1. No reward/state/action/backbone change was required.
2. EE or full multi-Catfish was not required.
3. The explicit `r3` low-throughput stop condition did not trigger by the
   implemented threshold, but `r3` still failed distinct intervention
   contribution because the `r2` buffer admitted the full sample set.

## Decision

```text
Phase 05A bounded diagnostic completion: PASS
Objective-specific buffer distinctness: FAIL
Phase 05B full multi-agent validation planning: BLOCK
Multi-Catfish effectiveness claim: DISALLOWED
Catfish-EE-MODQN claim: DISALLOWED
```

Phase 05A answers the immediate research question negatively for this bounded
surface: the current objective-specific buffer construction does not yet prove
that `r1` / `r2` / `r3` high-value buffers capture meaningfully distinct
experience distributions. The `r1` buffer is essentially the scalar/high-
throughput buffer, and `r2` is not selective enough under the observed handover
penalty distribution.

## Verification

Focused regression:

```text
.venv/bin/python -m pytest tests/test_catfish_phase05a_multi_buffer.py tests/test_catfish_phase04b.py -q
16 passed
```

Bounded diagnostic command:

```text
.venv/bin/python scripts/run_catfish_phase05a_multi_buffer_validation.py --config configs/catfish-modqn-phase-05a-multi-buffer-primary.resolved.yaml --output-dir artifacts/catfish-modqn-phase-05a-multi-buffer-primary-20ep --progress-every 0
```

Result:

```text
samples=20000
recommendation=BLOCK_PHASE_05B
```
