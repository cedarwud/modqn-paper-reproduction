# Phase 05B Execution Report: Multi-Catfish Bounded Pilot

**Date:** `2026-04-29`
**Status:** `PASS for bounded runnable evidence; FAIL for acceptance / effectiveness`
**Scope:** bounded Phase `05B` pilot over the original MODQN reward only. No
EE reward / objective, Catfish-EE-MODQN claim, RA-EE continuation, frozen
baseline mutation, full paper-faithful reproduction claim, or long training was
introduced.

## 1. Current State

Original MODQN reward semantics stayed fixed:

```text
r1 = throughput
r2 = handover penalty
r3 = load balance
```

Phase `05B` implemented the bounded plan authorized after Phase `05R`: guarded
residual objective buffers, objective-specialist Catfish learners, and matched
comparators under a fixed total Catfish intervention budget.

## 2. Changed Files

Implementation and analysis surfaces changed:

1. `src/modqn_paper_reproduction/runtime/trainer_spec.py`,
2. `src/modqn_paper_reproduction/config_loader.py`,
3. `src/modqn_paper_reproduction/runtime/catfish_replay.py`,
4. `src/modqn_paper_reproduction/algorithms/catfish_modqn.py`,
5. `src/modqn_paper_reproduction/algorithms/multi_catfish_modqn.py`,
6. `src/modqn_paper_reproduction/orchestration/train_main.py`,
7. `src/modqn_paper_reproduction/cli.py`,
8. `src/modqn_paper_reproduction/analysis/catfish_phase05b_bounded_pilot.py`,
9. `tests/test_catfish_phase05b_multi_catfish.py`.

## 3. Configs And Artifacts

Configs:

1. `configs/catfish-modqn-phase-05b-modqn-control.resolved.yaml`,
2. `configs/catfish-modqn-phase-05b-single-catfish-equal-budget.resolved.yaml`,
3. `configs/catfish-modqn-phase-05b-primary-multi-catfish-shaping-off.resolved.yaml`,
4. `configs/catfish-modqn-phase-05b-multi-buffer-single-learner.resolved.yaml`,
5. `configs/catfish-modqn-phase-05b-random-or-uniform-buffer-control.resolved.yaml`.

Artifacts:

1. `15` run directories under `artifacts/catfish-modqn-phase-05b-*-seed01..03/`,
2. summary JSON:
   `artifacts/catfish-modqn-phase-05b-bounded-pilot-summary/phase05b_bounded_pilot_summary.json`.

## 4. Protocol

The bounded run completed:

1. `5` comparators,
2. `3` matched seed triplets,
3. `15` total runs,
4. `20` episodes each,
5. evaluation seeds `[100, 200, 300, 400, 500]`,
6. eval / checkpoint cadence every `5` episodes and final,
7. configured total Catfish share `0.30`,
8. observed Catfish ratio for Catfish comparators `0.296875`.

Seed triplets:

```text
(42, 1337, 7)
(43, 1338, 8)
(44, 1339, 9)
```

## 5. Tests Run

The implementation report recorded:

1. `.venv/bin/python -m py_compile ...` passed,
2. `.venv/bin/python -m pytest tests/test_catfish_phase05b_multi_catfish.py -q`
   passed with `9` tests,
3. `.venv/bin/python -m pytest tests/test_catfish_phase04b.py tests/test_catfish_phase05a_multi_buffer.py tests/test_catfish_phase05b_multi_catfish.py -q`
   passed with `25` tests,
4. protected baseline config / artifact diff check found no protected baseline
   changes.

## 6. Metrics Summary

Final mean scalar by comparator:

| Comparator | Final mean scalar |
|---|---:|
| `single-catfish-equal-budget` | `609.209311` |
| `random-or-uniform-buffer-control` | `609.128592` |
| `modqn-control` | `608.359098` |
| `multi-buffer-single-learner` | `608.293825` |
| `primary-multi-catfish-shaping-off` | `608.037110` |

Primary Multi-Catfish versus single Catfish:

1. scalar reward was worse,
2. `r2` improved,
3. `r1` did not improve,
4. `r3` did not improve,
5. the result was not more than scalar evidence,
6. controls explained away the primary result.

Primary final means:

| Metric | Primary Multi-Catfish | Single Catfish |
|---|---:|---:|
| scalar | `608.037110` | `609.209311` |
| `r1_mean` | `1225.884447` | `1228.067570` |
| `r2_mean` | `-4.286667` | `-4.373333` |
| `r3_mean` | `-18.095565` | `-17.562371` |
| handovers | `857.333333` | `874.666667` |

## 7. Diagnostics Summary

Observed diagnostics:

1. primary per-update mix was visible as `45 main + 7 r1 + 6 r2 + 6 r3` per
   `64`-sample update,
2. primary final buffer Jaccards stayed low:
   `r1/r2 = 0.0517`, `r1/r3 = 0.1342`, `r2/r3 = 0.0928`,
3. no NaN was detected,
4. no action collapse was detected,
5. replay starvation counters were nonzero.

Summary acceptance checks:

| Check | Result |
|---|---|
| required roles complete | pass |
| actual Catfish ratio near `0.30` | pass |
| primary beats single on scalar | fail |
| component improvements vs single | `r2` only |
| not scalar reward alone | fail |
| multi-buffer control explains gain | fail / stop |
| random control explains gain | fail / stop |
| overall acceptance | fail |

Triggered stop conditions:

1. replay starvation detected,
2. multi-buffer / single-learner matched or exceeded primary,
3. random-buffer control matched or exceeded primary.

## 8. Claim Boundary

The summary recorded:

1. no Catfish-EE or EE claim was made,
2. no full paper-faithful reproduction claim was made,
3. Phase `05R` was not used as an effectiveness claim,
4. scalar reward alone was not used as success evidence.

## 9. Decision

```text
Phase 05B bounded runnable evidence: PASS
Phase 05B acceptance / effectiveness: FAIL
Multi-Catfish-MODQN promotion: BLOCK
Longer Phase 05B training by default: BLOCK
Catfish-EE / EE / RA-EE continuation claim: BLOCK
```

The bounded pilot is valuable as evidence because it demonstrates the
implementation surface can run and that diagnostics can observe the intended
multi-source replay composition. It does not establish that objective-specialist
learners add value beyond single Catfish or equal-budget control mixing.

## 10. Next State

Do not continue Phase `05B` by default with longer training, shaping-on primary
runs, or tuning. The current Multi-Catfish route should be treated as a negative
bounded result / paper boundary finding unless a new research question is
explicitly opened.

Any future reopening must define a new gate that explains why it would not be
explained away by:

1. single Catfish under equal budget,
2. multi-buffer / single-learner control,
3. random or uniform equal-budget buffer control,
4. replay starvation or intervention accounting artifacts.
