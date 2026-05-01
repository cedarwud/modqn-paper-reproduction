# Phase 05B Planning Report: Multi-Catfish Bounded Pilot

**Date:** `2026-04-29`
**Status:** `PASS for future implementation prompt draft only`
**Scope:** planning boundary only. No Phase `05B` implementation, training,
config generation, artifacts, full Multi-Catfish agents, or `catfish-r1` /
`catfish-r2` / `catfish-r3` learners were authorized or run.

## 1. Current State

Phase `05B` is planning-only. Original MODQN semantics stay fixed:

```text
r1 = throughput
r2 = handover penalty
r3 = load balance
```

Phase `05A` completed diagnostics but failed objective-buffer distinctness.
Phase `05R` passed only an offline buffer-admission redesign gate. No Phase
`05B` implementation, training, config generation, artifacts, or specialist
learners are authorized by this report.

## 2. Phase 05R Evidence Boundary

Use only the `guarded-residual-objective-admission` candidate. Evidence is
limited to offline replay diagnostics: bounded selective buffers, low pairwise
Jaccard, low scalar-buffer duplication, and simulated
`70 main + 10 r1 + 10 r2 + 10 r3` expected objective lift without significant
non-target damage.

This is not Multi-Catfish effectiveness evidence.

## 3. Phase 05B Planning Verdict

```text
ALLOW FUTURE PHASE 05B IMPLEMENTATION PROMPT DRAFT
BLOCK IMPLEMENTATION NOW
```

The minimal plan is valid only if it tests objective-specialized replay /
intervention against matched single Catfish under equal total Catfish budget.

## 4. Recommended Phase 05B Minimal Scope

Bounded pilot only:

1. `20` episodes, matching Phase `04C` bounded budget,
2. `3` matched seed triplets minimum, same triplets across all comparators,
3. evaluation seeds remain `[100, 200, 300, 400, 500]`,
4. eval cadence every `5` episodes and final,
5. checkpoints: final policy plus best weighted eval checkpoint,
6. batch / intervention budget: total Catfish share fixed at `0.30`.

No long training and no effectiveness claim from this pilot alone unless all
predeclared gates pass.

## 5. Allowed Code Surfaces

Future implementation may touch only opt-in Phase `05B` surfaces:

1. `src/modqn_paper_reproduction/config_loader.py`,
2. `src/modqn_paper_reproduction/runtime/trainer_spec.py`,
3. `src/modqn_paper_reproduction/runtime/catfish_replay.py`,
4. `src/modqn_paper_reproduction/algorithms/catfish_modqn.py` or a new sibling
   `multi_catfish_modqn.py`,
5. `src/modqn_paper_reproduction/orchestration/train_main.py` / `cli.py` only
   for explicit Phase `05B` routing,
6. new Phase `05B` analysis helpers,
7. new `tests/test_catfish_phase05b_*.py`.

## 6. Forbidden Code Surfaces

Do not mutate:

1. frozen baseline configs / artifacts,
2. `configs/modqn-paper-baseline*.yaml`,
3. `artifacts/pilot-02-best-eval/`, `run-9000/`, `table-ii-*`, `fig-*`,
4. environment / state / action / backbone semantics under `env/*`,
   `runtime/state_encoding.py`, or base MODQN behavior,
5. reward semantics or EE / RA-EE code paths,
6. Phase `05A` failed percentile-buffer construction as the starting design.

## 7. Config / Artifact Namespace

Use only:

```text
configs/catfish-modqn-phase-05b-*.resolved.yaml
artifacts/catfish-modqn-phase-05b-*/
```

Required planned configs:

1. `catfish-modqn-phase-05b-modqn-control`,
2. `catfish-modqn-phase-05b-single-catfish-equal-budget`,
3. `catfish-modqn-phase-05b-primary-multi-catfish-shaping-off`,
4. `catfish-modqn-phase-05b-multi-buffer-single-learner`,
5. `catfish-modqn-phase-05b-random-or-uniform-buffer-control`.

## 8. Comparator Design

Primary comparator: single Catfish-MODQN, shaping off, scalar Phase `04B`
high-value replay, total Catfish ratio `0.30`.

Also include:

1. matched MODQN control with Catfish disabled,
2. multi-buffer / single-learner comparator using the same guarded-residual
   buffers but only one Catfish learner,
3. random or uniform equal-budget buffer control to rule out extra data mixing.

## 9. Primary Run

Future primary `05B` run:

```text
main MODQN learner
+ catfish-r1 specialist replay/learner
+ catfish-r2 specialist replay/learner
+ catfish-r3 specialist replay/learner
```

Intervention mix:

```text
70% main + 10% r1 + 10% r2 + 10% r3
```

Primary run must keep competitive shaping off. Admission rules must be the
Phase `05R` guarded-residual rules with complete threshold ties only, no random
partial tie-breaking.

## 10. Ablations

Critical ablations:

1. multi-buffer / single-learner, same `70/10/10/10` mix,
2. no-intervention diagnostic, where buffers and specialists may train but main
   receives no Catfish samples,
3. random equal-budget buffer injection,
4. optional non-primary shaping-on branch only after shaping-off primary is
   complete.

## 11. Tests Required

Future implementation must add tests for:

1. config namespace gating and rejection of EE / reward changes,
2. total Catfish budget equality,
3. guarded-residual admission determinism and tie handling,
4. `70/10/10/10` mixed replay composition,
5. per-buffer source labels and sample counts,
6. baseline config / artifact non-mutation,
7. existing `test_catfish_phase04b.py` and
   `test_catfish_phase05a_multi_buffer.py` still passing.

## 12. Metrics / Diagnostics Required

Metrics:

1. scalar train / eval / final / best,
2. `r1`, `r2`, `r3`,
3. handover count,
4. AUC, episode-to-threshold, best-eval timing,
5. final-vs-best gap,
6. cross-seed variance,
7. TD loss, Q spikes, NaN, action collapse,
8. runtime / memory / agent-count cost.

Diagnostics:

1. per-buffer size / share / threshold / tie mass,
2. pairwise Jaccard and scalar-buffer Jaccard,
3. actual per-update replay composition,
4. per-buffer sample use in main updates,
5. specialist update counts / loss / Q maxima,
6. replay starvation and intervention skip counters,
7. non-target objective damage table.

## 13. Acceptance Criteria

To pass a future bounded `05B` pilot:

1. all comparators complete under identical budget / seeds / eval / checkpoint
   rules,
2. actual Catfish ratio stays near configured `0.30`, with per-buffer use
   visible,
3. Multi-Catfish beats single Catfish on more than scalar reward alone,
4. at least one component objective improves without significant non-target
   damage,
5. variance, final-vs-best gap, Q stability, and action diversity are not worse
   than single Catfish,
6. multi-buffer / single-learner and random-buffer controls do not explain away
   the gain.

## 14. Stop Conditions

Stop Phase `05B` if:

1. reward / state / action / backbone changes are needed,
2. EE or Catfish-EE framing appears,
3. primary result needs shaping-on,
4. any objective buffer becomes empty or all-samples,
5. actual intervention budget is unmatched,
6. replay starvation, NaN, Q divergence, or action collapse appears,
7. gains are scalar-only,
8. multi-buffer / single-learner matches full Multi-Catfish, making extra
   learners unnecessary.

## 15. Forbidden Claims

Do not claim:

1. Multi-Catfish effectiveness from Phase `05R`,
2. Catfish-EE-MODQN,
3. EE-MODQN or RA-EE continuation,
4. full paper-faithful reproduction,
5. physical energy saving,
6. scalar reward alone as success,
7. Phase `05B` implementation authorization from this planning draft.

## 16. Implementation Prompt Boundaries

A later implementation prompt must explicitly say Phase `05B` implementation
is authorized after review. It must name the exact configs, artifacts, bounded
run budget, tests, and whether training is authorized. Without that explicit
later prompt, only planning is allowed.

## 17. PASS / FAIL

```text
Phase 05B planning boundary: PASS
Future implementation prompt draft: ALLOWED AFTER REVIEW
Phase 05B implementation now: FORBIDDEN
Training now: FORBIDDEN
Artifact generation now: FORBIDDEN
Full Multi-Catfish agents now: FORBIDDEN
```
