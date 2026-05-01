# Draft Prompt: Phase 05B Multi-Catfish Bounded Pilot Implementation

Historical prompt note:

Phase `05B` bounded implementation has been executed and reported in
`05b-multi-catfish-bounded-pilot.execution-report.md`. Runnable evidence passed,
but acceptance / effectiveness failed. Do not rerun this prompt by default.

**Draft status:** not executable by default. Use this prompt only if the user
explicitly authorizes Phase `05B` implementation after reviewing
`05b-multi-catfish-planning.execution-report.md`.

Do not use this prompt for planning-only work. Do not use it to introduce EE,
Catfish-EE-MODQN, long training, or frozen-baseline mutation.

## Authorization Required

Before doing any code change or run, confirm the active user request explicitly
authorizes:

```text
Phase 05B bounded implementation
Phase 05B bounded training
new catfish-modqn-phase-05b configs/artifacts
```

If that authorization is absent, stop and report that only planning is allowed.

## Required Reading

Read first:

1. `AGENTS.md`
2. `docs/research/catfish-ee-modqn/development-guardrails.md`
3. `docs/research/catfish-ee-modqn/00-validation-master-plan.md`
4. `docs/research/catfish-ee-modqn/execution-handoff.md`
5. `docs/research/catfish-ee-modqn/05-multi-catfish-modqn-validation.md`
6. `docs/research/catfish-ee-modqn/04c-single-catfish-ablation-attribution.execution-report.md`
7. `docs/research/catfish-ee-modqn/05a-multi-buffer-validation.execution-report.md`
8. `docs/research/catfish-ee-modqn/05r-objective-buffer-redesign-gate.execution-report.md`
9. `docs/research/catfish-ee-modqn/05b-multi-catfish-planning.execution-report.md`

## Fixed Boundary

Original MODQN reward semantics remain fixed:

```text
r1 = throughput
r2 = handover penalty
r3 = load balance
```

Use only Phase `05R` guarded-residual objective admission:

```text
guarded-residual-objective-admission
complete threshold ties only
no random partial tie-breaking
```

Primary intervention mix:

```text
70% main + 10% r1 + 10% r2 + 10% r3
total Catfish share = 0.30
```

Primary run must keep competitive shaping off.

## Allowed Code Surfaces

Touch only opt-in Phase `05B` surfaces:

1. `src/modqn_paper_reproduction/config_loader.py`,
2. `src/modqn_paper_reproduction/runtime/trainer_spec.py`,
3. `src/modqn_paper_reproduction/runtime/catfish_replay.py`,
4. `src/modqn_paper_reproduction/algorithms/catfish_modqn.py` or new sibling
   `src/modqn_paper_reproduction/algorithms/multi_catfish_modqn.py`,
5. `src/modqn_paper_reproduction/orchestration/train_main.py` / `cli.py` only
   for explicit Phase `05B` routing,
6. new Phase `05B` analysis helpers,
7. new `tests/test_catfish_phase05b_*.py`.

## Forbidden Code Surfaces

Do not mutate:

1. `configs/modqn-paper-baseline*.yaml`,
2. frozen baseline artifacts such as `artifacts/pilot-02-best-eval/`,
   `artifacts/run-9000/`, `artifacts/table-ii-*`, and `artifacts/fig-*`,
3. environment / state / action / backbone semantics under `env/*`,
   `runtime/state_encoding.py`, or base MODQN behavior,
4. reward semantics,
5. EE / RA-EE code paths,
6. Phase `05A` failed percentile-buffer construction as the starting design.

## Required Configs

Use only:

```text
configs/catfish-modqn-phase-05b-*.resolved.yaml
```

Create exactly scoped configs for:

1. `catfish-modqn-phase-05b-modqn-control`,
2. `catfish-modqn-phase-05b-single-catfish-equal-budget`,
3. `catfish-modqn-phase-05b-primary-multi-catfish-shaping-off`,
4. `catfish-modqn-phase-05b-multi-buffer-single-learner`,
5. `catfish-modqn-phase-05b-random-or-uniform-buffer-control`.

## Artifact Namespace

Use only:

```text
artifacts/catfish-modqn-phase-05b-*/
```

Do not write into any baseline artifact directory.

## Bounded Run Budget

Use the bounded pilot only:

1. `20` episodes,
2. `3` matched seed triplets minimum,
3. identical seed triplets across all comparators,
4. evaluation seeds `[100, 200, 300, 400, 500]`,
5. eval cadence every `5` episodes and final,
6. final policy plus best weighted eval checkpoint.

Do not run long training.

## Required Comparators

Primary comparator:

1. single Catfish-MODQN,
2. shaping off,
3. scalar Phase `04B` high-value replay,
4. total Catfish ratio `0.30`.

Also implement:

1. matched MODQN control with Catfish disabled,
2. multi-buffer / single-learner comparator using guarded-residual buffers,
3. random or uniform equal-budget buffer control.

## Required Ablations

1. multi-buffer / single-learner, same `70/10/10/10` mix,
2. no-intervention diagnostic where buffers and specialists may train but main
   receives no Catfish samples,
3. random equal-budget buffer injection,
4. optional shaping-on branch only after shaping-off primary is complete.

## Required Tests

Add and run focused tests for:

1. config namespace gating and rejection of EE / reward changes,
2. total Catfish budget equality,
3. guarded-residual admission determinism and tie handling,
4. `70/10/10/10` mixed replay composition,
5. per-buffer source labels and sample counts,
6. baseline config / artifact non-mutation,
7. existing `test_catfish_phase04b.py` and
   `test_catfish_phase05a_multi_buffer.py` still passing.

## Required Metrics And Diagnostics

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

## Acceptance Criteria

Bounded `05B` pilot may pass only if:

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

## Stop Conditions

Stop if:

1. reward / state / action / backbone changes are needed,
2. EE or Catfish-EE framing appears,
3. primary result needs shaping-on,
4. any objective buffer becomes empty or all-samples,
5. actual intervention budget is unmatched,
6. replay starvation, NaN, Q divergence, or action collapse appears,
7. gains are scalar-only,
8. multi-buffer / single-learner matches full Multi-Catfish, making extra
   learners unnecessary.

## Forbidden Claims

Do not claim:

1. Multi-Catfish effectiveness from Phase `05R`,
2. Catfish-EE-MODQN,
3. EE-MODQN or RA-EE continuation,
4. full paper-faithful reproduction,
5. physical energy saving,
6. scalar reward alone as success,
7. full Multi-Catfish effectiveness from a bounded `20`-episode pilot unless
   all predeclared gates pass and claim scope remains bounded.

## Required Output

Return:

1. changed files,
2. configs created,
3. artifacts created,
4. tests run,
5. bounded runs completed,
6. metrics / diagnostics summary,
7. deviations from this prompt,
8. PASS / FAIL for bounded runnable evidence,
9. explicit statement that no Catfish-EE / EE / full paper-faithful claim was
   made.
