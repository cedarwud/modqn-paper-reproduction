# Prompt: Phase 05R Objective-Buffer Redesign Gate

Historical prompt note:

Phase `05R` has been executed and passed in
`05r-objective-buffer-redesign-gate.execution-report.md`. Do not rerun this
prompt by default. Use `05b-multi-catfish-planning.prompt.md` for the next
planning-only step unless the user explicitly asks to re-audit Phase `05R`.

請做 Phase `05R` objective-buffer-redesign gate。不要實作 full multi-Catfish，
不要跑訓練，不要開始 Phase `05B`。

請先讀：

1. `AGENTS.md`
2. `docs/research/catfish-ee-modqn/development-guardrails.md`
3. `docs/research/catfish-ee-modqn/00-validation-master-plan.md`
4. `docs/research/catfish-ee-modqn/execution-handoff.md`
5. `docs/research/catfish-ee-modqn/05-multi-catfish-modqn-validation.md`
6. `docs/research/catfish-ee-modqn/05a-multi-buffer-validation.execution-report.md`
7. `docs/research/catfish-ee-modqn/04c-single-catfish-ablation-attribution.execution-report.md`

Current disposition:

```text
Phase 05A bounded diagnostic completion: PASS
Objective-specific buffer distinctness: FAIL
Phase 05B full multi-agent validation planning: BLOCK
Full multi-Catfish agents now: FORBIDDEN
```

Phase `05R` exists only to answer:

```text
Can objective-buffer admission be redesigned so r1/r2/r3 buffers are selective,
distinct, and interpretable without changing original MODQN semantics?
```

Fixed reward semantics:

```text
r1 = throughput
r2 = handover penalty
r3 = load balance
```

Allowed scope:

1. analysis-first redesign of buffer admission / ranking / threshold / tie
   handling,
2. use existing Phase `05A` transition-level samples / diagnostics where
   sufficient,
3. add bounded diagnostics only if existing artifacts lack required fields,
4. simulate fixed total intervention-budget composition only, for example
   `70 main + 10 r1 + 10 r2 + 10 r3`,
5. produce a Phase `05R` redesign-gate report.

Forbidden scope:

1. no long training,
2. no full multi-Catfish agents,
3. no `catfish-r1`, `catfish-r2`, or `catfish-r3` learners,
4. no frozen baseline config or artifact mutation,
5. no original MODQN reward / state / action / backbone changes,
6. no EE reward, EE objective, EE-MODQN, RA-EE continuation, or
   Catfish-EE-MODQN claim,
7. no random tie-breaking to manufacture distinctness,
8. no scalar reward alone as success evidence.

Required diagnostics:

1. admission share per buffer,
2. thresholds and threshold direction per buffer,
3. tie mass and unique value count per objective,
4. pairwise Jaccard: `r1/r2`, `r1/r3`, `r2/r3`,
5. Jaccard versus scalar Phase `04` high-value replay,
6. each buffer's `r1` / `r2` / `r3` distributions,
7. distinct samples versus the union of scalar and other objective buffers,
8. non-target degradation table,
9. fixed-budget intervention-composition simulation,
10. explicit stop-condition report.

Acceptance criteria for allowing later Phase `05B` planning:

1. every objective buffer is a bounded subset, not all samples and not empty,
2. admission share targets are predeclared and plausibly bounded, such as
   `0.10` to `0.30`,
3. `r1` does not duplicate scalar/high-throughput replay under a predeclared
   Jaccard threshold,
4. `r2` coarse/tie degeneration is resolved by a non-arbitrary rule,
5. every objective buffer contributes measurable unique samples to the
   simulated intervention mix,
6. target-objective lift is visible without significant non-target damage,
7. the result is explainable as objective-specialized replay, not random extra
   data mixing,
8. success evidence does not depend on scalar reward alone.

Stop conditions:

1. any buffer admits all samples or no samples,
2. `r1` remains a scalar/high-throughput duplicate,
3. `r2` still degenerates because of coarse/tied values,
4. `r3` distinctness comes only from throughput collapse or non-target damage,
5. distinctness requires changing reward semantics, state, action, backbone, or
   adding EE,
6. full specialist learners are needed just to determine whether buffers are
   distinct,
7. the result can only be explained by scalar reward,
8. frozen baseline config or artifact mutation would be required.

Required output:

1. Current State
2. Phase 05A Disposition
3. Phase 05R Redesign Candidate
4. Diagnostics Results
5. Acceptance Criteria Check
6. Stop Conditions
7. Whether Phase 05B Planning Is Allowed
8. Forbidden Claims
9. PASS / FAIL

PASS means only that a later Phase `05B` planning prompt may be drafted. It
does not authorize full multi-Catfish implementation in the same run.

FAIL means close the current Multi-Catfish route or return to paper boundary
finding unless a new design gate is explicitly approved.
