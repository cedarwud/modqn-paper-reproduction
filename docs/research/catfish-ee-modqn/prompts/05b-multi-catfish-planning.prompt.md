# Prompt: Phase 05B Multi-Catfish Planning Draft

Historical prompt note:

Phase `05B` planning has been completed and passed in
`05b-multi-catfish-planning.execution-report.md`. Do not rerun this prompt by
default. The next prompt is
`05b-multi-catfish-implementation-draft.prompt.md`, but it is not executable
unless the user explicitly authorizes Phase `05B` implementation / bounded
training.

請只做 Phase `05B` multi-Catfish planning draft。不要實作、不要改程式、
不要跑訓練、不要啟動 full Multi-Catfish agents。

請先讀：

1. `AGENTS.md`
2. `docs/research/catfish-ee-modqn/development-guardrails.md`
3. `docs/research/catfish-ee-modqn/00-validation-master-plan.md`
4. `docs/research/catfish-ee-modqn/execution-handoff.md`
5. `docs/research/catfish-ee-modqn/05-multi-catfish-modqn-validation.md`
6. `docs/research/catfish-ee-modqn/04c-single-catfish-ablation-attribution.execution-report.md`
7. `docs/research/catfish-ee-modqn/05a-multi-buffer-validation.execution-report.md`
8. `docs/research/catfish-ee-modqn/05r-objective-buffer-redesign-gate.execution-report.md`

Current gate status:

```text
Phase 05A bounded diagnostic completion: PASS
Phase 05A objective-buffer distinctness: FAIL
Phase 05R objective-buffer admission redesign diagnostics: PASS
Phase 05B planning draft: ALLOWED
Phase 05B implementation now: FORBIDDEN
Full Multi-Catfish agents now: FORBIDDEN
```

Phase `05R` candidate to plan from:

```text
guarded-residual-objective-admission
```

Fixed original MODQN reward semantics:

```text
r1 = throughput
r2 = handover penalty
r3 = load balance
```

Planning goal:

Define the minimum bounded Phase `05B` implementation plan that could later
test whether objective-specialized Multi-Catfish adds value beyond single
Catfish under equal budget and unchanged original MODQN semantics.

This prompt authorizes only a planning draft. It does not authorize coding,
training, config generation, artifact generation, or `catfish-r1` / `catfish-r2`
/ `catfish-r3` learners in this run.

The Phase `05B` plan must define:

1. allowed code surfaces,
2. forbidden code surfaces,
3. config namespace,
4. artifact namespace,
5. primary comparator,
6. primary run,
7. critical ablations,
8. tests required,
9. metrics required,
10. diagnostics required,
11. acceptance criteria,
12. stop conditions,
13. forbidden claims,
14. implementation prompt boundaries.

Minimum planning constraints:

1. use original MODQN state, action, reward, and backbone unchanged,
2. keep frozen baseline configs and artifacts unchanged,
3. use new `catfish-modqn-phase-05b-*` config and artifact namespaces only,
4. keep total Catfish intervention budget matched against single Catfish,
5. use the Phase `05R` guarded-residual admission rules as the proposed buffer
   admission candidate,
6. include matched MODQN and single-Catfish comparators,
7. include a multi-buffer / single-learner comparator if needed to separate
   replay-admission value from extra learner capacity,
8. keep primary run shaping-off unless explicitly isolated as a non-primary
   ablation,
9. predeclare bounded pilot budget, seeds, checkpoint rule, and eval cadence,
10. require diagnostics for actual replay composition and per-buffer sample use.

Forbidden planning outcomes:

1. do not authorize Phase `05B` implementation in this same run,
2. do not authorize long training,
3. do not introduce EE reward or objective,
4. do not claim Catfish-EE-MODQN,
5. do not claim Multi-Catfish effectiveness from Phase `05R`,
6. do not use scalar reward alone as success evidence,
7. do not mutate frozen baseline configs or artifacts,
8. do not change original MODQN reward semantics,
9. do not start from the failed Phase `05A` percentile buffer construction.

Required output:

1. Current State
2. Phase 05R Evidence Boundary
3. Phase 05B Planning Verdict
4. Recommended Phase 05B Minimal Scope
5. Allowed Code Surfaces
6. Forbidden Code Surfaces
7. Config / Artifact Namespace
8. Comparator Design
9. Primary Run
10. Ablations
11. Tests Required
12. Metrics / Diagnostics Required
13. Acceptance Criteria
14. Stop Conditions
15. Forbidden Claims
16. Implementation Prompt Boundaries
17. PASS / FAIL

`PASS` means only that a future implementation prompt may be drafted after this
planning output is reviewed. It does not authorize implementation in the same
run.

`FAIL` means do not draft a Phase `05B` implementation prompt; either revise
the planning boundary or close Multi-Catfish as a paper boundary finding.
