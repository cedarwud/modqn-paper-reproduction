# Phase 01C: Comparator-Protocol Experiment SDD

**Status:** Proposed experimental SDD  
**Depends on:**
- [`phase-01b-paper-faithful-follow-on-sdd.md`](./phase-01b-paper-faithful-follow-on-sdd.md)
- [`phase-01b-slice-c-targeted-high-load-follow-on-sdd.md`](./phase-01b-slice-c-targeted-high-load-follow-on-sdd.md)
- [`../../artifacts/phase-01b-closeout-status-2026-04-14.md`](../../artifacts/phase-01b-closeout-status-2026-04-14.md)
- [`../../artifacts/public-summary-2026-04-14-phase-01b-closeout.md`](../../artifacts/public-summary-2026-04-14-phase-01b-closeout.md)
- [`../assumptions/modqn-reproduction-assumption-register.md`](../assumptions/modqn-reproduction-assumption-register.md)

## 1. Purpose

Phase 01C opens the only currently defensible follow-on branch after the
Phase 01B scenario-correction closeout.

Its purpose is narrow:

1. test whether comparator or evaluation protocol choices, rather than
   scenario fidelity or reward geometry, are still suppressing visible
   method separation
2. do that with the minimum auditable work needed to decide whether any
   protocol factor is worth further implementation
3. stop quickly if the protocol factors do not materially change the
   interpretation

Phase 01C is **not** a license to resume broad reproduction work.
The default outcome remains `stop unless a bounded protocol factor shows
real movement`.

## 2. Trigger

The trigger for this SDD is already recorded in the latest authority
artifacts:

1. the frozen comparison baseline is complete and should remain intact
2. the paper-backed scenario correction is complete and ended as a
   `negative-result stop`
3. reward-geometry re-scoring did not change winner identity or tie
   structure
4. the remaining defensible question is therefore whether the comparison
   protocol itself still contains a meaningful gap

If no further research is desired, stopping now is still fully
defensible.

If research continues anyway, it should do so only under this new
explicitly labeled `comparator-protocol` branch.

## 3. Scope

### 3.1 In Scope

Phase 01C includes:

1. auditing the currently fixed comparator and evaluation protocol
   surfaces
2. distinguishing paper-backed comparison facts from repo-fixed
   assumptions
3. using preserved artifacts and checkpoints first, before proposing any
   new training work
4. bounded replay or re-evaluation passes that isolate protocol
   variables without reopening scenario or reward changes
5. at most one small follow-on implementation slice if replay-only
   evidence shows a protocol factor is materially relevant

### 3.2 Out Of Scope

Phase 01C does **not** include:

1. reopening Phase 01B scenario-fidelity work
2. reopening reward geometry or reward-calibration as the main branch
3. expanding to full follow-on `Fig. 4` to `Fig. 6` sweeps
4. starting a new `9000`-episode long run
5. silently redefining the frozen comparison baseline
6. adding new benchmark families beyond the paper-backed comparator set
7. changing core environment, orbit, mobility, or reward equations

## 4. Paper-Backed Facts Vs Repo-Fixed Assumptions

### 4.1 Paper-Backed Comparator Facts

The paper source chain currently fixes these comparison facts:

1. comparator set:
   - `RSS_max`
   - `DQN_throughput`
   - `DQN_scalar`
2. the primary MODQN claim is one-time multi-objective training with
   post-training adaptation across preference weights
3. `DQN_throughput` is a throughput-only DQN comparator
4. `DQN_scalar` is a weighted scalar-reward DQN comparator
5. `Table II` uses the paper-disclosed weight rows
6. `Fig. 3` to `Fig. 6` use the paper-disclosed parameter ranges
7. the default figure interpretation uses weight
   `[0.5, 0.3, 0.2]`

The paper does **not** fully fix:

1. seed domains
2. evaluation aggregation mechanics
3. checkpoint reporting or checkpoint-selection rules
4. whether model selection and final reporting use the same seed set
5. exact executable bookkeeping for comparator retraining

### 4.2 Repo-Fixed Comparator And Evaluation Assumptions

The current repo has already frozen these protocol surfaces for the
comparison baseline:

1. `ASSUME-MODQN-REP-010`
   - evaluation aggregation is `mean-and-std` over
     `evaluation_seed_set`
2. `ASSUME-MODQN-REP-015`
   - checkpoint rule discloses:
     - primary report: `final-episode-policy`
     - secondary report: `best-weighted-reward-on-eval`
3. `ASSUME-MODQN-REP-016`
   - figure point sets are the resolved discrete executable points, not
     only the paper range endpoints
4. `ASSUME-MODQN-REP-017`
   - `DQN_scalar` retrains per weight row
5. `ASSUME-MODQN-REP-018`
   - fixed split seed policy:
     - train `42`
     - environment `1337`
     - mobility `7`
     - evaluation `{100, 200, 300, 400, 500}`
6. `ASSUME-MODQN-REP-011`
   - satellite-count changes use `per-topology retrain`

### 4.3 Current Implemented Protocol Surface

The current executable sweep surface further fixes:

1. `MODQN` is trained once per topology family and evaluated across
   `Table II` weight rows
2. `DQN_throughput` is trained once per executed sweep surface and then
   reused across reporting rows when the surface allows that reuse
3. `DQN_scalar` retrains per reporting weight row
4. `RSS_max` has no training checkpoint and is evaluated directly
5. learned-method sweep helpers restore the best-eval checkpoint when it
   exists, and emitted artifacts disclose the resulting
   `checkpoint_kind`
6. figure sweeps use the resolved baseline row `[0.5, 0.3, 0.2]`
7. Phase 01B follow-on artifacts kept the same seed and checkpoint
   disclosure surfaces rather than inventing a new comparator protocol

These are the repo-fixed baseline surfaces. Phase 01C may inspect them,
but it must not silently rewrite them.

## 5. Working Hypothesis

After scenario fidelity and reward geometry, the most plausible
remaining explanation is now protocol-level rather than model-level.

The working hypothesis is:

1. one or more comparator or evaluation protocol choices may be
   compressing a weak but real method difference
2. the most plausible remaining candidates are checkpoint-selection and
   reporting protocol, seed reuse between model selection and reporting,
   and aggregation strategy
3. if those protocol factors still do not move winner identity or tie
   structure, the repo should stop rather than escalate training

The null hypothesis is explicit:

1. current near-ties are robust to reasonable protocol changes
2. if so, the repo remains best interpreted as:
   - a disclosed comparison baseline
   - plus a negative-result scenario-correction follow-on

## 6. Candidate Protocol Factors

### 6.1 Highest-Priority Factors Worth Minimal Experiment

1. `CP-01` selection/reporting seed coupling
   - current best-eval checkpointing uses the declared evaluation seed
     set
   - the same seed set is also the current reporting surface
   - this is the cleanest high-value candidate because it may create
     selection-on-reporting leakage
2. `CP-02` final-vs-best checkpoint reporting
   - the paper does not fully specify whether reported comparisons use
     final policies, best observed policies, or both
   - the repo currently discloses both surfaces but sweep helpers
     preferentially restore best-eval checkpoints when available
3. `CP-03` aggregation sensitivity
   - current reporting uses five-seed mean and standard deviation
   - a bounded per-seed or leave-one-out audit may show whether the
     mean-only surface is hiding or overstating weak separation
4. `CP-04` comparator-specific reporting/disclosure surface
   - `DQN_throughput` is a throughput-only comparator, and its trainer
     currently selects best-eval checkpoints on its own throughput-only
     scalarization rather than on the baseline weighted reward
   - the narrower remaining question is whether the cross-comparator
     reporting surface, checkpoint labeling, or weighted-first summary
     presentation is still compressing a weak objective-aligned
     difference
   - this is lower-confidence than `CP-01` to `CP-03` and should only
     open if replay-only evidence points at a reporting-surface
     bottleneck

### 6.2 Lower-Priority Factors That Should Wait

These are allowed only if the highest-priority factors show real
movement first:

1. wider train-seed replication
2. alternative evaluation seed-cardinality studies
3. comparator-specific retraining-budget equalization
4. figure-family expansion beyond the minimal bounded surfaces

### 6.3 Factors Not Worth Opening In Phase 01C

These should stay closed:

1. new scenario geography or mobility changes
2. reward geometry or reward-calibration branches
3. new `9000`-episode long runs
4. new benchmark families not named in the paper
5. changing discrete figure point sets just to hunt for better-looking
   results
6. broad `Fig. 4` to `Fig. 6` follow-on sweeps
7. reinterpreting donor notes or external reports as new authority

## 7. Priority Order

Phase 01C should run in this order:

1. `CP-01` selection/reporting seed coupling
2. `CP-02` final-vs-best checkpoint reporting
3. `CP-03` aggregation sensitivity
4. `CP-04` comparator-specific reporting/disclosure surface
5. stop unless one of the above shows materially different comparison
   behavior

This order is mandatory because:

1. the first three factors can be answered largely from preserved
   artifacts and replay
2. they are cheaper and more auditable than new retraining
3. they directly test the current comparison surface rather than adding
   more model variance

## 8. Experiment-Only Variables

Phase 01C may vary only these explicit protocol variables:

1. reporting checkpoint policy:
   - `final-only`
   - `best-eval-on-selection-seeds`
   - side-by-side `final-vs-best`
2. evaluation seed role:
   - current selection/reporting seed set
   - one disjoint held-out reporting seed set fixed for all Phase 01C
     work: `{600, 700, 800, 900, 1000}`
3. aggregation mode:
   - current `mean-and-std`
   - per-seed winner table
   - leave-one-out stability check
4. comparator-specific reporting metric:
   - current weighted reward
   - explicitly labeled objective-aligned diagnostic views

Phase 01C must not substitute a different held-out reporting seed set
without a superseding SDD or decision note that explicitly replaces this
seed domain.

Phase 01C may **not** vary:

1. scenario fields
2. reward equations
3. reward normalization as a promoted interpretation surface
4. long-run budget
5. paper-backed comparator membership

## 9. Minimal Execution Slices

### 9.1 Slice A: Protocol Inventory And Artifact Audit

Goal:

1. produce a file-backed inventory of which preserved artifacts contain:
   - final checkpoints
   - best-eval checkpoints
   - evaluation seed disclosure
   - enough metadata to replay comparison results

Required outputs:

1. one protocol inventory note
2. one artifact availability matrix
3. one recommendation note saying whether replay-only work is possible
   without retraining

Acceptance:

1. every tested protocol factor is mapped to a source file or artifact
2. the audit states exactly which surfaces are baseline-fixed versus
   experiment-only
3. the next slice can be chosen without guessing about checkpoint
   availability

### 9.2 Slice B: Checkpoint And Seed-Coupling Replay

Slice B is replay-first and should only use preserved artifacts if Slice
A says the checkpoints exist.

Goal:

1. test `CP-01` and `CP-02` without new training

Minimal bounded surfaces:

1. `Table II` triage rows:
   - `1.0/1.0/1.0`
   - `1.0/0.0/0.0`
   - `0.0/1.0/0.0`
2. `Fig. 3` high-load points:
   - `160`
   - `180`
   - `200`

Required comparisons:

1. final checkpoint on current reporting seed set
2. best-eval checkpoint on current reporting seed set
3. final checkpoint on one disjoint held-out reporting seed set
4. best-eval checkpoint on one disjoint held-out reporting seed set

Frozen held-out reporting seeds for all Slice B replay:

1. `{600, 700, 800, 900, 1000}`

Required outputs:

1. one replay matrix manifest covering every exercised checkpoint kind
   and reporting-seed role
2. one machine-readable comparison table for the bounded `Table II`
   surfaces
3. one machine-readable comparison table for the bounded high-load
   `Fig. 3` surfaces
4. one short review note with an explicit `stop` or `promote` decision

Acceptance:

1. winner identity and tie structure are recorded for every protocol
   mode
2. each output records:
   - checkpoint kind
   - selection seed set
   - reporting seed set
   - source artifact
   - bounded surface identifier
3. the held-out reporting seed set is exactly
   `{600, 700, 800, 900, 1000}`
4. no new training run is started in this slice

### 9.3 Slice C: Aggregation Sensitivity Audit

Slice C only starts if Slice B is completed.

Goal:

1. test whether the current five-seed mean surface is hiding a stable
   per-seed separation signal

Required outputs:

1. per-seed winner table
2. leave-one-out winner table
3. short review note stating whether aggregation alone changes the
   interpretation

Acceptance:

1. the analysis is produced from the same bounded surfaces as Slice B
2. the note states whether any apparent separation survives across seed
   subsets rather than appearing only as a one-off winner flip

### 9.4 Slice D: Comparator-Reporting Micro-Experiment

Slice D is optional and should open only if Slice B or C shows
meaningful movement.

Goal:

1. test `CP-04` on one explicit experimental surface if replay-only
   evidence suggests the current reporting/disclosure surface is the
   real bottleneck

Allowed work:

1. one new explicitly labeled comparator-protocol config surface
2. one bounded micro-run on the same triage surfaces as Slice B
3. no full-sweep promotion from this slice alone

Required outputs:

1. one manifest naming:
   - the exact comparator-specific protocol variable under test
   - the checkpoint-selection rule used during training
   - the reporting metric shown in the output tables
2. one machine-readable comparison table for the exercised bounded
   surfaces
3. one review note stating whether `CP-04` justified any further branch

Recommended episode budget:

1. default: `20`
2. optional confirmation: `50`
3. never `9000`

Acceptance:

1. Slice D opens only if Slice B or C recorded material movement that
   specifically points to a reporting/disclosure bottleneck
2. all outputs explicitly distinguish training-time checkpoint-selection
   metric from reporting-time comparison metric
3. the exercised surfaces remain bounded to the same `Table II` rows and
   `Fig. 3` points as Slice B
4. the slice records an explicit `stop` or `promote-one-bounded-slice`
   decision

## 10. Promotion Rules

Promotion beyond replay-only work is allowed only if at least one
highest-priority factor does one of the following on the bounded
surfaces:

1. changes winner identity on at least `2` exercised rows or points
2. reduces exact-tie count by at least `2`
3. makes `MODQN` sole-best on at least `2` exercised rows or points
   where the baseline protocol previously reported a tie or loss
4. reveals that held-out reporting reverses the in-sample checkpoint
   conclusion in a way that makes the current comparator interpretation
   unstable

If those conditions are not met, do **not** promote to new
implementation or wider sweeps.

## 11. Stop Rules

Phase 01C should stop immediately if any of the following is true:

1. Slice A shows the preserved artifacts are insufficient for replay and
   the missing work would require broad retraining rather than one
   bounded protocol slice
2. Slice B shows that final-vs-best and held-out-vs-in-sample reporting
   do not materially change winner identity or tie structure
3. Slice C shows that apparent winner flips are only unstable seed noise
   rather than persistent method separation
4. the only observed differences remain confined to `r2` / handover and
   still do not open separation in `r1` or `r3`

The stop conclusion should be recorded as:

1. Phase 01C comparator-protocol check completed
2. no protocol factor justified reopening broader reproduction work

## 12. Artifact Requirements

Every executed Phase 01C decision slice must preserve:

1. one manifest recording:
   - protocol factor under test
   - source artifact or config
   - selection seed set
   - reporting seed set
   - checkpoint kind
   - bounded point or weight-row set
2. machine-readable comparison tables when the slice compares outcomes
   across protocol modes
3. one review note with an explicit stop or promote decision
4. stable references back to the frozen baseline bundle and, when used,
   the Phase 01B negative-result artifacts

Slice A may satisfy this rule with the protocol inventory note and
artifact availability matrix instead of comparison tables.

Review-only prose without the required file-backed outputs is not
sufficient.

## 13. Acceptance Gates

### Gate 1: Protocol Surface Inventory Closed

Gate 1 is closed only when:

1. paper-backed facts are separated from repo-fixed assumptions
2. the tested protocol factors are named explicitly
3. preserved artifact availability is recorded

### Gate 2: Replay-Only Decision Pass Closed

Gate 2 is closed only when:

1. Slice B is complete
2. checkpoint-kind and seed-role differences are file-backed
3. the repo records whether current comparison conclusions are robust to
   those changes

### Gate 3: Comparator Branch Outcome Declared

Gate 3 is closed only when the repo records one of these outcomes:

1. comparator protocol shows a real enough effect to justify one bounded
   implementation slice
2. comparator protocol does not materially change the interpretation, so
   the project stops with the current disclosed baseline plus negative
   follow-on evidence

## 14. Completion Boundary

Phase 01C is complete when:

1. the currently fixed comparator protocol has been audited
2. the highest-priority protocol factors have been tested on bounded
   surfaces
3. the repo records a clear `promote` or `stop` decision
4. the frozen comparison baseline and the closed Phase 01B follow-on both
   remain separately interpretable

Phase 01C completion does **not** mean that the repo has recovered a
full paper-faithful reproduction claim.
It only means the last currently defensible protocol-level question has
been tested without reopening broader retraining.
