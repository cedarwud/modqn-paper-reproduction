# Phase 01B Slice C: Targeted High-Load Follow-On SDD

**Status:** Proposed execution SDD  
**Depends on:**
- [`phase-01b-paper-faithful-follow-on-sdd.md`](./phase-01b-paper-faithful-follow-on-sdd.md)
- [`../../artifacts/phase-01b-slice-b-bounded-status-2026-04-14.md`](../../artifacts/phase-01b-slice-b-bounded-status-2026-04-14.md)

## 1. Purpose

This SDD defines the next bounded execution step after the first
Phase 01B Slice B follow-on evidence.

The goal is not to reopen broad retraining. The goal is to answer two
specific questions with the minimum additional work:

1. does the paper-backed scenario correction change behavior at the
   high-load end of `Fig. 3`, where the frozen baseline showed its
   largest local anomaly?
2. if not, should the next experiment move to reward geometry or to
   comparator protocol instead of more full sweeps?

## 2. Trigger

The current bounded follow-on status already shows:

1. the paper-backed scenario surface is now executable
2. the low-load bounded `Fig. 3` preview remains near-tied
3. the bounded `Table II` preview remains collapsed
4. reward-scale dominance still persists after the scenario correction

That means the next useful step is no longer a blind full sweep family.
It is a targeted decision gate.

## 3. Working Hypothesis

The most credible remaining scenario-backed signal is the high-load end
of `Fig. 3`.

Rationale:

1. the frozen baseline `Fig. 3` review reported its largest visible
   anomaly near `200` users
2. the bounded follow-on `Fig. 3` run only exercised the prefix
   `40, 60, 80`
3. that leaves the highest-load paper range endpoint untested on the
   corrected scenario surface

If the high-load end still stays near-tied, then scenario fidelity alone
is unlikely to be the main remaining bottleneck.

## 4. Scope

This execution SDD includes:

1. adding a clean point-override surface for figure sweeps so the repo
   can run non-prefix targeted subsets
2. running a targeted high-load `Fig. 3` follow-up on the corrected
   scenario surface
3. running a reward-geometry decision pass on the corrected follow-on
   `Table II` preview
4. recording one short decision note that says which branch should come
   next:
   - full follow-on sweep expansion
   - reward-geometry experiment
   - comparator-protocol experiment
   - negative-result stop

This execution SDD does **not** include:

1. a new `9000`-episode long run
2. a full follow-on `Fig. 4` to `Fig. 6` sweep family
3. silently redefining the frozen comparison baseline
4. claiming that the repo is already a paper-faithful baseline

## 5. Slice C1: Figure Point Override Surface

The current figure CLI only exposes `--max-figure-points`, which selects
the prefix of the configured point set.

That is insufficient for high-load validation because it cannot directly
target `160, 180, 200`.

Required work:

1. add a figure-point override input surface
   - CLI form is preferred:
     `--figure-points 160,180,200`
   - or an equivalent explicit config override surface
2. make the override apply only to figure suites
3. preserve the requested point set in the manifest and analysis outputs
4. add tests proving that non-prefix point selection works and does not
   silently mutate the default configured sweep set

Acceptance:

1. `fig-3` can be run on exactly the requested high-load points
2. the emitted manifest records those points explicitly
3. existing prefix-based behavior remains intact when no override is
   passed

## 6. Slice C2: Targeted High-Load Fig. 3 Follow-Up

After Slice C1 lands, run one targeted corrected-scenario `Fig. 3`
artifact with:

1. points: `160, 180, 200`
2. weight row: baseline `[0.5, 0.3, 0.2]`
3. methods: `MODQN`, `DQN_throughput`, `DQN_scalar`, `RSS_max`
4. reference run: `artifacts/scenario-corrected-pilot-01/`

Episode budget:

1. first pass: `20` episodes per learned method run
2. optional confirmation pass at `50` episodes only if the first pass
   shows materially different behavior from the frozen baseline

Required outputs:

1. machine-readable `fig-3` JSON and CSV files
2. weighted winners CSV
3. review note
4. one short comparison against:
   - `artifacts/fig-3-pilot-01/`
   - `artifacts/scenario-corrected-fig-3-bounded-01/`

## 7. Slice C3: Reward-Geometry / Comparator-Protocol Decision Gate

Run one decision-oriented analysis pass after the targeted high-load
`Fig. 3` artifact exists.

Required work:

1. run `reward-geometry` analysis on
   `artifacts/scenario-corrected-table-ii-bounded-01/`
2. inspect whether normalization scenarios materially change:
   - winner identity
   - scalar spread
   - tie count
3. write one short decision note that classifies the next likely
   bottleneck as one of:
   - reward geometry
   - comparator protocol
   - neither, because the corrected scenario already opens separation

Comparator-protocol work is only promoted if the reward-geometry pass
does not provide a stronger explanation.

## 8. Promotion Rules

Promotion to a broader follow-on sweep family is allowed only if the
targeted high-load `Fig. 3` artifact shows a materially stronger signal
than the current bounded follow-on note.

For this SDD, "materially stronger signal" means all of:

1. `MODQN` is sole-best on at least `2/3` high-load points
2. max weighted-reward spread across methods is at least `0.05`
3. at least one high-load point shows non-trivial cross-method
   separation outside `r2` / handover alone

If those conditions are not met, do **not** promote directly to a full
follow-on sweep family.

## 9. Negative-Result Rule

Negative results are valid outcomes.

If the targeted high-load `Fig. 3` follow-up still shows near-ties and
the reward-geometry pass does not materially change the interpretation,
the repo should record that:

1. scenario fidelity was corrected on the highest-impact disclosed
   mismatch
2. the near-tie structure survived that correction
3. the project remains best interpreted as:
   - a disclosed comparison baseline
   - plus a paper-faithful follow-on experiment that did not recover the
     claimed separation

## 10. Deliverables

This execution SDD is complete when the repo contains:

1. one merged implementation for figure-point overrides
2. one reviewed high-load corrected-scenario `Fig. 3` artifact
3. one reviewed reward-geometry decision artifact for the corrected
   `Table II` preview
4. one short status note that says which branch is next and why

## 11. Recommended Thread Boundary

After this SDD lands, implementation should preferably continue in a new
conversation.

Reason:

1. the current thread contains exploratory analysis, older baseline
   closeout context, and intermediate follow-on conclusions
2. the next work should be driven by this SDD plus the latest status
   notes rather than by the full historical discussion
3. the repo now contains enough authority surfaces that a clean
   implementation thread can start from file-backed context
