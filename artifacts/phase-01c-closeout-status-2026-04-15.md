# Phase 01C Closeout Status

Date: `2026-04-15`

## Scope Anchors

- prior repo-level closeout:
  - `artifacts/phase-01b-closeout-status-2026-04-14.md`
  - `artifacts/public-summary-2026-04-14-phase-01b-closeout.md`
- Phase 01C planning:
  - `docs/phases/phase-01c-comparator-protocol-experiment-sdd.md`
- Phase 01C execution evidence:
  - `artifacts/phase-01c-slice-a-protocol-inventory-2026-04-14/`
  - `artifacts/phase-01c-protocol-bounded-03/`

## Executive Status

Phase 01C is now closed.

It does **not** reopen the repo into a broader reproduction branch.
Instead, it closes the remaining comparator-protocol question as another
negative result.

The repo should now be interpreted as:

1. a working disclosed comparison baseline
2. plus a completed scenario-correction follow-on that ended as a
   negative result
3. plus a completed comparator-protocol follow-on that also ended as a
   negative result

It should still **not** be interpreted as a fully established
paper-faithful reproduction.

## What Phase 01C Checked

Phase 01C asked whether the remaining near-tie behavior could be an
artifact of comparator or evaluation protocol rather than of scenario
fidelity or reward geometry.

The explicit protocol factors were:

1. `CP-01` selection vs held-out reporting seed role
2. `CP-02` final-vs-best checkpoint reporting
3. `CP-03` aggregation sensitivity
4. `CP-04` comparator-specific reporting/disclosure surface

## Main Findings

1. Slice A showed the preserved artifacts were not sufficient for a full
   replay-first cross-comparator decision pass.
   - only `artifacts/scenario-corrected-pilot-01/` preserved replayable
     checkpoints
   - bounded sweep artifacts preserved aggregate outputs and manifests,
     but not comparator checkpoints or per-seed rows

2. The bounded protocol probe still answered the highest-priority
   comparator question on the same triage surfaces.
   - exercised `Table II` rows:
     `1.0/1.0/1.0`, `1.0/0.0/0.0`, `0.0/1.0/0.0`
   - exercised `Fig. 3` points:
     `160`, `180`, `200`
   - reporting seed roles:
     selection `{100,200,300,400,500}`
     vs held-out `{600,700,800,900,1000}`

3. `CP-02` is a no-op on the bounded `20`-episode surface.
   - final-vs-best changes:
     - changed winner surfaces: `0`
     - changed tie surfaces: `0`
     - new `MODQN` sole-best surfaces: `0`
   - the bounded learned runs all report `policy_episode = 19`, so the
     best-eval checkpoint collapses to the final-episode policy on this
     bounded surface

4. `CP-01` can reshuffle near-tied high-load `Fig. 3` points under the
   held-out reporting seeds.
   - changed winner/tie surfaces: `2`
   - changed surfaces:
     - `Fig. 3 @ 180 users`
     - `Fig. 3 @ 200 users`

5. Those seed-role flips do **not** recover meaningful paper-like method
   separation.
   - `maxScalarSpread = 0.028688`
   - `maxR1Spread = 0.0`
   - `maxR3Spread = 0.0`
   - `maxR2Spread = 0.095625`
   - new `MODQN` sole-best surfaces: `0`
   - the flips are entirely explained by small `r2` / handover
     differences between `DQN_scalar` and the `MODQN` /
     `DQN_throughput` tie pair

## Decision

Phase 01C ends as a `stop`.

Why:

1. final-vs-best reporting does not matter on the bounded surface
2. held-out reporting can reshuffle already near-tied high-load points,
   but only through `r2`
3. `r1` and `r3` still do not open any new cross-method separation
4. `MODQN` does not become newly sole-best anywhere

So the comparator-protocol branch does **not** justify any further
implementation slice under the current evidence.

## Recommended Interpretation

The strongest honest current claim is now:

1. standalone training and artifact surface: working
2. disclosed comparison baseline: ready and frozen
3. scenario-correction follow-on: completed negative result
4. comparator-protocol follow-on: completed negative result
5. full paper-faithful reproduction claim: still not established

## If Work Continues

No further reproduction work should continue by default under the
current authority set.

A future reopen would need at least one of:

1. new source-backed paper/runtime provenance
2. an explicit new SDD that reopens the claim boundary
3. a new external comparison requirement from the user
