# Phase 01B Closeout Status

Date: `2026-04-14`

## Scope Anchors

- frozen comparison-baseline status:
  - `artifacts/reproduction-status-2026-04-13.md`
  - `artifacts/public-summary-2026-04-13.md`
- Phase 01B planning and execution:
  - `docs/phases/phase-01b-paper-faithful-follow-on-sdd.md`
  - `docs/phases/phase-01b-slice-c-targeted-high-load-follow-on-sdd.md`
- follow-on runtime and pilot evidence:
  - `artifacts/scenario-corrected-pilot-01/`
  - `artifacts/scenario-corrected-table-ii-bounded-01/`
  - `artifacts/scenario-corrected-fig-3-bounded-01/`
  - `artifacts/scenario-corrected-fig-3-high-load-01/`
  - `artifacts/scenario-corrected-reward-geometry-bounded-01/`

## Executive Status

Phase 01B is now closed.

What Phase 01B established:

1. the highest-impact paper-backed scenario mismatch in the frozen
   baseline was corrected on an explicit follow-on track
2. that corrected scenario was exercised through bounded pilot evidence
   plus a targeted high-load `Fig. 3` follow-up
3. the corrected scenario did **not** recover paper-like method
   separation

That means the repo should now be interpreted as:

1. a working disclosed comparison baseline
2. plus a completed paper-faithful scenario-correction follow-on that
   ended in a negative result

It should **not** be interpreted as a fully established paper-faithful
baseline reproduction.

## What Phase 01B Changed

The follow-on track corrected the most visible scenario mismatch versus
the frozen baseline:

1. ground point moved to `(40.0, 116.0)`
2. user area changed to `200 km x 90 km`
3. user placement changed to `uniform-rectangle`
4. mobility changed to `random-wandering`

These changes landed on a separate experimental config surface and did
not silently redefine the frozen comparison baseline.

## Main Findings

1. The scenario-corrected runtime surface is real and stable enough for
   pilot work.
   - `artifacts/scenario-corrected-pilot-01/` completed `50` episodes
     and exported a replay bundle
   - best-eval checkpointing and metadata/export disclosure remain intact

2. Reward-scale dominance survived the scenario correction.
   - baseline corrected-scenario diagnostics still report
     `|r1| / |r2| ~= 982.6x`
   - targeted high-load `Fig. 3` diagnostics remained in the same
     throughput-dominant regime

3. The bounded corrected-scenario `Table II` surface stayed collapsed.
   - exact tie rows: `2/3`
   - max scalar spread across methods: `0.0140`
   - visible cross-method differences remained localized to `r2` /
     handover

4. The bounded corrected-scenario `Fig. 3` preview stayed near-tied.
   - exercised prefix points: `40`, `60`, `80`
   - exact tie points: `2/3`
   - max weighted-reward spread: `0.026625`

5. The targeted high-load corrected-scenario `Fig. 3` follow-up also did
   not open the method surface.
   - exercised points: `160`, `180`, `200`
   - `MODQN` sole-best points: `0/3`
   - exact tie points: `1/3`
   - max weighted-reward spread: `0.028688`
   - `r1` and `r3` remained identical across methods at all exercised
     points; differences were still only in `r2` / handover

6. Reward-geometry analysis did not change the follow-on interpretation.
   - winner identity did not change
   - tie count did not change
   - the largest diagnostic scalar spread remained localized to the
     handover-only row

## Decision

Phase 01B ends as a `negative-result stop`.

The repo should therefore:

1. keep the frozen comparison-baseline bundle intact
2. keep the Phase 01B follow-on artifacts as disclosed negative-result
   evidence
3. avoid expanding directly to full follow-on `Fig. 4` to `Fig. 6`
   sweeps
4. avoid starting a blind new `9000`-episode long run

## Recommended Interpretation

The strongest honest current claim is:

1. standalone training and artifact surface: working
2. disclosed comparison baseline: ready and frozen
3. paper-faithful scenario-correction follow-on: completed
4. full paper-faithful reproduction claim: still not established

## If Work Continues

If experimentation continues, it should not continue under the current
Phase 01B scenario-fidelity framing.

The only currently defensible next branch is a new explicitly labeled
`comparator-protocol` experiment surface.

That branch should begin with a new SDD rather than by expanding the
current follow-on sweep family.
