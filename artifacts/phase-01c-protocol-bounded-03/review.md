# Phase 01C Bounded Protocol Probe Review

Date: `2026-04-15`

## Scope

- artifact directory: `artifacts/phase-01c-protocol-bounded-03/`
- config: `configs/modqn-paper-baseline.paper-faithful-follow-on.resolved.yaml`
- exercised `Table II` rows:
  - `1.0/1.0/1.0`
  - `1.0/0.0/0.0`
  - `0.0/1.0/0.0`
- exercised `Fig. 3` points:
  - `160`
  - `180`
  - `200`
- reporting seed roles:
  - selection `{100, 200, 300, 400, 500}`
  - held-out `{600, 700, 800, 900, 1000}`
- checkpoint modes:
  - `final-episode-policy`
  - `best-weighted-reward-on-eval`
- learned-method training episodes: `20`

## Key Findings

1. `CP-02` is a no-op on these bounded surfaces.
   - `final` vs `best-eval` changes:
     - changed winner surfaces: `0`
     - changed tie surfaces: `0`
     - new `MODQN` sole-best surfaces: `0`
   - all learned bounded runs report `policy_episode = 19`, so the
     best-eval checkpoint surface collapses to the final-episode policy
     on this `20`-episode protocol probe

2. `CP-01` does move the bounded high-load `Fig. 3` surface under the
   held-out seeds, but only in a narrow `r2`-only way.
   - changed winner/tie surfaces under held-out reporting: `2`
   - changed surfaces:
     - `Fig. 3 @ 180 users`
       - selection: `MODQN` and `DQN_throughput` tie-best
       - held-out: `DQN_scalar` sole-best
     - `Fig. 3 @ 200 users`
       - selection: `DQN_scalar` sole-best
       - held-out: `MODQN` and `DQN_throughput` tie-best

3. Those held-out winner flips do **not** open meaningful new method
   separation.
   - `maxScalarSpread = 0.028688`
   - `maxR1Spread = 0.0`
   - `maxR3Spread = 0.0`
   - `maxR2Spread = 0.095625`
   - `MODQN` becomes newly sole-best on `0` exercised surfaces
   - the flips are entirely explained by small `r2` / handover changes
     between `DQN_scalar` and the `MODQN` / `DQN_throughput` tie pair

## Interpretation

The raw promotion heuristic in
`evaluation/decision-summary.json` fires because held-out reporting
changes winner identity and tie structure on `2` exercised surfaces.

However, the higher-priority Phase 01C stop condition still applies:

1. visible cross-method differences remain confined to `r2` / handover
2. neither `r1` nor `r3` opens any new separation signal
3. `MODQN` does not become newly sole-best anywhere

So this artifact should be interpreted as:

1. evidence that the reporting seed role can reshuffle near-tied
   high-load `Fig. 3` outcomes
2. not evidence that comparator protocol recovers paper-like method
   separation

## Decision

Final Phase 01C interpretation: `stop`

Rationale:

1. `CP-02` does not matter on the bounded `20`-episode surface
2. `CP-01` changes only two already near-tied `Fig. 3` points
3. the changed points remain `r2`-only with `r1/r3` still identical
4. this is not strong enough to justify another implementation slice
   under the current Phase 01C stop rules
