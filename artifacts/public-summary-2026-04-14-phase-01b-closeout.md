# Phase 01B Public Summary

Date: `2026-04-14`

## Current Status

`modqn-paper-reproduction` remains a working standalone reproduction
surface for `PAP-2024-MORL-MULTIBEAM`.

The comparison-baseline bundle is still valid and ready for downstream
comparison.

The paper-faithful follow-on track is now also complete enough to make a
clear statement:

- the corrected paper-backed user geography and mobility surface was
  implemented and tested
- that correction did **not** recover the paper's claimed method
  separation

## Main Result

The repo should now be presented as:

1. a disclosed engineering comparison baseline
2. plus a completed negative-result follow-on experiment on scenario
   fidelity

It should **not** be presented as a fully established paper-faithful
reproduction.

## Why The Full Reproduction Claim Remains Open

- the high-load corrected-scenario `Fig. 3` follow-up still did not open
  method separation
- the corrected-scenario `Table II` preview remained near-tied
- reward-scale dominance still persisted after the scenario correction
- reward-geometry re-scoring did not change the winner structure

## Recommended Next Step

Do not expand Phase 01B into full follow-on sweeps.

Do not start a new long run by default.

If further research is needed, open a new explicitly experimental
`comparator-protocol` branch instead of continuing the current
scenario-fidelity track.
