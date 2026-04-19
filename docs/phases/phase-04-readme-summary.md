# Phase 04 Summary

This note is the short read surface for the newer Phase 04 materials.
Use it before reading the full kickoff and slice SDDs.

## What Phase 04 Is

Phase 04 is the approved **internal hardening** direction for
`modqn-paper-reproduction`.

Its current intended scope is:

1. lock the existing producer contract with semantic golden tests,
2. make training-side artifact models explicit,
3. reduce internal duplication around artifact writing / reading,
4. prepare the standalone producer for cleaner downstream integration.

It is about internal cleanup and contract hardening, not about changing
the published bundle semantics.

## What Phase 04 Is Not

Phase 04 is **not** currently the landed producer authority.

It does not replace:

1. the Phase 01C closeout,
2. the Phase 03A replay bundle contract,
3. the Phase 03B additive `policyDiagnostics` surface,
4. the canonical sample fixture,
5. strict resolved-run training guardrails.

It is also not:

1. a new paper-faithful reproduction claim,
2. a trainer rewrite,
3. consumer-side work inside `ntn-sim-core`,
4. authority to couple downstream consumers to trainer internals.

## Current Interpretation

As of `2026-04-19`, read Phase 04 this way:

1. the kickoff and slice docs are real and relevant,
2. Phase 04A semantic-golden guardrail is now landed as an internal test gate,
3. Phase 04B training-artifact model seam is now landed as an internal model / serialization gate,
4. Phase 04C bundle-layer split is now landed as an internal bundle-contract gate,
5. later Phase 04 slices beyond Slice C remain planning / in-flight follow-on work,
6. none of this should be presented as a landed rewrite of the external producer authority.

For the authoritative interpretation, read
[`../../artifacts/phase-04-current-state-2026-04-19.md`](../../artifacts/phase-04-current-state-2026-04-19.md).

## When Phase 04 Can Be Treated As Landed

Phase 04A, Phase 04B, and Phase 04C may now be described as landed as
internal guardrail slices. The broader Phase 04 cleanup program should only be
described as landed when all of the following are true together:

1. the relevant docs / tests / status surfaces are committed and
   synchronized,
2. README / docs indexes point to the landed interpretation without
   contradiction,
3. an explicit landed status note says so,
4. the frozen external producer surfaces remain protected.

Until then, the safe wording is:

`Phase 04A, Phase 04B, and Phase 04C are landed as internal hardening slices, but the landed external producer authority remains Phase 03A / 03B plus the closeout/freeze surfaces.`
