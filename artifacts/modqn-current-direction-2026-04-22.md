# MODQN Current Direction

Date: `2026-04-22`

## Purpose

This memo records the current stop/recommendation state after the
bounded reopen work across `Phase 01E`, `Phase 01F`, and `Phase 01G`.

Treat this as the canonical quick-reference note for:

1. what is now established,
2. what is not established,
3. what should happen next,
4. what should **not** happen next.

## Inputs

This memo is based on:

1. `artifacts/read-only-audit-2026-04-22-baseline-contract-vs-reopen.md`
2. `artifacts/phase-01e-beam-semantics-status-2026-04-22.md`
3. `artifacts/phase-01f-bounded-pilot-status-2026-04-22.md`
4. `artifacts/phase-01g-atmospheric-sign-status-2026-04-22.md`

## Current Conclusions

### 1. Frozen Baseline Is Still The Disclosure Authority

The frozen baseline remains the repo's disclosed comparison authority.

Reason:

1. it is the stable, paper-published-sign baseline surface,
2. later work has been follow-on evaluation, not a silent baseline
   replacement.

### 2. Beam-Aware Eligibility Was A Useful Fix Surface

`Phase 01F` established a real and useful bounded change.

What is true:

1. beam-aware eligibility materially improves the bounded held-out
   surface versus the frozen baseline,
2. beam collapse is removed,
3. comparator degeneration is removed on the audited bounded surface.

This means the earlier beam-semantics diagnosis was real and worth
following up.

### 3. More Episodes Are Not Justified Right Now

The current evidence does **not** support running:

1. `500` episodes
2. `9000` episodes

Reason:

1. inside the beam-aware branch, `20` and `200` episode bounded pilots
   converge to the same held-out best-eval surface,
2. the bounded pilots establish semantic change, but not new value from
   longer training,
3. the repo already has historical evidence that longer runs are costly
   and can collapse late.

### 4. Atmospheric Sign Does Not Yet Justify A New Training Branch

`Phase 01G` established that the corrected lossy sign has a real but
modest diagnostic effect.

What is true:

1. replay under corrected-lossy reduces throughput-scale magnitudes,
2. reward-dominance diagnostics change only modestly,
3. bounded preserved-policy replay shows no action change on the
   audited traces,
4. this is true for both the frozen baseline `200`-episode artifact and
   the beam-aware `200`-episode artifact.

This means atmospheric sign, by itself, is not currently the right
reason to open a new training surface.

## Practical Decision

The current recommendation is:

1. stop the active repair/retraining line here,
2. do not start new longer training runs,
3. keep the frozen baseline as the disclosed baseline,
4. keep beam-aware eligibility as an explicit bounded follow-on result,
5. treat atmospheric-sign follow-up as closed at the evaluation-only
   level unless a new explicit experimental question is opened.

## What To Do Next

If future work resumes, it should begin from a **new explicitly labeled
experimental surface**, not from "just run more episodes".

Any such reopen should:

1. state the new scientific question clearly,
2. declare the changed semantics up front,
3. begin with evaluation-only or bounded smoke / `20` / `200` runs,
4. justify any later `500` or `9000` run with new bounded evidence.

## What Not To Do

Until a new explicit surface is authorized, do **not**:

1. silently replace the frozen baseline,
2. assume beam-aware should become the new default,
3. open a sign-only retraining branch,
4. use additional episodes as the default next step,
5. treat current follow-on results as proof that the paper's intended
   final method separation has now been reproduced.

## Bottom Line

Short version:

1. beam-aware semantics helped,
2. atmospheric-sign correction alone did not open a new policy signal,
3. longer training is not justified by the current bounded evidence,
4. the correct state is **pause, document, and only reopen under a new
   explicit experiment**.
