# Phase 01G Atmospheric-Sign Status

Date: `2026-04-22`

## Scope

This note closes the evaluation-only atmospheric-sign counterfactual
slice introduced after `Phase 01F`.

The question was narrow:

1. replay preserved checkpoints under the paper-published atmospheric
   sign and the corrected lossy sign,
2. measure whether `ASSUME-MODQN-REP-009` still creates a material
   policy-level effect after the bounded beam-aware follow-on,
3. decide whether any new training surface is justified.

No retraining was performed.

## Artifact Directories

### Beam-Aware 200

1. input run:
   `artifacts/codex-beam-aware-pilot-200ep-2026-04-22/`
2. counterfactual audit:
   `artifacts/codex-beam-aware-pilot-200ep-2026-04-22/atmospheric-sign-counterfactual-audit/`

### Frozen Baseline 200

1. input run:
   `artifacts/codex-pilot-200ep-2026-04-22/`
2. counterfactual audit:
   `artifacts/codex-pilot-200ep-2026-04-22/atmospheric-sign-counterfactual-audit/`

## Result Summary

### 1. Reward-Geometry Diagnostics Change, But Only Modestly

Both preserved `200`-episode surfaces produced the same diagnostic
change:

1. corrected/published `sample_r1` ratio: `0.9332543765022697`
2. corrected/published `|r1|/|r2|` ratio: `0.9332543765022697`
3. corrected/published `|r1|/|r3|` ratio: `1.0`

This means:

1. the corrected lossy sign reduces throughput-scale magnitudes by about
   `6.7%`,
2. it does not change the relative `r1` vs `r3` geometry,
3. it only reduces the `r1` vs `r2` dominance ratio from about
   `982.6x` to `917.0x`,
4. the dominance warning therefore remains firmly in place.

Interpretation:

1. `ASSUME-MODQN-REP-009` is visible in diagnostics,
2. but the sign change alone does not repair reward geometry.

### 2. Beam-Aware 200 Replay Effect

Beam-aware preserved replay reported:

1. diagnostics change scope: `notable`
2. `MODQN` change scope: `absent`
3. `RSS_max` change scope: `absent`

Exact replay deltas:

1. `MODQN` scalar reward: `502.8767 -> 469.2471` (`-33.6296`)
2. `MODQN` `r1`: `1013.7173 -> 945.9480`
3. `MODQN` `r2`: unchanged at `-0.545`
4. `MODQN` `r3`: `-19.0923 -> -17.8167`
5. `MODQN` handovers: unchanged at `109.0`
6. same-geometry action change rate: `0.0`
7. same-geometry satellite change rate: `0.0`

Interpretation:

1. replay values move because throughput-linked terms rescale,
2. the selected actions do not change on the audited geometry trace,
3. this is not evidence for a new training branch.

### 3. Frozen Baseline 200 Replay Effect

Frozen baseline preserved replay reported the same classification:

1. diagnostics change scope: `notable`
2. `MODQN` change scope: `absent`
3. `RSS_max` change scope: `absent`

Exact replay deltas:

1. `MODQN` scalar reward: `52.3709 -> 48.8575` (`-3.5133`)
2. `MODQN` `r1`: `175.0045 -> 163.2934`
3. `MODQN` `r2`: unchanged at `-0.435`
4. `MODQN` `r3`: `-175.0045 -> -163.2934`
5. `MODQN` handovers: unchanged at `87.0`
6. same-geometry action change rate: `0.0`
7. same-geometry satellite change rate: `0.0`

Interpretation:

1. the same pattern is not specific to the beam-aware branch,
2. the corrected sign behaves like a bounded throughput rescale on the
   preserved policy surface,
3. it still does not open a new policy-level signal on bounded replay.

## Decision

`Phase 01G does not justify new training by itself.`

Current recommendation:

1. stop again after the evaluation-only audit,
2. keep the paper-published sign as the disclosed baseline contract,
3. do not start a new `500` or `9000` run on the basis of this audit,
4. if a future follow-on is opened, it must be framed as a combined and
   explicitly experimental semantics surface rather than a sign-only
   retraining attempt.

## Practical Bottom Line

What is now established:

1. the corrected lossy sign has a real but modest diagnostic effect,
2. that effect is consistent across the frozen baseline and the
   beam-aware follow-on,
3. the effect does not change actions on the audited preserved replay
   traces,
4. the corrected sign alone does not remove reward dominance.

What is not established:

1. that corrected-lossy retraining would create a new policy surface
2. that atmospheric sign is the next highest-value training target
3. that any immediate long run is now justified
