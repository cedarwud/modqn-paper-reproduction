# Phase 03C-C Power-MDP Bounded Paired Pilot Review

Bounded paired pilot only. No long training, Catfish, multi-Catfish, frozen baseline mutation, HOBS optimizer claim, or scalar-reward-only success claim was performed.

## Protocol

- control run: `artifacts/ee-modqn-phase-03c-c-power-mdp-control-pilot`
- candidate run: `artifacts/ee-modqn-phase-03c-c-power-mdp-candidate-pilot`
- evaluation seeds: `[100, 200, 300, 400, 500]`
- primary checkpoint role: `best-eval`
- control power profile: `fixed-mid`
- candidate power selector: `runtime-ee-selector`
- shared budget: `8.0` W

## Metrics

- EE_system aggregate delta: `0.000270185470526485`
- throughput mean delta: `-4.36546863679886`
- throughput p05 delta: `-3.3914008498191834`
- served ratio delta: `0.0`
- handover count delta: `-494.0`

## Denominator

- candidate denominator varies in eval: `False`
- candidate selected profile distinct count: `1`
- candidate one-active-beam step ratio: `1.0`

## Ranking

- throughput rescore winner: `candidate`
- EE rescore winner: `candidate`
- same-policy rescore ranking changed: `False`
- candidate throughput-vs-EE Pearson: `1.0`
- candidate throughput-vs-EE Spearman: `0.9999999999999999`

## Decision

- Phase 03C-C decision: `BLOCKED`
- No EE-MODQN effectiveness claim is allowed unless the gate is `PROMOTE`.

## Forbidden Claims

- Do not claim EE-MODQN effectiveness unless this gate promotes.
- Do not claim HOBS optimizer behavior.
- Do not claim Catfish, multi-Catfish, or full paper-faithful reproduction.
- Do not treat per-user EE credit as system EE.
- Do not use scalar reward alone as success evidence.
