# RA-EE-02 Oracle Power-Allocation Audit Review

Offline fixed-trajectory oracle / heuristic audit only. No RL training, Catfish, multi-Catfish, frozen baseline mutation, HOBS optimizer claim, or old EE-MODQN effectiveness claim was performed.

## Protocol

- method label: `RA-EE-MDP / RA-EE-MODQN`
- config: `configs/ra-ee-02-oracle-power-allocation-audit.resolved.yaml`
- artifact namespace: `artifacts/ra-ee-02-oracle-power-allocation-audit`
- evaluation seeds: `[100, 200, 300, 400, 500]`
- trajectories: `['phase03c-c-candidate-best-eval', 'hold-current', 'random-valid', 'spread-valid']`
- candidates: `['fixed-control', 'load-concave', 'budget-trim', 'qos-tail-boost', 'constrained-oracle']`
- p05 throughput guardrail ratio: `0.95`

## Proof Flags

- denominator changed by power decision: `True`
- same-policy throughput-vs-EE ranking separates: `True`
- has budget-respecting candidate: `True`
- oracle/heuristic beats fixed control on EE: `True`
- QoS guardrails pass: `True`
- selected profile not single-point on non-collapsed trajectories: `True`
- active power not single-point on non-collapsed trajectories: `True`
- no budget violations for accepted candidate: `True`

## Accepted Candidates

- accepted count: `3`
- `hold-current::constrained-oracle` EE delta `3.189055583818231`, p05 ratio `0.9818337745575232`, QoS `True`, budget `True`
- `random-valid::constrained-oracle` EE delta `2.214164300102084`, p05 ratio `0.9731482959399329`, QoS `True`, budget `True`
- `spread-valid::constrained-oracle` EE delta `1.039861022592504`, p05 ratio `1.0117000462649277`, QoS `True`, budget `True`

## Decision

- RA-EE-02 decision: `PASS to RA-EE-03 design`
- allowed next step: RA-EE-03 resource-allocation MDP design only; no RL training claim yet

## Forbidden Claims

- Do not claim old EE-MODQN effectiveness.
- Do not claim HOBS optimizer behavior.
- Do not claim Catfish, multi-Catfish, or Catfish-EE-MODQN effectiveness.
- Do not treat per-user EE credit as system EE.
- Do not use scalar reward alone as success evidence.
- Do not claim full paper-faithful reproduction or absolute energy saving.
