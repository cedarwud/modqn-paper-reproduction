# Phase 03 Execution Report: Paired EE-MODQN Pilot

**Date:** `2026-04-28`  
**Status:** `NEEDS MORE EVIDENCE`  
**Scope:** paired objective-substitution pilot only. No Catfish,
multi-Catfish, or final Catfish-EE-MODQN work was performed.

## Execution Decision

Phase 03 is not promoted.

The bounded paired pilot changed only the first MODQN objective:

```text
MODQN-control: r1 = throughput
EE-MODQN:     r1 = per-user EE credit-assignment reward
r2/r3:        unchanged
```

Both sides used the same Phase 02B opt-in HOBS-linked SINR / power surface.
The run did not produce evidence that EE-MODQN is more energy-aware or more
effective. Best-eval `EE_system`, raw throughput, low-percentile throughput,
served ratio, active power, and load-balance metrics were tied. EE-MODQN had a
higher scalar reward only because the first objective changed scale, and it
showed worse handover / `r2`.

## Protocol

Configs:

1. `configs/ee-modqn-phase-03-control-pilot.resolved.yaml`
2. `configs/ee-modqn-phase-03-ee-pilot.resolved.yaml`

Artifact namespaces:

1. `artifacts/ee-modqn-phase-03-control-pilot/`
2. `artifacts/ee-modqn-phase-03-ee-pilot/`

Shared settings:

1. Phase 02B `active-load-concave` power surface.
2. `20` training episodes, bounded pilot only.
3. Seeds: train `42`, environment `1337`, mobility `7`.
4. Evaluation seed set: `[100, 200, 300, 400, 500]`.
5. Same target sync, replay capacity, objective weights, action space, state
   encoding, evaluation cadence, and checkpoint rule.

Checkpoint semantics:

1. final checkpoint: `final-episode-policy`
2. best-eval checkpoint: `best-weighted-reward-on-eval`
3. In this pilot, final and best-eval both selected episode `19`.

## Implemented Surface

1. Added explicit trainer gate `r1_reward_mode`.
2. Preserved baseline default `r1_reward_mode: throughput`.
3. Added Phase 03 opt-in `r1_reward_mode: per-user-ee-credit`.
4. Added per-user EE credit diagnostic:

```text
r1_i(t) = R_i(t) / (P_b(t) / N_b(t))
```

This is labeled as a modeling / credit-assignment assumption, not system EE.
Final evaluation reports:

```text
EE_system(t) = sum_i R_i(t) / sum_active_beams P_b(t)
```

## Runs And Artifacts

Denominator audits:

1. `artifacts/ee-modqn-phase-03-control-pilot/denominator-audit/`
2. `artifacts/ee-modqn-phase-03-ee-pilot/denominator-audit/`

Both config-level audits reported:

```text
classification = hobs-compatible-active-load-concave-power-surface
phase03_gate   = conditional-go-for-paired-phase-03-design
```

Training artifacts:

1. `training_log.json`
2. `run_metadata.json`
3. `checkpoints/final-episode-policy.pt`
4. `checkpoints/best-weighted-reward-on-eval.pt`

Paired comparison:

```text
artifacts/ee-modqn-phase-03-ee-pilot/paired-comparison-vs-control/
```

Outputs:

1. `phase03_paired_summary.json`
2. `phase03_episode_metrics.csv`
3. `phase03_user_step_metrics.csv`
4. `review.md`

## Best-Eval Metrics

Primary comparison uses best-eval checkpoints on the same evaluation seeds.

| Metric | MODQN-control | EE-MODQN | Result |
|---|---:|---:|---|
| `EE_system` aggregate bps/W | `873.0934588432312` | `873.0934588432312` | tie |
| mean raw throughput bps/user-step | `17.461869176864624` | `17.461869176864624` | tie |
| low p05 throughput bps | `13.576451826095582` | `13.576451826095582` | tie |
| served ratio | `1.0` | `1.0` | tie |
| mean active beam power W | `2.0` | `2.0` | tie |
| handover count | `84.4` | `92.4` | EE-MODQN worse |
| `r2` mean | `-0.422` | `-0.462` | EE-MODQN worse |
| `r3` mean | `-174.61869176864755` | `-174.61869176864755` | tie |
| load-balance gap bps | `1746.1869176864625` | `1746.1869176864625` | tie |
| scalar reward | `52.259007530593614` | `4330.4049558624265` | not comparable |

Scalar reward is supporting diagnostics only. The large scalar increase is a
reward-scale effect from changing r1 units, not an EE-system improvement.

## Rescaling Checks

Config-level denominator audit passed, but evaluated final/best policies did
not exercise denominator variability:

```text
denominator_varies_in_eval = false
mean active beam power     = 2.0 W
```

Throughput vs per-user EE reward:

```text
Pearson  = 1.0
Spearman = 0.9999999999999999
```

Same-policy replay rescoring:

| Rescore | MODQN-control | EE-MODQN | Winner |
|---|---:|---:|---|
| throughput-r1 scalar | `52.259007530593614` | `52.2470075305936` | MODQN-control |
| EE-r1 scalar | `4330.416955862427` | `4330.4049558624265` | MODQN-control |

The ranking did not change under same-policy rescoring. This is high
rescaling risk, not evidence of energy-aware learning.

## Reward-Hacking Checks

No classic service-collapse reward hacking was observed:

1. `EE_system` did not rise while throughput collapsed.
2. served ratio did not fall.
3. total active power did not fall because fewer users were served.
4. per-user EE did not rise without `EE_system`.

However, the absence of reward hacking is not a promotion signal here because
there was also no `EE_system` gain and the evaluated denominator was fixed.

## Validation

Focused tests:

```text
.venv/bin/python -m pytest \
  tests/test_phase03_ee_modqn.py \
  tests/test_ee_denominator_audit.py \
  tests/test_modqn_smoke.py \
  tests/test_training_hardening.py -q

38 passed
```

Additional focused check:

```text
.venv/bin/python -m pytest \
  tests/test_phase03_ee_modqn.py \
  tests/test_ee_denominator_audit.py \
  tests/test_step.py -q

62 passed
```

## Phase 03 Decision

`NEEDS MORE EVIDENCE`.

This pilot validates that the objective substitution path is gated and runnable,
but it does not validate EE-MODQN effectiveness. The paired evidence shows a
throughput-rescaling / policy-collapse risk in the evaluated policies and no
`EE_system` improvement.

## Remaining Blockers

1. Evaluated policies did not exercise active power denominator variability.
2. Per-user EE reward was perfectly correlated with throughput in the evaluated
   policies.
3. Same-policy replay rescoring did not change the method ranking.
4. EE-MODQN worsened handover / `r2` in the bounded pilot.
5. The Phase 02B power surface remains a disclosed synthesized proxy, not a
   paper-backed HOBS optimizer.
6. Catfish, multi-Catfish, and final Catfish-EE-MODQN claims remain blocked.
