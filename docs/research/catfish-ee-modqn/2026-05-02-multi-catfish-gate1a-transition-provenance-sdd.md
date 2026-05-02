# Multi-Catfish Gate 1A Transition Provenance SDD

**Date:** `2026-05-02`
**Status:** current next design gate after Gate 1 returned `NEEDS MORE DESIGN`
**Scope:** documentation / schema design only. This SDD does not authorize
implementation, training, pilot execution, simulator behavior changes, or
effectiveness promotion.

## Context

Gate 1 read-only / offline diagnostics completed against the existing
HEA-MODQN and Catfish-over-HEA artifact surfaces.

It found enough evidence to confirm the current Catfish-over-HEA bounded pilot
as a gate-level `BLOCK`, but not enough provenance to design or validate
Multi-Catfish transition-level scoring:

```text
Catfish-over-HEA mean J_delta = -1136.0548587210574
J non-worse cells = 7 / 15
EE_HO non-worse cells = 13 / 15
max handover delta = +7
max r2 worsening = 7
seed 5103/6103/7103 J_delta_sum = -37775.76469904102
```

The available artifacts expose cell, seed, rollout, aggregate, and checkpoint
metrics. They do not expose training transition rows, stable sample IDs,
per-transition reward components, per-transition `EE_HO` inputs, Catfish role
scores, coordinator quotas, rejection reasons, anti-starvation counters, or
overlap diagnostics.

## Decision

Do not proceed to Catfish implementation, training, or pilot work.

The next gate is:

```text
Gate 1A: transition-provenance schema and offline coordinator diagnostics design
```

Gate 1A must specify the minimum artifact schema required for later
Multi-Catfish offline diagnostics. A later implementation slice may emit that
schema, but only after this SDD is accepted by the controller.

## Fixed Boundaries

These remain fixed:

1. HEA utility:

   ```text
   J = throughputUtilityScale * throughput
       - lambda_HO * E_HO * handover_count
   ```

2. ratio-form handover-aware EE:

   ```text
   EE_HO =
     total_bits
     /
     (communication_energy_joules + E_HO_total_joules)

   E_HO_total_joules = handover_count * E_HO
   ```

3. simulator behavior, reward semantics, state/action surface, evaluation,
   policy selection, seeds, budgets, checkpoint protocol, and HEA artifact
   accounting must not change.
4. Catfish is a training-time external stimulus plus high-quality experience
   distillation mechanism.
5. Single-Catfish is a collapsed ablation of the same Multi-Catfish coordinator,
   not a separate old implementation route.

## Required Transition Record

Gate 1A should define an auditable transition record with at least these
fields.

### Identity And Boundary

| Field | Unit / type | Purpose |
|---|---|---|
| `artifact_schema_version` | string | schema identity |
| `run_id` | string | artifact run identity |
| `matched_boundary_id` | string | binds no-Catfish HEA, collapsed Single, full Multi, random, and equal-budget surfaces |
| `transition_id` | string | stable sample identity |
| `seed_triplet` | `[train, checkpoint, policy]` | matched seed provenance |
| `episode_index` | integer | training episode provenance |
| `step_index` | integer | within-episode step provenance |
| `source_role` | enum | control, candidate, ablation, random, equal-budget |
| `replay_source` | enum/string | base replay, role buffer, injected, resampled |
| `eval_context` | object/null | null for train transitions; explicit if an eval-derived diagnostic row |

### State / Action Provenance

| Field | Unit / type | Purpose |
|---|---|---|
| `state_hash` | string | auditable compact state identity |
| `next_state_hash` | string | auditable compact next-state identity |
| `state_features` | object | compact features used by scorers |
| `action_id` | string/integer | learner action identity |
| `serving_beam_id` | string/integer/null | handover / continuity provenance |
| `target_beam_id` | string/integer/null | action or post-action target |
| `handover_event` | boolean | per-transition continuity event |

The schema may use hashes plus compact features instead of full tensors, but it
must preserve enough provenance to recompute role scores.

### Reward And Metric Components

| Field | Unit / type | Purpose |
|---|---|---|
| `throughput_bits` | bits | local numerator input |
| `throughput_mbps` | Mbps, if used | declared source unit before conversion |
| `slot_duration_s` | seconds | conversion boundary |
| `communication_energy_joules` | J | local communication-energy input |
| `E_HO_joules` | J/event | scenario handover-energy value |
| `E_HO_total_joules` | J | local handover-energy attribution |
| `J_component_throughput` | utility units | utility numerator contribution |
| `J_component_handover_penalty` | utility units | HEA handover penalty contribution |
| `r1_throughput_component` | declared unit | original MODQN component, diagnostic only |
| `r2_handover_penalty` | declared unit | continuity / handover risk |
| `r3_load_component` | declared unit | load-balance / QoS support |
| `served_indicator` | boolean/null | local served support if attributable |
| `outage_indicator` | boolean/null | local outage support if attributable |
| `guardrail_window_id` | string/null | window boundary for p05 / served / outage diagnostics |

If `communication_energy_joules` or `E_HO_total_joules` cannot be attributed at
transition or declared window granularity, the artifact must say so explicitly
and Gate 1A cannot promote transition-level `catfish-ee` scoring.

### Role Scores

| Field | Unit / type | Purpose |
|---|---|---|
| `catfish_ee_score_raw` | numeric/null | field-backed scoped `J` / `EE_HO` admission pressure |
| `catfish_ho_score_raw` | numeric/null | field-backed handover / `r2` admission pressure |
| `catfish_qos_score_raw` | numeric/null | field-backed p05 / served / outage admission pressure |
| `catfish_ee_score_norm` | numeric/null | normalized score |
| `catfish_ho_score_norm` | numeric/null | normalized score |
| `catfish_qos_score_norm` | numeric/null | normalized score |
| `score_source_fields` | string[] | exact source fields used |
| `score_formula_id` | string | immutable formula identity |
| `score_window_id` | string/null | aggregation window for scores that cannot be per-step |

Scores must not be inferred from aggregate `J_delta`, aggregate `EE_HO_delta`,
scalar reward, seed-level summaries, or post-hoc result labels.

### Coordinator Diagnostics

| Field | Unit / type | Purpose |
|---|---|---|
| `update_id` | string/integer | training update boundary |
| `coordinator_policy_id` | string | quota / mix policy identity |
| `selected_role` | enum/null | admitted role |
| `requested_by_roles` | enum[] | roles requesting the transition |
| `admitted` | boolean | final admission decision |
| `rejection_reasons` | enum[] | auditable rejection |
| `role_quota_before` | object | quota state before decision |
| `role_quota_after` | object | quota state after decision |
| `main_replay_min_share` | ratio | non-starvation floor |
| `anti_starvation_counters` | object | role and main-replay starvation diagnostics |
| `duplicate_transition_ids` | string[] | duplicate / overlap proof |
| `role_overlap_bucket_id` | string/null | overlap grouping |
| `random_control_id` | string/null | random-control comparability |
| `equal_budget_control_id` | string/null | equal-budget comparability |

## Preliminary Score Design Constraints

Gate 1A may propose formulas, but they remain provisional until a future
artifact emits the required fields.

Acceptable score families:

1. `catfish-ee`: favors transitions or windows with field-backed positive
   scoped `J` / `EE_HO` contribution while rejecting communication-only EE
   overclaim and missing energy provenance.
2. `catfish-ho`: favors continuity-safe transitions with lower handover / `r2`
   risk and rejects ping-pong, handover increases, or seed-5103-like collapse
   patterns when identifiable.
3. `catfish-qos`: favors p05 / served / outage guardrail support and rejects
   transitions or windows that hide throughput collapse behind lower handover
   counts.

The coordinator must be designed before any pilot:

1. role quotas,
2. main-replay minimum share,
3. anti-starvation counters,
4. overlap diagnostics,
5. rejection reason taxonomy,
6. random and equal-budget control surfaces,
7. collapsed Single-Catfish ablation using the same score and coordinator
   family.

## Acceptance Criteria

Gate 1A may pass only if:

1. a concrete transition-provenance schema is accepted;
2. every score field has source-field provenance and unit semantics;
3. energy fields distinguish communication energy from `E_HO_total`;
4. `lambda_HO` is carried only as policy / objective context, not as an
   `EE_HO` denominator term;
5. coordinator diagnostics cover quotas, anti-starvation, overlap, rejection,
   random control, and equal-budget control;
6. known bad Catfish-over-HEA patterns can be represented at least as
   window-level rejection diagnostics;
7. no source field uses `J_delta`, aggregate `EE_HO_delta`, active-TX EE, scalar
   reward, or seed-level summaries as a transition score substitute;
8. the design does not require simulator behavior, reward, state/action,
   evaluation, policy-selection, or HEA / `EE_HO` formula changes.

## Stop Conditions

Stop Gate 1A if:

1. the required scoring surface needs transition fields that cannot be emitted
   without changing simulator behavior;
2. `catfish-ee`, `catfish-ho`, and `catfish-qos` reduce to the same scalar
   view;
3. the coordinator cannot compare against random and equal-budget controls;
4. the design requires post-hoc subset selection;
5. the design hides the blocked Catfish-over-HEA pilot failure;
6. the design tries to recover active-TX EE, Catfish-EE, Phase `06`, or general
   communication-only EE.

## Next Authorized Output

The next worker output, if Gate 1A is opened, should be a schema / readiness
design only:

1. proposed transition record schema,
2. proposed coordinator diagnostics schema,
3. proposed score formula IDs and source fields,
4. validator acceptance rules,
5. implementation readiness decision.

It must not include a training run, pilot run, algorithm effectiveness claim, or
promotion decision.

