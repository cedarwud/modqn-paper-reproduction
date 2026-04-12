# ADR-001: Start The MODQN Baseline As A Separate Python Project

## Status
Accepted

## Date
2026-04-11

## Context

The current workspace already contains `ntn-sim-core`, which is the long-term simulator platform. That repo already ships a reviewed MODQN bridge and a UI-facing result/view-model path, but its current MODQN baseline surface is not a zero-assumption paper reproduction:

1. it uses a disclosed `2 x 2` orbital proxy,
2. it includes runtime bridge assumptions needed by the platform,
3. it is shaped around platform/UI/result-consumption contracts rather than a first clean trainer implementation,
4. the target paper itself states the simulation platform is Python.

The immediate task is to reproduce `PAP-2024-MORL-MULTIBEAM` as a baseline first, then later present the resulting outputs through the `ntn-sim-core` 3D UI.

## Decision

Create a new standalone project at `modqn-paper-reproduction/` and implement the first baseline reproduction there in Python.

`ntn-sim-core` integration is deferred to a later artifact-bridge phase.

## Alternatives Considered

### Implement directly inside `ntn-sim-core`

Pros:

1. existing runtime/result/view-model surfaces already exist
2. later UI consumption would be closer

Cons:

1. platform assumptions would contaminate the first paper-baseline implementation
2. trainer design would be constrained by current platform contracts
3. the platform repo is explicitly not positioned as a one-paper reproduction codebase

Rejected because the first milestone is a clean paper-baseline reproduction, not immediate platform coupling.

### Place the work under `project/`

Pros:

1. physically separate from the platform repo

Cons:

1. repo-root governance treats `project/*` primarily as donor/reference space
2. this work is an active implementation target, not a donor project

Rejected because it misclassifies the new reproduction effort.

## Consequences

1. Phase 1 can stay Python-native and paper-focused.
2. Assumptions can be recorded against the paper rather than hidden in platform defaults.
3. Phase 2 must define an explicit artifact schema for later `ntn-sim-core` ingestion.
4. Phase 3 can focus on 3D presentation and replay mapping without porting the trainer into the platform repo.
