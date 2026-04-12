# Phase 03: `ntn-sim-core` Visual Integration SDD

**Status:** Planned SDD  
**Depends on:** Phase 02 artifact bundle freeze

## 1. Purpose

Phase 03 presents the paper-reproduction outputs through `ntn-sim-core`'s 3D and overlay surfaces without turning `ntn-sim-core` into the primary trainer implementation for this paper.

## 2. Integration Rule

`ntn-sim-core` should ingest exported artifacts through a dedicated adapter layer.

It should not:

1. reimplement Phase 01 trainer logic as the primary source of truth
2. silently replace the Python bundle with platform-default assumptions
3. hide `reproduction-assumption` fields from the UI
4. consume unstable Phase 01 internal dataclasses or ad hoc CSV headers directly

## 3. Intended Visual Surfaces

The eventual integration should support:

1. baseline run summary panel
2. training convergence plots
3. sweep comparison panel
4. per-slot replay of beam selection and handover events
5. assumption/provenance disclosure panel

## 4. Data Needed From Phase 02

Minimum required artifact inputs:

1. resolved config
2. training curves
3. KPI summary
4. sweep outputs
5. timeline step trace
6. field-level provenance map
7. schema and producer versions

## 5. Presentation Rule

The UI must distinguish:

1. `paper-backed`
2. `recovered-from-paper`
3. `reproduction-assumption`
4. `platform-visualization-only`

No visual layer may relabel a `reproduction-assumption` runtime proxy as a `paper-backed` result.

## 6. Completion Boundary

Phase 03 is complete only when:

1. one exported reproduction bundle can be loaded into `ntn-sim-core`
2. the UI can display training/evaluation outputs from bundle truth
3. the 3D replay can step through handover events from exported timeline data
4. all `reproduction-assumption` fields remain visible in metadata/provenance views
5. the adapter layer consumes only the Phase 02 frozen bundle surface and does not rewrite trainer semantics
