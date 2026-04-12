# Phase 02: Artifact Bridge SDD

**Status:** Planned SDD  
**Depends on:** Phase 01 output contracts

## 1. Purpose

Phase 02 freezes the export bundle that will allow the standalone Python reproduction to feed external consumers, especially `ntn-sim-core`, without moving the trainer into the platform repo.

## 2. Principle

The Python project remains the source of truth for:

1. training
2. sweep execution
3. paper-style plots
4. `reproduction-assumption` disclosure

External consumers receive stable artifacts, not internal trainer state.

## 3. Bundle Layout

Each run should eventually export:

```text
run-id/
├── manifest.json
├── provenance-map.json
├── config-resolved.json
├── assumptions.json
├── training/
│   ├── episode_metrics.csv
│   └── loss_curves.csv
├── evaluation/
│   ├── summary.json
│   ├── baselines.csv
│   └── sweeps/
├── figures/
│   ├── table-ii.csv
│   ├── fig-3.csv
│   ├── fig-4.csv
│   ├── fig-5.csv
│   ├── fig-6.csv
│   └── *.png
└── timeline/
    └── step-trace.jsonl
```

## 4. Required Metadata

The bundle must carry:

1. `paperId`
2. `runId`
3. `bundleSchemaVersion`
4. `producerVersion`
5. `gitCommit` if available
6. seed(s)
7. resolved baseline parameters
8. assumption IDs and values
9. baseline/comparator identities
10. completion timestamp
11. field-level provenance surface

Field-level provenance must allow a consumer to distinguish:

1. `paper-backed`
2. `recovered-from-paper`
3. `reproduction-assumption`
4. `platform-visualization-only`

## 5. Timeline Contract

The `timeline/step-trace.jsonl` surface is the future replay bridge. Each row should be append-only and contain enough truth for later 3D mapping:

1. slot index
2. simulation time
3. per-user identity
4. per-user serving satellite/beam
5. per-user position and coordinate frame
6. satellite and beam identities referenced in the row
7. candidate beam list with stable ordering semantics
8. candidate beam scores
9. visibility mask
10. action-validity mask
11. reward vector
12. scalarized reward
13. handover event classification
14. handover event ID
15. beam loads
16. per-slot KPI overlay fields
17. geometry snapshot or inline geometry payload

Geometry and identity fields are mandatory, not optional. A consumer must not invent geometry that was not present in the exported artifact.

## 6. Stability Rule

After Phase 02 is accepted:

1. field names may be added
2. existing semantics may not silently change
3. any breaking change requires a new bundle version

Phase 01 dataclasses and internal CSV headers remain unstable until Phase 02 freeze. The artifact bundle is the first external compatibility surface.

## 7. Completion Boundary

Phase 02 is complete only when:

1. Phase 01 can export one stable run bundle
2. the bundle contains figures, training curves, and timeline truth
3. the bundle can be consumed without importing Python trainer internals
4. the bundle declares `bundleSchemaVersion` and `producerVersion`
5. the bundle includes field-level provenance for UI disclosure
6. the timeline contract includes geometry, identity, visibility, and event semantics required for future 3D replay
