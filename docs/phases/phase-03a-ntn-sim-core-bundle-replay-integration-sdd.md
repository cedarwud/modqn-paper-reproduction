# Phase 03A: `ntn-sim-core` Bundle Replay Integration SDD

**Status:** Promoted follow-on SDD; Slice A landed.
**Date:** `2026-04-13`
**Producer schema:** `phase-03a-replay-bundle-v1`
**Slice A closeout:**
[`../../artifacts/reproduction-status-2026-04-13-phase-03a-slice-a.md`](../../artifacts/reproduction-status-2026-04-13-phase-03a-slice-a.md)
**Canonical sample bundle:** `tests/fixtures/sample-bundle-v1/`

**Depends on:**

1. `docs/baseline-acceptance-checklist.md`
2. `docs/phases/phase-02-artifact-bridge-sdd.md`
3. `docs/phases/phase-03-ntn-sim-core-visual-integration-sdd.md`
4. `artifacts/reproduction-status-2026-04-13.md`

## 1. Purpose

This follow-on defines the first low-coupling path for presenting
`modqn-paper-reproduction` results inside `ntn-sim-core` without turning
`ntn-sim-core` into the primary trainer or handover-decision authority
for `PAP-2024-MORL-MULTIBEAM`.

The target outcome is a **bundle-driven replay mode**:

1. `modqn-paper-reproduction` remains the producer of training, replay,
   handover truth, and assumption disclosure
2. `ntn-sim-core` becomes a consumer that loads a frozen export bundle
   through an adapter layer
3. the frontend can switch between native `ntn-sim-core` runtime truth
   and MODQN bundle-replay truth

## 2. Why This Follow-On Exists

The comparison-baseline bundle is now complete, but the current repo
still stops at:

1. training artifacts
2. figure/table outputs
3. analysis notes
4. incomplete Phase 02 export-bundle freeze

The user goal is to present the MODQN baseline through
`ntn-sim-core`'s 3D/replay UI while keeping coupling shallow.

That means the next step is **not** more baseline retraining.
It is a contract/export/integration task:

1. freeze the minimum producer bundle needed for replay
2. add one consumer adapter in `ntn-sim-core`
3. expose a mode switch in the UI

## 3. Scope

### 3.1 In Scope

1. a stable bundle contract exported from `modqn-paper-reproduction`
2. one adapter layer in `ntn-sim-core` that consumes that bundle
3. one frontend mode switch between native simulator truth and bundle
   replay truth
4. bundle-driven rendering of:
   - satellites
   - beams
   - serving links
   - handover events
   - training/evaluation summary
   - assumption/provenance metadata
5. validation that the replayed handover path comes from exported truth,
   not from re-running native `ntn-sim-core` handover logic

### 3.2 Explicitly Out Of Scope

1. porting the Python trainer into `ntn-sim-core`
2. making `ntn-sim-core` the new baseline truth source for this paper
3. silently replacing bundle truth with native simulator recomputation
4. making a new paper-faithful reproduction claim
5. deep profile/config merging between the two repos
6. full multi-paper artifact generalization in the first pass

## 4. Repo Boundary And Authority

### 4.1 Producer: `modqn-paper-reproduction`

This repo remains the authority for:

1. training
2. checkpoint selection
3. sweep execution
4. handover truth for the exported run
5. `reproduction-assumption` disclosure
6. paper-specific result interpretation

### 4.2 Consumer: `ntn-sim-core`

`ntn-sim-core` is responsible only for:

1. loading the frozen bundle
2. validating schema/version compatibility
3. adapting bundle truth into replay/view-model surfaces
4. rendering bundle truth in UI/3D
5. surfacing provenance and limitations

### 4.3 Non-Negotiable Boundary Rules

1. `ntn-sim-core` must not recompute the MODQN handover decision as the
   primary replay truth in bundle mode.
2. `ntn-sim-core` must not import Python trainer internals.
3. `modqn-paper-reproduction` must not export unstable internal
   dataclasses as if they were frozen public contracts.
4. If bundle geometry is missing, the consumer should reject the bundle
   rather than invent geometry.

## 5. Integration Principle

The first integration should be **artifact-driven, not engine-driven**.

The intended layering is:

```text
modqn-paper-reproduction
  -> export bundle (frozen schema)
  -> ntn-sim-core adapter
  -> ntn-sim-core replay/view-model layer
  -> ntn-sim-core frontend mode switch + overlays
```

The adapter is the only integration seam. No direct trainer/runtime
coupling is allowed across repos.

## 6. Mode Switch Design

The UI should expose two distinct modes:

### 6.1 Native Simulator Mode

Uses current `ntn-sim-core` runtime truth:

1. native scenario/profile/runtime execution
2. native handover logic
3. native replay/benchmark flow

### 6.2 MODQN Bundle Replay Mode

Uses exported MODQN truth:

1. bundle-provided satellite / beam / serving state
2. bundle-provided handover event classification
3. bundle-provided timeline and KPI overlays
4. bundle-provided assumptions / provenance / reproduction metadata

### 6.3 Mode Switch Rule

The mode switch must change the **truth source**, not just the overlay
labels.

In bundle replay mode:

1. native handover logic is not the source of truth
2. native profile defaults are not the source of truth
3. the adapter-fed bundle model is the source of truth

## 7. Bundle Contract: Minimum V1 Surface

The first frozen bundle version does not need to solve every future
consumer need, but it must be sufficient for one reviewed replay mode in
`ntn-sim-core`.

### 7.1 Required Root Files

```text
bundle-root/
├── manifest.json
├── config-resolved.json
├── assumptions.json
├── provenance-map.json
├── training/
│   ├── episode_metrics.csv
│   └── loss_curves.csv
├── evaluation/
│   ├── summary.json
│   └── sweeps/
├── figures/
└── timeline/
    └── step-trace.jsonl
```

### 7.2 Required `manifest.json` Fields

1. `paperId`
2. `runId`
3. `bundleSchemaVersion`
4. `producerVersion`
5. `exportedAt`
6. `sourceArtifactDir`
7. `checkpointRule`
8. `replayTruthMode`
9. `timelineFormatVersion`
10. `coordinateFrame`

### 7.3 Required Timeline Fields

Each `timeline/step-trace.jsonl` row must include enough information to
replay one slot without consulting the Python runtime:

1. `slotIndex`
2. `timeSec`
3. `userId`
4. `userPosition`
   - at minimum: local tangent frame coordinates
5. `previousServing`
   - `satId`
   - `beamId`
6. `selectedServing`
   - `satId`
   - `beamId`
7. `handoverEvent`
   - `kind`
   - `eventId`
8. `visibilityMask`
9. `actionValidityMask`
10. `beamLoads`
11. `rewardVector`
12. `scalarReward`
13. `satelliteStates`
14. `beamStates`
15. `kpiOverlay`

### 7.4 Required Identity/Geometry Rules

The exported timeline must use stable IDs:

1. satellite IDs
2. beam IDs
3. user IDs
4. stable beam ordering semantics

The consumer must not infer:

1. satellite positions
2. beam centers
3. candidate validity
4. handover classification

If the bundle does not provide them, the bundle is not replay-complete.

### 7.5 Optional V1 Fields

These are useful but not required for the first replay landing:

1. per-beam objective/Q scores
2. extra debug traces for rejected actions
3. richer chart-ready denormalized sweep summaries

## 8. Consumer Adapter Contract In `ntn-sim-core`

`ntn-sim-core` should add a dedicated adapter layer rather than
consuming bundle JSON/CSV ad hoc inside overlays.

The adapter should:

1. validate `bundleSchemaVersion`
2. parse manifest/config/assumptions/provenance
3. parse `step-trace.jsonl`
4. convert rows into a consumer-native replay domain model
5. expose one typed view-model surface to the UI

The adapter must not:

1. call native handover code to derive serving beams
2. backfill missing geometry from native profiles
3. reinterpret `reproduction-assumption` values as native defaults

## 9. Intended UI Surfaces In `ntn-sim-core`

The first integrated consumer should support:

1. one bundle-picker or fixed sample-bundle entry
2. one mode switch:
   - native mode
   - MODQN bundle replay mode
3. one replay timeline surface for stepping through handover events
4. one summary panel for training/evaluation outputs
5. one provenance/assumption disclosure panel

### 9.1 First-Pass Render Requirement

The first pass does not need to recreate every paper figure in the
frontend. It needs to prove:

1. the right satellites/beams are rendered from bundle truth
2. the serving link changes follow bundle handover truth
3. the metadata/disclosure stays attached to that replay

## 10. Implementation Slices

### 10.1 Slice A: Producer Bundle Freeze

Repo: `modqn-paper-reproduction`

Deliverables:

1. add `bundleSchemaVersion`
2. freeze root bundle layout
3. export `config-resolved.json`
4. export `provenance-map.json`
5. export replay-complete `timeline/step-trace.jsonl`
6. add exporter validation for required fields

Acceptance:

1. one existing reviewed run can be exported into a replay-complete
   bundle
2. exporter rejects incomplete replay bundles

### 10.2 Slice B: Consumer Adapter

Repo: `ntn-sim-core`

Deliverables:

1. typed bundle loader
2. schema/version guard
3. replay-frame adapter
4. minimal sample-bundle fixture

Acceptance:

1. adapter can load one exported bundle without importing Python code
2. adapter can expose one slot-by-slot replay model

### 10.3 Slice C: Mode Switch And Replay Presentation

Repo: `ntn-sim-core`

Deliverables:

1. UI switch between native runtime and MODQN bundle replay
2. replay controls for stepping bundle slots
3. serving-link / handover overlay driven by bundle truth

Acceptance:

1. switching mode changes truth source, not just labels
2. bundle replay shows serving-beam transitions exactly from the bundle

### 10.4 Slice D: Metadata / Provenance UI

Repo: `ntn-sim-core`

Deliverables:

1. assumptions panel
2. provenance surface
3. training/eval summary panel for the loaded bundle

Acceptance:

1. `reproduction-assumption` values remain visible in the UI
2. the UI does not imply these values are native simulator defaults

## 11. Validation Plan

### 11.1 Producer-Side Validation

Expected checks:

1. exporter unit/integration test for bundle manifest
2. exporter unit/integration test for replay timeline completeness
3. fixture export from one reviewed artifact directory

### 11.2 Consumer-Side Validation

Expected checks:

1. adapter fixture-load test
2. replay-frame consistency test
3. UI state test for mode switching
4. browser-visible smoke for bundle replay

### 11.3 Cross-Repo Acceptance

The follow-on is only accepted when:

1. one reviewed `modqn-paper-reproduction` run exports a replay-complete
   frozen bundle
2. `ntn-sim-core` loads that bundle through the adapter
3. the frontend shows bundle-driven satellite/beam/handover replay
4. handover transitions match bundle truth rather than native
   recomputation
5. provenance/assumptions remain visible end to end

## 12. Non-Goals For The First Landing

Do not expand the first landing into:

1. generic multi-paper ingestion
2. trainer-in-the-browser execution
3. new scientific claims about paper-faithful reproduction
4. automatic bundle-to-native-profile conversion
5. deep engine coupling between the two repos

## 13. Working Rule For Future Dialogs

When implementing this follow-on:

1. read this SDD first
2. treat `modqn-paper-reproduction` as producer and `ntn-sim-core` as
   consumer
3. keep changes split by slice
4. do not reopen baseline interpretation work unless the task explicitly
   changes goals
5. if `ntn-sim-core` code is changed, also follow
   `/home/u24/papers/ntn-sim-core/agent-governance.md` and update its
   local status/doc surfaces in the same change set when contracts or
   entry guidance change
