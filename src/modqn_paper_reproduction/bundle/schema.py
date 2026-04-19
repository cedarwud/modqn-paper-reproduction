"""Bundle contract constants for the frozen Phase 03A / 03B surface."""

from __future__ import annotations


BUNDLE_SCHEMA_VERSION = "phase-03a-replay-bundle-v1"
TIMELINE_FORMAT_VERSION = "step-trace.jsonl/v1"
REPLAY_TRUTH_MODE = "selected-checkpoint-greedy-replay"
BEAM_CATALOG_ORDER = "satellite-major-beam-minor"
SLOT_INDEX_OFFSET = 1
SLOT_INDEX_NOTE = (
    "Slot indices start at 1. Index N corresponds to the post-step "
    "state at time_s = N * slot_duration_s after the environment reset. "
    "Slot 0 is the reset state and has no exported row."
)

POLICY_DIAGNOSTICS_VERSION = "phase-03b-policy-diagnostics-v1"
POLICY_DIAGNOSTICS_TIMELINE_FIELD = "policyDiagnostics"
POLICY_DIAGNOSTICS_TOP_CANDIDATES = 3
POLICY_DIAGNOSTICS_SELECTED_ACTION_SOURCE = "selectedServing.beamIndex"
POLICY_DIAGNOSTICS_NOTE = (
    "Rows include policyDiagnostics only when the exporter can derive "
    "stable greedy masked candidate scores from the replay checkpoint "
    "without changing Phase 03A required field semantics."
)

REQUIRED_BUNDLE_RELATIVE_PATHS = (
    "manifest.json",
    "config-resolved.json",
    "provenance-map.json",
    "assumptions.json",
    "training/episode_metrics.csv",
    "training/loss_curves.csv",
    "evaluation/summary.json",
    "evaluation/sweeps",
    "timeline/step-trace.jsonl",
)

REQUIRED_MANIFEST_FIELDS = (
    "paperId",
    "runId",
    "bundleSchemaVersion",
    "producerVersion",
    "exportedAt",
    "sourceArtifactDir",
    "checkpointRule",
    "replayTruthMode",
    "timelineFormatVersion",
    "coordinateFrame",
)

REQUIRED_NON_EMPTY_MANIFEST_FIELDS = (
    "paperId",
    "runId",
    "bundleSchemaVersion",
    "producerVersion",
    "timelineFormatVersion",
    "replayTruthMode",
)

TIMELINE_ROW_REQUIRED_FIELDS = (
    "slotIndex",
    "timeSec",
    "userId",
    "userPosition",
    "previousServing",
    "selectedServing",
    "handoverEvent",
    "visibilityMask",
    "actionValidityMask",
    "beamLoads",
    "rewardVector",
    "scalarReward",
    "satelliteStates",
    "beamStates",
    "kpiOverlay",
)

POLICY_DIAGNOSTICS_REQUIRED_FIELDS = (
    "diagnosticsVersion",
    "objectiveWeights",
    "selectedScalarizedQ",
    "runnerUpScalarizedQ",
    "scalarizedMarginToRunnerUp",
    "availableActionCount",
    "topCandidates",
)

POLICY_DIAGNOSTICS_TOP_CANDIDATE_REQUIRED_FIELDS = (
    "beamId",
    "beamIndex",
    "satId",
    "satIndex",
    "localBeamIndex",
    "validUnderDecisionMask",
    "objectiveQ",
    "scalarizedQ",
)

OPTIONAL_POLICY_DIAGNOSTICS_REQUIRED_FIELDS = (
    "present",
    "timelineField",
    "diagnosticsVersion",
    "requiredByBundleSchema",
    "producerOwned",
    "selectedActionSource",
    "topCandidateLimit",
    "rowsWithDiagnostics",
    "rowsWithoutDiagnostics",
    "note",
)
