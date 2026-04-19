"""Phase 03A replay-bundle freeze tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modqn_paper_reproduction.cli import export_main, train_main
from modqn_paper_reproduction.export.replay_bundle import (
    BUNDLE_SCHEMA_VERSION,
    POLICY_DIAGNOSTICS_VERSION,
    TIMELINE_FORMAT_VERSION,
    trim_replay_bundle_for_sample,
    validate_replay_bundle,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_BUNDLE = REPO_ROOT / "tests" / "fixtures" / "sample-bundle-v1"
RESOLVED_CONFIG = "configs/modqn-paper-baseline.resolved-template.yaml"


_REQUIRED_BUNDLE_PATHS = (
    "manifest.json",
    "config-resolved.json",
    "assumptions.json",
    "provenance-map.json",
    "training/episode_metrics.csv",
    "training/loss_curves.csv",
    "evaluation/summary.json",
    "evaluation/sweeps",
    "timeline/step-trace.jsonl",
    "figures/training-scalar-reward.png",
    "figures/training-objectives.png",
)


_REQUIRED_MANIFEST_FIELDS = (
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


_REQUIRED_TIMELINE_ROW_FIELDS = (
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


_REQUIRED_PROVENANCE_CATEGORIES = (
    "paper-backed",
    "recovered-from-paper",
    "reproduction-assumption",
    "platform-visualization-only",
    "artifact-derived",
)


@pytest.fixture(scope="module")
def sample_bundle_dir() -> Path:
    if not SAMPLE_BUNDLE.exists():
        pytest.skip(
            "Sample bundle fixture is missing. Run "
            "`./.venv/bin/python scripts/generate_sample_bundle.py` to refresh it."
        )
    return SAMPLE_BUNDLE


def test_sample_bundle_required_paths_present(sample_bundle_dir: Path) -> None:
    for relative in _REQUIRED_BUNDLE_PATHS:
        path = sample_bundle_dir / relative
        assert path.exists(), f"missing {relative}"


def test_sample_bundle_validates(sample_bundle_dir: Path) -> None:
    validate_replay_bundle(sample_bundle_dir)


def test_sample_bundle_manifest_fields(sample_bundle_dir: Path) -> None:
    manifest = json.loads((sample_bundle_dir / "manifest.json").read_text())
    for field in _REQUIRED_MANIFEST_FIELDS:
        assert field in manifest, f"manifest missing field {field}"
    assert manifest["bundleSchemaVersion"] == BUNDLE_SCHEMA_VERSION
    assert manifest["timelineFormatVersion"] == TIMELINE_FORMAT_VERSION
    assert manifest["paperId"] == "PAP-2024-MORL-MULTIBEAM"
    assert manifest["replayTruthMode"] == "selected-checkpoint-greedy-replay"
    coord = manifest["coordinateFrame"]
    for key in ("userPosition", "satellitePosition", "beamCenter", "groundPoint"):
        assert key in coord, f"coordinateFrame missing {key}"
    ground_point = coord["groundPoint"]
    assert {"latDeg", "lonDeg"} <= set(ground_point)
    assert isinstance(ground_point["latDeg"], (int, float))
    assert isinstance(ground_point["lonDeg"], (int, float))
    slot_semantics = manifest["slotIndexSemantics"]
    assert slot_semantics["firstIndex"] == 1
    assert "note" in slot_semantics
    summary = manifest["replaySummary"]
    assert summary["rowCount"] >= 1
    assert summary["slotCount"] >= 1
    assert summary["slotIndexOffset"] == 1
    assert summary["replaySeedSource"]
    assert summary["replaySeedSource"] != ""
    sample = summary["sampleSubset"]
    assert sample["maxUsers"] >= 1
    assert sample["sourceFullRowCount"] >= summary["rowCount"]
    optional_diagnostics = manifest["optionalPolicyDiagnostics"]
    assert optional_diagnostics["present"] is True
    assert optional_diagnostics["timelineField"] == "policyDiagnostics"
    assert optional_diagnostics["rowsWithDiagnostics"] == summary["rowCount"]
    assert optional_diagnostics["rowsWithoutDiagnostics"] == 0
    # Slot indices in the timeline must start at the declared offset,
    # not at 0.
    with (sample_bundle_dir / "timeline" / "step-trace.jsonl").open() as handle:
        first_row = json.loads(handle.readline())
    assert first_row["slotIndex"] >= slot_semantics["firstIndex"]


def test_sample_bundle_timeline_rows(sample_bundle_dir: Path) -> None:
    rows = []
    with (sample_bundle_dir / "timeline" / "step-trace.jsonl").open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    assert rows, "sample bundle timeline must contain at least one row"
    for row in rows:
        for field in _REQUIRED_TIMELINE_ROW_FIELDS:
            assert field in row, f"timeline row missing field {field}"
        prev = row["previousServing"]
        cur = row["selectedServing"]
        for key in ("satId", "beamId", "satIndex", "beamIndex"):
            assert key in prev
            assert key in cur
        sat_states = row["satelliteStates"]
        assert isinstance(sat_states, list) and sat_states
        for sat in sat_states:
            assert {"satId", "satIndex", "subSatellitePoint"} <= set(sat)
        beam_states = row["beamStates"]
        assert isinstance(beam_states, list) and beam_states
        for beam in beam_states:
            assert {"satId", "beamId", "satIndex", "beamIndex", "localBeamIndex"} <= set(beam)
        reward = row["rewardVector"]
        assert {"r1Throughput", "r2Handover", "r3LoadBalance"} == set(reward)
        diagnostics = row["policyDiagnostics"]
        assert diagnostics["diagnosticsVersion"] == POLICY_DIAGNOSTICS_VERSION
        assert diagnostics["availableActionCount"] == sum(
            1 for value in row["decisionActionValidityMask"] if value
        )
        assert diagnostics["topCandidates"][0]["beamIndex"] == cur["beamIndex"]
        if len(diagnostics["topCandidates"]) > 1:
            assert diagnostics["runnerUpScalarizedQ"] == pytest.approx(
                diagnostics["topCandidates"][1]["scalarizedQ"]
            )
            expected_margin = (
                diagnostics["selectedScalarizedQ"]
                - diagnostics["runnerUpScalarizedQ"]
            )
            assert diagnostics["scalarizedMarginToRunnerUp"] == pytest.approx(
                expected_margin
            )


def test_sample_bundle_provenance_map(sample_bundle_dir: Path) -> None:
    provenance = json.loads(
        (sample_bundle_dir / "provenance-map.json").read_text()
    )
    assert provenance["bundleSchemaVersion"] == BUNDLE_SCHEMA_VERSION
    legend = provenance["classificationLegend"]
    for key in _REQUIRED_PROVENANCE_CATEGORIES:
        assert key in legend, f"provenance map legend missing category {key}"
    fields = provenance["fields"]
    assert isinstance(fields, dict) and fields
    used_categories = {entry["primaryClassification"] for entry in fields.values()}
    assert used_categories <= set(_REQUIRED_PROVENANCE_CATEGORIES)
    # The map must cover at least one paper-backed and one
    # reproduction-assumption field so consumer disclosure UIs always
    # have data to render.
    assert "paper-backed" in used_categories
    assert "reproduction-assumption" in used_categories


def test_sample_bundle_assumptions_present(sample_bundle_dir: Path) -> None:
    assumptions = json.loads(
        (sample_bundle_dir / "assumptions.json").read_text()
    )
    assert isinstance(assumptions, dict)
    assert assumptions, "sample bundle assumptions must not be empty"
    has_assumption_id = False
    for name, block in assumptions.items():
        assert isinstance(block, dict), f"{name} must be an object"
        assert "value" in block, f"{name} must carry a value field"
        if "assumption_id" in block:
            assert str(block["assumption_id"]).startswith("ASSUME-MODQN-REP-")
            has_assumption_id = True
    assert has_assumption_id, "at least one entry must carry an assumption_id"


def test_sample_bundle_evaluation_summary(sample_bundle_dir: Path) -> None:
    summary = json.loads(
        (sample_bundle_dir / "evaluation" / "summary.json").read_text()
    )
    assert summary["paper_id"] == "PAP-2024-MORL-MULTIBEAM"
    assert summary["bundle_schema_version"] == BUNDLE_SCHEMA_VERSION
    assert "replay_timeline" in summary


def test_validate_replay_bundle_rejects_missing_manifest(tmp_path: Path) -> None:
    """An incomplete bundle must be rejected by validate_replay_bundle."""
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "config-resolved.json").write_text("{}")
    with pytest.raises(ValueError):
        validate_replay_bundle(incomplete)


def _write_valid_manifest(bundle: Path) -> None:
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "paperId": "x",
                "runId": "x",
                "bundleSchemaVersion": BUNDLE_SCHEMA_VERSION,
                "producerVersion": "0.1.0",
                "exportedAt": "now",
                "sourceArtifactDir": "x",
                "checkpointRule": {},
                "replayTruthMode": "selected-checkpoint-greedy-replay",
                "timelineFormatVersion": TIMELINE_FORMAT_VERSION,
                "coordinateFrame": {},
            }
        )
    )


def _prepare_incomplete_bundle(bundle: Path) -> None:
    """Create a bundle with every required path present but a placeholder
    timeline that can be overwritten per test."""
    (bundle / "training").mkdir(parents=True)
    (bundle / "evaluation" / "sweeps").mkdir(parents=True)
    (bundle / "timeline").mkdir(parents=True)
    _write_valid_manifest(bundle)
    (bundle / "config-resolved.json").write_text("{}")
    (bundle / "assumptions.json").write_text("{}")
    (bundle / "provenance-map.json").write_text("{}")
    (bundle / "training" / "episode_metrics.csv").write_text("episode\n0\n")
    (bundle / "training" / "loss_curves.csv").write_text(
        "episode,loss_q1,loss_q2,loss_q3\n0,0,0,0\n"
    )
    (bundle / "evaluation" / "summary.json").write_text("{}")


def test_validate_replay_bundle_rejects_missing_timeline_field(
    tmp_path: Path,
) -> None:
    """A timeline row with missing required field must be rejected by the
    timeline-field validation branch (not the required-paths branch)."""
    bundle = tmp_path / "bundle"
    _prepare_incomplete_bundle(bundle)
    # Timeline row is valid JSON but missing most required fields, so the
    # required-paths check should pass and the timeline-field check should
    # be the one to reject the bundle.
    (bundle / "timeline" / "step-trace.jsonl").write_text(
        json.dumps({"slotIndex": 0, "timeSec": 0.0}) + "\n"
    )
    with pytest.raises(ValueError, match="missing required fields"):
        validate_replay_bundle(bundle)


def test_validate_replay_bundle_rejects_empty_timeline(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _prepare_incomplete_bundle(bundle)
    (bundle / "timeline" / "step-trace.jsonl").write_text("")
    with pytest.raises(ValueError, match="empty"):
        validate_replay_bundle(bundle)


def test_validate_replay_bundle_rejects_second_row_missing_field(
    tmp_path: Path,
) -> None:
    """Every row must carry required fields, not just the first one."""
    bundle = tmp_path / "bundle"
    _prepare_incomplete_bundle(bundle)
    good_row = {field: None for field in _REQUIRED_TIMELINE_ROW_FIELDS}
    good_row["slotIndex"] = 1
    bad_row = dict(good_row)
    bad_row.pop("kpiOverlay")
    (bundle / "timeline" / "step-trace.jsonl").write_text(
        json.dumps(good_row) + "\n" + json.dumps(bad_row) + "\n"
    )
    with pytest.raises(ValueError, match="row 2"):
        validate_replay_bundle(bundle)


def test_validate_replay_bundle_rejects_malformed_json_row(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    _prepare_incomplete_bundle(bundle)
    good_row = {field: None for field in _REQUIRED_TIMELINE_ROW_FIELDS}
    good_row["slotIndex"] = 1
    (bundle / "timeline" / "step-trace.jsonl").write_text(
        json.dumps(good_row) + "\n{ this is not valid json\n"
    )
    with pytest.raises(ValueError, match="not valid JSON"):
        validate_replay_bundle(bundle)


def test_validate_replay_bundle_rejects_empty_paper_id(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _prepare_incomplete_bundle(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["paperId"] = ""
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    good_row = {field: None for field in _REQUIRED_TIMELINE_ROW_FIELDS}
    good_row["slotIndex"] = 1
    (bundle / "timeline" / "step-trace.jsonl").write_text(
        json.dumps(good_row) + "\n"
    )
    with pytest.raises(ValueError, match="empty identity fields"):
        validate_replay_bundle(bundle)


def test_validate_replay_bundle_rejects_misaligned_policy_diagnostics(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    _prepare_incomplete_bundle(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["optionalPolicyDiagnostics"] = {
        "present": True,
        "timelineField": "policyDiagnostics",
        "diagnosticsVersion": POLICY_DIAGNOSTICS_VERSION,
        "requiredByBundleSchema": False,
        "producerOwned": True,
        "selectedActionSource": "selectedServing.beamIndex",
        "topCandidateLimit": 3,
        "rowsWithDiagnostics": 1,
        "rowsWithoutDiagnostics": 0,
        "note": "test payload",
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest))

    row = {field: None for field in _REQUIRED_TIMELINE_ROW_FIELDS}
    row.update(
        {
            "slotIndex": 1,
            "timeSec": 1.0,
            "userId": "user-0",
            "userPosition": {},
            "previousServing": {"beamIndex": 0},
            "selectedServing": {"beamIndex": 2},
            "handoverEvent": {"kind": "none", "eventId": None},
            "visibilityMask": [True, True, False],
            "actionValidityMask": [True, True, False],
            "beamLoads": [0, 1, 0],
            "rewardVector": {
                "r1Throughput": 1.0,
                "r2Handover": 0.0,
                "r3LoadBalance": -1.0,
            },
            "scalarReward": 0.5,
            "satelliteStates": [],
            "beamStates": [],
            "kpiOverlay": {},
            "decisionActionValidityMask": [True, True, False],
            "policyDiagnostics": {
                "diagnosticsVersion": POLICY_DIAGNOSTICS_VERSION,
                "objectiveWeights": {
                    "r1Throughput": 0.5,
                    "r2Handover": 0.3,
                    "r3LoadBalance": 0.2,
                },
                "selectedScalarizedQ": 1.0,
                "runnerUpScalarizedQ": 0.8,
                "scalarizedMarginToRunnerUp": 0.2,
                "availableActionCount": 2,
                "topCandidates": [
                    {
                        "beamId": "sat-0-beam-0",
                        "beamIndex": 0,
                        "satId": "sat-0",
                        "satIndex": 0,
                        "localBeamIndex": 0,
                        "validUnderDecisionMask": True,
                        "objectiveQ": {
                            "r1Throughput": 1.0,
                            "r2Handover": 0.0,
                            "r3LoadBalance": -1.0,
                        },
                        "scalarizedQ": 1.0,
                    },
                    {
                        "beamId": "sat-0-beam-1",
                        "beamIndex": 1,
                        "satId": "sat-0",
                        "satIndex": 0,
                        "localBeamIndex": 1,
                        "validUnderDecisionMask": True,
                        "objectiveQ": {
                            "r1Throughput": 0.8,
                            "r2Handover": 0.0,
                            "r3LoadBalance": -0.5,
                        },
                        "scalarizedQ": 0.8,
                    },
                ],
            },
        }
    )
    (bundle / "timeline" / "step-trace.jsonl").write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="align topCandidates\\[0\\] with selectedServing"):
        validate_replay_bundle(bundle)


def test_trim_replay_bundle_round_trip(tmp_path: Path) -> None:
    """End-to-end: train a smoke run, export it, trim it, validate."""
    run_dir = tmp_path / "run"
    full_bundle = tmp_path / "full"
    trimmed_bundle = tmp_path / "trimmed"

    train_rc = train_main(
        [
            "--config",
            RESOLVED_CONFIG,
            "--episodes",
            "1",
            "--progress-every",
            "0",
            "--output-dir",
            str(run_dir),
        ]
    )
    assert train_rc == 0
    export_rc = export_main(
        [
            "--input",
            str(run_dir),
            "--output-dir",
            str(full_bundle),
        ]
    )
    assert export_rc == 0
    validate_replay_bundle(full_bundle)

    full_manifest = json.loads((full_bundle / "manifest.json").read_text())
    full_row_count = int(full_manifest["replaySummary"]["rowCount"])
    full_slot_count = int(full_manifest["replaySummary"]["slotCount"])
    full_handover_count = int(
        full_manifest["replaySummary"]["handoverEventCount"]
    )
    assert full_row_count >= 2
    assert full_slot_count >= 2
    # The handover count is environment-driven so only a lower-bound is
    # meaningful; the trimmed subset should never exceed the full count.
    assert full_handover_count >= 0

    report = trim_replay_bundle_for_sample(
        full_bundle,
        trimmed_bundle,
        max_users=1,
        max_slots=2,
    )
    assert report["userCount"] == 1
    assert report["slotCount"] == 2
    assert report["rowCount"] == 2
    assert report["handoverEventCount"] <= full_handover_count

    validate_replay_bundle(trimmed_bundle)
    manifest = json.loads((trimmed_bundle / "manifest.json").read_text())
    summary = json.loads((trimmed_bundle / "evaluation" / "summary.json").read_text())
    assert manifest["replaySummary"]["rowCount"] == 2
    assert manifest["replaySummary"]["slotCount"] == 2
    assert summary["replay_timeline"] == manifest["replaySummary"]
    sample_subset = manifest["replaySummary"]["sampleSubset"]
    assert sample_subset["maxUsers"] == 1
    assert sample_subset["maxSlots"] == 2
    assert sample_subset["sourceFullRowCount"] == full_row_count
    assert sample_subset["sourceFullSlotCount"] == full_slot_count
    assert sample_subset["sourceFullHandoverEventCount"] == full_handover_count
    assert "sampleNote" in manifest


def test_trim_replay_bundle_refuses_symlink_target(tmp_path: Path) -> None:
    """Symlink targets must not be overwritten by the trim helper."""
    run_dir = tmp_path / "run"
    full_bundle = tmp_path / "full"
    train_rc = train_main(
        [
            "--config",
            RESOLVED_CONFIG,
            "--episodes",
            "1",
            "--progress-every",
            "0",
            "--output-dir",
            str(run_dir),
        ]
    )
    assert train_rc == 0
    export_rc = export_main(
        [
            "--input",
            str(run_dir),
            "--output-dir",
            str(full_bundle),
        ]
    )
    assert export_rc == 0

    sensitive_dir = tmp_path / "sensitive"
    sensitive_dir.mkdir()
    sentinel = sensitive_dir / "keep.txt"
    sentinel.write_text("important")
    symlink = tmp_path / "symlink-target"
    symlink.symlink_to(sensitive_dir)

    with pytest.raises(ValueError, match="symlink"):
        trim_replay_bundle_for_sample(full_bundle, symlink, max_users=1)
    assert sentinel.exists(), "symlink target must not be wiped"
