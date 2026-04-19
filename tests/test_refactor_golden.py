"""Phase 04 Slice A — artifact-level semantic golden tests.

Locks the current producer contract so that Slice B/C/D/E refactor
work cannot silently drift a Phase 03A / 03B required surface. No
source under ``src/`` is touched by this file.

Families:
    F1  ``run_metadata.json`` key / shape
    F2  ``training_log.json`` per-episode row shape
    F3  checkpoint payload envelope (``format_version == 1``)
    F4  ``manifest.replaySummary`` <-> ``evaluation.summary.replay_timeline``
        cross-file consistency
    F5  timeline row geometry invariants
    F6  ``scripts/generate_sample_bundle.py`` regeneration determinism
"""

from __future__ import annotations

import filecmp
import math
import subprocess
import sys
from dataclasses import fields as dc_fields
from pathlib import Path

import pytest
import torch

from modqn_paper_reproduction.algorithms.modqn import TrainerConfig
from modqn_paper_reproduction.cli import export_main, train_main
from modqn_paper_reproduction.env.step import local_tangent_offset_km

from tests.refactor_golden.helpers import (
    assert_key_set,
    assert_key_superset,
    assert_optional_type,
    assert_type,
    iter_directory_files,
    load_json,
    load_timeline_rows,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_BUNDLE = REPO_ROOT / "tests" / "fixtures" / "sample-bundle-v1"
SMOKE_CONFIG = (
    REPO_ROOT / "configs" / "modqn-paper-baseline.resolved-template.yaml"
)
GENERATE_SCRIPT = REPO_ROOT / "scripts" / "generate_sample_bundle.py"


# --- Fixtures ---------------------------------------------------------


@pytest.fixture(scope="module")
def smoke_run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One-episode smoke training artifact directory, reused across F1-F3/F4/F5."""
    run_dir = tmp_path_factory.mktemp("golden-smoke-run") / "run"
    rc = train_main(
        [
            "--config",
            str(SMOKE_CONFIG),
            "--episodes",
            "1",
            "--progress-every",
            "0",
            "--output-dir",
            str(run_dir),
        ]
    )
    assert rc == 0, f"train_main exited with {rc}"
    return run_dir


@pytest.fixture(scope="module")
def smoke_bundle_dir(
    smoke_run_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Bundle exported from the smoke training run, reused across F4/F5."""
    bundle_dir = tmp_path_factory.mktemp("golden-smoke-bundle") / "bundle"
    rc = export_main(
        [
            "--input",
            str(smoke_run_dir),
            "--output-dir",
            str(bundle_dir),
        ]
    )
    assert rc == 0, f"export_main exited with {rc}"
    return bundle_dir


@pytest.fixture(scope="module")
def sample_bundle_dir() -> Path:
    """Checked-in trimmed sample bundle fixture (Phase 03A canonical)."""
    if not SAMPLE_BUNDLE.exists():
        pytest.skip(
            "Sample bundle fixture is missing. Run "
            "`scripts/generate_sample_bundle.py` first."
        )
    return SAMPLE_BUNDLE


# --- Identity invariants shared across families ----------------------


PAPER_ID = "PAP-2024-MORL-MULTIBEAM"
CHECKPOINT_ASSUMPTION_ID = "ASSUME-MODQN-REP-015"
PRIMARY_REPORT = "final-episode-policy"
SECONDARY_REPORT = "best-weighted-reward-on-eval"
VALID_HANDOVER_KINDS = {
    "none",
    "intra-satellite-beam-switch",
    "inter-satellite-handover",
}
SLOT_DURATION_S_EXPECTED = 1.0


# --- F1 -- run_metadata.json key / shape golden -----------------------


F1_TOP_LEVEL_KEYS = {
    "paper_id",
    "package_version",
    "config_path",
    "config_role",
    "resolved_config_snapshot",
    "training_experiment",
    "seeds",
    "checkpoint_rule",
    "reward_calibration",
    "checkpoint_files",
    "resolved_assumptions",
    "runtime_environment",
    "trainer_config",
    "best_eval_summary",
    "resume_from",
    "training_summary",
}

F1_SEEDS_KEYS = {
    "train_seed",
    "environment_seed",
    "mobility_seed",
    "evaluation_seed_set",
}

F1_CHECKPOINT_RULE_KEYS = {
    "assumption_id",
    "primary_report",
    "secondary_report",
    "secondary_implemented",
    "secondary_status",
}

F1_CHECKPOINT_FILES_KEYS = {
    "primary_final",
    "secondary_best_eval",
}

F1_TRAINING_SUMMARY_KEYS = {
    "episodes_requested",
    "episodes_completed",
    "elapsed_s",
    "final_episode_index",
    "final_scalar_reward",
}

F1_REWARD_CALIBRATION_KEYS = {
    "enabled",
    "mode",
    "source",
    "scales",
    "training_experiment_kind",
    "training_experiment_id",
    "evaluation_metrics",
    "checkpoint_selection_metric",
}


def test_f1_run_metadata_shape(smoke_run_dir: Path) -> None:
    """F1: run_metadata.json must carry its current key set and identity values."""
    metadata = load_json(smoke_run_dir / "run_metadata.json")
    assert_type(metadata, dict, context="run_metadata.json")
    assert_key_set(
        metadata, F1_TOP_LEVEL_KEYS, context="run_metadata.json top level"
    )

    # Identity values that pin the paper / assumption contract.
    assert metadata["paper_id"] == PAPER_ID
    assert_type(
        metadata["package_version"], str, context="run_metadata.package_version"
    )
    assert_type(metadata["config_path"], str, context="run_metadata.config_path")

    # Seeds shape.
    assert_type(metadata["seeds"], dict, context="run_metadata.seeds")
    assert_key_set(
        metadata["seeds"], F1_SEEDS_KEYS, context="run_metadata.seeds"
    )
    assert_type(
        metadata["seeds"]["train_seed"],
        int,
        context="run_metadata.seeds.train_seed",
    )
    assert_type(
        metadata["seeds"]["evaluation_seed_set"],
        list,
        context="run_metadata.seeds.evaluation_seed_set",
    )

    # Checkpoint rule identity.
    rule = metadata["checkpoint_rule"]
    assert_type(rule, dict, context="run_metadata.checkpoint_rule")
    assert_key_set(
        rule, F1_CHECKPOINT_RULE_KEYS, context="run_metadata.checkpoint_rule"
    )
    assert rule["assumption_id"] == CHECKPOINT_ASSUMPTION_ID
    assert rule["primary_report"] == PRIMARY_REPORT
    assert rule["secondary_report"] == SECONDARY_REPORT
    assert_type(
        rule["secondary_implemented"],
        bool,
        context="run_metadata.checkpoint_rule.secondary_implemented",
    )

    # Checkpoint files shape.
    files = metadata["checkpoint_files"]
    assert_type(files, dict, context="run_metadata.checkpoint_files")
    assert_key_set(
        files,
        F1_CHECKPOINT_FILES_KEYS,
        context="run_metadata.checkpoint_files",
    )
    assert_type(
        files["primary_final"],
        str,
        context="run_metadata.checkpoint_files.primary_final",
    )
    assert_optional_type(
        files["secondary_best_eval"],
        str,
        context="run_metadata.checkpoint_files.secondary_best_eval",
    )

    # Training summary shape.
    summary = metadata["training_summary"]
    assert_type(summary, dict, context="run_metadata.training_summary")
    assert_key_set(
        summary,
        F1_TRAINING_SUMMARY_KEYS,
        context="run_metadata.training_summary",
    )

    # Reward calibration shape.
    calib = metadata["reward_calibration"]
    assert_type(calib, dict, context="run_metadata.reward_calibration")
    assert_key_set(
        calib,
        F1_REWARD_CALIBRATION_KEYS,
        context="run_metadata.reward_calibration",
    )

    # Optional or env-dependent nested blocks — type only.
    assert_type(
        metadata["resolved_config_snapshot"],
        dict,
        context="run_metadata.resolved_config_snapshot",
    )
    assert_type(
        metadata["resolved_assumptions"],
        dict,
        context="run_metadata.resolved_assumptions",
    )
    assert_type(
        metadata["runtime_environment"],
        dict,
        context="run_metadata.runtime_environment",
    )
    assert_type(
        metadata["trainer_config"],
        dict,
        context="run_metadata.trainer_config",
    )
    assert_optional_type(
        metadata["training_experiment"],
        dict,
        context="run_metadata.training_experiment",
    )
    assert_optional_type(
        metadata["best_eval_summary"],
        dict,
        context="run_metadata.best_eval_summary",
    )
    assert_optional_type(
        metadata["resume_from"],
        dict,
        context="run_metadata.resume_from",
    )


def test_f1_trainer_config_shape_matches_dataclass(smoke_run_dir: Path) -> None:
    """F1b: ``trainer_config`` block must cover every TrainerConfig field.

    This is the invariant that protects Slice D's facade-preserving
    refactor: ``TrainerConfig`` may move to ``runtime/trainer_spec.py``
    but its field set must remain identical.
    """
    metadata = load_json(smoke_run_dir / "run_metadata.json")
    expected_fields = {f.name for f in dc_fields(TrainerConfig)}
    assert_key_set(
        metadata["trainer_config"],
        expected_fields,
        context="run_metadata.trainer_config",
    )


# --- F2 -- training_log.json row shape golden -------------------------


F2_ROW_KEYS = {
    "episode",
    "epsilon",
    "r1_mean",
    "r2_mean",
    "r3_mean",
    "scalar_reward",
    "total_handovers",
    "replay_size",
    "losses",
}


def test_f2_training_log_row_shape(smoke_run_dir: Path) -> None:
    """F2: every row in training_log.json has the documented key set."""
    log = load_json(smoke_run_dir / "training_log.json")
    assert_type(log, list, context="training_log.json")
    assert len(log) >= 1, "training_log.json must contain at least one row"

    previous_episode: int | None = None
    for i, row in enumerate(log):
        assert_type(row, dict, context=f"training_log.json row {i}")
        assert_key_set(
            row, F2_ROW_KEYS, context=f"training_log.json row {i}"
        )
        assert_type(row["episode"], int, context=f"row {i}.episode")
        assert_type(row["epsilon"], float, context=f"row {i}.epsilon")
        assert_type(
            row["total_handovers"], int, context=f"row {i}.total_handovers"
        )
        assert_type(row["replay_size"], int, context=f"row {i}.replay_size")
        for key in ("r1_mean", "r2_mean", "r3_mean", "scalar_reward"):
            assert_type(row[key], (int, float), context=f"row {i}.{key}")

        losses = row["losses"]
        assert_type(losses, list, context=f"row {i}.losses")
        assert len(losses) == 3, (
            f"row {i}.losses must have 3 elements, got {len(losses)}"
        )
        for j, value in enumerate(losses):
            assert_type(value, (int, float), context=f"row {i}.losses[{j}]")

        if previous_episode is None:
            assert row["episode"] == 0, (
                "first row must have episode == 0, "
                f"got {row['episode']!r}"
            )
        else:
            assert row["episode"] == previous_episode + 1, (
                f"row {i}.episode must be monotonic +1, "
                f"got {row['episode']!r} after {previous_episode!r}"
            )
        previous_episode = row["episode"]


# --- F3 -- checkpoint payload envelope golden -------------------------


F3_REQUIRED_TOP_KEYS = {
    "format_version",
    "checkpoint_kind",
    "episode",
    "train_seed",
    "env_seed",
    "mobility_seed",
    "state_dim",
    "action_dim",
    "trainer_config",
    "checkpoint_rule",
    "q_networks",
    "target_networks",
}

F3_EVALUATION_SUMMARY_REQUIRED_KEYS = {
    "episode",
    "evaluation_every_episodes",
    "eval_seeds",
    "mean_scalar_reward",
    "std_scalar_reward",
    "mean_r1",
    "std_r1",
    "mean_r2",
    "std_r2",
    "mean_r3",
    "std_r3",
    "mean_total_handovers",
    "std_total_handovers",
}


def _load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def test_f3_primary_checkpoint_envelope(smoke_run_dir: Path) -> None:
    """F3a: primary (final-episode) checkpoint carries the V1 envelope."""
    metadata = load_json(smoke_run_dir / "run_metadata.json")
    primary_path = Path(metadata["checkpoint_files"]["primary_final"])
    payload = _load_checkpoint(primary_path)

    assert_type(payload, dict, context="primary checkpoint payload")
    assert payload["format_version"] == 1, (
        f"primary checkpoint format_version must be 1, "
        f"got {payload['format_version']!r}"
    )
    assert_key_superset(
        payload, F3_REQUIRED_TOP_KEYS, context="primary checkpoint payload"
    )
    assert payload["checkpoint_kind"] == PRIMARY_REPORT
    assert "optimizers" in payload, (
        "primary checkpoint must include optimizers (include_optimizers=True)"
    )
    assert_type(
        payload["q_networks"], list, context="primary checkpoint.q_networks"
    )
    assert_type(
        payload["target_networks"],
        list,
        context="primary checkpoint.target_networks",
    )
    assert len(payload["q_networks"]) == len(payload["target_networks"]), (
        "q_networks and target_networks must have equal length"
    )


def test_f3_best_eval_checkpoint_envelope_when_present(
    smoke_run_dir: Path,
) -> None:
    """F3b: best-eval checkpoint adds evaluation_summary with documented keys."""
    metadata = load_json(smoke_run_dir / "run_metadata.json")
    secondary_path_raw = metadata["checkpoint_files"].get("secondary_best_eval")
    if not secondary_path_raw:
        pytest.skip(
            "Best-eval checkpoint was not produced by this smoke run; "
            "F3b only runs when secondary_best_eval is present."
        )
    payload = _load_checkpoint(Path(secondary_path_raw))

    assert payload["format_version"] == 1
    assert_key_superset(
        payload,
        F3_REQUIRED_TOP_KEYS | {"evaluation_summary"},
        context="best-eval checkpoint payload",
    )
    assert payload["checkpoint_kind"] == SECONDARY_REPORT
    summary = payload["evaluation_summary"]
    assert_type(summary, dict, context="evaluation_summary")
    assert_key_set(
        summary,
        F3_EVALUATION_SUMMARY_REQUIRED_KEYS,
        context="evaluation_summary",
    )


# --- F4 -- manifest.replaySummary <-> summary.replay_timeline ---------


def _replay_summary_from_manifest(bundle_dir: Path) -> dict:
    manifest = load_json(bundle_dir / "manifest.json")
    replay = manifest.get("replaySummary")
    assert isinstance(replay, dict), (
        f"{bundle_dir}/manifest.json missing replaySummary"
    )
    return replay


def _replay_timeline_from_summary(bundle_dir: Path) -> dict:
    summary = load_json(bundle_dir / "evaluation" / "summary.json")
    replay = summary.get("replay_timeline")
    assert isinstance(replay, dict), (
        f"{bundle_dir}/evaluation/summary.json missing replay_timeline"
    )
    return replay


@pytest.mark.parametrize(
    "bundle_source",
    ["sample_bundle_dir", "smoke_bundle_dir"],
    ids=["fixture", "fresh-smoke-export"],
)
def test_f4_manifest_summary_cross_file_consistency(
    bundle_source: str, request: pytest.FixtureRequest
) -> None:
    """F4: manifest.replaySummary and evaluation.summary.replay_timeline equal.

    Slice B/C refactor work must preserve this: both files must
    serialize the same underlying data, even if they sit on different
    top-level casing conventions.
    """
    bundle_dir = request.getfixturevalue(bundle_source)
    manifest_replay = _replay_summary_from_manifest(bundle_dir)
    summary_replay = _replay_timeline_from_summary(bundle_dir)

    assert manifest_replay == summary_replay, (
        "manifest.replaySummary and evaluation.summary.replay_timeline "
        "must carry identical content.\n"
        f"Only-in-manifest: "
        f"{set(manifest_replay.keys()) - set(summary_replay.keys())}\n"
        f"Only-in-summary: "
        f"{set(summary_replay.keys()) - set(manifest_replay.keys())}"
    )


# --- F5 -- timeline row geometry invariants --------------------------


TIME_SEC_TOLERANCE = 1e-9
LOCAL_TANGENT_KM_TOLERANCE = 1e-6


def _beams_per_satellite(row: dict) -> int:
    num_sats = len(row["satelliteStates"])
    assert num_sats > 0, "row has no satelliteStates"
    num_beams = len(row["beamStates"])
    assert num_beams % num_sats == 0, (
        f"beamStates length {num_beams} not divisible by "
        f"satelliteStates length {num_sats}"
    )
    return num_beams // num_sats


@pytest.mark.parametrize(
    "bundle_source",
    ["sample_bundle_dir", "smoke_bundle_dir"],
    ids=["fixture", "fresh-smoke-export"],
)
def test_f5_timeline_row_geometry(
    bundle_source: str, request: pytest.FixtureRequest
) -> None:
    """F5: every timeline row satisfies Phase 03A geometry invariants."""
    bundle_dir = request.getfixturevalue(bundle_source)
    rows = load_timeline_rows(bundle_dir)
    assert rows, f"{bundle_dir}/timeline/step-trace.jsonl is empty"

    manifest = load_json(bundle_dir / "manifest.json")
    ground_point = manifest["coordinateFrame"]["groundPoint"]
    ground_lat = float(ground_point["latDeg"])
    ground_lon = float(ground_point["lonDeg"])

    config = load_json(bundle_dir / "config-resolved.json")
    slot_duration_s = float(config.get("baseline", {}).get("slot_duration_s"))

    last_slot_by_user: dict[str, int] = {}
    for i, row in enumerate(rows):
        context = f"timeline row {i}"

        assert row["beamCatalogOrder"] == "satellite-major-beam-minor", (
            f"{context}.beamCatalogOrder must be satellite-major-beam-minor, "
            f"got {row['beamCatalogOrder']!r}"
        )
        num_sats = len(row["satelliteStates"])
        bps = _beams_per_satellite(row)
        num_beams = len(row["beamStates"])
        assert num_beams == num_sats * bps, (
            f"{context}: beamStates ({num_beams}) != "
            f"satelliteStates ({num_sats}) * beams_per_satellite ({bps})"
        )
        assert len(row["actionValidityMask"]) == num_beams, (
            f"{context}: actionValidityMask length must equal beamStates"
        )
        assert len(row["visibilityMask"]) == num_beams, (
            f"{context}: visibilityMask length must equal beamStates"
        )

        selected = int(row["selectedServing"]["beamIndex"])
        assert 0 <= selected < num_beams, (
            f"{context}: selectedServing.beamIndex {selected} out of range"
        )
        assert bool(row["actionValidityMask"][selected]), (
            f"{context}: selectedServing.beamIndex {selected} not valid "
            f"under actionValidityMask"
        )

        assert row["handoverEvent"]["kind"] in VALID_HANDOVER_KINDS, (
            f"{context}: handoverEvent.kind {row['handoverEvent']['kind']!r} "
            f"not in {sorted(VALID_HANDOVER_KINDS)}"
        )

        slot_index = int(row["slotIndex"])
        assert slot_index >= 1, (
            f"{context}: slotIndex must be >= 1, got {slot_index}"
        )
        user_id = str(row["userId"])
        if user_id in last_slot_by_user:
            assert slot_index > last_slot_by_user[user_id], (
                f"{context}: slotIndex must be strictly monotonic per userId"
            )
        last_slot_by_user[user_id] = slot_index

        time_sec = float(row["timeSec"])
        expected_time = slot_index * slot_duration_s
        assert abs(time_sec - expected_time) < TIME_SEC_TOLERANCE, (
            f"{context}: timeSec {time_sec} inconsistent with "
            f"slotIndex * slot_duration_s = {expected_time}"
        )

        # Local-tangent offset must match recomputation from groundPoint.
        pos = row["userPosition"]
        east_actual = float(pos["localTangentKm"]["east"])
        north_actual = float(pos["localTangentKm"]["north"])
        east_expected, north_expected = local_tangent_offset_km(
            ground_lat,
            ground_lon,
            float(pos["latDeg"]),
            float(pos["lonDeg"]),
        )
        assert math.isclose(
            east_actual, east_expected, abs_tol=LOCAL_TANGENT_KM_TOLERANCE
        ), (
            f"{context}: userPosition.localTangentKm.east "
            f"{east_actual} != recomputed {east_expected}"
        )
        assert math.isclose(
            north_actual, north_expected, abs_tol=LOCAL_TANGENT_KM_TOLERANCE
        ), (
            f"{context}: userPosition.localTangentKm.north "
            f"{north_actual} != recomputed {north_expected}"
        )


def test_f5_slot_duration_config_matches_expectation(
    smoke_bundle_dir: Path,
) -> None:
    """F5 support: the smoke bundle runs with slot_duration_s == 1.0.

    This pins the assumption F5 relies on; if a future config overrides
    slot_duration_s, this test should fail loudly rather than have F5
    silently start passing under a different time scale.
    """
    config = load_json(smoke_bundle_dir / "config-resolved.json")
    slot = float(config.get("baseline", {}).get("slot_duration_s"))
    assert slot == SLOT_DURATION_S_EXPECTED, (
        f"slot_duration_s must be {SLOT_DURATION_S_EXPECTED} for the "
        f"baseline resolved template; got {slot}"
    )


# --- F6 -- fixture regeneration determinism ---------------------------


def _diff_directories(lhs: Path, rhs: Path) -> list[str]:
    """Return a list of human-readable differences between two directories."""
    lhs_files = {p.relative_to(lhs): p for p in iter_directory_files(lhs)}
    rhs_files = {p.relative_to(rhs): p for p in iter_directory_files(rhs)}

    only_lhs = set(lhs_files) - set(rhs_files)
    only_rhs = set(rhs_files) - set(lhs_files)

    diffs: list[str] = []
    for rel in sorted(only_lhs):
        diffs.append(f"only in expected: {rel}")
    for rel in sorted(only_rhs):
        diffs.append(f"only in actual: {rel}")
    for rel in sorted(set(lhs_files) & set(rhs_files)):
        if not filecmp.cmp(lhs_files[rel], rhs_files[rel], shallow=False):
            diffs.append(f"content differs: {rel}")
    return diffs


def test_f6_sample_bundle_regeneration_is_deterministic(
    tmp_path: Path,
) -> None:
    """F6: regenerating the sample bundle matches the checked-in fixture.

    This is the highest-risk family: if torch / numpy RNG paths or
    serialization orderings introduce non-determinism, this will
    surface it. Per SDD section 11 this test may be marked xfail with
    a status note if the non-determinism is unrelated to refactor
    scope.
    """
    if not SAMPLE_BUNDLE.exists():
        pytest.skip("sample bundle fixture missing; nothing to compare")
    if not GENERATE_SCRIPT.exists():
        pytest.skip("generate_sample_bundle.py missing")

    output_dir = tmp_path / "regenerated"
    # Deliberately omit --config so the script's default relative
    # `configs/modqn-paper-baseline.resolved-template.yaml` applies.
    # The checked-in fixture was generated with the default path, and
    # the resulting relative string is embedded into manifest.configPath
    # and summary.config_path. Passing an absolute path here would
    # silently diverge those fields even though training is
    # deterministic.
    completed = subprocess.run(
        [
            sys.executable,
            str(GENERATE_SCRIPT),
            "--output",
            str(output_dir),
            "--episodes",
            "1",
            "--max-users",
            "1",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert completed.returncode == 0, (
        "generate_sample_bundle.py failed:\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )

    diffs = _diff_directories(SAMPLE_BUNDLE, output_dir)
    assert not diffs, (
        "Regenerated sample bundle does not byte-match the fixture. "
        "If this is the first time running after a known non-deterministic "
        "upstream change, mark this test xfail with a status note per "
        "phase-04a SDD section 11.\n  " + "\n  ".join(diffs)
    )
