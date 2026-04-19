"""Phase 04B training-artifact model and I/O seam tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from modqn_paper_reproduction.artifacts import (
    CheckpointCatalog,
    CheckpointPayloadV1,
    CheckpointRuleV1,
    ResumeFromV1,
    RewardCalibrationV1,
    RunArtifactPaths,
    RunMetadataV1,
    RuntimeEnvironmentV1,
    SeedsBlock,
    TrainingLogRow,
    TrainingSummaryV1,
    read_checkpoint,
    read_run_metadata,
    read_training_log,
    write_checkpoint,
    write_run_metadata,
    write_training_log,
)
from modqn_paper_reproduction.cli import train_main


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = (
    REPO_ROOT / "configs" / "modqn-paper-baseline.resolved-template.yaml"
)
SAMPLE_BUNDLE = REPO_ROOT / "tests" / "fixtures" / "sample-bundle-v1"


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert set(left.keys()) == set(right.keys())
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for lhs, rhs in zip(left, right):
            _assert_nested_equal(lhs, rhs)
        return
    assert left == right


def _sample_run_metadata() -> RunMetadataV1:
    return RunMetadataV1(
        paper_id="PAP-2024-MORL-MULTIBEAM",
        package_version="0.1.0",
        config_path="configs/modqn-paper-baseline.resolved-template.yaml",
        config_role="resolved-run-template",
        resolved_config_snapshot={
            "config_role": "resolved-run-template",
            "baseline": {"users": 100},
        },
        training_experiment={"kind": "baseline", "experiment_id": "EXP-BASE-001"},
        seeds=SeedsBlock(
            train_seed=42,
            environment_seed=1337,
            mobility_seed=7,
            evaluation_seed_set=(100, 200, 300),
        ),
        checkpoint_rule=CheckpointRuleV1(
            assumption_id="ASSUME-MODQN-REP-015",
            primary_report="final-episode-policy",
            secondary_report="best-weighted-reward-on-eval",
            secondary_implemented=True,
            secondary_status="best-eval checkpoint available",
        ),
        reward_calibration=RewardCalibrationV1(
            enabled=False,
            mode="raw-unscaled",
            source="raw-unscaled",
            scales=(1.0, 1.0, 1.0),
            training_experiment_kind="baseline",
            training_experiment_id="",
            evaluation_metrics="raw-paper-metrics",
            checkpoint_selection_metric="raw-weighted-eval",
        ),
        checkpoint_files=CheckpointCatalog(
            primary_final=Path("artifacts/run-1/checkpoints/final-episode-policy.pt"),
            secondary_best_eval=Path(
                "artifacts/run-1/checkpoints/best-weighted-reward-on-eval.pt"
            ),
        ).to_v1(),
        resolved_assumptions={"seed_and_rng_policy": {"assumption_id": "A-001"}},
        runtime_environment=RuntimeEnvironmentV1(
            num_users=100,
            num_satellites=301,
            beams_per_satellite=7,
            user_lat_deg=25.0,
            user_lon_deg=121.0,
            r3_gap_scope="all-reachable-beams",
            r3_empty_beam_throughput=0.0,
            user_heading_stride_rad=2.3998277,
            user_scatter_radius_km=50.0,
            user_scatter_distribution="uniform-circular",
            user_area_width_km=0.0,
            user_area_height_km=0.0,
            mobility_model="deterministic-heading",
            random_wandering_max_turn_rad=0.0,
        ),
        trainer_config={
            "episodes": 1,
            "objective_weights": (0.5, 0.3, 0.2),
        },
        best_eval_summary={
            "episode": 0,
            "evaluation_every_episodes": 50,
            "eval_seeds": [100, 200, 300],
            "mean_scalar_reward": 1.0,
            "std_scalar_reward": 0.0,
            "mean_r1": 10.0,
            "std_r1": 0.0,
            "mean_r2": -1.0,
            "std_r2": 0.0,
            "mean_r3": -2.0,
            "std_r3": 0.0,
            "mean_total_handovers": 1.0,
            "std_total_handovers": 0.0,
        },
        resume_from=ResumeFromV1(
            path="artifacts/run-0/checkpoints/final-episode-policy.pt",
            checkpoint_kind="final-episode-policy",
            episode=0,
        ),
        training_summary=TrainingSummaryV1(
            episodes_requested=1,
            episodes_completed=1,
            elapsed_s=0.123,
            final_episode_index=0,
            final_scalar_reward=1.5,
        ),
    )


def _sample_checkpoint_payload() -> CheckpointPayloadV1:
    return CheckpointPayloadV1(
        format_version=1,
        checkpoint_kind="final-episode-policy",
        episode=0,
        train_seed=42,
        env_seed=1337,
        mobility_seed=7,
        state_dim=15,
        action_dim=21,
        trainer_config={
            "objective_weights": (0.5, 0.3, 0.2),
            "reward_calibration_scales": (1.0, 1.0, 1.0),
        },
        checkpoint_rule=CheckpointRuleV1(
            assumption_id="ASSUME-MODQN-REP-015",
            primary_report="final-episode-policy",
            secondary_report="best-weighted-reward-on-eval",
            secondary_implemented=True,
            secondary_status="best-eval checkpoint available",
        ),
        q_networks=[{"weight": torch.tensor([1.0, 2.0, 3.0])}],
        target_networks=[{"weight": torch.tensor([4.0, 5.0, 6.0])}],
        optimizers=[{"lr": 0.01}],
        last_episode_log={
            "episode": 0,
            "epsilon": 1.0,
            "losses": (0.1, 0.2, 0.3),
        },
        evaluation_summary={
            "eval_seeds": [100, 200],
            "episode": 0,
            "mean_scalar_reward": 1.0,
        },
    )


@pytest.fixture(scope="module")
def smoke_run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    run_dir = tmp_path_factory.mktemp("artifacts-models") / "run"
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
    assert rc == 0
    return run_dir


def test_run_metadata_round_trip_and_paths() -> None:
    metadata = _sample_run_metadata()
    reloaded = RunMetadataV1.from_dict(metadata.to_dict())
    assert reloaded == metadata

    run_paths = RunArtifactPaths(Path("artifacts/run-1"))
    assert run_paths.training_log_json == Path("artifacts/run-1/training_log.json")
    assert run_paths.run_metadata_json == Path("artifacts/run-1/run_metadata.json")
    assert run_paths.primary_checkpoint(metadata.checkpoint_rule) == Path(
        "artifacts/run-1/checkpoints/final-episode-policy.pt"
    )


def test_training_log_row_round_trip() -> None:
    row = TrainingLogRow(
        episode=0,
        epsilon=1.0,
        r1_mean=1.1,
        r2_mean=-0.2,
        r3_mean=-0.3,
        scalar_reward=0.4,
        total_handovers=5,
        replay_size=10,
        losses=(0.01, 0.02, 0.03),
    )
    assert TrainingLogRow.from_dict(row.to_dict()) == row


def test_checkpoint_payload_round_trip() -> None:
    payload = _sample_checkpoint_payload()
    round_trip = CheckpointPayloadV1.from_dict(payload.to_dict())
    _assert_nested_equal(round_trip.to_dict(), payload.to_dict())


def test_run_metadata_io_round_trip(tmp_path: Path) -> None:
    metadata = _sample_run_metadata()
    metadata_path = tmp_path / "run_metadata.json"
    write_run_metadata(metadata_path, metadata)
    loaded = read_run_metadata(metadata_path)
    assert loaded == metadata
    assert metadata_path.read_text() == json.dumps(metadata.to_dict(), indent=2)


def test_training_log_io_round_trip(tmp_path: Path) -> None:
    rows = [
        TrainingLogRow(
            episode=0,
            epsilon=1.0,
            r1_mean=1.0,
            r2_mean=-0.1,
            r3_mean=-0.2,
            scalar_reward=0.3,
            total_handovers=4,
            replay_size=8,
            losses=(0.01, 0.02, 0.03),
        )
    ]
    path = tmp_path / "training_log.json"
    write_training_log(path, rows)
    assert read_training_log(path) == rows


def test_checkpoint_io_round_trip(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    payload = _sample_checkpoint_payload()
    write_checkpoint(checkpoint_path, payload)
    reloaded = read_checkpoint(checkpoint_path)
    _assert_nested_equal(reloaded.to_dict(), payload.to_dict())


def test_sample_bundle_best_eval_summary_fits_run_metadata_shape() -> None:
    summary_payload = json.loads(
        (SAMPLE_BUNDLE / "evaluation" / "summary.json").read_text()
    )
    metadata = _sample_run_metadata()
    metadata_with_fixture_summary = RunMetadataV1(
        paper_id=metadata.paper_id,
        package_version=metadata.package_version,
        config_path=metadata.config_path,
        config_role=metadata.config_role,
        resolved_config_snapshot=metadata.resolved_config_snapshot,
        training_experiment=metadata.training_experiment,
        seeds=metadata.seeds,
        checkpoint_rule=metadata.checkpoint_rule,
        reward_calibration=metadata.reward_calibration,
        checkpoint_files=metadata.checkpoint_files,
        resolved_assumptions=metadata.resolved_assumptions,
        runtime_environment=metadata.runtime_environment,
        trainer_config=metadata.trainer_config,
        best_eval_summary=summary_payload["best_eval_summary"],
        resume_from=metadata.resume_from,
        training_summary=metadata.training_summary,
    )
    assert (
        metadata_with_fixture_summary.to_dict()["best_eval_summary"]
        == summary_payload["best_eval_summary"]
    )


def test_train_main_run_metadata_reserializes_byte_equal(
    smoke_run_dir: Path,
    tmp_path: Path,
) -> None:
    metadata_path = smoke_run_dir / "run_metadata.json"
    original = metadata_path.read_text()
    metadata = read_run_metadata(metadata_path)

    rewritten_path = tmp_path / "run_metadata.json"
    write_run_metadata(rewritten_path, metadata)
    assert rewritten_path.read_text() == original


def test_train_main_checkpoint_round_trip_semantics(
    smoke_run_dir: Path,
    tmp_path: Path,
) -> None:
    metadata = read_run_metadata(smoke_run_dir / "run_metadata.json")
    checkpoint_path = Path(metadata.checkpoint_files.primary_final)
    payload = read_checkpoint(checkpoint_path)

    rewritten_path = tmp_path / "checkpoint.pt"
    write_checkpoint(rewritten_path, payload)
    reloaded = read_checkpoint(rewritten_path)
    _assert_nested_equal(reloaded.to_dict(), payload.to_dict())
