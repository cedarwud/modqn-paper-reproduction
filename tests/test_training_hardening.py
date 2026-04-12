from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from modqn_paper_reproduction.algorithms import apply_reward_calibration
from modqn_paper_reproduction.cli import train_main
from modqn_paper_reproduction.config_loader import (
    ConfigValidationError,
    build_environment,
    build_trainer_config,
    load_training_yaml,
    load_yaml,
)


RESOLVED_CONFIG = "configs/modqn-paper-baseline.resolved-template.yaml"
REWARD_CALIBRATION_CONFIG = "configs/modqn-paper-baseline.reward-calibration.resolved.yaml"
PAPER_ENVELOPE_CONFIG = "configs/modqn-paper-baseline.yaml"


def _resolved_cfg() -> dict:
    return copy.deepcopy(load_yaml(RESOLVED_CONFIG))


def test_training_loader_rejects_paper_envelope() -> None:
    try:
        load_training_yaml(PAPER_ENVELOPE_CONFIG)
    except ConfigValidationError as exc:
        assert "resolved-run config" in str(exc)
    else:
        raise AssertionError("paper-envelope training input should be rejected")


def test_train_cli_rejects_paper_envelope(capsys) -> None:
    rc = train_main([
        "--config",
        PAPER_ENVELOPE_CONFIG,
        "--episodes",
        "1",
        "--progress-every",
        "0",
    ])
    captured = capsys.readouterr()
    assert rc == 2
    assert "resolved-run config" in captured.err


def test_resolved_config_drives_r3_gap_runtime() -> None:
    base_cfg = _resolved_cfg()
    base_cfg["baseline"]["users"] = 5

    env = build_environment(base_cfg)
    rng = np.random.default_rng(42)
    env.reset(rng, np.random.default_rng(7))
    all_on_one_beam = np.zeros(5, dtype=np.int32)
    result = env.step(all_on_one_beam, rng)
    assert result.rewards[0].r3_load_balance < 0.0

    occupied_only_cfg = _resolved_cfg()
    occupied_only_cfg["baseline"]["users"] = 5
    occupied_only_cfg["resolved_assumptions"]["r3_gap_beam_scope"]["value"]["scope"] = (
        "occupied-beams-only"
    )

    occupied_env = build_environment(occupied_only_cfg)
    occupied_rng = np.random.default_rng(42)
    occupied_env.reset(occupied_rng, np.random.default_rng(7))
    occupied_result = occupied_env.step(all_on_one_beam, occupied_rng)
    assert occupied_result.rewards[0].r3_load_balance == 0.0


def test_resolved_config_drives_heading_stride_and_scatter_radius() -> None:
    cfg = _resolved_cfg()
    cfg["baseline"]["users"] = 2
    cfg["resolved_assumptions"]["user_heading_stride"]["value"]["stride_rad"] = 0.0
    cfg["resolved_assumptions"]["user_scatter_radius"]["value"]["radius_km"] = 0.0

    env = build_environment(cfg)
    rng = np.random.default_rng(42)
    states, _, _ = env.reset(rng, np.random.default_rng(7))

    assert env._user_positions[0] == (0.0, 0.0)
    assert env._user_positions[1] == (0.0, 0.0)

    actions = np.array([
        int(np.argmax(s.access_vector)) for s in states
    ], dtype=np.int32)
    env.step(actions, np.random.default_rng(99))
    assert env._user_positions[0] == env._user_positions[1]


def test_train_cli_writes_final_checkpoint_and_metadata(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    rc = train_main([
        "--config",
        RESOLVED_CONFIG,
        "--episodes",
        "1",
        "--progress-every",
        "0",
        "--output-dir",
        str(out_dir),
    ])
    assert rc == 0

    checkpoint_path = out_dir / "checkpoints" / "final-episode-policy.pt"
    best_eval_checkpoint_path = out_dir / "checkpoints" / "best-weighted-reward-on-eval.pt"
    metadata_path = out_dir / "run_metadata.json"
    log_path = out_dir / "training_log.json"

    assert checkpoint_path.exists()
    assert best_eval_checkpoint_path.exists()
    assert metadata_path.exists()
    assert log_path.exists()

    metadata = json.loads(metadata_path.read_text())
    assert metadata["checkpoint_rule"]["assumption_id"] == "ASSUME-MODQN-REP-015"
    assert metadata["checkpoint_rule"]["primary_report"] == "final-episode-policy"
    assert metadata["checkpoint_rule"]["secondary_report"] == (
        "best-weighted-reward-on-eval"
    )
    assert metadata["checkpoint_rule"]["secondary_implemented"] is True
    assert metadata["checkpoint_files"]["secondary_best_eval"] == str(best_eval_checkpoint_path)
    assert metadata["best_eval_summary"] is not None
    assert metadata["runtime_environment"]["r3_gap_scope"] == "all-reachable-beams"
    assert metadata["runtime_environment"]["user_heading_stride_rad"] == 2.3998277
    assert metadata["runtime_environment"]["user_scatter_radius_km"] == 50.0


def test_reward_calibration_experiment_is_opt_in() -> None:
    baseline_cfg = load_training_yaml(RESOLVED_CONFIG)
    baseline_trainer_cfg = build_trainer_config(baseline_cfg)
    assert baseline_trainer_cfg.reward_calibration_enabled is False
    assert baseline_trainer_cfg.reward_calibration_mode == "raw-unscaled"

    experiment_cfg = load_training_yaml(REWARD_CALIBRATION_CONFIG)
    experiment_trainer_cfg = build_trainer_config(experiment_cfg)
    assert experiment_trainer_cfg.training_experiment_kind == "reward-calibration"
    assert experiment_trainer_cfg.training_experiment_id == "EXP-MODQN-CAL-001"
    assert experiment_trainer_cfg.reward_calibration_enabled is True
    assert experiment_trainer_cfg.reward_calibration_mode == "divide-by-fixed-scales"
    np.testing.assert_allclose(
        experiment_trainer_cfg.reward_calibration_scales,
        np.array([491.28614764527117, 0.5, 17.54593384447397], dtype=np.float64),
    )

    raw = np.array([491.28614764527117, -0.5, -17.54593384447397], dtype=np.float64)
    np.testing.assert_allclose(
        apply_reward_calibration(raw, experiment_trainer_cfg),
        np.array([1.0, -1.0, -1.0], dtype=np.float64),
    )
    np.testing.assert_allclose(
        apply_reward_calibration(raw, baseline_trainer_cfg),
        raw,
    )


def test_train_cli_writes_reward_calibration_experiment_metadata(tmp_path: Path) -> None:
    out_dir = tmp_path / "reward-calibration-run"
    rc = train_main([
        "--config",
        REWARD_CALIBRATION_CONFIG,
        "--episodes",
        "1",
        "--progress-every",
        "0",
        "--output-dir",
        str(out_dir),
    ])
    assert rc == 0

    metadata = json.loads((out_dir / "run_metadata.json").read_text())
    assert metadata["config_role"] == "resolved-run-experiment"
    assert metadata["training_experiment"]["kind"] == "reward-calibration"
    assert metadata["training_experiment"]["experiment_id"] == "EXP-MODQN-CAL-001"
    assert metadata["reward_calibration"]["enabled"] is True
    assert metadata["reward_calibration"]["mode"] == "divide-by-fixed-scales"
    assert metadata["reward_calibration"]["training_experiment_kind"] == "reward-calibration"
    assert metadata["trainer_config"]["reward_calibration_enabled"] is True
