from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from modqn_paper_reproduction.algorithms import apply_reward_calibration
from modqn_paper_reproduction.cli import (
    atmospheric_sign_counterfactual_main,
    beam_counterfactual_main,
    beam_semantics_audit_main,
    export_main,
    train_main,
)
from modqn_paper_reproduction.config_loader import (
    ConfigValidationError,
    build_environment,
    build_trainer_config,
    load_training_yaml,
    load_yaml,
)


RESOLVED_CONFIG = "configs/modqn-paper-baseline.resolved-template.yaml"
FOLLOW_ON_CONFIG = "configs/modqn-paper-baseline.paper-faithful-follow-on.resolved.yaml"
BEAM_AWARE_FOLLOW_ON_CONFIG = (
    "configs/modqn-paper-baseline.beam-aware-eligibility-follow-on.resolved.yaml"
)
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
    assert metadata["resolved_config_snapshot"]["config_role"] == "resolved-run-template"
    assert metadata["resolved_config_snapshot"]["baseline"]["users"] == 100
    assert metadata["runtime_environment"]["r3_gap_scope"] == "all-reachable-beams"
    assert (
        metadata["runtime_environment"]["action_mask_eligibility_mode"]
        == "satellite-visible-all-beams"
    )
    assert metadata["runtime_environment"]["user_heading_stride_rad"] == 2.3998277
    assert metadata["runtime_environment"]["user_scatter_radius_km"] == 50.0
    assert metadata["runtime_environment"]["user_scatter_distribution"] == "uniform-circular"
    assert metadata["runtime_environment"]["mobility_model"] == "deterministic-heading"


def test_follow_on_config_drives_rectangle_area_and_random_wandering() -> None:
    cfg = load_training_yaml(FOLLOW_ON_CONFIG)
    env = build_environment(cfg)

    assert env.config.user_lat_deg == 40.0
    assert env.config.user_lon_deg == 116.0
    assert env.config.user_scatter_distribution == "uniform-rectangle"
    assert env.config.user_area_width_km == 200.0
    assert env.config.user_area_height_km == 90.0
    assert env.config.mobility_model == "random-wandering"

    rng = np.random.default_rng(42)
    states, masks, _ = env.reset(rng, np.random.default_rng(7))
    assert min(mask.num_valid for mask in masks) > 0
    actions = np.array([
        int(np.where(mask.mask)[0][0]) for mask in masks
    ], dtype=np.int32)
    result = env.step(actions, rng)
    assert any(reward.r1_throughput > 0.0 for reward in result.rewards)


def test_follow_on_train_cli_writes_scenario_runtime_metadata(tmp_path: Path) -> None:
    out_dir = tmp_path / "follow-on-run"
    rc = train_main([
        "--config",
        FOLLOW_ON_CONFIG,
        "--episodes",
        "1",
        "--progress-every",
        "0",
        "--output-dir",
        str(out_dir),
    ])
    assert rc == 0

    metadata = json.loads((out_dir / "run_metadata.json").read_text())
    assert metadata["resolved_config_snapshot"]["config_role"] == "resolved-run-follow-on"
    runtime = metadata["runtime_environment"]
    assert runtime["user_lat_deg"] == 40.0
    assert runtime["user_lon_deg"] == 116.0
    assert runtime["user_scatter_distribution"] == "uniform-rectangle"
    assert runtime["user_area_width_km"] == 200.0
    assert runtime["user_area_height_km"] == 90.0
    assert runtime["mobility_model"] == "random-wandering"
    assert runtime["random_wandering_max_turn_rad"] > 0.0


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


def test_beam_aware_follow_on_config_drives_nearest_beam_masks() -> None:
    cfg = load_training_yaml(BEAM_AWARE_FOLLOW_ON_CONFIG)
    env = build_environment(cfg)
    assert env.config.action_mask_eligibility_mode == "nearest-beam-per-visible-satellite"

    rng = np.random.default_rng(42)
    _, masks, _ = env.reset(rng, np.random.default_rng(7))
    first_mask = masks[0].mask
    for sat_index in range(env.orbit.num_satellites):
        block = first_mask[sat_index * env.beam_pattern.num_beams: (sat_index + 1) * env.beam_pattern.num_beams]
        assert int(np.sum(block)) in {0, 1}


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


def test_beam_aware_follow_on_train_and_export_disclose_mask_mode(tmp_path: Path) -> None:
    run_dir = tmp_path / "beam-aware-run"
    export_dir = tmp_path / "beam-aware-export"

    rc = train_main([
        "--config",
        BEAM_AWARE_FOLLOW_ON_CONFIG,
        "--episodes",
        "1",
        "--progress-every",
        "0",
        "--output-dir",
        str(run_dir),
    ])
    assert rc == 0

    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["config_role"] == "resolved-run-follow-on"
    assert metadata["resolved_config_snapshot"]["track"]["phase"] == "phase-01f"
    assert (
        metadata["resolved_assumptions"]["action_masking_semantics"]["value"]["eligibility_mode"]
        == "nearest-beam-per-visible-satellite"
    )
    assert (
        metadata["runtime_environment"]["action_mask_eligibility_mode"]
        == "nearest-beam-per-visible-satellite"
    )

    export_rc = export_main([
        "--input",
        str(run_dir),
        "--output-dir",
        str(export_dir),
    ])
    assert export_rc == 0

    manifest = json.loads((export_dir / "manifest.json").read_text())
    assert (
        manifest["scenarioSurface"]["actionMaskEligibilityMode"]
        == "nearest-beam-per-visible-satellite"
    )


def test_beam_semantics_audit_cli_writes_expected_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    audit_dir = tmp_path / "beam-audit"

    rc = train_main([
        "--config",
        RESOLVED_CONFIG,
        "--episodes",
        "1",
        "--progress-every",
        "0",
        "--output-dir",
        str(run_dir),
    ])
    assert rc == 0

    audit_rc = beam_semantics_audit_main([
        "--input",
        str(run_dir),
        "--output-dir",
        str(audit_dir),
        "--evaluation-seed",
        "100",
        "--max-steps",
        "1",
        "--max-users",
        "2",
    ])
    assert audit_rc == 0

    summary_path = audit_dir / "beam_semantics_summary.json"
    beam_csv_path = audit_dir / "beam_tie_metrics.csv"
    decision_csv_path = audit_dir / "decision_margin_metrics.csv"
    review_path = audit_dir / "review.md"

    assert summary_path.exists()
    assert beam_csv_path.exists()
    assert decision_csv_path.exists()
    assert review_path.exists()

    summary = json.loads(summary_path.read_text())
    assert summary["effective_audit_surface"]["steps_audited"] == 1
    assert summary["effective_audit_surface"]["users_audited_per_step"] == 2
    assert summary["decision_rows_audited"] == 2
    assert summary["beam_block_rows_audited"] >= 2
    assert summary["evaluation_seed"] == 100
    assert summary["checkpoint_kind"] == "best-weighted-reward-on-eval"
    assert summary["valid_mask_collapse_classification"] in {
        "absent",
        "rare",
        "common",
        "pervasive",
    }
    assert "without retraining" in review_path.read_text()


def test_beam_counterfactual_cli_writes_expected_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    counter_dir = tmp_path / "beam-counterfactual"

    rc = train_main([
        "--config",
        RESOLVED_CONFIG,
        "--episodes",
        "1",
        "--progress-every",
        "0",
        "--output-dir",
        str(run_dir),
    ])
    assert rc == 0

    counter_rc = beam_counterfactual_main([
        "--input",
        str(run_dir),
        "--output-dir",
        str(counter_dir),
        "--evaluation-seed",
        "100",
        "--max-steps",
        "1",
        "--max-users",
        "2",
    ])
    assert counter_rc == 0

    summary_path = counter_dir / "counterfactual_eval_summary.json"
    comparison_path = counter_dir / "counterfactual_vs_baseline.csv"
    review_path = counter_dir / "review.md"

    assert summary_path.exists()
    assert comparison_path.exists()
    assert review_path.exists()

    summary = json.loads(summary_path.read_text())
    assert summary["counterfactual_mode"] == "nearest-beam-per-visible-satellite"
    assert summary["evaluation_seed"] == 100
    assert summary["same_state_decision_comparison"]["steps_audited"] == 1
    assert summary["same_state_decision_comparison"]["users_compared_per_step"] == 2
    assert summary["baseline_eval"]["modqn"]["steps_audited"] == 1
    assert summary["counterfactual_eval"]["rss_max"]["steps_audited"] == 1
    assert summary["interpretation"]["modqn_change_scope"] in {
        "absent",
        "mostly-tie-break",
        "material",
    }
    assert "counterfactual" in review_path.read_text().lower()


def test_atmospheric_sign_counterfactual_cli_writes_expected_artifacts(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    counter_dir = tmp_path / "atmospheric-counterfactual"

    rc = train_main([
        "--config",
        RESOLVED_CONFIG,
        "--episodes",
        "1",
        "--progress-every",
        "0",
        "--output-dir",
        str(run_dir),
    ])
    assert rc == 0

    counter_rc = atmospheric_sign_counterfactual_main([
        "--input",
        str(run_dir),
        "--output-dir",
        str(counter_dir),
        "--evaluation-seed",
        "100",
        "--max-steps",
        "1",
        "--max-users",
        "2",
    ])
    assert counter_rc == 0

    summary_path = counter_dir / "atmospheric_sign_counterfactual_summary.json"
    comparison_path = counter_dir / "atmospheric_sign_vs_baseline.csv"
    diagnostics_path = counter_dir / "reward_geometry_diagnostics_comparison.csv"
    review_path = counter_dir / "review.md"

    assert summary_path.exists()
    assert comparison_path.exists()
    assert diagnostics_path.exists()
    assert review_path.exists()

    summary = json.loads(summary_path.read_text())
    assert summary["baseline_mode"] == "paper-published"
    assert summary["counterfactual_mode"] == "corrected-lossy"
    assert summary["evaluation_seed"] == 100
    assert summary["same_geometry_decision_comparison"]["steps_audited"] == 1
    assert summary["same_geometry_decision_comparison"]["users_compared_per_step"] == 2
    assert summary["baseline_eval"]["modqn"]["steps_audited"] == 1
    assert summary["counterfactual_eval"]["rss_max"]["steps_audited"] == 1
    assert summary["interpretation"]["diagnostics_change_scope"] in {
        "absent",
        "notable",
        "material",
    }
    assert summary["interpretation"]["modqn_change_scope"] in {
        "absent",
        "notable",
        "material",
    }
    assert "without retraining" in review_path.read_text()
