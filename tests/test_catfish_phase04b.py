from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from modqn_paper_reproduction.cli import train_main
from modqn_paper_reproduction.config_loader import (
    ConfigValidationError,
    build_trainer_config,
    get_seeds,
    load_training_yaml,
)
from modqn_paper_reproduction.runtime.catfish_replay import (
    quality_score,
    sample_mixed_replay_batch,
)
from modqn_paper_reproduction.runtime.replay_buffer import ReplayBuffer


BASELINE_CONFIG = "configs/modqn-paper-baseline.resolved-template.yaml"
PAPER_ENVELOPE_CONFIG = "configs/modqn-paper-baseline.yaml"
PRIMARY_CONFIG = "configs/catfish-modqn-phase-04-b-primary-shaping-off.resolved.yaml"
CONTROL_CONFIG = "configs/catfish-modqn-phase-04-b-control.resolved.yaml"
NO_INTERVENTION_CONFIG = (
    "configs/catfish-modqn-phase-04-b-no-intervention.resolved.yaml"
)
NO_ASYMMETRIC_GAMMA_CONFIG = (
    "configs/catfish-modqn-phase-04-b-no-asymmetric-gamma.resolved.yaml"
)


def test_baseline_loads_as_baseline_and_catfish_is_opt_in() -> None:
    cfg = load_training_yaml(BASELINE_CONFIG)
    trainer_cfg = build_trainer_config(cfg)

    assert trainer_cfg.training_experiment_kind == "baseline"
    assert trainer_cfg.method_family == "MODQN-baseline"
    assert trainer_cfg.r1_reward_mode == "throughput"
    assert trainer_cfg.catfish_enabled is False
    assert trainer_cfg.catfish_intervention_enabled is False


def test_paper_envelope_still_rejected_for_training() -> None:
    with pytest.raises(ConfigValidationError):
        load_training_yaml(PAPER_ENVELOPE_CONFIG)


def test_phase04b_catfish_configs_validate_only_under_phase04b_kind() -> None:
    cfg = load_training_yaml(PRIMARY_CONFIG)
    trainer_cfg = build_trainer_config(cfg)
    assert trainer_cfg.training_experiment_kind == (
        "phase-04-b-single-catfish-feasibility"
    )
    assert trainer_cfg.method_family == "Catfish-MODQN"
    assert trainer_cfg.catfish_enabled is True

    wrong_kind = copy.deepcopy(cfg)
    wrong_kind["training_experiment"]["kind"] = "baseline"
    with pytest.raises((ConfigValidationError, ValueError)):
        build_trainer_config(wrong_kind)

    wrong_method = copy.deepcopy(cfg)
    wrong_method["training_experiment"]["method_family"] = "MODQN-control"
    with pytest.raises((ConfigValidationError, ValueError)):
        build_trainer_config(wrong_method)


def test_phase04b_rejects_ee_reward_modes() -> None:
    cfg = load_training_yaml(PRIMARY_CONFIG)
    cfg["training_experiment"]["phase_04_b_single_catfish"][
        "r1_reward_mode"
    ] = "per-user-ee-credit"
    with pytest.raises((ConfigValidationError, ValueError)):
        build_trainer_config(cfg)


def test_primary_shaping_off_and_required_ablations_are_configured() -> None:
    primary = load_training_yaml(PRIMARY_CONFIG)
    primary_trainer = build_trainer_config(primary)
    assert primary_trainer.catfish_competitive_shaping_enabled is False
    assert primary_trainer.catfish_intervention_enabled is True
    assert primary_trainer.catfish_discount_factor > primary_trainer.discount_factor

    no_intervention = build_trainer_config(load_training_yaml(NO_INTERVENTION_CONFIG))
    assert no_intervention.catfish_ablation == "no-intervention"
    assert no_intervention.catfish_enabled is True
    assert no_intervention.catfish_intervention_enabled is False

    no_asymmetric = build_trainer_config(load_training_yaml(NO_ASYMMETRIC_GAMMA_CONFIG))
    assert no_asymmetric.catfish_ablation == "no-asymmetric-gamma"
    assert no_asymmetric.catfish_enabled is True
    assert no_asymmetric.catfish_intervention_enabled is True
    assert no_asymmetric.catfish_discount_factor == no_asymmetric.discount_factor


def test_phase04b_control_and_primary_share_protocol() -> None:
    primary = load_training_yaml(PRIMARY_CONFIG)
    control = load_training_yaml(CONTROL_CONFIG)
    primary_trainer = build_trainer_config(primary)
    control_trainer = build_trainer_config(control)

    assert get_seeds(primary) == get_seeds(control)
    assert primary_trainer.episodes == control_trainer.episodes == 20
    assert primary_trainer.batch_size == control_trainer.batch_size == 64
    assert primary_trainer.target_update_every_episodes == (
        control_trainer.target_update_every_episodes
    )
    assert primary_trainer.checkpoint_primary_report == (
        control_trainer.checkpoint_primary_report
    )
    assert primary_trainer.checkpoint_secondary_report == (
        control_trainer.checkpoint_secondary_report
    )
    assert primary_trainer.r1_reward_mode == control_trainer.r1_reward_mode
    assert control_trainer.method_family == "MODQN-control"
    assert control_trainer.catfish_enabled is False


def test_quality_score_uses_configured_r1_r2_r3_weights() -> None:
    cfg = build_trainer_config(load_training_yaml(PRIMARY_CONFIG))
    reward = np.array([10.0, -2.0, -4.0], dtype=np.float64)
    assert quality_score(reward, cfg.catfish_quality_weights) == pytest.approx(
        0.5 * 10.0 + 0.3 * -2.0 + 0.2 * -4.0
    )


def test_mixed_replay_batch_reports_actual_composition() -> None:
    main = ReplayBuffer(capacity=100)
    catfish = ReplayBuffer(capacity=100)
    for idx in range(60):
        main.push(*_transition(idx))
    for idx in range(40):
        catfish.push(*_transition(1000 + idx))

    batch = sample_mixed_replay_batch(
        main_replay=main,
        catfish_replay=catfish,
        batch_size=10,
        catfish_ratio=0.30,
        rng=np.random.default_rng(42),
    )

    assert batch.states.shape == (10, 4)
    assert batch.composition["configured_catfish_ratio"] == 0.30
    assert batch.composition["target_catfish_sample_count"] == 3
    assert batch.composition["actual_catfish_sample_count"] == 3
    assert batch.composition["actual_main_sample_count"] == 7
    assert batch.composition["actual_catfish_ratio"] == pytest.approx(0.3)


def test_phase04b_bounded_smoke_writes_metadata_and_diagnostics(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "catfish-smoke"
    rc = train_main(
        [
            "--config",
            PRIMARY_CONFIG,
            "--episodes",
            "1",
            "--progress-every",
            "0",
            "--output-dir",
            str(run_dir),
        ]
    )
    assert rc == 0

    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    diagnostics = json.loads((run_dir / "catfish_diagnostics.json").read_text())
    training_log = json.loads((run_dir / "training_log.json").read_text())

    assert metadata["training_experiment"]["kind"] == (
        "phase-04-b-single-catfish-feasibility"
    )
    assert metadata["training_experiment"]["method_family"] == "Catfish-MODQN"
    assert metadata["trainer_config"]["catfish_enabled"] is True
    assert metadata["trainer_config"]["r1_reward_mode"] == "throughput"
    assert (
        metadata["resolved_assumptions"]["phase_04_b_catfish_local_assumptions"][
            "value"
        ]["modqn_quality_score_adaptation"]["weights"]
        == {"r1": 0.5, "r2": 0.3, "r3": 0.2}
    )

    assert len(training_log) == 1
    assert diagnostics["final_replay"]["catfish_replay_size"] > 0
    assert diagnostics["cumulative"]["intervention_trigger_count"] > 0
    assert (
        diagnostics["cumulative"]["actual_catfish_samples_used_in_main_updates"]
        > 0
    )
    assert diagnostics["cumulative"]["nan_detected"] is False
    assert diagnostics["final_replay"]["quality_threshold"] is not None
    assert (
        diagnostics["final_replay"][
            "catfish_replay_reward_component_distribution"
        ]["r1"]["count"]
        > 0
    )
    assert (
        diagnostics["episode_diagnostics"][-1]["replay_starvation"][
            "catfish_replay_empty_after_warmup"
        ]
        is False
    )


def _transition(idx: int) -> tuple:
    state = np.full(4, float(idx), dtype=np.float32)
    next_state = state + 1.0
    reward = np.array([idx, -idx % 3, -idx % 5], dtype=np.float32)
    mask = np.ones(3, dtype=bool)
    next_mask = np.ones(3, dtype=bool)
    return state, idx % 3, reward, next_state, mask, next_mask, False
