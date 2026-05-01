from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from modqn_paper_reproduction.algorithms.multi_catfish_modqn import (
    MultiCatfishMODQNTrainer,
)
from modqn_paper_reproduction.analysis.catfish_phase05b_bounded_pilot import (
    PHASE_05B_CONFIGS,
    PHASE_05B_EVAL_SEEDS,
    validate_phase05b_training_config,
)
from modqn_paper_reproduction.config_loader import (
    ConfigValidationError,
    build_environment,
    build_trainer_config,
    get_seeds,
    load_training_yaml,
)
from modqn_paper_reproduction.runtime.catfish_replay import (
    sample_multi_source_replay_batch,
    target_source_sample_counts,
)
from modqn_paper_reproduction.runtime.replay_buffer import ReplayBuffer


PRIMARY_CONFIG = (
    "configs/catfish-modqn-phase-05b-primary-multi-catfish-shaping-off.resolved.yaml"
)
SINGLE_CONFIG = (
    "configs/catfish-modqn-phase-05b-single-catfish-equal-budget.resolved.yaml"
)
CONTROL_CONFIG = "configs/catfish-modqn-phase-05b-modqn-control.resolved.yaml"
MULTI_BUFFER_CONFIG = (
    "configs/catfish-modqn-phase-05b-multi-buffer-single-learner.resolved.yaml"
)
RANDOM_CONFIG = (
    "configs/catfish-modqn-phase-05b-random-or-uniform-buffer-control.resolved.yaml"
)


def test_phase05b_configs_validate_only_in_allowed_namespace() -> None:
    for config_path in PHASE_05B_CONFIGS:
        assert Path(config_path).name.startswith("catfish-modqn-phase-05b-")
        cfg = load_training_yaml(config_path)
        trainer_cfg = build_trainer_config(cfg)
        phase_block = validate_phase05b_training_config(cfg, trainer_cfg)

        assert trainer_cfg.episodes == 20
        assert trainer_cfg.target_update_every_episodes == 5
        assert get_seeds(cfg)["evaluation_seed_set"] == PHASE_05B_EVAL_SEEDS
        assert len(phase_block["seed_triplets"]) >= 3
        assert trainer_cfg.r1_reward_mode == "throughput"
        assert trainer_cfg.reward_calibration_enabled is False


def test_phase05b_rejects_ee_and_reward_surface_changes() -> None:
    cfg = load_training_yaml(PRIMARY_CONFIG)
    cfg["training_experiment"]["phase_05b_multi_catfish"][
        "r1_reward_mode"
    ] = "per-user-ee-credit"
    with pytest.raises((ConfigValidationError, ValueError)):
        build_trainer_config(cfg)

    cfg = load_training_yaml(PRIMARY_CONFIG)
    cfg["training_experiment"]["phase_05b_multi_catfish"]["reward_surface"][
        "r1"
    ] = "energy-efficiency"
    with pytest.raises(ConfigValidationError):
        build_trainer_config(cfg)

    cfg = load_training_yaml(PRIMARY_CONFIG)
    cfg["training_experiment"]["method_family"] = "Catfish-EE-MODQN"
    with pytest.raises((ConfigValidationError, ValueError)):
        build_trainer_config(cfg)


def test_phase05b_total_catfish_budget_equality() -> None:
    primary = build_trainer_config(load_training_yaml(PRIMARY_CONFIG))
    single = build_trainer_config(load_training_yaml(SINGLE_CONFIG))
    multi_buffer = build_trainer_config(load_training_yaml(MULTI_BUFFER_CONFIG))
    random_control = build_trainer_config(load_training_yaml(RANDOM_CONFIG))
    control = build_trainer_config(load_training_yaml(CONTROL_CONFIG))

    assert control.catfish_total_intervention_ratio == 0.0
    for cfg in (primary, single, multi_buffer, random_control):
        assert cfg.catfish_total_intervention_ratio == pytest.approx(0.30)
        assert cfg.catfish_intervention_enabled is True
        assert cfg.catfish_competitive_shaping_enabled is False

    assert primary.catfish_source_ratios == pytest.approx((0.10, 0.10, 0.10))
    assert multi_buffer.catfish_source_ratios == pytest.approx((0.10, 0.10, 0.10))
    assert random_control.catfish_source_ratios == pytest.approx((0.10, 0.10, 0.10))
    assert single.catfish_intervention_catfish_ratio == pytest.approx(0.30)


def test_guarded_residual_admission_is_deterministic_and_uses_complete_ties() -> None:
    cfg = load_training_yaml(PRIMARY_CONFIG)
    trainer_cfg = build_trainer_config(cfg)
    trainer = MultiCatfishMODQNTrainer(
        build_environment(cfg),
        trainer_cfg,
        train_seed=1,
        env_seed=2,
        mobility_seed=3,
    )

    zero_handover_tie = np.array([50.0, 0.0, -2.0], dtype=np.float64)
    assert trainer._guarded_residual_sources(
        reward_raw=zero_handover_tie,
        scalar_admitted=False,
    ) == ["r2"]
    assert trainer._guarded_residual_sources(
        reward_raw=zero_handover_tie,
        scalar_admitted=False,
    ) == ["r2"]

    next_coarse_tie = np.array([50.0, -0.5, -2.0], dtype=np.float64)
    assert trainer._guarded_residual_sources(
        reward_raw=next_coarse_tie,
        scalar_admitted=False,
    ) == []

    scalar_duplicate_r1 = np.array([150.0, -0.5, -1.0], dtype=np.float64)
    assert "r1" not in trainer._guarded_residual_sources(
        reward_raw=scalar_duplicate_r1,
        scalar_admitted=True,
    )


def test_phase05b_mixed_replay_uses_70_10_10_10_composition() -> None:
    counts = target_source_sample_counts(
        batch_size=64,
        source_ratios={"r1": 0.10, "r2": 0.10, "r3": 0.10},
    )
    assert sum(counts.values()) == 19
    assert counts == {"r1": 7, "r2": 6, "r3": 6}

    main = ReplayBuffer(capacity=128)
    sources = {name: ReplayBuffer(capacity=128) for name in ("r1", "r2", "r3")}
    for idx in range(80):
        main.push(*_transition(idx))
    for source_idx, name in enumerate(sources):
        for idx in range(20):
            sources[name].push(*_transition(1000 * (source_idx + 1) + idx))

    batch = sample_multi_source_replay_batch(
        main_replay=main,
        source_replays=sources,
        batch_size=64,
        source_ratios={"r1": 0.10, "r2": 0.10, "r3": 0.10},
        rng=np.random.default_rng(42),
    )

    assert batch.states.shape == (64, 4)
    assert batch.composition["actual_main_sample_count"] == 45
    assert batch.composition["actual_catfish_sample_count"] == 19
    assert batch.composition["actual_catfish_ratio"] == pytest.approx(19 / 64)
    assert batch.composition["source_counts"] == {
        "main": 45,
        "r1": 7,
        "r2": 6,
        "r3": 6,
    }


def test_phase05b_per_buffer_labels_are_reported_in_main_update_samples() -> None:
    cfg = load_training_yaml(PRIMARY_CONFIG)
    trainer_cfg = build_trainer_config(cfg)
    trainer = MultiCatfishMODQNTrainer(
        build_environment(cfg),
        trainer_cfg,
        train_seed=4,
        env_seed=5,
        mobility_seed=6,
    )
    for idx in range(80):
        trainer.replay.push(*_transition(idx))
    for source_idx, name in enumerate(("r1", "r2", "r3")):
        for idx in range(20):
            trainer.objective_replays[name].push(
                *_transition(1000 * (source_idx + 1) + idx)
            )

    batch, composition = trainer._sample_main_update_batch()

    assert batch is not None
    assert composition is not None
    assert composition["source_counts"] == {"main": 45, "r1": 7, "r2": 6, "r3": 6}
    assert composition["target_source_sample_counts"] == {"r1": 7, "r2": 6, "r3": 6}


def test_phase05b_baseline_configs_and_protected_artifacts_are_not_targets() -> None:
    for baseline in (
        "configs/modqn-paper-baseline.yaml",
        "configs/modqn-paper-baseline.resolved-template.yaml",
    ):
        text = Path(baseline).read_text(encoding="utf-8")
        assert "phase-05-b-multi-catfish-bounded-pilot" not in text
        assert "Multi-Catfish-MODQN" not in text

    for config_path in PHASE_05B_CONFIGS:
        cfg_text = Path(config_path).read_text(encoding="utf-8")
        assert "artifacts/pilot-02-best-eval" not in cfg_text
        assert "artifacts/run-9000" not in cfg_text
        assert "artifacts/table-ii-" not in cfg_text
        assert "artifacts/fig-" not in cfg_text


def test_phase05b_seed_triplets_match_across_comparators() -> None:
    triplets = []
    for config_path in PHASE_05B_CONFIGS:
        cfg = load_training_yaml(config_path)
        triplets.append(
            cfg["training_experiment"]["phase_05b_multi_catfish"]["seed_triplets"]
        )
    assert all(item == triplets[0] for item in triplets)


def test_phase05b_control_config_does_not_enable_catfish() -> None:
    cfg = build_trainer_config(load_training_yaml(CONTROL_CONFIG))

    assert cfg.method_family == "MODQN-control"
    assert cfg.catfish_enabled is False
    assert cfg.catfish_intervention_enabled is False
    assert cfg.r1_reward_mode == "throughput"


def _transition(idx: int) -> tuple:
    state = np.full(4, float(idx), dtype=np.float32)
    next_state = state + 1.0
    reward = np.array([idx, -idx % 3, -idx % 5], dtype=np.float32)
    mask = np.ones(3, dtype=bool)
    next_mask = np.ones(3, dtype=bool)
    return state, idx % 3, reward, next_state, mask, next_mask, False
