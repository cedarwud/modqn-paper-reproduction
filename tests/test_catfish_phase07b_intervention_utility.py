from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from modqn_paper_reproduction.algorithms.catfish_modqn import CatfishMODQNTrainer
from modqn_paper_reproduction.analysis.catfish_phase07b_bounded_pilot import (
    PHASE_07B_CONFIGS,
    PHASE_07B_EVAL_SEEDS,
    validate_phase07b_training_config,
)
from modqn_paper_reproduction.cli import train_main
from modqn_paper_reproduction.config_loader import (
    ConfigValidationError,
    build_environment,
    build_trainer_config,
    get_seeds,
    load_training_yaml,
)
from modqn_paper_reproduction.runtime.catfish_replay import (
    sample_equal_budget_random_replay_batch,
)
from modqn_paper_reproduction.runtime.replay_buffer import ReplayBuffer


CONTROL_CONFIG = "configs/catfish-modqn-phase-07b-modqn-control.resolved.yaml"
PRIMARY_CONFIG = (
    "configs/catfish-modqn-phase-07b-single-catfish-primary-shaping-off.resolved.yaml"
)
NO_INTERVENTION_CONFIG = (
    "configs/catfish-modqn-phase-07b-no-intervention.resolved.yaml"
)
RANDOM_CONFIG = (
    "configs/catfish-modqn-phase-07b-random-equal-budget-injection.resolved.yaml"
)
REPLAY_ONLY_CONFIG = (
    "configs/catfish-modqn-phase-07b-replay-only-single-learner.resolved.yaml"
)
NO_ASYMMETRIC_GAMMA_CONFIG = (
    "configs/catfish-modqn-phase-07b-no-asymmetric-gamma.resolved.yaml"
)


def test_phase07b_configs_validate_only_in_allowed_namespace() -> None:
    for config_path in PHASE_07B_CONFIGS:
        assert Path(config_path).name.startswith("catfish-modqn-phase-07b-")
        cfg = load_training_yaml(config_path)
        trainer_cfg = build_trainer_config(cfg)
        phase_block = validate_phase07b_training_config(cfg, trainer_cfg)

        assert trainer_cfg.episodes == 20
        assert trainer_cfg.target_update_every_episodes == 5
        assert get_seeds(cfg)["evaluation_seed_set"] == PHASE_07B_EVAL_SEEDS
        assert len(phase_block["seed_triplets"]) >= 3
        assert trainer_cfg.r1_reward_mode == "throughput"
        assert trainer_cfg.reward_calibration_enabled is False


def test_phase07b_baseline_config_and_artifacts_are_not_targets() -> None:
    for baseline in (
        "configs/modqn-paper-baseline.yaml",
        "configs/modqn-paper-baseline.resolved-template.yaml",
    ):
        text = Path(baseline).read_text(encoding="utf-8")
        assert "phase-07-b-single-catfish-intervention-utility" not in text
        assert "Catfish-EE" not in text
        assert "RA-EE" not in text

    for config_path in PHASE_07B_CONFIGS:
        cfg_text = Path(config_path).read_text(encoding="utf-8")
        assert "artifacts/pilot-02-best-eval" not in cfg_text
        assert "artifacts/run-9000" not in cfg_text
        assert "artifacts/table-ii-" not in cfg_text
        assert "artifacts/fig-" not in cfg_text


def test_phase07b_rejects_reward_semantics_ee_and_catfish_ee() -> None:
    cfg = load_training_yaml(PRIMARY_CONFIG)
    cfg["training_experiment"]["phase_07b_catfish_utility"][
        "r1_reward_mode"
    ] = "per-user-ee-credit"
    with pytest.raises((ConfigValidationError, ValueError)):
        build_trainer_config(cfg)

    cfg = load_training_yaml(PRIMARY_CONFIG)
    cfg["training_experiment"]["phase_07b_catfish_utility"]["reward_surface"][
        "r1"
    ] = "energy-efficiency"
    with pytest.raises(ConfigValidationError):
        build_trainer_config(cfg)

    cfg = load_training_yaml(PRIMARY_CONFIG)
    cfg["training_experiment"]["method_family"] = "Catfish-EE-MODQN"
    with pytest.raises((ConfigValidationError, ValueError)):
        build_trainer_config(cfg)


def test_phase07b_primary_shaping_off_and_equal_budget_contract() -> None:
    control = build_trainer_config(load_training_yaml(CONTROL_CONFIG))
    primary = build_trainer_config(load_training_yaml(PRIMARY_CONFIG))
    no_intervention = build_trainer_config(load_training_yaml(NO_INTERVENTION_CONFIG))
    random_control = build_trainer_config(load_training_yaml(RANDOM_CONFIG))
    replay_only = build_trainer_config(load_training_yaml(REPLAY_ONLY_CONFIG))
    no_asymmetric = build_trainer_config(
        load_training_yaml(NO_ASYMMETRIC_GAMMA_CONFIG)
    )

    assert control.catfish_enabled is False
    assert control.catfish_intervention_enabled is False
    assert control.catfish_intervention_catfish_ratio == pytest.approx(0.0)

    for cfg in (primary, random_control, replay_only, no_asymmetric):
        assert cfg.catfish_intervention_enabled is True
        assert cfg.catfish_intervention_catfish_ratio == pytest.approx(0.30)
        assert cfg.catfish_competitive_shaping_enabled is False

    assert no_intervention.catfish_enabled is True
    assert no_intervention.catfish_intervention_enabled is False
    assert no_intervention.catfish_intervention_catfish_ratio == pytest.approx(0.0)
    assert no_intervention.catfish_intervention_source_mode == "disabled"


def test_phase07b_random_equal_budget_replay_uses_matched_budget() -> None:
    main = ReplayBuffer(capacity=128)
    for idx in range(80):
        main.push(*_transition(idx))

    batch = sample_equal_budget_random_replay_batch(
        main_replay=main,
        batch_size=64,
        injected_ratio=0.30,
        rng=np.random.default_rng(42),
    )

    assert batch.states.shape == (64, 4)
    assert batch.composition["actual_main_sample_count"] == 45
    assert batch.composition["actual_random_control_sample_count"] == 19
    assert batch.composition["actual_catfish_sample_count"] == 19
    assert batch.composition["source_counts"] == {
        "main": 45,
        "catfish": 0,
        "random-control": 19,
    }


def test_phase07b_random_control_main_update_reports_matched_source() -> None:
    cfg = load_training_yaml(RANDOM_CONFIG)
    trainer_cfg = build_trainer_config(cfg)
    trainer = CatfishMODQNTrainer(
        build_environment(cfg),
        trainer_cfg,
        train_seed=1,
        env_seed=2,
        mobility_seed=3,
    )
    for idx in range(80):
        trainer.replay.push(*_transition(idx))

    batch, composition = trainer._sample_main_update_batch()

    assert batch is not None
    assert composition is not None
    assert composition["source_mode"] == "random-main-replay"
    assert composition["source_counts"] == {
        "main": 45,
        "catfish": 0,
        "random-control": 19,
    }
    assert composition["actual_injected_ratio"] == pytest.approx(19 / 64)


def test_phase07b_replay_only_single_learner_has_no_challenger_networks() -> None:
    cfg = load_training_yaml(REPLAY_ONLY_CONFIG)
    trainer_cfg = build_trainer_config(cfg)
    trainer = CatfishMODQNTrainer(
        build_environment(cfg),
        trainer_cfg,
        train_seed=4,
        env_seed=5,
        mobility_seed=6,
    )

    assert trainer_cfg.catfish_challenger_enabled is False
    assert len(trainer.catfish_q_nets) == 0
    assert len(trainer.catfish_target_nets) == 0
    assert trainer.catfish_optimizers == []


def test_phase07b_no_asymmetric_gamma_equalizes_discount_factor() -> None:
    cfg = build_trainer_config(load_training_yaml(NO_ASYMMETRIC_GAMMA_CONFIG))

    assert cfg.catfish_phase07b_variant == "no-asymmetric-gamma"
    assert cfg.catfish_discount_factor == cfg.discount_factor


def test_phase07b_no_intervention_injects_zero_catfish_samples(tmp_path: Path) -> None:
    run_dir = tmp_path / "phase07b-no-intervention"
    rc = train_main(
        [
            "--config",
            NO_INTERVENTION_CONFIG,
            "--episodes",
            "1",
            "--progress-every",
            "0",
            "--output-dir",
            str(run_dir),
        ]
    )
    assert rc == 0

    diagnostics = json.loads((run_dir / "catfish_diagnostics.json").read_text())
    assert diagnostics["cumulative"]["intervention_trigger_count"] == 0
    assert (
        diagnostics["cumulative"]["actual_catfish_samples_used_in_main_updates"]
        == 0
    )
    assert diagnostics["intervention_utility"]["window_count"] == 0


def test_phase07b_primary_smoke_emits_lineage_starvation_and_windows(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "phase07b-primary-smoke"
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

    diagnostics = json.loads((run_dir / "catfish_diagnostics.json").read_text())
    lineage = diagnostics["final_replay"]["sample_lineage_summary"]
    starvation = diagnostics["replay_starvation"]

    assert diagnostics["phase07b_variant"] == "single-catfish-primary-shaping-off"
    assert diagnostics["config"]["intervention_source_mode"] == "catfish-replay"
    assert diagnostics["config"]["challenger_enabled"] is True
    assert diagnostics["intervention_utility"]["window_count"] > 0
    assert "sample_id" in lineage["fields"]
    assert "source_buffer" in lineage["fields"]
    assert lineage["accepted_sample_count"] > 0
    assert "by_source" in starvation
    assert diagnostics["cumulative"]["nan_detected"] is False


def test_phase07b_seed_triplets_match_across_comparators() -> None:
    triplets = []
    for config_path in PHASE_07B_CONFIGS:
        cfg = load_training_yaml(config_path)
        triplets.append(
            cfg["training_experiment"]["phase_07b_catfish_utility"][
                "seed_triplets"
            ]
        )
    assert all(item == triplets[0] for item in triplets)


def test_phase07b_bad_kind_and_executable_prompt_like_surfaces_rejected() -> None:
    cfg = load_training_yaml(PRIMARY_CONFIG)
    wrong_kind = copy.deepcopy(cfg)
    wrong_kind["training_experiment"]["kind"] = "phase-05-b-multi-catfish-bounded-pilot"
    with pytest.raises((ConfigValidationError, ValueError)):
        build_trainer_config(wrong_kind)

    cfg = load_training_yaml(PRIMARY_CONFIG)
    cfg["training_experiment"]["phase_07b_catfish_utility"][
        "competitive_shaping"
    ]["enabled"] = True
    with pytest.raises((ConfigValidationError, ValueError)):
        build_trainer_config(cfg)


def _transition(idx: int) -> tuple:
    state = np.full(4, float(idx), dtype=np.float32)
    next_state = state + 1.0
    reward = np.array([idx, -idx % 3, -idx % 5], dtype=np.float32)
    mask = np.ones(3, dtype=bool)
    next_mask = np.ones(3, dtype=bool)
    return state, idx % 3, reward, next_state, mask, next_mask, False
