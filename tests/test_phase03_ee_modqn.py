from __future__ import annotations

import numpy as np

from modqn_paper_reproduction.algorithms.modqn import MODQNTrainer
from modqn_paper_reproduction.config_loader import (
    build_environment,
    build_trainer_config,
    load_training_yaml,
)


CONTROL_CONFIG = "configs/ee-modqn-phase-03-control-pilot.resolved.yaml"
EE_CONFIG = "configs/ee-modqn-phase-03-ee-pilot.resolved.yaml"
P03B_CONTROL_CONFIG = (
    "configs/ee-modqn-phase-03b-control-objective-geometry.resolved.yaml"
)
P03B_EE_CONFIG = "configs/ee-modqn-phase-03b-ee-objective-geometry.resolved.yaml"


def test_phase03_configs_gate_only_r1_objective() -> None:
    control_cfg = load_training_yaml(CONTROL_CONFIG)
    ee_cfg = load_training_yaml(EE_CONFIG)

    control_trainer = build_trainer_config(control_cfg)
    ee_trainer = build_trainer_config(ee_cfg)

    assert control_cfg["track"]["phase"] == "phase-03"
    assert ee_cfg["track"]["phase"] == "phase-03"
    assert control_trainer.training_experiment_kind == "phase-03-objective-substitution"
    assert ee_trainer.training_experiment_kind == "phase-03-objective-substitution"
    assert control_trainer.r1_reward_mode == "throughput"
    assert ee_trainer.r1_reward_mode == "per-user-ee-credit"
    assert control_trainer.objective_weights == ee_trainer.objective_weights
    assert control_trainer.target_update_every_episodes == (
        ee_trainer.target_update_every_episodes
    )
    assert control_trainer.replay_capacity == ee_trainer.replay_capacity
    assert control_trainer.episodes == ee_trainer.episodes == 20

    control_power = control_cfg["resolved_assumptions"]["hobs_power_surface"]["value"]
    ee_power = ee_cfg["resolved_assumptions"]["hobs_power_surface"]["value"]
    assert control_power == ee_power
    assert control_power["mode"] == "active-load-concave"


def test_phase03_ee_reward_is_explicit_credit_assignment() -> None:
    control_cfg = load_training_yaml(CONTROL_CONFIG)
    ee_cfg = load_training_yaml(EE_CONFIG)
    control_cfg["baseline"]["users"] = 3
    ee_cfg["baseline"]["users"] = 3

    env = build_environment(ee_cfg)
    rng = np.random.default_rng(42)
    states, masks, _diag = env.reset(rng, np.random.default_rng(7))
    actions = np.array(
        [int(np.flatnonzero(mask.mask)[0]) for mask in masks],
        dtype=np.int32,
    )
    result = env.step(actions, rng)

    selected_beam = int(np.argmax(result.user_states[0].access_vector))
    beam_load = float(result.user_states[0].beam_loads[selected_beam])
    beam_power_w = float(result.beam_transmit_power_w[selected_beam])
    expected_credit = (
        result.rewards[0].r1_throughput / (beam_power_w / beam_load)
    )
    assert result.rewards[0].r1_energy_efficiency_credit == expected_credit

    control_trainer = MODQNTrainer(
        env=build_environment(control_cfg),
        config=build_trainer_config(control_cfg),
    )
    ee_trainer = MODQNTrainer(
        env=build_environment(ee_cfg),
        config=build_trainer_config(ee_cfg),
    )

    control_reward = control_trainer.reward_vector_from_step_result(result, 0)
    ee_reward = ee_trainer.reward_vector_from_step_result(result, 0)

    assert control_reward[0] == result.rewards[0].r1_throughput
    assert ee_reward[0] == result.rewards[0].r1_energy_efficiency_credit
    assert control_reward[1] == ee_reward[1] == result.rewards[0].r2_handover
    assert control_reward[2] == ee_reward[2] == result.rewards[0].r3_load_balance


def test_phase03b_geometry_is_opt_in_and_leaves_baseline_default_raw() -> None:
    baseline_cfg = load_training_yaml("configs/modqn-paper-baseline.resolved-template.yaml")
    baseline_trainer = build_trainer_config(baseline_cfg)
    assert baseline_trainer.training_experiment_kind == "baseline"
    assert baseline_trainer.r1_reward_mode == "throughput"
    assert baseline_trainer.reward_calibration_enabled is False
    assert baseline_trainer.reward_normalization_mode == "raw-unscaled"
    assert baseline_trainer.load_balance_calibration_mode == "baseline-paper-weight"
    assert baseline_trainer.objective_weights == (0.5, 0.3, 0.2)

    control_cfg = load_training_yaml(P03B_CONTROL_CONFIG)
    ee_cfg = load_training_yaml(P03B_EE_CONFIG)
    control_trainer = build_trainer_config(control_cfg)
    ee_trainer = build_trainer_config(ee_cfg)

    assert control_trainer.training_experiment_kind == "phase-03b-objective-geometry"
    assert ee_trainer.training_experiment_kind == "phase-03b-objective-geometry"
    assert control_trainer.phase == ee_trainer.phase == "phase-03b"
    assert control_trainer.comparison_role == "paired-control"
    assert ee_trainer.comparison_role == "paired-ee-objective"
    assert control_trainer.r1_reward_mode == "throughput"
    assert ee_trainer.r1_reward_mode == "per-user-beam-ee-credit"
    assert control_trainer.reward_calibration_enabled is True
    assert ee_trainer.reward_calibration_enabled is True
    assert control_trainer.reward_normalization_mode == "divide-by-fixed-scales"
    assert ee_trainer.reward_normalization_mode == "divide-by-fixed-scales"
    assert control_trainer.load_balance_calibration_mode == (
        "r3-weight-0.60-with-r1-normalized"
    )
    assert ee_trainer.load_balance_calibration_mode == (
        "r3-weight-0.60-with-r1-normalized"
    )
    assert control_trainer.objective_weights == ee_trainer.objective_weights
    assert control_trainer.objective_weights == (0.2, 0.2, 0.6)


def test_phase03b_ee_r1_path_uses_beam_power_credit() -> None:
    control_cfg = load_training_yaml(P03B_CONTROL_CONFIG)
    ee_cfg = load_training_yaml(P03B_EE_CONFIG)
    control_cfg["baseline"]["users"] = 4
    ee_cfg["baseline"]["users"] = 4

    env = build_environment(ee_cfg)
    rng = np.random.default_rng(42)
    _states, masks, _diag = env.reset(rng, np.random.default_rng(7))
    shared_valid_beam = int(np.flatnonzero(masks[0].mask)[0])
    actions = np.full(len(masks), shared_valid_beam, dtype=np.int32)
    result = env.step(actions, rng)

    selected_beam = int(np.argmax(result.user_states[0].access_vector))
    beam_load = float(result.user_states[0].beam_loads[selected_beam])
    beam_power_w = float(result.beam_transmit_power_w[selected_beam])
    allocated_credit = result.rewards[0].r1_throughput / (beam_power_w / beam_load)
    beam_power_credit = result.rewards[0].r1_throughput / beam_power_w

    assert result.rewards[0].r1_energy_efficiency_credit == allocated_credit
    assert result.rewards[0].r1_beam_power_efficiency_credit == beam_power_credit

    control_trainer = MODQNTrainer(
        env=build_environment(control_cfg),
        config=build_trainer_config(control_cfg),
    )
    ee_trainer = MODQNTrainer(
        env=build_environment(ee_cfg),
        config=build_trainer_config(ee_cfg),
    )
    control_reward = control_trainer.reward_vector_from_step_result(result, 0)
    ee_reward = ee_trainer.reward_vector_from_step_result(result, 0)

    assert control_reward[0] == result.rewards[0].r1_throughput
    assert ee_reward[0] == result.rewards[0].r1_beam_power_efficiency_credit
    assert ee_reward[0] != result.rewards[0].r1_energy_efficiency_credit
    assert control_reward[1] == ee_reward[1] == result.rewards[0].r2_handover
    assert control_reward[2] == ee_reward[2] == result.rewards[0].r3_load_balance
