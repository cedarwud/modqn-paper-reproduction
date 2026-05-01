"""Tests for the HOBS active-TX EE anti-collapse design gate."""

from __future__ import annotations

import copy

import numpy as np

from modqn_paper_reproduction.algorithms.modqn import MODQNTrainer
from modqn_paper_reproduction.analysis.hobs_active_tx_ee_anti_collapse import (
    CANDIDATE_CONFIG,
    CONTROL_CONFIG,
    interpret_anti_collapse_verdict,
    predeclared_tolerances,
    prove_matched_boundary,
)
from modqn_paper_reproduction.analysis.hobs_active_tx_ee_qos_sticky_anti_collapse import (
    CANDIDATE_CONFIG as QOS_STICKY_CANDIDATE_CONFIG,
    CONTROL_CONFIG as QOS_STICKY_CONTROL_CONFIG,
)
from modqn_paper_reproduction.config_loader import (
    build_environment,
    build_trainer_config,
    load_training_yaml,
)
from modqn_paper_reproduction.env.step import (
    ActionMask,
    HOBS_POWER_SURFACE_DPC_SIDECAR,
    StepConfig,
    StepEnvironment,
    UserState,
)
from modqn_paper_reproduction.runtime.trainer_spec import (
    HOBS_ACTIVE_TX_EE_ANTI_COLLAPSE_KIND,
    R1_REWARD_MODE_HOBS_ACTIVE_TX_EE,
    TrainerConfig,
)


def _summary(**overrides):
    diagnostics = {
        "denominator_varies_in_eval": True,
        "all_evaluated_steps_one_active_beam": False,
        "active_power_single_point_distribution": False,
        "raw_throughput_mean_bps": 1000.0,
        "p05_throughput_bps": 100.0,
        "served_ratio": 1.0,
        "outage_ratio": 0.0,
        "handover_count": 100,
        "r2_mean": -0.2,
        "load_balance_metric": -1.0,
        "budget_violation_count": 0,
        "per_beam_power_violation_count": 0,
        "inactive_beam_nonzero_power_step_count": 0,
        "episode_scalar_reward_diagnostic_mean": 5.0,
    }
    diagnostics.update(overrides)
    return {"diagnostics": diagnostics}


def test_anti_collapse_configs_are_gated_and_matched() -> None:
    control_cfg = load_training_yaml(CONTROL_CONFIG)
    candidate_cfg = load_training_yaml(CANDIDATE_CONFIG)
    control_trainer = build_trainer_config(control_cfg)
    candidate_trainer = build_trainer_config(candidate_cfg)
    control_env = build_environment(control_cfg)
    candidate_env = build_environment(candidate_cfg)

    assert control_trainer.training_experiment_kind == HOBS_ACTIVE_TX_EE_ANTI_COLLAPSE_KIND
    assert candidate_trainer.training_experiment_kind == HOBS_ACTIVE_TX_EE_ANTI_COLLAPSE_KIND
    assert control_trainer.r1_reward_mode == R1_REWARD_MODE_HOBS_ACTIVE_TX_EE
    assert candidate_trainer.r1_reward_mode == R1_REWARD_MODE_HOBS_ACTIVE_TX_EE
    assert control_trainer.episodes == candidate_trainer.episodes == 5
    assert not control_trainer.anti_collapse_action_constraint_enabled
    assert candidate_trainer.anti_collapse_action_constraint_enabled
    assert (
        control_trainer.anti_collapse_max_users_per_beam
        == candidate_trainer.anti_collapse_max_users_per_beam
        == 50
    )
    assert (
        control_env.power_surface_config.hobs_power_surface_mode
        == candidate_env.power_surface_config.hobs_power_surface_mode
        == HOBS_POWER_SURFACE_DPC_SIDECAR
    )


def test_anti_collapse_matched_boundary_proves_only_candidate_toggle() -> None:
    proof = prove_matched_boundary(CONTROL_CONFIG, CANDIDATE_CONFIG)
    assert proof["matched_boundary_pass"] is True
    assert proof["checks"]["both_r1_are_hobs_active_tx_ee"] is True
    assert proof["checks"]["same_dpc_sidecar"] is True
    assert proof["checks"]["same_seed_block"] is True
    assert proof["checks"]["same_checkpoint_rule"] is True
    assert proof["checks"]["same_constraint_parameters"] is True
    assert proof["checks"]["control_constraint_disabled"] is True
    assert proof["checks"]["candidate_constraint_enabled"] is True
    assert proof["predeclared_tolerances"] == predeclared_tolerances()


def test_capacity_aware_assignment_splits_collapsed_greedy_choices(monkeypatch) -> None:
    env = StepEnvironment(step_config=StepConfig(num_users=4))
    cfg = TrainerConfig(
        training_experiment_kind=HOBS_ACTIVE_TX_EE_ANTI_COLLAPSE_KIND,
        comparison_role="matched-candidate",
        r1_reward_mode=R1_REWARD_MODE_HOBS_ACTIVE_TX_EE,
        anti_collapse_action_constraint_enabled=True,
        anti_collapse_constraint_mode="capacity-aware-greedy-assignment",
        anti_collapse_max_users_per_beam=2,
        anti_collapse_min_active_beams_target=2,
    )
    trainer = MODQNTrainer(env=env, config=cfg)

    def fake_predict(states_encoded):
        q = np.zeros((4, trainer.action_dim), dtype=np.float64)
        q[:, 0] = 10.0
        q[:, 1] = 9.0
        q[:, 2] = 1.0
        return [q, np.zeros_like(q), np.zeros_like(q)]

    monkeypatch.setattr(trainer, "_predict_objective_q_values", fake_predict)
    masks = [ActionMask(mask=np.ones(trainer.action_dim, dtype=bool)) for _ in range(4)]
    states = np.zeros((4, trainer.state_dim), dtype=np.float32)

    actions = trainer.select_actions(states, masks, eps=0.0)

    assert actions.tolist() == [0, 0, 1, 1]
    assert len(set(actions.tolist())) == 2


def _user_state_for_current_beam(action_dim: int, current_beam: int) -> UserState:
    access = np.zeros(action_dim, dtype=np.float32)
    access[current_beam] = 1.0
    return UserState(
        access_vector=access,
        channel_quality=np.ones(action_dim, dtype=np.float32),
        beam_offsets=np.zeros((action_dim, 2), dtype=np.float32),
        beam_loads=np.zeros(action_dim, dtype=np.float32),
    )


def test_qos_sticky_configs_are_gated_matched_and_non_forced() -> None:
    control_cfg = load_training_yaml(QOS_STICKY_CONTROL_CONFIG)
    candidate_cfg = load_training_yaml(QOS_STICKY_CANDIDATE_CONFIG)
    control_trainer = build_trainer_config(control_cfg)
    candidate_trainer = build_trainer_config(candidate_cfg)
    proof = prove_matched_boundary(
        QOS_STICKY_CONTROL_CONFIG,
        QOS_STICKY_CANDIDATE_CONFIG,
    )

    assert proof["matched_boundary_pass"] is True
    assert not control_trainer.anti_collapse_action_constraint_enabled
    assert candidate_trainer.anti_collapse_action_constraint_enabled
    assert (
        candidate_trainer.anti_collapse_constraint_mode
        == "qos-sticky-overflow-reassignment"
    )
    assert candidate_trainer.anti_collapse_min_active_beams_target == 0
    assert candidate_trainer.anti_collapse_overload_threshold_users_per_beam == 50
    assert candidate_trainer.anti_collapse_qos_ratio_min == 0.95
    assert not candidate_trainer.anti_collapse_allow_nonsticky_moves
    assert candidate_trainer.anti_collapse_nonsticky_move_budget == 0
    assert proof["checks"]["same_constraint_parameters"] is True


def test_qos_sticky_overflow_uses_only_safe_current_beam(monkeypatch) -> None:
    env = StepEnvironment(step_config=StepConfig(num_users=4))
    cfg = TrainerConfig(
        training_experiment_kind=HOBS_ACTIVE_TX_EE_ANTI_COLLAPSE_KIND,
        comparison_role="matched-candidate",
        r1_reward_mode=R1_REWARD_MODE_HOBS_ACTIVE_TX_EE,
        anti_collapse_action_constraint_enabled=True,
        anti_collapse_constraint_mode="qos-sticky-overflow-reassignment",
        anti_collapse_overload_threshold_users_per_beam=2,
        anti_collapse_qos_ratio_min=0.95,
        anti_collapse_allow_nonsticky_moves=False,
        anti_collapse_nonsticky_move_budget=0,
    )
    trainer = MODQNTrainer(env=env, config=cfg)

    def fake_predict(states_encoded):
        q = np.zeros((4, trainer.action_dim), dtype=np.float64)
        q[:, 0] = 10.0
        q[:, 1] = 1.0
        return [q, np.zeros_like(q), np.zeros_like(q)]

    monkeypatch.setattr(trainer, "_predict_objective_q_values", fake_predict)
    masks = [ActionMask(mask=np.ones(trainer.action_dim, dtype=bool)) for _ in range(4)]
    raw_states = [
        _user_state_for_current_beam(trainer.action_dim, beam)
        for beam in [0, 0, 1, 1]
    ]
    encoded = np.zeros((4, trainer.state_dim), dtype=np.float32)

    actions = trainer.select_actions(
        encoded,
        masks,
        eps=0.0,
        raw_states=raw_states,
    )
    diag = trainer.get_anti_collapse_diagnostics()

    assert actions.tolist() == [0, 0, 1, 1]
    assert diag["overflow_steps"] == 1
    assert diag["overflow_user_count"] == 2
    assert diag["sticky_override_count"] == 2
    assert diag["nonsticky_move_count"] == 0


def test_qos_sticky_overflow_blocks_unsafe_sticky_without_nonsticky(monkeypatch) -> None:
    env = StepEnvironment(step_config=StepConfig(num_users=4))
    cfg = TrainerConfig(
        training_experiment_kind=HOBS_ACTIVE_TX_EE_ANTI_COLLAPSE_KIND,
        comparison_role="matched-candidate",
        r1_reward_mode=R1_REWARD_MODE_HOBS_ACTIVE_TX_EE,
        anti_collapse_action_constraint_enabled=True,
        anti_collapse_constraint_mode="qos-sticky-overflow-reassignment",
        anti_collapse_overload_threshold_users_per_beam=2,
        anti_collapse_qos_ratio_min=0.95,
        anti_collapse_allow_nonsticky_moves=False,
        anti_collapse_nonsticky_move_budget=0,
    )
    trainer = MODQNTrainer(env=env, config=cfg)

    def fake_predict(states_encoded):
        q = np.zeros((4, trainer.action_dim), dtype=np.float64)
        q[:, 0] = 10.0
        q[:, 1] = 1.0
        return [q, np.zeros_like(q), np.zeros_like(q)]

    monkeypatch.setattr(trainer, "_predict_objective_q_values", fake_predict)
    masks = [ActionMask(mask=np.ones(trainer.action_dim, dtype=bool)) for _ in range(4)]
    raw_states = [
        _user_state_for_current_beam(trainer.action_dim, beam)
        for beam in [0, 0, 1, 1]
    ]
    for state in raw_states[2:]:
        state.channel_quality[0] = 10.0
        state.channel_quality[1] = 0.01
    encoded = np.zeros((4, trainer.state_dim), dtype=np.float32)

    actions = trainer.select_actions(
        encoded,
        masks,
        eps=0.0,
        raw_states=raw_states,
    )
    diag = trainer.get_anti_collapse_diagnostics()

    assert actions.tolist() == [0, 0, 0, 0]
    assert diag["sticky_override_count"] == 0
    assert diag["nonsticky_move_count"] == 0
    assert diag["qos_guard_reject_count"] == 2
    assert diag["handover_guard_reject_count"] == 2


def test_anti_collapse_verdict_passes_when_all_predeclared_guards_pass() -> None:
    boundary = {"matched_boundary_pass": True}
    verdict = interpret_anti_collapse_verdict(
        _summary(),
        _summary(
            raw_throughput_mean_bps=990.0,
            p05_throughput_bps=98.0,
            served_ratio=1.0,
            outage_ratio=0.0,
            handover_count=120,
            r2_mean=-0.24,
            episode_scalar_reward_diagnostic_mean=4.0,
        ),
        boundary,
    )
    assert verdict["status"] == "PASS"
    assert verdict["candidate_minus_control"]["p05_throughput_ratio_vs_control"] == 0.98


def test_anti_collapse_verdict_blocks_candidate_one_beam_collapse() -> None:
    boundary = {"matched_boundary_pass": True}
    verdict = interpret_anti_collapse_verdict(
        _summary(),
        _summary(all_evaluated_steps_one_active_beam=True),
        boundary,
    )
    assert verdict["status"] == "BLOCK"
    assert "candidate still all_evaluated_steps_one_active_beam=true" in verdict["reasons"]


def test_anti_collapse_verdict_needs_more_design_when_boundary_unproven() -> None:
    boundary = copy.deepcopy(prove_matched_boundary(CONTROL_CONFIG, CANDIDATE_CONFIG))
    boundary["matched_boundary_pass"] = False
    verdict = interpret_anti_collapse_verdict(_summary(), _summary(), boundary)
    assert verdict["status"] == "NEEDS MORE DESIGN"
    assert "candidate/control boundary cannot be proven matched" in verdict["reasons"]
