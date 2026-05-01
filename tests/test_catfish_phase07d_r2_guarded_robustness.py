from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from modqn_paper_reproduction.algorithms.catfish_modqn import CatfishMODQNTrainer
from modqn_paper_reproduction.analysis.catfish_phase07d_r2_guarded_robustness import (
    PHASE_07D_CONFIGS,
    PHASE_07D_EVAL_SEEDS,
    PHASE_07D_REQUIRED_ROLES,
    summarize_phase07d_runs,
    validate_phase07d_training_config,
)
from modqn_paper_reproduction.config_loader import (
    ConfigValidationError,
    build_environment,
    build_trainer_config,
    get_seeds,
    load_training_yaml,
)
from modqn_paper_reproduction.runtime.catfish_replay import (
    sample_equal_budget_random_replay_batch,
    sample_r2_guarded_mixed_replay_batch,
)
from modqn_paper_reproduction.runtime.replay_buffer import ReplayBuffer


CONTROL_CONFIG = "configs/catfish-modqn-phase-07d-modqn-control.resolved.yaml"
PRIMARY_CONFIG = (
    "configs/catfish-modqn-phase-07d-r2-guarded-primary-shaping-off.resolved.yaml"
)
RANDOM_CONFIG = (
    "configs/catfish-modqn-phase-07d-random-equal-budget-injection.resolved.yaml"
)
REPLAY_ONLY_CONFIG = (
    "configs/catfish-modqn-phase-07d-replay-only-single-learner.resolved.yaml"
)
NO_ASYMMETRIC_GAMMA_CONFIG = (
    "configs/catfish-modqn-phase-07d-no-asymmetric-gamma.resolved.yaml"
)
ADMISSION_ONLY_CONFIG = (
    "configs/catfish-modqn-phase-07d-admission-only-guard.resolved.yaml"
)
INTERVENTION_ONLY_CONFIG = (
    "configs/catfish-modqn-phase-07d-intervention-only-guard.resolved.yaml"
)
FULL_GUARD_CONFIG = (
    "configs/catfish-modqn-phase-07d-full-admission-intervention-guard.resolved.yaml"
)


def test_phase07d_configs_validate_namespace_kind_and_roles() -> None:
    roles = []
    for config_path in PHASE_07D_CONFIGS:
        assert Path(config_path).name.startswith("catfish-modqn-phase-07d-")
        cfg = load_training_yaml(config_path)
        trainer_cfg = build_trainer_config(cfg)
        phase_block = validate_phase07d_training_config(cfg, trainer_cfg)

        assert trainer_cfg.training_experiment_kind == (
            "phase-07-d-r2-guarded-single-catfish-robustness"
        )
        assert trainer_cfg.episodes == 20
        assert trainer_cfg.target_update_every_episodes == 5
        assert get_seeds(cfg)["evaluation_seed_set"] == PHASE_07D_EVAL_SEEDS
        assert len(phase_block["seed_triplets"]) >= 3
        roles.append(trainer_cfg.comparison_role)

    assert set(PHASE_07D_REQUIRED_ROLES).issubset(set(roles))


def test_phase07d_reward_semantics_and_forbidden_modes_rejected() -> None:
    cfg = load_training_yaml(PRIMARY_CONFIG)
    trainer_cfg = build_trainer_config(cfg)
    assert trainer_cfg.r1_reward_mode == "throughput"
    assert trainer_cfg.reward_calibration_enabled is False
    assert trainer_cfg.catfish_competitive_shaping_enabled is False

    bad = copy.deepcopy(cfg)
    bad["training_experiment"]["phase_07d_r2_guarded_robustness"][
        "r1_reward_mode"
    ] = "per-user-ee-credit"
    with pytest.raises((ConfigValidationError, ValueError)):
        build_trainer_config(bad)

    bad = copy.deepcopy(cfg)
    bad["training_experiment"]["method_family"] = "Catfish-EE-MODQN"
    with pytest.raises((ConfigValidationError, ValueError)):
        build_trainer_config(bad)

    bad = copy.deepcopy(cfg)
    bad["training_experiment"]["method_family"] = "Multi-Catfish-MODQN"
    with pytest.raises((ConfigValidationError, ValueError)):
        build_trainer_config(bad)

    bad = copy.deepcopy(cfg)
    bad["training_experiment"]["phase_07d_r2_guarded_robustness"][
        "competitive_shaping"
    ]["enabled"] = True
    with pytest.raises((ConfigValidationError, ValueError)):
        build_trainer_config(bad)


def test_phase07d_guard_variants_have_expected_switches() -> None:
    primary = build_trainer_config(load_training_yaml(PRIMARY_CONFIG))
    admission = build_trainer_config(load_training_yaml(ADMISSION_ONLY_CONFIG))
    intervention = build_trainer_config(load_training_yaml(INTERVENTION_ONLY_CONFIG))
    full = build_trainer_config(load_training_yaml(FULL_GUARD_CONFIG))

    assert primary.catfish_r2_guard_enabled is True
    assert primary.catfish_r2_admission_guard_enabled is True
    assert primary.catfish_r2_intervention_guard_enabled is True
    assert primary.catfish_handover_spike_guard_enabled is True

    assert admission.catfish_r2_admission_guard_enabled is True
    assert admission.catfish_r2_intervention_guard_enabled is False

    assert intervention.catfish_r2_admission_guard_enabled is False
    assert intervention.catfish_r2_intervention_guard_enabled is True

    assert full.catfish_r2_admission_guard_enabled is True
    assert full.catfish_r2_intervention_guard_enabled is True


def test_phase07d_guard_sampler_does_not_fall_back_to_unguarded_injection() -> None:
    main = ReplayBuffer(capacity=128)
    catfish = ReplayBuffer(capacity=128)
    for idx in range(80):
        main.push(*_transition(idx, r2=0.0))
        catfish.push(*_transition(idx, r2=-0.5))

    batch, guard = sample_r2_guarded_mixed_replay_batch(
        main_replay=main,
        catfish_replay=catfish,
        batch_size=64,
        catfish_ratio=0.30,
        rng=np.random.default_rng(42),
        max_attempts=2,
    )

    assert batch is None
    assert guard["passed"] is False
    assert guard["reason"] == "r2-batch-share-exceeds-main-replay"


def test_phase07d_random_equal_budget_keeps_matched_budget() -> None:
    main = ReplayBuffer(capacity=128)
    for idx in range(80):
        main.push(*_transition(idx, r2=-0.5 if idx % 2 else 0.0))

    batch = sample_equal_budget_random_replay_batch(
        main_replay=main,
        batch_size=64,
        injected_ratio=0.30,
        rng=np.random.default_rng(42),
    )

    assert batch.composition["actual_main_sample_count"] == 45
    assert batch.composition["actual_random_control_sample_count"] == 19
    assert batch.composition["actual_injected_ratio"] == pytest.approx(19 / 64)


def test_phase07d_replay_only_has_no_challenger_and_no_asym_gamma_equalizes() -> None:
    replay_only_cfg = load_training_yaml(REPLAY_ONLY_CONFIG)
    trainer_cfg = build_trainer_config(replay_only_cfg)
    trainer = CatfishMODQNTrainer(
        build_environment(replay_only_cfg),
        trainer_cfg,
        train_seed=4,
        env_seed=5,
        mobility_seed=6,
    )

    assert trainer_cfg.catfish_challenger_enabled is False
    assert len(trainer.catfish_q_nets) == 0
    assert len(trainer.catfish_target_nets) == 0
    assert trainer.catfish_optimizers == []

    no_asym = build_trainer_config(load_training_yaml(NO_ASYMMETRIC_GAMMA_CONFIG))
    assert no_asym.catfish_phase07d_variant == "no-asymmetric-gamma"
    assert no_asym.catfish_discount_factor == no_asym.discount_factor


def test_phase07d_summary_fails_predeclared_failure_modes(tmp_path: Path) -> None:
    run_dirs = _write_fake_phase07d_runs(tmp_path)
    passing = summarize_phase07d_runs(
        run_dirs=run_dirs,
        output_dir=tmp_path / "summary-pass",
    )
    assert passing["pass"] is True

    scalar_only_dirs = _write_fake_phase07d_runs(
        tmp_path / "scalar-only",
        primary_overrides={"r1_mean": 99.0, "r3_mean": -20.0},
    )
    scalar_only = summarize_phase07d_runs(
        run_dirs=scalar_only_dirs,
        output_dir=tmp_path / "summary-scalar-only",
    )
    assert scalar_only["acceptance_checks"]["scalar_only_success"] is True
    assert scalar_only["pass"] is False

    r2_bad_dirs = _write_fake_phase07d_runs(
        tmp_path / "r2-bad",
        primary_overrides={"r2_mean": -4.13},
    )
    r2_bad = summarize_phase07d_runs(
        run_dirs=r2_bad_dirs,
        output_dir=tmp_path / "summary-r2-bad",
    )
    assert r2_bad["acceptance_checks"][
        "r2_noninferior_vs_matched_modqn_and_random"
    ] is False

    starvation_dirs = _write_fake_phase07d_runs(
        tmp_path / "starvation",
        primary_diagnostics_overrides={
            "replay_starvation": {
                "main_replay_starved_updates_cumulative": 1,
                "catfish_replay_starved_intervention_cumulative": 0,
            }
        },
    )
    starvation = summarize_phase07d_runs(
        run_dirs=starvation_dirs,
        output_dir=tmp_path / "summary-starvation",
    )
    assert starvation["acceptance_checks"]["starvation_stop_trigger_absent"] is False

    missing_diag_dirs = _write_fake_phase07d_runs(
        tmp_path / "missing-diagnostics",
        omit_primary_diagnostics=True,
    )
    missing_diag = summarize_phase07d_runs(
        run_dirs=missing_diag_dirs,
        output_dir=tmp_path / "summary-missing-diagnostics",
    )
    assert missing_diag["acceptance_checks"]["required_diagnostics_present"] is False


def test_phase07d_summary_fails_frozen_namespace_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    run_dirs = _write_fake_phase07d_runs(Path("."))
    run_dirs[0] = _write_fake_run(
        Path("artifacts/run-9000/phase07d-bad-control"),
        role="matched-modqn-control",
        variant="modqn-control",
        seed_triplet=[42, 1337, 7],
        metrics=_role_metrics("matched-modqn-control"),
        diagnostics=None,
    )
    summary = summarize_phase07d_runs(
        run_dirs=run_dirs,
        output_dir=Path("artifacts/catfish-modqn-phase-07d-summary-frozen-test"),
    )

    assert summary["protected_surface_check"]["frozen_namespace_used"] is True
    assert summary["pass"] is False


def _write_fake_phase07d_runs(
    root: Path,
    *,
    primary_overrides: dict | None = None,
    primary_diagnostics_overrides: dict | None = None,
    omit_primary_diagnostics: bool = False,
) -> list[str]:
    run_dirs = []
    seed_triplets = ([42, 1337, 7], [43, 1338, 8], [44, 1339, 9])
    for role in PHASE_07D_REQUIRED_ROLES:
        for index, seed_triplet in enumerate(seed_triplets, start=1):
            metrics = _role_metrics(role)
            diagnostics = None if role == "matched-modqn-control" else _diagnostics(role)
            if role == "r2-guarded-primary-shaping-off":
                metrics.update(primary_overrides or {})
                if omit_primary_diagnostics:
                    diagnostics = None
                elif diagnostics is not None:
                    diagnostics = _deep_merge(
                        diagnostics,
                        primary_diagnostics_overrides or {},
                    )
            run_dirs.append(
                _write_fake_run(
                    root / f"artifacts/catfish-modqn-phase-07d-{role}-seed{index:02d}",
                    role=role,
                    variant=role if role != "matched-modqn-control" else "modqn-control",
                    seed_triplet=seed_triplet,
                    metrics=metrics,
                    diagnostics=diagnostics,
                )
            )
    return run_dirs


def _role_metrics(role: str) -> dict[str, float]:
    base = {
        "matched-modqn-control": {
            "scalar_reward": 609.0,
            "r1_mean": 1200.0,
            "r2_mean": -4.09,
            "r3_mean": -18.0,
            "total_handovers": 818,
        },
        "r2-guarded-primary-shaping-off": {
            "scalar_reward": 610.0,
            "r1_mean": 1201.0,
            "r2_mean": -4.10,
            "r3_mean": -17.0,
            "total_handovers": 820,
        },
        "no-intervention": {
            "scalar_reward": 608.0,
            "r1_mean": 1198.0,
            "r2_mean": -4.11,
            "r3_mean": -19.0,
            "total_handovers": 822,
        },
        "random-equal-budget-injection": {
            "scalar_reward": 609.5,
            "r1_mean": 1199.0,
            "r2_mean": -4.09,
            "r3_mean": -18.5,
            "total_handovers": 819,
        },
        "replay-only-single-learner": {
            "scalar_reward": 609.2,
            "r1_mean": 1199.5,
            "r2_mean": -4.10,
            "r3_mean": -18.2,
            "total_handovers": 820,
        },
    }
    default = {
        "scalar_reward": 609.1,
        "r1_mean": 1199.0,
        "r2_mean": -4.10,
        "r3_mean": -18.4,
        "total_handovers": 820,
    }
    return dict(base.get(role, default))


def _diagnostics(role: str) -> dict:
    guard_enabled = role not in {
        "no-intervention",
        "random-equal-budget-injection",
    }
    return {
        "cumulative": {
            "actual_injected_ratio_in_mixed_updates": 0.296875,
            "nan_detected": False,
        },
        "final_replay": {
            "sample_lineage_summary": {
                "tracking_enabled": True,
                "accepted_sample_count": 50,
                "fields": ["sample_id", "source_buffer"],
            }
        },
        "intervention_utility": {
            "window_count": 10,
            "windows": [{"main_update_index": 1}],
        },
        "r2_handover_guard": {
            "enabled": guard_enabled,
            "admission_guard_enabled": guard_enabled,
            "intervention_guard_enabled": guard_enabled,
            "admission_guard_pass_count": 20 if guard_enabled else 0,
            "admission_guard_skip_count": 5 if guard_enabled else 0,
            "intervention_guard_pass_count": 10 if guard_enabled else 0,
            "intervention_guard_skip_count": 0,
            "skip_reasons": {},
            "guarded_batch_count": 10 if guard_enabled else 0,
            "guarded_batch_pass_count": 10 if guard_enabled else 0,
            "guarded_batch_violation_count": 0,
            "injected_batch_r2_negative_share_distribution": {"count": 10, "mean": 0.1},
            "matched_main_batch_r2_negative_share_distribution": {"count": 10, "mean": 0.2},
            "recent_batch_records": [],
        },
        "replay_starvation": {
            "main_replay_starved_updates_cumulative": 0,
            "catfish_replay_starved_intervention_cumulative": 0,
        },
        "runtime_cost": {"agent_count": 2},
        "episode_diagnostics": [
            {
                "action_diversity": {"action_collapse_detected": False},
                "td_loss": {"main": {}, "catfish": {}},
                "q_stability": {"nan_detected": False},
            }
        ],
    }


def _write_fake_run(
    run_dir: Path,
    *,
    role: str,
    variant: str,
    seed_triplet: list[int],
    metrics: dict[str, float],
    diagnostics: dict | None,
) -> str:
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "trainer_config": {
            "comparison_role": role,
            "method_family": "MODQN-control" if role == "matched-modqn-control" else "Catfish-MODQN",
            "catfish_phase07d_variant": variant,
        },
        "seeds": {
            "train_seed": seed_triplet[0],
            "environment_seed": seed_triplet[1],
            "mobility_seed": seed_triplet[2],
            "evaluation_seed_set": PHASE_07D_EVAL_SEEDS,
        },
        "training_summary": {"episodes_completed": 20, "elapsed_s": 1.0},
        "runtime_environment": {"num_users": 100},
        "resolved_config_snapshot": {
            "baseline": {
                "users": 100,
                "slot_duration_s": 1.0,
                "episode_duration_s": 10.0,
            }
        },
        "best_eval_summary": {"episode": 4, "mean_scalar_reward": 52.0},
    }
    final = {
        "episode": 19,
        "epsilon": 0.99,
        "replay_size": 20000,
        "losses": [1.0, 1.0, 1.0],
        **metrics,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_dir / "training_log.json").write_text(json.dumps([final]), encoding="utf-8")
    if diagnostics is not None:
        (run_dir / "catfish_diagnostics.json").write_text(
            json.dumps(diagnostics),
            encoding="utf-8",
        )
    return str(run_dir)


def _deep_merge(base: dict, overlay: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _transition(idx: int, *, r2: float) -> tuple:
    state = np.full(4, float(idx), dtype=np.float32)
    next_state = state + 1.0
    reward = np.array([float(idx), r2, -idx % 5], dtype=np.float32)
    mask = np.ones(3, dtype=bool)
    next_mask = np.ones(3, dtype=bool)
    return state, idx % 3, reward, next_state, mask, next_mask, False
