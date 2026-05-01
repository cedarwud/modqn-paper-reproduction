"""Route D tiny matched learned-policy denominator check with DPC.

This module trains only the predeclared 5-episode matched control/candidate
pair and evaluates the selected greedy policies for denominator collapse.

It is not an effectiveness pilot. Scalar reward is exported only as a
diagnostic, and the frozen baseline is not modified.
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..algorithms.modqn import MODQNTrainer
from ..artifacts import (
    RunArtifactPaths,
    TrainingLogRow,
    write_training_log,
)
from ..config_loader import (
    build_environment,
    build_trainer_config,
    get_seeds,
    load_training_yaml,
)
from ..env.step import HOBS_POWER_SURFACE_DPC_SIDECAR
from ..runtime.objective_math import scalarize_objectives
from ..runtime.trainer_spec import (
    R1_REWARD_MODE_HOBS_ACTIVE_TX_EE,
    R1_REWARD_MODE_THROUGHPUT,
)
from ._common import write_json

CONTROL_CONFIG = Path("configs/hobs-dpc-denominator-check-control.resolved.yaml")
CANDIDATE_CONFIG = Path("configs/hobs-dpc-denominator-check-candidate.resolved.yaml")
CONTROL_ARTIFACT_DIR = Path("artifacts/hobs-dpc-denominator-check-control")
CANDIDATE_ARTIFACT_DIR = Path("artifacts/hobs-dpc-denominator-check-candidate")
PAIRED_COMPARISON_DIR = CANDIDATE_ARTIFACT_DIR / "paired-comparison-vs-control"
MAX_TINY_EPISODES = 10

_MIN_CORRELATION_SAMPLES = 5
_PEARSON_BLOCK_THRESHOLD = 0.95
_P05_COLLAPSE_RATIO = 0.95
_SERVED_RATIO_MATERIAL_DROP = 0.02


def _unique_sorted(values: list[float], *, places: int = 9) -> list[float]:
    return sorted({round(float(v), places) for v in values})


def _distribution(values: list[float] | list[int], *, places: int = 9) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(round(float(value), places))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: float(item[0])))


def _config_power_surface_value(cfg: dict[str, Any]) -> dict[str, Any]:
    block = cfg.get("resolved_assumptions", {}).get("hobs_power_surface", {})
    value = block.get("value", {}) if isinstance(block, dict) else {}
    return dict(value) if isinstance(value, dict) else {}


def _seed_block(cfg: dict[str, Any]) -> dict[str, Any]:
    seeds = get_seeds(cfg)
    return {
        "train_seed": int(seeds["train_seed"]),
        "environment_seed": int(seeds["environment_seed"]),
        "mobility_seed": int(seeds["mobility_seed"]),
        "evaluation_seed_set": [int(seed) for seed in seeds["evaluation_seed_set"]],
    }


def _dpc_parameter_subset(power_surface: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "mode",
        "inactive_beam_policy",
        "dpc_initial_power_w",
        "dpc_step_size_w",
        "dpc_p_min_w",
        "dpc_p_beam_max_w",
        "dpc_p_sat_max_w",
        "dpc_qos_thr_bps",
        "dpc_epsilon_p_w",
        "units",
    )
    return {key: power_surface.get(key) for key in keys}


def prove_matched_boundary(
    control_config_path: str | Path = CONTROL_CONFIG,
    candidate_config_path: str | Path = CANDIDATE_CONFIG,
) -> dict[str, Any]:
    """Prove that Route D control/candidate differ only in r1 role."""
    control_cfg = load_training_yaml(control_config_path)
    candidate_cfg = load_training_yaml(candidate_config_path)
    control_trainer = build_trainer_config(control_cfg)
    candidate_trainer = build_trainer_config(candidate_cfg)
    control_env = build_environment(control_cfg)
    candidate_env = build_environment(candidate_cfg)

    control_power = _config_power_surface_value(control_cfg)
    candidate_power = _config_power_surface_value(candidate_cfg)
    control_dpc_params = _dpc_parameter_subset(control_power)
    candidate_dpc_params = _dpc_parameter_subset(candidate_power)
    control_seeds = _seed_block(control_cfg)
    candidate_seeds = _seed_block(candidate_cfg)

    checks = {
        "control_r1_is_throughput": (
            control_trainer.r1_reward_mode == R1_REWARD_MODE_THROUGHPUT
        ),
        "candidate_r1_is_hobs_active_tx_ee": (
            candidate_trainer.r1_reward_mode == R1_REWARD_MODE_HOBS_ACTIVE_TX_EE
        ),
        "same_training_experiment_kind": (
            control_trainer.training_experiment_kind
            == candidate_trainer.training_experiment_kind
        ),
        "same_phase": control_trainer.phase == candidate_trainer.phase,
        "same_episode_budget": control_trainer.episodes == candidate_trainer.episodes,
        "tiny_episode_budget": 5 <= int(candidate_trainer.episodes) <= MAX_TINY_EPISODES,
        "same_seed_block": control_seeds == candidate_seeds,
        "same_objective_weights": (
            control_trainer.objective_weights == candidate_trainer.objective_weights
        ),
        "same_training_hyperparameters": (
            control_trainer.hidden_layers == candidate_trainer.hidden_layers
            and control_trainer.activation == candidate_trainer.activation
            and control_trainer.learning_rate == candidate_trainer.learning_rate
            and control_trainer.discount_factor == candidate_trainer.discount_factor
            and control_trainer.batch_size == candidate_trainer.batch_size
            and control_trainer.epsilon_start == candidate_trainer.epsilon_start
            and control_trainer.epsilon_end == candidate_trainer.epsilon_end
            and control_trainer.epsilon_decay_episodes
            == candidate_trainer.epsilon_decay_episodes
            and control_trainer.target_update_every_episodes
            == candidate_trainer.target_update_every_episodes
            and control_trainer.replay_capacity == candidate_trainer.replay_capacity
        ),
        "same_checkpoint_rule": (
            control_trainer.checkpoint_assumption_id
            == candidate_trainer.checkpoint_assumption_id
            and control_trainer.checkpoint_primary_report
            == candidate_trainer.checkpoint_primary_report
            and control_trainer.checkpoint_secondary_report
            == candidate_trainer.checkpoint_secondary_report
        ),
        "same_dpc_sidecar": control_dpc_params == candidate_dpc_params,
        "dpc_sidecar_enabled": (
            control_env.power_surface_config.hobs_power_surface_mode
            == candidate_env.power_surface_config.hobs_power_surface_mode
            == HOBS_POWER_SURFACE_DPC_SIDECAR
        ),
        "same_environment_boundary": (
            control_env.config == candidate_env.config
            and control_env.orbit.config == candidate_env.orbit.config
            and control_env.beam_pattern.config == candidate_env.beam_pattern.config
            and control_env.channel_config == candidate_env.channel_config
        ),
    }
    matched = all(bool(value) for value in checks.values())
    return {
        "matched_boundary_pass": matched,
        "checks": checks,
        "allowed_difference": "r1_reward_mode and method/label metadata only",
        "control": {
            "config_path": str(control_config_path),
            "r1_reward_mode": control_trainer.r1_reward_mode,
            "method_family": control_trainer.method_family,
            "comparison_role": control_trainer.comparison_role,
        },
        "candidate": {
            "config_path": str(candidate_config_path),
            "r1_reward_mode": candidate_trainer.r1_reward_mode,
            "method_family": candidate_trainer.method_family,
            "comparison_role": candidate_trainer.comparison_role,
        },
        "episodes": int(candidate_trainer.episodes),
        "evaluation_seed_set": candidate_seeds["evaluation_seed_set"],
        "evaluation_every_episodes": int(candidate_trainer.target_update_every_episodes),
        "checkpoint_rule": {
            "primary_report": candidate_trainer.checkpoint_primary_report,
            "secondary_report": candidate_trainer.checkpoint_secondary_report,
        },
        "dpc_sidecar_parameters": candidate_dpc_params,
    }


def _evaluate_learned_greedy_policy(
    trainer: MODQNTrainer,
    eval_seeds: tuple[int, ...],
    *,
    max_steps_per_episode: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    step_records: list[dict[str, Any]] = []
    dpc_totals = {
        "dpc_step_count": 0,
        "dpc_sign_flip_count": 0,
        "dpc_qos_guard_count": 0,
        "dpc_per_beam_cap_violations": 0,
        "dpc_sat_cap_violations": 0,
    }
    within_seed_power_changes = 0
    within_seed_power_pairs = 0
    episode_scalars: list[float] = []
    episode_raw_throughputs: list[float] = []
    handover_count = 0
    all_user_throughputs: list[float] = []

    for seed in eval_seeds:
        env_seed_seq, mobility_seed_seq = np.random.SeedSequence(int(seed)).spawn(2)
        env_rng = np.random.default_rng(env_seed_seq)
        mobility_rng = np.random.default_rng(mobility_seed_seq)

        states, masks, _diag = trainer.env.reset(env_rng, mobility_rng)
        encoded = trainer.encode_states(states)
        seed_total_powers: list[float] = []
        episode_reward = np.zeros(3, dtype=np.float64)
        episode_throughput = 0.0

        steps_seen = 0
        while True:
            if max_steps_per_episode is not None and steps_seen >= max_steps_per_episode:
                break
            actions = trainer.select_actions(
                encoded,
                masks,
                eps=0.0,
                objective_weights=trainer.config.objective_weights,
            )
            result = trainer.env.step(actions, env_rng)
            steps_seen += 1

            active_mask = result.active_beam_mask.astype(bool)
            active_count = int(np.sum(active_mask))
            user_throughputs = [
                float(reward.r1_throughput) for reward in result.rewards
            ]
            total_throughput = float(np.sum(user_throughputs, dtype=np.float64))
            ee_active_tx = (
                float(result.rewards[0].r1_hobs_active_tx_ee)
                if result.rewards else 0.0
            )

            step_reward = np.zeros(3, dtype=np.float64)
            for uid, reward in enumerate(result.rewards):
                reward_vec = trainer.reward_vector_from_step_result(result, uid)
                step_reward += reward_vec
                episode_reward += reward_vec
                if reward.r2_handover < 0.0:
                    handover_count += 1
            step_avg_reward = step_reward / max(trainer.num_users, 1)
            step_scalar = scalarize_objectives(
                step_avg_reward,
                trainer.config.objective_weights,
            )

            total_power = float(result.total_active_beam_power_w)
            seed_total_powers.append(total_power)
            episode_throughput += total_throughput
            all_user_throughputs.extend(user_throughputs)

            step_records.append({
                "eval_seed": int(seed),
                "step_index": int(result.step_index),
                "active_beam_count": active_count,
                "total_active_power_w": total_power,
                "sum_throughput_bps": total_throughput,
                "ee_active_tx": ee_active_tx,
                "scalar_reward_diagnostic": float(step_scalar),
                "active_power_vals": result.beam_transmit_power_w[active_mask].tolist(),
                "selected_power_profile": result.selected_power_profile,
            })

            if result.done:
                break
            encoded = trainer.encode_states(result.user_states)
            masks = result.action_masks

        for i in range(1, len(seed_total_powers)):
            within_seed_power_pairs += 1
            if abs(seed_total_powers[i] - seed_total_powers[i - 1]) > 1e-10:
                within_seed_power_changes += 1

        dpc_diag = trainer.env.get_dpc_diagnostics()
        for key in dpc_totals:
            dpc_totals[key] += int(dpc_diag.get(key, 0))

        episode_avg_reward = episode_reward / max(trainer.num_users, 1)
        episode_scalars.append(float(
            scalarize_objectives(
                episode_avg_reward,
                trainer.config.objective_weights,
            )
        ))
        episode_raw_throughputs.append(float(episode_throughput))

    if not step_records:
        return {
            "error": "no greedy evaluation steps recorded",
            "denominator_varies_in_eval": False,
            "throughput_proxy_risk_flag": True,
        }, step_records

    active_counts = [int(row["active_beam_count"]) for row in step_records]
    total_powers = [float(row["total_active_power_w"]) for row in step_records]
    throughputs = [float(row["sum_throughput_bps"]) for row in step_records]
    ees = [float(row["ee_active_tx"]) for row in step_records]
    scalars = [float(row["scalar_reward_diagnostic"]) for row in step_records]
    active_power_values: list[float] = []
    for row in step_records:
        active_power_values.extend(float(v) for v in row["active_power_vals"])

    distinct_total = _unique_sorted(total_powers)
    distinct_active = _unique_sorted(active_power_values)
    denominator_varies = len(distinct_total) > 1
    active_power_single_point = len(distinct_active) <= 1
    all_one_beam = all(count <= 1 for count in active_counts)
    power_activity_rate = (
        within_seed_power_changes / within_seed_power_pairs
        if within_seed_power_pairs > 0 else 0.0
    )

    pearson: float | None = None
    if len(throughputs) >= _MIN_CORRELATION_SAMPLES:
        thr_arr = np.asarray(throughputs, dtype=np.float64)
        ee_arr = np.asarray(ees, dtype=np.float64)
        if float(np.std(thr_arr)) > 0.0 and float(np.std(ee_arr)) > 0.0:
            pearson = float(np.corrcoef(thr_arr, ee_arr)[0, 1])
        else:
            pearson = 1.0 if float(np.std(thr_arr)) == float(np.std(ee_arr)) else None

    ranked_by_throughput = sorted(
        range(len(step_records)),
        key=lambda idx: (-throughputs[idx], idx),
    )
    ranked_by_ee = sorted(
        range(len(step_records)),
        key=lambda idx: (-ees[idx], idx),
    )
    ranking_change = ranked_by_throughput != ranked_by_ee

    served = [value > 0.0 for value in all_user_throughputs]
    served_ratio = float(np.mean(served)) if served else 0.0
    p05 = (
        float(np.percentile(np.asarray(all_user_throughputs, dtype=np.float64), 5))
        if all_user_throughputs else 0.0
    )

    return {
        "steps_evaluated": len(step_records),
        "eval_seed_set": [int(seed) for seed in eval_seeds],
        "denominator_varies_in_eval": denominator_varies,
        "all_evaluated_steps_one_active_beam": all_one_beam,
        "active_beam_count_distribution": _distribution(active_counts, places=0),
        "total_active_power_distribution": _distribution(total_powers),
        "active_power_single_point_distribution": active_power_single_point,
        "distinct_total_active_power_w_values": distinct_total,
        "distinct_active_power_w_values": distinct_active,
        "power_control_activity_rate": float(power_activity_rate),
        "dpc_sign_flip_count": int(dpc_totals["dpc_sign_flip_count"]),
        "dpc_step_count": int(dpc_totals["dpc_step_count"]),
        "dpc_qos_guard_count": int(dpc_totals["dpc_qos_guard_count"]),
        "dpc_per_beam_cap_violations": int(dpc_totals["dpc_per_beam_cap_violations"]),
        "dpc_sat_cap_violations": int(dpc_totals["dpc_sat_cap_violations"]),
        "throughput_proxy_risk_flag": (
            (not denominator_varies) or active_power_single_point
        ),
        "throughput_vs_ee_pearson": pearson,
        "same_policy_throughput_vs_ee_rescore_ranking_change": ranking_change,
        "top_throughput_step": {
            key: step_records[ranked_by_throughput[0]][key]
            for key in (
                "eval_seed",
                "step_index",
                "sum_throughput_bps",
                "ee_active_tx",
                "total_active_power_w",
            )
        },
        "top_ee_step": {
            key: step_records[ranked_by_ee[0]][key]
            for key in (
                "eval_seed",
                "step_index",
                "sum_throughput_bps",
                "ee_active_tx",
                "total_active_power_w",
            )
        },
        "raw_throughput_mean_bps": float(np.mean(throughputs)),
        "raw_episode_throughput_mean_bps": float(np.mean(episode_raw_throughputs)),
        "p05_throughput_bps": p05,
        "served_ratio": served_ratio,
        "handover_count": int(handover_count),
        "scalar_reward_diagnostic_mean": float(np.mean(scalars)),
        "episode_scalar_reward_diagnostic_mean": float(np.mean(episode_scalars)),
        "forbidden_claims": [
            "Do not claim EE-MODQN effectiveness.",
            "Do not claim old Phase 03 failures are overturned.",
            "Do not claim physical energy saving.",
            "Do not claim DPC is MODQN-paper-backed.",
            "Do not use scalar reward alone as success evidence.",
            "Do not introduce Catfish or RA-EE association.",
        ],
    }, step_records


def _write_step_trace(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "eval_seed",
        "step_index",
        "active_beam_count",
        "total_active_power_w",
        "sum_throughput_bps",
        "ee_active_tx",
        "scalar_reward_diagnostic",
        "selected_power_profile",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _train_and_evaluate_arm(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    max_steps_per_eval_episode: int | None = None,
) -> dict[str, Any]:
    cfg = load_training_yaml(config_path)
    env = build_environment(cfg)
    trainer_cfg = build_trainer_config(cfg)
    seeds = get_seeds(cfg)
    eval_seeds = tuple(int(seed) for seed in seeds["evaluation_seed_set"])

    if int(trainer_cfg.episodes) > MAX_TINY_EPISODES:
        raise ValueError(
            "Route D denominator check is capped at "
            f"{MAX_TINY_EPISODES} episodes, got {trainer_cfg.episodes}."
        )
    if int(trainer_cfg.episodes) < 5:
        raise ValueError(
            f"Route D denominator check requires at least 5 episodes, got {trainer_cfg.episodes}."
        )
    if not eval_seeds:
        raise ValueError("Route D denominator check requires a non-empty eval seed set.")

    trainer = MODQNTrainer(
        env=env,
        config=trainer_cfg,
        train_seed=int(seeds["train_seed"]),
        env_seed=int(seeds["environment_seed"]),
        mobility_seed=int(seeds["mobility_seed"]),
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = RunArtifactPaths(out)

    t0 = time.time()
    logs = trainer.train(
        progress_every=0,
        evaluation_seed_set=eval_seeds,
        evaluation_every_episodes=trainer_cfg.target_update_every_episodes,
    )
    elapsed_s = time.time() - t0

    write_training_log(
        paths.training_log_json,
        [
            TrainingLogRow(
                episode=log.episode,
                epsilon=log.epsilon,
                r1_mean=log.r1_mean,
                r2_mean=log.r2_mean,
                r3_mean=log.r3_mean,
                scalar_reward=log.scalar_reward,
                total_handovers=log.total_handovers,
                replay_size=log.replay_size,
                losses=log.losses,
            )
            for log in logs
        ],
    )

    checkpoint_rule = trainer.checkpoint_rule()
    final_episode = int(logs[-1].episode) if logs else -1
    final_checkpoint = trainer.save_checkpoint(
        paths.primary_checkpoint(checkpoint_rule),
        episode=final_episode,
        checkpoint_kind=trainer_cfg.checkpoint_primary_report,
        logs=logs,
        include_optimizers=True,
    )
    best_eval_checkpoint = None
    checkpoint_used_for_greedy_eval = trainer_cfg.checkpoint_primary_report
    best_eval_summary = None
    if trainer.has_best_eval_checkpoint():
        best_eval_checkpoint = trainer.save_best_eval_checkpoint(
            paths.secondary_checkpoint(checkpoint_rule)
        )
        best_eval_summary_obj = trainer.best_eval_summary()
        best_eval_summary = (
            asdict(best_eval_summary_obj) if best_eval_summary_obj is not None else None
        )
        trainer.restore_best_eval_checkpoint(load_optimizers=False)
        checkpoint_used_for_greedy_eval = trainer_cfg.checkpoint_secondary_report

    diagnostics, step_trace = _evaluate_learned_greedy_policy(
        trainer,
        eval_seeds,
        max_steps_per_episode=max_steps_per_eval_episode,
    )
    _write_step_trace(out / "greedy_eval_step_trace.csv", step_trace)

    summary = {
        "namespace": str(Path(output_dir)),
        "route": "Route D - tiny matched learned-policy denominator check with DPC",
        "config_path": str(config_path),
        "training_completed": True,
        "episodes_completed": len(logs),
        "elapsed_s": elapsed_s,
        "checkpoint_used_for_greedy_eval": checkpoint_used_for_greedy_eval,
        "checkpoint_files": {
            "final_episode_policy": str(final_checkpoint),
            "best_weighted_reward_on_eval": (
                None if best_eval_checkpoint is None else str(best_eval_checkpoint)
            ),
        },
        "best_eval_summary": best_eval_summary,
        "trainer_config": {
            "training_experiment_kind": trainer_cfg.training_experiment_kind,
            "method_family": trainer_cfg.method_family,
            "comparison_role": trainer_cfg.comparison_role,
            "r1_reward_mode": trainer_cfg.r1_reward_mode,
            "episodes": int(trainer_cfg.episodes),
            "target_update_every_episodes": int(trainer_cfg.target_update_every_episodes),
            "objective_weights": [float(x) for x in trainer_cfg.objective_weights],
        },
        "seeds": _seed_block(cfg),
        "dpc_sidecar_parameters": _config_power_surface_value(cfg),
        "diagnostics": diagnostics,
    }
    summary_path = write_json(out / "summary.json", summary)
    _write_arm_review(out / "review.md", summary)
    summary["artifact_paths"] = {
        "summary_json": str(summary_path),
        "training_log_json": str(paths.training_log_json),
        "step_trace_csv": str(out / "greedy_eval_step_trace.csv"),
        "review_md": str(out / "review.md"),
    }
    write_json(out / "summary.json", summary)
    return summary


def interpret_route_d_verdict(
    control_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    boundary_proof: dict[str, Any],
) -> dict[str, Any]:
    """Interpret Route D hard-stop conditions for the paired run."""
    control_diag = control_summary["diagnostics"]
    candidate_diag = candidate_summary["diagnostics"]

    reasons: list[str] = []
    if not bool(boundary_proof.get("matched_boundary_pass", False)):
        return {
            "route_d_status": "NEEDS MORE DESIGN",
            "reasons": ["candidate/control boundary cannot be proven matched"],
            "candidate_denominator_diagnostics_pass": False,
        }

    pearson = candidate_diag.get("throughput_vs_ee_pearson")
    pearson_block = pearson is not None and float(pearson) > _PEARSON_BLOCK_THRESHOLD
    candidate_denominator_pass = (
        bool(candidate_diag.get("denominator_varies_in_eval", False))
        and not bool(candidate_diag.get("all_evaluated_steps_one_active_beam", True))
        and not bool(candidate_diag.get("active_power_single_point_distribution", True))
        and not bool(candidate_diag.get("throughput_proxy_risk_flag", True))
        and not pearson_block
    )

    if not bool(candidate_diag.get("denominator_varies_in_eval", False)):
        reasons.append("denominator_varies_in_eval=false under learned greedy eval")
    if bool(candidate_diag.get("all_evaluated_steps_one_active_beam", False)):
        reasons.append("all_evaluated_steps_one_active_beam=true")
    if bool(candidate_diag.get("active_power_single_point_distribution", False)):
        reasons.append("active_power_single_point_distribution=true")
    if pearson_block:
        reasons.append("throughput_vs_ee_pearson > 0.95")

    control_p05 = float(control_diag.get("p05_throughput_bps", 0.0))
    candidate_p05 = float(candidate_diag.get("p05_throughput_bps", 0.0))
    p05_ratio = (
        candidate_p05 / control_p05
        if control_p05 > 0.0 else (1.0 if candidate_p05 == 0.0 else float("inf"))
    )
    served_delta = (
        float(candidate_diag.get("served_ratio", 0.0))
        - float(control_diag.get("served_ratio", 0.0))
    )
    material_throughput_collapse = (
        p05_ratio < _P05_COLLAPSE_RATIO
        or served_delta < -_SERVED_RATIO_MATERIAL_DROP
    )
    if material_throughput_collapse:
        reasons.append("candidate materially collapses p05 throughput or served ratio")

    scalar_delta = (
        float(candidate_diag.get("episode_scalar_reward_diagnostic_mean", 0.0))
        - float(control_diag.get("episode_scalar_reward_diagnostic_mean", 0.0))
    )
    scalar_only_improvement = scalar_delta > 0.0 and not candidate_denominator_pass
    if scalar_only_improvement:
        reasons.append("candidate improves scalar reward but not denominator diagnostics")

    status = "BLOCK" if reasons else "PASS"
    return {
        "route_d_status": status,
        "reasons": reasons,
        "candidate_denominator_diagnostics_pass": candidate_denominator_pass,
        "candidate_minus_control": {
            "raw_throughput_mean_bps": (
                float(candidate_diag.get("raw_throughput_mean_bps", 0.0))
                - float(control_diag.get("raw_throughput_mean_bps", 0.0))
            ),
            "p05_throughput_bps": candidate_p05 - control_p05,
            "served_ratio": served_delta,
            "handover_count": (
                int(candidate_diag.get("handover_count", 0))
                - int(control_diag.get("handover_count", 0))
            ),
            "episode_scalar_reward_diagnostic_mean": scalar_delta,
        },
        "p05_throughput_ratio_candidate_vs_control": float(p05_ratio),
        "served_ratio_delta_candidate_vs_control": float(served_delta),
        "material_throughput_collapse": material_throughput_collapse,
        "scalar_only_improvement_without_denominator": scalar_only_improvement,
    }


def export_tiny_matched_denominator_check(
    control_config_path: str | Path = CONTROL_CONFIG,
    candidate_config_path: str | Path = CANDIDATE_CONFIG,
    *,
    control_output_dir: str | Path = CONTROL_ARTIFACT_DIR,
    candidate_output_dir: str | Path = CANDIDATE_ARTIFACT_DIR,
    paired_output_dir: str | Path = PAIRED_COMPARISON_DIR,
    max_steps_per_eval_episode: int | None = None,
) -> dict[str, Any]:
    """Run and export the tiny matched Route D denominator check."""
    boundary = prove_matched_boundary(control_config_path, candidate_config_path)
    control_summary = _train_and_evaluate_arm(
        control_config_path,
        control_output_dir,
        max_steps_per_eval_episode=max_steps_per_eval_episode,
    )
    candidate_summary = _train_and_evaluate_arm(
        candidate_config_path,
        candidate_output_dir,
        max_steps_per_eval_episode=max_steps_per_eval_episode,
    )
    verdict = interpret_route_d_verdict(control_summary, candidate_summary, boundary)

    paired = {
        "namespace": str(paired_output_dir),
        "route": "Route D - tiny matched learned-policy denominator check with DPC",
        "boundary_proof": boundary,
        "control_summary_path": str(Path(control_output_dir) / "summary.json"),
        "candidate_summary_path": str(Path(candidate_output_dir) / "summary.json"),
        "control_diagnostics": control_summary["diagnostics"],
        "candidate_diagnostics": candidate_summary["diagnostics"],
        "paired_verdict": verdict,
        "what_route_d_proves": (
            "Whether the learned greedy candidate policy avoids denominator "
            "collapse under the same DPC sidecar and matched tiny budget."
        ),
        "what_route_d_does_not_prove": [
            "EE-MODQN effectiveness",
            "physical energy saving",
            "HOBS optimizer reproduction",
            "DPC support in the MODQN paper",
            "overturning old Phase 03 failures",
        ],
    }
    out = Path(paired_output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = write_json(out / "summary.json", paired)
    review_path = out / "review.md"
    _write_paired_review(review_path, paired)
    paired["artifact_paths"] = {
        "summary_json": str(summary_path),
        "review_md": str(review_path),
    }
    write_json(out / "summary.json", paired)
    return paired


def _write_arm_review(path: Path, summary: dict[str, Any]) -> None:
    diag = summary["diagnostics"]
    lines = [
        "# Route D DPC Denominator Check Arm",
        "",
        f"Config: `{summary['config_path']}`",
        f"R1 mode: `{summary['trainer_config']['r1_reward_mode']}`",
        f"Checkpoint used for greedy eval: `{summary['checkpoint_used_for_greedy_eval']}`",
        "",
        "## Greedy Diagnostics",
        "",
        f"- `denominator_varies_in_eval`: `{diag['denominator_varies_in_eval']}`",
        f"- `all_evaluated_steps_one_active_beam`: `{diag['all_evaluated_steps_one_active_beam']}`",
        f"- `active_power_single_point_distribution`: `{diag['active_power_single_point_distribution']}`",
        f"- `power_control_activity_rate`: `{diag['power_control_activity_rate']}`",
        f"- `dpc_sign_flip_count`: `{diag['dpc_sign_flip_count']}`",
        f"- `throughput_vs_ee_pearson`: `{diag['throughput_vs_ee_pearson']}`",
        f"- `raw_throughput_mean_bps`: `{diag['raw_throughput_mean_bps']}`",
        f"- `p05_throughput_bps`: `{diag['p05_throughput_bps']}`",
        f"- `served_ratio`: `{diag['served_ratio']}`",
        f"- `handover_count`: `{diag['handover_count']}`",
        "",
        "## Claim Boundary",
        "",
        "- Scalar reward is diagnostic only.",
        "- This arm alone does not prove effectiveness.",
        "- DPC is opt-in and not MODQN-paper-backed.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _write_paired_review(path: Path, paired: dict[str, Any]) -> None:
    verdict = paired["paired_verdict"]
    cdiag = paired["candidate_diagnostics"]
    lines = [
        "# Route D Tiny Matched Denominator Check",
        "",
        f"Status: **{verdict['route_d_status']}**",
        "",
        "## Matched Boundary",
        "",
        f"- `matched_boundary_pass`: `{paired['boundary_proof']['matched_boundary_pass']}`",
        f"- `same_dpc_sidecar`: `{paired['boundary_proof']['checks']['same_dpc_sidecar']}`",
        f"- `same_seed_block`: `{paired['boundary_proof']['checks']['same_seed_block']}`",
        f"- `same_checkpoint_rule`: `{paired['boundary_proof']['checks']['same_checkpoint_rule']}`",
        "",
        "## Candidate Greedy Diagnostics",
        "",
        f"- `denominator_varies_in_eval`: `{cdiag['denominator_varies_in_eval']}`",
        f"- `all_evaluated_steps_one_active_beam`: `{cdiag['all_evaluated_steps_one_active_beam']}`",
        f"- `active_power_single_point_distribution`: `{cdiag['active_power_single_point_distribution']}`",
        f"- `throughput_vs_ee_pearson`: `{cdiag['throughput_vs_ee_pearson']}`",
        f"- `raw_throughput_mean_bps`: `{cdiag['raw_throughput_mean_bps']}`",
        f"- `p05_throughput_bps`: `{cdiag['p05_throughput_bps']}`",
        f"- `served_ratio`: `{cdiag['served_ratio']}`",
        "",
        "## Reasons",
        "",
    ]
    if verdict["reasons"]:
        lines.extend(f"- {reason}" for reason in verdict["reasons"])
    else:
        lines.append("- No predeclared hard stop was triggered.")
    lines.extend([
        "",
        "## Forbidden Claims",
        "",
        "- Do not claim EE-MODQN effectiveness.",
        "- Do not claim physical energy saving.",
        "- Do not claim DPC is MODQN-paper-backed.",
        "- Do not claim old Phase 03 failures are overturned.",
        "- Do not use scalar reward alone as success evidence.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


__all__ = [
    "CANDIDATE_CONFIG",
    "CONTROL_CONFIG",
    "export_tiny_matched_denominator_check",
    "interpret_route_d_verdict",
    "prove_matched_boundary",
]
