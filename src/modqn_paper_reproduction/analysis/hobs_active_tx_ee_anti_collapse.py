"""Bounded HOBS active-TX EE anti-collapse design gate.

This gate compares two tiny matched HOBS active-TX EE runs.  Both arms keep
the same DPC sidecar and the same HOBS active-TX EE reward.  The only intended
difference is the candidate's explicit, config-gated capacity-aware assignment
constraint.
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..algorithms.modqn import MODQNTrainer
from ..artifacts import RunArtifactPaths, TrainingLogRow, write_training_log
from ..config_loader import (
    build_environment,
    build_trainer_config,
    get_seeds,
    load_training_yaml,
)
from ..env.step import HOBS_POWER_SURFACE_DPC_SIDECAR
from ..env.step import HOBS_POWER_SURFACE_NON_CODEBOOK_CONTINUOUS_POWER
from ..runtime.objective_math import scalarize_objectives
from ..runtime.trainer_spec import (
    HOBS_ACTIVE_TX_EE_ANTI_COLLAPSE_KIND,
    R1_REWARD_MODE_HOBS_ACTIVE_TX_EE,
)
from ._common import write_json

CONTROL_CONFIG = Path("configs/hobs-active-tx-ee-anti-collapse-control.resolved.yaml")
CANDIDATE_CONFIG = Path(
    "configs/hobs-active-tx-ee-anti-collapse-candidate.resolved.yaml"
)
CONTROL_ARTIFACT_DIR = Path("artifacts/hobs-active-tx-ee-anti-collapse-control")
CANDIDATE_ARTIFACT_DIR = Path("artifacts/hobs-active-tx-ee-anti-collapse-candidate")
PAIRED_COMPARISON_DIR = CANDIDATE_ARTIFACT_DIR / "paired-comparison-vs-control"

MAX_TINY_EPISODES = 5
P05_THROUGHPUT_RATIO_MIN = 0.95
HANDOVER_DELTA_MAX = 25
R2_MEAN_DELTA_MIN = -0.05
SERVED_RATIO_MIN_DELTA = 0.0
OUTAGE_RATIO_MAX_DELTA = 0.0
_MIN_CORRELATION_SAMPLES = 5


def _unique_sorted(values: list[float], *, places: int = 9) -> list[float]:
    return sorted({round(float(v), places) for v in values})


def _distribution(values: list[float] | list[int], *, places: int = 9) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(round(float(value), places))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: float(item[0])))


def _string_distribution(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


def _rank_average_ties(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=np.float64)
    i = 0
    while i < arr.size:
        j = i + 1
        while j < arr.size and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def _spearman(values_a: list[float], values_b: list[float]) -> float | None:
    if len(values_a) < _MIN_CORRELATION_SAMPLES:
        return None
    ranks_a = _rank_average_ties(values_a)
    ranks_b = _rank_average_ties(values_b)
    if float(np.std(ranks_a)) == 0.0 or float(np.std(ranks_b)) == 0.0:
        return 1.0 if np.array_equal(ranks_a, ranks_b) else None
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


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


def _anti_collapse_subset(trainer_cfg) -> dict[str, Any]:
    return {
        "mode": trainer_cfg.anti_collapse_constraint_mode,
        "max_users_per_beam": int(trainer_cfg.anti_collapse_max_users_per_beam),
        "min_active_beams_target": int(
            trainer_cfg.anti_collapse_min_active_beams_target
        ),
        "assignment_order": trainer_cfg.anti_collapse_assignment_order,
        "overload_threshold_users_per_beam": int(
            trainer_cfg.anti_collapse_overload_threshold_users_per_beam
        ),
        "qos_ratio_min": float(trainer_cfg.anti_collapse_qos_ratio_min),
        "allow_nonsticky_moves": bool(
            trainer_cfg.anti_collapse_allow_nonsticky_moves
        ),
        "nonsticky_move_budget": int(
            trainer_cfg.anti_collapse_nonsticky_move_budget
        ),
    }


def prove_matched_boundary(
    control_config_path: str | Path = CONTROL_CONFIG,
    candidate_config_path: str | Path = CANDIDATE_CONFIG,
) -> dict[str, Any]:
    """Prove that the pilot differs only by the candidate constraint toggle."""
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
    control_constraint = _anti_collapse_subset(control_trainer)
    candidate_constraint = _anti_collapse_subset(candidate_trainer)

    checks = {
        "both_r1_are_hobs_active_tx_ee": (
            control_trainer.r1_reward_mode
            == candidate_trainer.r1_reward_mode
            == R1_REWARD_MODE_HOBS_ACTIVE_TX_EE
        ),
        "same_training_experiment_kind": (
            control_trainer.training_experiment_kind
            == candidate_trainer.training_experiment_kind
            == HOBS_ACTIVE_TX_EE_ANTI_COLLAPSE_KIND
        ),
        "same_phase": control_trainer.phase == candidate_trainer.phase,
        "same_episode_budget": control_trainer.episodes == candidate_trainer.episodes,
        "tiny_episode_budget": int(candidate_trainer.episodes) == MAX_TINY_EPISODES,
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
        "same_constraint_parameters": control_constraint == candidate_constraint,
        "control_constraint_disabled": (
            not control_trainer.anti_collapse_action_constraint_enabled
        ),
        "candidate_constraint_enabled": (
            candidate_trainer.anti_collapse_action_constraint_enabled
        ),
    }
    matched = all(bool(value) for value in checks.values())
    return {
        "matched_boundary_pass": matched,
        "checks": checks,
        "allowed_difference": (
            "candidate enables the opt-in anti-collapse assignment constraint; "
            "method/label metadata may identify matched roles"
        ),
        "control": {
            "config_path": str(control_config_path),
            "r1_reward_mode": control_trainer.r1_reward_mode,
            "comparison_role": control_trainer.comparison_role,
            "anti_collapse_enabled": (
                control_trainer.anti_collapse_action_constraint_enabled
            ),
        },
        "candidate": {
            "config_path": str(candidate_config_path),
            "r1_reward_mode": candidate_trainer.r1_reward_mode,
            "comparison_role": candidate_trainer.comparison_role,
            "anti_collapse_enabled": (
                candidate_trainer.anti_collapse_action_constraint_enabled
            ),
        },
        "episodes": int(candidate_trainer.episodes),
        "evaluation_seed_set": candidate_seeds["evaluation_seed_set"],
        "evaluation_every_episodes": int(candidate_trainer.target_update_every_episodes),
        "checkpoint_rule": {
            "primary_report": candidate_trainer.checkpoint_primary_report,
            "secondary_report": candidate_trainer.checkpoint_secondary_report,
        },
        "dpc_sidecar_parameters": candidate_dpc_params,
        "anti_collapse_constraint": candidate_constraint,
        "predeclared_tolerances": predeclared_tolerances(),
    }


def predeclared_tolerances() -> dict[str, Any]:
    """Return the acceptance tolerances fixed before pilot execution."""
    return {
        "p05_throughput_ratio_vs_control_min": P05_THROUGHPUT_RATIO_MIN,
        "served_ratio_delta_min": SERVED_RATIO_MIN_DELTA,
        "outage_ratio_delta_max": OUTAGE_RATIO_MAX_DELTA,
        "handover_delta_max": HANDOVER_DELTA_MAX,
        "r2_mean_delta_min": R2_MEAN_DELTA_MIN,
        "budget_violation_count_max": 0,
        "per_beam_power_violation_count_max": 0,
        "inactive_beam_nonzero_power_step_count_max": 0,
    }


def _evaluate_learned_greedy_policy(
    trainer: MODQNTrainer,
    eval_seeds: tuple[int, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trainer.reset_anti_collapse_diagnostics()
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
    episode_r2: list[float] = []
    episode_r3: list[float] = []
    episode_raw_throughputs: list[float] = []
    handover_count = 0
    all_user_throughputs: list[float] = []
    load_gap_values: list[float] = []
    inactive_nonzero_steps = 0
    per_beam_power_violation_count = 0
    budget_violation_count = 0

    env_power_cfg = trainer.env.power_surface_config
    if (
        env_power_cfg.hobs_power_surface_mode
        == HOBS_POWER_SURFACE_NON_CODEBOOK_CONTINUOUS_POWER
    ):
        per_beam_cap = float(
            env_power_cfg.max_power_w
            if env_power_cfg.max_power_w is not None
            else env_power_cfg.continuous_p_active_hi_w
        )
    else:
        per_beam_cap = float(env_power_cfg.dpc_p_beam_max_w)
    sat_cap = env_power_cfg.dpc_p_sat_max_w
    num_sats = trainer.env.orbit.num_satellites
    beams_per_sat = trainer.env.beam_pattern.num_beams

    for seed in eval_seeds:
        env_seed_seq, mobility_seed_seq = np.random.SeedSequence(int(seed)).spawn(2)
        env_rng = np.random.default_rng(env_seed_seq)
        mobility_rng = np.random.default_rng(mobility_seed_seq)

        states, masks, _diag = trainer.env.reset(env_rng, mobility_rng)
        encoded = trainer.encode_states(states)
        seed_total_powers: list[float] = []
        episode_reward = np.zeros(3, dtype=np.float64)
        episode_throughput = 0.0

        while True:
            actions = trainer.select_actions(
                encoded,
                masks,
                eps=0.0,
                objective_weights=trainer.config.objective_weights,
                raw_states=states,
            )
            result = trainer.env.step(actions, env_rng)

            active_mask = result.active_beam_mask.astype(bool)
            active_count = int(np.sum(active_mask))
            beam_loads = result.user_states[0].beam_loads.astype(np.float64)
            active_loads = beam_loads[active_mask]
            load_gap = (
                float(np.max(active_loads) - np.min(active_loads))
                if active_loads.size > 0
                else 0.0
            )
            load_gap_values.append(load_gap)

            user_throughputs = [float(reward.r1_throughput) for reward in result.rewards]
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

            powers = result.beam_transmit_power_w.astype(np.float64)
            if np.any(np.abs(powers[~active_mask]) > 1e-9):
                inactive_nonzero_steps += 1
            if np.any(powers[active_mask] > per_beam_cap + 1e-9):
                per_beam_power_violation_count += 1
            if sat_cap is not None:
                for sat_idx in range(num_sats):
                    start = sat_idx * beams_per_sat
                    stop = start + beams_per_sat
                    sat_power = float(np.sum(powers[start:stop], dtype=np.float64))
                    if sat_power > float(sat_cap) + 1e-9:
                        budget_violation_count += 1
            if result.power_budget_violation:
                budget_violation_count += 1

            total_power = float(result.total_active_beam_power_w)
            seed_total_powers.append(total_power)
            episode_throughput += total_throughput
            all_user_throughputs.extend(user_throughputs)

            step_records.append({
                "eval_seed": int(seed),
                "step_index": int(result.step_index),
                "active_beam_count": active_count,
                "active_beam_load_gap": load_gap,
                "max_active_beam_load": (
                    float(np.max(active_loads)) if active_loads.size > 0 else 0.0
                ),
                "total_active_power_w": total_power,
                "sum_throughput_bps": total_throughput,
                "ee_active_tx": ee_active_tx,
                "r2_step_mean": float(step_avg_reward[1]),
                "r3_step_mean": float(step_avg_reward[2]),
                "scalar_reward_diagnostic": float(step_scalar),
                "active_power_vals": powers[active_mask].tolist(),
                "selected_power_profile": result.selected_power_profile,
            })

            if result.done:
                break
            states = result.user_states
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
        episode_r2.append(float(episode_avg_reward[1]))
        episode_r3.append(float(episode_avg_reward[2]))
        episode_raw_throughputs.append(float(episode_throughput))

    if not step_records:
        anti_diag = trainer.get_anti_collapse_diagnostics()
        return {
            "error": "no greedy evaluation steps recorded",
            "denominator_varies_in_eval": False,
            "all_evaluated_steps_one_active_beam": True,
            **anti_diag,
        }, step_records

    active_counts = [int(row["active_beam_count"]) for row in step_records]
    total_powers = [float(row["total_active_power_w"]) for row in step_records]
    throughputs = [float(row["sum_throughput_bps"]) for row in step_records]
    ees = [float(row["ee_active_tx"]) for row in step_records]
    scalars = [float(row["scalar_reward_diagnostic"]) for row in step_records]
    selected_profiles = [str(row["selected_power_profile"]) for row in step_records]
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
    total_throughput_eval = float(np.sum(throughputs, dtype=np.float64))
    total_active_power_eval = float(np.sum(total_powers, dtype=np.float64))
    ee_eval_aggregate = (
        total_throughput_eval / total_active_power_eval
        if total_active_power_eval > 0.0 else 0.0
    )

    pearson: float | None = None
    spearman: float | None = None
    if len(throughputs) >= _MIN_CORRELATION_SAMPLES:
        thr_arr = np.asarray(throughputs, dtype=np.float64)
        ee_arr = np.asarray(ees, dtype=np.float64)
        if float(np.std(thr_arr)) > 0.0 and float(np.std(ee_arr)) > 0.0:
            pearson = float(np.corrcoef(thr_arr, ee_arr)[0, 1])
        else:
            pearson = 1.0 if float(np.std(thr_arr)) == float(np.std(ee_arr)) else None
        spearman = _spearman(throughputs, ees)

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
    outage_ratio = 1.0 - served_ratio
    p05 = (
        float(np.percentile(np.asarray(all_user_throughputs, dtype=np.float64), 5))
        if all_user_throughputs else 0.0
    )
    anti_diag = trainer.get_anti_collapse_diagnostics()

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
        "selected_power_profile_distribution": _string_distribution(selected_profiles),
        "selected_power_profile_absent": set(selected_profiles) == {""},
        "power_control_activity_rate": float(power_activity_rate),
        "continuous_power_activity_rate": float(power_activity_rate),
        "dpc_sign_flip_count": int(dpc_totals["dpc_sign_flip_count"]),
        "dpc_step_count": int(dpc_totals["dpc_step_count"]),
        "dpc_qos_guard_count": int(dpc_totals["dpc_qos_guard_count"]),
        "dpc_per_beam_cap_clip_count": int(dpc_totals["dpc_per_beam_cap_violations"]),
        "dpc_sat_cap_clip_count": int(dpc_totals["dpc_sat_cap_violations"]),
        "throughput_vs_ee_pearson": pearson,
        "throughput_vs_ee_spearman": spearman,
        "same_policy_throughput_vs_ee_rescore_ranking_change": ranking_change,
        "eta_EE_active_TX": float(ee_eval_aggregate),
        "eta_EE_active_TX_eval_aggregate": float(ee_eval_aggregate),
        "eta_EE_active_TX_step_mean": float(np.mean(ees)),
        "EE_system": float(ee_eval_aggregate),
        "EE_system_eval_aggregate": float(ee_eval_aggregate),
        "EE_system_step_mean": float(np.mean(ees)),
        "raw_throughput_mean_bps": float(np.mean(throughputs)),
        "raw_episode_throughput_mean_bps": float(np.mean(episode_raw_throughputs)),
        "p05_throughput_bps": p05,
        "served_ratio": served_ratio,
        "outage_ratio": outage_ratio,
        "handover_count": int(handover_count),
        "r2": float(np.mean(episode_r2)),
        "r2_mean": float(np.mean(episode_r2)),
        "load_balance_metric": float(np.mean(episode_r3)),
        "r3_mean": float(np.mean(episode_r3)),
        "active_beam_load_gap_mean": float(np.mean(load_gap_values)),
        "scalar_reward_diagnostic_mean": float(np.mean(scalars)),
        "episode_scalar_reward_diagnostic_mean": float(np.mean(episode_scalars)),
        "budget_violation_count": int(budget_violation_count),
        "per_beam_power_violation_count": int(per_beam_power_violation_count),
        "inactive_beam_nonzero_power_step_count": int(inactive_nonzero_steps),
        "overflow_steps": int(anti_diag["overflow_steps"]),
        "overflow_user_count": int(anti_diag["overflow_user_count"]),
        "sticky_override_count": int(anti_diag["sticky_override_count"]),
        "nonsticky_move_count": int(anti_diag["nonsticky_move_count"]),
        "qos_guard_reject_count": int(anti_diag["qos_guard_reject_count"]),
        "handover_guard_reject_count": int(
            anti_diag["handover_guard_reject_count"]
        ),
        "forbidden_claims": [
            "Do not claim EE-MODQN effectiveness.",
            "Do not claim physical energy saving.",
            "Do not claim HOBS optimizer reproduction.",
            "Do not use scalar reward alone as success evidence.",
            "Do not introduce Catfish, Multi-Catfish, or RA-EE association.",
        ],
    }, step_records


def _write_step_trace(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "eval_seed",
        "step_index",
        "active_beam_count",
        "active_beam_load_gap",
        "max_active_beam_load",
        "total_active_power_w",
        "sum_throughput_bps",
        "ee_active_tx",
        "r2_step_mean",
        "r3_step_mean",
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
    route_label: str = "HOBS active-TX EE anti-collapse bounded design gate",
) -> dict[str, Any]:
    cfg = load_training_yaml(config_path)
    env = build_environment(cfg)
    trainer_cfg = build_trainer_config(cfg)
    seeds = get_seeds(cfg)
    eval_seeds = tuple(int(seed) for seed in seeds["evaluation_seed_set"])

    if int(trainer_cfg.episodes) != MAX_TINY_EPISODES:
        raise ValueError(
            "Anti-collapse gate must use the unchanged tiny "
            f"{MAX_TINY_EPISODES}-episode budget, got {trainer_cfg.episodes}."
        )
    if not eval_seeds:
        raise ValueError("Anti-collapse gate requires a non-empty eval seed set.")

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

    diagnostics, step_trace = _evaluate_learned_greedy_policy(trainer, eval_seeds)
    _write_step_trace(out / "greedy_eval_step_trace.csv", step_trace)

    summary = {
        "namespace": str(Path(output_dir)),
        "route": route_label,
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
            "anti_collapse_action_constraint_enabled": (
                trainer_cfg.anti_collapse_action_constraint_enabled
            ),
            "anti_collapse_constraint": _anti_collapse_subset(trainer_cfg),
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


def interpret_anti_collapse_verdict(
    control_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
    boundary_proof: dict[str, Any],
) -> dict[str, Any]:
    """Interpret hard-stop and acceptance criteria for the bounded gate."""
    control_diag = control_summary["diagnostics"]
    candidate_diag = candidate_summary["diagnostics"]
    tol = predeclared_tolerances()

    if not bool(boundary_proof.get("matched_boundary_pass", False)):
        return {
            "status": "NEEDS MORE DESIGN",
            "reasons": ["candidate/control boundary cannot be proven matched"],
            "acceptance_pass": False,
            "predeclared_tolerances": tol,
        }

    reasons: list[str] = []
    if bool(candidate_diag.get("all_evaluated_steps_one_active_beam", True)):
        reasons.append("candidate still all_evaluated_steps_one_active_beam=true")
    if not bool(candidate_diag.get("denominator_varies_in_eval", False)):
        reasons.append("candidate denominator_varies_in_eval=false")
    if bool(candidate_diag.get("active_power_single_point_distribution", True)):
        reasons.append("candidate active_power_single_point_distribution=true")

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
    outage_delta = (
        float(candidate_diag.get("outage_ratio", 0.0))
        - float(control_diag.get("outage_ratio", 0.0))
    )
    handover_delta = (
        int(candidate_diag.get("handover_count", 0))
        - int(control_diag.get("handover_count", 0))
    )
    r2_delta = (
        float(candidate_diag.get("r2_mean", 0.0))
        - float(control_diag.get("r2_mean", 0.0))
    )
    if p05_ratio < P05_THROUGHPUT_RATIO_MIN:
        reasons.append("candidate p05_throughput_ratio_vs_control below 0.95")
    if served_delta < SERVED_RATIO_MIN_DELTA:
        reasons.append("candidate served_ratio decreases vs control")
    if outage_delta > OUTAGE_RATIO_MAX_DELTA:
        reasons.append("candidate outage_ratio increases vs control")
    if handover_delta > HANDOVER_DELTA_MAX:
        reasons.append("candidate handover regression exceeds predeclared tolerance")
    if r2_delta < R2_MEAN_DELTA_MIN:
        reasons.append("candidate r2 regression exceeds predeclared tolerance")

    violation_keys = (
        "budget_violation_count",
        "per_beam_power_violation_count",
        "inactive_beam_nonzero_power_step_count",
    )
    for key in violation_keys:
        if int(candidate_diag.get(key, 0)) > 0:
            reasons.append(f"candidate {key} > 0")

    scalar_delta = (
        float(candidate_diag.get("episode_scalar_reward_diagnostic_mean", 0.0))
        - float(control_diag.get("episode_scalar_reward_diagnostic_mean", 0.0))
    )
    scalar_only_success = scalar_delta > 0.0 and bool(reasons)
    if scalar_only_success:
        reasons.append("candidate scalar reward improves but acceptance criteria fail")

    status = "PASS" if not reasons else "BLOCK"
    return {
        "status": status,
        "acceptance_pass": status == "PASS",
        "reasons": reasons,
        "predeclared_tolerances": tol,
        "candidate_minus_control": {
            "raw_throughput_mean_bps": (
                float(candidate_diag.get("raw_throughput_mean_bps", 0.0))
                - float(control_diag.get("raw_throughput_mean_bps", 0.0))
            ),
            "p05_throughput_bps": candidate_p05 - control_p05,
            "p05_throughput_ratio_vs_control": float(p05_ratio),
            "served_ratio": float(served_delta),
            "outage_ratio": float(outage_delta),
            "handover_count": int(handover_delta),
            "r2": float(r2_delta),
            "load_balance_metric": (
                float(candidate_diag.get("load_balance_metric", 0.0))
                - float(control_diag.get("load_balance_metric", 0.0))
            ),
            "episode_scalar_reward_diagnostic_mean": float(scalar_delta),
        },
        "scalar_only_success": bool(scalar_only_success),
    }


def export_tiny_matched_anti_collapse_pilot(
    control_config_path: str | Path = CONTROL_CONFIG,
    candidate_config_path: str | Path = CANDIDATE_CONFIG,
    *,
    control_output_dir: str | Path = CONTROL_ARTIFACT_DIR,
    candidate_output_dir: str | Path = CANDIDATE_ARTIFACT_DIR,
    paired_output_dir: str | Path = PAIRED_COMPARISON_DIR,
) -> dict[str, Any]:
    """Run and export the bounded matched anti-collapse pilot."""
    boundary = prove_matched_boundary(control_config_path, candidate_config_path)
    control_summary = _train_and_evaluate_arm(control_config_path, control_output_dir)
    candidate_summary = _train_and_evaluate_arm(
        candidate_config_path,
        candidate_output_dir,
    )
    verdict = interpret_anti_collapse_verdict(
        control_summary,
        candidate_summary,
        boundary,
    )
    mechanism = str(boundary["anti_collapse_constraint"]["mode"])
    if mechanism == "qos-sticky-overflow-reassignment":
        gate_proof = (
            "Whether a QoS- and stickiness-preserving overflow-only "
            "reassignment hook can remove one-active-beam learned greedy "
            "collapse under the same HOBS active-TX EE reward and DPC sidecar."
        )
    else:
        gate_proof = (
            "Whether a minimal opt-in capacity-aware assignment constraint can "
            "remove one-active-beam learned greedy collapse under the same HOBS "
            "active-TX EE reward and DPC sidecar."
        )

    paired = {
        "namespace": str(paired_output_dir),
        "route": f"HOBS active-TX EE anti-collapse bounded design gate: {mechanism}",
        "boundary_proof": boundary,
        "control_summary_path": str(Path(control_output_dir) / "summary.json"),
        "candidate_summary_path": str(Path(candidate_output_dir) / "summary.json"),
        "control_diagnostics": control_summary["diagnostics"],
        "candidate_diagnostics": candidate_summary["diagnostics"],
        "paired_verdict": verdict,
        "what_this_gate_proves": gate_proof,
        "what_this_gate_does_not_prove": [
            "EE-MODQN effectiveness",
            "physical energy saving",
            "HOBS optimizer reproduction",
            "DPC support in the MODQN paper",
            "Catfish or Multi-Catfish effectiveness",
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
        "# HOBS Active-TX EE Anti-Collapse Arm",
        "",
        f"Config: `{summary['config_path']}`",
        f"R1 mode: `{summary['trainer_config']['r1_reward_mode']}`",
        "Anti-collapse enabled: "
        f"`{summary['trainer_config']['anti_collapse_action_constraint_enabled']}`",
        f"Checkpoint used for greedy eval: `{summary['checkpoint_used_for_greedy_eval']}`",
        "",
        "## Greedy Diagnostics",
        "",
        f"- `denominator_varies_in_eval`: `{diag['denominator_varies_in_eval']}`",
        f"- `all_evaluated_steps_one_active_beam`: `{diag['all_evaluated_steps_one_active_beam']}`",
        f"- `active_beam_count_distribution`: `{diag['active_beam_count_distribution']}`",
        f"- `active_power_single_point_distribution`: `{diag['active_power_single_point_distribution']}`",
        f"- `power_control_activity_rate`: `{diag['power_control_activity_rate']}`",
        f"- `throughput_vs_ee_pearson`: `{diag['throughput_vs_ee_pearson']}`",
        f"- `raw_throughput_mean_bps`: `{diag['raw_throughput_mean_bps']}`",
        f"- `p05_throughput_bps`: `{diag['p05_throughput_bps']}`",
        f"- `served_ratio`: `{diag['served_ratio']}`",
        f"- `outage_ratio`: `{diag['outage_ratio']}`",
        f"- `handover_count`: `{diag['handover_count']}`",
        f"- `r2`: `{diag['r2']}`",
        f"- `load_balance_metric`: `{diag['load_balance_metric']}`",
        f"- `budget_violation_count`: `{diag['budget_violation_count']}`",
        f"- `per_beam_power_violation_count`: `{diag['per_beam_power_violation_count']}`",
        f"- `inactive_beam_nonzero_power_step_count`: `{diag['inactive_beam_nonzero_power_step_count']}`",
        f"- `overflow_steps`: `{diag['overflow_steps']}`",
        f"- `overflow_user_count`: `{diag['overflow_user_count']}`",
        f"- `sticky_override_count`: `{diag['sticky_override_count']}`",
        f"- `nonsticky_move_count`: `{diag['nonsticky_move_count']}`",
        f"- `qos_guard_reject_count`: `{diag['qos_guard_reject_count']}`",
        f"- `handover_guard_reject_count`: `{diag['handover_guard_reject_count']}`",
        "",
        "## Claim Boundary",
        "",
        "- Scalar reward is diagnostic only.",
        "- This arm alone does not prove effectiveness.",
        "- The constraint is opt-in and changes only this new gate surface.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _write_paired_review(path: Path, paired: dict[str, Any]) -> None:
    verdict = paired["paired_verdict"]
    cdiag = paired["candidate_diagnostics"]
    delta = verdict.get("candidate_minus_control", {})
    lines = [
        "# HOBS Active-TX EE Anti-Collapse Gate",
        "",
        f"Status: **{verdict['status']}**",
        "",
        "## Matched Boundary",
        "",
        f"- `matched_boundary_pass`: `{paired['boundary_proof']['matched_boundary_pass']}`",
        f"- `same_dpc_sidecar`: `{paired['boundary_proof']['checks']['same_dpc_sidecar']}`",
        f"- `same_seed_block`: `{paired['boundary_proof']['checks']['same_seed_block']}`",
        f"- `same_checkpoint_rule`: `{paired['boundary_proof']['checks']['same_checkpoint_rule']}`",
        f"- `same_constraint_parameters`: `{paired['boundary_proof']['checks']['same_constraint_parameters']}`",
        f"- `candidate_constraint_enabled`: `{paired['boundary_proof']['checks']['candidate_constraint_enabled']}`",
        "",
        "## Candidate Metrics",
        "",
        f"- `all_evaluated_steps_one_active_beam`: `{cdiag['all_evaluated_steps_one_active_beam']}`",
        f"- `active_beam_count_distribution`: `{cdiag['active_beam_count_distribution']}`",
        f"- `denominator_varies_in_eval`: `{cdiag['denominator_varies_in_eval']}`",
        f"- `active_power_single_point_distribution`: `{cdiag['active_power_single_point_distribution']}`",
        f"- `distinct_total_active_power_w_values`: `{cdiag['distinct_total_active_power_w_values']}`",
        f"- `power_control_activity_rate`: `{cdiag['power_control_activity_rate']}`",
        f"- `throughput_vs_ee_pearson`: `{cdiag['throughput_vs_ee_pearson']}`",
        f"- `same_policy_throughput_vs_ee_rescore_ranking_change`: `{cdiag['same_policy_throughput_vs_ee_rescore_ranking_change']}`",
        f"- `raw_throughput_mean_bps`: `{cdiag['raw_throughput_mean_bps']}`",
        f"- `p05_throughput_bps`: `{cdiag['p05_throughput_bps']}`",
        f"- `served_ratio`: `{cdiag['served_ratio']}`",
        f"- `outage_ratio`: `{cdiag['outage_ratio']}`",
        f"- `handover_count`: `{cdiag['handover_count']}`",
        f"- `r2`: `{cdiag['r2']}`",
        f"- `load_balance_metric`: `{cdiag['load_balance_metric']}`",
        f"- `scalar_reward_diagnostic_mean`: `{cdiag['scalar_reward_diagnostic_mean']}`",
        f"- `budget_violation_count`: `{cdiag['budget_violation_count']}`",
        f"- `per_beam_power_violation_count`: `{cdiag['per_beam_power_violation_count']}`",
        f"- `inactive_beam_nonzero_power_step_count`: `{cdiag['inactive_beam_nonzero_power_step_count']}`",
        f"- `overflow_steps`: `{cdiag['overflow_steps']}`",
        f"- `overflow_user_count`: `{cdiag['overflow_user_count']}`",
        f"- `sticky_override_count`: `{cdiag['sticky_override_count']}`",
        f"- `nonsticky_move_count`: `{cdiag['nonsticky_move_count']}`",
        f"- `qos_guard_reject_count`: `{cdiag['qos_guard_reject_count']}`",
        f"- `handover_guard_reject_count`: `{cdiag['handover_guard_reject_count']}`",
        "",
        "## Candidate Minus Control",
        "",
        f"- `p05_throughput_ratio_vs_control`: `{delta.get('p05_throughput_ratio_vs_control')}`",
        f"- `served_ratio`: `{delta.get('served_ratio')}`",
        f"- `outage_ratio`: `{delta.get('outage_ratio')}`",
        f"- `handover_count`: `{delta.get('handover_count')}`",
        f"- `r2`: `{delta.get('r2')}`",
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
        "- Do not claim HOBS optimizer reproduction.",
        "- Do not claim DPC as MODQN-paper-backed optimizer.",
        "- Do not use scalar reward alone as success evidence.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


__all__ = [
    "CANDIDATE_CONFIG",
    "CONTROL_CONFIG",
    "export_tiny_matched_anti_collapse_pilot",
    "interpret_anti_collapse_verdict",
    "predeclared_tolerances",
    "prove_matched_boundary",
]
