"""Phase 03 paired MODQN-control vs EE-MODQN validation helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from ..algorithms.modqn import MODQNTrainer
from ..artifacts import read_run_metadata
from ..config_loader import build_environment, build_trainer_config
from ..runtime.objective_math import scalarize_objectives
from ..runtime.trainer_spec import (
    R1_REWARD_MODE_PER_USER_BEAM_EE_CREDIT,
    R1_REWARD_MODE_PER_USER_EE_CREDIT,
)
from ._common import write_json
from .phase03a_diagnostics import distribution


_CHECKPOINT_ROLES = (
    ("final", "primary_final"),
    ("best-eval", "secondary_best_eval"),
)


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _correlation(x_values: list[float], y_values: list[float]) -> dict[str, float | None]:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return {"pearson": None, "spearman": None}
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        pearson = None
    else:
        pearson = float(np.corrcoef(x, y)[0, 1])
    rx = _rank(x)
    ry = _rank(y)
    if float(np.std(rx)) == 0.0 or float(np.std(ry)) == 0.0:
        spearman = None
    else:
        spearman = float(np.corrcoef(rx, ry)[0, 1])
    return {"pearson": pearson, "spearman": spearman}


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def _winner(control_value: float | None, ee_value: float | None) -> str:
    if control_value is None or ee_value is None:
        return "unavailable"
    if ee_value > control_value:
        return "EE-MODQN"
    if control_value > ee_value:
        return "MODQN-control"
    return "tie"


def _pct_delta(control_value: float | None, ee_value: float | None) -> float | None:
    if control_value is None or ee_value is None:
        return None
    if abs(control_value) < 1e-12:
        return None
    return float((ee_value - control_value) / abs(control_value))


def _ee_credit_for_mode(reward: Any, r1_reward_mode: str) -> float:
    """Return the EE diagnostic value used for same-policy EE rescoring."""
    if r1_reward_mode == R1_REWARD_MODE_PER_USER_BEAM_EE_CREDIT:
        return float(reward.r1_beam_power_efficiency_credit)
    if r1_reward_mode == R1_REWARD_MODE_PER_USER_EE_CREDIT:
        return float(reward.r1_energy_efficiency_credit)
    return float(reward.r1_energy_efficiency_credit)


def _checkpoint_paths(metadata) -> dict[str, Path]:
    files = metadata.checkpoint_files
    paths: dict[str, Path] = {"final": Path(files.primary_final)}
    if files.secondary_best_eval is not None:
        paths["best-eval"] = Path(files.secondary_best_eval)
    return paths


def _evaluate_checkpoint(
    *,
    run_dir: Path,
    method_label: str,
    checkpoint_role: str,
    checkpoint_path: Path,
    evaluation_seed_set: tuple[int, ...],
    ee_rescore_r1_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    metadata = read_run_metadata(run_dir / "run_metadata.json")
    cfg = metadata.resolved_config_snapshot
    env = build_environment(cfg)
    trainer_cfg = build_trainer_config(cfg)
    trainer = MODQNTrainer(
        env=env,
        config=trainer_cfg,
        train_seed=metadata.seeds.train_seed,
        env_seed=metadata.seeds.environment_seed,
        mobility_seed=metadata.seeds.mobility_seed,
    )
    payload = trainer.load_checkpoint(checkpoint_path, load_optimizers=False)
    checkpoint_kind = str(payload.get("checkpoint_kind"))
    checkpoint_episode = int(payload.get("episode"))

    episode_rows: list[dict[str, Any]] = []
    user_step_rows: list[dict[str, Any]] = []
    all_throughput: list[float] = []
    all_ee_credit: list[float] = []
    all_active_beam_counts: list[int] = []
    all_total_active_power_w: list[float] = []

    for eval_seed in evaluation_seed_set:
        env_seed_seq, mobility_seed_seq = np.random.SeedSequence(eval_seed).spawn(2)
        env_rng = np.random.default_rng(env_seed_seq)
        mobility_rng = np.random.default_rng(mobility_seed_seq)

        states, masks, _diag = trainer.env.reset(env_rng, mobility_rng)
        encoded = trainer.encode_states(states)

        objective_sum = np.zeros(3, dtype=np.float64)
        throughput_sum = 0.0
        ee_credit_sum = 0.0
        ee_rescore_sum = 0.0
        beam_power_ee_credit_sum = 0.0
        r2_sum = 0.0
        r3_sum = 0.0
        handovers = 0
        intra_handovers = 0
        inter_handovers = 0
        served = 0
        unserved = 0
        raw_values: list[float] = []
        ee_values: list[float] = []
        step_ee_system_values: list[float] = []
        step_power_values: list[float] = []
        step_load_gap_values: list[float] = []

        for _step_idx in range(trainer.env.config.steps_per_episode):
            actions = trainer.select_actions(encoded, masks, eps=0.0)
            result = trainer.env.step(actions, env_rng)

            active_power_w = float(
                np.sum(result.beam_transmit_power_w[result.active_beam_mask])
            )
            active_beam_count = int(np.count_nonzero(result.active_beam_mask))
            all_active_beam_counts.append(active_beam_count)
            all_total_active_power_w.append(active_power_w)
            step_throughput = float(
                np.sum([rw.r1_throughput for rw in result.rewards], dtype=np.float64)
            )
            step_ee_system = (
                None if active_power_w <= 0.0 else step_throughput / active_power_w
            )
            if step_ee_system is not None:
                step_ee_system_values.append(float(step_ee_system))
            step_power_values.append(active_power_w)
            if result.rewards:
                step_load_gap_values.append(
                    float(-result.rewards[0].r3_load_balance * trainer.num_users)
                )

            for uid, rw in enumerate(result.rewards):
                reward_vec = trainer.reward_vector_from_step_result(result, uid)
                objective_sum += reward_vec

                raw_thr = float(rw.r1_throughput)
                ee_credit = float(rw.r1_energy_efficiency_credit)
                beam_power_ee_credit = float(rw.r1_beam_power_efficiency_credit)
                ee_rescore_credit = _ee_credit_for_mode(rw, ee_rescore_r1_mode)
                throughput_sum += raw_thr
                ee_credit_sum += ee_credit
                beam_power_ee_credit_sum += beam_power_ee_credit
                ee_rescore_sum += ee_rescore_credit
                r2_sum += float(rw.r2_handover)
                r3_sum += float(rw.r3_load_balance)
                raw_values.append(raw_thr)
                ee_values.append(ee_rescore_credit)
                all_throughput.append(raw_thr)
                all_ee_credit.append(ee_rescore_credit)

                if raw_thr > 0.0:
                    served += 1
                else:
                    unserved += 1
                if rw.r2_handover < 0.0:
                    handovers += 1
                    if np.isclose(rw.r2_handover, -trainer.env.config.phi1):
                        intra_handovers += 1
                    elif np.isclose(rw.r2_handover, -trainer.env.config.phi2):
                        inter_handovers += 1

                user_step_rows.append(
                    {
                        "method": method_label,
                        "checkpoint_role": checkpoint_role,
                        "checkpoint_kind": checkpoint_kind,
                        "checkpoint_episode": checkpoint_episode,
                        "evaluation_seed": int(eval_seed),
                        "step_index": int(result.step_index),
                        "user_index": int(uid),
                        "raw_throughput_bps": raw_thr,
                        "per_user_ee_credit_bps_per_w": ee_credit,
                        "per_user_beam_power_ee_credit_bps_per_w": (
                            beam_power_ee_credit
                        ),
                        "ee_rescore_credit_bps_per_w": ee_rescore_credit,
                        "r1_objective": float(reward_vec[0]),
                        "r2_handover": float(rw.r2_handover),
                        "r3_load_balance": float(rw.r3_load_balance),
                        "active_beam_count": active_beam_count,
                        "total_active_beam_power_w": active_power_w,
                        "ee_system_bps_per_w": step_ee_system,
                    }
                )

            if result.done:
                break
            encoded = trainer.encode_states(result.user_states)
            masks = result.action_masks

        user_step_count = max(len(raw_values), 1)
        active_power_sum = float(np.sum(step_power_values, dtype=np.float64))
        avg_objective = objective_sum / max(trainer.num_users, 1)
        throughput_rescore = np.array(
            [throughput_sum / max(trainer.num_users, 1),
             r2_sum / max(trainer.num_users, 1),
             r3_sum / max(trainer.num_users, 1)],
            dtype=np.float64,
        )
        ee_rescore = np.array(
            [ee_rescore_sum / max(trainer.num_users, 1),
             r2_sum / max(trainer.num_users, 1),
             r3_sum / max(trainer.num_users, 1)],
            dtype=np.float64,
        )

        episode_rows.append(
            {
                "method": method_label,
                "checkpoint_role": checkpoint_role,
                "checkpoint_kind": checkpoint_kind,
                "checkpoint_episode": checkpoint_episode,
                "evaluation_seed": int(eval_seed),
                "r1_reward_mode": trainer_cfg.r1_reward_mode,
                "scalar_reward": scalarize_objectives(
                    avg_objective,
                    trainer_cfg.objective_weights,
                ),
                "objective_r1_mean": float(avg_objective[0]),
                "r2_mean": float(avg_objective[1]),
                "r3_mean": float(avg_objective[2]),
                "raw_throughput_episode_sum_bps": throughput_sum,
                "raw_throughput_episode_mean_per_user_bps": (
                    throughput_sum / max(trainer.num_users, 1)
                ),
                "raw_throughput_mean_user_step_bps": (
                    throughput_sum / user_step_count
                ),
                "raw_throughput_low_p05_bps": float(np.percentile(raw_values, 5)),
                "served_ratio": served / max(served + unserved, 1),
                "unserved_ratio": unserved / max(served + unserved, 1),
                "ee_system_aggregate_bps_per_w": (
                    None if active_power_sum <= 0.0 else throughput_sum / active_power_sum
                ),
                "ee_system_step_mean_bps_per_w": (
                    None
                    if not step_ee_system_values
                    else float(np.mean(step_ee_system_values))
                ),
                "per_user_ee_reward_mean_bps_per_w": (
                    ee_credit_sum / user_step_count
                ),
                "per_user_beam_power_ee_reward_mean_bps_per_w": (
                    beam_power_ee_credit_sum / user_step_count
                ),
                "ee_rescore_reward_mean_bps_per_w": (
                    ee_rescore_sum / user_step_count
                ),
                "total_active_beam_power_episode_sum_w": active_power_sum,
                "total_active_beam_power_mean_w": float(np.mean(step_power_values)),
                "total_active_beam_power_min_w": float(np.min(step_power_values)),
                "total_active_beam_power_max_w": float(np.max(step_power_values)),
                "active_beam_count_mean": float(
                    np.mean(all_active_beam_counts[-len(step_power_values):])
                ),
                "active_beam_count_min": float(
                    np.min(all_active_beam_counts[-len(step_power_values):])
                ),
                "active_beam_count_max": float(
                    np.max(all_active_beam_counts[-len(step_power_values):])
                ),
                "handover_count": int(handovers),
                "handover_per_user_episode": handovers / max(trainer.num_users, 1),
                "intra_handover_count": int(intra_handovers),
                "inter_handover_count": int(inter_handovers),
                "load_balance_gap_mean_bps": (
                    None
                    if not step_load_gap_values
                    else float(np.mean(step_load_gap_values))
                ),
                "counterfactual_scalar_throughput_r1": scalarize_objectives(
                    throughput_rescore,
                    trainer_cfg.objective_weights,
                ),
                "counterfactual_scalar_ee_r1": scalarize_objectives(
                    ee_rescore,
                    trainer_cfg.objective_weights,
                ),
            }
        )

    summary = _summarize_checkpoint_rows(
        method_label=method_label,
        checkpoint_role=checkpoint_role,
        checkpoint_kind=checkpoint_kind,
        checkpoint_episode=checkpoint_episode,
        r1_reward_mode=trainer_cfg.r1_reward_mode,
        episode_rows=episode_rows,
        all_throughput=all_throughput,
        all_ee_credit=all_ee_credit,
        all_active_beam_counts=all_active_beam_counts,
        all_total_active_power_w=all_total_active_power_w,
    )
    return episode_rows, user_step_rows, summary


def _summarize_checkpoint_rows(
    *,
    method_label: str,
    checkpoint_role: str,
    checkpoint_kind: str,
    checkpoint_episode: int,
    r1_reward_mode: str,
    episode_rows: list[dict[str, Any]],
    all_throughput: list[float],
    all_ee_credit: list[float],
    all_active_beam_counts: list[int],
    all_total_active_power_w: list[float],
) -> dict[str, Any]:
    numeric_keys = [
        key
        for key, value in episode_rows[0].items()
        if isinstance(value, (int, float)) and key not in {"evaluation_seed"}
    ]
    summary: dict[str, Any] = {
        "method": method_label,
        "checkpoint_role": checkpoint_role,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_episode": checkpoint_episode,
        "r1_reward_mode": r1_reward_mode,
        "evaluation_seed_count": len(episode_rows),
    }
    for key in numeric_keys:
        values = [
            float(row[key])
            for row in episode_rows
            if row.get(key) is not None and np.isfinite(float(row[key]))
        ]
        mean, std = _mean_std(values)
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
    summary["throughput_vs_per_user_ee_correlation"] = _correlation(
        all_throughput,
        all_ee_credit,
    )
    summary["active_beam_count_distribution"] = distribution(all_active_beam_counts)
    summary["total_active_beam_power_w_distribution"] = distribution(
        all_total_active_power_w
    )
    summary["denominator_varies_in_eval"] = (
        summary["total_active_beam_power_w_distribution"]["max"] is not None
        and summary["total_active_beam_power_w_distribution"]["min"] is not None
        and float(summary["total_active_beam_power_w_distribution"]["max"])
        > float(summary["total_active_beam_power_w_distribution"]["min"])
    )
    summary["all_evaluated_steps_one_active_beam"] = (
        bool(all_active_beam_counts)
        and min(all_active_beam_counts) == 1
        and max(all_active_beam_counts) == 1
    )
    return summary


def _build_comparison(
    *,
    control_summary: dict[str, Any],
    ee_summary: dict[str, Any],
) -> dict[str, Any]:
    metrics = [
        "ee_system_aggregate_bps_per_w",
        "ee_system_step_mean_bps_per_w",
        "per_user_ee_reward_mean_bps_per_w",
        "per_user_beam_power_ee_reward_mean_bps_per_w",
        "ee_rescore_reward_mean_bps_per_w",
        "raw_throughput_mean_user_step_bps",
        "raw_throughput_low_p05_bps",
        "served_ratio",
        "active_beam_count_mean",
        "total_active_beam_power_mean_w",
        "handover_count",
        "r2_mean",
        "r3_mean",
        "load_balance_gap_mean_bps",
        "scalar_reward",
        "counterfactual_scalar_throughput_r1",
        "counterfactual_scalar_ee_r1",
    ]
    deltas: dict[str, Any] = {}
    for metric in metrics:
        control_value = control_summary.get(f"{metric}_mean")
        ee_value = ee_summary.get(f"{metric}_mean")
        deltas[metric] = {
            "control_mean": control_value,
            "ee_mean": ee_value,
            "delta": (
                None
                if control_value is None or ee_value is None
                else float(ee_value - control_value)
            ),
            "pct_delta": _pct_delta(control_value, ee_value),
            "winner": _winner(control_value, ee_value),
        }

    reward_hacking_flags = {
        "ee_system_rises_while_mean_throughput_drops_gt_10pct": (
            deltas["ee_system_aggregate_bps_per_w"]["delta"] is not None
            and deltas["ee_system_aggregate_bps_per_w"]["delta"] > 0.0
            and deltas["raw_throughput_mean_user_step_bps"]["pct_delta"] is not None
            and deltas["raw_throughput_mean_user_step_bps"]["pct_delta"] < -0.10
        ),
        "ee_system_rises_while_low_p05_throughput_drops_gt_10pct": (
            deltas["ee_system_aggregate_bps_per_w"]["delta"] is not None
            and deltas["ee_system_aggregate_bps_per_w"]["delta"] > 0.0
            and deltas["raw_throughput_low_p05_bps"]["pct_delta"] is not None
            and deltas["raw_throughput_low_p05_bps"]["pct_delta"] < -0.10
        ),
        "ee_system_rises_while_served_ratio_drops_gt_0_05": (
            deltas["ee_system_aggregate_bps_per_w"]["delta"] is not None
            and deltas["ee_system_aggregate_bps_per_w"]["delta"] > 0.0
            and deltas["served_ratio"]["delta"] is not None
            and deltas["served_ratio"]["delta"] < -0.05
        ),
        "power_falls_with_served_ratio_drop": (
            deltas["total_active_beam_power_mean_w"]["delta"] is not None
            and deltas["total_active_beam_power_mean_w"]["delta"] < 0.0
            and deltas["served_ratio"]["delta"] is not None
            and deltas["served_ratio"]["delta"] < -0.05
        ),
        "per_user_ee_rises_but_ee_system_does_not": (
            deltas["ee_rescore_reward_mean_bps_per_w"]["delta"] is not None
            and deltas["ee_rescore_reward_mean_bps_per_w"]["delta"] > 0.0
            and (
                deltas["ee_system_aggregate_bps_per_w"]["delta"] is None
                or deltas["ee_system_aggregate_bps_per_w"]["delta"] <= 0.0
            )
        ),
    }
    reward_hacking_flags["any"] = any(bool(value) for value in reward_hacking_flags.values())

    ranking = {
        "throughput_rescore_winner": deltas[
            "counterfactual_scalar_throughput_r1"
        ]["winner"],
        "ee_rescore_winner": deltas["counterfactual_scalar_ee_r1"]["winner"],
        "ranking_changes_under_same_policy_rescore": (
            deltas["counterfactual_scalar_throughput_r1"]["winner"]
            != deltas["counterfactual_scalar_ee_r1"]["winner"]
        ),
        "raw_throughput_winner": deltas["raw_throughput_mean_user_step_bps"]["winner"],
        "ee_system_winner": deltas["ee_system_aggregate_bps_per_w"]["winner"],
        "scalar_reward_winner": deltas["scalar_reward"]["winner"],
    }
    control_corr = control_summary["throughput_vs_per_user_ee_correlation"]
    ee_corr = ee_summary["throughput_vs_per_user_ee_correlation"]
    rescaling_check = {
        "control_denominator_varies_in_eval": bool(
            control_summary["denominator_varies_in_eval"]
        ),
        "ee_denominator_varies_in_eval": bool(
            ee_summary["denominator_varies_in_eval"]
        ),
        "control_all_evaluated_steps_one_active_beam": bool(
            control_summary["all_evaluated_steps_one_active_beam"]
        ),
        "ee_all_evaluated_steps_one_active_beam": bool(
            ee_summary["all_evaluated_steps_one_active_beam"]
        ),
        "control_active_beam_count_distribution": control_summary[
            "active_beam_count_distribution"
        ],
        "ee_active_beam_count_distribution": ee_summary[
            "active_beam_count_distribution"
        ],
        "control_total_active_beam_power_w_distribution": control_summary[
            "total_active_beam_power_w_distribution"
        ],
        "ee_total_active_beam_power_w_distribution": ee_summary[
            "total_active_beam_power_w_distribution"
        ],
        "control_throughput_ee_pearson": control_corr["pearson"],
        "control_throughput_ee_spearman": control_corr["spearman"],
        "ee_throughput_ee_pearson": ee_corr["pearson"],
        "ee_throughput_ee_spearman": ee_corr["spearman"],
        "same_policy_rescore_changes_ranking": ranking[
            "ranking_changes_under_same_policy_rescore"
        ],
        "scalar_reward_increases_without_ee_system_change": (
            deltas["scalar_reward"]["winner"] == "EE-MODQN"
            and deltas["ee_system_aggregate_bps_per_w"]["winner"] != "EE-MODQN"
        ),
    }
    rescaling_check["high_rescaling_risk"] = (
        not rescaling_check["ee_denominator_varies_in_eval"]
        and rescaling_check["ee_throughput_ee_pearson"] is not None
        and abs(float(rescaling_check["ee_throughput_ee_pearson"])) >= 0.99
        and not rescaling_check["same_policy_rescore_changes_ranking"]
    )

    return {
        "checkpoint_role": control_summary["checkpoint_role"],
        "deltas": deltas,
        "ranking_check": ranking,
        "rescaling_check": rescaling_check,
        "reward_hacking_flags": reward_hacking_flags,
    }


def _phase03_decision(primary_comparison: dict[str, Any], *, phase: str) -> str:
    if primary_comparison["reward_hacking_flags"]["any"]:
        return "BLOCK"
    if phase == "phase-03b":
        deltas = primary_comparison["deltas"]
        rescaling = primary_comparison["rescaling_check"]
        ranking = primary_comparison["ranking_check"]

        ee_denominator_varies = bool(rescaling["ee_denominator_varies_in_eval"])
        no_one_beam_collapse = not bool(
            rescaling["ee_all_evaluated_steps_one_active_beam"]
        )
        ee_delta = deltas["ee_system_aggregate_bps_per_w"]["delta"]
        ee_improves = ee_delta is not None and ee_delta > 0.0
        throughput_guardrail = (
            deltas["raw_throughput_low_p05_bps"]["pct_delta"] is None
            or deltas["raw_throughput_low_p05_bps"]["pct_delta"] >= -0.10
        ) and (
            deltas["served_ratio"]["delta"] is None
            or deltas["served_ratio"]["delta"] >= -0.05
        )
        handover_guardrail = (
            deltas["handover_count"]["pct_delta"] is None
            or deltas["handover_count"]["pct_delta"] <= 0.10
        )
        ranking_not_identical = bool(
            ranking["ranking_changes_under_same_policy_rescore"]
        ) or (
            rescaling["ee_throughput_ee_pearson"] is not None
            and abs(float(rescaling["ee_throughput_ee_pearson"])) < 0.99
        )
        not_scalar_only = not bool(
            rescaling["scalar_reward_increases_without_ee_system_change"]
        )
        if (
            ee_denominator_varies
            and no_one_beam_collapse
            and ee_improves
            and throughput_guardrail
            and handover_guardrail
            and ranking_not_identical
            and not_scalar_only
        ):
            return "PROMOTE"
    return "NEEDS MORE EVIDENCE"


def _review_lines(summary: dict[str, Any]) -> list[str]:
    primary = summary["primary_comparison"]
    decision = summary["phase_03_decision"]
    protocol = summary["protocol"]
    return [
        f"# {protocol['phase']} EE-MODQN Paired Pilot Review",
        "",
        "This is a bounded paired pilot. It does not run Catfish or "
        "multi-Catfish.",
        "",
        "## Protocol",
        "",
        f"- control run: `{summary['inputs']['control_run_dir']}`",
        f"- EE run: `{summary['inputs']['ee_run_dir']}`",
        f"- primary checkpoint role: `{primary['checkpoint_role']}`",
        f"- control r1: `{protocol['control_r1']}`",
        f"- EE r1: `{protocol['ee_r1']}`",
        f"- reward normalization: `{protocol['reward_normalization_mode']}`",
        f"- load-balance calibration: `{protocol['load_balance_calibration_mode']}`",
        "- shared Phase 02B power surface: `active-load-concave`",
        "",
        "## Primary Deltas",
        "",
        f"- EE_system aggregate delta: "
        f"`{primary['deltas']['ee_system_aggregate_bps_per_w']['delta']}`",
        f"- raw throughput mean delta: "
        f"`{primary['deltas']['raw_throughput_mean_user_step_bps']['delta']}`",
        f"- served ratio delta: `{primary['deltas']['served_ratio']['delta']}`",
        f"- active beam count delta: "
        f"`{primary['deltas']['active_beam_count_mean']['delta']}`",
        f"- total active beam power delta: "
        f"`{primary['deltas']['total_active_beam_power_mean_w']['delta']}`",
        f"- EE denominator varies in eval: "
        f"`{primary['rescaling_check']['ee_denominator_varies_in_eval']}`",
        f"- EE all evaluated steps one active beam: "
        f"`{primary['rescaling_check']['ee_all_evaluated_steps_one_active_beam']}`",
        f"- same-policy rescore ranking changed: "
        f"`{primary['ranking_check']['ranking_changes_under_same_policy_rescore']}`",
        f"- high rescaling risk: "
        f"`{primary['rescaling_check']['high_rescaling_risk']}`",
        f"- reward-hacking flag: `{primary['reward_hacking_flags']['any']}`",
        "",
        "## Decision",
        "",
        f"- Phase 03 decision: `{decision}`",
        (
            "- Promotion, if present, is limited to this bounded Phase 03B "
            "objective-geometry gate; Catfish and final-method claims remain blocked."
            if decision == "PROMOTE"
            else "- No EE-MODQN effectiveness claim is promoted from this bounded pilot."
        ),
    ]


def export_phase03_paired_validation(
    *,
    control_run_dir: str | Path,
    ee_run_dir: str | Path,
    output_dir: str | Path,
    evaluation_seed_set: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Evaluate paired Phase 03 runs and export comparison artifacts."""
    control_dir = Path(control_run_dir)
    ee_dir = Path(ee_run_dir)
    out_dir = Path(output_dir)
    control_metadata = read_run_metadata(control_dir / "run_metadata.json")
    ee_metadata = read_run_metadata(ee_dir / "run_metadata.json")
    control_trainer_cfg = build_trainer_config(control_metadata.resolved_config_snapshot)
    ee_trainer_cfg = build_trainer_config(ee_metadata.resolved_config_snapshot)

    control_eval_seeds = tuple(control_metadata.seeds.evaluation_seed_set)
    ee_eval_seeds = tuple(ee_metadata.seeds.evaluation_seed_set)
    eval_seeds = tuple(
        int(seed)
        for seed in (
            evaluation_seed_set
            if evaluation_seed_set is not None
            else control_eval_seeds
        )
    )
    if not eval_seeds:
        raise ValueError("Phase 03 paired validation requires evaluation seeds.")
    if control_eval_seeds != ee_eval_seeds:
        raise ValueError(
            "Control and EE run metadata must use the same evaluation seed set."
        )
    phase = str(ee_trainer_cfg.phase or "phase-03")

    all_episode_rows: list[dict[str, Any]] = []
    all_user_step_rows: list[dict[str, Any]] = []
    checkpoint_summaries: list[dict[str, Any]] = []

    for method_label, run_dir, metadata in (
        ("MODQN-control", control_dir, control_metadata),
        ("EE-MODQN", ee_dir, ee_metadata),
    ):
        paths = _checkpoint_paths(metadata)
        for checkpoint_role, _field in _CHECKPOINT_ROLES:
            checkpoint_path = paths.get(checkpoint_role)
            if checkpoint_path is None:
                continue
            episode_rows, user_step_rows, checkpoint_summary = _evaluate_checkpoint(
                run_dir=run_dir,
                method_label=method_label,
                checkpoint_role=checkpoint_role,
                checkpoint_path=checkpoint_path,
                evaluation_seed_set=eval_seeds,
                ee_rescore_r1_mode=ee_trainer_cfg.r1_reward_mode,
            )
            all_episode_rows.extend(episode_rows)
            all_user_step_rows.extend(user_step_rows)
            checkpoint_summaries.append(checkpoint_summary)

    summary_by_key = {
        (row["method"], row["checkpoint_role"]): row
        for row in checkpoint_summaries
    }
    comparisons: list[dict[str, Any]] = []
    for checkpoint_role, _field in _CHECKPOINT_ROLES:
        control_summary = summary_by_key.get(("MODQN-control", checkpoint_role))
        ee_summary = summary_by_key.get(("EE-MODQN", checkpoint_role))
        if control_summary is None or ee_summary is None:
            continue
        comparisons.append(
            _build_comparison(
                control_summary=control_summary,
                ee_summary=ee_summary,
            )
        )
    if not comparisons:
        raise ValueError("No paired checkpoint comparisons were available.")

    primary_comparison = next(
        (
            comparison
            for comparison in comparisons
            if comparison["checkpoint_role"] == "best-eval"
        ),
        comparisons[0],
    )

    summary = {
        "inputs": {
            "control_run_dir": str(control_dir),
            "ee_run_dir": str(ee_dir),
            "output_dir": str(out_dir),
            "evaluation_seed_set": list(eval_seeds),
        },
        "protocol": {
            "phase": phase,
            "paired": True,
            "catfish": "disabled",
            "multi_catfish": "disabled",
            "control_r1": control_trainer_cfg.r1_reward_mode,
            "ee_r1": ee_trainer_cfg.r1_reward_mode,
            "r2": "handover penalty unchanged",
            "r3": "load balance calibrated only if disclosed by Phase 03B config",
            "reward_normalization_mode": ee_trainer_cfg.reward_normalization_mode,
            "load_balance_calibration_mode": (
                ee_trainer_cfg.load_balance_calibration_mode
            ),
            "power_surface": "Phase 02B active-load-concave",
            "claim_boundary": (
                "bounded objective-substitution / reward-geometry pilot only; "
                "no final Catfish-EE-MODQN or full paper-faithful claim"
            ),
        },
        "checkpoint_summaries": checkpoint_summaries,
        "comparisons": comparisons,
        "primary_comparison": primary_comparison,
        "phase_03_decision": _phase03_decision(
            primary_comparison,
            phase=phase,
        ),
        "remaining_blockers": [
            "Pilot budget is intentionally bounded and does not replace a larger approved run.",
            "The Phase 02B power surface is a disclosed synthesized proxy, not a paper-backed optimizer.",
            "Catfish, multi-Catfish, and final Catfish-EE-MODQN remain out of scope.",
        ],
    }

    episode_csv = _write_csv(
        out_dir / "phase03_episode_metrics.csv",
        all_episode_rows,
        fieldnames=list(all_episode_rows[0].keys()),
    )
    user_step_csv = _write_csv(
        out_dir / "phase03_user_step_metrics.csv",
        all_user_step_rows,
        fieldnames=list(all_user_step_rows[0].keys()),
    )
    summary_path = write_json(out_dir / "phase03_paired_summary.json", summary)
    review_path = out_dir / "review.md"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text("\n".join(_review_lines(summary)) + "\n")

    return {
        "phase03_paired_summary": summary_path,
        "phase03_episode_metrics": episode_csv,
        "phase03_user_step_metrics": user_step_csv,
        "review_md": review_path,
        "summary": summary,
    }


__all__ = ["export_phase03_paired_validation"]
