"""Phase 03C-C bounded paired power-MDP pilot comparison.

This module evaluates matched control/candidate run artifacts. It is bounded
pilot evidence only: no Catfish, no multi-Catfish, no frozen baseline mutation,
and scalar reward is reported only as a diagnostic.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..algorithms.modqn import MODQNTrainer
from ..artifacts import read_run_metadata
from ..config_loader import build_environment, build_trainer_config
from ..runtime.objective_math import scalarize_objectives
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


def _format_vector(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.12g}" for value in values.tolist())


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
    pearson = None
    if float(np.std(x)) != 0.0 and float(np.std(y)) != 0.0:
        pearson = float(np.corrcoef(x, y)[0, 1])
    rx = _rank(x)
    ry = _rank(y)
    spearman = None
    if float(np.std(rx)) != 0.0 and float(np.std(ry)) != 0.0:
        spearman = float(np.corrcoef(rx, ry)[0, 1])
    return {"pearson": pearson, "spearman": spearman}


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return None, None
    arr = np.asarray(clean, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def _pct_delta(control_value: float | None, candidate_value: float | None) -> float | None:
    if control_value is None or candidate_value is None or abs(control_value) < 1e-12:
        return None
    return float((candidate_value - control_value) / abs(control_value))


def _winner(control_value: float | None, candidate_value: float | None) -> str:
    if control_value is None or candidate_value is None:
        return "unavailable"
    if candidate_value > control_value:
        return "candidate"
    if control_value > candidate_value:
        return "control"
    return "tie"


def _categorical_distribution(values: list[str]) -> dict[str, Any]:
    counts = Counter(str(value) for value in values)
    return {
        "count": int(sum(counts.values())),
        "distinct": sorted(counts),
        "distinct_count": int(len(counts)),
        "histogram": {
            key: int(value)
            for key, value in sorted(counts.items(), key=lambda item: item[0])
        },
    }


def _unique_float_values(values: list[float], *, places: int = 12) -> list[float]:
    return sorted({round(float(value), places) for value in values})


def _checkpoint_paths(metadata: Any) -> dict[str, Path]:
    files = metadata.checkpoint_files
    paths: dict[str, Path] = {"final": Path(files.primary_final)}
    if files.secondary_best_eval is not None:
        paths["best-eval"] = Path(files.secondary_best_eval)
    return paths


def _power_surface_value(cfg: dict[str, Any]) -> dict[str, Any]:
    value = (
        cfg.get("resolved_assumptions", {})
        .get("hobs_power_surface", {})
        .get("value", {})
    )
    return value if isinstance(value, dict) else {}


def _evaluate_checkpoint(
    *,
    run_dir: Path,
    method_label: str,
    checkpoint_role: str,
    checkpoint_path: Path,
    evaluation_seed_set: tuple[int, ...],
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
    budget_w = env.power_surface_config.total_power_budget_w

    step_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    user_throughputs: list[float] = []
    step_throughputs: list[float] = []
    step_ee_throughputs: list[float] = []
    step_ee_values: list[float] = []
    active_beam_counts: list[int] = []
    total_active_powers: list[float] = []
    selected_profiles: list[str] = []
    r2_values: list[float] = []
    r3_values: list[float] = []
    scalar_values: list[float] = []
    throughput_rescore_values: list[float] = []
    ee_rescore_values: list[float] = []
    budget_excess_values: list[float] = []
    budget_violation_count = 0
    served_count = 0
    outage_count = 0
    handover_count = 0
    intra_handover_count = 0
    inter_handover_count = 0

    for eval_seed in evaluation_seed_set:
        env_seed_seq, mobility_seed_seq = np.random.SeedSequence(eval_seed).spawn(2)
        env_rng = np.random.default_rng(env_seed_seq)
        mobility_rng = np.random.default_rng(mobility_seed_seq)
        states, masks, _diag = trainer.env.reset(env_rng, mobility_rng)
        encoded = trainer.encode_states(states)

        objective_sum = np.zeros(3, dtype=np.float64)
        episode_throughput_sum = 0.0
        episode_power_sum = 0.0
        episode_r2_sum = 0.0
        episode_r3_sum = 0.0

        for _step_idx in range(trainer.env.config.steps_per_episode):
            actions = trainer.select_actions(encoded, masks, eps=0.0)
            result = trainer.env.step(actions, env_rng)

            active_mask = result.active_beam_mask.astype(bool, copy=False)
            beam_power = result.beam_transmit_power_w.astype(np.float64, copy=False)
            total_active_power = float(
                np.sum(beam_power[active_mask], dtype=np.float64)
            )
            active_beam_count = int(np.count_nonzero(active_mask))
            selected_profile = str(result.selected_power_profile)
            step_user_throughputs = [
                float(reward.r1_throughput)
                for reward in result.rewards
            ]
            step_throughput = float(np.sum(step_user_throughputs, dtype=np.float64))
            step_ee = None if total_active_power <= 0.0 else (
                step_throughput / total_active_power
            )
            step_served = sum(1 for value in step_user_throughputs if value > 0.0)
            step_outage = len(step_user_throughputs) - step_served
            step_handovers = sum(
                1 for reward in result.rewards if reward.r2_handover < 0.0
            )
            step_intra = sum(
                1
                for reward in result.rewards
                if np.isclose(reward.r2_handover, -trainer.env.config.phi1)
            )
            step_inter = sum(
                1
                for reward in result.rewards
                if np.isclose(reward.r2_handover, -trainer.env.config.phi2)
            )
            step_r2_values = [float(reward.r2_handover) for reward in result.rewards]
            step_r3_values = [float(reward.r3_load_balance) for reward in result.rewards]
            budget_excess = (
                0.0
                if budget_w is None
                else max(0.0, total_active_power - float(budget_w))
            )
            budget_violation = budget_excess > 1e-12

            step_throughputs.append(step_throughput)
            if step_ee is not None:
                step_ee_throughputs.append(step_throughput)
                step_ee_values.append(float(step_ee))
            active_beam_counts.append(active_beam_count)
            total_active_powers.append(total_active_power)
            selected_profiles.append(selected_profile)
            user_throughputs.extend(step_user_throughputs)
            r2_values.extend(step_r2_values)
            r3_values.extend(step_r3_values)
            budget_excess_values.append(budget_excess)
            budget_violation_count += int(budget_violation)
            served_count += step_served
            outage_count += step_outage
            handover_count += step_handovers
            intra_handover_count += step_intra
            inter_handover_count += step_inter
            episode_throughput_sum += step_throughput
            episode_power_sum += total_active_power
            episode_r2_sum += float(np.sum(step_r2_values, dtype=np.float64))
            episode_r3_sum += float(np.sum(step_r3_values, dtype=np.float64))

            for uid in range(trainer.num_users):
                objective_sum += trainer.reward_vector_from_step_result(result, uid)

            step_rows.append(
                {
                    "method": method_label,
                    "checkpoint_role": checkpoint_role,
                    "checkpoint_kind": checkpoint_kind,
                    "checkpoint_episode": checkpoint_episode,
                    "evaluation_seed": int(eval_seed),
                    "step_index": int(result.step_index),
                    "selected_power_profile": selected_profile,
                    "active_beam_count": active_beam_count,
                    "active_beam_mask": " ".join(
                        "1" if value else "0" for value in active_mask
                    ),
                    "beam_transmit_power_w": _format_vector(beam_power),
                    "total_active_beam_power_w": total_active_power,
                    "total_power_budget_w": budget_w,
                    "budget_violation": budget_violation,
                    "budget_excess_w": budget_excess,
                    "sum_user_throughput_bps": step_throughput,
                    "throughput_mean_user_step_bps": (
                        step_throughput / max(len(step_user_throughputs), 1)
                    ),
                    "throughput_p05_user_step_bps": float(
                        np.percentile(step_user_throughputs, 5)
                    ),
                    "EE_system_bps_per_w": step_ee,
                    "served_count": int(step_served),
                    "outage_count": int(step_outage),
                    "served_ratio": step_served / max(len(step_user_throughputs), 1),
                    "outage_ratio": step_outage / max(len(step_user_throughputs), 1),
                    "handover_count": int(step_handovers),
                    "r2_mean": float(np.mean(step_r2_values)),
                    "r3_mean": float(np.mean(step_r3_values)),
                }
            )

            if result.done:
                break
            encoded = trainer.encode_states(result.user_states)
            masks = result.action_masks

        avg_objective = objective_sum / max(trainer.num_users, 1)
        scalar = scalarize_objectives(avg_objective, trainer_cfg.objective_weights)
        episode_ee_system = (
            None
            if episode_power_sum <= 0.0
            else episode_throughput_sum / episode_power_sum
        )
        throughput_rescore = scalarize_objectives(
            np.array(
                [
                    episode_throughput_sum / max(trainer.num_users, 1),
                    episode_r2_sum / max(trainer.num_users, 1),
                    episode_r3_sum / max(trainer.num_users, 1),
                ],
                dtype=np.float64,
            ),
            trainer_cfg.objective_weights,
        )
        ee_rescore = scalarize_objectives(
            np.array(
                [
                    0.0 if episode_ee_system is None else episode_ee_system,
                    episode_r2_sum / max(trainer.num_users, 1),
                    episode_r3_sum / max(trainer.num_users, 1),
                ],
                dtype=np.float64,
            ),
            trainer_cfg.objective_weights,
        )
        scalar_values.append(float(scalar))
        throughput_rescore_values.append(float(throughput_rescore))
        ee_rescore_values.append(float(ee_rescore))
        episode_rows.append(
            {
                "method": method_label,
                "checkpoint_role": checkpoint_role,
                "checkpoint_kind": checkpoint_kind,
                "checkpoint_episode": checkpoint_episode,
                "evaluation_seed": int(eval_seed),
                "scalar_reward": float(scalar),
                "objective_r1_mean": float(avg_objective[0]),
                "r2_mean": float(avg_objective[1]),
                "r3_mean": float(avg_objective[2]),
                "raw_throughput_episode_sum_bps": episode_throughput_sum,
                "EE_system_aggregate_bps_per_w": episode_ee_system,
                "total_active_beam_power_episode_sum_w": episode_power_sum,
                "throughput_rescore_scalar": float(throughput_rescore),
                "EE_rescore_scalar": float(ee_rescore),
            }
        )

    total_throughput = float(np.sum(step_throughputs, dtype=np.float64))
    total_power = float(np.sum(total_active_powers, dtype=np.float64))
    scalar_mean, scalar_std = _mean_std(scalar_values)
    throughput_rescore_mean, throughput_rescore_std = _mean_std(
        throughput_rescore_values
    )
    ee_rescore_mean, ee_rescore_std = _mean_std(ee_rescore_values)

    selected_profile_distribution = _categorical_distribution(selected_profiles)
    total_power_distinct = _unique_float_values(total_active_powers)
    summary = {
        "method": method_label,
        "checkpoint_role": checkpoint_role,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_episode": checkpoint_episode,
        "phase": trainer_cfg.phase,
        "method_family": trainer_cfg.method_family,
        "comparison_role": trainer_cfg.comparison_role,
        "r1_reward_mode": trainer_cfg.r1_reward_mode,
        "power_surface_mode": env.power_surface_config.hobs_power_surface_mode,
        "configured_power_codebook_profile": (
            env.power_surface_config.power_codebook_profile
        ),
        "total_power_budget_w": budget_w,
        "evaluation_seed_count": len(evaluation_seed_set),
        "step_count": len(step_rows),
        "EE_system_aggregate_bps_per_w": (
            None if total_power <= 0.0 else total_throughput / total_power
        ),
        "EE_system_step_mean_bps_per_w": (
            None if not step_ee_values else float(np.mean(step_ee_values))
        ),
        "throughput_mean_user_step_bps": (
            None if not user_throughputs else float(np.mean(user_throughputs))
        ),
        "throughput_p05_user_step_bps": (
            None
            if not user_throughputs
            else float(np.percentile(user_throughputs, 5))
        ),
        "served_ratio": served_count / max(served_count + outage_count, 1),
        "outage_ratio": outage_count / max(served_count + outage_count, 1),
        "handover_count": int(handover_count),
        "intra_handover_count": int(intra_handover_count),
        "inter_handover_count": int(inter_handover_count),
        "r2_mean": _mean_std(r2_values)[0],
        "r3_mean": _mean_std(r3_values)[0],
        "scalar_reward_mean": scalar_mean,
        "scalar_reward_std": scalar_std,
        "throughput_rescore_scalar_mean": throughput_rescore_mean,
        "throughput_rescore_scalar_std": throughput_rescore_std,
        "EE_rescore_scalar_mean": ee_rescore_mean,
        "EE_rescore_scalar_std": ee_rescore_std,
        "active_beam_count_distribution": distribution(active_beam_counts),
        "total_active_beam_power_w_distribution": distribution(total_active_powers),
        "selected_power_profile_distribution": selected_profile_distribution,
        "selected_profile_distinct_count": selected_profile_distribution[
            "distinct_count"
        ],
        "denominator_varies_in_eval": len(total_power_distinct) > 1,
        "one_active_beam_step_ratio": (
            0.0
            if not active_beam_counts
            else float(np.mean([count == 1 for count in active_beam_counts]))
        ),
        "budget_violations": {
            "budget_w": budget_w,
            "step_count": int(budget_violation_count),
            "step_ratio": budget_violation_count / max(len(step_rows), 1),
            "max_excess_w": (
                None if not budget_excess_values else float(np.max(budget_excess_values))
            ),
        },
        "throughput_vs_EE_system_correlation": _correlation(
            step_ee_throughputs,
            step_ee_values,
        ),
    }
    return episode_rows, step_rows, summary


def _build_deltas(
    *,
    control_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
) -> dict[str, Any]:
    metrics = [
        "EE_system_aggregate_bps_per_w",
        "EE_system_step_mean_bps_per_w",
        "throughput_mean_user_step_bps",
        "throughput_p05_user_step_bps",
        "served_ratio",
        "outage_ratio",
        "handover_count",
        "r2_mean",
        "r3_mean",
        "scalar_reward_mean",
        "throughput_rescore_scalar_mean",
        "EE_rescore_scalar_mean",
    ]
    deltas: dict[str, Any] = {}
    for metric in metrics:
        control_value = control_summary.get(metric)
        candidate_value = candidate_summary.get(metric)
        deltas[metric] = {
            "control": control_value,
            "candidate": candidate_value,
            "delta": (
                None
                if control_value is None or candidate_value is None
                else float(candidate_value - control_value)
            ),
            "pct_delta": _pct_delta(control_value, candidate_value),
            "winner": _winner(control_value, candidate_value),
        }
    return deltas


def _handover_guardrail(deltas: dict[str, Any]) -> bool:
    control = deltas["handover_count"]["control"]
    candidate = deltas["handover_count"]["candidate"]
    pct_delta = deltas["handover_count"]["pct_delta"]
    if control is None or candidate is None:
        return True
    if abs(float(control)) < 1e-12:
        return float(candidate) <= 0.0
    return pct_delta is None or pct_delta <= 0.10


def _build_comparison(
    *,
    control_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
) -> dict[str, Any]:
    deltas = _build_deltas(
        control_summary=control_summary,
        candidate_summary=candidate_summary,
    )
    throughput_rescore_winner = deltas["throughput_rescore_scalar_mean"]["winner"]
    ee_rescore_winner = deltas["EE_rescore_scalar_mean"]["winner"]
    candidate_corr = candidate_summary["throughput_vs_EE_system_correlation"]
    ranking_separates = (
        throughput_rescore_winner != ee_rescore_winner
        or (
            candidate_corr["pearson"] is not None
            and abs(float(candidate_corr["pearson"])) < 0.99
        )
        or (
            candidate_corr["spearman"] is not None
            and abs(float(candidate_corr["spearman"])) < 0.99
        )
    )
    scalar_reward_only_success = (
        deltas["scalar_reward_mean"]["winner"] == "candidate"
        and deltas["EE_system_aggregate_bps_per_w"]["winner"] != "candidate"
    )
    throughput_guardrail = (
        deltas["throughput_p05_user_step_bps"]["pct_delta"] is None
        or deltas["throughput_p05_user_step_bps"]["pct_delta"] >= -0.10
    )
    served_guardrail = (
        deltas["served_ratio"]["delta"] is None
        or deltas["served_ratio"]["delta"] >= -0.05
    )
    handover_guardrail = _handover_guardrail(deltas)
    reward_hacking_flags = {
        "EE_gain_from_low_p05_throughput_collapse": (
            deltas["EE_system_aggregate_bps_per_w"]["winner"] == "candidate"
            and not throughput_guardrail
        ),
        "EE_gain_from_served_ratio_collapse": (
            deltas["EE_system_aggregate_bps_per_w"]["winner"] == "candidate"
            and not served_guardrail
        ),
        "EE_gain_from_handover_collapse": (
            deltas["EE_system_aggregate_bps_per_w"]["winner"] == "candidate"
            and not handover_guardrail
        ),
        "candidate_only_wins_scalar_reward": scalar_reward_only_success,
    }
    reward_hacking_flags["any"] = any(
        bool(value) for value in reward_hacking_flags.values()
    )

    acceptance_gate = {
        "denominator_varies_in_eval": bool(
            candidate_summary["denominator_varies_in_eval"]
        ),
        "selected_power_profile_not_single_point": (
            int(candidate_summary["selected_profile_distinct_count"]) > 1
        ),
        "active_power_not_single_point": (
            int(
                len(
                    candidate_summary["total_active_beam_power_w_distribution"][
                        "distinct"
                    ]
                )
            )
            > 1
        ),
        "EE_system_improves_over_control": (
            deltas["EE_system_aggregate_bps_per_w"]["winner"] == "candidate"
        ),
        "low_p05_throughput_guardrail": throughput_guardrail,
        "served_ratio_guardrail": served_guardrail,
        "handover_guardrail": handover_guardrail,
        "not_all_one_active_beam": (
            float(candidate_summary["one_active_beam_step_ratio"]) < 1.0
        ),
        "ranking_separates_or_rescore_changes": bool(ranking_separates),
        "no_scalar_reward_only_success_claim": not scalar_reward_only_success,
    }
    stop_conditions = {
        "denominator_remains_fixed_in_eval": not acceptance_gate[
            "denominator_varies_in_eval"
        ],
        "selected_profile_collapses_to_one_choice": not acceptance_gate[
            "selected_power_profile_not_single_point"
        ],
        "evaluated_policy_all_one_active_beam": not acceptance_gate[
            "not_all_one_active_beam"
        ],
        "EE_gain_from_throughput_or_service_or_handover_collapse": (
            reward_hacking_flags["EE_gain_from_low_p05_throughput_collapse"]
            or reward_hacking_flags["EE_gain_from_served_ratio_collapse"]
            or reward_hacking_flags["EE_gain_from_handover_collapse"]
        ),
        "candidate_only_wins_scalar_reward": scalar_reward_only_success,
    }
    all_acceptance_passed = all(bool(value) for value in acceptance_gate.values())
    if all_acceptance_passed and not reward_hacking_flags["any"]:
        decision = "PROMOTE"
    elif any(bool(value) for value in stop_conditions.values()):
        decision = "BLOCKED"
    else:
        decision = "NEEDS MORE EVIDENCE"

    return {
        "checkpoint_role": control_summary["checkpoint_role"],
        "deltas": deltas,
        "ranking_check": {
            "throughput_rescore_winner": throughput_rescore_winner,
            "EE_rescore_winner": ee_rescore_winner,
            "same_policy_throughput_rescore_vs_EE_rescore_ranking_changes": (
                throughput_rescore_winner != ee_rescore_winner
            ),
            "ranking_separates_or_correlation_changes": bool(ranking_separates),
            "control_throughput_EE_pearson": control_summary[
                "throughput_vs_EE_system_correlation"
            ]["pearson"],
            "control_throughput_EE_spearman": control_summary[
                "throughput_vs_EE_system_correlation"
            ]["spearman"],
            "candidate_throughput_EE_pearson": candidate_corr["pearson"],
            "candidate_throughput_EE_spearman": candidate_corr["spearman"],
        },
        "denominator_variability_result": {
            "control_denominator_varies_in_eval": bool(
                control_summary["denominator_varies_in_eval"]
            ),
            "candidate_denominator_varies_in_eval": bool(
                candidate_summary["denominator_varies_in_eval"]
            ),
            "candidate_one_active_beam_step_ratio": candidate_summary[
                "one_active_beam_step_ratio"
            ],
            "candidate_selected_profile_distinct_count": candidate_summary[
                "selected_profile_distinct_count"
            ],
            "candidate_selected_power_profile_distribution": candidate_summary[
                "selected_power_profile_distribution"
            ],
            "candidate_active_beam_count_distribution": candidate_summary[
                "active_beam_count_distribution"
            ],
            "candidate_total_active_beam_power_w_distribution": candidate_summary[
                "total_active_beam_power_w_distribution"
            ],
        },
        "acceptance_gate": acceptance_gate,
        "stop_conditions": stop_conditions,
        "reward_hacking_flags": reward_hacking_flags,
        "phase_03c_c_decision": decision,
    }


def _review_lines(summary: dict[str, Any]) -> list[str]:
    primary = summary["primary_comparison"]
    protocol = summary["protocol"]
    variability = primary["denominator_variability_result"]
    ranking = primary["ranking_check"]
    return [
        "# Phase 03C-C Power-MDP Bounded Paired Pilot Review",
        "",
        "Bounded paired pilot only. No long training, Catfish, multi-Catfish, "
        "frozen baseline mutation, HOBS optimizer claim, or scalar-reward-only "
        "success claim was performed.",
        "",
        "## Protocol",
        "",
        f"- control run: `{summary['inputs']['control_run_dir']}`",
        f"- candidate run: `{summary['inputs']['candidate_run_dir']}`",
        f"- evaluation seeds: `{summary['inputs']['evaluation_seed_set']}`",
        f"- primary checkpoint role: `{primary['checkpoint_role']}`",
        f"- control power profile: `{protocol['control_power_profile']}`",
        f"- candidate power selector: `{protocol['candidate_power_profile']}`",
        f"- shared budget: `{protocol['total_power_budget_w']}` W",
        "",
        "## Metrics",
        "",
        f"- EE_system aggregate delta: "
        f"`{primary['deltas']['EE_system_aggregate_bps_per_w']['delta']}`",
        f"- throughput mean delta: "
        f"`{primary['deltas']['throughput_mean_user_step_bps']['delta']}`",
        f"- throughput p05 delta: "
        f"`{primary['deltas']['throughput_p05_user_step_bps']['delta']}`",
        f"- served ratio delta: `{primary['deltas']['served_ratio']['delta']}`",
        f"- handover count delta: `{primary['deltas']['handover_count']['delta']}`",
        "",
        "## Denominator",
        "",
        f"- candidate denominator varies in eval: "
        f"`{variability['candidate_denominator_varies_in_eval']}`",
        f"- candidate selected profile distinct count: "
        f"`{variability['candidate_selected_profile_distinct_count']}`",
        f"- candidate one-active-beam step ratio: "
        f"`{variability['candidate_one_active_beam_step_ratio']}`",
        "",
        "## Ranking",
        "",
        f"- throughput rescore winner: `{ranking['throughput_rescore_winner']}`",
        f"- EE rescore winner: `{ranking['EE_rescore_winner']}`",
        f"- same-policy rescore ranking changed: "
        f"`{ranking['same_policy_throughput_rescore_vs_EE_rescore_ranking_changes']}`",
        f"- candidate throughput-vs-EE Pearson: "
        f"`{ranking['candidate_throughput_EE_pearson']}`",
        f"- candidate throughput-vs-EE Spearman: "
        f"`{ranking['candidate_throughput_EE_spearman']}`",
        "",
        "## Decision",
        "",
        f"- Phase 03C-C decision: `{summary['phase_03c_c_decision']}`",
        "- No EE-MODQN effectiveness claim is allowed unless the gate is `PROMOTE`.",
        "",
        "## Forbidden Claims",
        "",
        "- Do not claim EE-MODQN effectiveness unless this gate promotes.",
        "- Do not claim HOBS optimizer behavior.",
        "- Do not claim Catfish, multi-Catfish, or full paper-faithful reproduction.",
        "- Do not treat per-user EE credit as system EE.",
        "- Do not use scalar reward alone as success evidence.",
    ]


def export_phase03c_c_power_mdp_paired_validation(
    *,
    control_run_dir: str | Path,
    candidate_run_dir: str | Path,
    output_dir: str | Path,
    evaluation_seed_set: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Evaluate paired Phase 03C-C power-MDP runs and export artifacts."""
    control_dir = Path(control_run_dir)
    candidate_dir = Path(candidate_run_dir)
    out_dir = Path(output_dir)
    control_metadata = read_run_metadata(control_dir / "run_metadata.json")
    candidate_metadata = read_run_metadata(candidate_dir / "run_metadata.json")
    control_cfg = control_metadata.resolved_config_snapshot
    candidate_cfg = candidate_metadata.resolved_config_snapshot
    control_trainer_cfg = build_trainer_config(control_cfg)
    candidate_trainer_cfg = build_trainer_config(candidate_cfg)

    control_eval_seeds = tuple(control_metadata.seeds.evaluation_seed_set)
    candidate_eval_seeds = tuple(candidate_metadata.seeds.evaluation_seed_set)
    eval_seeds = tuple(
        int(seed)
        for seed in (
            evaluation_seed_set
            if evaluation_seed_set is not None
            else control_eval_seeds
        )
    )
    if not eval_seeds:
        raise ValueError("Phase 03C-C paired validation requires evaluation seeds.")
    if control_eval_seeds != candidate_eval_seeds:
        raise ValueError(
            "Control and candidate run metadata must use the same evaluation seed set."
        )

    control_power = _power_surface_value(control_cfg)
    candidate_power = _power_surface_value(candidate_cfg)
    if control_power.get("total_power_budget_w") != candidate_power.get(
        "total_power_budget_w"
    ):
        raise ValueError("Control and candidate must use the same power budget.")
    if control_metadata.checkpoint_rule.to_dict() != candidate_metadata.checkpoint_rule.to_dict():
        raise ValueError("Control and candidate must use the same checkpoint rule.")

    all_episode_rows: list[dict[str, Any]] = []
    all_step_rows: list[dict[str, Any]] = []
    checkpoint_summaries: list[dict[str, Any]] = []
    for method_label, run_dir, metadata in (
        ("control", control_dir, control_metadata),
        ("candidate", candidate_dir, candidate_metadata),
    ):
        paths = _checkpoint_paths(metadata)
        for checkpoint_role, _field in _CHECKPOINT_ROLES:
            checkpoint_path = paths.get(checkpoint_role)
            if checkpoint_path is None:
                continue
            episode_rows, step_rows, checkpoint_summary = _evaluate_checkpoint(
                run_dir=run_dir,
                method_label=method_label,
                checkpoint_role=checkpoint_role,
                checkpoint_path=checkpoint_path,
                evaluation_seed_set=eval_seeds,
            )
            all_episode_rows.extend(episode_rows)
            all_step_rows.extend(step_rows)
            checkpoint_summaries.append(checkpoint_summary)

    summary_by_key = {
        (row["method"], row["checkpoint_role"]): row
        for row in checkpoint_summaries
    }
    comparisons: list[dict[str, Any]] = []
    for checkpoint_role, _field in _CHECKPOINT_ROLES:
        control_summary = summary_by_key.get(("control", checkpoint_role))
        candidate_summary = summary_by_key.get(("candidate", checkpoint_role))
        if control_summary is None or candidate_summary is None:
            continue
        comparisons.append(
            _build_comparison(
                control_summary=control_summary,
                candidate_summary=candidate_summary,
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
            "candidate_run_dir": str(candidate_dir),
            "output_dir": str(out_dir),
            "evaluation_seed_set": list(eval_seeds),
        },
        "protocol": {
            "phase": "phase-03c-c",
            "paired": True,
            "bounded_training": True,
            "catfish": "disabled",
            "multi_catfish": "disabled",
            "frozen_baseline_mutation": "forbidden/not-performed",
            "control_r1": control_trainer_cfg.r1_reward_mode,
            "candidate_r1": candidate_trainer_cfg.r1_reward_mode,
            "control_power_profile": control_power.get("power_codebook_profile"),
            "candidate_power_profile": candidate_power.get("power_codebook_profile"),
            "total_power_budget_w": control_power.get("total_power_budget_w"),
            "same_evaluation_seeds": control_eval_seeds == candidate_eval_seeds,
            "same_checkpoint_rule": (
                control_metadata.checkpoint_rule.to_dict()
                == candidate_metadata.checkpoint_rule.to_dict()
            ),
            "system_EE_primary": True,
            "per_user_EE_credit_is_system_EE": False,
            "scalar_reward_success_basis": False,
            "controller_claim": "new-extension / HOBS-inspired; not HOBS optimizer",
        },
        "checkpoint_summaries": checkpoint_summaries,
        "comparisons": comparisons,
        "primary_comparison": primary_comparison,
        "phase_03c_c_decision": primary_comparison["phase_03c_c_decision"],
        "remaining_blockers": [
            "Pilot budget is intentionally bounded and does not replace a larger approved run.",
            "The runtime selector is a new-extension / HOBS-inspired controller, not a HOBS optimizer.",
            "Catfish, multi-Catfish, and final Catfish-EE-MODQN remain out of scope.",
        ],
        "forbidden_claims_still_active": [
            "Do not claim EE-MODQN effectiveness unless the Phase 03C-C gate promotes.",
            "Do not claim full paper-faithful reproduction.",
            "Do not claim Catfish, multi-Catfish, or final Catfish-EE-MODQN.",
            "Do not treat per-user EE credit as system EE.",
            "Do not use scalar reward alone as success evidence.",
            "Do not label the Phase 03C-C selector as a HOBS optimizer.",
        ],
    }

    episode_csv = _write_csv(
        out_dir / "phase03c_c_power_mdp_episode_metrics.csv",
        all_episode_rows,
        fieldnames=list(all_episode_rows[0].keys()),
    )
    step_csv = _write_csv(
        out_dir / "phase03c_c_power_mdp_step_metrics.csv",
        all_step_rows,
        fieldnames=list(all_step_rows[0].keys()),
    )
    summary_csv = _write_csv(
        out_dir / "phase03c_c_power_mdp_checkpoint_summary.csv",
        [
            {
                "method": row["method"],
                "checkpoint_role": row["checkpoint_role"],
                "EE_system_aggregate_bps_per_w": row[
                    "EE_system_aggregate_bps_per_w"
                ],
                "EE_system_step_mean_bps_per_w": row[
                    "EE_system_step_mean_bps_per_w"
                ],
                "throughput_mean_user_step_bps": row[
                    "throughput_mean_user_step_bps"
                ],
                "throughput_p05_user_step_bps": row[
                    "throughput_p05_user_step_bps"
                ],
                "served_ratio": row["served_ratio"],
                "outage_ratio": row["outage_ratio"],
                "handover_count": row["handover_count"],
                "r2_mean": row["r2_mean"],
                "r3_mean": row["r3_mean"],
                "denominator_varies_in_eval": row["denominator_varies_in_eval"],
                "one_active_beam_step_ratio": row["one_active_beam_step_ratio"],
                "selected_profile_distinct_count": row[
                    "selected_profile_distinct_count"
                ],
                "budget_violation_step_count": row["budget_violations"][
                    "step_count"
                ],
                "throughput_EE_pearson": row[
                    "throughput_vs_EE_system_correlation"
                ]["pearson"],
                "throughput_EE_spearman": row[
                    "throughput_vs_EE_system_correlation"
                ]["spearman"],
                "scalar_reward_mean": row["scalar_reward_mean"],
            }
            for row in checkpoint_summaries
        ],
        fieldnames=[
            "method",
            "checkpoint_role",
            "EE_system_aggregate_bps_per_w",
            "EE_system_step_mean_bps_per_w",
            "throughput_mean_user_step_bps",
            "throughput_p05_user_step_bps",
            "served_ratio",
            "outage_ratio",
            "handover_count",
            "r2_mean",
            "r3_mean",
            "denominator_varies_in_eval",
            "one_active_beam_step_ratio",
            "selected_profile_distinct_count",
            "budget_violation_step_count",
            "throughput_EE_pearson",
            "throughput_EE_spearman",
            "scalar_reward_mean",
        ],
    )
    summary_path = write_json(out_dir / "phase03c_c_power_mdp_summary.json", summary)
    review_path = out_dir / "review.md"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text("\n".join(_review_lines(summary)) + "\n")

    return {
        "phase03c_c_power_mdp_summary": summary_path,
        "phase03c_c_power_mdp_episode_metrics": episode_csv,
        "phase03c_c_power_mdp_step_metrics": step_csv,
        "phase03c_c_power_mdp_checkpoint_summary": summary_csv,
        "review_md": review_path,
        "summary": summary,
    }


__all__ = ["export_phase03c_c_power_mdp_paired_validation"]
