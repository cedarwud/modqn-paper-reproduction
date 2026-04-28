"""Phase 03C-B static/counterfactual power-MDP audit.

This module is audit-only. It fixes handover trajectories, then replays the
same beam decisions under multiple power semantics to check whether system EE
can be separated from throughput by an explicit power decision. It does not
train, does not invoke Catfish, and does not mutate frozen baseline artifacts.
"""

from __future__ import annotations

import copy
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..algorithms.modqn import MODQNTrainer
from ..artifacts import read_run_metadata
from ..config_loader import (
    build_environment,
    build_trainer_config,
    get_seeds,
    load_training_yaml,
)
from ..env.step import (
    HOBS_POWER_SURFACE_ACTIVE_LOAD_CONCAVE,
    HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK,
    HOBS_POWER_SURFACE_STATIC_CONFIG,
)
from ._common import write_json
from .phase03a_diagnostics import distribution, select_counterfactual_actions


DEFAULT_PHASE03B_EE_RUN_DIR = (
    "artifacts/ee-modqn-phase-03b-ee-objective-geometry-pilot"
)

POWER_SEMANTICS = (
    "fixed-2w",
    "phase-02b-proxy",
    "fixed-low",
    "fixed-mid",
    "fixed-high",
    "load-concave",
    "qos-tail-boost",
    "budget-trim",
)

COUNTERFACTUAL_POLICIES = (
    "hold-current",
    "random-valid",
    "spread-valid",
)

_POLICY_SEED_OFFSETS = {
    "hold-current": 1009,
    "random-valid": 2003,
    "spread-valid": 4001,
}


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


def _unique_float_values(values: list[float], *, places: int = 12) -> list[float]:
    return sorted({round(float(value), places) for value in values})


def _mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return None
    return float(np.mean(np.asarray(clean, dtype=np.float64)))


def _pct_delta(reference: float | None, value: float | None) -> float | None:
    if reference is None or value is None or abs(reference) < 1e-12:
        return None
    return float((value - reference) / abs(reference))


def _power_value_template(cfg: dict[str, Any]) -> dict[str, Any]:
    value = (
        cfg.get("resolved_assumptions", {})
        .get("hobs_power_surface", {})
        .get("value", {})
    )
    if not isinstance(value, dict):
        value = {}
    template = {
        "inactive_beam_policy": value.get("inactive_beam_policy", "zero-w"),
        "active_base_power_w": float(value.get("active_base_power_w", 0.25)),
        "load_scale_power_w": float(value.get("load_scale_power_w", 0.35)),
        "load_exponent": float(value.get("load_exponent", 0.5)),
        "max_power_w": value.get("max_power_w", 2.0),
        "power_codebook_profile": value.get("power_codebook_profile", "fixed-mid"),
        "power_codebook_levels_w": list(value.get("power_codebook_levels_w", [0.5, 1.0, 2.0])),
        "total_power_budget_w": value.get("total_power_budget_w", 8.0),
        "units": "linear-w",
    }
    if template["max_power_w"] is not None:
        template["max_power_w"] = float(template["max_power_w"])
    if template["total_power_budget_w"] is not None:
        template["total_power_budget_w"] = float(template["total_power_budget_w"])
    return template


def _cfg_for_power_semantics(
    base_cfg: dict[str, Any],
    power_semantics: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if power_semantics not in POWER_SEMANTICS:
        raise ValueError(
            f"Unsupported Phase 03C-B power semantics {power_semantics!r}; "
            f"supported values are {POWER_SEMANTICS!r}."
        )

    cfg = copy.deepcopy(base_cfg)
    resolved = cfg.setdefault("resolved_assumptions", {})
    block = resolved.setdefault("hobs_power_surface", {})
    block["assumption_id"] = "ASSUME-MODQN-REP-025"
    value = _power_value_template(base_cfg)

    metadata = {
        "power_semantics": power_semantics,
        "category": "comparator",
        "phase_02b_proxy": False,
        "phase_03c_b_controller": False,
        "new_extension": False,
        "hobs_inspired": False,
        "hobs_optimizer": False,
        "profile": None,
        "budget_w": value.get("total_power_budget_w"),
    }

    if power_semantics == "fixed-2w":
        value.update(
            {
                "mode": HOBS_POWER_SURFACE_STATIC_CONFIG,
                "inactive_beam_policy": "excluded-from-active-beams",
                "max_power_w": 2.0,
                "provenance": "Fixed 2W comparator; not allocated power control.",
            }
        )
        metadata["label"] = "fixed 2W comparator"
    elif power_semantics == "phase-02b-proxy":
        value.update(
            {
                "mode": HOBS_POWER_SURFACE_ACTIVE_LOAD_CONCAVE,
                "inactive_beam_policy": "zero-w",
                "provenance": (
                    "Phase 02B synthesized active-load-concave proxy; not a "
                    "paper-backed HOBS optimizer."
                ),
            }
        )
        metadata.update(
            {
                "label": "Phase 02B proxy",
                "phase_02b_proxy": True,
            }
        )
    else:
        value.update(
            {
                "mode": HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK,
                "inactive_beam_policy": "zero-w",
                "power_codebook_profile": power_semantics,
                "provenance": (
                    "Phase 03C-B new-extension / HOBS-inspired centralized "
                    "discrete codebook controller; not a HOBS optimizer."
                ),
            }
        )
        metadata.update(
            {
                "label": f"Phase 03C-B codebook {power_semantics}",
                "category": "phase-03c-b-power-codebook",
                "phase_03c_b_controller": True,
                "new_extension": True,
                "hobs_inspired": True,
                "profile": power_semantics,
            }
        )

    block["value"] = value
    return cfg, metadata


def _checkpoint_path(metadata: Any) -> tuple[str, Path] | None:
    files = metadata.checkpoint_files
    if files.secondary_best_eval is not None:
        return "best-eval", Path(files.secondary_best_eval)
    if files.primary_final is not None:
        return "final", Path(files.primary_final)
    return None


def _rollout_learned_trajectory(
    *,
    run_dir: Path,
    evaluation_seed_set: tuple[int, ...],
    max_steps: int | None,
) -> tuple[dict[str, dict[int, list[np.ndarray]]], dict[str, Any]]:
    metadata = read_run_metadata(run_dir / "run_metadata.json")
    checkpoint = _checkpoint_path(metadata)
    if checkpoint is None:
        raise ValueError(f"No checkpoint is available in learned run {run_dir}.")
    checkpoint_role, checkpoint_path = checkpoint
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
    policy_label = f"phase03b-ee-{checkpoint_role}"

    trajectories: dict[str, dict[int, list[np.ndarray]]] = {
        policy_label: defaultdict(list)
    }
    for eval_seed in evaluation_seed_set:
        env_seed_seq, mobility_seed_seq = np.random.SeedSequence(eval_seed).spawn(2)
        env_rng = np.random.default_rng(env_seed_seq)
        mobility_rng = np.random.default_rng(mobility_seed_seq)
        states, masks, _diag = trainer.env.reset(env_rng, mobility_rng)
        encoded = trainer.encode_states(states)
        steps_seen = 0
        while True:
            if max_steps is not None and steps_seen >= max_steps:
                break
            actions = trainer.select_actions(encoded, masks, eps=0.0)
            result = trainer.env.step(actions, env_rng)
            trajectories[policy_label][int(eval_seed)].append(actions.copy())
            steps_seen += 1
            if result.done:
                break
            encoded = trainer.encode_states(result.user_states)
            masks = result.action_masks

    return trajectories, {
        "available": True,
        "run_dir": str(run_dir),
        "policy_label": policy_label,
        "checkpoint_role": checkpoint_role,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_kind": str(payload.get("checkpoint_kind")),
        "checkpoint_episode": int(payload.get("episode")),
    }


def _rollout_counterfactual_trajectories(
    *,
    cfg: dict[str, Any],
    evaluation_seed_set: tuple[int, ...],
    max_steps: int | None,
    policies: tuple[str, ...],
) -> dict[str, dict[int, list[np.ndarray]]]:
    trajectories: dict[str, dict[int, list[np.ndarray]]] = {}
    for policy in policies:
        if policy not in COUNTERFACTUAL_POLICIES:
            raise ValueError(
                f"Unsupported Phase 03C-B trajectory policy {policy!r}; "
                f"supported values are {COUNTERFACTUAL_POLICIES!r}."
            )
        env = build_environment(cfg)
        policy_rows: dict[int, list[np.ndarray]] = defaultdict(list)
        for eval_seed in evaluation_seed_set:
            env_seed_seq, mobility_seed_seq = np.random.SeedSequence(eval_seed).spawn(2)
            env_rng = np.random.default_rng(env_seed_seq)
            mobility_rng = np.random.default_rng(mobility_seed_seq)
            policy_rng = np.random.default_rng(
                int(eval_seed) + _POLICY_SEED_OFFSETS[policy]
            )
            _states, masks, _diag = env.reset(env_rng, mobility_rng)
            steps_seen = 0
            while True:
                if max_steps is not None and steps_seen >= max_steps:
                    break
                current_assignments = env.current_assignments()
                internal_policy = (
                    "spread-valid-heuristic"
                    if policy == "spread-valid"
                    else policy
                )
                actions, _diagnostics = select_counterfactual_actions(
                    internal_policy,
                    current_assignments=current_assignments,
                    masks=masks,
                    rng=policy_rng,
                )
                result = env.step(actions, env_rng)
                policy_rows[int(eval_seed)].append(actions.copy())
                steps_seen += 1
                if result.done:
                    break
                masks = result.action_masks
        trajectories[policy] = policy_rows
    return trajectories


def _step_metric_row(
    *,
    policy_label: str,
    power_metadata: dict[str, Any],
    evaluation_seed: int,
    result: Any,
    budget_w: float | None,
) -> tuple[dict[str, Any], list[float]]:
    active_mask = result.active_beam_mask.astype(bool, copy=False)
    beam_power = result.beam_transmit_power_w.astype(np.float64, copy=False)
    total_active_power = float(np.sum(beam_power[active_mask], dtype=np.float64))
    user_throughputs = [
        float(reward.r1_throughput)
        for reward in result.rewards
    ]
    throughput_sum = float(np.sum(user_throughputs, dtype=np.float64))
    ee_system = None if total_active_power <= 0.0 else throughput_sum / total_active_power
    served = sum(1 for value in user_throughputs if value > 0.0)
    handovers = sum(1 for reward in result.rewards if reward.r2_handover < 0.0)
    budget_excess = (
        None
        if budget_w is None
        else max(0.0, total_active_power - float(budget_w))
    )
    row = {
        "trajectory_policy": policy_label,
        "power_semantics": power_metadata["power_semantics"],
        "power_category": power_metadata["category"],
        "power_profile": power_metadata["profile"],
        "phase_02b_proxy": bool(power_metadata["phase_02b_proxy"]),
        "phase_03c_b_controller": bool(power_metadata["phase_03c_b_controller"]),
        "new_extension": bool(power_metadata["new_extension"]),
        "hobs_inspired": bool(power_metadata["hobs_inspired"]),
        "hobs_optimizer": bool(power_metadata["hobs_optimizer"]),
        "evaluation_seed": int(evaluation_seed),
        "step_index": int(result.step_index),
        "active_beam_count": int(np.count_nonzero(active_mask)),
        "active_beam_mask": " ".join("1" if value else "0" for value in active_mask),
        "beam_transmit_power_w": _format_vector(beam_power),
        "total_active_beam_power_w": total_active_power,
        "sum_user_throughput_bps": throughput_sum,
        "throughput_mean_user_step_bps": throughput_sum / max(len(user_throughputs), 1),
        "throughput_p05_user_step_bps": float(np.percentile(user_throughputs, 5)),
        "ee_system_bps_per_w": ee_system,
        "served_count": int(served),
        "outage_count": int(len(user_throughputs) - served),
        "served_ratio": served / max(len(user_throughputs), 1),
        "outage_ratio": 1.0 - (served / max(len(user_throughputs), 1)),
        "handover_count": int(handovers),
        "total_power_budget_w": budget_w,
        "budget_violation": bool(budget_excess is not None and budget_excess > 1e-12),
        "budget_excess_w": budget_excess,
    }
    return row, user_throughputs


def _replay_trajectories(
    *,
    base_cfg: dict[str, Any],
    trajectories: dict[str, dict[int, list[np.ndarray]]],
    power_semantics: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[float]]]:
    rows: list[dict[str, Any]] = []
    user_throughputs_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    for semantics in power_semantics:
        cfg, metadata = _cfg_for_power_semantics(base_cfg, semantics)
        env = build_environment(cfg)
        budget_w = metadata.get("budget_w")
        for policy_label, by_seed in trajectories.items():
            for eval_seed, action_rows in by_seed.items():
                env_seed_seq, mobility_seed_seq = np.random.SeedSequence(eval_seed).spawn(2)
                env_rng = np.random.default_rng(env_seed_seq)
                mobility_rng = np.random.default_rng(mobility_seed_seq)
                env.reset(env_rng, mobility_rng)
                for actions in action_rows:
                    result = env.step(actions, env_rng)
                    row, user_throughputs = _step_metric_row(
                        policy_label=policy_label,
                        power_metadata=metadata,
                        evaluation_seed=eval_seed,
                        result=result,
                        budget_w=budget_w,
                    )
                    rows.append(row)
                    user_throughputs_by_key[(policy_label, semantics)].extend(
                        user_throughputs
                    )
                    if result.done:
                        break
    return rows, user_throughputs_by_key


def _summarize_group(
    *,
    rows: list[dict[str, Any]],
    user_throughputs: list[float],
) -> dict[str, Any]:
    first = rows[0]
    active_counts = [int(row["active_beam_count"]) for row in rows]
    total_powers = [float(row["total_active_beam_power_w"]) for row in rows]
    beam_vectors = {str(row["beam_transmit_power_w"]) for row in rows}
    step_throughput = [float(row["sum_user_throughput_bps"]) for row in rows]
    step_pairs = [
        (float(row["sum_user_throughput_bps"]), float(row["ee_system_bps_per_w"]))
        for row in rows
        if row["ee_system_bps_per_w"] is not None
    ]
    step_ee = [ee for _throughput, ee in step_pairs]
    total_throughput = float(np.sum(step_throughput, dtype=np.float64))
    total_power = float(np.sum(total_powers, dtype=np.float64))
    budget_excess = [
        float(row["budget_excess_w"])
        for row in rows
        if row["budget_excess_w"] is not None
    ]
    served_count = sum(int(row["served_count"]) for row in rows)
    outage_count = sum(int(row["outage_count"]) for row in rows)
    handovers = sum(int(row["handover_count"]) for row in rows)
    denominator_distinct = _unique_float_values(total_powers)

    return {
        "trajectory_policy": first["trajectory_policy"],
        "power_semantics": first["power_semantics"],
        "power_category": first["power_category"],
        "power_profile": first["power_profile"],
        "phase_02b_proxy": bool(first["phase_02b_proxy"]),
        "phase_03c_b_controller": bool(first["phase_03c_b_controller"]),
        "new_extension": bool(first["new_extension"]),
        "hobs_inspired": bool(first["hobs_inspired"]),
        "hobs_optimizer": bool(first["hobs_optimizer"]),
        "step_count": len(rows),
        "evaluation_seeds": sorted({int(row["evaluation_seed"]) for row in rows}),
        "EE_system_aggregate_bps_per_w": (
            None if total_power <= 0.0 else total_throughput / total_power
        ),
        "EE_system_step_mean_bps_per_w": _mean(step_ee),
        "active_beam_count_distribution": distribution(active_counts),
        "total_active_beam_power_w_distribution": distribution(total_powers),
        "beam_power_vector_distinct_count": len(beam_vectors),
        "denominator_varies_in_eval": len(denominator_distinct) > 1,
        "one_active_beam_step_ratio": float(
            np.mean([count == 1 for count in active_counts])
        ),
        "throughput_mean_user_step_bps": _mean(user_throughputs),
        "throughput_p05_user_step_bps": (
            None
            if not user_throughputs
            else float(np.percentile(user_throughputs, 5))
        ),
        "served_ratio": served_count / max(served_count + outage_count, 1),
        "outage_ratio": outage_count / max(served_count + outage_count, 1),
        "handover_count": handovers,
        "budget_violations": {
            "budget_w": first["total_power_budget_w"],
            "step_count": int(sum(bool(row["budget_violation"]) for row in rows)),
            "step_ratio": float(np.mean([bool(row["budget_violation"]) for row in rows])),
            "max_excess_w": None if not budget_excess else float(np.max(budget_excess)),
        },
        "throughput_vs_EE_system_correlation": _correlation(
            [throughput for throughput, _ee in step_pairs],
            step_ee,
        ),
        "total_throughput_bps": total_throughput,
        "total_active_beam_power_sum_w": total_power,
    }


def _summarize_all(
    *,
    rows: list[dict[str, Any]],
    user_throughputs_by_key: dict[tuple[str, str], list[float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(str(row["trajectory_policy"]), str(row["power_semantics"]))].append(row)

    summaries = [
        _summarize_group(
            rows=group_rows,
            user_throughputs=user_throughputs_by_key[(policy, semantics)],
        )
        for (policy, semantics), group_rows in sorted(by_key.items())
    ]

    policy_ranking_checks: list[dict[str, Any]] = []
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        by_policy[str(summary["trajectory_policy"])].append(summary)
    for policy, policy_summaries in sorted(by_policy.items()):
        throughput_ranking = [
            row["power_semantics"]
            for row in sorted(
                policy_summaries,
                key=lambda row: (
                    row["throughput_mean_user_step_bps"]
                    if row["throughput_mean_user_step_bps"] is not None
                    else -math.inf
                ),
                reverse=True,
            )
        ]
        ee_ranking = [
            row["power_semantics"]
            for row in sorted(
                policy_summaries,
                key=lambda row: (
                    row["EE_system_aggregate_bps_per_w"]
                    if row["EE_system_aggregate_bps_per_w"] is not None
                    else -math.inf
                ),
                reverse=True,
            )
        ]
        fixed = next(
            (
                row for row in policy_summaries
                if row["power_semantics"] == "fixed-2w"
            ),
            None,
        )
        candidate_deltas = {}
        for row in policy_summaries:
            candidate_deltas[row["power_semantics"]] = {
                "throughput_pct_delta_vs_fixed_2w": _pct_delta(
                    None if fixed is None else fixed["throughput_mean_user_step_bps"],
                    row["throughput_mean_user_step_bps"],
                ),
                "EE_system_pct_delta_vs_fixed_2w": _pct_delta(
                    None if fixed is None else fixed["EE_system_aggregate_bps_per_w"],
                    row["EE_system_aggregate_bps_per_w"],
                ),
            }
        policy_ranking_checks.append(
            {
                "trajectory_policy": policy,
                "throughput_rescore_ranking": throughput_ranking,
                "EE_rescore_ranking": ee_ranking,
                "throughput_rescore_winner": throughput_ranking[0],
                "EE_rescore_winner": ee_ranking[0],
                "same_policy_throughput_rescore_vs_EE_rescore_ranking_changes": (
                    throughput_ranking != ee_ranking
                ),
                "same_policy_throughput_rescore_vs_EE_rescore_top_changes": (
                    throughput_ranking[0] != ee_ranking[0]
                ),
                "candidate_deltas_vs_fixed_2w": candidate_deltas,
            }
        )
    return summaries, policy_ranking_checks


def _build_decision(
    *,
    candidate_summaries: list[dict[str, Any]],
    ranking_checks: list[dict[str, Any]],
) -> dict[str, Any]:
    codebook = [
        row for row in candidate_summaries
        if row["phase_03c_b_controller"]
    ]
    denominator_changed_by_power_decision = False
    for policy in sorted({row["trajectory_policy"] for row in candidate_summaries}):
        values = [
            row["total_active_beam_power_sum_w"]
            for row in candidate_summaries
            if row["trajectory_policy"] == policy
        ]
        if len(_unique_float_values(values, places=9)) > 1:
            denominator_changed_by_power_decision = True
            break

    has_budget_respecting_codebook = any(
        row["budget_violations"]["step_count"] == 0 for row in codebook
    )
    ranking_separates = any(
        bool(row["same_policy_throughput_rescore_vs_EE_rescore_ranking_changes"])
        for row in ranking_checks
    )
    fixed_denominator_caught = any(
        row["power_semantics"] in {"fixed-2w", "fixed-low", "fixed-mid", "fixed-high"}
        and not bool(row["denominator_varies_in_eval"])
        for row in candidate_summaries
    )

    if (
        denominator_changed_by_power_decision
        and ranking_separates
        and has_budget_respecting_codebook
    ):
        phase_decision = "PASS to bounded paired pilot"
    elif not candidate_summaries:
        phase_decision = "BLOCKED"
    else:
        phase_decision = "NEEDS MORE EVIDENCE"

    return {
        "phase_03c_b_decision": phase_decision,
        "denominator_changed_by_power_decision": denominator_changed_by_power_decision,
        "ranking_separates_under_same_policy_rescore": ranking_separates,
        "has_budget_respecting_codebook_candidate": has_budget_respecting_codebook,
        "fixed_denominator_audit_catches_fixed_denominator": fixed_denominator_caught,
        "allowed_next_step": (
            "bounded paired pilot only; no EE-MODQN effectiveness claim"
            if phase_decision == "PASS to bounded paired pilot"
            else "do not start paired pilot without more evidence or design revision"
        ),
    }


def _review_lines(summary: dict[str, Any]) -> list[str]:
    decision = summary["phase_03c_b_decision"]
    proof = summary["denominator_variability_result"]
    ranking = summary["ranking_separation_result"]
    return [
        "# Phase 03C-B Power-MDP Audit Review",
        "",
        "Static/counterfactual audit only. No training, Catfish, multi-Catfish, "
        "frozen baseline mutation, or HOBS optimizer claim was performed.",
        "",
        "## Protocol",
        "",
        f"- config: `{summary['inputs']['config_path']}`",
        f"- artifact namespace: `{summary['inputs']['output_dir']}`",
        f"- evaluation seeds: `{summary['inputs']['evaluation_seed_set']}`",
        f"- trajectory policies: `{summary['protocol']['trajectory_policies']}`",
        f"- power semantics: `{summary['protocol']['power_semantics']}`",
        "",
        "## Denominator",
        "",
        f"- changed by power decision: "
        f"`{proof['denominator_changed_by_power_decision']}`",
        f"- fixed denominator caught: "
        f"`{proof['fixed_denominator_audit_catches_fixed_denominator']}`",
        "",
        "## Ranking Separation",
        "",
        f"- same-policy throughput-vs-EE ranking separates: "
        f"`{ranking['ranking_separates_under_same_policy_rescore']}`",
        f"- policies with top-rank changes: "
        f"`{ranking['policies_with_top_rank_change']}`",
        "",
        "## Decision",
        "",
        f"- Phase 03C-B decision: `{decision['phase_03c_b_decision']}`",
        f"- allowed next step: {decision['allowed_next_step']}",
        "",
        "## Forbidden Claims",
        "",
        "- Do not claim EE-MODQN effectiveness.",
        "- Do not treat per-user EE credit as system EE.",
        "- Do not use scalar reward as the success basis.",
        "- Do not call the Phase 03C-B controller a HOBS optimizer.",
        "- Do not claim Catfish, multi-Catfish, or full paper-faithful reproduction.",
    ]


def export_phase03c_b_power_mdp_audit(
    config_path: str | Path = "configs/ee-modqn-phase-03c-b-power-mdp-audit.resolved.yaml",
    output_dir: str | Path = "artifacts/ee-modqn-phase-03c-b-power-mdp-audit",
    *,
    learned_run_dir: str | Path | None = DEFAULT_PHASE03B_EE_RUN_DIR,
    include_learned: bool = True,
    evaluation_seed_set: tuple[int, ...] | None = None,
    max_steps: int | None = None,
    policies: tuple[str, ...] = COUNTERFACTUAL_POLICIES,
    power_semantics: tuple[str, ...] = POWER_SEMANTICS,
) -> dict[str, Any]:
    """Export the Phase 03C-B static/counterfactual power-codebook audit."""
    if max_steps is not None and max_steps < 1:
        raise ValueError(f"max_steps must be >= 1 when provided, got {max_steps!r}")
    if not power_semantics:
        raise ValueError("At least one power semantics candidate is required.")

    cfg = load_training_yaml(config_path)
    seeds = get_seeds(cfg)
    eval_seeds = tuple(
        int(seed)
        for seed in (
            evaluation_seed_set
            if evaluation_seed_set is not None
            else tuple(seeds["evaluation_seed_set"])
        )
    )
    if not eval_seeds:
        raise ValueError("Phase 03C-B audit requires evaluation seeds.")

    trajectories: dict[str, dict[int, list[np.ndarray]]] = {}
    learned_trajectory = {"available": False, "reason": "not-requested"}
    if include_learned and learned_run_dir is not None:
        run_dir = Path(learned_run_dir)
        if run_dir.exists() and (run_dir / "run_metadata.json").exists():
            learned_paths, learned_trajectory = _rollout_learned_trajectory(
                run_dir=run_dir,
                evaluation_seed_set=eval_seeds,
                max_steps=max_steps,
            )
            trajectories.update(learned_paths)
        else:
            learned_trajectory = {
                "available": False,
                "run_dir": str(run_dir),
                "reason": "run directory or run_metadata.json unavailable",
            }

    trajectories.update(
        _rollout_counterfactual_trajectories(
            cfg=cfg,
            evaluation_seed_set=eval_seeds,
            max_steps=max_steps,
            policies=policies,
        )
    )

    step_rows, user_throughputs_by_key = _replay_trajectories(
        base_cfg=cfg,
        trajectories=trajectories,
        power_semantics=power_semantics,
    )
    candidate_summaries, ranking_checks = _summarize_all(
        rows=step_rows,
        user_throughputs_by_key=user_throughputs_by_key,
    )
    decision = _build_decision(
        candidate_summaries=candidate_summaries,
        ranking_checks=ranking_checks,
    )
    denominator_variability_result = {
        "denominator_changed_by_power_decision": decision[
            "denominator_changed_by_power_decision"
        ],
        "fixed_denominator_audit_catches_fixed_denominator": decision[
            "fixed_denominator_audit_catches_fixed_denominator"
        ],
        "candidate_denominator_varies": {
            f"{row['trajectory_policy']}::{row['power_semantics']}": bool(
                row["denominator_varies_in_eval"]
            )
            for row in candidate_summaries
        },
    }
    ranking_separation_result = {
        "ranking_separates_under_same_policy_rescore": decision[
            "ranking_separates_under_same_policy_rescore"
        ],
        "policies_with_rank_change": [
            row["trajectory_policy"]
            for row in ranking_checks
            if row["same_policy_throughput_rescore_vs_EE_rescore_ranking_changes"]
        ],
        "policies_with_top_rank_change": [
            row["trajectory_policy"]
            for row in ranking_checks
            if row["same_policy_throughput_rescore_vs_EE_rescore_top_changes"]
        ],
        "ranking_checks": ranking_checks,
    }

    out_dir = Path(output_dir)
    summary = {
        "inputs": {
            "config_path": str(Path(config_path)),
            "output_dir": str(out_dir),
            "learned_run_dir": None if learned_run_dir is None else str(learned_run_dir),
            "evaluation_seed_set": list(eval_seeds),
            "max_steps": max_steps,
        },
        "protocol": {
            "phase": "phase-03c-b",
            "training": "not-run",
            "catfish": "disabled",
            "multi_catfish": "disabled",
            "frozen_baseline_mutation": "forbidden/not-performed",
            "controller": (
                "hierarchical handover plus centralized discrete power-codebook"
            ),
            "controller_claim": "new-extension / HOBS-inspired; not HOBS optimizer",
            "trajectory_policies": list(trajectories.keys()),
            "counterfactual_policies": list(policies),
            "power_semantics": list(power_semantics),
            "system_EE_primary": True,
            "per_user_EE_credit_is_system_EE": False,
            "scalar_reward_success_basis": False,
        },
        "learned_trajectory": learned_trajectory,
        "candidate_summaries": candidate_summaries,
        "denominator_variability_result": denominator_variability_result,
        "ranking_separation_result": ranking_separation_result,
        "phase_03c_b_decision": decision,
        "remaining_blockers": [
            "This is static/counterfactual evidence, not learned policy evidence.",
            "The controller is a new-extension / HOBS-inspired codebook surface, not a HOBS optimizer.",
            "Phase 02B remains a synthesized proxy comparator.",
            "Any training pilot must be separately bounded and paired.",
        ],
        "forbidden_claims_still_active": [
            "Do not claim EE-MODQN effectiveness.",
            "Do not claim full paper-faithful reproduction.",
            "Do not claim Catfish, multi-Catfish, or final Catfish-EE-MODQN.",
            "Do not treat per-user EE credit as system EE.",
            "Do not use scalar reward alone as success evidence.",
            "Do not label Phase 03C-B as a HOBS optimizer.",
        ],
    }

    step_csv = _write_csv(
        out_dir / "phase03c_b_power_mdp_step_metrics.csv",
        step_rows,
        fieldnames=list(step_rows[0].keys()),
    )
    summary_csv = _write_csv(
        out_dir / "phase03c_b_power_mdp_candidate_summary.csv",
        [
            {
                "trajectory_policy": row["trajectory_policy"],
                "power_semantics": row["power_semantics"],
                "EE_system_aggregate_bps_per_w": row["EE_system_aggregate_bps_per_w"],
                "EE_system_step_mean_bps_per_w": row["EE_system_step_mean_bps_per_w"],
                "denominator_varies_in_eval": row["denominator_varies_in_eval"],
                "beam_power_vector_distinct_count": row[
                    "beam_power_vector_distinct_count"
                ],
                "one_active_beam_step_ratio": row["one_active_beam_step_ratio"],
                "throughput_mean_user_step_bps": row[
                    "throughput_mean_user_step_bps"
                ],
                "throughput_p05_user_step_bps": row[
                    "throughput_p05_user_step_bps"
                ],
                "served_ratio": row["served_ratio"],
                "outage_ratio": row["outage_ratio"],
                "handover_count": row["handover_count"],
                "budget_violation_step_count": row["budget_violations"]["step_count"],
                "budget_violation_step_ratio": row["budget_violations"]["step_ratio"],
                "throughput_EE_pearson": row[
                    "throughput_vs_EE_system_correlation"
                ]["pearson"],
                "throughput_EE_spearman": row[
                    "throughput_vs_EE_system_correlation"
                ]["spearman"],
            }
            for row in candidate_summaries
        ],
        fieldnames=[
            "trajectory_policy",
            "power_semantics",
            "EE_system_aggregate_bps_per_w",
            "EE_system_step_mean_bps_per_w",
            "denominator_varies_in_eval",
            "beam_power_vector_distinct_count",
            "one_active_beam_step_ratio",
            "throughput_mean_user_step_bps",
            "throughput_p05_user_step_bps",
            "served_ratio",
            "outage_ratio",
            "handover_count",
            "budget_violation_step_count",
            "budget_violation_step_ratio",
            "throughput_EE_pearson",
            "throughput_EE_spearman",
        ],
    )
    summary_path = write_json(
        out_dir / "phase03c_b_power_mdp_audit_summary.json",
        summary,
    )
    review_path = out_dir / "review.md"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text("\n".join(_review_lines(summary)) + "\n")

    return {
        "phase03c_b_power_mdp_audit_summary": summary_path,
        "phase03c_b_power_mdp_step_metrics": step_csv,
        "phase03c_b_power_mdp_candidate_summary": summary_csv,
        "review_md": review_path,
        "summary": summary,
    }


__all__ = ["export_phase03c_b_power_mdp_audit"]
