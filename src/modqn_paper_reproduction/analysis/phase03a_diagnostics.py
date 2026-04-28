"""Phase 03A diagnostic-only policy and denominator exercise helpers."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..algorithms.modqn import MODQNTrainer
from ..artifacts import read_run_metadata
from ..config_loader import build_environment, build_trainer_config
from ._common import write_json


_CHECKPOINT_ROLES = (
    ("final", "primary_final"),
    ("best-eval", "secondary_best_eval"),
)

_COUNTERFACTUAL_POLICIES = (
    "hold-current",
    "random-valid",
    "first-valid",
    "spread-valid-heuristic",
)

_POLICY_SEED_OFFSETS = {
    "hold-current": 1009,
    "random-valid": 2003,
    "first-valid": 3001,
    "spread-valid-heuristic": 4001,
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


def _clean_float(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def distribution(values: list[float | int]) -> dict[str, Any]:
    """Return compact numeric distribution diagnostics."""
    clean = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not clean:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "distinct": [],
            "histogram": {},
        }
    arr = np.asarray(clean, dtype=np.float64)
    counts = Counter(clean)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "distinct": sorted(float(value) for value in counts),
        "histogram": {
            str(value): int(count)
            for value, count in sorted(counts.items(), key=lambda item: item[0])
        },
    }


def concentration_metrics(counts: np.ndarray | list[int | float]) -> dict[str, float | int | bool]:
    """Compute entropy and concentration over nonzero counts."""
    arr = np.asarray(counts, dtype=np.float64)
    active = arr[arr > 0.0]
    total = float(np.sum(active))
    if active.size == 0 or total <= 0.0:
        return {
            "active_count": 0,
            "entropy_nats": 0.0,
            "entropy_normalized": 0.0,
            "hhi": 0.0,
            "top1_share": 0.0,
            "max_count": 0.0,
            "min_active_count": 0.0,
            "all_mass_one_bucket": False,
        }
    p = active / total
    entropy = float(-np.sum(p * np.log(p)))
    entropy_norm = 0.0 if active.size <= 1 else float(entropy / math.log(active.size))
    return {
        "active_count": int(active.size),
        "entropy_nats": entropy,
        "entropy_normalized": entropy_norm,
        "hhi": float(np.sum(np.square(p))),
        "top1_share": float(np.max(p)),
        "max_count": float(np.max(active)),
        "min_active_count": float(np.min(active)),
        "all_mass_one_bucket": bool(active.size == 1),
    }


def select_counterfactual_actions(
    policy: str,
    *,
    current_assignments: np.ndarray,
    masks: list[Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Select actions for non-learned diagnostic policies."""
    if policy not in _COUNTERFACTUAL_POLICIES:
        raise ValueError(f"Unsupported Phase 03A counterfactual policy: {policy!r}")

    actions = np.zeros(len(masks), dtype=np.int32)
    diagnostics: list[dict[str, Any]] = []
    assigned_counts = np.zeros_like(masks[0].mask, dtype=np.int32)

    for uid, mask_obj in enumerate(masks):
        mask = mask_obj.mask
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            actions[uid] = 0
            diagnostics.append(
                {
                    "availableActionCount": 0,
                    "selectedAction": 0,
                    "fallbackUsed": True,
                    "selectionRule": "no-valid-action-default-zero",
                }
            )
            continue

        current = int(current_assignments[uid])
        fallback_used = False
        if policy == "hold-current":
            if 0 <= current < mask.size and bool(mask[current]):
                selected = current
                rule = "current-valid"
            else:
                selected = int(valid[0])
                fallback_used = True
                rule = "current-invalid-first-valid"
        elif policy == "random-valid":
            selected = int(rng.choice(valid))
            rule = "uniform-random-valid"
        elif policy == "first-valid":
            selected = int(valid[0])
            rule = "lowest-index-valid"
        else:
            best_load = int(np.min(assigned_counts[valid]))
            selected = int(valid[assigned_counts[valid] == best_load][0])
            rule = "least-assigned-valid-then-lowest-index"

        actions[uid] = selected
        assigned_counts[selected] += 1
        diagnostics.append(
            {
                "availableActionCount": int(valid.size),
                "selectedAction": int(selected),
                "fallbackUsed": bool(fallback_used),
                "selectionRule": rule,
            }
        )

    return actions, diagnostics


def _checkpoint_paths(metadata: Any) -> dict[str, Path]:
    files = metadata.checkpoint_files
    paths: dict[str, Path] = {"final": Path(files.primary_final)}
    if files.secondary_best_eval is not None:
        paths["best-eval"] = Path(files.secondary_best_eval)
    return paths


def _action_metrics(actions: np.ndarray, num_actions: int) -> dict[str, Any]:
    counts = np.bincount(actions.astype(np.int64), minlength=num_actions)
    metrics = concentration_metrics(counts)
    return {
        "distinct_selected_action_count": metrics["active_count"],
        "selected_action_entropy_normalized": metrics["entropy_normalized"],
        "selected_action_hhi": metrics["hhi"],
        "selected_action_top1_share": metrics["top1_share"],
        "selected_action_max_count": metrics["max_count"],
        "all_users_one_selected_action": metrics["all_mass_one_bucket"],
    }


def _q_step_metrics(q_diagnostics: list[dict[str, Any] | None]) -> dict[str, Any]:
    margins = [
        float(diag["scalarizedMarginToRunnerUp"])
        for diag in q_diagnostics
        if diag is not None and diag.get("scalarizedMarginToRunnerUp") is not None
    ]
    selected_q = [
        float(diag["selectedScalarizedQ"])
        for diag in q_diagnostics
        if diag is not None and diag.get("selectedScalarizedQ") is not None
    ]
    if not margins:
        return {
            "q_margin_mean": None,
            "q_margin_min": None,
            "q_margin_zero_ratio": None,
            "q_margin_near_zero_1e_6_ratio": None,
            "selected_scalarized_q_mean": None,
        }
    arr = np.asarray(margins, dtype=np.float64)
    return {
        "q_margin_mean": float(np.mean(arr)),
        "q_margin_min": float(np.min(arr)),
        "q_margin_zero_ratio": float(np.mean(np.isclose(arr, 0.0, atol=1e-12))),
        "q_margin_near_zero_1e_6_ratio": float(np.mean(np.abs(arr) <= 1e-6)),
        "selected_scalarized_q_mean": (
            None if not selected_q else float(np.mean(selected_q))
        ),
    }


def _decision_rows(
    *,
    method: str,
    policy_label: str,
    policy_kind: str,
    checkpoint_role: str,
    checkpoint_kind: str | None,
    checkpoint_episode: int | None,
    evaluation_seed: int,
    step_index: int,
    actions: np.ndarray,
    current_assignments: np.ndarray,
    masks: list[Any],
    q_diagnostics: list[dict[str, Any] | None] | None,
    counterfactual_diagnostics: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for uid, mask_obj in enumerate(masks):
        mask = mask_obj.mask
        valid = np.flatnonzero(mask)
        selected = int(actions[uid])
        first_valid = None if valid.size == 0 else int(valid[0])
        q_diag = None if q_diagnostics is None else q_diagnostics[uid]
        cf_diag = (
            None
            if counterfactual_diagnostics is None
            else counterfactual_diagnostics[uid]
        )
        top_candidates = None
        if q_diag is not None:
            top_candidates = json.dumps(q_diag.get("topCandidates", []))
        rows.append(
            {
                "method": method,
                "policy_label": policy_label,
                "policy_kind": policy_kind,
                "checkpoint_role": checkpoint_role,
                "checkpoint_kind": checkpoint_kind,
                "checkpoint_episode": checkpoint_episode,
                "evaluation_seed": int(evaluation_seed),
                "step_index": int(step_index),
                "user_index": int(uid),
                "previous_assignment": int(current_assignments[uid]),
                "selected_action": selected,
                "selected_action_valid": bool(0 <= selected < mask.size and mask[selected]),
                "selected_current_assignment": bool(selected == int(current_assignments[uid])),
                "first_valid_action": first_valid,
                "selected_first_valid_action": bool(first_valid is not None and selected == first_valid),
                "valid_action_count": int(valid.size),
                "counterfactual_selection_rule": (
                    None if cf_diag is None else cf_diag["selectionRule"]
                ),
                "counterfactual_fallback_used": (
                    None if cf_diag is None else bool(cf_diag["fallbackUsed"])
                ),
                "selected_scalarized_q": (
                    None if q_diag is None else q_diag["selectedScalarizedQ"]
                ),
                "runner_up_action": (
                    None if q_diag is None else q_diag["runnerUpAction"]
                ),
                "runner_up_scalarized_q": (
                    None if q_diag is None else q_diag["runnerUpScalarizedQ"]
                ),
                "scalarized_margin_to_runner_up": (
                    None if q_diag is None else q_diag["scalarizedMarginToRunnerUp"]
                ),
                "top_candidates_json": top_candidates,
            }
        )
    return rows


def _step_row(
    *,
    method: str,
    policy_label: str,
    policy_kind: str,
    checkpoint_role: str,
    checkpoint_kind: str | None,
    checkpoint_episode: int | None,
    evaluation_seed: int,
    result: Any,
    actions: np.ndarray,
    current_assignments: np.ndarray,
    masks: list[Any],
    q_diagnostics: list[dict[str, Any] | None] | None,
    max_power_w: float | None,
) -> dict[str, Any]:
    beam_loads = result.user_states[0].beam_loads.astype(np.float64, copy=False)
    active_mask = result.active_beam_mask.astype(bool, copy=False)
    active_power = result.beam_transmit_power_w[active_mask].astype(
        np.float64,
        copy=False,
    )
    total_power_w = float(np.sum(active_power, dtype=np.float64))
    throughput_sum = float(
        np.sum([reward.r1_throughput for reward in result.rewards], dtype=np.float64)
    )
    ee_credit_values = [
        float(reward.r1_energy_efficiency_credit)
        for reward in result.rewards
    ]
    active_count = int(np.count_nonzero(active_mask))
    load_metrics = concentration_metrics(beam_loads)
    action_metrics = _action_metrics(actions, result.beam_transmit_power_w.size)
    q_metrics = _q_step_metrics(q_diagnostics or [])
    valid_counts = np.asarray([mask.num_valid for mask in masks], dtype=np.float64)
    selected_first_valid = [
        bool(row)
        for row in (
            actions[uid] == int(np.flatnonzero(masks[uid].mask)[0])
            for uid in range(len(masks))
            if np.flatnonzero(masks[uid].mask).size > 0
        )
    ]
    selected_current = [
        int(actions[uid]) == int(current_assignments[uid])
        for uid in range(len(actions))
    ]
    saturated = (
        np.zeros_like(active_power, dtype=bool)
        if max_power_w is None
        else np.isclose(active_power, float(max_power_w), atol=1e-12)
    )
    return {
        "method": method,
        "policy_label": policy_label,
        "policy_kind": policy_kind,
        "checkpoint_role": checkpoint_role,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_episode": checkpoint_episode,
        "evaluation_seed": int(evaluation_seed),
        "step_index": int(result.step_index),
        "active_beam_count": active_count,
        "total_active_beam_power_w": total_power_w,
        "active_beam_power_min_w": None if active_power.size == 0 else float(np.min(active_power)),
        "active_beam_power_max_w": None if active_power.size == 0 else float(np.max(active_power)),
        "saturated_active_beam_count": int(np.sum(saturated)),
        "saturated_active_beam_ratio": (
            None if active_count == 0 else float(np.mean(saturated))
        ),
        "beam_load_entropy_normalized": load_metrics["entropy_normalized"],
        "beam_load_hhi": load_metrics["hhi"],
        "beam_load_top1_share": load_metrics["top1_share"],
        "beam_load_max_users": load_metrics["max_count"],
        "beam_load_min_active_users": load_metrics["min_active_count"],
        "all_users_one_beam": bool(load_metrics["all_mass_one_bucket"]),
        "distinct_selected_action_count": action_metrics["distinct_selected_action_count"],
        "selected_action_entropy_normalized": action_metrics["selected_action_entropy_normalized"],
        "selected_action_hhi": action_metrics["selected_action_hhi"],
        "selected_action_top1_share": action_metrics["selected_action_top1_share"],
        "selected_action_max_count": action_metrics["selected_action_max_count"],
        "all_users_one_selected_action": action_metrics["all_users_one_selected_action"],
        "valid_action_count_min": float(np.min(valid_counts)),
        "valid_action_count_mean": float(np.mean(valid_counts)),
        "valid_action_count_max": float(np.max(valid_counts)),
        "selected_first_valid_ratio": (
            None if not selected_first_valid else float(np.mean(selected_first_valid))
        ),
        "selected_current_assignment_ratio": float(np.mean(selected_current)),
        "throughput_sum_bps": throughput_sum,
        "ee_system_bps_per_w": None if total_power_w <= 0.0 else throughput_sum / total_power_w,
        "mean_per_user_ee_credit_bps_per_w": float(np.mean(ee_credit_values)),
        "mean_ee_credit_to_throughput_ratio": (
            None
            if throughput_sum <= 0.0
            else float(np.sum(ee_credit_values, dtype=np.float64) / throughput_sum)
        ),
        "r3_load_balance_mean": float(np.mean([reward.r3_load_balance for reward in result.rewards])),
        "load_balance_gap_bps": float(-result.rewards[0].r3_load_balance * len(result.rewards)),
        **q_metrics,
    }


def _rollout_learned_checkpoint(
    *,
    run_dir: Path,
    method: str,
    checkpoint_role: str,
    checkpoint_path: Path,
    evaluation_seed_set: tuple[int, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
    policy_label = f"{method}/{checkpoint_role}"

    step_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for eval_seed in evaluation_seed_set:
        env_seed_seq, mobility_seed_seq = np.random.SeedSequence(eval_seed).spawn(2)
        env_rng = np.random.default_rng(env_seed_seq)
        mobility_rng = np.random.default_rng(mobility_seed_seq)
        states, masks, _diag = trainer.env.reset(env_rng, mobility_rng)
        encoded = trainer.encode_states(states)
        for _ in range(trainer.env.config.steps_per_episode):
            current_assignments = trainer.env.current_assignments()
            actions, q_diagnostics = trainer.select_actions_with_diagnostics(
                encoded,
                masks,
                top_k=3,
            )
            result = trainer.env.step(actions, env_rng)
            step_rows.append(
                _step_row(
                    method=method,
                    policy_label=policy_label,
                    policy_kind="learned-checkpoint",
                    checkpoint_role=checkpoint_role,
                    checkpoint_kind=checkpoint_kind,
                    checkpoint_episode=checkpoint_episode,
                    evaluation_seed=eval_seed,
                    result=result,
                    actions=actions,
                    current_assignments=current_assignments,
                    masks=masks,
                    q_diagnostics=q_diagnostics,
                    max_power_w=trainer.env.power_surface_config.max_power_w,
                )
            )
            action_rows.extend(
                _decision_rows(
                    method=method,
                    policy_label=policy_label,
                    policy_kind="learned-checkpoint",
                    checkpoint_role=checkpoint_role,
                    checkpoint_kind=checkpoint_kind,
                    checkpoint_episode=checkpoint_episode,
                    evaluation_seed=eval_seed,
                    step_index=result.step_index,
                    actions=actions,
                    current_assignments=current_assignments,
                    masks=masks,
                    q_diagnostics=q_diagnostics,
                    counterfactual_diagnostics=None,
                )
            )
            if result.done:
                break
            encoded = trainer.encode_states(result.user_states)
            masks = result.action_masks
    return step_rows, action_rows


def _rollout_counterfactual_policy(
    *,
    cfg: dict[str, Any],
    policy: str,
    evaluation_seed_set: tuple[int, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    env = build_environment(cfg)
    policy_label = f"counterfactual/{policy}"
    step_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for eval_seed in evaluation_seed_set:
        env_seed_seq, mobility_seed_seq = np.random.SeedSequence(eval_seed).spawn(2)
        env_rng = np.random.default_rng(env_seed_seq)
        mobility_rng = np.random.default_rng(mobility_seed_seq)
        policy_rng = np.random.default_rng(
            int(eval_seed) + _POLICY_SEED_OFFSETS[policy]
        )
        states, masks, _diag = env.reset(env_rng, mobility_rng)
        del states
        for _ in range(env.config.steps_per_episode):
            current_assignments = env.current_assignments()
            actions, cf_diagnostics = select_counterfactual_actions(
                policy,
                current_assignments=current_assignments,
                masks=masks,
                rng=policy_rng,
            )
            result = env.step(actions, env_rng)
            step_rows.append(
                _step_row(
                    method="non-learned-counterfactual",
                    policy_label=policy_label,
                    policy_kind="non-learned-counterfactual",
                    checkpoint_role="none",
                    checkpoint_kind=None,
                    checkpoint_episode=None,
                    evaluation_seed=eval_seed,
                    result=result,
                    actions=actions,
                    current_assignments=current_assignments,
                    masks=masks,
                    q_diagnostics=None,
                    max_power_w=env.power_surface_config.max_power_w,
                )
            )
            action_rows.extend(
                _decision_rows(
                    method="non-learned-counterfactual",
                    policy_label=policy_label,
                    policy_kind="non-learned-counterfactual",
                    checkpoint_role="none",
                    checkpoint_kind=None,
                    checkpoint_episode=None,
                    evaluation_seed=eval_seed,
                    step_index=result.step_index,
                    actions=actions,
                    current_assignments=current_assignments,
                    masks=masks,
                    q_diagnostics=None,
                    counterfactual_diagnostics=cf_diagnostics,
                )
            )
            if result.done:
                break
            masks = result.action_masks
    return step_rows, action_rows


def _summarize_policy_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    fields = [
        "active_beam_count",
        "total_active_beam_power_w",
        "beam_load_entropy_normalized",
        "beam_load_hhi",
        "beam_load_top1_share",
        "beam_load_max_users",
        "distinct_selected_action_count",
        "selected_action_entropy_normalized",
        "selected_action_top1_share",
        "valid_action_count_mean",
        "selected_first_valid_ratio",
        "selected_current_assignment_ratio",
        "throughput_sum_bps",
        "ee_system_bps_per_w",
        "mean_per_user_ee_credit_bps_per_w",
        "mean_ee_credit_to_throughput_ratio",
        "r3_load_balance_mean",
        "load_balance_gap_bps",
        "saturated_active_beam_ratio",
        "q_margin_mean",
        "q_margin_min",
        "q_margin_zero_ratio",
        "q_margin_near_zero_1e_6_ratio",
    ]
    metrics = {
        field: distribution(
            [
                value
                for value in (_clean_float(row.get(field)) for row in rows)
                if value is not None
            ]
        )
        for field in fields
    }
    return {
        "method": first["method"],
        "policy_label": first["policy_label"],
        "policy_kind": first["policy_kind"],
        "checkpoint_role": first["checkpoint_role"],
        "checkpoint_kind": first["checkpoint_kind"],
        "checkpoint_episode": first["checkpoint_episode"],
        "step_count": len(rows),
        "evaluation_seeds": sorted({int(row["evaluation_seed"]) for row in rows}),
        "active_beam_count_distribution": metrics["active_beam_count"],
        "total_active_beam_power_w_distribution": metrics[
            "total_active_beam_power_w"
        ],
        "beam_load_metrics": {
            "entropy_normalized": metrics["beam_load_entropy_normalized"],
            "hhi": metrics["beam_load_hhi"],
            "top1_share": metrics["beam_load_top1_share"],
            "max_users": metrics["beam_load_max_users"],
            "all_users_one_beam_step_ratio": float(
                np.mean([bool(row["all_users_one_beam"]) for row in rows])
            ),
            "saturated_active_beam_ratio": metrics["saturated_active_beam_ratio"],
        },
        "selected_action_diversity": {
            "distinct_selected_action_count": metrics[
                "distinct_selected_action_count"
            ],
            "entropy_normalized": metrics["selected_action_entropy_normalized"],
            "top1_share": metrics["selected_action_top1_share"],
            "selected_first_valid_ratio": metrics["selected_first_valid_ratio"],
            "selected_current_assignment_ratio": metrics[
                "selected_current_assignment_ratio"
            ],
            "valid_action_count_mean": metrics["valid_action_count_mean"],
        },
        "reward_and_metric_scaling": {
            "throughput_sum_bps": metrics["throughput_sum_bps"],
            "ee_system_bps_per_w": metrics["ee_system_bps_per_w"],
            "mean_per_user_ee_credit_bps_per_w": metrics[
                "mean_per_user_ee_credit_bps_per_w"
            ],
            "mean_ee_credit_to_throughput_ratio": metrics[
                "mean_ee_credit_to_throughput_ratio"
            ],
            "r3_load_balance_mean": metrics["r3_load_balance_mean"],
            "load_balance_gap_bps": metrics["load_balance_gap_bps"],
        },
        "q_margin_diagnostics": {
            "q_margin_mean": metrics["q_margin_mean"],
            "q_margin_min": metrics["q_margin_min"],
            "q_margin_zero_ratio": metrics["q_margin_zero_ratio"],
            "q_margin_near_zero_1e_6_ratio": metrics[
                "q_margin_near_zero_1e_6_ratio"
            ],
        },
        "denominator_varies_in_eval": (
            metrics["total_active_beam_power_w"]["max"] is not None
            and metrics["total_active_beam_power_w"]["min"] is not None
            and float(metrics["total_active_beam_power_w"]["max"])
            > float(metrics["total_active_beam_power_w"]["min"])
        ),
    }


def _build_root_cause_summary(policy_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    learned = [
        summary
        for summary in policy_summaries
        if summary["policy_kind"] == "learned-checkpoint"
    ]
    counterfactual = [
        summary
        for summary in policy_summaries
        if summary["policy_kind"] == "non-learned-counterfactual"
    ]
    learned_fixed = all(
        not bool(summary["denominator_varies_in_eval"]) for summary in learned
    )
    counter_varies = any(
        bool(summary["denominator_varies_in_eval"]) for summary in counterfactual
    )
    learned_one_beam = all(
        float(summary["beam_load_metrics"]["all_users_one_beam_step_ratio"]) >= 0.999
        for summary in learned
    )
    learned_saturated = all(
        summary["beam_load_metrics"]["saturated_active_beam_ratio"]["mean"] is not None
        and float(summary["beam_load_metrics"]["saturated_active_beam_ratio"]["mean"]) >= 0.999
        for summary in learned
    )
    learned_valid_action_mean = [
        summary["selected_action_diversity"]["valid_action_count_mean"]["mean"]
        for summary in learned
        if summary["selected_action_diversity"]["valid_action_count_mean"]["mean"]
        is not None
    ]
    learned_margin_near_zero = [
        summary["q_margin_diagnostics"]["q_margin_near_zero_1e_6_ratio"]["mean"]
        for summary in learned
        if summary["q_margin_diagnostics"]["q_margin_near_zero_1e_6_ratio"]["mean"]
        is not None
    ]
    return {
        "denominator_variability_exists_in_environment": bool(counter_varies),
        "learned_policies_exercise_denominator_variability": not learned_fixed,
        "learned_policies_all_users_one_beam": bool(learned_one_beam),
        "learned_active_beams_saturated_at_max_power": bool(learned_saturated),
        "action_mask_forces_single_beam": bool(
            learned_valid_action_mean and max(learned_valid_action_mean) <= 1.0
        ),
        "q_value_near_tie_is_primary_cause": bool(
            learned_margin_near_zero
            and max(float(value) for value in learned_margin_near_zero) >= 0.95
        ),
        "r3_load_balance_insufficient_to_prevent_collapse": bool(
            learned_one_beam
            and learned_valid_action_mean
            and max(float(value) for value in learned_valid_action_mean) > 1.0
        ),
        "per_user_ee_credit_degenerates_to_throughput_scaling_under_collapse": bool(
            learned_one_beam and learned_saturated
        ),
        "phase_02b_power_formula_is_globally_fixed": not bool(counter_varies),
        "assessment": (
            "The Phase 02B denominator is variable under valid counterfactual "
            "actions, but the learned Phase 03 policies collapse every evaluated "
            "step to one active beam at the per-beam max-power cap. Under that "
            "collapse, per-user EE credit becomes a constant multiple of "
            "throughput, and the configured r3 load-balance term is not strong "
            "enough in this pilot to prevent the collapse. Evaluation therefore "
            "cannot distinguish EE behavior from throughput scaling."
        ),
    }


def _review_lines(summary: dict[str, Any]) -> list[str]:
    root = summary["root_cause_assessment"]
    primary = summary["primary_learned_policy_summary"]
    policies = {
        row["policy_label"]: row
        for row in summary["policy_summaries"]
    }
    random_valid = policies.get("counterfactual/random-valid")
    hold_current = policies.get("counterfactual/hold-current")
    spread_valid = policies.get("counterfactual/spread-valid-heuristic")
    next_gate = summary["recommended_next_gate"]
    return [
        "# Phase 03A Diagnostic Review",
        "",
        "Diagnostic-only follow-up. No reward change, Catfish, multi-Catfish, "
        "or long 9000-episode training was run.",
        "",
        "## Primary Learned Policy",
        "",
        f"- policy: `{primary['policy_label']}`",
        f"- active beam count distinct values: "
        f"`{primary['active_beam_count_distribution']['distinct']}`",
        f"- total active beam power distinct values: "
        f"`{primary['total_active_beam_power_w_distribution']['distinct']}`",
        f"- all-users-one-beam step ratio: "
        f"`{primary['beam_load_metrics']['all_users_one_beam_step_ratio']}`",
        f"- valid action count mean: "
        f"`{primary['selected_action_diversity']['valid_action_count_mean']['mean']}`",
        f"- mean EE-credit / throughput ratio: "
        f"`{primary['reward_and_metric_scaling']['mean_ee_credit_to_throughput_ratio']['mean']}`",
        f"- Q near-zero margin ratio: "
        f"`{primary['q_margin_diagnostics']['q_margin_near_zero_1e_6_ratio']['mean']}`",
        f"- denominator varies in learned eval: "
        f"`{primary['denominator_varies_in_eval']}`",
        "",
        "## Counterfactuals",
        "",
        "- first-valid is a deliberate collapse control and keeps one active beam.",
        (
            "- hold-current active beam count distinct values: "
            f"`{hold_current['active_beam_count_distribution']['distinct']}`"
            if hold_current is not None
            else "- hold-current unavailable"
        ),
        (
            "- random-valid total active beam power distinct values include: "
            f"`{random_valid['total_active_beam_power_w_distribution']['distinct'][:5]}`"
            if random_valid is not None
            else "- random-valid unavailable"
        ),
        (
            "- spread-valid active beam count distinct values: "
            f"`{spread_valid['active_beam_count_distribution']['distinct']}`"
            if spread_valid is not None
            else "- spread-valid unavailable"
        ),
        "",
        "## Root Cause",
        "",
        f"- denominator exists in environment: "
        f"`{root['denominator_variability_exists_in_environment']}`",
        f"- learned policies exercise denominator variability: "
        f"`{root['learned_policies_exercise_denominator_variability']}`",
        f"- action mask forces single beam: `{root['action_mask_forces_single_beam']}`",
        f"- Q near-tie primary cause: `{root['q_value_near_tie_is_primary_cause']}`",
        f"- r3 load balance insufficient to prevent collapse: "
        f"`{root['r3_load_balance_insufficient_to_prevent_collapse']}`",
        f"- per-user EE degenerates under collapse: "
        f"`{root['per_user_ee_credit_degenerates_to_throughput_scaling_under_collapse']}`",
        f"- Phase 02B power formula globally fixed: "
        f"`{root['phase_02b_power_formula_is_globally_fixed']}`",
        "",
        "## Next Gate",
        "",
        f"- more training budget: `{next_gate['more_training_budget']}`",
        f"- reward normalization: `{next_gate['reward_normalization']}`",
        f"- load-balance calibration: `{next_gate['load_balance_calibration']}`",
        "- denominator-sensitive action/power design remains required before "
        "larger training.",
        "",
        "## Decision",
        "",
        "- Phase 03 remains `NEEDS MORE EVIDENCE`.",
        "- Do not claim EE-MODQN effectiveness from this diagnostic.",
    ]


def export_phase03a_diagnostics(
    *,
    control_run_dir: str | Path,
    ee_run_dir: str | Path,
    output_dir: str | Path,
    evaluation_seed_set: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Export diagnostic-only Phase 03A policy collapse artifacts."""
    control_dir = Path(control_run_dir)
    ee_dir = Path(ee_run_dir)
    out_dir = Path(output_dir)
    control_metadata = read_run_metadata(control_dir / "run_metadata.json")
    ee_metadata = read_run_metadata(ee_dir / "run_metadata.json")
    control_eval_seeds = tuple(control_metadata.seeds.evaluation_seed_set)
    ee_eval_seeds = tuple(ee_metadata.seeds.evaluation_seed_set)
    if control_eval_seeds != ee_eval_seeds:
        raise ValueError(
            "Control and EE run metadata must use the same evaluation seed set."
        )
    eval_seeds = tuple(
        int(seed)
        for seed in (
            evaluation_seed_set
            if evaluation_seed_set is not None
            else control_eval_seeds
        )
    )
    if not eval_seeds:
        raise ValueError("Phase 03A diagnostics require evaluation seeds.")

    all_step_rows: list[dict[str, Any]] = []
    all_action_rows: list[dict[str, Any]] = []
    for method, run_dir, metadata in (
        ("MODQN-control", control_dir, control_metadata),
        ("EE-MODQN", ee_dir, ee_metadata),
    ):
        paths = _checkpoint_paths(metadata)
        for checkpoint_role, _field in _CHECKPOINT_ROLES:
            checkpoint_path = paths.get(checkpoint_role)
            if checkpoint_path is None:
                continue
            step_rows, action_rows = _rollout_learned_checkpoint(
                run_dir=run_dir,
                method=method,
                checkpoint_role=checkpoint_role,
                checkpoint_path=checkpoint_path,
                evaluation_seed_set=eval_seeds,
            )
            all_step_rows.extend(step_rows)
            all_action_rows.extend(action_rows)

    counterfactual_cfg = control_metadata.resolved_config_snapshot
    for policy in _COUNTERFACTUAL_POLICIES:
        step_rows, action_rows = _rollout_counterfactual_policy(
            cfg=counterfactual_cfg,
            policy=policy,
            evaluation_seed_set=eval_seeds,
        )
        all_step_rows.extend(step_rows)
        all_action_rows.extend(action_rows)

    rows_by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_step_rows:
        rows_by_policy[str(row["policy_label"])].append(row)
    policy_summaries = [
        _summarize_policy_rows(rows)
        for _label, rows in sorted(rows_by_policy.items())
    ]
    primary_learned = next(
        summary
        for summary in policy_summaries
        if summary["policy_label"] == "EE-MODQN/best-eval"
    )
    root_cause = _build_root_cause_summary(policy_summaries)
    summary = {
        "inputs": {
            "control_run_dir": str(control_dir),
            "ee_run_dir": str(ee_dir),
            "output_dir": str(out_dir),
            "evaluation_seed_set": list(eval_seeds),
        },
        "protocol": {
            "phase": "phase-03a",
            "diagnostic_only": True,
            "catfish": "disabled",
            "multi_catfish": "disabled",
            "reward_changes": "none",
            "long_training": "not-run",
            "artifact_namespace": str(out_dir),
            "counterfactual_policies": list(_COUNTERFACTUAL_POLICIES),
        },
        "policy_summaries": policy_summaries,
        "primary_learned_policy_summary": primary_learned,
        "root_cause_assessment": root_cause,
        "phase_03_decision": "NEEDS MORE EVIDENCE",
        "recommended_next_gate": {
            "more_training_budget": (
                "not the next gate by itself; first make the objective surface "
                "exercise denominator variability"
            ),
            "reward_normalization": "required design candidate before larger training",
            "load_balance_calibration": "required design candidate before larger training",
            "stronger_denominator_sensitive_action_space": (
                "required if EE objective is expected to choose lower-power states"
            ),
            "different_power_allocation_model": (
                "candidate sensitivity check; current proxy is variable but collapses "
                "under learned one-beam policies"
            ),
            "blocked": True,
        },
        "forbidden_claims_still_active": [
            "Do not claim EE-MODQN effectiveness.",
            "Do not claim energy-aware learning while evaluated denominator is fixed.",
            "Do not claim Catfish, multi-Catfish, or final Catfish-EE-MODQN.",
            "Do not claim full paper-faithful reproduction.",
        ],
    }

    step_csv = _write_csv(
        out_dir / "phase03a_step_diagnostics.csv",
        all_step_rows,
        fieldnames=list(all_step_rows[0].keys()),
    )
    action_csv = _write_csv(
        out_dir / "phase03a_action_diagnostics.csv",
        all_action_rows,
        fieldnames=list(all_action_rows[0].keys()),
    )
    summary_path = write_json(out_dir / "phase03a_diagnostic_summary.json", summary)
    review_path = out_dir / "review.md"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text("\n".join(_review_lines(summary)) + "\n")
    return {
        "phase03a_diagnostic_summary": summary_path,
        "phase03a_step_diagnostics": step_csv,
        "phase03a_action_diagnostics": action_csv,
        "review_md": review_path,
        "summary": summary,
    }


__all__ = [
    "concentration_metrics",
    "distribution",
    "export_phase03a_diagnostics",
    "select_counterfactual_actions",
]
