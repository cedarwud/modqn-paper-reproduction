"""Evaluation-only counterfactual beam-eligibility helpers for Phase 01E2."""

from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any

import numpy as np

from ..algorithms.modqn import MODQNTrainer
from ..artifacts import RunArtifactPaths, read_run_metadata
from ..artifacts.compat import (
    _select_timeline_seed,
    resolve_training_config_snapshot,
    select_replay_checkpoint,
)
from ..config_loader import build_environment, build_trainer_config
from ..env.step import ActionMask, StepEnvironment, UserState
from ._common import write_json

_COUNTERFACTUAL_MODE = "nearest-beam-per-visible-satellite"
_COMPARISON_FIELDS = [
    "policy",
    "baseline_scalar_reward",
    "counterfactual_scalar_reward",
    "delta_scalar_reward",
    "baseline_r1_mean",
    "counterfactual_r1_mean",
    "delta_r1_mean",
    "baseline_r2_mean",
    "counterfactual_r2_mean",
    "delta_r2_mean",
    "baseline_r3_mean",
    "counterfactual_r3_mean",
    "delta_r3_mean",
    "baseline_total_handovers",
    "counterfactual_total_handovers",
    "delta_total_handovers",
    "baseline_mean_valid_action_count",
    "counterfactual_mean_valid_action_count",
    "baseline_rss_tie_rate",
    "counterfactual_rss_tie_rate",
    "baseline_top1_top2_same_satellite_rate",
    "counterfactual_top1_top2_same_satellite_rate",
    "same_state_action_changed_rate",
    "same_state_satellite_changed_rate",
]


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str],
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return target


def _action_satellite_index(
    action: int | None,
    beams_per_satellite: int,
) -> int | None:
    if action is None:
        return None
    return int(action // beams_per_satellite)


def _rss_selection(
    state: UserState,
    mask: np.ndarray,
) -> tuple[int | None, int, bool]:
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        return (None, 0, False)
    valid_channels = state.channel_quality[valid].astype(np.float64, copy=False)
    max_channel = float(np.max(valid_channels))
    tied = np.isclose(valid_channels, max_channel, rtol=1e-6, atol=1e-12)
    tied_actions = valid[tied]
    return (int(tied_actions[0]), int(tied_actions.size), bool(tied_actions.size > 1))


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _fraction_or_none(values: list[bool]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _classify_effect(
    *,
    satellite_change_rate: float | None,
    action_change_rate: float | None,
    baseline_scalar_reward: float,
    counterfactual_scalar_reward: float,
) -> str:
    scalar_delta = abs(counterfactual_scalar_reward - baseline_scalar_reward)
    material_delta_floor = max(abs(baseline_scalar_reward) * 0.10, 10.0)

    if satellite_change_rate is not None and satellite_change_rate >= 0.25:
        return "material"
    if scalar_delta >= material_delta_floor:
        return "material"
    if action_change_rate is not None and action_change_rate > 0.0:
        return "mostly-tie-break"
    return "absent"


def _build_trainer_from_artifact(
    input_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, str, dict[str, Any], tuple[float, float, float]]:
    metadata = read_run_metadata(RunArtifactPaths(input_dir).run_metadata_json)
    cfg = resolve_training_config_snapshot(metadata, artifact_dir=input_dir)
    trainer_cfg = build_trainer_config(cfg)
    seeds = metadata.seeds.to_dict()

    env = build_environment(copy.deepcopy(cfg))
    trainer = MODQNTrainer(
        env=env,
        config=trainer_cfg,
        train_seed=int(seeds["train_seed"]),
        env_seed=int(seeds["environment_seed"]),
        mobility_seed=int(seeds["mobility_seed"]),
    )
    checkpoint_path, checkpoint_kind = select_replay_checkpoint(
        metadata,
        artifact_dir=input_dir,
    )
    checkpoint_payload = trainer.load_checkpoint(
        checkpoint_path,
        load_optimizers=False,
    )
    objective_weights = tuple(
        float(value)
        for value in checkpoint_payload.get(
            "trainer_config",
            {},
        ).get("objective_weights", trainer.config.objective_weights)
    )
    return (
        metadata.to_dict(),
        cfg,
        checkpoint_path,
        checkpoint_kind,
        checkpoint_payload,
        objective_weights,
    )


def _counterfactual_masks_for_env(
    env: StepEnvironment,
    masks: list[ActionMask],
) -> list[ActionMask]:
    sats = env.current_satellites()
    user_positions = env.current_user_positions()
    beams_per_satellite = env.beam_pattern.num_beams

    counterfactual_masks: list[ActionMask] = []
    for uid, mask in enumerate(masks):
        updated = np.zeros_like(mask.mask, dtype=bool)
        ulat, ulon = user_positions[uid]
        for sat_index in range(env.orbit.num_satellites):
            start = sat_index * beams_per_satellite
            stop = start + beams_per_satellite
            block = mask.mask[start:stop]
            if not np.any(block):
                continue

            local_beam = env.beam_pattern.nearest_beam(
                sats[sat_index],
                ulat,
                ulon,
            )
            local_candidates = np.flatnonzero(block)
            if local_beam not in local_candidates:
                local_beam = int(local_candidates[0])
            updated[start + local_beam] = True

        counterfactual_masks.append(ActionMask(mask=updated))
    return counterfactual_masks


def _policy_decisions(
    trainer: MODQNTrainer,
    states: list[UserState],
    masks: list[ActionMask],
    *,
    policy: str,
    objective_weights: tuple[float, float, float],
) -> tuple[np.ndarray, list[dict[str, Any] | None], dict[str, float]]:
    if policy == "modqn":
        encoded = trainer.encode_states(states)
        actions, diagnostics = trainer.select_actions_with_diagnostics(
            encoded,
            masks,
            objective_weights=objective_weights,
            top_k=2,
        )
        valid_counts = [int(mask.num_valid) for mask in masks]
        same_satellite_flags = []
        for row in diagnostics:
            if row is None or row["runnerUpAction"] is None:
                continue
            same_satellite_flags.append(
                _action_satellite_index(
                    int(row["selectedAction"]),
                    trainer.env.beam_pattern.num_beams,
                )
                == _action_satellite_index(
                    int(row["runnerUpAction"]),
                    trainer.env.beam_pattern.num_beams,
                )
            )
        return (
            actions,
            diagnostics,
            {
                "mean_valid_action_count": float(np.mean(valid_counts)) if valid_counts else 0.0,
                "rss_tie_rate": 0.0,
                "top1_top2_same_satellite_rate": (
                    float(np.mean(same_satellite_flags))
                    if same_satellite_flags else 0.0
                ),
            },
        )

    if policy != "rss_max":
        raise ValueError(f"Unsupported policy {policy!r}")

    actions = np.zeros(len(masks), dtype=np.int32)
    diagnostics: list[dict[str, Any] | None] = []
    valid_counts = []
    rss_tie_flags = []
    for uid, mask in enumerate(masks):
        action, tie_width, has_tie = _rss_selection(states[uid], mask.mask)
        selected_action = 0 if action is None else int(action)
        actions[uid] = selected_action
        valid_counts.append(int(mask.num_valid))
        rss_tie_flags.append(bool(has_tie))
        diagnostics.append(
            {
                "selectedAction": selected_action,
                "runnerUpAction": None,
                "rssTieWidth": int(tie_width),
            }
        )
    return (
        actions,
        diagnostics,
        {
            "mean_valid_action_count": float(np.mean(valid_counts)) if valid_counts else 0.0,
            "rss_tie_rate": float(np.mean(rss_tie_flags)) if rss_tie_flags else 0.0,
            "top1_top2_same_satellite_rate": 0.0,
        },
    )


def _rollout_policy(
    cfg: dict[str, Any],
    seeds: dict[str, Any],
    checkpoint_path: Path,
    *,
    objective_weights: tuple[float, float, float],
    evaluation_seed: int,
    policy: str,
    use_counterfactual_mask: bool,
    max_steps: int | None,
    max_users: int | None,
) -> dict[str, Any]:
    env = build_environment(copy.deepcopy(cfg))
    trainer = MODQNTrainer(
        env=env,
        config=build_trainer_config(cfg),
        train_seed=int(seeds["train_seed"]),
        env_seed=int(seeds["environment_seed"]),
        mobility_seed=int(seeds["mobility_seed"]),
    )
    trainer.load_checkpoint(checkpoint_path, load_optimizers=False)

    env_seed_seq, mobility_seed_seq = np.random.SeedSequence(evaluation_seed).spawn(2)
    env_rng = np.random.default_rng(env_seed_seq)
    mobility_rng = np.random.default_rng(mobility_seed_seq)
    states, masks, _diag = trainer.env.reset(env_rng, mobility_rng)

    users_total = len(states)
    users_to_score = users_total if max_users is None else min(int(max_users), users_total)
    steps_audited = 0
    ep_reward = np.zeros(3, dtype=np.float64)
    ep_handovers = 0
    mean_valid_action_counts: list[float] = []
    rss_tie_rates: list[float] = []
    same_satellite_rates: list[float] = []

    while True:
        if max_steps is not None and steps_audited >= int(max_steps):
            break

        decision_masks = (
            _counterfactual_masks_for_env(trainer.env, masks)
            if use_counterfactual_mask
            else masks
        )
        active_states = states[:users_to_score]
        active_masks = decision_masks[:users_to_score]
        actions_prefix, _diagnostics, policy_metrics = _policy_decisions(
            trainer,
            active_states,
            active_masks,
            policy=policy,
            objective_weights=objective_weights,
        )
        actions = trainer.env.current_assignments()
        actions[:users_to_score] = actions_prefix

        result = trainer.env.step(actions, env_rng)
        for uid in range(users_to_score):
            rw = result.rewards[uid]
            reward_vec = np.array(
                [rw.r1_throughput, rw.r2_handover, rw.r3_load_balance],
                dtype=np.float64,
            )
            ep_reward += reward_vec
            if rw.r2_handover < 0:
                ep_handovers += 1

        mean_valid_action_counts.append(float(policy_metrics["mean_valid_action_count"]))
        rss_tie_rates.append(float(policy_metrics["rss_tie_rate"]))
        same_satellite_rates.append(
            float(policy_metrics["top1_top2_same_satellite_rate"])
        )

        steps_audited += 1
        if result.done:
            break
        states = result.user_states
        masks = result.action_masks

    avg_reward = ep_reward / max(users_to_score, 1)
    scalar = float(np.dot(objective_weights, avg_reward))
    return {
        "policy": policy,
        "mask_mode": (
            _COUNTERFACTUAL_MODE if use_counterfactual_mask else "baseline-visible-satellite-block"
        ),
        "evaluation_seed": int(evaluation_seed),
        "steps_audited": int(steps_audited),
        "users_scored_per_step": int(users_to_score),
        "mean_scalar_reward": scalar,
        "mean_r1": float(avg_reward[0]),
        "mean_r2": float(avg_reward[1]),
        "mean_r3": float(avg_reward[2]),
        "mean_total_handovers": float(ep_handovers),
        "mean_valid_action_count": _mean_or_none(mean_valid_action_counts),
        "rss_tie_rate": _mean_or_none(rss_tie_rates),
        "top1_top2_same_satellite_rate": _mean_or_none(same_satellite_rates),
    }


def _same_state_decision_comparison(
    cfg: dict[str, Any],
    seeds: dict[str, Any],
    checkpoint_path: Path,
    *,
    objective_weights: tuple[float, float, float],
    evaluation_seed: int,
    max_steps: int | None,
    max_users: int | None,
) -> dict[str, Any]:
    env = build_environment(copy.deepcopy(cfg))
    trainer = MODQNTrainer(
        env=env,
        config=build_trainer_config(cfg),
        train_seed=int(seeds["train_seed"]),
        env_seed=int(seeds["environment_seed"]),
        mobility_seed=int(seeds["mobility_seed"]),
    )
    trainer.load_checkpoint(checkpoint_path, load_optimizers=False)

    env_seed_seq, mobility_seed_seq = np.random.SeedSequence(evaluation_seed).spawn(2)
    env_rng = np.random.default_rng(env_seed_seq)
    mobility_rng = np.random.default_rng(mobility_seed_seq)
    states, masks, _diag = trainer.env.reset(env_rng, mobility_rng)

    users_total = len(states)
    users_to_compare = users_total if max_users is None else min(int(max_users), users_total)
    steps_audited = 0

    metrics: dict[str, dict[str, list[bool] | list[float]]] = {
        "modqn": {
            "action_changed": [],
            "satellite_changed": [],
            "baseline_valid_action_count": [],
            "counterfactual_valid_action_count": [],
            "baseline_rss_tie_rate": [],
            "counterfactual_rss_tie_rate": [],
            "baseline_top1_top2_same_satellite_rate": [],
            "counterfactual_top1_top2_same_satellite_rate": [],
        },
        "rss_max": {
            "action_changed": [],
            "satellite_changed": [],
            "baseline_valid_action_count": [],
            "counterfactual_valid_action_count": [],
            "baseline_rss_tie_rate": [],
            "counterfactual_rss_tie_rate": [],
            "baseline_top1_top2_same_satellite_rate": [],
            "counterfactual_top1_top2_same_satellite_rate": [],
        },
    }

    while True:
        if max_steps is not None and steps_audited >= int(max_steps):
            break

        baseline_masks = masks[:users_to_compare]
        counter_masks = _counterfactual_masks_for_env(
            trainer.env,
            masks,
        )[:users_to_compare]
        active_states = states[:users_to_compare]

        for policy in ("modqn", "rss_max"):
            baseline_actions, baseline_diag, baseline_policy_metrics = _policy_decisions(
                trainer,
                active_states,
                baseline_masks,
                policy=policy,
                objective_weights=objective_weights,
            )
            counter_actions, counter_diag, counter_policy_metrics = _policy_decisions(
                trainer,
                active_states,
                counter_masks,
                policy=policy,
                objective_weights=objective_weights,
            )

            metrics[policy]["baseline_valid_action_count"].append(
                float(baseline_policy_metrics["mean_valid_action_count"])
            )
            metrics[policy]["counterfactual_valid_action_count"].append(
                float(counter_policy_metrics["mean_valid_action_count"])
            )
            metrics[policy]["baseline_rss_tie_rate"].append(
                float(baseline_policy_metrics["rss_tie_rate"])
            )
            metrics[policy]["counterfactual_rss_tie_rate"].append(
                float(counter_policy_metrics["rss_tie_rate"])
            )
            metrics[policy]["baseline_top1_top2_same_satellite_rate"].append(
                float(baseline_policy_metrics["top1_top2_same_satellite_rate"])
            )
            metrics[policy]["counterfactual_top1_top2_same_satellite_rate"].append(
                float(counter_policy_metrics["top1_top2_same_satellite_rate"])
            )

            for uid in range(users_to_compare):
                baseline_action = int(baseline_actions[uid])
                counter_action = int(counter_actions[uid])
                metrics[policy]["action_changed"].append(
                    baseline_action != counter_action
                )
                metrics[policy]["satellite_changed"].append(
                    _action_satellite_index(
                        baseline_action,
                        trainer.env.beam_pattern.num_beams,
                    )
                    != _action_satellite_index(
                        counter_action,
                        trainer.env.beam_pattern.num_beams,
                    )
                )

        baseline_actions_full, _, _ = _policy_decisions(
            trainer,
            states,
            masks,
            policy="modqn",
            objective_weights=objective_weights,
        )
        result = trainer.env.step(baseline_actions_full, env_rng)
        steps_audited += 1
        if result.done:
            break
        states = result.user_states
        masks = result.action_masks

    return {
        "evaluation_seed": int(evaluation_seed),
        "steps_audited": int(steps_audited),
        "users_compared_per_step": int(users_to_compare),
        "modqn": {
            "same_state_action_changed_rate": _fraction_or_none(
                metrics["modqn"]["action_changed"]
            ),
            "same_state_satellite_changed_rate": _fraction_or_none(
                metrics["modqn"]["satellite_changed"]
            ),
            "baseline_mean_valid_action_count": _mean_or_none(
                metrics["modqn"]["baseline_valid_action_count"]
            ),
            "counterfactual_mean_valid_action_count": _mean_or_none(
                metrics["modqn"]["counterfactual_valid_action_count"]
            ),
            "baseline_rss_tie_rate": _mean_or_none(
                metrics["modqn"]["baseline_rss_tie_rate"]
            ),
            "counterfactual_rss_tie_rate": _mean_or_none(
                metrics["modqn"]["counterfactual_rss_tie_rate"]
            ),
            "baseline_top1_top2_same_satellite_rate": _mean_or_none(
                metrics["modqn"]["baseline_top1_top2_same_satellite_rate"]
            ),
            "counterfactual_top1_top2_same_satellite_rate": _mean_or_none(
                metrics["modqn"]["counterfactual_top1_top2_same_satellite_rate"]
            ),
        },
        "rss_max": {
            "same_state_action_changed_rate": _fraction_or_none(
                metrics["rss_max"]["action_changed"]
            ),
            "same_state_satellite_changed_rate": _fraction_or_none(
                metrics["rss_max"]["satellite_changed"]
            ),
            "baseline_mean_valid_action_count": _mean_or_none(
                metrics["rss_max"]["baseline_valid_action_count"]
            ),
            "counterfactual_mean_valid_action_count": _mean_or_none(
                metrics["rss_max"]["counterfactual_valid_action_count"]
            ),
            "baseline_rss_tie_rate": _mean_or_none(
                metrics["rss_max"]["baseline_rss_tie_rate"]
            ),
            "counterfactual_rss_tie_rate": _mean_or_none(
                metrics["rss_max"]["counterfactual_rss_tie_rate"]
            ),
            "baseline_top1_top2_same_satellite_rate": _mean_or_none(
                metrics["rss_max"]["baseline_top1_top2_same_satellite_rate"]
            ),
            "counterfactual_top1_top2_same_satellite_rate": _mean_or_none(
                metrics["rss_max"]["counterfactual_top1_top2_same_satellite_rate"]
            ),
        },
    }


def _comparison_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in ("modqn", "rss_max"):
        baseline = summary["baseline_eval"][policy]
        counter = summary["counterfactual_eval"][policy]
        compare = summary["same_state_decision_comparison"][policy]
        rows.append(
            {
                "policy": policy,
                "baseline_scalar_reward": baseline["mean_scalar_reward"],
                "counterfactual_scalar_reward": counter["mean_scalar_reward"],
                "delta_scalar_reward": (
                    counter["mean_scalar_reward"] - baseline["mean_scalar_reward"]
                ),
                "baseline_r1_mean": baseline["mean_r1"],
                "counterfactual_r1_mean": counter["mean_r1"],
                "delta_r1_mean": counter["mean_r1"] - baseline["mean_r1"],
                "baseline_r2_mean": baseline["mean_r2"],
                "counterfactual_r2_mean": counter["mean_r2"],
                "delta_r2_mean": counter["mean_r2"] - baseline["mean_r2"],
                "baseline_r3_mean": baseline["mean_r3"],
                "counterfactual_r3_mean": counter["mean_r3"],
                "delta_r3_mean": counter["mean_r3"] - baseline["mean_r3"],
                "baseline_total_handovers": baseline["mean_total_handovers"],
                "counterfactual_total_handovers": counter["mean_total_handovers"],
                "delta_total_handovers": (
                    counter["mean_total_handovers"] - baseline["mean_total_handovers"]
                ),
                "baseline_mean_valid_action_count": (
                    compare["baseline_mean_valid_action_count"]
                ),
                "counterfactual_mean_valid_action_count": (
                    compare["counterfactual_mean_valid_action_count"]
                ),
                "baseline_rss_tie_rate": compare["baseline_rss_tie_rate"],
                "counterfactual_rss_tie_rate": compare["counterfactual_rss_tie_rate"],
                "baseline_top1_top2_same_satellite_rate": (
                    compare["baseline_top1_top2_same_satellite_rate"]
                ),
                "counterfactual_top1_top2_same_satellite_rate": (
                    compare["counterfactual_top1_top2_same_satellite_rate"]
                ),
                "same_state_action_changed_rate": (
                    compare["same_state_action_changed_rate"]
                ),
                "same_state_satellite_changed_rate": (
                    compare["same_state_satellite_changed_rate"]
                ),
            }
        )
    return rows


def _review_lines(summary: dict[str, Any]) -> list[str]:
    modqn_same_state = summary["same_state_decision_comparison"]["modqn"]
    rss_same_state = summary["same_state_decision_comparison"]["rss_max"]
    return [
        "# Counterfactual Eligibility Review",
        "",
        "This note compares the frozen baseline mask against the evaluation-only "
        f"counterfactual `{_COUNTERFACTUAL_MODE}`.",
        "",
        "## Surface",
        "",
        f"- checkpoint: `{summary['checkpoint_kind']}`",
        f"- evaluation seed: `{summary['evaluation_seed']}` "
        f"({summary['evaluation_seed_source']})",
        f"- bounded steps: `{summary['limits']['max_steps']}`",
        f"- bounded users: `{summary['limits']['max_users']}`",
        "",
        "## Comparator Meaning",
        "",
        f"- MODQN same-state action-change rate: "
        f"`{modqn_same_state['same_state_action_changed_rate']}`",
        f"- MODQN same-state satellite-change rate: "
        f"`{modqn_same_state['same_state_satellite_changed_rate']}`",
        f"- MODQN overall effect classification: "
        f"`{summary['interpretation']['modqn_change_scope']}`",
        f"- RSS_max same-state action-change rate: "
        f"`{rss_same_state['same_state_action_changed_rate']}`",
        f"- RSS_max same-state satellite-change rate: "
        f"`{rss_same_state['same_state_satellite_changed_rate']}`",
        f"- RSS_max overall effect classification: "
        f"`{summary['interpretation']['rss_max_change_scope']}`",
        f"- MODQN baseline vs counterfactual scalar reward delta: "
        f"`{summary['counterfactual_eval']['modqn']['mean_scalar_reward'] - summary['baseline_eval']['modqn']['mean_scalar_reward']}`",
        f"- RSS_max baseline vs counterfactual scalar reward delta: "
        f"`{summary['counterfactual_eval']['rss_max']['mean_scalar_reward'] - summary['baseline_eval']['rss_max']['mean_scalar_reward']}`",
        "",
        "## Tie Structure",
        "",
        f"- RSS_max tie rate baseline -> counterfactual: "
        f"`{summary['same_state_decision_comparison']['rss_max']['baseline_rss_tie_rate']}` "
        f"-> `{summary['same_state_decision_comparison']['rss_max']['counterfactual_rss_tie_rate']}`",
        f"- MODQN top-1/top-2 same-satellite rate baseline -> counterfactual: "
        f"`{summary['same_state_decision_comparison']['modqn']['baseline_top1_top2_same_satellite_rate']}` "
        f"-> `{summary['same_state_decision_comparison']['modqn']['counterfactual_top1_top2_same_satellite_rate']}`",
        "",
        "## Interpretation",
        "",
        "- A zero satellite-change rate does not automatically mean the effect is trivial.",
        "- If beam-level action changes materially reshape load allocation and weighted reward, "
        "this slice has already moved beyond a cosmetic tie-break.",
    ]


def export_counterfactual_eligibility_eval(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    evaluation_seed: int | None = None,
    max_steps: int | None = None,
    max_users: int | None = None,
) -> dict[str, Any]:
    """Run the Phase 01E2 evaluation-only nearest-beam counterfactual."""
    if max_steps is not None and max_steps < 1:
        raise ValueError(f"max_steps must be >= 1 when provided, got {max_steps!r}")
    if max_users is not None and max_users < 1:
        raise ValueError(f"max_users must be >= 1 when provided, got {max_users!r}")

    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    metadata, cfg, checkpoint_path, checkpoint_kind, checkpoint_payload, objective_weights = (
        _build_trainer_from_artifact(in_dir)
    )

    seeds = metadata["seeds"]
    if evaluation_seed is None:
        selected_seed, selected_seed_source = _select_timeline_seed(
            read_run_metadata(RunArtifactPaths(in_dir).run_metadata_json),
            checkpoint_payload,
            cfg=cfg,
        )
    else:
        selected_seed = int(evaluation_seed)
        selected_seed_source = "cli.override"

    same_state = _same_state_decision_comparison(
        cfg,
        seeds,
        checkpoint_path,
        objective_weights=objective_weights,
        evaluation_seed=selected_seed,
        max_steps=max_steps,
        max_users=max_users,
    )

    baseline_modqn = _rollout_policy(
        cfg,
        seeds,
        checkpoint_path,
        objective_weights=objective_weights,
        evaluation_seed=selected_seed,
        policy="modqn",
        use_counterfactual_mask=False,
        max_steps=max_steps,
        max_users=max_users,
    )
    counter_modqn = _rollout_policy(
        cfg,
        seeds,
        checkpoint_path,
        objective_weights=objective_weights,
        evaluation_seed=selected_seed,
        policy="modqn",
        use_counterfactual_mask=True,
        max_steps=max_steps,
        max_users=max_users,
    )
    baseline_rss = _rollout_policy(
        cfg,
        seeds,
        checkpoint_path,
        objective_weights=objective_weights,
        evaluation_seed=selected_seed,
        policy="rss_max",
        use_counterfactual_mask=False,
        max_steps=max_steps,
        max_users=max_users,
    )
    counter_rss = _rollout_policy(
        cfg,
        seeds,
        checkpoint_path,
        objective_weights=objective_weights,
        evaluation_seed=selected_seed,
        policy="rss_max",
        use_counterfactual_mask=True,
        max_steps=max_steps,
        max_users=max_users,
    )

    summary = {
        "paper_id": metadata["paper_id"],
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_kind": checkpoint_kind,
        "policy_episode": int(checkpoint_payload.get("episode", -1)),
        "evaluation_seed": int(selected_seed),
        "evaluation_seed_source": selected_seed_source,
        "counterfactual_mode": _COUNTERFACTUAL_MODE,
        "objective_weights": [float(value) for value in objective_weights],
        "limits": {
            "max_steps": None if max_steps is None else int(max_steps),
            "max_users": None if max_users is None else int(max_users),
        },
        "same_state_decision_comparison": same_state,
        "baseline_eval": {
            "modqn": baseline_modqn,
            "rss_max": baseline_rss,
        },
        "counterfactual_eval": {
            "modqn": counter_modqn,
            "rss_max": counter_rss,
        },
        "interpretation": {
            "modqn_change_scope": _classify_effect(
                satellite_change_rate=same_state["modqn"]["same_state_satellite_changed_rate"],
                action_change_rate=same_state["modqn"]["same_state_action_changed_rate"],
                baseline_scalar_reward=baseline_modqn["mean_scalar_reward"],
                counterfactual_scalar_reward=counter_modqn["mean_scalar_reward"],
            ),
            "rss_max_change_scope": _classify_effect(
                satellite_change_rate=same_state["rss_max"]["same_state_satellite_changed_rate"],
                action_change_rate=same_state["rss_max"]["same_state_action_changed_rate"],
                baseline_scalar_reward=baseline_rss["mean_scalar_reward"],
                counterfactual_scalar_reward=counter_rss["mean_scalar_reward"],
            ),
        },
    }

    summary_path = write_json(out_dir / "counterfactual_eval_summary.json", summary)
    comparison_rows = _comparison_rows(summary)
    comparison_path = _write_csv(
        out_dir / "counterfactual_vs_baseline.csv",
        comparison_rows,
        fieldnames=_COMPARISON_FIELDS,
    )
    review_path = out_dir / "review.md"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text("\n".join(_review_lines(summary)) + "\n")

    return {
        "counterfactual_eval_summary": summary_path,
        "counterfactual_vs_baseline_csv": comparison_path,
        "review_md": review_path,
        "summary": summary,
    }


__all__ = ["export_counterfactual_eligibility_eval"]
