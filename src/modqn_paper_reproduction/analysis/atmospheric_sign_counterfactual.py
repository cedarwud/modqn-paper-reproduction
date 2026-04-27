"""Evaluation-only atmospheric-sign counterfactual helpers for Phase 01G."""

from __future__ import annotations

import contextlib
import copy
import csv
import io
from pathlib import Path
from typing import Any

import numpy as np

from ..algorithms.modqn import MODQNTrainer
from ..artifacts import RunArtifactPaths, read_run_metadata
from ..artifacts.compat import (
    resolve_training_config_snapshot,
    select_replay_checkpoint,
)
from ..config_loader import build_environment, build_trainer_config
from ..env.channel import AtmosphericSignMode
from ..env.step import ActionMask, UserState
from ._common import safe_abs_scale, write_json
from .reward_geometry import collect_reward_diagnostics

_PUBLISHED_MODE = AtmosphericSignMode.PAPER_PUBLISHED
_CORRECTED_MODE = AtmosphericSignMode.CORRECTED_LOSSY

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
    "same_geometry_action_changed_rate",
    "same_geometry_satellite_changed_rate",
]
_DIAGNOSTIC_FIELDS = [
    "mode",
    "zenith_snr_db",
    "zenith_throughput_bps",
    "sample_r1",
    "sample_r2_beam_change",
    "sample_r2_sat_change",
    "sample_r3",
    "r1_r2_ratio",
    "r1_r3_ratio",
]


def _silence_stderr():
    return contextlib.redirect_stderr(io.StringIO())


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str],
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return target


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _fraction_or_none(values: list[bool]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


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
) -> int:
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        return 0
    valid_channels = state.channel_quality[valid].astype(np.float64, copy=False)
    max_channel = float(np.max(valid_channels))
    tied = np.isclose(valid_channels, max_channel, rtol=1e-6, atol=1e-12)
    return int(valid[tied][0])


def _override_atmospheric_sign(
    cfg: dict[str, Any],
    mode: AtmosphericSignMode,
) -> dict[str, Any]:
    updated = copy.deepcopy(cfg)
    assumptions = updated.setdefault("resolved_assumptions", {})
    if not isinstance(assumptions, dict):
        raise ValueError("resolved_assumptions must be a mapping for atmospheric override")
    block = assumptions.setdefault(
        "atmospheric_formula_sign",
        {"assumption_id": "ASSUME-MODQN-REP-009", "value": {}},
    )
    if not isinstance(block, dict):
        raise ValueError("resolved_assumptions.atmospheric_formula_sign must be a mapping")
    value = block.setdefault("value", {})
    if not isinstance(value, dict):
        raise ValueError(
            "resolved_assumptions.atmospheric_formula_sign.value must be a mapping"
        )

    if mode is _CORRECTED_MODE:
        value["primary_run_formula"] = "corrected-lossy-sign"
        value["primary_expression"] = "A_corrected(d) = 10^(-3 d chi / (10 h))"
        value["sign_anomaly_disclosure"] = (
            "evaluation-only-corrected-lossy-sign-counterfactual"
        )
        value["required_sensitivity_formula"] = value["primary_expression"]
    else:
        value["primary_run_formula"] = "paper-published-sign"
        value["primary_expression"] = "A(d) = 10^(+3 d chi / (10 h))"
        value["sign_anomaly_disclosure"] = (
            "published-sign-yields-gain-for-typical-values"
        )
        value.setdefault(
            "required_sensitivity_formula",
            "A_corrected(d) = 10^(-3 d chi / (10 h))",
        )
    return updated


def _build_artifact_context(
    input_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, str, tuple[float, float, float]]:
    metadata = read_run_metadata(RunArtifactPaths(input_dir).run_metadata_json)
    cfg = resolve_training_config_snapshot(metadata, artifact_dir=input_dir)
    seeds = metadata.seeds.to_dict()

    trainer = MODQNTrainer(
        env=build_environment(copy.deepcopy(cfg)),
        config=build_trainer_config(cfg),
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
        for value in checkpoint_payload.get("trainer_config", {}).get(
            "objective_weights",
            trainer.config.objective_weights,
        )
    )
    return (
        metadata.to_dict(),
        cfg,
        checkpoint_path,
        checkpoint_kind,
        objective_weights,
    )


def _build_rng_pair(evaluation_seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    state = np.random.SeedSequence(int(evaluation_seed)).generate_state(
        2, dtype=np.uint64
    )
    return (
        np.random.default_rng(int(state[0])),
        np.random.default_rng(int(state[1])),
    )


def _policy_actions(
    trainer: MODQNTrainer,
    states: list[UserState],
    masks: list[ActionMask],
    *,
    policy: str,
    objective_weights: tuple[float, float, float],
) -> np.ndarray:
    if policy == "modqn":
        encoded = trainer.encode_states(states)
        actions, _diag = trainer.select_actions_with_diagnostics(
            encoded,
            masks,
            objective_weights=objective_weights,
            top_k=2,
        )
        return actions
    if policy != "rss_max":
        raise ValueError(f"Unsupported policy {policy!r}")
    actions = np.zeros(len(masks), dtype=np.int32)
    for uid, mask in enumerate(masks):
        actions[uid] = _rss_selection(states[uid], mask.mask)
    return actions


def _rollout_policy(
    cfg: dict[str, Any],
    seeds: dict[str, Any],
    checkpoint_path: Path,
    *,
    mode: AtmosphericSignMode,
    objective_weights: tuple[float, float, float],
    evaluation_seed: int,
    policy: str,
    max_steps: int | None,
    max_users: int | None,
) -> dict[str, Any]:
    active_cfg = _override_atmospheric_sign(cfg, mode)
    trainer = MODQNTrainer(
        env=build_environment(copy.deepcopy(active_cfg)),
        config=build_trainer_config(active_cfg),
        train_seed=int(seeds["train_seed"]),
        env_seed=int(seeds["environment_seed"]),
        mobility_seed=int(seeds["mobility_seed"]),
    )
    trainer.load_checkpoint(checkpoint_path, load_optimizers=False)

    env_rng, mobility_rng = _build_rng_pair(evaluation_seed)
    with _silence_stderr():
        states, masks, _diag = trainer.env.reset(env_rng, mobility_rng)

    users_total = len(states)
    users_to_score = users_total if max_users is None else min(int(max_users), users_total)
    steps_audited = 0
    ep_reward = np.zeros(3, dtype=np.float64)
    ep_handovers = 0

    while True:
        if max_steps is not None and steps_audited >= int(max_steps):
            break

        actions_prefix = _policy_actions(
            trainer,
            states[:users_to_score],
            masks[:users_to_score],
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

        steps_audited += 1
        if result.done:
            break
        states = result.user_states
        masks = result.action_masks

    avg_reward = ep_reward / max(users_to_score, 1)
    scalar = float(np.dot(objective_weights, avg_reward))
    return {
        "policy": policy,
        "atmospheric_sign_mode": mode.value,
        "evaluation_seed": int(evaluation_seed),
        "steps_audited": int(steps_audited),
        "users_scored_per_step": int(users_to_score),
        "mean_scalar_reward": scalar,
        "mean_r1": float(avg_reward[0]),
        "mean_r2": float(avg_reward[1]),
        "mean_r3": float(avg_reward[2]),
        "mean_total_handovers": float(ep_handovers),
    }


def _same_geometry_decision_comparison(
    cfg: dict[str, Any],
    seeds: dict[str, Any],
    checkpoint_path: Path,
    *,
    objective_weights: tuple[float, float, float],
    evaluation_seed: int,
    max_steps: int | None,
    max_users: int | None,
) -> dict[str, Any]:
    published_cfg = _override_atmospheric_sign(cfg, _PUBLISHED_MODE)
    corrected_cfg = _override_atmospheric_sign(cfg, _CORRECTED_MODE)

    published_trainer = MODQNTrainer(
        env=build_environment(copy.deepcopy(published_cfg)),
        config=build_trainer_config(published_cfg),
        train_seed=int(seeds["train_seed"]),
        env_seed=int(seeds["environment_seed"]),
        mobility_seed=int(seeds["mobility_seed"]),
    )
    corrected_trainer = MODQNTrainer(
        env=build_environment(copy.deepcopy(corrected_cfg)),
        config=build_trainer_config(corrected_cfg),
        train_seed=int(seeds["train_seed"]),
        env_seed=int(seeds["environment_seed"]),
        mobility_seed=int(seeds["mobility_seed"]),
    )
    published_trainer.load_checkpoint(checkpoint_path, load_optimizers=False)
    corrected_trainer.load_checkpoint(checkpoint_path, load_optimizers=False)

    published_env_rng, published_mobility_rng = _build_rng_pair(evaluation_seed)
    corrected_env_rng, corrected_mobility_rng = _build_rng_pair(evaluation_seed)
    with _silence_stderr():
        published_states, published_masks, _ = published_trainer.env.reset(
            published_env_rng,
            published_mobility_rng,
        )
        corrected_states, corrected_masks, _ = corrected_trainer.env.reset(
            corrected_env_rng,
            corrected_mobility_rng,
        )

    users_total = len(published_states)
    users_to_compare = users_total if max_users is None else min(int(max_users), users_total)
    steps_audited = 0
    beams_per_satellite = published_trainer.env.beam_pattern.num_beams

    metrics: dict[str, dict[str, list[bool]]] = {
        "modqn": {"action_changed": [], "satellite_changed": []},
        "rss_max": {"action_changed": [], "satellite_changed": []},
    }

    while True:
        if max_steps is not None and steps_audited >= int(max_steps):
            break

        for policy in ("modqn", "rss_max"):
            published_actions = _policy_actions(
                published_trainer,
                published_states[:users_to_compare],
                published_masks[:users_to_compare],
                policy=policy,
                objective_weights=objective_weights,
            )
            corrected_actions = _policy_actions(
                corrected_trainer,
                corrected_states[:users_to_compare],
                corrected_masks[:users_to_compare],
                policy=policy,
                objective_weights=objective_weights,
            )
            for uid in range(users_to_compare):
                published_action = int(published_actions[uid])
                corrected_action = int(corrected_actions[uid])
                metrics[policy]["action_changed"].append(
                    published_action != corrected_action
                )
                metrics[policy]["satellite_changed"].append(
                    _action_satellite_index(published_action, beams_per_satellite)
                    != _action_satellite_index(corrected_action, beams_per_satellite)
                )

        published_step_actions = _policy_actions(
            published_trainer,
            published_states,
            published_masks,
            policy="modqn",
            objective_weights=objective_weights,
        )
        published_result = published_trainer.env.step(
            published_step_actions,
            published_env_rng,
        )
        corrected_result = corrected_trainer.env.step(
            published_step_actions,
            corrected_env_rng,
        )

        steps_audited += 1
        if published_result.done or corrected_result.done:
            break
        published_states = published_result.user_states
        published_masks = published_result.action_masks
        corrected_states = corrected_result.user_states
        corrected_masks = corrected_result.action_masks

    return {
        "evaluation_seed": int(evaluation_seed),
        "steps_audited": int(steps_audited),
        "users_compared_per_step": int(users_to_compare),
        "modqn": {
            "same_geometry_action_changed_rate": _fraction_or_none(
                metrics["modqn"]["action_changed"]
            ),
            "same_geometry_satellite_changed_rate": _fraction_or_none(
                metrics["modqn"]["satellite_changed"]
            ),
        },
        "rss_max": {
            "same_geometry_action_changed_rate": _fraction_or_none(
                metrics["rss_max"]["action_changed"]
            ),
            "same_geometry_satellite_changed_rate": _fraction_or_none(
                metrics["rss_max"]["satellite_changed"]
            ),
        },
    }


def _diagnostic_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"mode": "paper-published", **summary["diagnostics"]["paper_published"]},
        {"mode": "corrected-lossy", **summary["diagnostics"]["corrected_lossy"]},
    ]


def _comparison_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in ("modqn", "rss_max"):
        baseline = summary["baseline_eval"][policy]
        counter = summary["counterfactual_eval"][policy]
        compare = summary["same_geometry_decision_comparison"][policy]
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
                "same_geometry_action_changed_rate": (
                    compare["same_geometry_action_changed_rate"]
                ),
                "same_geometry_satellite_changed_rate": (
                    compare["same_geometry_satellite_changed_rate"]
                ),
            }
        )
    return rows


def _classify_policy_effect(
    *,
    baseline_scalar_reward: float,
    counterfactual_scalar_reward: float,
    action_change_rate: float | None,
    satellite_change_rate: float | None,
) -> str:
    scalar_delta = abs(counterfactual_scalar_reward - baseline_scalar_reward)
    material_delta_floor = max(abs(baseline_scalar_reward) * 0.10, 10.0)

    if satellite_change_rate is not None and satellite_change_rate >= 0.25:
        return "material"
    if scalar_delta >= material_delta_floor:
        return "material"
    if action_change_rate is not None and action_change_rate > 0.0:
        return "notable"
    return "absent"


def _classify_diagnostics_change(
    published: dict[str, Any],
    corrected: dict[str, Any],
) -> str:
    throughput_ratio = (
        safe_abs_scale(corrected["sample_r1"]) / safe_abs_scale(published["sample_r1"])
    )
    r1_r2_ratio = (
        safe_abs_scale(corrected["r1_r2_ratio"])
        / safe_abs_scale(published["r1_r2_ratio"])
    )
    r1_r3_ratio = (
        safe_abs_scale(corrected["r1_r3_ratio"])
        / safe_abs_scale(published["r1_r3_ratio"])
    )

    if (
        throughput_ratio <= 0.75
        or throughput_ratio >= 1.25
        or r1_r2_ratio <= 0.75
        or r1_r2_ratio >= 1.25
        or r1_r3_ratio <= 0.75
        or r1_r3_ratio >= 1.25
    ):
        return "material"
    if (
        throughput_ratio <= 0.95
        or throughput_ratio >= 1.05
        or r1_r2_ratio <= 0.95
        or r1_r2_ratio >= 1.05
        or r1_r3_ratio <= 0.95
        or r1_r3_ratio >= 1.05
    ):
        return "notable"
    return "absent"


def _review_lines(summary: dict[str, Any]) -> list[str]:
    published = summary["diagnostics"]["paper_published"]
    corrected = summary["diagnostics"]["corrected_lossy"]
    ratios = summary["diagnostics"]["comparison"]
    modqn_change = summary["interpretation"]["modqn_change_scope"]
    rss_change = summary["interpretation"]["rss_max_change_scope"]
    return [
        "# Atmospheric-Sign Counterfactual Review",
        "",
        "This note replays the preserved checkpoint without retraining under the "
        "paper-published atmospheric sign and the corrected lossy sign.",
        "",
        "## Reward Geometry",
        "",
        f"- published sample r1: `{published['sample_r1']}`",
        f"- corrected sample r1: `{corrected['sample_r1']}`",
        f"- sample r1 ratio corrected/published: `{ratios['sample_r1_ratio_corrected_vs_published']}`",
        f"- published |r1|/|r2| ratio: `{published['r1_r2_ratio']}`",
        f"- corrected |r1|/|r2| ratio: `{corrected['r1_r2_ratio']}`",
        f"- published |r1|/|r3| ratio: `{published['r1_r3_ratio']}`",
        f"- corrected |r1|/|r3| ratio: `{corrected['r1_r3_ratio']}`",
        "",
        "## Policy Effect",
        "",
        f"- MODQN change scope: `{modqn_change}`",
        f"- RSS_max change scope: `{rss_change}`",
        f"- diagnostics change scope: `{summary['interpretation']['diagnostics_change_scope']}`",
        "",
        "## Interpretation",
        "",
        "- This surface is evaluation-only and does not authorize a new long training run by itself.",
        "- If the corrected sign remains material after beam-aware eligibility, the next follow-on should be a disclosed combined semantics surface rather than a blind episode increase.",
    ]


def export_atmospheric_sign_counterfactual_eval(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    evaluation_seed: int | None = None,
    max_steps: int | None = None,
    max_users: int | None = None,
) -> dict[str, Any]:
    """Replay a preserved checkpoint under published vs corrected atmospheric sign."""
    artifact_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata, cfg, checkpoint_path, checkpoint_kind, objective_weights = (
        _build_artifact_context(artifact_dir)
    )
    seeds = metadata["seeds"]
    if evaluation_seed is not None:
        chosen_seed = int(evaluation_seed)
    else:
        best_eval_summary = metadata.get("best_eval_summary", {})
        best_eval_seeds = (
            best_eval_summary.get("eval_seeds", [])
            if isinstance(best_eval_summary, dict)
            else []
        )
        evaluation_seed_set = seeds.get("evaluation_seed_set", [])
        if best_eval_seeds:
            chosen_seed = int(best_eval_seeds[0])
        elif evaluation_seed_set:
            chosen_seed = int(evaluation_seed_set[0])
        else:
            chosen_seed = int(seeds["train_seed"])

    published_cfg = _override_atmospheric_sign(cfg, _PUBLISHED_MODE)
    corrected_cfg = _override_atmospheric_sign(cfg, _CORRECTED_MODE)
    with _silence_stderr():
        published_diag = collect_reward_diagnostics(copy.deepcopy(published_cfg))
        corrected_diag = collect_reward_diagnostics(copy.deepcopy(corrected_cfg))

    summary = {
        "input_run_dir": str(artifact_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_kind": checkpoint_kind,
        "evaluation_seed": chosen_seed,
        "baseline_mode": _PUBLISHED_MODE.value,
        "counterfactual_mode": _CORRECTED_MODE.value,
        "baseline_eval": {
            policy: _rollout_policy(
                cfg,
                seeds,
                checkpoint_path,
                mode=_PUBLISHED_MODE,
                objective_weights=objective_weights,
                evaluation_seed=chosen_seed,
                policy=policy,
                max_steps=max_steps,
                max_users=max_users,
            )
            for policy in ("modqn", "rss_max")
        },
        "counterfactual_eval": {
            policy: _rollout_policy(
                cfg,
                seeds,
                checkpoint_path,
                mode=_CORRECTED_MODE,
                objective_weights=objective_weights,
                evaluation_seed=chosen_seed,
                policy=policy,
                max_steps=max_steps,
                max_users=max_users,
            )
            for policy in ("modqn", "rss_max")
        },
        "same_geometry_decision_comparison": _same_geometry_decision_comparison(
            cfg,
            seeds,
            checkpoint_path,
            objective_weights=objective_weights,
            evaluation_seed=chosen_seed,
            max_steps=max_steps,
            max_users=max_users,
        ),
        "diagnostics": {
            "paper_published": published_diag,
            "corrected_lossy": corrected_diag,
            "comparison": {
                "sample_r1_ratio_corrected_vs_published": (
                    safe_abs_scale(corrected_diag["sample_r1"])
                    / safe_abs_scale(published_diag["sample_r1"])
                ),
                "r1_r2_ratio_corrected_vs_published": (
                    safe_abs_scale(corrected_diag["r1_r2_ratio"])
                    / safe_abs_scale(published_diag["r1_r2_ratio"])
                ),
                "r1_r3_ratio_corrected_vs_published": (
                    safe_abs_scale(corrected_diag["r1_r3_ratio"])
                    / safe_abs_scale(published_diag["r1_r3_ratio"])
                ),
            },
        },
    }
    summary["interpretation"] = {
        "diagnostics_change_scope": _classify_diagnostics_change(
            published_diag,
            corrected_diag,
        ),
        "modqn_change_scope": _classify_policy_effect(
            baseline_scalar_reward=summary["baseline_eval"]["modqn"]["mean_scalar_reward"],
            counterfactual_scalar_reward=summary["counterfactual_eval"]["modqn"][
                "mean_scalar_reward"
            ],
            action_change_rate=summary["same_geometry_decision_comparison"]["modqn"][
                "same_geometry_action_changed_rate"
            ],
            satellite_change_rate=summary["same_geometry_decision_comparison"]["modqn"][
                "same_geometry_satellite_changed_rate"
            ],
        ),
        "rss_max_change_scope": _classify_policy_effect(
            baseline_scalar_reward=summary["baseline_eval"]["rss_max"]["mean_scalar_reward"],
            counterfactual_scalar_reward=summary["counterfactual_eval"]["rss_max"][
                "mean_scalar_reward"
            ],
            action_change_rate=summary["same_geometry_decision_comparison"]["rss_max"][
                "same_geometry_action_changed_rate"
            ],
            satellite_change_rate=summary["same_geometry_decision_comparison"]["rss_max"][
                "same_geometry_satellite_changed_rate"
            ],
        ),
    }

    summary_path = write_json(
        out_dir / "atmospheric_sign_counterfactual_summary.json",
        summary,
    )
    comparison_csv_path = _write_csv(
        out_dir / "atmospheric_sign_vs_baseline.csv",
        _comparison_rows(summary),
        fieldnames=_COMPARISON_FIELDS,
    )
    diagnostics_csv_path = _write_csv(
        out_dir / "reward_geometry_diagnostics_comparison.csv",
        _diagnostic_rows(summary),
        fieldnames=_DIAGNOSTIC_FIELDS,
    )
    review_path = out_dir / "review.md"
    review_path.write_text("\n".join(_review_lines(summary)) + "\n")

    return {
        "summary_path": summary_path,
        "comparison_csv_path": comparison_csv_path,
        "diagnostics_csv_path": diagnostics_csv_path,
        "review_path": review_path,
        "summary": summary,
    }


__all__ = ["export_atmospheric_sign_counterfactual_eval"]
