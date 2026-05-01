"""Phase 05-B bounded-pilot validation and artifact summarization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..config_loader import ConfigValidationError, get_seeds
from ..runtime.trainer_spec import (
    PHASE_05_B_MULTI_CATFISH_KIND,
    R1_REWARD_MODE_THROUGHPUT,
    TrainerConfig,
)


PHASE_05B_CONFIGS = (
    "configs/catfish-modqn-phase-05b-modqn-control.resolved.yaml",
    "configs/catfish-modqn-phase-05b-single-catfish-equal-budget.resolved.yaml",
    "configs/catfish-modqn-phase-05b-primary-multi-catfish-shaping-off.resolved.yaml",
    "configs/catfish-modqn-phase-05b-multi-buffer-single-learner.resolved.yaml",
    "configs/catfish-modqn-phase-05b-random-or-uniform-buffer-control.resolved.yaml",
)
PHASE_05B_EVAL_SEEDS = [100, 200, 300, 400, 500]
OBJECTIVE_SOURCES = ("r1", "r2", "r3")
PROTECTED_CONFIGS = (
    "configs/modqn-paper-baseline.yaml",
    "configs/modqn-paper-baseline.resolved-template.yaml",
)
PROTECTED_ARTIFACT_PREFIXES = (
    "artifacts/pilot-02-best-eval",
    "artifacts/run-9000",
    "artifacts/table-ii-",
    "artifacts/fig-",
)


def validate_phase05b_training_config(
    cfg: dict[str, Any],
    trainer_cfg: TrainerConfig,
) -> dict[str, Any]:
    """Validate the Phase 05-B bounded-pilot contract beyond dataclass checks."""
    if trainer_cfg.training_experiment_kind != PHASE_05_B_MULTI_CATFISH_KIND:
        raise ConfigValidationError(
            "Phase 05B requires training_experiment.kind="
            f"{PHASE_05_B_MULTI_CATFISH_KIND!r}."
        )
    if trainer_cfg.episodes != 20:
        raise ConfigValidationError("Phase 05B bounded run budget must be 20 episodes.")
    if trainer_cfg.target_update_every_episodes != 5:
        raise ConfigValidationError("Phase 05B eval/checkpoint cadence must be 5 episodes.")
    if trainer_cfg.r1_reward_mode != R1_REWARD_MODE_THROUGHPUT:
        raise ConfigValidationError("Phase 05B requires original r1='throughput'.")
    if trainer_cfg.reward_calibration_enabled:
        raise ConfigValidationError("Phase 05B must not enable reward calibration.")
    if trainer_cfg.catfish_competitive_shaping_enabled:
        raise ConfigValidationError("Phase 05B primary/configs keep shaping off.")

    seeds = get_seeds(cfg)
    if seeds["evaluation_seed_set"] != PHASE_05B_EVAL_SEEDS:
        raise ConfigValidationError(
            f"Phase 05B eval seeds must be {PHASE_05B_EVAL_SEEDS}, "
            f"got {seeds['evaluation_seed_set']}."
        )
    phase_block = cfg.get("training_experiment", {}).get("phase_05b_multi_catfish")
    if not isinstance(phase_block, dict):
        raise ConfigValidationError(
            "Phase 05B requires training_experiment.phase_05b_multi_catfish."
        )
    seed_triplets = phase_block.get("seed_triplets", [])
    if len(seed_triplets) < 3:
        raise ConfigValidationError("Phase 05B requires at least 3 seed triplets.")

    reward_surface = phase_block.get("reward_surface", {})
    if reward_surface != {
        "r1": "throughput",
        "r2": "handover penalty",
        "r3": "load balance",
    }:
        raise ConfigValidationError("Phase 05B reward surface was modified.")

    forbidden_terms = ("EE-MODQN", "Catfish-EE-MODQN", "RA-EE", "per-user-ee")
    serialized = json.dumps(cfg, sort_keys=True)
    if any(term in serialized for term in forbidden_terms):
        raise ConfigValidationError(
            "Phase 05B configs must not introduce EE, Catfish-EE, or RA-EE framing."
        )
    return phase_block


def summarize_phase05b_runs(
    *,
    run_dirs: Iterable[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Summarize completed Phase 05-B run artifacts across seed triplets."""
    rows = [_load_run(Path(path)) for path in run_dirs]
    if not rows:
        raise ValueError("summarize_phase05b_runs requires at least one run dir.")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["role"], []).append(row)

    role_summaries = {
        role: _role_summary(role_rows)
        for role, role_rows in sorted(grouped.items())
    }
    protocol = _protocol_summary(rows)
    acceptance = _acceptance_summary(role_summaries)
    summary = {
        "phase": "05B",
        "protocol": protocol,
        "role_summaries": role_summaries,
        "acceptance_checks": acceptance,
        "protected_surface_check": {
            "created_run_dirs_under_allowed_namespace": all(
                str(row["run_dir"]).startswith("artifacts/catfish-modqn-phase-05b-")
                for row in rows
            ),
            "protected_configs_declared_unchanged": list(PROTECTED_CONFIGS),
            "protected_artifact_prefixes_not_written": list(PROTECTED_ARTIFACT_PREFIXES),
        },
        "claim_boundary": {
            "catfish_ee_or_ee_claim_made": False,
            "full_paper_faithful_claim_made": False,
            "scalar_reward_alone_used_as_success": False,
            "phase05r_used_as_effectiveness_claim": False,
        },
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "phase05b_bounded_pilot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["artifact_paths"] = {"summary_json": str(summary_path)}
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _load_run(run_dir: Path) -> dict[str, Any]:
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    training_log = json.loads((run_dir / "training_log.json").read_text(encoding="utf-8"))
    diagnostics_path = run_dir / "catfish_diagnostics.json"
    diagnostics = (
        json.loads(diagnostics_path.read_text(encoding="utf-8"))
        if diagnostics_path.exists()
        else None
    )
    trainer_config = metadata["trainer_config"]
    role = str(trainer_config.get("comparison_role", "not-applicable"))
    final = training_log[-1] if training_log else {}
    best_eval = metadata.get("best_eval_summary") or {}
    return {
        "run_dir": str(run_dir),
        "role": role,
        "method_family": trainer_config.get("method_family"),
        "variant": trainer_config.get("catfish_phase05b_variant"),
        "seed_triplet": [
            metadata["seeds"]["train_seed"],
            metadata["seeds"]["environment_seed"],
            metadata["seeds"]["mobility_seed"],
        ],
        "episodes_completed": metadata["training_summary"]["episodes_completed"],
        "eval_seeds": metadata["seeds"]["evaluation_seed_set"],
        "final": final,
        "auc": _auc(training_log, "scalar_reward"),
        "best_eval": best_eval,
        "final_minus_best_eval_scalar": (
            None
            if not best_eval
            else float(final["scalar_reward"] - best_eval["mean_scalar_reward"])
        ),
        "diagnostics": diagnostics,
    }


def _role_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    final = rows[-1]["final"]
    best_scalars = [
        row["best_eval"].get("mean_scalar_reward")
        for row in rows
        if row["best_eval"]
    ]
    final_scalars = [row["final"]["scalar_reward"] for row in rows]
    diagnostics = [row["diagnostics"] for row in rows if row["diagnostics"]]
    return {
        "run_count": len(rows),
        "seed_triplets": [row["seed_triplet"] for row in rows],
        "episodes_completed": [row["episodes_completed"] for row in rows],
        "final_mean": _metric_means(rows, "final"),
        "final_std": _metric_stds(rows, "final"),
        "auc_mean_scalar": float(np.mean([row["auc"] for row in rows])),
        "best_eval_mean_scalar": _mean_or_none(best_scalars),
        "best_eval_std_scalar": _std_or_none(best_scalars),
        "best_eval_episode_mean": _mean_or_none(
            [
                row["best_eval"].get("episode")
                for row in rows
                if row["best_eval"]
            ]
        ),
        "final_vs_best_gap_mean": _mean_or_none(
            [
                row["final_minus_best_eval_scalar"]
                for row in rows
                if row["final_minus_best_eval_scalar"] is not None
            ]
        ),
        "cross_seed_variance_final_scalar": float(np.var(final_scalars)),
        "representative_final": final,
        "catfish_diagnostics": _diagnostics_summary(diagnostics),
    }


def _metric_means(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    return {
        metric: float(np.mean([row[key][metric] for row in rows]))
        for metric in ("scalar_reward", "r1_mean", "r2_mean", "r3_mean", "total_handovers")
    }


def _metric_stds(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    return {
        metric: float(np.std([row[key][metric] for row in rows]))
        for metric in ("scalar_reward", "r1_mean", "r2_mean", "r3_mean", "total_handovers")
    }


def _diagnostics_summary(diagnostics_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not diagnostics_rows:
        return {"available": False}
    ratios = [
        row.get("cumulative", {}).get("actual_catfish_ratio_in_mixed_updates")
        for row in diagnostics_rows
        if row.get("cumulative", {}).get("actual_catfish_ratio_in_mixed_updates") is not None
    ]
    final = diagnostics_rows[-1]
    q_nan = [
        bool(row.get("cumulative", {}).get("nan_detected"))
        for row in diagnostics_rows
    ]
    return {
        "available": True,
        "actual_catfish_ratio_mean": _mean_or_none(ratios),
        "actual_catfish_ratio_std": _std_or_none(ratios),
        "final_replay": final.get("final_replay", {}),
        "overlap": final.get("overlap", {}),
        "non_target_objective_damage": final.get("non_target_objective_damage", {}),
        "cumulative": final.get("cumulative", {}),
        "replay_starvation": final.get("replay_starvation", {}),
        "runtime_cost": final.get("runtime_cost", {}),
        "any_nan_detected": bool(any(q_nan)),
        "last_episode_action_diversity": (
            final.get("episode_diagnostics", [{}])[-1].get("action_diversity", {})
            if final.get("episode_diagnostics")
            else {}
        ),
        "last_episode_td_loss": (
            final.get("episode_diagnostics", [{}])[-1].get("td_loss", {})
            if final.get("episode_diagnostics")
            else {}
        ),
        "last_episode_q_stability": (
            final.get("episode_diagnostics", [{}])[-1].get("q_stability", {})
            if final.get("episode_diagnostics")
            else {}
        ),
    }


def _protocol_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "bounded_run_budget_episodes": sorted({row["episodes_completed"] for row in rows}),
        "evaluation_seed_sets": sorted({tuple(row["eval_seeds"]) for row in rows}),
        "seed_triplets": sorted({tuple(row["seed_triplet"]) for row in rows}),
        "role_count": len({row["role"] for row in rows}),
        "run_count": len(rows),
    }


def _acceptance_summary(role_summaries: dict[str, Any]) -> dict[str, Any]:
    primary = role_summaries.get("primary-multi-catfish-shaping-off")
    single = role_summaries.get("single-catfish-equal-budget")
    multi_buffer = role_summaries.get("multi-buffer-single-learner")
    random_control = role_summaries.get("random-or-uniform-buffer-control")
    required_roles_complete = all(
        role in role_summaries and role_summaries[role]["run_count"] >= 3
        for role in (
            "modqn-control",
            "single-catfish-equal-budget",
            "primary-multi-catfish-shaping-off",
            "multi-buffer-single-learner",
            "random-or-uniform-buffer-control",
        )
    )
    if not primary or not single:
        return {"required_roles_complete": required_roles_complete, "pass": False}

    primary_final = primary["final_mean"]
    single_final = single["final_mean"]
    component_improvements = {
        key: primary_final[key] > single_final[key]
        for key in ("r1_mean", "r2_mean", "r3_mean")
    }
    primary_diag = primary["catfish_diagnostics"]
    ratio = primary_diag.get("actual_catfish_ratio_mean")
    ratio_near = ratio is not None and abs(float(ratio) - 0.30) <= 0.03
    multi_buffer_explains = (
        bool(multi_buffer)
        and multi_buffer["best_eval_mean_scalar"] is not None
        and primary["best_eval_mean_scalar"] is not None
        and multi_buffer["best_eval_mean_scalar"] >= primary["best_eval_mean_scalar"]
    )
    random_explains = (
        bool(random_control)
        and random_control["best_eval_mean_scalar"] is not None
        and primary["best_eval_mean_scalar"] is not None
        and random_control["best_eval_mean_scalar"] >= primary["best_eval_mean_scalar"]
    )
    primary_beats_single_scalar = (
        primary["best_eval_mean_scalar"] is not None
        and single["best_eval_mean_scalar"] is not None
        and primary["best_eval_mean_scalar"] > single["best_eval_mean_scalar"]
    )
    objective_not_scalar_only = primary_beats_single_scalar and any(
        component_improvements.values()
    )
    replay_starvation = _has_replay_starvation(
        primary_diag.get("replay_starvation", {})
    )
    action_collapse = bool(
        primary_diag.get("last_episode_action_diversity", {}).get(
            "action_collapse_detected"
        )
    )
    stop_conditions = {
        "nan_detected": bool(primary_diag.get("any_nan_detected")),
        "replay_starvation_detected": bool(replay_starvation),
        "action_collapse_detected": bool(action_collapse),
        "multi_buffer_single_learner_matches_or_exceeds_primary": bool(
            multi_buffer_explains
        ),
        "random_buffer_control_matches_or_exceeds_primary": bool(random_explains),
        "actual_intervention_budget_unmatched": not ratio_near,
    }
    return {
        "required_roles_complete": required_roles_complete,
        "actual_catfish_ratio_near_0_30": bool(ratio_near),
        "primary_beats_single_scalar": bool(primary_beats_single_scalar),
        "component_improvements_vs_single": component_improvements,
        "not_scalar_reward_alone": bool(objective_not_scalar_only),
        "multi_buffer_control_explains_gain": bool(multi_buffer_explains),
        "random_control_explains_gain": bool(random_explains),
        "stop_conditions": stop_conditions,
        "pass": bool(
            required_roles_complete
            and ratio_near
            and objective_not_scalar_only
            and not any(stop_conditions.values())
        ),
    }


def _auc(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return float(np.mean(values))


def _mean_or_none(values: Iterable[Any]) -> float | None:
    vals = [float(value) for value in values if value is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def _std_or_none(values: Iterable[Any]) -> float | None:
    vals = [float(value) for value in values if value is not None]
    if not vals:
        return None
    return float(np.std(vals))


def _has_replay_starvation(payload: dict[str, Any]) -> bool:
    if not payload:
        return False
    scalar_keys = (
        "catfish_replay_starved_intervention_cumulative",
        "catfish_replay_starved_training_cumulative",
        "main_replay_starved_updates_cumulative",
    )
    if any(int(payload.get(key, 0) or 0) > 0 for key in scalar_keys):
        return True
    mapping_keys = (
        "source_replay_starved_training_cumulative",
        "source_replay_starved_intervention_cumulative",
    )
    for key in mapping_keys:
        values = payload.get(key, {})
        if isinstance(values, dict) and any(int(value or 0) > 0 for value in values.values()):
            return True
    return bool(payload.get("any_objective_replay_empty_after_warmup"))
