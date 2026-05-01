"""Phase 07-B single-Catfish intervention utility pilot helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..config_loader import ConfigValidationError, get_seeds
from ..runtime.trainer_spec import (
    PHASE_07_B_SINGLE_CATFISH_UTILITY_KIND,
    R1_REWARD_MODE_THROUGHPUT,
    TrainerConfig,
)


PHASE_07B_CONFIGS = (
    "configs/catfish-modqn-phase-07b-modqn-control.resolved.yaml",
    "configs/catfish-modqn-phase-07b-single-catfish-primary-shaping-off.resolved.yaml",
    "configs/catfish-modqn-phase-07b-no-intervention.resolved.yaml",
    "configs/catfish-modqn-phase-07b-random-equal-budget-injection.resolved.yaml",
    "configs/catfish-modqn-phase-07b-replay-only-single-learner.resolved.yaml",
    "configs/catfish-modqn-phase-07b-no-asymmetric-gamma.resolved.yaml",
)
PHASE_07B_OPTIONAL_CONFIGS = (
    "configs/catfish-modqn-phase-07b-shaping-on-ablation.resolved.yaml",
)
PHASE_07B_EVAL_SEEDS = [100, 200, 300, 400, 500]
PHASE_07B_REQUIRED_ROLES = (
    "matched-modqn-control",
    "single-catfish-primary-shaping-off",
    "no-intervention",
    "random-equal-budget-injection",
    "replay-only-single-learner",
    "no-asymmetric-gamma",
)
PROTECTED_CONFIGS = (
    "configs/modqn-paper-baseline.yaml",
    "configs/modqn-paper-baseline.resolved-template.yaml",
)


def validate_phase07b_training_config(
    cfg: dict[str, Any],
    trainer_cfg: TrainerConfig,
) -> dict[str, Any]:
    """Validate the Phase 07-B bounded-pilot contract beyond dataclass checks."""
    if trainer_cfg.training_experiment_kind != PHASE_07_B_SINGLE_CATFISH_UTILITY_KIND:
        raise ConfigValidationError(
            "Phase 07B requires training_experiment.kind="
            f"{PHASE_07_B_SINGLE_CATFISH_UTILITY_KIND!r}."
        )
    if trainer_cfg.episodes != 20:
        raise ConfigValidationError("Phase 07B bounded run budget must be 20 episodes.")
    if trainer_cfg.target_update_every_episodes != 5:
        raise ConfigValidationError(
            "Phase 07B eval/checkpoint cadence must be 5 episodes."
        )
    if trainer_cfg.r1_reward_mode != R1_REWARD_MODE_THROUGHPUT:
        raise ConfigValidationError("Phase 07B requires original r1='throughput'.")
    if trainer_cfg.reward_calibration_enabled:
        raise ConfigValidationError("Phase 07B must not enable reward calibration.")
    if trainer_cfg.catfish_competitive_shaping_enabled:
        raise ConfigValidationError("Phase 07B primary/configs keep shaping off.")

    seeds = get_seeds(cfg)
    if seeds["evaluation_seed_set"] != PHASE_07B_EVAL_SEEDS:
        raise ConfigValidationError(
            f"Phase 07B eval seeds must be {PHASE_07B_EVAL_SEEDS}, "
            f"got {seeds['evaluation_seed_set']}."
        )
    phase_block = cfg.get("training_experiment", {}).get(
        "phase_07b_catfish_utility"
    )
    if not isinstance(phase_block, dict):
        raise ConfigValidationError(
            "Phase 07B requires training_experiment.phase_07b_catfish_utility."
        )
    seed_triplets = phase_block.get("seed_triplets", [])
    if len(seed_triplets) < 3:
        raise ConfigValidationError("Phase 07B requires at least 3 seed triplets.")

    reward_surface = phase_block.get("reward_surface", {})
    expected_reward = {
        "r1": "throughput",
        "r2": "handover penalty",
        "r3": "load balance",
    }
    if reward_surface != expected_reward:
        raise ConfigValidationError("Phase 07B reward surface was modified.")

    forbidden_terms = ("EE-MODQN", "Catfish-EE-MODQN", "RA-EE", "per-user-ee")
    serialized = json.dumps(cfg, sort_keys=True)
    if any(term in serialized for term in forbidden_terms):
        raise ConfigValidationError(
            "Phase 07B configs must not introduce EE, Catfish-EE, or RA-EE framing."
        )
    return phase_block


def summarize_phase07b_runs(
    *,
    run_dirs: Iterable[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Summarize completed Phase 07-B run artifacts across seed triplets."""
    rows = [_load_run(Path(path)) for path in run_dirs]
    if not rows:
        raise ValueError("summarize_phase07b_runs requires at least one run dir.")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["role"], []).append(row)

    role_summaries = {
        role: _role_summary(role_rows)
        for role, role_rows in sorted(grouped.items())
    }
    summary = {
        "phase": "07B",
        "protocol": _protocol_summary(rows),
        "role_summaries": role_summaries,
        "comparator_deltas": _comparator_deltas(role_summaries),
        "acceptance_checks": _acceptance_summary(role_summaries),
        "protected_surface_check": {
            "created_run_dirs_under_allowed_namespace": all(
                str(row["run_dir"]).startswith("artifacts/catfish-modqn-phase-07b-")
                for row in rows
            ),
            "protected_configs_declared_unchanged": list(PROTECTED_CONFIGS),
        },
        "claim_boundary": {
            "catfish_ee_or_ee_claim_made": False,
            "phase06_claim_made": False,
            "multi_catfish_redesign_performed": False,
            "full_paper_faithful_claim_made": False,
            "scalar_reward_alone_used_as_success": False,
            "phase05b_negative_result_used_as_effectiveness": False,
        },
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "phase07b_bounded_pilot_summary.json"
    summary["artifact_paths"] = {"summary_json": str(summary_path)}
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
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
    final = training_log[-1] if training_log else {}
    best_eval = metadata.get("best_eval_summary") or {}
    return {
        "run_dir": str(run_dir),
        "role": trainer_config.get("comparison_role", "not-applicable"),
        "method_family": trainer_config.get("method_family"),
        "variant": trainer_config.get("catfish_phase07b_variant"),
        "seed_triplet": [
            metadata["seeds"]["train_seed"],
            metadata["seeds"]["environment_seed"],
            metadata["seeds"]["mobility_seed"],
        ],
        "episodes_completed": metadata["training_summary"]["episodes_completed"],
        "elapsed_s": metadata["training_summary"]["elapsed_s"],
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
    final_scalars = [row["final"]["scalar_reward"] for row in rows]
    best_scalars = [
        row["best_eval"].get("mean_scalar_reward")
        for row in rows
        if row["best_eval"]
    ]
    diagnostics = [row["diagnostics"] for row in rows if row["diagnostics"]]
    return {
        "run_count": len(rows),
        "seed_triplets": [row["seed_triplet"] for row in rows],
        "episodes_completed": [row["episodes_completed"] for row in rows],
        "runtime_s_mean": float(np.mean([row["elapsed_s"] for row in rows])),
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
        for metric in (
            "scalar_reward",
            "r1_mean",
            "r2_mean",
            "r3_mean",
            "total_handovers",
        )
    }


def _metric_stds(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    return {
        metric: float(np.std([row[key][metric] for row in rows]))
        for metric in (
            "scalar_reward",
            "r1_mean",
            "r2_mean",
            "r3_mean",
            "total_handovers",
        )
    }


def _diagnostics_summary(diagnostics_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not diagnostics_rows:
        return {"available": False}
    ratios = [
        row.get("cumulative", {}).get("actual_injected_ratio_in_mixed_updates")
        for row in diagnostics_rows
        if row.get("cumulative", {}).get("actual_injected_ratio_in_mixed_updates")
        is not None
    ]
    windows = [
        int(row.get("intervention_utility", {}).get("window_count", 0))
        for row in diagnostics_rows
    ]
    final = diagnostics_rows[-1]
    q_nan = [
        bool(row.get("cumulative", {}).get("nan_detected"))
        for row in diagnostics_rows
    ]
    return {
        "available": True,
        "actual_injected_ratio_mean": _mean_or_none(ratios),
        "actual_injected_ratio_std": _std_or_none(ratios),
        "intervention_window_count_mean": _mean_or_none(windows),
        "final_replay": final.get("final_replay", {}),
        "sample_lineage_summary": final.get("final_replay", {}).get(
            "sample_lineage_summary",
            {},
        ),
        "intervention_utility": final.get("intervention_utility", {}),
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


def _comparator_deltas(role_summaries: dict[str, Any]) -> dict[str, Any]:
    primary = role_summaries.get("single-catfish-primary-shaping-off")
    if not primary:
        return {}
    primary_final = primary["final_mean"]
    deltas: dict[str, Any] = {}
    for role, summary in role_summaries.items():
        if role == "single-catfish-primary-shaping-off":
            continue
        final = summary["final_mean"]
        deltas[role] = {
            "scalar_delta": primary_final["scalar_reward"] - final["scalar_reward"],
            "r1_delta": primary_final["r1_mean"] - final["r1_mean"],
            "r2_delta": primary_final["r2_mean"] - final["r2_mean"],
            "r3_delta": primary_final["r3_mean"] - final["r3_mean"],
            "handover_delta": (
                primary_final["total_handovers"] - final["total_handovers"]
            ),
        }
    return deltas


def _acceptance_summary(role_summaries: dict[str, Any]) -> dict[str, Any]:
    required_roles_complete = all(
        role in role_summaries and role_summaries[role]["run_count"] >= 3
        for role in PHASE_07B_REQUIRED_ROLES
    )
    primary = role_summaries.get("single-catfish-primary-shaping-off")
    if not primary:
        return {"required_roles_complete": required_roles_complete, "pass": False}

    required_controls = (
        "no-intervention",
        "random-equal-budget-injection",
        "replay-only-single-learner",
    )
    primary_final = primary["final_mean"]
    scalar_beats_required_controls = all(
        role in role_summaries
        and primary_final["scalar_reward"]
        > role_summaries[role]["final_mean"]["scalar_reward"]
        for role in required_controls
    )
    component_or_stability_improvement = any(
        role in role_summaries
        and (
            primary_final["r1_mean"] > role_summaries[role]["final_mean"]["r1_mean"]
            or primary_final["r2_mean"] > role_summaries[role]["final_mean"]["r2_mean"]
            or primary_final["r3_mean"] > role_summaries[role]["final_mean"]["r3_mean"]
            or primary["final_vs_best_gap_mean"]
            is not None
            and role_summaries[role]["final_vs_best_gap_mean"] is not None
            and abs(primary["final_vs_best_gap_mean"])
            < abs(role_summaries[role]["final_vs_best_gap_mean"])
        )
        for role in required_controls
    )
    primary_diag = primary["catfish_diagnostics"]
    ratio = primary_diag.get("actual_injected_ratio_mean")
    ratio_near = ratio is not None and abs(float(ratio) - 0.30) <= 0.03
    intervention_windows_present = (
        primary_diag.get("intervention_window_count_mean") is not None
        and float(primary_diag["intervention_window_count_mean"]) > 0
    )
    starvation = primary_diag.get("replay_starvation", {})
    starvation_triggered = any(
        int(starvation.get(key, 0) or 0) > 0
        for key in (
            "main_replay_starved_updates_cumulative",
            "catfish_replay_starved_intervention_cumulative",
        )
    )
    no_nan = not bool(primary_diag.get("any_nan_detected"))

    scalar_only = scalar_beats_required_controls and not component_or_stability_improvement
    pass_ = all(
        [
            required_roles_complete,
            scalar_beats_required_controls,
            component_or_stability_improvement,
            ratio_near,
            intervention_windows_present,
            no_nan,
            not starvation_triggered,
            not scalar_only,
        ]
    )
    return {
        "required_roles_complete": required_roles_complete,
        "primary_beats_no_intervention_random_and_replay_only": (
            scalar_beats_required_controls
        ),
        "component_or_stability_improvement_present": (
            component_or_stability_improvement
        ),
        "primary_actual_injected_ratio_near_0_30": ratio_near,
        "intervention_window_diagnostics_present": intervention_windows_present,
        "no_nan_detected": no_nan,
        "starvation_stop_trigger_absent": not starvation_triggered,
        "scalar_only_success": scalar_only,
        "pass": pass_,
    }


def _auc(training_log: list[dict[str, Any]], key: str) -> float:
    if not training_log:
        return 0.0
    values = np.asarray([row[key] for row in training_log], dtype=np.float64)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, dx=1.0))
    return float(np.sum((values[:-1] + values[1:]) * 0.5))


def _mean_or_none(values: Iterable[Any]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return None if not clean else float(np.mean(clean))


def _std_or_none(values: Iterable[Any]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return None if not clean else float(np.std(clean))
