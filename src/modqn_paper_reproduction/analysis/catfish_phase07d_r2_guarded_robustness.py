"""Phase 07-D r2-guarded single-Catfish robustness helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..config_loader import ConfigValidationError, get_seeds
from ..runtime.trainer_spec import (
    PHASE_07_D_R2_GUARDED_ROBUSTNESS_KIND,
    R1_REWARD_MODE_THROUGHPUT,
    TrainerConfig,
)


PHASE_07D_CONFIGS = (
    "configs/catfish-modqn-phase-07d-modqn-control.resolved.yaml",
    "configs/catfish-modqn-phase-07d-r2-guarded-primary-shaping-off.resolved.yaml",
    "configs/catfish-modqn-phase-07d-no-intervention.resolved.yaml",
    "configs/catfish-modqn-phase-07d-random-equal-budget-injection.resolved.yaml",
    "configs/catfish-modqn-phase-07d-replay-only-single-learner.resolved.yaml",
    "configs/catfish-modqn-phase-07d-no-asymmetric-gamma.resolved.yaml",
    "configs/catfish-modqn-phase-07d-admission-only-guard.resolved.yaml",
    "configs/catfish-modqn-phase-07d-intervention-only-guard.resolved.yaml",
    "configs/catfish-modqn-phase-07d-full-admission-intervention-guard.resolved.yaml",
)
PHASE_07D_OPTIONAL_CONFIGS = (
    "configs/catfish-modqn-phase-07d-strict-no-handover-sample-guard.resolved.yaml",
)
PHASE_07D_EVAL_SEEDS = [100, 200, 300, 400, 500]
PHASE_07D_REQUIRED_ROLES = (
    "matched-modqn-control",
    "r2-guarded-primary-shaping-off",
    "no-intervention",
    "random-equal-budget-injection",
    "replay-only-single-learner",
    "no-asymmetric-gamma",
    "admission-only-guard",
    "intervention-only-guard",
    "full-admission-intervention-guard",
)
PHASE_07D_PRIMARY_ROLE = "r2-guarded-primary-shaping-off"
PHASE_07D_NONINFERIORITY_CONTROLS = (
    "matched-modqn-control",
    "random-equal-budget-injection",
)
PHASE_07D_SCALAR_CONTROLS = (
    "matched-modqn-control",
    "no-intervention",
    "random-equal-budget-injection",
    "replay-only-single-learner",
)
R2_DELTA_MARGIN = -0.02
HANDOVER_DELTA_MARGIN = 5.0
PROTECTED_CONFIGS = (
    "configs/modqn-paper-baseline.yaml",
    "configs/modqn-paper-baseline.resolved-template.yaml",
)
FROZEN_ARTIFACT_PREFIXES = (
    "artifacts/pilot-02-best-eval",
    "artifacts/run-9000",
    "artifacts/table-ii-200ep-01",
    "artifacts/fig-3-pilot-01",
    "artifacts/fig-4-pilot-01",
    "artifacts/fig-5-pilot-01",
    "artifacts/fig-6-pilot-01",
)


def validate_phase07d_training_config(
    cfg: dict[str, Any],
    trainer_cfg: TrainerConfig,
) -> dict[str, Any]:
    """Validate the Phase 07-D bounded robustness contract."""
    if trainer_cfg.training_experiment_kind != PHASE_07_D_R2_GUARDED_ROBUSTNESS_KIND:
        raise ConfigValidationError(
            "Phase 07D requires training_experiment.kind="
            f"{PHASE_07_D_R2_GUARDED_ROBUSTNESS_KIND!r}."
        )
    if trainer_cfg.episodes != 20:
        raise ConfigValidationError("Phase 07D bounded run budget must be 20 episodes.")
    if trainer_cfg.target_update_every_episodes != 5:
        raise ConfigValidationError(
            "Phase 07D eval/checkpoint cadence must be 5 episodes."
        )
    if trainer_cfg.r1_reward_mode != R1_REWARD_MODE_THROUGHPUT:
        raise ConfigValidationError("Phase 07D requires original r1='throughput'.")
    if trainer_cfg.reward_calibration_enabled:
        raise ConfigValidationError("Phase 07D must not enable reward calibration.")
    if trainer_cfg.catfish_competitive_shaping_enabled:
        raise ConfigValidationError("Phase 07D keeps primary shaping off.")

    seeds = get_seeds(cfg)
    if seeds["evaluation_seed_set"] != PHASE_07D_EVAL_SEEDS:
        raise ConfigValidationError(
            f"Phase 07D eval seeds must be {PHASE_07D_EVAL_SEEDS}, "
            f"got {seeds['evaluation_seed_set']}."
        )
    phase_block = cfg.get("training_experiment", {}).get(
        "phase_07d_r2_guarded_robustness"
    )
    if not isinstance(phase_block, dict):
        raise ConfigValidationError(
            "Phase 07D requires training_experiment."
            "phase_07d_r2_guarded_robustness."
        )
    seed_triplets = phase_block.get("seed_triplets", [])
    if len(seed_triplets) < 3:
        raise ConfigValidationError("Phase 07D requires at least 3 seed triplets.")

    expected_reward = {
        "r1": "throughput",
        "r2": "handover penalty",
        "r3": "load balance",
    }
    if phase_block.get("reward_surface", {}) != expected_reward:
        raise ConfigValidationError("Phase 07D reward surface was modified.")

    serialized = json.dumps(cfg, sort_keys=True)
    forbidden_terms = (
        "EE-MODQN",
        "Catfish-EE-MODQN",
        "RA-EE",
        "per-user-ee",
        "Multi-Catfish-MODQN",
        "phase-06",
    )
    if any(term in serialized for term in forbidden_terms):
        raise ConfigValidationError(
            "Phase 07D configs must not introduce EE, Catfish-EE, RA-EE, "
            "Multi-Catfish, or Phase 06 framing."
        )
    return phase_block


def summarize_phase07d_runs(
    *,
    run_dirs: Iterable[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Summarize completed Phase 07-D run artifacts across seed triplets."""
    rows = [_load_run(Path(path)) for path in run_dirs]
    if not rows:
        raise ValueError("summarize_phase07d_runs requires at least one run dir.")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["role"], []).append(row)

    role_summaries = {
        role: _role_summary(role_rows)
        for role, role_rows in sorted(grouped.items())
    }
    deltas = _comparator_deltas(role_summaries)
    acceptance = _acceptance_summary(role_summaries, rows, deltas)
    summary = {
        "phase": "07D",
        "protocol": _protocol_summary(rows),
        "predeclared_noninferiority_margins": {
            "r2_delta_min": R2_DELTA_MARGIN,
            "handover_delta_max": HANDOVER_DELTA_MARGIN,
        },
        "required_roles": list(PHASE_07D_REQUIRED_ROLES),
        "role_summaries": role_summaries,
        "comparator_deltas": deltas,
        "acceptance_checks": acceptance,
        "protected_surface_check": _protected_surface_check(rows),
        "claim_boundary": {
            "catfish_ee_or_ee_claim_made": False,
            "phase06_claim_made": False,
            "multi_catfish_reopening_performed": False,
            "full_paper_faithful_claim_made": False,
            "scalar_reward_alone_used_as_success": False,
            "asymmetric_gamma_mechanism_claim_made": False,
            "frozen_baseline_config_or_artifact_modified_by_summary": False,
        },
    }
    summary["pass"] = bool(
        acceptance["pass"]
        and summary["protected_surface_check"]["created_run_dirs_under_allowed_namespace"]
        and not summary["protected_surface_check"]["frozen_namespace_used"]
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "phase07d_r2_guarded_robustness_summary.json"
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
        "variant": trainer_config.get("catfish_phase07d_variant"),
        "seed_triplet": [
            metadata["seeds"]["train_seed"],
            metadata["seeds"]["environment_seed"],
            metadata["seeds"]["mobility_seed"],
        ],
        "episodes_completed": metadata["training_summary"]["episodes_completed"],
        "elapsed_s": metadata["training_summary"]["elapsed_s"],
        "eval_seeds": metadata["seeds"]["evaluation_seed_set"],
        "runtime": metadata.get("runtime_environment", {}),
        "config_snapshot": metadata.get("resolved_config_snapshot", {}),
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
        "seed_wise_final": [
            {
                "seed_triplet": row["seed_triplet"],
                "scalar_reward": row["final"]["scalar_reward"],
                "r1_mean": row["final"]["r1_mean"],
                "r2_mean": row["final"]["r2_mean"],
                "r3_mean": row["final"]["r3_mean"],
                "total_handovers": row["final"]["total_handovers"],
                "handover_rate": _handover_rate(row),
            }
            for row in rows
        ],
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
        "scalar_reward": float(np.mean([row[key]["scalar_reward"] for row in rows])),
        "r1_mean": float(np.mean([row[key]["r1_mean"] for row in rows])),
        "r2_mean": float(np.mean([row[key]["r2_mean"] for row in rows])),
        "r3_mean": float(np.mean([row[key]["r3_mean"] for row in rows])),
        "total_handovers": float(
            np.mean([row[key]["total_handovers"] for row in rows])
        ),
        "handover_rate": float(np.mean([_handover_rate(row) for row in rows])),
    }


def _metric_stds(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    return {
        "scalar_reward": float(np.std([row[key]["scalar_reward"] for row in rows])),
        "r1_mean": float(np.std([row[key]["r1_mean"] for row in rows])),
        "r2_mean": float(np.std([row[key]["r2_mean"] for row in rows])),
        "r3_mean": float(np.std([row[key]["r3_mean"] for row in rows])),
        "total_handovers": float(
            np.std([row[key]["total_handovers"] for row in rows])
        ),
        "handover_rate": float(np.std([_handover_rate(row) for row in rows])),
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
    guard_rows = [
        row.get("r2_handover_guard", {})
        for row in diagnostics_rows
        if row.get("r2_handover_guard") is not None
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
        "r2_handover_guard": _guard_summary(guard_rows),
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


def _guard_summary(guard_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not guard_rows:
        return {"available": False}
    final = guard_rows[-1]
    skip_reasons: dict[str, int] = {}
    for row in guard_rows:
        for reason, count in (row.get("skip_reasons") or {}).items():
            skip_reasons[reason] = skip_reasons.get(reason, 0) + int(count)
    return {
        "available": True,
        "enabled": bool(final.get("enabled")),
        "admission_guard_enabled": bool(final.get("admission_guard_enabled")),
        "intervention_guard_enabled": bool(final.get("intervention_guard_enabled")),
        "strict_no_handover_sample_guard": bool(
            final.get("strict_no_handover_sample_guard")
        ),
        "admission_guard_pass_count": int(
            final.get("admission_guard_pass_count", 0) or 0
        ),
        "admission_guard_skip_count": int(
            final.get("admission_guard_skip_count", 0) or 0
        ),
        "intervention_guard_pass_count": int(
            final.get("intervention_guard_pass_count", 0) or 0
        ),
        "intervention_guard_skip_count": int(
            final.get("intervention_guard_skip_count", 0) or 0
        ),
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "guarded_batch_count": int(final.get("guarded_batch_count", 0) or 0),
        "guarded_batch_pass_count": int(
            final.get("guarded_batch_pass_count", 0) or 0
        ),
        "guarded_batch_violation_count": int(
            final.get("guarded_batch_violation_count", 0) or 0
        ),
        "injected_batch_r2_negative_share_distribution": final.get(
            "injected_batch_r2_negative_share_distribution", {}
        ),
        "matched_main_batch_r2_negative_share_distribution": final.get(
            "matched_main_batch_r2_negative_share_distribution", {}
        ),
        "recent_batch_records": final.get("recent_batch_records", []),
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
    primary = role_summaries.get(PHASE_07D_PRIMARY_ROLE)
    if not primary:
        return {}
    primary_final = primary["final_mean"]
    deltas: dict[str, Any] = {}
    for role, summary in role_summaries.items():
        if role == PHASE_07D_PRIMARY_ROLE:
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
            "handover_rate_delta": (
                primary_final["handover_rate"] - final["handover_rate"]
            ),
        }
    return deltas


def _acceptance_summary(
    role_summaries: dict[str, Any],
    rows: list[dict[str, Any]],
    deltas: dict[str, Any],
) -> dict[str, Any]:
    required_roles_complete = all(
        role in role_summaries and role_summaries[role]["run_count"] >= 3
        for role in PHASE_07D_REQUIRED_ROLES
    )
    primary = role_summaries.get(PHASE_07D_PRIMARY_ROLE)
    if not primary:
        return {"required_roles_complete": required_roles_complete, "pass": False}

    primary_final = primary["final_mean"]
    scalar_beats_controls = all(
        role in role_summaries
        and primary_final["scalar_reward"]
        > role_summaries[role]["final_mean"]["scalar_reward"]
        for role in PHASE_07D_SCALAR_CONTROLS
    )
    component_support = any(
        role in role_summaries
        and (
            primary_final["r1_mean"] > role_summaries[role]["final_mean"]["r1_mean"]
            or primary_final["r3_mean"] > role_summaries[role]["final_mean"]["r3_mean"]
        )
        for role in PHASE_07D_SCALAR_CONTROLS
    )
    scalar_only = scalar_beats_controls and not component_support
    r2_noninferior = all(
        role in deltas and deltas[role]["r2_delta"] >= R2_DELTA_MARGIN
        for role in PHASE_07D_NONINFERIORITY_CONTROLS
    )
    handover_noninferior = all(
        role in deltas and deltas[role]["handover_delta"] <= HANDOVER_DELTA_MARGIN
        for role in PHASE_07D_NONINFERIORITY_CONTROLS
    )

    primary_diag = primary["catfish_diagnostics"]
    ratio = primary_diag.get("actual_injected_ratio_mean")
    ratio_near = ratio is not None and abs(float(ratio) - 0.30) <= 0.03
    windows_present = (
        primary_diag.get("intervention_window_count_mean") is not None
        and float(primary_diag["intervention_window_count_mean"]) > 0
    )
    guard = primary_diag.get("r2_handover_guard", {})
    guard_diagnostics_present = bool(guard.get("available")) and bool(
        guard.get("enabled")
    )
    guard_no_violations = int(guard.get("guarded_batch_violation_count", 1) or 0) == 0
    guard_passed = int(guard.get("intervention_guard_pass_count", 0) or 0) > 0
    hidden_unguarded_fallback_absent = guard_diagnostics_present and guard_no_violations

    starvation = primary_diag.get("replay_starvation", {})
    starvation_triggered = any(
        int(starvation.get(key, 0) or 0) > 0
        for key in (
            "main_replay_starved_updates_cumulative",
            "catfish_replay_starved_intervention_cumulative",
        )
    )
    no_nan = not bool(primary_diag.get("any_nan_detected"))
    no_action_collapse = not bool(
        primary_diag.get("last_episode_action_diversity", {}).get(
            "action_collapse_detected"
        )
    )
    diagnostics_complete = all(_row_has_required_diagnostics(row) for row in rows)
    frozen_namespace_absent = not _protected_surface_check(rows)["frozen_namespace_used"]

    pass_ = all(
        [
            required_roles_complete,
            scalar_beats_controls,
            component_support,
            not scalar_only,
            r2_noninferior,
            handover_noninferior,
            ratio_near,
            windows_present,
            guard_diagnostics_present,
            guard_passed,
            hidden_unguarded_fallback_absent,
            no_nan,
            no_action_collapse,
            not starvation_triggered,
            diagnostics_complete,
            frozen_namespace_absent,
        ]
    )
    return {
        "required_roles_complete": required_roles_complete,
        "primary_beats_matched_modqn_no_intervention_random_and_replay_only": (
            scalar_beats_controls
        ),
        "component_support_not_scalar_only": component_support,
        "scalar_only_success": scalar_only,
        "r2_noninferior_vs_matched_modqn_and_random": r2_noninferior,
        "handover_noninferior_vs_matched_modqn_and_random": handover_noninferior,
        "primary_actual_injected_ratio_near_0_30": ratio_near,
        "intervention_window_diagnostics_present": windows_present,
        "guard_diagnostics_present": guard_diagnostics_present,
        "guard_intervention_passed": guard_passed,
        "hidden_unguarded_fallback_absent": hidden_unguarded_fallback_absent,
        "no_nan_detected": no_nan,
        "no_action_collapse_detected": no_action_collapse,
        "starvation_stop_trigger_absent": not starvation_triggered,
        "required_diagnostics_present": diagnostics_complete,
        "frozen_namespace_absent": frozen_namespace_absent,
        "pass": pass_,
    }


def _row_has_required_diagnostics(row: dict[str, Any]) -> bool:
    if row["role"] == "matched-modqn-control":
        return True
    diagnostics = row.get("diagnostics") or {}
    if not diagnostics:
        return False
    if row["role"] == "random-equal-budget-injection":
        return bool(diagnostics.get("intervention_utility"))
    guard = diagnostics.get("r2_handover_guard", {})
    return all(
        key in diagnostics
        for key in (
            "cumulative",
            "final_replay",
            "intervention_utility",
            "replay_starvation",
            "episode_diagnostics",
        )
    ) and bool(guard)


def _protected_surface_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    run_dirs = [str(row["run_dir"]) for row in rows]
    frozen_namespace_used = any(_path_has_prefix(path, FROZEN_ARTIFACT_PREFIXES) for path in run_dirs)
    return {
        "created_run_dirs_under_allowed_namespace": all(
            _path_has_prefix(path, ("artifacts/catfish-modqn-phase-07d-",))
            for path in run_dirs
        ),
        "frozen_namespace_used": frozen_namespace_used,
        "protected_configs_declared_unchanged": list(PROTECTED_CONFIGS),
        "frozen_artifact_prefixes": list(FROZEN_ARTIFACT_PREFIXES),
    }


def _path_has_prefix(path: str, prefixes: tuple[str, ...]) -> bool:
    normalized = path.replace("\\", "/")
    return any(
        normalized.startswith(prefix) or f"/{prefix}" in normalized
        for prefix in prefixes
    )


def _handover_rate(row: dict[str, Any]) -> float:
    runtime = row.get("runtime", {})
    config = row.get("config_snapshot", {})
    baseline = config.get("baseline", {})
    num_users = float(runtime.get("num_users", baseline.get("users", 100)) or 100)
    slot_s = float(baseline.get("slot_duration_s", 1.0) or 1.0)
    duration_s = float(baseline.get("episode_duration_s", 10.0) or 10.0)
    steps = max(int(round(duration_s / slot_s)), 1)
    return float(row["final"]["total_handovers"] / max(num_users * steps, 1.0))


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
