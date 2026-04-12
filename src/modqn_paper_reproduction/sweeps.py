"""Phase 01 sweep helpers."""

from __future__ import annotations

import copy
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Any

from . import PACKAGE_VERSION, PAPER_ID
from .algorithms.dqn_scalar import ScalarDQNPolicyConfig, ScalarDQNTrainer
from .algorithms.modqn import MODQNTrainer, TrainerConfig
from .baselines.rss_max import evaluate_rss_max
from .config_loader import (
    build_environment,
    build_trainer_config,
    get_seeds,
)
from .export.pipeline import (
    export_figure_sweep_results,
    export_table_ii_results,
)


FIGURE_SUITES: dict[str, dict[str, Any]] = {
    "fig-3": {
        "suite": "fig-3",
        "figure_id": "Fig. 3",
        "parameter_name": "user_count",
        "parameter_label": "User Count",
        "parameter_unit": "users",
        "plot_title": "Performance Comparison Under Different User Numbers",
    },
    "fig-4": {
        "suite": "fig-4",
        "figure_id": "Fig. 4",
        "parameter_name": "satellite_count",
        "parameter_label": "Satellite Count",
        "parameter_unit": "satellites",
        "plot_title": "Performance Comparison Under Different Satellite Numbers",
    },
    "fig-5": {
        "suite": "fig-5",
        "figure_id": "Fig. 5",
        "parameter_name": "user_speed_kmh",
        "parameter_label": "User Speed",
        "parameter_unit": "km/h",
        "plot_title": "Performance Comparison Under User Speeds",
    },
    "fig-6": {
        "suite": "fig-6",
        "figure_id": "Fig. 6",
        "parameter_name": "satellite_speed_km_s",
        "parameter_label": "Satellite Speed",
        "parameter_unit": "km/s",
        "plot_title": "Performance Comparison Under Different Satellite Speeds",
    },
}


def table_ii_weight_rows(cfg: dict[str, Any]) -> list[tuple[float, float, float]]:
    """Extract Table II weight rows from the merged config surface."""
    block = cfg.get("paper_backed_weight_rows", {})
    rows = block.get("table_ii", []) if isinstance(block, dict) else []
    return [tuple(float(x) for x in row) for row in rows]


def figure_point_sets(cfg: dict[str, Any]) -> dict[str, list[float]]:
    """Extract resolved discrete point sets for Fig. 3 to Fig. 6."""
    resolved = cfg.get("resolved_assumptions", {})
    block = resolved.get("figure_discrete_point_set", {})
    value = block.get("value", {}) if isinstance(block, dict) else {}
    if not isinstance(value, dict):
        value = {}
    return {
        "user_count": [float(x) for x in value.get("user_count", [])],
        "satellite_count": [float(x) for x in value.get("satellite_count", [])],
        "user_speed_kmh": [float(x) for x in value.get("user_speed_kmh", [])],
        "satellite_speed_km_s": [float(x) for x in value.get("satellite_speed_km_s", [])],
    }


def default_figure_weight_row(cfg: dict[str, Any]) -> tuple[float, float, float]:
    """Use the baseline objective weights for Fig. 3 to Fig. 6."""
    base = cfg.get("baseline", cfg)
    return tuple(float(x) for x in base.get("objective_weights", [0.5, 0.3, 0.2]))


def _override_episodes(trainer_cfg: TrainerConfig, episodes: int | None) -> TrainerConfig:
    if episodes is None:
        return trainer_cfg
    kwargs = {f.name: getattr(trainer_cfg, f.name) for f in dc_fields(trainer_cfg)}
    kwargs["episodes"] = episodes
    return TrainerConfig(**kwargs)


def _clone_trainer_config(trainer_cfg: TrainerConfig) -> TrainerConfig:
    return TrainerConfig(
        **{f.name: getattr(trainer_cfg, f.name) for f in dc_fields(trainer_cfg)}
    )


def _train_modqn(
    cfg: dict[str, Any],
    *,
    trainer_cfg: TrainerConfig,
    seeds: dict[str, Any],
    eval_seeds: tuple[int, ...],
    progress_every: int,
) -> tuple[MODQNTrainer, str]:
    env = build_environment(copy.deepcopy(cfg))
    trainer = MODQNTrainer(
        env=env,
        config=trainer_cfg,
        train_seed=seeds["train_seed"],
        env_seed=seeds["environment_seed"],
        mobility_seed=seeds["mobility_seed"],
    )
    trainer.train(
        progress_every=progress_every,
        evaluation_seed_set=eval_seeds,
        evaluation_every_episodes=trainer_cfg.target_update_every_episodes,
    )
    checkpoint_kind = trainer_cfg.checkpoint_primary_report
    if trainer.has_best_eval_checkpoint():
        trainer.restore_best_eval_checkpoint(load_optimizers=False)
        checkpoint_kind = trainer_cfg.checkpoint_secondary_report
    return trainer, checkpoint_kind


def _train_scalar_dqn(
    cfg: dict[str, Any],
    *,
    trainer_cfg: TrainerConfig,
    seeds: dict[str, Any],
    eval_seeds: tuple[int, ...],
    progress_every: int,
    policy: ScalarDQNPolicyConfig,
) -> tuple[ScalarDQNTrainer, str]:
    env = build_environment(copy.deepcopy(cfg))
    trainer = ScalarDQNTrainer(
        env=env,
        config=trainer_cfg,
        policy=policy,
        train_seed=seeds["train_seed"],
        env_seed=seeds["environment_seed"],
        mobility_seed=seeds["mobility_seed"],
    )
    trainer.train(
        progress_every=progress_every,
        evaluation_seed_set=eval_seeds,
        evaluation_every_episodes=trainer_cfg.target_update_every_episodes,
    )
    checkpoint_kind = trainer_cfg.checkpoint_primary_report
    if trainer.has_best_eval_checkpoint():
        trainer.restore_best_eval_checkpoint()
        checkpoint_kind = trainer_cfg.checkpoint_secondary_report
    return trainer, checkpoint_kind


def _summary_row(
    *,
    method: str,
    weight_row: tuple[float, float, float],
    summary: Any,
    checkpoint_kind: str | None,
    training_episodes: int,
) -> dict[str, Any]:
    return {
        "method": method,
        "weight_label": "/".join(f"{x:.1f}" for x in weight_row),
        "w1": weight_row[0],
        "w2": weight_row[1],
        "w3": weight_row[2],
        "mean_scalar_reward": summary.mean_scalar_reward,
        "std_scalar_reward": summary.std_scalar_reward,
        "mean_r1": summary.mean_r1,
        "std_r1": summary.std_r1,
        "mean_r2": summary.mean_r2,
        "std_r2": summary.std_r2,
        "mean_r3": summary.mean_r3,
        "std_r3": summary.std_r3,
        "mean_total_handovers": summary.mean_total_handovers,
        "std_total_handovers": summary.std_total_handovers,
        "policy_episode": summary.episode,
        "eval_seed_count": len(summary.eval_seeds),
        "training_episodes": training_episodes,
        "checkpoint_kind": checkpoint_kind,
    }


def _figure_summary_row(
    *,
    suite_spec: dict[str, Any],
    parameter_value: float,
    method: str,
    weight_row: tuple[float, float, float],
    summary: Any,
    checkpoint_kind: str | None,
    training_episodes: int,
) -> dict[str, Any]:
    row = _summary_row(
        method=method,
        weight_row=weight_row,
        summary=summary,
        checkpoint_kind=checkpoint_kind,
        training_episodes=training_episodes,
    )
    row.update(
        {
            "suite": suite_spec["suite"],
            "figure_id": suite_spec["figure_id"],
            "parameter_name": suite_spec["parameter_name"],
            "parameter_label": suite_spec["parameter_label"],
            "parameter_unit": suite_spec["parameter_unit"],
            "parameter_value": float(parameter_value),
        }
    )
    return row


def _figure_cfg_for_point(
    cfg: dict[str, Any],
    *,
    suite: str,
    parameter_value: float,
) -> dict[str, Any]:
    """Apply one figure-sweep point to a config copy."""
    point_cfg = copy.deepcopy(cfg)
    baseline = point_cfg.setdefault("baseline", {})
    resolved = point_cfg.setdefault("resolved_assumptions", {})
    orbit_layout = resolved.setdefault("orbit_layout", {})
    orbit_value = orbit_layout.setdefault("value", {})

    if suite == "fig-3":
        baseline["users"] = int(parameter_value)
    elif suite == "fig-4":
        baseline["satellites"] = int(parameter_value)
        orbit_value["satellites_per_plane"] = int(parameter_value)
    elif suite == "fig-5":
        baseline["user_speed_kmh"] = float(parameter_value)
    elif suite == "fig-6":
        baseline["satellite_speed_km_s"] = float(parameter_value)
    else:
        raise ValueError(f"Unsupported figure suite {suite!r}")

    return point_cfg


def run_table_ii(
    cfg: dict[str, Any],
    *,
    output_dir: str | Path,
    episodes: int | None = None,
    progress_every: int = 100,
    max_weight_rows: int | None = None,
    methods: tuple[str, ...] = ("modqn", "dqn_throughput", "dqn_scalar", "rss_max"),
    reference_run_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Run the first executable Table II sweep slice."""
    trainer_cfg = _override_episodes(build_trainer_config(cfg), episodes)
    seeds = get_seeds(cfg)
    eval_seeds = tuple(seeds["evaluation_seed_set"])
    rows = table_ii_weight_rows(cfg)
    if max_weight_rows is not None:
        rows = rows[: max(0, int(max_weight_rows))]
    if not rows:
        raise ValueError("No Table II weight rows are available in the config")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    method_list = tuple(m.strip().lower() for m in methods if m.strip())

    if "modqn" in method_list:
        trainer, checkpoint_kind = _train_modqn(
            cfg,
            trainer_cfg=trainer_cfg,
            seeds=seeds,
            eval_seeds=eval_seeds,
            progress_every=progress_every,
        )
        for weight_row in rows:
            summary = trainer.evaluate_policy(
                eval_seeds,
                episode=(
                    trainer.best_eval_summary().episode
                    if trainer.best_eval_summary()
                    else trainer_cfg.episodes - 1
                ),
                evaluation_every_episodes=trainer_cfg.target_update_every_episodes,
                objective_weights=weight_row,
            )
            results.append(
                _summary_row(
                    method="MODQN",
                    weight_row=weight_row,
                    summary=summary,
                    checkpoint_kind=checkpoint_kind,
                    training_episodes=trainer_cfg.episodes,
                )
            )

    if "dqn_throughput" in method_list:
        throughput_cfg = _clone_trainer_config(trainer_cfg)
        trainer, checkpoint_kind = _train_scalar_dqn(
            cfg,
            trainer_cfg=throughput_cfg,
            seeds=seeds,
            eval_seeds=eval_seeds,
            progress_every=progress_every,
            policy=ScalarDQNPolicyConfig(
                name="DQN_throughput",
                scalar_reward_weights=(1.0, 0.0, 0.0),
            ),
        )
        for weight_row in rows:
            summary = trainer.evaluate_policy(
                eval_seeds,
                episode=(
                    trainer.best_eval_summary().episode
                    if trainer.best_eval_summary()
                    else throughput_cfg.episodes - 1
                ),
                evaluation_every_episodes=throughput_cfg.target_update_every_episodes,
                scalarization_weights=weight_row,
            )
            results.append(
                _summary_row(
                    method="DQN_throughput",
                    weight_row=weight_row,
                    summary=summary,
                    checkpoint_kind=checkpoint_kind,
                    training_episodes=throughput_cfg.episodes,
                )
            )

    if "dqn_scalar" in method_list:
        for weight_row in rows:
            scalar_cfg = _clone_trainer_config(trainer_cfg)
            trainer, checkpoint_kind = _train_scalar_dqn(
                cfg,
                trainer_cfg=scalar_cfg,
                seeds=seeds,
                eval_seeds=eval_seeds,
                progress_every=progress_every,
                policy=ScalarDQNPolicyConfig(
                    name="DQN_scalar",
                    scalar_reward_weights=weight_row,
                ),
            )
            summary = trainer.evaluate_policy(
                eval_seeds,
                episode=(
                    trainer.best_eval_summary().episode
                    if trainer.best_eval_summary()
                    else scalar_cfg.episodes - 1
                ),
                evaluation_every_episodes=scalar_cfg.target_update_every_episodes,
                scalarization_weights=weight_row,
            )
            results.append(
                _summary_row(
                    method="DQN_scalar",
                    weight_row=weight_row,
                    summary=summary,
                    checkpoint_kind=checkpoint_kind,
                    training_episodes=scalar_cfg.episodes,
                )
            )

    if "rss_max" in method_list:
        for weight_row in rows:
            env = build_environment(copy.deepcopy(cfg))
            summary = evaluate_rss_max(
                env,
                evaluation_seed_set=eval_seeds,
                scalarization_weights=weight_row,
            )
            results.append(
                _summary_row(
                    method="RSS_max",
                    weight_row=weight_row,
                    summary=summary,
                    checkpoint_kind="not-applicable",
                    training_episodes=0,
                )
            )

    manifest = {
        "paperId": PAPER_ID,
        "producerVersion": PACKAGE_VERSION,
        "configRole": cfg.get("config_role"),
        "trainingExperiment": cfg.get("training_experiment"),
        "methods": list(dict.fromkeys(row["method"] for row in results)),
        "tableIiWeightRows": [list(row) for row in rows],
        "seeds": seeds,
        "checkpointRule": cfg.get("resolved_assumptions", {}).get(
            "checkpoint_selection_rule",
            {},
        ),
        "aggregationRule": cfg.get("resolved_assumptions", {}).get(
            "evaluation_aggregation",
            {},
        ),
    }
    return export_table_ii_results(
        out_dir,
        rows=results,
        manifest=manifest,
        reference_run_dir=reference_run_dir,
    )


def run_figure_suite(
    cfg: dict[str, Any],
    *,
    suite: str,
    output_dir: str | Path,
    episodes: int | None = None,
    progress_every: int = 100,
    max_points: int | None = None,
    methods: tuple[str, ...] = ("modqn", "dqn_throughput", "dqn_scalar", "rss_max"),
    reference_run_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Run one executable figure sweep for Fig. 3 to Fig. 6."""
    if suite not in FIGURE_SUITES:
        raise ValueError(f"Unsupported figure suite {suite!r}")

    suite_spec = FIGURE_SUITES[suite]
    trainer_cfg = _override_episodes(build_trainer_config(cfg), episodes)
    seeds = get_seeds(cfg)
    eval_seeds = tuple(seeds["evaluation_seed_set"])
    weight_row = default_figure_weight_row(cfg)
    point_sets = figure_point_sets(cfg)
    points = point_sets.get(suite_spec["parameter_name"], [])
    if max_points is not None:
        points = points[: max(0, int(max_points))]
    if not points:
        raise ValueError(f"No point set is available for {suite_spec['figure_id']}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    method_list = tuple(m.strip().lower() for m in methods if m.strip())

    for point in points:
        point_cfg = _figure_cfg_for_point(cfg, suite=suite, parameter_value=point)

        if "modqn" in method_list:
            trainer, checkpoint_kind = _train_modqn(
                point_cfg,
                trainer_cfg=trainer_cfg,
                seeds=seeds,
                eval_seeds=eval_seeds,
                progress_every=progress_every,
            )
            summary = trainer.evaluate_policy(
                eval_seeds,
                episode=(
                    trainer.best_eval_summary().episode
                    if trainer.best_eval_summary()
                    else trainer_cfg.episodes - 1
                ),
                evaluation_every_episodes=trainer_cfg.target_update_every_episodes,
                objective_weights=weight_row,
            )
            results.append(
                _figure_summary_row(
                    suite_spec=suite_spec,
                    parameter_value=point,
                    method="MODQN",
                    weight_row=weight_row,
                    summary=summary,
                    checkpoint_kind=checkpoint_kind,
                    training_episodes=trainer_cfg.episodes,
                )
            )

        if "dqn_throughput" in method_list:
            throughput_cfg = _clone_trainer_config(trainer_cfg)
            trainer, checkpoint_kind = _train_scalar_dqn(
                point_cfg,
                trainer_cfg=throughput_cfg,
                seeds=seeds,
                eval_seeds=eval_seeds,
                progress_every=progress_every,
                policy=ScalarDQNPolicyConfig(
                    name="DQN_throughput",
                    scalar_reward_weights=(1.0, 0.0, 0.0),
                ),
            )
            summary = trainer.evaluate_policy(
                eval_seeds,
                episode=(
                    trainer.best_eval_summary().episode
                    if trainer.best_eval_summary()
                    else throughput_cfg.episodes - 1
                ),
                evaluation_every_episodes=throughput_cfg.target_update_every_episodes,
                scalarization_weights=weight_row,
            )
            results.append(
                _figure_summary_row(
                    suite_spec=suite_spec,
                    parameter_value=point,
                    method="DQN_throughput",
                    weight_row=weight_row,
                    summary=summary,
                    checkpoint_kind=checkpoint_kind,
                    training_episodes=throughput_cfg.episodes,
                )
            )

        if "dqn_scalar" in method_list:
            scalar_cfg = _clone_trainer_config(trainer_cfg)
            trainer, checkpoint_kind = _train_scalar_dqn(
                point_cfg,
                trainer_cfg=scalar_cfg,
                seeds=seeds,
                eval_seeds=eval_seeds,
                progress_every=progress_every,
                policy=ScalarDQNPolicyConfig(
                    name="DQN_scalar",
                    scalar_reward_weights=weight_row,
                ),
            )
            summary = trainer.evaluate_policy(
                eval_seeds,
                episode=(
                    trainer.best_eval_summary().episode
                    if trainer.best_eval_summary()
                    else scalar_cfg.episodes - 1
                ),
                evaluation_every_episodes=scalar_cfg.target_update_every_episodes,
                scalarization_weights=weight_row,
            )
            results.append(
                _figure_summary_row(
                    suite_spec=suite_spec,
                    parameter_value=point,
                    method="DQN_scalar",
                    weight_row=weight_row,
                    summary=summary,
                    checkpoint_kind=checkpoint_kind,
                    training_episodes=scalar_cfg.episodes,
                )
            )

        if "rss_max" in method_list:
            env = build_environment(copy.deepcopy(point_cfg))
            summary = evaluate_rss_max(
                env,
                evaluation_seed_set=eval_seeds,
                scalarization_weights=weight_row,
            )
            results.append(
                _figure_summary_row(
                    suite_spec=suite_spec,
                    parameter_value=point,
                    method="RSS_max",
                    weight_row=weight_row,
                    summary=summary,
                    checkpoint_kind="not-applicable",
                    training_episodes=0,
                )
            )

    manifest = {
        "paperId": PAPER_ID,
        "producerVersion": PACKAGE_VERSION,
        "configRole": cfg.get("config_role"),
        "trainingExperiment": cfg.get("training_experiment"),
        "suite": suite,
        "figureId": suite_spec["figure_id"],
        "sweepParameter": suite_spec["parameter_name"],
        "sweepPointSet": points,
        "baselineWeightRow": list(weight_row),
        "methods": list(dict.fromkeys(row["method"] for row in results)),
        "seeds": seeds,
        "checkpointRule": cfg.get("resolved_assumptions", {}).get(
            "checkpoint_selection_rule",
            {},
        ),
        "aggregationRule": cfg.get("resolved_assumptions", {}).get(
            "evaluation_aggregation",
            {},
        ),
    }
    return export_figure_sweep_results(
        out_dir,
        rows=results,
        manifest=manifest,
        suite_spec=suite_spec,
        reference_run_dir=reference_run_dir,
    )
