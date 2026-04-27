"""Training-command orchestration."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path


def run_train_command(
    args: argparse.Namespace,
    *,
    paper_id: str,
    package_version: str,
) -> int:
    from ..algorithms.modqn import MODQNTrainer
    from ..artifacts import (
        CheckpointCatalog,
        ResumeFromV1,
        RewardCalibrationV1,
        RunArtifactPaths,
        RunMetadataV1,
        RuntimeEnvironmentV1,
        SeedsBlock,
        TrainingLogRow,
        TrainingSummaryV1,
        write_run_metadata,
        write_training_log,
    )
    from ..config_loader import (
        ConfigValidationError,
        build_environment,
        build_trainer_config,
        get_seeds,
        load_training_yaml,
    )

    print(f"[modqn-train] paper={paper_id} version={package_version}")
    print(f"[modqn-train] config={args.config}")

    try:
        cfg = load_training_yaml(args.config)
        env = build_environment(cfg)
        trainer_cfg = build_trainer_config(cfg)
        seeds = get_seeds(cfg)
    except ConfigValidationError as exc:
        print(f"[modqn-train] ERROR: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"[modqn-train] ERROR: invalid training config: {exc}", file=sys.stderr)
        return 2

    print(
        f"[modqn-train] env: {env.config.num_users} users, "
        f"{env.orbit.num_satellites} sats, "
        f"{env.beam_pattern.num_beams} beams/sat, "
        f"total_beams={env.num_beams_total}"
    )

    from dataclasses import fields as dc_fields

    from ..runtime.trainer_spec import TrainerConfig

    if args.episodes is not None:
        kwargs = {f.name: getattr(trainer_cfg, f.name) for f in dc_fields(trainer_cfg)}
        kwargs["episodes"] = args.episodes
        trainer_cfg = TrainerConfig(**kwargs)

    print(f"[modqn-train] seeds: {seeds}")
    print(
        f"[modqn-train] trainer: episodes={trainer_cfg.episodes}, "
        f"lr={trainer_cfg.learning_rate}, "
        f"batch={trainer_cfg.batch_size}, "
        f"replay={trainer_cfg.replay_capacity}, "
        f"eps={trainer_cfg.epsilon_start}->{trainer_cfg.epsilon_end} "
        f"over {trainer_cfg.epsilon_decay_episodes} eps"
    )
    print(
        "[modqn-train] checkpoint-rule: "
        f"primary={trainer_cfg.checkpoint_primary_report}, "
        f"secondary={trainer_cfg.checkpoint_secondary_report} "
        "(secondary best-eval checkpoint uses the configured evaluation seed set)"
    )
    if trainer_cfg.reward_calibration_enabled:
        print(
            "[modqn-train] experiment: "
            f"kind={trainer_cfg.training_experiment_kind} "
            f"id={trainer_cfg.training_experiment_id or '<unspecified>'} "
            f"reward-calibration={trainer_cfg.reward_calibration_mode} "
            f"source={trainer_cfg.reward_calibration_source} "
            f"scales={trainer_cfg.reward_calibration_scales}"
        )
        print(
            "[modqn-train] experiment-reporting: "
            "training rewards are calibrated in the trainer, while evaluation, "
            "logged r1/r2/r3, and checkpoint selection remain raw paper metrics"
        )

    trainer = MODQNTrainer(
        env=env,
        config=trainer_cfg,
        train_seed=seeds["train_seed"],
        env_seed=seeds["environment_seed"],
        mobility_seed=seeds["mobility_seed"],
    )

    resumed_from: dict | None = None
    if args.resume_from:
        payload = trainer.load_checkpoint(args.resume_from)
        resumed_from = {
            "path": str(Path(args.resume_from)),
            "checkpoint_kind": payload.get("checkpoint_kind"),
            "episode": payload.get("episode"),
        }
        print(
            "[modqn-train] resumed-from: "
            f"{resumed_from['path']} "
            f"(kind={resumed_from['checkpoint_kind']}, episode={resumed_from['episode']})"
        )

    t0 = time.time()
    evaluation_seed_set = tuple(seeds.get("evaluation_seed_set", ()))
    logs = trainer.train(
        progress_every=args.progress_every,
        evaluation_seed_set=evaluation_seed_set,
        evaluation_every_episodes=trainer_cfg.target_update_every_episodes,
    )
    elapsed = time.time() - t0

    print(f"[modqn-train] done: {len(logs)} episodes in {elapsed:.1f}s")

    if logs:
        final = logs[-1]
        print(
            f"[modqn-train] final: scalar={final.scalar_reward:.4e}, "
            f"r1={final.r1_mean:.4e}, r2={final.r2_mean:.4f}, "
            f"r3={final.r3_mean:.4e}, ho={final.total_handovers}"
        )

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        artifact_paths = RunArtifactPaths(out_dir)

        log_path = write_training_log(
            artifact_paths.training_log_json,
            [
                TrainingLogRow(
                    episode=l.episode,
                    epsilon=l.epsilon,
                    r1_mean=l.r1_mean,
                    r2_mean=l.r2_mean,
                    r3_mean=l.r3_mean,
                    scalar_reward=l.scalar_reward,
                    total_handovers=l.total_handovers,
                    replay_size=l.replay_size,
                    losses=l.losses,
                )
                for l in logs
            ],
        )
        print(f"[modqn-train] logs saved to {log_path}")

        final_episode = logs[-1].episode if logs else (
            resumed_from["episode"] if resumed_from else -1
        )
        checkpoint_rule = trainer.checkpoint_rule()
        best_eval_summary = None
        final_checkpoint_path = trainer.save_checkpoint(
            artifact_paths.primary_checkpoint(checkpoint_rule),
            episode=final_episode,
            checkpoint_kind=trainer_cfg.checkpoint_primary_report,
            logs=logs,
            include_optimizers=True,
        )
        print(f"[modqn-train] final checkpoint saved to {final_checkpoint_path}")

        secondary_checkpoint_path = None
        if trainer.has_best_eval_checkpoint():
            best_eval_summary_obj = trainer.best_eval_summary()
            best_eval_summary = (
                asdict(best_eval_summary_obj)
                if best_eval_summary_obj is not None
                else None
            )
            secondary_checkpoint_path = trainer.save_best_eval_checkpoint(
                artifact_paths.secondary_checkpoint(checkpoint_rule)
            )
            print(
                "[modqn-train] best-eval checkpoint saved to "
                f"{secondary_checkpoint_path}"
            )

        checkpoint_catalog = CheckpointCatalog(
            primary_final=final_checkpoint_path,
            secondary_best_eval=secondary_checkpoint_path,
        )
        metadata = RunMetadataV1(
            paper_id=paper_id,
            package_version=package_version,
            config_path=str(Path(args.config)),
            config_role=cfg.get("config_role"),
            resolved_config_snapshot=cfg,
            training_experiment=cfg.get("training_experiment"),
            seeds=SeedsBlock(
                train_seed=seeds["train_seed"],
                environment_seed=seeds["environment_seed"],
                mobility_seed=seeds["mobility_seed"],
                evaluation_seed_set=tuple(seeds.get("evaluation_seed_set", ())),
            ),
            checkpoint_rule=checkpoint_rule,
            reward_calibration=RewardCalibrationV1(
                enabled=trainer_cfg.reward_calibration_enabled,
                mode=trainer_cfg.reward_calibration_mode,
                source=trainer_cfg.reward_calibration_source,
                scales=tuple(trainer_cfg.reward_calibration_scales),
                training_experiment_kind=trainer_cfg.training_experiment_kind,
                training_experiment_id=trainer_cfg.training_experiment_id,
                evaluation_metrics="raw-paper-metrics",
                checkpoint_selection_metric="raw-weighted-eval",
            ),
            checkpoint_files=checkpoint_catalog.to_v1(),
            resolved_assumptions=cfg.get("resolved_assumptions", {}),
            runtime_environment=RuntimeEnvironmentV1(
                num_users=env.config.num_users,
                num_satellites=env.orbit.num_satellites,
                beams_per_satellite=env.beam_pattern.num_beams,
                user_lat_deg=env.config.user_lat_deg,
                user_lon_deg=env.config.user_lon_deg,
                r3_gap_scope=env.config.r3_gap_scope,
                r3_empty_beam_throughput=env.config.r3_empty_beam_throughput,
                user_heading_stride_rad=env.config.user_heading_stride_rad,
                user_scatter_radius_km=env.config.user_scatter_radius_km,
                user_scatter_distribution=env.config.user_scatter_distribution,
                user_area_width_km=env.config.user_area_width_km,
                user_area_height_km=env.config.user_area_height_km,
                mobility_model=env.config.mobility_model,
                random_wandering_max_turn_rad=env.config.random_wandering_max_turn_rad,
                action_mask_eligibility_mode=env.config.action_mask_eligibility_mode,
            ),
            trainer_config=asdict(trainer_cfg),
            best_eval_summary=best_eval_summary,
            resume_from=(
                None
                if resumed_from is None
                else ResumeFromV1(
                    path=resumed_from["path"],
                    checkpoint_kind=resumed_from["checkpoint_kind"],
                    episode=resumed_from["episode"],
                )
            ),
            training_summary=TrainingSummaryV1(
                episodes_requested=trainer_cfg.episodes,
                episodes_completed=len(logs),
                elapsed_s=elapsed,
                final_episode_index=final_episode,
                final_scalar_reward=logs[-1].scalar_reward if logs else None,
            ),
        )
        metadata_path = write_run_metadata(artifact_paths.run_metadata_json, metadata)
        print(f"[modqn-train] metadata saved to {metadata_path}")

    return 0
