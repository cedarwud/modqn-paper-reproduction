"""CLI entry points for MODQN paper reproduction.

These are the real training / sweep / export entry points,
not scaffold banners.  Each loads the resolved-run config,
builds the env and trainer, and runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from . import PAPER_ID, PACKAGE_VERSION


def _build_parser(command_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=command_name)
    parser.add_argument(
        "--config",
        default="configs/modqn-paper-baseline.resolved-template.yaml",
        help="Path to the resolved-run config file.",
    )
    return parser


def train_main(argv: list[str] | None = None) -> int:
    """Train MODQN on the paper baseline scenario."""
    parser = _build_parser("modqn-train")
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Override episode count (for quick tests).",
    )
    parser.add_argument(
        "--progress-every", type=int, default=100,
        help="Print progress every N episodes (0 = silent).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to write training logs (JSON).",
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Optional checkpoint file to load before training.",
    )
    args = parser.parse_args(argv)

    from .config_loader import (
        ConfigValidationError,
        build_environment,
        build_trainer_config,
        get_seeds,
        load_training_yaml,
    )
    from .algorithms.modqn import MODQNTrainer

    print(f"[modqn-train] paper={PAPER_ID} version={PACKAGE_VERSION}")
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

    # Build trainer config (with optional episode override)
    from dataclasses import fields as dc_fields
    from .algorithms.modqn import TrainerConfig

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
        "(secondary save not yet implemented in this pass)"
    )

    # Build and run trainer
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
    logs = trainer.train(progress_every=args.progress_every)
    elapsed = time.time() - t0

    print(f"[modqn-train] done: {len(logs)} episodes in {elapsed:.1f}s")

    if logs:
        final = logs[-1]
        print(
            f"[modqn-train] final: scalar={final.scalar_reward:.4e}, "
            f"r1={final.r1_mean:.4e}, r2={final.r2_mean:.4f}, "
            f"r3={final.r3_mean:.4e}, ho={final.total_handovers}"
        )

    # Save logs if output dir specified
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        log_path = out_dir / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(
                [
                    {
                        "episode": l.episode,
                        "epsilon": l.epsilon,
                        "r1_mean": l.r1_mean,
                        "r2_mean": l.r2_mean,
                        "r3_mean": l.r3_mean,
                        "scalar_reward": l.scalar_reward,
                        "total_handovers": l.total_handovers,
                        "replay_size": l.replay_size,
                        "losses": list(l.losses),
                    }
                    for l in logs
                ],
                f,
                indent=2,
            )
        print(f"[modqn-train] logs saved to {log_path}")

        final_episode = logs[-1].episode if logs else (
            resumed_from["episode"] if resumed_from else -1
        )
        checkpoint_dir = out_dir / "checkpoints"
        final_checkpoint_path = trainer.save_checkpoint(
            checkpoint_dir / f"{trainer_cfg.checkpoint_primary_report}.pt",
            episode=final_episode,
            checkpoint_kind=trainer_cfg.checkpoint_primary_report,
            logs=logs,
            include_optimizers=True,
        )
        print(f"[modqn-train] final checkpoint saved to {final_checkpoint_path}")

        metadata = {
            "paper_id": PAPER_ID,
            "package_version": PACKAGE_VERSION,
            "config_path": str(Path(args.config)),
            "config_role": cfg.get("config_role"),
            "seeds": seeds,
            "checkpoint_rule": trainer.checkpoint_rule(),
            "checkpoint_files": {
                "primary_final": str(final_checkpoint_path),
                "secondary_best_eval": None,
            },
            "resolved_assumptions": cfg.get("resolved_assumptions", {}),
            "runtime_environment": {
                "num_users": env.config.num_users,
                "num_satellites": env.orbit.num_satellites,
                "beams_per_satellite": env.beam_pattern.num_beams,
                "r3_gap_scope": env.config.r3_gap_scope,
                "r3_empty_beam_throughput": env.config.r3_empty_beam_throughput,
                "user_heading_stride_rad": env.config.user_heading_stride_rad,
                "user_scatter_radius_km": env.config.user_scatter_radius_km,
                "user_scatter_distribution": env.config.user_scatter_distribution,
            },
            "trainer_config": asdict(trainer_cfg),
            "resume_from": resumed_from,
            "training_summary": {
                "episodes_requested": trainer_cfg.episodes,
                "episodes_completed": len(logs),
                "elapsed_s": elapsed,
                "final_episode_index": final_episode,
                "final_scalar_reward": logs[-1].scalar_reward if logs else None,
            },
        }
        metadata_path = out_dir / "run_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[modqn-train] metadata saved to {metadata_path}")

    return 0


def sweep_main(argv: list[str] | None = None) -> int:
    parser = _build_parser("modqn-sweeps")
    args = parser.parse_args(argv)
    print(f"[modqn-sweeps] paper={PAPER_ID} version={PACKAGE_VERSION}")
    print(f"[modqn-sweeps] config={args.config}")
    print("[modqn-sweeps] sweep runner not yet implemented (Phase 1 focus: MODQN trainer)")
    print("[modqn-sweeps] next-step: implement Table II and Fig. 3-6 sweep runners")
    return 0


def export_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="modqn-export")
    parser.add_argument(
        "--input", required=True,
        help="Path to a completed run artifact directory.",
    )
    args = parser.parse_args(argv)
    print(f"[modqn-export] paper={PAPER_ID}")
    print(f"[modqn-export] input={Path(args.input)}")
    print("[modqn-export] export not yet implemented (Phase 2)")
    return 0
