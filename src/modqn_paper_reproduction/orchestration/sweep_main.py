"""Sweep-command orchestration."""

from __future__ import annotations

import argparse
import sys


def run_sweep_command(
    args: argparse.Namespace,
    *,
    paper_id: str,
    package_version: str,
    figure_points: tuple[float, ...] | None,
) -> int:
    from ..config_loader import ConfigValidationError, load_training_yaml
    from ..sweeps import (
        FIGURE_SUITES,
        FigurePointSelectionError,
        run_figure_suite,
        run_table_ii,
    )
    from ..analysis.reward_geometry import export_reward_geometry_analysis

    print(f"[modqn-sweeps] paper={paper_id} version={package_version}")
    print(f"[modqn-sweeps] config={args.config}")
    print(f"[modqn-sweeps] suite={args.suite}")
    print(f"[modqn-sweeps] output-dir={args.output_dir}")

    try:
        cfg = load_training_yaml(args.config)
    except ConfigValidationError as exc:
        print(f"[modqn-sweeps] ERROR: {exc}", file=sys.stderr)
        return 2

    methods = tuple(
        part.strip().lower() for part in args.methods.split(",") if part.strip()
    )

    if args.suite == "table-ii":
        outputs = run_table_ii(
            cfg,
            output_dir=args.output_dir,
            episodes=args.episodes,
            progress_every=args.progress_every,
            max_weight_rows=args.max_weight_rows,
            methods=methods,
            reference_run_dir=args.reference_run,
        )
    elif args.suite == "reward-geometry":
        if not args.input_table_ii:
            print(
                "[modqn-sweeps] ERROR: --input-table-ii is required for reward-geometry",
                file=sys.stderr,
            )
            return 2
        outputs = export_reward_geometry_analysis(
            args.output_dir,
            cfg=cfg,
            table_ii_dir=args.input_table_ii,
            reference_run_dir=args.reference_run,
        )
    elif args.suite in FIGURE_SUITES:
        try:
            outputs = run_figure_suite(
                cfg,
                suite=args.suite,
                output_dir=args.output_dir,
                episodes=args.episodes,
                progress_every=args.progress_every,
                max_points=args.max_figure_points,
                figure_points=figure_points,
                methods=methods,
                reference_run_dir=args.reference_run,
            )
        except FigurePointSelectionError as exc:
            print(f"[modqn-sweeps] ERROR: {exc}", file=sys.stderr)
            return 2
    else:
        print(f"[modqn-sweeps] ERROR: unsupported suite {args.suite!r}", file=sys.stderr)
        return 2

    for key, path in outputs.items():
        print(f"[modqn-sweeps] {key}={path}")
    return 0
