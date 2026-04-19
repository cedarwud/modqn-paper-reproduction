"""CLI entry points for MODQN paper reproduction."""

from __future__ import annotations

import argparse
import sys

from . import PAPER_ID, PACKAGE_VERSION


def _build_parser(command_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=command_name)
    parser.add_argument(
        "--config",
        default="configs/modqn-paper-baseline.resolved-template.yaml",
        help="Path to the resolved-run config file.",
    )
    return parser


def _parse_figure_points_arg(raw: str) -> tuple[float, ...]:
    """Parse a comma-separated explicit figure point list."""
    parts = [part.strip() for part in raw.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError("figure point override must be a comma-separated numeric list")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(
            "figure point override must contain only numeric values"
        ) from exc


def train_main(argv: list[str] | None = None) -> int:
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

    from .orchestration.train_main import run_train_command

    return run_train_command(
        args,
        paper_id=PAPER_ID,
        package_version=PACKAGE_VERSION,
    )


def sweep_main(argv: list[str] | None = None) -> int:
    parser = _build_parser("modqn-sweeps")
    parser.add_argument(
        "--suite",
        choices=["table-ii", "reward-geometry", "fig-3", "fig-4", "fig-5", "fig-6"],
        default="table-ii",
        help="Sweep/export suite to run.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override episode count for sweep training jobs.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N episodes (0 = silent).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write sweep outputs.",
    )
    parser.add_argument(
        "--max-weight-rows",
        type=int,
        default=None,
        help="Optional limit on Table II weight rows for quick runs.",
    )
    parser.add_argument(
        "--max-figure-points",
        type=int,
        default=None,
        help="Optional limit on Fig. 3-6 point counts for quick runs.",
    )
    parser.add_argument(
        "--figure-points",
        default=None,
        help="Optional explicit comma-separated Fig. 3-6 point override, for example 160,180,200.",
    )
    parser.add_argument(
        "--methods",
        default="modqn,dqn_throughput,dqn_scalar,rss_max",
        help="Comma-separated method list.",
    )
    parser.add_argument(
        "--reference-run",
        default=None,
        help="Optional reference training artifact directory for analysis linkage.",
    )
    parser.add_argument(
        "--input-table-ii",
        default=None,
        help="Existing Table II artifact directory for analysis-only suites.",
    )
    args = parser.parse_args(argv)

    from .sweeps import FIGURE_SUITES
    from .orchestration.sweep_main import run_sweep_command

    figure_points = None
    if args.figure_points is not None:
        if args.suite not in FIGURE_SUITES:
            print(
                "[modqn-sweeps] ERROR: --figure-points is only supported for fig-3 to fig-6",
                file=sys.stderr,
            )
            return 2
        if args.max_figure_points is not None:
            print(
                "[modqn-sweeps] ERROR: --figure-points cannot be combined with --max-figure-points",
                file=sys.stderr,
            )
            return 2
        try:
            figure_points = _parse_figure_points_arg(args.figure_points)
        except ValueError as exc:
            print(f"[modqn-sweeps] ERROR: {exc}", file=sys.stderr)
            return 2

    return run_sweep_command(
        args,
        paper_id=PAPER_ID,
        package_version=PACKAGE_VERSION,
        figure_points=figure_points,
    )


def export_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="modqn-export")
    parser.add_argument(
        "--input", required=True,
        help="Path to a completed run artifact directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write exported CSV/PNG bundle surfaces.",
    )
    args = parser.parse_args(argv)

    from .orchestration.export_main import run_export_command

    return run_export_command(args, paper_id=PAPER_ID)
