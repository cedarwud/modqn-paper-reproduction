"""Export-command orchestration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def run_export_command(
    args: argparse.Namespace,
    *,
    paper_id: str,
) -> int:
    from ..export.pipeline import export_training_run

    print(f"[modqn-export] paper={paper_id}")
    print(f"[modqn-export] input={Path(args.input)}")

    input_dir = Path(args.input)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else input_dir / "export-bundle"
    )
    try:
        outputs = export_training_run(
            input_dir,
            output_dir,
            replay_start_time_s=float(args.replay_start_time_s),
            replay_slot_count=(
                None
                if args.replay_slot_count is None
                else int(args.replay_slot_count)
            ),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[modqn-export] ERROR: {exc}", file=sys.stderr)
        return 2

    for key, path in outputs.items():
        print(f"[modqn-export] {key}={path}")
    return 0
