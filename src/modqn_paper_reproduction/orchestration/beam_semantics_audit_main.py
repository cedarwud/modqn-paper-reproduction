"""Beam-semantics audit command orchestration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def run_beam_semantics_audit_command(
    args: argparse.Namespace,
    *,
    paper_id: str,
) -> int:
    from ..analysis import export_beam_semantics_audit

    print(f"[modqn-beam-audit] paper={paper_id}")
    print(f"[modqn-beam-audit] input={Path(args.input)}")

    input_dir = Path(args.input)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else input_dir / "beam-semantics-audit"
    )

    try:
        outputs = export_beam_semantics_audit(
            input_dir,
            output_dir,
            evaluation_seed=args.evaluation_seed,
            max_steps=args.max_steps,
            max_users=args.max_users,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[modqn-beam-audit] ERROR: {exc}", file=sys.stderr)
        return 2

    for key, value in outputs.items():
        if key == "summary":
            continue
        print(f"[modqn-beam-audit] {key}={value}")
    print(
        "[modqn-beam-audit] collapse="
        f"mask:{outputs['summary']['valid_mask_collapse_classification']} "
        f"channel:{outputs['summary']['channel_value_collapse_classification']} "
        f"rss:{outputs['summary']['comparator_degeneration_classification']}"
    )
    return 0
