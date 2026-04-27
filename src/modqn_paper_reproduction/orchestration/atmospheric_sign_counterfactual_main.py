"""Atmospheric-sign counterfactual evaluation orchestration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def run_atmospheric_sign_counterfactual_command(
    args: argparse.Namespace,
    *,
    paper_id: str,
) -> int:
    from ..analysis.atmospheric_sign_counterfactual import (
        export_atmospheric_sign_counterfactual_eval,
    )

    print(f"[modqn-atmospheric-sign-counterfactual] paper={paper_id}")
    print(f"[modqn-atmospheric-sign-counterfactual] input={Path(args.input)}")

    input_dir = Path(args.input)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else input_dir / "atmospheric-sign-counterfactual-audit"
    )
    try:
        outputs = export_atmospheric_sign_counterfactual_eval(
            input_dir,
            output_dir,
            evaluation_seed=args.evaluation_seed,
            max_steps=args.max_steps,
            max_users=args.max_users,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[modqn-atmospheric-sign-counterfactual] ERROR: {exc}", file=sys.stderr)
        return 2

    for key, value in outputs.items():
        if key == "summary":
            continue
        print(f"[modqn-atmospheric-sign-counterfactual] {key}={value}")
    print(
        "[modqn-atmospheric-sign-counterfactual] change-scope="
        f"diagnostics:{outputs['summary']['interpretation']['diagnostics_change_scope']} "
        f"modqn:{outputs['summary']['interpretation']['modqn_change_scope']} "
        f"rss_max:{outputs['summary']['interpretation']['rss_max_change_scope']}"
    )
    return 0
