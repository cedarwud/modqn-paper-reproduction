"""Phase 02 EE denominator audit command orchestration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def run_ee_denominator_audit_command(
    args: argparse.Namespace,
    *,
    paper_id: str,
) -> int:
    from ..analysis import export_ee_denominator_audit
    from ..config_loader import ConfigValidationError

    print(f"[modqn-ee-denominator-audit] paper={paper_id}")
    print(f"[modqn-ee-denominator-audit] config={Path(args.config)}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("artifacts") / "ee-denominator-audit-phase-02"
    )
    policies = tuple(
        part.strip()
        for part in str(args.policies).split(",")
        if part.strip()
    )

    try:
        outputs = export_ee_denominator_audit(
            args.config,
            output_dir,
            evaluation_seed=int(args.evaluation_seed),
            max_steps=args.max_steps,
            policies=policies,
        )
    except (ConfigValidationError, FileNotFoundError, ValueError) as exc:
        print(f"[modqn-ee-denominator-audit] ERROR: {exc}", file=sys.stderr)
        return 2

    for key, value in outputs.items():
        if key == "summary":
            continue
        print(f"[modqn-ee-denominator-audit] {key}={value}")
    print(
        "[modqn-ee-denominator-audit] decision="
        f"{outputs['summary']['phase_02_decision']['status']} "
        f"phase03={outputs['summary']['phase_02_decision']['phase_03_gate']} "
        f"classification={outputs['summary']['denominator_classification']}"
    )
    return 0
