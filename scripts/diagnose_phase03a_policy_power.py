"""Run Phase 03A diagnostic-only policy power and diversity reports."""

from __future__ import annotations

import argparse

from modqn_paper_reproduction.analysis.phase03a_diagnostics import (
    export_phase03a_diagnostics,
)


def main() -> int:
    parser = argparse.ArgumentParser(prog="diagnose-phase03a-policy-power")
    parser.add_argument("--control-run-dir", required=True)
    parser.add_argument("--ee-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--evaluation-seeds",
        default=None,
        help="Optional comma-separated eval seed override. Defaults to run metadata.",
    )
    args = parser.parse_args()

    evaluation_seed_set = None
    if args.evaluation_seeds:
        evaluation_seed_set = tuple(
            int(part.strip())
            for part in args.evaluation_seeds.split(",")
            if part.strip()
        )

    result = export_phase03a_diagnostics(
        control_run_dir=args.control_run_dir,
        ee_run_dir=args.ee_run_dir,
        output_dir=args.output_dir,
        evaluation_seed_set=evaluation_seed_set,
    )
    summary = result["summary"]
    root = summary["root_cause_assessment"]
    print(
        "[phase03a-diagnostics] decision="
        f"{summary['phase_03_decision']} "
        "env_denominator_varies="
        f"{root['denominator_variability_exists_in_environment']} "
        "learned_denominator_varies="
        f"{root['learned_policies_exercise_denominator_variability']} "
        f"summary={result['phase03a_diagnostic_summary']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
