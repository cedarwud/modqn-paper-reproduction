"""Compare paired Phase 03C-C power-MDP bounded pilot runs."""

from __future__ import annotations

import argparse

from modqn_paper_reproduction.analysis import (
    export_phase03c_c_power_mdp_paired_validation,
)


def main() -> int:
    parser = argparse.ArgumentParser(prog="compare-phase03c-c-power-mdp")
    parser.add_argument("--control-run-dir", required=True)
    parser.add_argument("--candidate-run-dir", required=True)
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

    result = export_phase03c_c_power_mdp_paired_validation(
        control_run_dir=args.control_run_dir,
        candidate_run_dir=args.candidate_run_dir,
        output_dir=args.output_dir,
        evaluation_seed_set=evaluation_seed_set,
    )
    summary = result["summary"]
    print(
        "[phase03c-c-power-mdp-compare] decision="
        f"{summary['phase_03c_c_decision']} "
        f"primary={summary['primary_comparison']['checkpoint_role']} "
        f"summary={result['phase03c_c_power_mdp_summary']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
