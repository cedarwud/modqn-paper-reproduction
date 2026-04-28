"""Compare paired Phase 03 MODQN-control and EE-MODQN pilot runs."""

from __future__ import annotations

import argparse

from modqn_paper_reproduction.analysis import export_phase03_paired_validation


def main() -> int:
    parser = argparse.ArgumentParser(prog="compare-phase03-ee-modqn")
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

    result = export_phase03_paired_validation(
        control_run_dir=args.control_run_dir,
        ee_run_dir=args.ee_run_dir,
        output_dir=args.output_dir,
        evaluation_seed_set=evaluation_seed_set,
    )
    summary = result["summary"]
    print(
        "[phase03-compare] decision="
        f"{summary['phase_03_decision']} "
        f"primary={summary['primary_comparison']['checkpoint_role']} "
        f"summary={result['phase03_paired_summary']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
