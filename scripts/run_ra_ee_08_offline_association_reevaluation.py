"""Run RA-EE-08 offline association re-evaluation gate."""

from __future__ import annotations

import argparse

from modqn_paper_reproduction.analysis.ra_ee_08_offline_association_reevaluation import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    export_ra_ee_08_offline_association_reevaluation,
)


def _parse_csv_ints(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _parse_csv_strings(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run-ra-ee-08-offline-association-reevaluation"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--calibration-seeds", default=None)
    parser.add_argument("--held-out-seeds", default=None)
    parser.add_argument("--candidate-association-policies", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--skip-oracle",
        action="store_true",
        help="Do not export constrained or association-oracle diagnostic rows.",
    )
    args = parser.parse_args()

    result = export_ra_ee_08_offline_association_reevaluation(
        config_path=args.config,
        output_dir=args.output_dir,
        calibration_seed_set=_parse_csv_ints(args.calibration_seeds),
        held_out_seed_set=_parse_csv_ints(args.held_out_seeds),
        candidate_association_policies=_parse_csv_strings(
            args.candidate_association_policies
        ),
        max_steps=args.max_steps,
        include_oracle=not args.skip_oracle,
    )
    summary = result["summary"]
    proof = summary["proof_flags"]
    held_out = summary["bucket_results"].get("held-out", {})
    print(
        "[ra-ee-08-offline-association-reevaluation] decision="
        f"{summary['ra_ee_08_decision']} "
        f"positive={proof['majority_or_predeclared_primary_held_out_positive_EE_delta']} "
        f"accepted={proof['majority_or_predeclared_primary_held_out_accepted']} "
        f"qos={proof['p05_throughput_guardrail_pass_for_accepted_held_out']} "
        f"oracle_gap={held_out.get('aggregate_oracle_gap_closed_ratio')} "
        f"summary={result['ra_ee_08_offline_association_reevaluation_summary']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
