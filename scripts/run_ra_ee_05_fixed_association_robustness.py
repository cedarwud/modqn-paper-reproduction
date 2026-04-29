"""Run RA-EE-05 fixed-association robustness and held-out validation."""

from __future__ import annotations

import argparse

from modqn_paper_reproduction.analysis.ra_ee_05_fixed_association_robustness import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    export_ra_ee_05_fixed_association_robustness,
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
    parser = argparse.ArgumentParser(prog="run-ra-ee-05-fixed-association-robustness")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--calibration-seeds",
        default=None,
        help="Optional comma-separated calibration seed override.",
    )
    parser.add_argument(
        "--held-out-seeds",
        default=None,
        help="Optional comma-separated held-out seed override.",
    )
    parser.add_argument(
        "--calibration-policies",
        default=None,
        help="Optional comma-separated calibration trajectory override.",
    )
    parser.add_argument(
        "--held-out-policies",
        default=None,
        help="Optional comma-separated held-out trajectory override.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional per-trajectory step cap for smoke validation.",
    )
    parser.add_argument(
        "--skip-oracle",
        action="store_true",
        help="Do not export constrained-oracle upper-bound diagnostics.",
    )
    args = parser.parse_args()

    result = export_ra_ee_05_fixed_association_robustness(
        config_path=args.config,
        output_dir=args.output_dir,
        calibration_seed_set=_parse_csv_ints(args.calibration_seeds),
        held_out_seed_set=_parse_csv_ints(args.held_out_seeds),
        calibration_policies=_parse_csv_strings(args.calibration_policies),
        held_out_policies=_parse_csv_strings(args.held_out_policies),
        max_steps=args.max_steps,
        include_oracle=not args.skip_oracle,
    )
    summary = result["summary"]
    proof = summary["proof_flags"]
    print(
        "[ra-ee-05-fixed-association-robustness] decision="
        f"{summary['ra_ee_05_decision']} "
        f"held_out={proof['held_out_bucket_exists_and_reported_separately']} "
        f"majority_positive="
        f"{proof['majority_noncollapsed_held_out_positive_EE_delta']} "
        f"qos={proof['p05_throughput_guardrail_pass_for_accepted_held_out']} "
        f"ranking="
        f"{proof['throughput_winner_vs_EE_winner_separate_for_accepted_held_out']} "
        f"summary={result['ra_ee_05_fixed_association_robustness_summary']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
