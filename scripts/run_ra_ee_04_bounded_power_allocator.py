"""Run the RA-EE-04 fixed-association power-allocation pilot."""

from __future__ import annotations

import argparse

from modqn_paper_reproduction.analysis.ra_ee_04_bounded_power_allocator import (
    DEFAULT_CANDIDATE_CONFIG,
    DEFAULT_CANDIDATE_OUTPUT_DIR,
    DEFAULT_COMPARISON_OUTPUT_DIR,
    DEFAULT_CONTROL_CONFIG,
    DEFAULT_CONTROL_OUTPUT_DIR,
    export_ra_ee_04_bounded_power_allocator_pilot,
)


def _parse_csv_ints(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _parse_csv_strings(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser(prog="run-ra-ee-04-bounded-power-allocator")
    parser.add_argument("--control-config", default=DEFAULT_CONTROL_CONFIG)
    parser.add_argument("--candidate-config", default=DEFAULT_CANDIDATE_CONFIG)
    parser.add_argument("--control-output-dir", default=DEFAULT_CONTROL_OUTPUT_DIR)
    parser.add_argument("--candidate-output-dir", default=DEFAULT_CANDIDATE_OUTPUT_DIR)
    parser.add_argument("--comparison-output-dir", default=DEFAULT_COMPARISON_OUTPUT_DIR)
    parser.add_argument(
        "--evaluation-seeds",
        default=None,
        help="Optional comma-separated evaluation seed override.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional per-trajectory step cap for smoke validation.",
    )
    parser.add_argument(
        "--policies",
        default="hold-current,random-valid,spread-valid",
        help="Comma-separated fixed association trajectory policies.",
    )
    parser.add_argument(
        "--skip-oracle",
        action="store_true",
        help="Do not export constrained-oracle upper-bound diagnostics.",
    )
    args = parser.parse_args()

    result = export_ra_ee_04_bounded_power_allocator_pilot(
        control_config_path=args.control_config,
        candidate_config_path=args.candidate_config,
        control_output_dir=args.control_output_dir,
        candidate_output_dir=args.candidate_output_dir,
        comparison_output_dir=args.comparison_output_dir,
        evaluation_seed_set=_parse_csv_ints(args.evaluation_seeds),
        max_steps=args.max_steps,
        policies=_parse_csv_strings(args.policies),
        include_oracle=not args.skip_oracle,
    )
    summary = result["summary"]
    proof = summary["proof_flags"]
    print(
        "[ra-ee-04-bounded-power-allocator] decision="
        f"{summary['ra_ee_04_decision']} "
        f"denominator_varies={proof['denominator_varies_in_eval']} "
        f"qos={proof['QoS_guardrails_pass']} "
        f"ranking_separates={proof['ranking_separates_or_rescore_changes']} "
        f"summary={result['ra_ee_04_bounded_power_allocator_summary']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
