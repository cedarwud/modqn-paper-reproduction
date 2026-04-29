"""Run RA-EE-07 constrained-power allocator distillation gate."""

from __future__ import annotations

import argparse

from modqn_paper_reproduction.analysis.ra_ee_07_constrained_power_allocator_distillation import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    export_ra_ee_07_constrained_power_allocator_distillation,
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
        prog="run-ra-ee-07-constrained-power-allocator-distillation"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--calibration-seeds", default=None)
    parser.add_argument("--held-out-seeds", default=None)
    parser.add_argument("--diagnostic-seeds", default=None)
    parser.add_argument("--calibration-policies", default=None)
    parser.add_argument("--held-out-policies", default=None)
    parser.add_argument("--diagnostic-association-policies", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--skip-oracle",
        action="store_true",
        help="Do not export constrained-power oracle diagnostics.",
    )
    parser.add_argument(
        "--skip-association-diagnostics",
        action="store_true",
        help="Do not export RA-EE-06B association proposal diagnostic buckets.",
    )
    args = parser.parse_args()

    result = export_ra_ee_07_constrained_power_allocator_distillation(
        config_path=args.config,
        output_dir=args.output_dir,
        calibration_seed_set=_parse_csv_ints(args.calibration_seeds),
        held_out_seed_set=_parse_csv_ints(args.held_out_seeds),
        diagnostic_seed_set=_parse_csv_ints(args.diagnostic_seeds),
        calibration_policies=_parse_csv_strings(args.calibration_policies),
        held_out_policies=_parse_csv_strings(args.held_out_policies),
        diagnostic_association_policies=_parse_csv_strings(
            args.diagnostic_association_policies
        ),
        max_steps=args.max_steps,
        include_oracle=not args.skip_oracle,
        include_association_diagnostics=not args.skip_association_diagnostics,
    )
    summary = result["summary"]
    proof = summary["proof_flags"]
    held_out = summary["bucket_results"].get("held-out", {})
    print(
        "[ra-ee-07-constrained-power-allocator-distillation] decision="
        f"{summary['ra_ee_07_decision']} "
        f"majority_positive="
        f"{proof['majority_noncollapsed_held_out_positive_EE_delta']} "
        f"oracle_gap_closure={held_out.get('aggregate_oracle_gap_closed_ratio')} "
        f"qos={proof['p05_throughput_guardrail_pass_for_accepted_held_out']} "
        f"summary={result['ra_ee_07_constrained_power_allocator_distillation_summary']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
