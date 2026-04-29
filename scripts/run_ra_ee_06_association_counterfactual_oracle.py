"""Run RA-EE-06 association counterfactual / oracle design gate."""

from __future__ import annotations

import argparse

from modqn_paper_reproduction.analysis.ra_ee_06_association_counterfactual_oracle import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    export_ra_ee_06_association_counterfactual_oracle,
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
        prog="run-ra-ee-06-association-counterfactual-oracle"
    )
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
        "--candidate-association-policies",
        default=None,
        help="Optional comma-separated candidate association policy override.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional per-policy step cap for smoke validation.",
    )
    parser.add_argument(
        "--skip-oracle",
        action="store_true",
        help="Do not export constrained association + power oracle diagnostics.",
    )
    parser.add_argument(
        "--skip-fixed-1w-control",
        action="store_true",
        help="Do not export the fixed-1W matched-control diagnostic rows.",
    )
    args = parser.parse_args()

    result = export_ra_ee_06_association_counterfactual_oracle(
        config_path=args.config,
        output_dir=args.output_dir,
        calibration_seed_set=_parse_csv_ints(args.calibration_seeds),
        held_out_seed_set=_parse_csv_ints(args.held_out_seeds),
        candidate_association_policies=_parse_csv_strings(
            args.candidate_association_policies
        ),
        max_steps=args.max_steps,
        include_oracle=not args.skip_oracle,
        include_fixed_1w_control=not args.skip_fixed_1w_control,
    )
    summary = result["summary"]
    proof = summary["proof_flags"]
    print(
        "[ra-ee-06-association-counterfactual-oracle] decision="
        f"{summary['ra_ee_06_decision']} "
        f"held_out={proof['held_out_bucket_exists_and_reported_separately']} "
        f"accepted={proof['majority_noncollapsed_held_out_accepted']} "
        f"qos={proof['p05_throughput_guardrail_pass_for_accepted_held_out']} "
        f"noncollapse="
        f"{proof['one_active_beam_collapse_avoided_for_accepted_held_out']} "
        f"summary={result['ra_ee_06_association_counterfactual_oracle_summary']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
