"""Run RA-EE-09 Slice 09E matched held-out resource-allocation replay."""

from __future__ import annotations

import argparse

from modqn_paper_reproduction.analysis.ra_ee_09_fixed_association_rb_bandwidth import (
    DEFAULT_COMPARISON_OUTPUT_DIR,
    DEFAULT_CONFIG,
    export_ra_ee_09_fixed_association_rb_bandwidth_matched_comparison,
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
        prog="run-ra-ee-09-fixed-association-rb-bandwidth-matched-comparison"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_COMPARISON_OUTPUT_DIR)
    parser.add_argument("--held-out-seeds", default=None)
    parser.add_argument("--held-out-policies", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    result = export_ra_ee_09_fixed_association_rb_bandwidth_matched_comparison(
        config_path=args.config,
        output_dir=args.output_dir,
        held_out_seed_set=_parse_csv_ints(args.held_out_seeds),
        held_out_policies=_parse_csv_strings(args.held_out_policies),
        max_steps=args.max_steps,
    )
    summary = result["summary"]
    proof = summary["proof_flags"]
    deltas = summary["matched_comparison"]["deltas"]
    print(
        "[ra-ee-09-fixed-association-rb-bandwidth-matched-comparison] decision="
        f"{summary['ra_ee_09_slice_09e_decision']} "
        f"ee_delta={deltas['simulated_EE_system_delta_bps_per_w']} "
        f"p05_ratio={deltas['p05_throughput_ratio']} "
        f"served_delta={deltas['served_ratio_delta']} "
        f"outage_delta={deltas['outage_ratio_delta']} "
        f"same_power={proof['same_effective_power_schedule_hash']} "
        f"summary={result['paired_comparison']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
