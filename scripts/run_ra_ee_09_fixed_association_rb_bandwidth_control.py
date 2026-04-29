"""Run RA-EE-09 fixed-association RB / bandwidth control replay."""

from __future__ import annotations

import argparse

from modqn_paper_reproduction.analysis.ra_ee_09_fixed_association_rb_bandwidth import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    export_ra_ee_09_fixed_association_rb_bandwidth_control,
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
        prog="run-ra-ee-09-fixed-association-rb-bandwidth-control"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--calibration-seeds", default=None)
    parser.add_argument("--held-out-seeds", default=None)
    parser.add_argument("--calibration-policies", default=None)
    parser.add_argument("--held-out-policies", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    result = export_ra_ee_09_fixed_association_rb_bandwidth_control(
        config_path=args.config,
        output_dir=args.output_dir,
        calibration_seed_set=_parse_csv_ints(args.calibration_seeds),
        held_out_seed_set=_parse_csv_ints(args.held_out_seeds),
        calibration_policies=_parse_csv_strings(args.calibration_policies),
        held_out_policies=_parse_csv_strings(args.held_out_policies),
        max_steps=args.max_steps,
    )
    summary = result["summary"]
    proof = summary["proof_flags"]
    print(
        "[ra-ee-09-fixed-association-rb-bandwidth-control] decision="
        f"{summary['ra_ee_09_slice_09a_09c_decision']} "
        f"parity={proof['equal_share_throughput_parity']} "
        f"resource_sum={proof['active_beam_resource_sum_exact']} "
        f"inactive_zero={proof['inactive_beam_zero_resource']} "
        f"same_power={proof['same_power_vector_as_control']} "
        f"summary={result['ra_ee_09_fixed_association_rb_bandwidth_control_summary']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
