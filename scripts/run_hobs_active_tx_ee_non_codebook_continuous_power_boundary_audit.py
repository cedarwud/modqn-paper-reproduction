"""Run CP-base non-codebook continuous-power boundary audit."""

from __future__ import annotations

import argparse

from modqn_paper_reproduction.analysis.hobs_active_tx_ee_non_codebook_continuous_power_boundary_audit import (
    DEFAULT_AUDIT_CONFIG,
    DEFAULT_CANDIDATE_CONFIG,
    DEFAULT_CONTROL_CONFIG,
    DEFAULT_OUTPUT_DIR,
    export_boundary_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run-hobs-active-tx-ee-non-codebook-continuous-power-boundary-audit"
    )
    parser.add_argument("--control-config", default=DEFAULT_CONTROL_CONFIG)
    parser.add_argument("--candidate-config", default=DEFAULT_CANDIDATE_CONFIG)
    parser.add_argument("--audit-config", default=DEFAULT_AUDIT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    summary = export_boundary_audit(
        control_config_path=args.control_config,
        candidate_config_path=args.candidate_config,
        audit_config_path=args.audit_config,
        output_dir=args.output_dir,
    )
    print(
        "[hobs-active-tx-ee-non-codebook-continuous-power-boundary-audit] "
        f"decision={summary['acceptance_result']} "
        f"matched_boundary={summary['boundary_proof']['matched_boundary_pass']} "
        f"artifact={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
