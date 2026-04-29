"""Run the RA-EE-02 offline oracle power-allocation audit."""

from __future__ import annotations

import argparse

from modqn_paper_reproduction.analysis.ra_ee_02_oracle_power_allocation import (
    DEFAULT_PHASE03C_C_LEARNED_RUN_DIR,
    export_ra_ee_02_oracle_power_allocation_audit,
)


def _parse_csv_ints(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _parse_csv_strings(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser(prog="audit-ra-ee-02-oracle-power-allocation")
    parser.add_argument(
        "--config",
        default="configs/ra-ee-02-oracle-power-allocation-audit.resolved.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/ra-ee-02-oracle-power-allocation-audit",
    )
    parser.add_argument(
        "--learned-run-dir",
        default=DEFAULT_PHASE03C_C_LEARNED_RUN_DIR,
        help="Phase 03C-C candidate run to use for learned trajectory if available.",
    )
    parser.add_argument(
        "--skip-learned",
        action="store_true",
        help="Do not include the Phase 03C-C learned trajectory.",
    )
    parser.add_argument(
        "--evaluation-seeds",
        default=None,
        help="Optional comma-separated evaluation seed override.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional per-trajectory step cap for bounded smoke audits.",
    )
    parser.add_argument(
        "--policies",
        default="hold-current,random-valid,spread-valid",
        help="Comma-separated fixed association trajectory policies.",
    )
    parser.add_argument(
        "--power-candidates",
        default=(
            "fixed-control,load-concave,budget-trim,"
            "qos-tail-boost,constrained-oracle"
        ),
        help="Comma-separated power-allocation candidates.",
    )
    args = parser.parse_args()

    result = export_ra_ee_02_oracle_power_allocation_audit(
        config_path=args.config,
        output_dir=args.output_dir,
        learned_run_dir=args.learned_run_dir,
        include_learned=not args.skip_learned,
        evaluation_seed_set=_parse_csv_ints(args.evaluation_seeds),
        max_steps=args.max_steps,
        policies=_parse_csv_strings(args.policies),
        power_candidates=_parse_csv_strings(args.power_candidates),
    )
    summary = result["summary"]
    proof = summary["proof_flags"]
    print(
        "[ra-ee-02-oracle-power-allocation] decision="
        f"{summary['ra_ee_02_decision']} "
        "denominator_changed="
        f"{proof['denominator_changed_by_power_decision']} "
        "ranking_separates="
        f"{proof['ranking_separates_under_same_policy_rescore']} "
        "beats_fixed_control="
        f"{proof['oracle_or_heuristic_beats_fixed_control_on_EE']} "
        f"summary={result['ra_ee_02_oracle_power_allocation_summary']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
