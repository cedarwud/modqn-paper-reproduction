"""Run the Phase 03C-B static/counterfactual power-MDP audit."""

from __future__ import annotations

import argparse

from modqn_paper_reproduction.analysis.phase03c_b_power_mdp_audit import (
    DEFAULT_PHASE03B_EE_RUN_DIR,
    export_phase03c_b_power_mdp_audit,
)


def _parse_csv_ints(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _parse_csv_strings(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser(prog="audit-phase03c-b-power-mdp")
    parser.add_argument(
        "--config",
        default="configs/ee-modqn-phase-03c-b-power-mdp-audit.resolved.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/ee-modqn-phase-03c-b-power-mdp-audit",
    )
    parser.add_argument(
        "--learned-run-dir",
        default=DEFAULT_PHASE03B_EE_RUN_DIR,
        help="Phase 03B EE run to use for learned trajectory if available.",
    )
    parser.add_argument(
        "--skip-learned",
        action="store_true",
        help="Do not include the Phase 03B learned policy trajectory.",
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
        help="Comma-separated fixed handover trajectory policies.",
    )
    parser.add_argument(
        "--power-semantics",
        default=(
            "fixed-2w,phase-02b-proxy,fixed-low,fixed-mid,fixed-high,"
            "load-concave,qos-tail-boost,budget-trim"
        ),
        help="Comma-separated power semantics candidates.",
    )
    args = parser.parse_args()

    result = export_phase03c_b_power_mdp_audit(
        config_path=args.config,
        output_dir=args.output_dir,
        learned_run_dir=args.learned_run_dir,
        include_learned=not args.skip_learned,
        evaluation_seed_set=_parse_csv_ints(args.evaluation_seeds),
        max_steps=args.max_steps,
        policies=_parse_csv_strings(args.policies),
        power_semantics=_parse_csv_strings(args.power_semantics),
    )
    summary = result["summary"]
    decision = summary["phase_03c_b_decision"]
    print(
        "[phase03c-b-power-mdp-audit] decision="
        f"{decision['phase_03c_b_decision']} "
        "denominator_changed="
        f"{decision['denominator_changed_by_power_decision']} "
        "ranking_separates="
        f"{decision['ranking_separates_under_same_policy_rescore']} "
        f"summary={result['phase03c_b_power_mdp_audit_summary']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
