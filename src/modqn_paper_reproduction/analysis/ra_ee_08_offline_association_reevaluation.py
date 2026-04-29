"""RA-EE-08 offline association re-evaluation gate.

This module is the public exporter shell.  Protocol parsing, deterministic
replay, and gate metrics live in the RA-EE-08 helper modules so each surface has
a narrow responsibility.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..config_loader import build_environment, load_training_yaml
from ..env.step import HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK
from ._common import write_json
from .ra_ee_02_oracle_power_allocation import _build_unit_power_snapshots, _summarize_all
from .ra_ee_07_constrained_power_allocator_distillation import _fieldnames
from .ra_ee_08_metrics import (
    _augment_summaries,
    _build_decision,
    _build_guardrail_checks,
    _bucket_results,
    _compact_summary_rows,
    _guardrail_result,
    _oracle_gap_diagnostics,
    _ranking_checks,
    _seed_level_results,
)
from .ra_ee_08_protocol import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    RA_EE_08_ASSOC_ORACLE_CONSTRAINED,
    RA_EE_08_ASSOC_ORACLE_DEPLOYABLE,
    RA_EE_08_CANDIDATE,
    RA_EE_08_FIXED_CONSTRAINED_ORACLE,
    RA_EE_08_FIXED_DEPLOYABLE_CONTROL,
    RA_EE_08_FIXED_SAFE_GREEDY,
    RA_EE_08_METHOD_LABEL,
    RA_EE_08_PROPOSAL_POLICIES,
    RA_EE_08_PROPOSAL_SAFE_GREEDY,
    _RAEE08Settings,
    _select_actions_for_association_policy,
    _settings_from_config,
    _validate_association_policies,
)
from .ra_ee_08_replay import _evaluation_rows, _rollout_association_trajectories
from .ra_ee_05_fixed_association_robustness import (
    CALIBRATION_BUCKET,
    HELD_OUT_BUCKET,
    _BucketSpec,
)


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: list[str],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path

def _review_lines(summary: dict[str, Any]) -> list[str]:
    decision = summary["ra_ee_08_decision"]
    proof = summary["proof_flags"]
    held_out = summary["bucket_results"][HELD_OUT_BUCKET]
    lines = [
        "# RA-EE-08 Offline Association Re-Evaluation Review",
        "",
        "Offline deterministic association proposal replay only. Primary rows "
        "compare fixed association + deployable stronger power allocator against "
        "proposal association + the same deployable stronger power allocator. "
        "Safe-greedy and oracle rows are diagnostic-only. No training, learned "
        "association, hierarchical RL, joint association + power training, "
        "Catfish, multi-Catfish, RB / bandwidth allocation, oracle runtime method, "
        "or frozen baseline mutation was performed.",
        "",
        "## Protocol",
        "",
        f"- method label: `{summary['protocol']['method_label']}`",
        f"- implementation sublabel: `{summary['protocol']['implementation_sublabel']}`",
        f"- config: `{summary['inputs']['config_path']}`",
        f"- artifact namespace: `{summary['inputs']['output_dir']}`",
        f"- primary control: `{summary['protocol']['primary_control']}`",
        f"- primary candidate: `{summary['protocol']['primary_candidate']}`",
        f"- proposal families: `{summary['protocol']['candidate_association_policies']}`",
        f"- deployable allocator candidates: `{summary['protocol']['deployable_allocator_candidates']}`",
        f"- primary deployable allocator: `{summary['protocol']['primary_deployable_allocator']}`",
        f"- diagnostics: `{summary['protocol']['diagnostic_rows']}`",
        "",
        "## Held-Out Gate",
        "",
        f"- noncollapsed candidates: `{held_out['noncollapsed_candidate_policies']}`",
        f"- positive EE delta candidates: `{held_out['positive_EE_delta_candidate_policies']}`",
        f"- accepted candidates: `{held_out['accepted_candidate_policies']}`",
        f"- rejection reasons: `{held_out['rejection_reasons']}`",
        f"- majority or predeclared primary positive EE delta: `{held_out['majority_or_predeclared_primary_positive_EE_delta']}`",
        f"- gains not concentrated in one policy: `{held_out['gains_not_concentrated_in_one_policy']}`",
        f"- gains not concentrated in one seed: `{held_out['gains_not_concentrated_in_one_seed']}`",
        f"- QoS guardrails pass for accepted: `{held_out['qos_guardrails_pass_for_accepted']}`",
        f"- handover burden bounded: `{held_out['handover_burden_bounded_for_accepted']}`",
        f"- aggregate oracle gap closure: `{held_out['aggregate_oracle_gap_closed_ratio']}`",
        "",
        "## Gate Flags",
        "",
    ]
    for key, value in proof.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- RA-EE-08 decision: `{decision}`",
            f"- allowed claim: {summary['decision_detail']['allowed_claim']}",
            "",
            "## Forbidden Claims",
            "",
        ]
    )
    for claim in summary["forbidden_claims_still_active"]:
        lines.append(f"- {claim}")
    return lines


def export_ra_ee_08_offline_association_reevaluation(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    calibration_seed_set: tuple[int, ...] | None = None,
    held_out_seed_set: tuple[int, ...] | None = None,
    candidate_association_policies: tuple[str, ...] | None = None,
    max_steps: int | None = None,
    include_oracle: bool = True,
) -> dict[str, Any]:
    """Export RA-EE-08 offline association re-evaluation artifacts."""
    if max_steps is not None and max_steps < 1:
        raise ValueError(f"max_steps must be >= 1 when provided, got {max_steps!r}")
    cfg = load_training_yaml(config_path)
    env = build_environment(cfg)
    if env.power_surface_config.hobs_power_surface_mode != (
        HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK
    ):
        raise ValueError("RA-EE-08 config must opt into the power-codebook surface.")

    settings = _settings_from_config(cfg)
    bucket_specs: list[_BucketSpec] = []
    for spec in settings.bucket_specs:
        bucket_specs.append(
            _BucketSpec(
                name=spec.name,
                trajectory_families=spec.trajectory_families,
                evaluation_seed_set=(
                    tuple(calibration_seed_set)
                    if spec.name == CALIBRATION_BUCKET and calibration_seed_set is not None
                    else (
                        tuple(held_out_seed_set)
                        if spec.name == HELD_OUT_BUCKET and held_out_seed_set is not None
                        else spec.evaluation_seed_set
                    )
                ),
            )
        )
    run_candidates = (
        tuple(candidate_association_policies)
        if candidate_association_policies is not None
        else settings.candidate_association_policies
    )
    _validate_association_policies("candidate policies", run_candidates)
    run_settings = _RAEE08Settings(
        method_label=settings.method_label,
        implementation_sublabel=settings.implementation_sublabel,
        audit=settings.audit,
        bucket_specs=tuple(bucket_specs),
        matched_control_association_policy=settings.matched_control_association_policy,
        candidate_association_policies=run_candidates,
        oracle_association_policies=tuple(
            policy for policy in settings.oracle_association_policies if policy in run_candidates
        ),
        predeclared_primary_association_policy=(
            settings.predeclared_primary_association_policy
            if settings.predeclared_primary_association_policy in run_candidates
            else None
        ),
        deployable_allocators=settings.deployable_allocators,
        primary_deployable_allocator=settings.primary_deployable_allocator,
        min_active_beams=settings.min_active_beams,
        max_active_beams=settings.max_active_beams,
        target_users_per_active_beam=settings.target_users_per_active_beam,
        load_cap_overflow_users=settings.load_cap_overflow_users,
        candidate_max_demoted_beams=settings.candidate_max_demoted_beams,
        candidate_step_p05_guardrail_margin=settings.candidate_step_p05_guardrail_margin,
        max_one_active_beam_ratio_for_acceptance=(
            settings.max_one_active_beam_ratio_for_acceptance
        ),
        max_two_beam_overload_step_ratio=settings.max_two_beam_overload_step_ratio,
        max_moved_user_ratio=settings.max_moved_user_ratio,
        max_moved_user_ratio_per_step=settings.max_moved_user_ratio_per_step,
        min_oracle_gap_closed_ratio=settings.min_oracle_gap_closed_ratio,
        quality_margin_for_move=settings.quality_margin_for_move,
        local_search_swap_limit=settings.local_search_swap_limit,
        trace_top_k=settings.trace_top_k,
        p05_trim_max_moves=settings.p05_trim_max_moves,
        local_search_max_moves=settings.local_search_max_moves,
        dp_max_profile_count=settings.dp_max_profile_count,
    )

    trajectories, association_metadata, traces_by_key = _rollout_association_trajectories(
        cfg=cfg,
        settings=run_settings,
        max_steps=max_steps,
    )
    snapshots = _build_unit_power_snapshots(
        base_cfg=cfg,
        settings=run_settings.audit,
        trajectories=trajectories,
    )
    step_rows, user_throughputs_by_key = _evaluation_rows(
        snapshots=snapshots,
        traces_by_key=traces_by_key,
        settings=run_settings,
        include_oracle=include_oracle,
    )
    summaries = _summarize_all(
        rows=step_rows,
        user_throughputs_by_key=user_throughputs_by_key,
    )
    summaries = _augment_summaries(
        summaries,
        step_rows=step_rows,
        settings=run_settings,
    )
    guardrail_checks = _build_guardrail_checks(
        summaries=summaries,
        settings=run_settings,
    )
    ranking_checks = _ranking_checks(summaries)
    oracle_gap_diagnostics = _oracle_gap_diagnostics(summaries)
    seed_results = _seed_level_results(step_rows=step_rows, bucket=HELD_OUT_BUCKET)
    bucket_results = _bucket_results(
        settings=run_settings,
        summaries=summaries,
        guardrail_checks=guardrail_checks,
        ranking_checks=ranking_checks,
        oracle_gap_diagnostics=oracle_gap_diagnostics,
        seed_results=seed_results,
    )
    decision_detail = _build_decision(
        summaries=summaries,
        guardrail_checks=guardrail_checks,
        bucket_results=bucket_results,
        include_oracle=include_oracle,
    )

    out_dir = Path(output_dir)
    step_csv = _write_csv(
        out_dir / "ra_ee_08_step_metrics.csv",
        step_rows,
        fieldnames=_fieldnames(step_rows),
    )
    compact_rows = _compact_summary_rows(summaries)
    summary_csv = _write_csv(
        out_dir / "ra_ee_08_candidate_summary.csv",
        compact_rows,
        fieldnames=list(compact_rows[0].keys()),
    )
    guardrail_csv = _write_csv(
        out_dir / "ra_ee_08_guardrail_checks.csv",
        guardrail_checks,
        fieldnames=list(guardrail_checks[0].keys()) if guardrail_checks else [],
    )

    protocol = {
        "phase": "RA-EE-08",
        "method_label": run_settings.method_label,
        "method_family": "RA-EE offline deterministic association re-evaluation",
        "implementation_sublabel": run_settings.implementation_sublabel,
        "training": "none; offline replay only",
        "offline_replay_only": True,
        "deterministic_association_proposals_only": True,
        "learned_association": "disabled",
        "learned_hierarchical_RL": "disabled",
        "association_training": "disabled",
        "joint_association_power_training": "disabled",
        "catfish": "disabled",
        "multi_catfish": "disabled",
        "rb_bandwidth_allocation": "disabled/not-modeled",
        "old_EE_MODQN_continuation": "forbidden/not-performed",
        "frozen_baseline_mutation": "forbidden/not-performed",
        "hobs_optimizer_claim": "forbidden/not-made",
        "physical_energy_saving_claim": "forbidden/not-made",
        "association_action_contract": "deterministic-active-set-served-set-proposal-rule",
        "matched_control_association_policy": run_settings.matched_control_association_policy,
        "candidate_association_policies": list(run_settings.candidate_association_policies),
        "predeclared_primary_association_policy": (
            run_settings.predeclared_primary_association_policy
        ),
        "primary_control": RA_EE_08_FIXED_DEPLOYABLE_CONTROL,
        "primary_candidate": RA_EE_08_CANDIDATE,
        "primary_power_allocator_pairing": "same-deployable-stronger-power-allocator",
        "deployable_allocator_candidates": list(run_settings.deployable_allocators),
        "primary_deployable_allocator": run_settings.primary_deployable_allocator,
        "diagnostic_rows": [
            RA_EE_08_PROPOSAL_SAFE_GREEDY,
            RA_EE_08_FIXED_SAFE_GREEDY,
            RA_EE_08_FIXED_CONSTRAINED_ORACLE,
            RA_EE_08_ASSOC_ORACLE_CONSTRAINED,
            RA_EE_08_ASSOC_ORACLE_DEPLOYABLE,
        ],
        "oracle_diagnostic_only": include_oracle,
        "oracle_runtime_method": "forbidden/not-performed",
        "system_EE_primary": True,
        "scalar_reward_success_basis": False,
        "per_user_EE_credit_success_basis": False,
    }
    constraints = {
        "min_active_beams": run_settings.min_active_beams,
        "max_active_beams": run_settings.max_active_beams,
        "target_users_per_active_beam": run_settings.target_users_per_active_beam,
        "load_cap_overflow_users": run_settings.load_cap_overflow_users,
        "max_one_active_beam_ratio_for_acceptance": (
            run_settings.max_one_active_beam_ratio_for_acceptance
        ),
        "max_two_beam_overload_step_ratio": (
            run_settings.max_two_beam_overload_step_ratio
        ),
        "max_moved_user_ratio": run_settings.max_moved_user_ratio,
        "max_moved_user_ratio_per_step": run_settings.max_moved_user_ratio_per_step,
        "min_oracle_gap_closed_ratio": run_settings.min_oracle_gap_closed_ratio,
        "per_beam_max_power_w": run_settings.audit.per_beam_max_power_w,
        "total_power_budget_w": run_settings.audit.total_power_budget_w,
        "inactive_beam_policy": "zero-w",
        "codebook_levels_w": list(run_settings.audit.codebook_levels_w),
        "p05_throughput_min_ratio_vs_control": (
            run_settings.audit.p05_min_ratio_vs_control
        ),
        "served_ratio_min_delta_vs_control": (
            run_settings.audit.served_ratio_min_delta_vs_control
        ),
        "outage_ratio_max_delta_vs_control": (
            run_settings.audit.outage_ratio_max_delta_vs_control
        ),
        "candidate_max_demoted_beams": run_settings.candidate_max_demoted_beams,
        "candidate_step_p05_guardrail_margin": (
            run_settings.candidate_step_p05_guardrail_margin
        ),
        "p05_trim_max_moves": run_settings.p05_trim_max_moves,
        "local_search_max_moves": run_settings.local_search_max_moves,
        "dp_max_profile_count": run_settings.dp_max_profile_count,
        "power_repair": "not-used; requested and effective vectors are exported",
        "effective_power_vector_contract": (
            "same effective_power_vector_w feeds SINR numerator, throughput, "
            "EE denominator, audit logs, and budget checks"
        ),
    }
    summary = {
        "inputs": {
            "config_path": str(Path(config_path)),
            "output_dir": str(out_dir),
            "max_steps": max_steps,
        },
        "protocol": protocol,
        "constraints": constraints,
        "association_metadata": association_metadata,
        "step_metrics_schema_fields": list(step_rows[0].keys()),
        "candidate_summaries": summaries,
        "guardrail_checks": guardrail_checks,
        "ranking_separation_result": {
            "comparison_fixed_deployable_vs_proposal_deployable": ranking_checks,
        },
        "bucket_results": bucket_results,
        "seed_level_results": seed_results,
        "oracle_gap_diagnostics": oracle_gap_diagnostics,
        "decision_detail": decision_detail,
        "proof_flags": decision_detail["proof_flags"],
        "stop_conditions": decision_detail["stop_conditions"],
        "ra_ee_08_decision": decision_detail["ra_ee_08_decision"],
        "remaining_blockers": [
            "This is offline deterministic association replay evidence only.",
            "No learned association, hierarchical RL, or full RA-EE-MODQN policy exists.",
            "No joint association + power training exists.",
            "No RB / bandwidth allocation is included.",
            "Safe-greedy and oracle rows remain diagnostic-only.",
        ],
        "forbidden_claims_still_active": [
            "Do not call RA-EE-08 full RA-EE-MODQN.",
            "Do not claim learned hierarchical RL or learned association effectiveness.",
            "Do not claim joint association + power training.",
            "Do not claim old EE-MODQN effectiveness.",
            "Do not claim HOBS optimizer behavior.",
            "Do not claim Catfish, multi-Catfish, or Catfish-EE-MODQN effectiveness.",
            "Do not add or claim RB / bandwidth allocation.",
            "Do not treat per-user EE credit as system EE.",
            "Do not use scalar reward alone as success evidence.",
            "Do not use oracle rows as deployable runtime methods.",
            "Do not claim full paper-faithful reproduction.",
            "Do not claim physical energy saving.",
        ],
    }
    summary_path = write_json(
        out_dir / "ra_ee_08_offline_association_reevaluation_summary.json",
        summary,
    )
    review_path = out_dir / "review.md"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text("\n".join(_review_lines(summary)) + "\n")
    return {
        "ra_ee_08_offline_association_reevaluation_summary": summary_path,
        "ra_ee_08_candidate_summary_csv": summary_csv,
        "ra_ee_08_guardrail_checks_csv": guardrail_csv,
        "ra_ee_08_step_metrics": step_csv,
        "review_md": review_path,
        "summary": summary,
    }

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT_DIR",
    "RA_EE_08_ASSOC_ORACLE_CONSTRAINED",
    "RA_EE_08_ASSOC_ORACLE_DEPLOYABLE",
    "RA_EE_08_CANDIDATE",
    "RA_EE_08_FIXED_CONSTRAINED_ORACLE",
    "RA_EE_08_FIXED_DEPLOYABLE_CONTROL",
    "RA_EE_08_FIXED_SAFE_GREEDY",
    "RA_EE_08_METHOD_LABEL",
    "RA_EE_08_PROPOSAL_POLICIES",
    "RA_EE_08_PROPOSAL_SAFE_GREEDY",
    "_guardrail_result",
    "_select_actions_for_association_policy",
    "_settings_from_config",
    "export_ra_ee_08_offline_association_reevaluation",
]
