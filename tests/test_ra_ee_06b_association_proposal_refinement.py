from __future__ import annotations

from pathlib import Path

import numpy as np

from modqn_paper_reproduction.analysis.ra_ee_06_association_counterfactual_oracle import (
    DEFAULT_CONFIG as RA_EE_06_CONFIG,
)
from modqn_paper_reproduction.analysis.ra_ee_06b_association_proposal_refinement import (
    BOUNDED_MOVE_SERVED_SET,
    DEFAULT_CONFIG,
    ORACLE_SCORE_TOPK_ACTIVE_SET,
    P05_SLACK_AWARE_ACTIVE_SET,
    POWER_RESPONSE_AWARE_LOAD_BALANCE,
    RA_EE_06B_CANDIDATE,
    RA_EE_06B_GREEDY_DIAGNOSTIC,
    RA_EE_06B_MATCHED_CONTROL,
    RA_EE_06B_METHOD_LABEL,
    RA_EE_06B_ORACLE_CONSTRAINED,
    RA_EE_06B_ORACLE_SAFE_GREEDY,
    RA_EE_06B_PROPOSAL_FIXED_1W,
    RA_EE_06B_PROPOSAL_POLICIES,
    STICKY_ORACLE_COUNT_LOCAL_SEARCH,
    _handover_burden,
    _p05_ratio_and_slack,
    _select_actions_for_association_policy,
    _settings_from_config,
    export_ra_ee_06b_association_proposal_refinement,
)
from modqn_paper_reproduction.config_loader import build_environment, load_training_yaml
from modqn_paper_reproduction.env.step import (
    HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK,
    HOBS_POWER_SURFACE_STATIC_CONFIG,
)


BASELINE_CONFIG = "configs/modqn-paper-baseline.resolved-template.yaml"
RA_EE_04_CONTROL_CONFIG = "configs/ra-ee-04-bounded-power-allocator-control.resolved.yaml"
RA_EE_05_CONFIG = "configs/ra-ee-05-fixed-association-robustness.resolved.yaml"
RA_EE_06B_CONFIG = DEFAULT_CONFIG
FROZEN_BASELINE_ARTIFACTS = (
    "artifacts/pilot-02-best-eval",
    "artifacts/run-9000",
    "artifacts/table-ii-200ep-01",
    "artifacts/fig-3-pilot-01",
    "artifacts/fig-4-pilot-01",
    "artifacts/fig-5-pilot-01",
    "artifacts/fig-6-pilot-01",
)


def test_ra_ee_06b_is_opt_in_and_forbidden_features_disabled() -> None:
    baseline_cfg = load_training_yaml(BASELINE_CONFIG)
    ra_ee_06b_cfg = load_training_yaml(RA_EE_06B_CONFIG)
    baseline_env = build_environment(baseline_cfg)
    ra_ee_06b_env = build_environment(ra_ee_06b_cfg)
    gate = ra_ee_06b_cfg["resolved_assumptions"][
        "ra_ee_06b_association_proposal_refinement"
    ]["value"]

    assert baseline_env.power_surface_config.hobs_power_surface_mode == (
        HOBS_POWER_SURFACE_STATIC_CONFIG
    )
    assert ra_ee_06b_env.power_surface_config.hobs_power_surface_mode == (
        HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK
    )
    assert ra_ee_06b_cfg["track"]["phase"] == "RA-EE-06B"
    assert gate["method_label"] == RA_EE_06B_METHOD_LABEL
    assert gate["offline_trace_export_only"] is True
    assert gate["deterministic_proposal_refinement_only"] is True
    assert gate["learned_hierarchical_RL"] == "disabled"
    assert gate["learned_association"] == "disabled"
    assert gate["association_training"] == "disabled"
    assert gate["joint_association_power_training"] == "disabled"
    assert gate["catfish"] == "disabled"
    assert gate["multi_catfish"] == "disabled"
    assert gate["rb_bandwidth_allocation"] == "disabled"
    assert gate["oracle_upper_bound_role"] == "diagnostic-only"


def test_ra_ee_06b_proposal_rules_are_deterministic_and_bounded() -> None:
    cfg = load_training_yaml(RA_EE_06B_CONFIG)
    settings = _settings_from_config(cfg)
    env = build_environment(cfg)
    env_seed_seq, mobility_seed_seq = np.random.SeedSequence(600).spawn(2)
    states, masks, _diag = env.reset(
        np.random.default_rng(env_seed_seq),
        np.random.default_rng(mobility_seed_seq),
    )
    current = env.current_assignments()

    for policy in RA_EE_06B_PROPOSAL_POLICIES:
        first = _select_actions_for_association_policy(
            policy,
            user_states=states,
            masks=masks,
            current_assignments=current,
            settings=settings,
        )
        second = _select_actions_for_association_policy(
            policy,
            user_states=states,
            masks=masks,
            current_assignments=current,
            settings=settings,
        )
        active_count = len(set(int(value) for value in first.tolist()))
        assert np.array_equal(first, second)
        assert settings.min_active_beams <= active_count <= settings.max_active_beams


def test_ra_ee_06b_p05_slack_and_handover_burden_calculation() -> None:
    ratio, slack = _p05_ratio_and_slack(
        control_p05_bps=100.0,
        candidate_p05_bps=96.0,
        threshold_ratio=0.95,
    )
    burden = _handover_burden(moved_user_count=12, user_step_count=100)

    assert ratio == 0.96
    assert slack == 1.0
    assert burden["moved_user_count"] == 12
    assert burden["moved_user_ratio"] == 0.12


def test_ra_ee_06b_export_keeps_frozen_baseline_config_and_artifacts_untouched(
    tmp_path,
) -> None:
    before = Path(BASELINE_CONFIG).read_text()
    outputs = export_ra_ee_06b_association_proposal_refinement(
        config_path=RA_EE_06B_CONFIG,
        output_dir=tmp_path / "ra-ee-06b",
        calibration_seed_set=(100,),
        held_out_seed_set=(600,),
        candidate_association_policies=(BOUNDED_MOVE_SERVED_SET,),
        max_steps=1,
        include_oracle=True,
    )
    after = Path(BASELINE_CONFIG).read_text()

    assert before == after
    output_paths = {str(path) for key, path in outputs.items() if key != "summary"}
    for frozen in FROZEN_BASELINE_ARTIFACTS:
        assert all(not path.startswith(frozen) for path in output_paths)
    assert outputs["summary"]["proof_flags"]["frozen_baseline_mutation"] is False


def test_ra_ee_06b_trace_schema_and_review_exports_include_required_flags(tmp_path) -> None:
    outputs = export_ra_ee_06b_association_proposal_refinement(
        config_path=RA_EE_06B_CONFIG,
        output_dir=tmp_path / "ra-ee-06b",
        calibration_seed_set=(100,),
        held_out_seed_set=(600,),
        candidate_association_policies=(BOUNDED_MOVE_SERVED_SET,),
        max_steps=1,
        include_oracle=True,
    )
    summary = outputs["summary"]
    schema = set(summary["oracle_trace_schema_fields"])

    assert outputs["ra_ee_06b_association_proposal_refinement_summary"].exists()
    assert outputs["ra_ee_06b_candidate_summary_csv"].exists()
    assert outputs["ra_ee_06b_guardrail_checks_csv"].exists()
    assert outputs["ra_ee_06b_oracle_trace"].exists()
    assert outputs["review_md"].exists()
    assert "RA-EE-06B Association Proposal Refinement Review" in (
        outputs["review_md"].read_text()
    )
    for field in (
        "active_beam_count",
        "active_beam_mask",
        "active_set_size",
        "active_set_source_policy",
        "beam_load_distribution",
        "beam_load_max",
        "beam_load_min",
        "beam_load_std",
        "load_cap_slack",
        "beam_load_balance_gap",
        "per_user_selected_beam_quality",
        "per_user_top_k_quality",
        "best_vs_selected_margin",
        "valid_beam_count",
        "current_beam",
        "control_beam",
        "oracle_beam",
        "moved_flag",
        "beam_rank_distance",
        "beam_offset_distance_proxy",
        "moved_user_count",
        "handover_count",
        "r2_mean",
        "p05_throughput_control_bps",
        "p05_throughput_candidate_bps",
        "p05_throughput_oracle_bps",
        "p05_ratio_vs_matched_control",
        "p05_slack_to_0_95_threshold_bps",
        "effective_power_vector_w",
        "demoted_beams",
        "total_active_beam_power_w",
        "EE_denominator_w",
        "safe_greedy_accepted_demotions",
        "safe_greedy_rejected_demotion_count",
        "active_beam_throughput_gap_bps",
        "tail_user_ids",
        "oracle_selected_association_policy",
        "oracle_power_profile",
        "EE_delta_vs_matched_control",
        "accepted_flag",
        "rejection_reason",
    ):
        assert field in schema
    assert summary["proof_flags"]["offline_trace_export_only"] is True
    assert summary["proof_flags"]["deterministic_proposal_refinement_only"] is True
    assert summary["proof_flags"]["oracle_diagnostic_only"] is True
    assert summary["protocol"]["scalar_reward_success_basis"] is False


def test_ra_ee_06b_exports_required_comparators_and_same_allocator(tmp_path) -> None:
    outputs = export_ra_ee_06b_association_proposal_refinement(
        config_path=RA_EE_06B_CONFIG,
        output_dir=tmp_path / "ra-ee-06b",
        calibration_seed_set=(100,),
        held_out_seed_set=(600,),
        candidate_association_policies=(BOUNDED_MOVE_SERVED_SET,),
        max_steps=1,
        include_oracle=True,
        include_fixed_1w_diagnostic=True,
        include_matched_fixed_constrained_isolation=True,
    )
    summary = outputs["summary"]
    semantics = {row["power_semantics"] for row in summary["candidate_summaries"]}
    safe_rows = [
        row
        for row in summary["candidate_summaries"]
        if row["power_semantics"] in {RA_EE_06B_MATCHED_CONTROL, RA_EE_06B_CANDIDATE}
    ]

    assert RA_EE_06B_MATCHED_CONTROL in semantics
    assert RA_EE_06B_CANDIDATE in semantics
    assert RA_EE_06B_PROPOSAL_FIXED_1W in semantics
    assert RA_EE_06B_GREEDY_DIAGNOSTIC in semantics
    assert RA_EE_06B_ORACLE_SAFE_GREEDY in semantics
    assert RA_EE_06B_ORACLE_CONSTRAINED in semantics
    assert all(row["power_allocator"] == "safe-greedy-power-allocator" for row in safe_rows)
    assert summary["proof_flags"]["matched_control_uses_same_power_allocator"] is True
    assert summary["proof_flags"]["safe_greedy_allocator_retained"] is True


def test_ra_ee_06b_all_proposal_names_are_configured() -> None:
    cfg = load_training_yaml(RA_EE_06B_CONFIG)
    gate = cfg["resolved_assumptions"][
        "ra_ee_06b_association_proposal_refinement"
    ]["value"]

    assert tuple(gate["candidate_association_policies"]) == (
        STICKY_ORACLE_COUNT_LOCAL_SEARCH,
        P05_SLACK_AWARE_ACTIVE_SET,
        POWER_RESPONSE_AWARE_LOAD_BALANCE,
        BOUNDED_MOVE_SERVED_SET,
        ORACLE_SCORE_TOPK_ACTIVE_SET,
    )


def test_ra_ee_06b_does_not_relabel_prior_ra_ee_configs() -> None:
    ra_ee_04_cfg = load_training_yaml(RA_EE_04_CONTROL_CONFIG)
    ra_ee_05_cfg = load_training_yaml(RA_EE_05_CONFIG)
    ra_ee_06_cfg = load_training_yaml(RA_EE_06_CONFIG)
    ra_ee_06b_cfg = load_training_yaml(RA_EE_06B_CONFIG)

    assert ra_ee_04_cfg["track"]["phase"] == "RA-EE-04"
    assert ra_ee_05_cfg["track"]["phase"] == "RA-EE-05"
    assert ra_ee_06_cfg["track"]["phase"] == "RA-EE-06"
    assert ra_ee_06b_cfg["track"]["phase"] == "RA-EE-06B"
    assert "ra_ee_06b_association_proposal_refinement" not in (
        ra_ee_06_cfg["resolved_assumptions"]
    )
