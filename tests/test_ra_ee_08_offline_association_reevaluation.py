from __future__ import annotations

from pathlib import Path

import numpy as np

from modqn_paper_reproduction.analysis.ra_ee_05_fixed_association_robustness import (
    export_ra_ee_05_fixed_association_robustness,
)
from modqn_paper_reproduction.analysis.ra_ee_06_association_counterfactual_oracle import (
    ACTIVE_SET_LOAD_SPREAD,
    ACTIVE_SET_QUALITY_SPREAD,
    ACTIVE_SET_STICKY_SPREAD,
    export_ra_ee_06_association_counterfactual_oracle,
)
from modqn_paper_reproduction.analysis.ra_ee_06b_association_proposal_refinement import (
    BOUNDED_MOVE_SERVED_SET,
    ORACLE_SCORE_TOPK_ACTIVE_SET,
    P05_SLACK_AWARE_ACTIVE_SET,
    POWER_RESPONSE_AWARE_LOAD_BALANCE,
    STICKY_ORACLE_COUNT_LOCAL_SEARCH,
    export_ra_ee_06b_association_proposal_refinement,
)
from modqn_paper_reproduction.analysis.ra_ee_07_constrained_power_allocator_distillation import (
    export_ra_ee_07_constrained_power_allocator_distillation,
)
from modqn_paper_reproduction.analysis.ra_ee_08_offline_association_reevaluation import (
    DEFAULT_CONFIG,
    RA_EE_08_ASSOC_ORACLE_CONSTRAINED,
    RA_EE_08_ASSOC_ORACLE_DEPLOYABLE,
    RA_EE_08_CANDIDATE,
    RA_EE_08_FIXED_CONSTRAINED_ORACLE,
    RA_EE_08_FIXED_DEPLOYABLE_CONTROL,
    RA_EE_08_FIXED_SAFE_GREEDY,
    RA_EE_08_METHOD_LABEL,
    RA_EE_08_PROPOSAL_POLICIES,
    RA_EE_08_PROPOSAL_SAFE_GREEDY,
    _guardrail_result,
    _select_actions_for_association_policy,
    _settings_from_config,
    export_ra_ee_08_offline_association_reevaluation,
)
from modqn_paper_reproduction.config_loader import build_environment, load_training_yaml
from modqn_paper_reproduction.env.step import (
    HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK,
    HOBS_POWER_SURFACE_STATIC_CONFIG,
)


BASELINE_CONFIG = "configs/modqn-paper-baseline.resolved-template.yaml"
RA_EE_05_CONFIG = "configs/ra-ee-05-fixed-association-robustness.resolved.yaml"
RA_EE_06_CONFIG = "configs/ra-ee-06-association-counterfactual-oracle.resolved.yaml"
RA_EE_06B_CONFIG = "configs/ra-ee-06b-association-proposal-refinement.resolved.yaml"
RA_EE_07_CONFIG = "configs/ra-ee-07-constrained-power-allocator-distillation.resolved.yaml"
RA_EE_08_CONFIG = DEFAULT_CONFIG
FROZEN_BASELINE_ARTIFACTS = (
    "artifacts/pilot-02-best-eval",
    "artifacts/run-9000",
    "artifacts/table-ii-200ep-01",
    "artifacts/fig-3-pilot-01",
    "artifacts/fig-4-pilot-01",
    "artifacts/fig-5-pilot-01",
    "artifacts/fig-6-pilot-01",
)


def _smoke_export(tmp_path, *, policies=(ACTIVE_SET_STICKY_SPREAD,)):
    return export_ra_ee_08_offline_association_reevaluation(
        config_path=RA_EE_08_CONFIG,
        output_dir=tmp_path / "ra-ee-08",
        calibration_seed_set=(100,),
        held_out_seed_set=(600,),
        candidate_association_policies=policies,
        max_steps=1,
        include_oracle=True,
    )


def test_ra_ee_08_is_opt_in_and_forbidden_features_disabled() -> None:
    baseline_cfg = load_training_yaml(BASELINE_CONFIG)
    ra_ee_08_cfg = load_training_yaml(RA_EE_08_CONFIG)
    baseline_env = build_environment(baseline_cfg)
    ra_ee_08_env = build_environment(ra_ee_08_cfg)
    gate = ra_ee_08_cfg["resolved_assumptions"][
        "ra_ee_08_offline_association_reevaluation"
    ]["value"]

    assert baseline_env.power_surface_config.hobs_power_surface_mode == (
        HOBS_POWER_SURFACE_STATIC_CONFIG
    )
    assert ra_ee_08_env.power_surface_config.hobs_power_surface_mode == (
        HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK
    )
    assert ra_ee_08_cfg["track"]["phase"] == "RA-EE-08"
    assert gate["method_label"] == RA_EE_08_METHOD_LABEL
    assert gate["offline_replay_only"] is True
    assert gate["deterministic_association_proposals_only"] is True
    assert gate["learned_association"] == "disabled"
    assert gate["learned_hierarchical_RL"] == "disabled"
    assert gate["association_training"] == "disabled"
    assert gate["joint_association_power_training"] == "disabled"
    assert gate["catfish"] == "disabled"
    assert gate["multi_catfish"] == "disabled"
    assert gate["rb_bandwidth_allocation"] == "disabled"
    assert gate["oracle_runtime_method"] == "forbidden"


def test_ra_ee_08_all_proposal_families_are_replayed_and_deterministic() -> None:
    cfg = load_training_yaml(RA_EE_08_CONFIG)
    settings = _settings_from_config(cfg)
    env = build_environment(cfg)
    env_seed_seq, mobility_seed_seq = np.random.SeedSequence(600).spawn(2)
    states, masks, _diag = env.reset(
        np.random.default_rng(env_seed_seq),
        np.random.default_rng(mobility_seed_seq),
    )
    current = env.current_assignments()

    assert tuple(settings.candidate_association_policies) == (
        ACTIVE_SET_LOAD_SPREAD,
        ACTIVE_SET_QUALITY_SPREAD,
        ACTIVE_SET_STICKY_SPREAD,
        STICKY_ORACLE_COUNT_LOCAL_SEARCH,
        P05_SLACK_AWARE_ACTIVE_SET,
        POWER_RESPONSE_AWARE_LOAD_BALANCE,
        BOUNDED_MOVE_SERVED_SET,
        ORACLE_SCORE_TOPK_ACTIVE_SET,
    )
    assert tuple(settings.candidate_association_policies) == RA_EE_08_PROPOSAL_POLICIES

    for policy in settings.candidate_association_policies:
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


def test_ra_ee_08_export_keeps_frozen_baseline_untouched(tmp_path) -> None:
    before = Path(BASELINE_CONFIG).read_text()
    outputs = _smoke_export(tmp_path)
    after = Path(BASELINE_CONFIG).read_text()

    assert before == after
    output_paths = {str(path) for key, path in outputs.items() if key != "summary"}
    for frozen in FROZEN_BASELINE_ARTIFACTS:
        assert all(not path.startswith(frozen) for path in output_paths)
    assert outputs["summary"]["proof_flags"]["frozen_baseline_mutation"] is False


def test_ra_ee_08_primary_comparison_uses_same_deployable_allocator(tmp_path) -> None:
    summary = _smoke_export(tmp_path)["summary"]
    protocol = summary["protocol"]
    primary_rows = [
        row
        for row in summary["candidate_summaries"]
        if row["power_semantics"]
        in {RA_EE_08_FIXED_DEPLOYABLE_CONTROL, RA_EE_08_CANDIDATE}
    ]

    assert protocol["primary_control"] == RA_EE_08_FIXED_DEPLOYABLE_CONTROL
    assert protocol["primary_candidate"] == RA_EE_08_CANDIDATE
    assert protocol["primary_power_allocator_pairing"] == (
        "same-deployable-stronger-power-allocator"
    )
    assert all(
        row["power_allocator"] == "deployable-stronger-power-allocator"
        for row in primary_rows
    )
    assert summary["proof_flags"]["primary_comparison_uses_same_deployable_allocator"]
    assert summary["proof_flags"]["primary_comparison_no_step_cap_mismatch"]


def test_ra_ee_08_diagnostic_and_oracle_rows_do_not_count_for_acceptance(tmp_path) -> None:
    summary = _smoke_export(tmp_path)["summary"]
    semantics = {row["power_semantics"] for row in summary["candidate_summaries"]}

    assert RA_EE_08_PROPOSAL_SAFE_GREEDY in semantics
    assert RA_EE_08_FIXED_SAFE_GREEDY in semantics
    assert RA_EE_08_FIXED_CONSTRAINED_ORACLE in semantics
    assert RA_EE_08_ASSOC_ORACLE_DEPLOYABLE in semantics
    assert RA_EE_08_ASSOC_ORACLE_CONSTRAINED in semantics
    diagnostic_checks = [
        row
        for row in summary["guardrail_checks"]
        if row["power_semantics"] != RA_EE_08_CANDIDATE
    ]
    assert diagnostic_checks
    assert all(row["accepted"] is False for row in diagnostic_checks)
    assert all(row["rejection_reason"] == "diagnostic-only" for row in diagnostic_checks)
    assert summary["proof_flags"]["oracle_rows_excluded_from_acceptance"] is True


def test_ra_ee_08_p05_guardrail_is_enforced() -> None:
    settings = _settings_from_config(load_training_yaml(RA_EE_08_CONFIG))
    control = {
        "evaluation_bucket": "held-out",
        "trajectory_policy": "held-out:test",
        "candidate_association_policy": "test",
        "power_semantics": RA_EE_08_FIXED_DEPLOYABLE_CONTROL,
        "power_allocator": "deployable-stronger-power-allocator",
        "EE_system_aggregate_bps_per_w": 10.0,
        "throughput_p05_user_step_bps": 100.0,
        "throughput_mean_user_step_bps": 100.0,
        "served_ratio": 1.0,
        "outage_ratio": 0.0,
    }
    candidate = {
        **control,
        "power_semantics": RA_EE_08_CANDIDATE,
        "EE_system_aggregate_bps_per_w": 11.0,
        "throughput_p05_user_step_bps": 90.0,
        "budget_violations": {"step_count": 0},
        "per_beam_power_violations": {"step_count": 0},
        "inactive_beam_nonzero_power_step_count": 0,
        "one_active_beam_step_ratio": 0.0,
        "two_beam_overload_step_ratio": 0.0,
        "denominator_varies_in_eval": True,
        "handover_burden": {"moved_user_count": 1, "moved_user_ratio": 0.01},
        "oracle_labels_used_for_runtime_decision": False,
        "future_outcomes_used_for_runtime_decision": False,
        "held_out_answers_used_for_runtime_decision": False,
    }
    oracle = {
        **control,
        "power_semantics": RA_EE_08_ASSOC_ORACLE_CONSTRAINED,
        "EE_system_aggregate_bps_per_w": 12.0,
    }

    result = _guardrail_result(
        candidate=candidate,
        control=control,
        oracle=oracle,
        settings=settings,
    )

    assert result["p05_guardrail_pass"] is False
    assert result["accepted"] is False
    assert "p05-ratio-below-threshold" in result["rejection_reason"]


def test_ra_ee_08_exports_handover_burden_and_gate_flags(tmp_path) -> None:
    outputs = _smoke_export(tmp_path)
    summary = outputs["summary"]

    assert outputs["ra_ee_08_offline_association_reevaluation_summary"].exists()
    assert outputs["ra_ee_08_candidate_summary_csv"].exists()
    assert outputs["ra_ee_08_guardrail_checks_csv"].exists()
    assert outputs["ra_ee_08_step_metrics"].exists()
    assert outputs["review_md"].exists()
    assert "RA-EE-08 Offline Association Re-Evaluation Review" in (
        outputs["review_md"].read_text()
    )
    candidate = next(
        row
        for row in summary["candidate_summaries"]
        if row["power_semantics"] == RA_EE_08_CANDIDATE
    )
    assert "moved_user_count_total" in candidate
    assert "handover_burden" in candidate
    assert "moved_user_ratio_distribution" in candidate
    assert summary["proof_flags"]["offline_replay_only"] is True
    assert summary["proof_flags"]["deterministic_association_proposals_only"] is True
    assert summary["proof_flags"]["catfish_disabled"] is True
    assert summary["proof_flags"]["rb_bandwidth_allocation_disabled"] is True
    assert summary["proof_flags"]["scalar_reward_success_basis"] is False
    assert summary["proof_flags"]["full_RA_EE_MODQN_claim"] is False


def test_ra_ee_08_ra_ee_05_06_06b_07_regression_compatibility(tmp_path) -> None:
    ra05 = export_ra_ee_05_fixed_association_robustness(
        output_dir=tmp_path / "ra05",
        calibration_seed_set=(100,),
        held_out_seed_set=(600,),
        calibration_policies=("hold-current",),
        held_out_policies=("random-valid-heldout",),
        max_steps=1,
        include_oracle=True,
    )
    ra06 = export_ra_ee_06_association_counterfactual_oracle(
        output_dir=tmp_path / "ra06",
        calibration_seed_set=(100,),
        held_out_seed_set=(600,),
        candidate_association_policies=(ACTIVE_SET_STICKY_SPREAD,),
        max_steps=1,
        include_oracle=True,
    )
    ra06b = export_ra_ee_06b_association_proposal_refinement(
        output_dir=tmp_path / "ra06b",
        calibration_seed_set=(100,),
        held_out_seed_set=(600,),
        candidate_association_policies=(STICKY_ORACLE_COUNT_LOCAL_SEARCH,),
        max_steps=1,
        include_oracle=True,
    )
    ra07 = export_ra_ee_07_constrained_power_allocator_distillation(
        output_dir=tmp_path / "ra07",
        calibration_seed_set=(100,),
        held_out_seed_set=(600,),
        diagnostic_seed_set=(600,),
        calibration_policies=("hold-current",),
        held_out_policies=("random-valid-heldout",),
        diagnostic_association_policies=(STICKY_ORACLE_COUNT_LOCAL_SEARCH,),
        max_steps=1,
        include_oracle=True,
        include_association_diagnostics=True,
    )

    assert "ra_ee_05_decision" in ra05["summary"]
    assert "ra_ee_06_decision" in ra06["summary"]
    assert "ra_ee_06b_decision" in ra06b["summary"]
    assert "ra_ee_07_decision" in ra07["summary"]
