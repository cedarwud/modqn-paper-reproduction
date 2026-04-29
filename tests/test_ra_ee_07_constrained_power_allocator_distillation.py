from __future__ import annotations

from pathlib import Path

import numpy as np

from modqn_paper_reproduction.analysis.ra_ee_02_oracle_power_allocation import (
    _StepSnapshot,
    _evaluate_power_vector,
)
from modqn_paper_reproduction.analysis.ra_ee_04_bounded_power_allocator import (
    export_ra_ee_04_bounded_power_allocator_pilot,
)
from modqn_paper_reproduction.analysis.ra_ee_05_fixed_association_robustness import (
    export_ra_ee_05_fixed_association_robustness,
)
from modqn_paper_reproduction.analysis.ra_ee_06_association_counterfactual_oracle import (
    export_ra_ee_06_association_counterfactual_oracle,
)
from modqn_paper_reproduction.analysis.ra_ee_06b_association_proposal_refinement import (
    export_ra_ee_06b_association_proposal_refinement,
)
from modqn_paper_reproduction.analysis.ra_ee_07_constrained_power_allocator_distillation import (
    BOUNDED_LOCAL_SEARCH,
    DETERMINISTIC_HYBRID,
    FINITE_CODEBOOK_DP,
    P05_SLACK_TRIM_TAIL_PROTECT,
    RA_EE_07_ASSOC_ORACLE_CONSTRAINED,
    RA_EE_07_ASSOC_PROPOSAL_DEPLOYABLE,
    RA_EE_07_CONSTRAINED_ORACLE,
    RA_EE_07_DEPLOYABLE,
    RA_EE_07_FIXED_1W_DIAGNOSTIC,
    RA_EE_07_METHOD_LABEL,
    RA_EE_07_SAFE_GREEDY_CONTROL,
    _candidate_step_passes,
    _settings_from_config,
    export_ra_ee_07_constrained_power_allocator_distillation,
)
from modqn_paper_reproduction.config_loader import build_environment, load_training_yaml
from modqn_paper_reproduction.env.step import (
    HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK,
    HOBS_POWER_SURFACE_STATIC_CONFIG,
)


BASELINE_CONFIG = "configs/modqn-paper-baseline.resolved-template.yaml"
RA_EE_07_CONFIG = (
    "configs/ra-ee-07-constrained-power-allocator-distillation.resolved.yaml"
)
FROZEN_BASELINE_ARTIFACTS = (
    "artifacts/pilot-02-best-eval",
    "artifacts/run-9000",
    "artifacts/table-ii-200ep-01",
    "artifacts/fig-3-pilot-01",
    "artifacts/fig-4-pilot-01",
    "artifacts/fig-5-pilot-01",
    "artifacts/fig-6-pilot-01",
)


def _snapshot() -> _StepSnapshot:
    return _StepSnapshot(
        trajectory_policy="synthetic-heldout",
        evaluation_seed=600,
        step_index=1,
        assignments=np.array([0, 1], dtype=np.int32),
        active_mask=np.array([True, True, False], dtype=bool),
        beam_loads=np.array([1.0, 1.0, 0.0], dtype=np.float64),
        unit_snr_by_user=np.array([1.0, 1.0], dtype=np.float64),
        bandwidth_hz=1.0,
        handover_count=0,
        r2_mean=0.0,
    )


def _smoke_export(tmp_path):
    return export_ra_ee_07_constrained_power_allocator_distillation(
        config_path=RA_EE_07_CONFIG,
        output_dir=tmp_path / "ra-ee-07",
        calibration_seed_set=(100,),
        held_out_seed_set=(600,),
        diagnostic_seed_set=(600,),
        calibration_policies=("hold-current",),
        held_out_policies=("random-valid-heldout",),
        diagnostic_association_policies=("sticky-oracle-count-local-search",),
        max_steps=1,
        include_oracle=True,
        include_association_diagnostics=True,
    )


def test_ra_ee_07_is_opt_in_and_forbidden_features_disabled() -> None:
    baseline_cfg = load_training_yaml(BASELINE_CONFIG)
    ra_ee_07_cfg = load_training_yaml(RA_EE_07_CONFIG)
    baseline_env = build_environment(baseline_cfg)
    ra_ee_07_env = build_environment(ra_ee_07_cfg)
    gate = ra_ee_07_cfg["resolved_assumptions"][
        "ra_ee_07_constrained_power_allocator_distillation"
    ]["value"]

    assert baseline_env.power_surface_config.hobs_power_surface_mode == (
        HOBS_POWER_SURFACE_STATIC_CONFIG
    )
    assert ra_ee_07_env.power_surface_config.hobs_power_surface_mode == (
        HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK
    )
    assert ra_ee_07_cfg["track"]["phase"] == "RA-EE-07"
    assert gate["method_label"] == RA_EE_07_METHOD_LABEL
    assert gate["offline_replay_only"] is True
    assert gate["learned_association"] == "disabled"
    assert gate["learned_hierarchical_RL"] == "disabled"
    assert gate["joint_association_power_training"] == "disabled"
    assert gate["catfish"] == "disabled"
    assert gate["multi_catfish"] == "disabled"
    assert gate["rb_bandwidth_allocation"] == "disabled"


def test_ra_ee_07_export_keeps_frozen_baseline_untouched(tmp_path) -> None:
    before = Path(BASELINE_CONFIG).read_text()
    outputs = _smoke_export(tmp_path)
    after = Path(BASELINE_CONFIG).read_text()

    assert before == after
    output_paths = {str(path) for key, path in outputs.items() if key != "summary"}
    for frozen in FROZEN_BASELINE_ARTIFACTS:
        assert all(not path.startswith(frozen) for path in output_paths)


def test_ra_ee_07_protocol_and_oracle_rows_are_diagnostic_only(tmp_path) -> None:
    summary = _smoke_export(tmp_path)["summary"]
    protocol = summary["protocol"]

    assert protocol["offline_replay_only"] is True
    assert protocol["primary_control"] == RA_EE_07_SAFE_GREEDY_CONTROL
    assert protocol["primary_candidate"] == RA_EE_07_DEPLOYABLE
    assert protocol["fixed_1w_diagnostic"] == RA_EE_07_FIXED_1W_DIAGNOSTIC
    assert protocol["constrained_power_oracle_isolation"] == RA_EE_07_CONSTRAINED_ORACLE
    assert protocol["association_proposal_diagnostic"] == RA_EE_07_ASSOC_PROPOSAL_DEPLOYABLE
    assert protocol["association_oracle_upper_bound"] == RA_EE_07_ASSOC_ORACLE_CONSTRAINED
    assert protocol["oracle_diagnostic_only"] is True

    oracle_summaries = [
        row
        for row in summary["candidate_summaries"]
        if row["power_semantics"]
        in {RA_EE_07_CONSTRAINED_ORACLE, RA_EE_07_ASSOC_ORACLE_CONSTRAINED}
    ]
    assert oracle_summaries
    assert all(row["diagnostic_only"] is True for row in oracle_summaries)
    assert all(row["primary_candidate"] is False for row in oracle_summaries)


def test_ra_ee_07_candidate_has_no_oracle_or_future_leakage(tmp_path) -> None:
    summary = _smoke_export(tmp_path)["summary"]
    primary = [
        row
        for row in summary["candidate_summaries"]
        if row["power_semantics"] == RA_EE_07_DEPLOYABLE
    ]

    assert primary
    assert all(row["primary_candidate"] is True for row in primary)
    assert all(row["diagnostic_only"] is False for row in primary)
    assert all(row["oracle_labels_used_for_runtime_decision"] is False for row in primary)
    assert all(row["future_outcomes_used_for_runtime_decision"] is False for row in primary)
    assert all(row["held_out_answers_used_for_runtime_decision"] is False for row in primary)


def test_ra_ee_07_deployable_allocator_candidates_are_exported(tmp_path) -> None:
    summary = _smoke_export(tmp_path)["summary"]
    candidate_labels = {
        row["allocator_label"]
        for row in summary["candidate_summaries"]
        if row["power_semantics"].startswith("deployable-candidate::")
        or row["power_semantics"] == RA_EE_07_DEPLOYABLE
    }

    assert {
        P05_SLACK_TRIM_TAIL_PROTECT,
        BOUNDED_LOCAL_SEARCH,
        FINITE_CODEBOOK_DP,
        DETERMINISTIC_HYBRID,
    }.issubset(candidate_labels)


def test_ra_ee_07_same_power_vector_feeds_numerator_and_denominator() -> None:
    settings = _settings_from_config(load_training_yaml(RA_EE_07_CONFIG))
    snapshot = _snapshot()
    one_w = _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics=RA_EE_07_FIXED_1W_DIAGNOSTIC,
        selected_power_profile="fixed-1w",
        power_vector=np.array([1.0, 1.0, 0.0], dtype=np.float64),
        settings=settings.audit,
    )
    two_w = _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics=RA_EE_07_DEPLOYABLE,
        selected_power_profile="fixed-2w",
        power_vector=np.array([2.0, 2.0, 0.0], dtype=np.float64),
        settings=settings.audit,
    )

    assert two_w["sum_user_throughput_bps"] > one_w["sum_user_throughput_bps"]
    assert two_w["total_active_beam_power_w"] > one_w["total_active_beam_power_w"]
    assert two_w["EE_system_bps_per_w"] != one_w["EE_system_bps_per_w"]


def test_ra_ee_07_budget_inactive_and_p05_guardrails() -> None:
    settings = _settings_from_config(load_training_yaml(RA_EE_07_CONFIG))
    snapshot = _snapshot()
    control = _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics=RA_EE_07_SAFE_GREEDY_CONTROL,
        selected_power_profile="fixed-1w",
        power_vector=np.array([1.0, 1.0, 0.0], dtype=np.float64),
        settings=settings.audit,
    )
    low_power = _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics=RA_EE_07_DEPLOYABLE,
        selected_power_profile="low-power-fake",
        power_vector=np.array([0.5, 0.5, 0.0], dtype=np.float64),
        settings=settings.audit,
    )
    bad_budget = _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics=RA_EE_07_DEPLOYABLE,
        selected_power_profile="budget-test",
        power_vector=np.array([2.0, 2.0, 0.25], dtype=np.float64),
        settings=settings.audit,
    )

    assert not _candidate_step_passes(
        row=low_power,
        matched_safe_row=control,
        settings=settings,
    )
    assert bad_budget["inactive_beam_nonzero_power"] is True
    assert bad_budget["per_beam_power_violation"] is False


def test_ra_ee_07_summary_and_review_exports_required_fields(tmp_path) -> None:
    outputs = _smoke_export(tmp_path)
    summary = outputs["summary"]

    assert outputs["ra_ee_07_constrained_power_allocator_distillation_summary"].exists()
    assert outputs["ra_ee_07_candidate_summary_csv"].exists()
    assert outputs["ra_ee_07_guardrail_checks_csv"].exists()
    assert outputs["ra_ee_07_step_metrics"].exists()
    assert outputs["review_md"].exists()
    assert "RA-EE-07 Constrained-Power Allocator Distillation Review" in (
        outputs["review_md"].read_text()
    )
    assert summary["bucket_results"]["held-out"]
    assert summary["seed_level_results"]
    assert summary["oracle_gap_diagnostics"]
    assert summary["proof_flags"]["scalar_reward_success_basis"] is False
    assert summary["proof_flags"]["catfish_disabled"] is True

    candidate = next(
        row
        for row in summary["candidate_summaries"]
        if row["power_semantics"] == RA_EE_07_DEPLOYABLE
    )
    for field in (
        "evaluation_bucket",
        "association_policy",
        "association_role",
        "allocator_label",
        "selected_allocator_candidate_distribution",
        "EE_system_aggregate_bps_per_w",
        "throughput_p05_user_step_bps",
        "served_ratio",
        "outage_ratio",
        "selected_power_vector_distribution",
        "total_active_beam_power_w_distribution",
        "denominator_varies_in_eval",
        "budget_violations",
        "per_beam_power_violations",
        "inactive_beam_nonzero_power_step_count",
        "oracle_gap_closed_ratio_distribution",
        "candidate_regret_bps_per_w_distribution",
    ):
        assert field in candidate


def test_ra_ee_07_ra_ee_04_05_06_06b_regression_smokes(tmp_path) -> None:
    ra04 = export_ra_ee_04_bounded_power_allocator_pilot(
        control_output_dir=tmp_path / "ra04-control",
        candidate_output_dir=tmp_path / "ra04-candidate",
        comparison_output_dir=tmp_path / "ra04-comparison",
        evaluation_seed_set=(100,),
        max_steps=1,
        policies=("hold-current",),
        include_oracle=True,
    )
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
        candidate_association_policies=("active-set-sticky-spread",),
        max_steps=1,
        include_oracle=True,
    )
    ra06b = export_ra_ee_06b_association_proposal_refinement(
        output_dir=tmp_path / "ra06b",
        calibration_seed_set=(100,),
        held_out_seed_set=(600,),
        candidate_association_policies=("sticky-oracle-count-local-search",),
        max_steps=1,
        include_oracle=True,
    )

    assert "ra_ee_04_decision" in ra04["summary"]
    assert "ra_ee_05_decision" in ra05["summary"]
    assert "ra_ee_06_decision" in ra06["summary"]
    assert "ra_ee_06b_decision" in ra06b["summary"]
