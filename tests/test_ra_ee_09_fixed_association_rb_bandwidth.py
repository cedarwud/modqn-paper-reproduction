from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from modqn_paper_reproduction.analysis.ra_ee_02_oracle_power_allocation import (
    _StepSnapshot,
    _evaluate_power_vector,
)
from modqn_paper_reproduction.analysis.ra_ee_07_constrained_power_allocator_distillation import (
    export_ra_ee_07_constrained_power_allocator_distillation,
)
from modqn_paper_reproduction.analysis.ra_ee_09_fixed_association_rb_bandwidth import (
    DEFAULT_CONFIG,
    RA_EE_09_ASSUMPTION_KEY,
    RA_EE_09_CANDIDATE_ALLOCATOR,
    RA_EE_09_CANDIDATE_MAX_EQUAL_SHARE_MULTIPLIER,
    RA_EE_09_CANDIDATE_MIN_EQUAL_SHARE_MULTIPLIER,
    RA_EE_09_CONTROL,
    RA_EE_09_EQUAL_SHARE_ALLOCATOR,
    RA_EE_09_GATE_ID,
    RA_EE_09_POWER_ALLOCATOR_ID,
    RA_EE_09_PREDECLARED_RESOURCE_EFFICIENCY_METRIC,
    RA_EE_09_RESOURCE_UNIT,
    _audit_resource_accounting,
    _bounded_qos_slack_resource_share_allocator,
    _candidate_settings_from_control,
    _compute_user_throughputs_from_resource,
    _equal_share_resource_fractions,
    _settings_from_config,
    export_ra_ee_09_fixed_association_rb_bandwidth_candidate,
    export_ra_ee_09_fixed_association_rb_bandwidth_control,
    export_ra_ee_09_fixed_association_rb_bandwidth_matched_comparison,
    ra_ee_09_resource_accounting_enabled,
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
RA_EE_09_CONFIG = DEFAULT_CONFIG
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
        assignments=np.array([0, 0, 1], dtype=np.int32),
        active_mask=np.array([True, True, False], dtype=bool),
        beam_loads=np.array([2.0, 1.0, 0.0], dtype=np.float64),
        unit_snr_by_user=np.array([1.0, 3.0, 0.5], dtype=np.float64),
        bandwidth_hz=1.0,
        handover_count=0,
        r2_mean=0.0,
    )


def _smoke_export(tmp_path):
    return export_ra_ee_09_fixed_association_rb_bandwidth_control(
        config_path=RA_EE_09_CONFIG,
        output_dir=tmp_path / "ra-ee-09",
        calibration_seed_set=(100,),
        held_out_seed_set=(600,),
        calibration_policies=("hold-current",),
        held_out_policies=("random-valid-heldout",),
        max_steps=1,
    )


def _smoke_candidate_export(tmp_path):
    return export_ra_ee_09_fixed_association_rb_bandwidth_candidate(
        config_path=RA_EE_09_CONFIG,
        output_dir=tmp_path / "ra-ee-09-candidate",
        calibration_seed_set=(100,),
        held_out_seed_set=(600,),
        calibration_policies=("hold-current",),
        held_out_policies=("random-valid-heldout",),
        max_steps=1,
    )


def _smoke_matched_export(tmp_path):
    return export_ra_ee_09_fixed_association_rb_bandwidth_matched_comparison(
        config_path=RA_EE_09_CONFIG,
        output_dir=tmp_path / "ra-ee-09-matched",
        held_out_seed_set=(600,),
        held_out_policies=("random-valid-heldout",),
        max_steps=1,
    )


def test_ra_ee_09_is_explicit_opt_in_and_disabled_elsewhere() -> None:
    baseline_cfg = load_training_yaml(BASELINE_CONFIG)
    ra_ee_07_cfg = load_training_yaml(RA_EE_07_CONFIG)
    ra_ee_09_cfg = load_training_yaml(RA_EE_09_CONFIG)
    baseline_env = build_environment(baseline_cfg)
    ra_ee_09_env = build_environment(ra_ee_09_cfg)
    gate = ra_ee_09_cfg["resolved_assumptions"][RA_EE_09_ASSUMPTION_KEY]["value"]
    ra_ee_07_gate = ra_ee_07_cfg["resolved_assumptions"][
        "ra_ee_07_constrained_power_allocator_distillation"
    ]["value"]

    assert not ra_ee_09_resource_accounting_enabled(
        baseline_cfg,
        config_path=BASELINE_CONFIG,
    )
    assert not ra_ee_09_resource_accounting_enabled(
        ra_ee_07_cfg,
        config_path=RA_EE_07_CONFIG,
    )
    assert ra_ee_09_resource_accounting_enabled(
        ra_ee_09_cfg,
        config_path=RA_EE_09_CONFIG,
    )
    assert baseline_env.power_surface_config.hobs_power_surface_mode == (
        HOBS_POWER_SURFACE_STATIC_CONFIG
    )
    assert ra_ee_09_env.power_surface_config.hobs_power_surface_mode == (
        HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK
    )
    assert ra_ee_09_cfg["track"]["phase"] == RA_EE_09_GATE_ID
    assert gate["ra_ee_gate_id"] == RA_EE_09_GATE_ID
    assert gate["resource_unit"] == RA_EE_09_RESOURCE_UNIT
    assert gate["resource_allocator_id"] == RA_EE_09_EQUAL_SHARE_ALLOCATOR
    assert gate["power_allocator_id"] == RA_EE_09_POWER_ALLOCATOR_ID
    assert gate["learned_association_disabled"] is True
    assert gate["hierarchical_RL_disabled"] is True
    assert gate["catfish_disabled"] is True
    assert gate["phase03c_continuation_disabled"] is True
    assert gate["scalar_reward_success_basis"] is False
    assert ra_ee_07_gate["rb_bandwidth_allocation"] == "disabled"


def test_ra_ee_09_namespace_gating_rejects_non_ra_ee_09_configs(tmp_path) -> None:
    ra_ee_09_cfg = load_training_yaml(RA_EE_09_CONFIG)

    with pytest.raises(ValueError, match="ra-ee-09"):
        _settings_from_config(
            ra_ee_09_cfg,
            config_path="configs/not-ra-ee-09-control.resolved.yaml",
        )
    with pytest.raises(ValueError, match="ra-ee-09"):
        export_ra_ee_09_fixed_association_rb_bandwidth_control(
            config_path=RA_EE_07_CONFIG,
            output_dir=tmp_path / "bad",
            max_steps=1,
        )


def test_equal_share_resource_formula_matches_existing_throughput_path() -> None:
    settings = _settings_from_config(
        load_training_yaml(RA_EE_09_CONFIG),
        config_path=RA_EE_09_CONFIG,
    )
    snapshot = _snapshot()
    power_vector = np.array([1.5, 0.75, 0.0], dtype=np.float64)
    shares = _equal_share_resource_fractions(snapshot)
    existing = _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics=RA_EE_09_CONTROL,
        selected_power_profile="synthetic",
        power_vector=power_vector,
        settings=settings.power_settings.audit,
    )
    generalized = _compute_user_throughputs_from_resource(
        snapshot,
        power_vector,
        shares,
    )
    accounting = _audit_resource_accounting(snapshot, shares, settings.resource)

    np.testing.assert_allclose(
        generalized,
        existing["_user_throughputs"],
        rtol=0.0,
        atol=settings.resource.resource_sum_tolerance,
    )
    assert accounting["active_beam_resource_sum_exact"] is True
    assert accounting["inactive_beam_nonzero_resource"] is False
    assert accounting["resource_budget_violation"] is False
    assert accounting["per_beam_resource_sum"] == "1 1 0"


def test_resource_accounting_reports_overuse_underuse_and_inactive_nonzero() -> None:
    settings = _settings_from_config(
        load_training_yaml(RA_EE_09_CONFIG),
        config_path=RA_EE_09_CONFIG,
    )
    snapshot = _snapshot()
    shares = _equal_share_resource_fractions(snapshot)
    overused = shares.copy()
    overused[0] += 0.1
    inactive_usage = np.array([0.0, 0.0, 0.01], dtype=np.float64)

    report = _audit_resource_accounting(
        snapshot,
        overused,
        settings.resource,
        inactive_beam_resource_usage=inactive_usage,
    )

    assert report["active_beam_resource_sum_exact"] is False
    assert report["active_beam_resource_sum_violation_count"] == 1
    assert report["inactive_beam_nonzero_resource"] is True
    assert report["inactive_beam_nonzero_resource_count"] == 1
    assert report["per_user_max_resource_violation_count"] == 1
    assert report["resource_budget_violation"] is True
    assert report["resource_budget_violation_count"] >= 1


def test_ra_ee_09_control_export_keeps_baseline_untouched_and_proves_parity(
    tmp_path,
) -> None:
    before = Path(BASELINE_CONFIG).read_text()
    outputs = _smoke_export(tmp_path)
    after = Path(BASELINE_CONFIG).read_text()
    summary = outputs["summary"]
    proof = summary["proof_flags"]
    metadata = summary["metadata"]

    assert before == after
    output_paths = {str(path) for key, path in outputs.items() if key != "summary"}
    for frozen in FROZEN_BASELINE_ARTIFACTS:
        assert all(not path.startswith(frozen) for path in output_paths)
    assert outputs["ra_ee_09_fixed_association_rb_bandwidth_control_summary"].exists()
    assert outputs["ra_ee_09_control_summary_csv"].exists()
    assert outputs["ra_ee_09_step_resource_trace"].exists()
    assert outputs["resource_budget_report"].exists()
    assert outputs["review_md"].exists()
    assert "RA-EE-09 Fixed-Association" in outputs["review_md"].read_text()

    assert summary["ra_ee_09_slice_09a_09c_decision"] == "PASS_TO_SLICE_09D"
    assert metadata["ra_ee_gate_id"] == RA_EE_09_GATE_ID
    assert metadata["association_mode"] == "fixed-replay"
    assert metadata["power_allocator_id"] == RA_EE_09_POWER_ALLOCATOR_ID
    assert metadata["resource_unit"] == RA_EE_09_RESOURCE_UNIT
    assert metadata["learned_association_disabled"] is True
    assert metadata["hierarchical_RL_disabled"] is True
    assert metadata["catfish_disabled"] is True
    assert metadata["phase03c_continuation_disabled"] is True
    assert metadata["scalar_reward_success_basis"] is False
    assert proof["equal_share_throughput_parity"] is True
    assert proof["active_beam_resource_sum_exact"] is True
    assert proof["inactive_beam_zero_resource"] is True
    assert proof["same_power_vector_as_control"] is True
    assert proof["fixed_association_enforced"] is True
    assert proof["resource_allocation_feedback_to_power_decision"] is False
    assert proof["frozen_baseline_mutation"] is False


def test_ra_ee_09_summary_exposes_fixed_association_and_resource_schema(
    tmp_path,
) -> None:
    summary = _smoke_export(tmp_path)["summary"]
    control = summary["control_summaries"][0]
    budget = summary["resource_budget_report"]["overall"]

    assert control["power_semantics"] == RA_EE_09_CONTROL
    assert control["association_mode"] == "fixed-replay"
    assert control["resource_unit"] == RA_EE_09_RESOURCE_UNIT
    assert control["resource_allocator_id"] == RA_EE_09_EQUAL_SHARE_ALLOCATOR
    assert control["candidate_allocator_enabled"] is False
    assert control["fixed_association_enforced"] is True
    assert control["same_power_vector_as_control"] is True
    assert control["resource_allocation_after_power_vector_selection"] is True
    assert control["resource_allocation_feedback_to_power_decision"] is False
    assert control["equal_share_throughput_parity"] is True
    assert control["active_beam_resource_sum_exact"] is True
    assert control["inactive_beam_zero_resource"] is True
    assert budget["resource_budget_violation_count"] == 0
    assert budget["active_beam_resource_sum_exact"] is True
    assert budget["inactive_beam_zero_resource"] is True
    assert "resource_fractions" in summary["step_resource_trace_schema_fields"]
    assert "per_beam_resource_sum" in summary["step_resource_trace_schema_fields"]
    assert "inactive_beam_nonzero_resource" in summary["step_resource_trace_schema_fields"]


def test_ra_ee_09_does_not_change_existing_ra_ee_07_export_contract(tmp_path) -> None:
    ra07 = export_ra_ee_07_constrained_power_allocator_distillation(
        output_dir=tmp_path / "ra07",
        calibration_seed_set=(100,),
        held_out_seed_set=(600,),
        diagnostic_seed_set=(600,),
        calibration_policies=("hold-current",),
        held_out_policies=("random-valid-heldout",),
        diagnostic_association_policies=("sticky-oracle-count-local-search",),
        max_steps=1,
        include_oracle=False,
        include_association_diagnostics=False,
    )

    assert ra07["summary"]["protocol"]["rb_bandwidth_allocation"] == "disabled/not-modeled"
    assert ra07["summary"]["proof_flags"]["frozen_baseline_mutation"] is False
    assert "ra_ee_07_decision" in ra07["summary"]


def test_bounded_qos_slack_candidate_resource_budget_and_bounds_are_exact() -> None:
    control_settings = _settings_from_config(
        load_training_yaml(RA_EE_09_CONFIG),
        config_path=RA_EE_09_CONFIG,
    )
    candidate_settings = _candidate_settings_from_control(control_settings)
    snapshot = _snapshot()
    power_vector = np.array([1.5, 0.75, 0.0], dtype=np.float64)

    shares = _bounded_qos_slack_resource_share_allocator(
        snapshot,
        power_vector,
        candidate_settings.resource,
    )
    accounting = _audit_resource_accounting(
        snapshot,
        shares,
        candidate_settings.resource,
    )

    assert accounting["active_beam_resource_sum_exact"] is True
    assert accounting["inactive_beam_nonzero_resource"] is False
    assert accounting["resource_budget_violation"] is False
    assert accounting["per_beam_resource_sum"] == "1 1 0"
    for uid, beam_idx in enumerate(snapshot.assignments.tolist()):
        load = max(float(snapshot.beam_loads[int(beam_idx)]), 1.0)
        lower = RA_EE_09_CANDIDATE_MIN_EQUAL_SHARE_MULTIPLIER / load
        upper = min(RA_EE_09_CANDIDATE_MAX_EQUAL_SHARE_MULTIPLIER / load, 1.0)
        assert float(shares[uid]) >= lower - candidate_settings.resource.resource_sum_tolerance
        assert float(shares[uid]) <= upper + candidate_settings.resource.resource_sum_tolerance


def test_ra_ee_09_candidate_export_keeps_association_handover_and_power_fixed(
    tmp_path,
) -> None:
    outputs = _smoke_candidate_export(tmp_path)
    summary = outputs["summary"]
    metadata = summary["metadata"]
    proof = summary["proof_flags"]
    candidate = summary["candidate_summaries"][0]
    budget = summary["resource_budget_report"]["overall"]

    assert outputs["ra_ee_09_fixed_association_rb_bandwidth_candidate_summary"].exists()
    assert outputs["ra_ee_09_candidate_summary_csv"].exists()
    assert outputs["ra_ee_09_candidate_step_resource_trace"].exists()
    assert outputs["resource_budget_report"].exists()
    assert outputs["review_md"].exists()
    assert summary["ra_ee_09_slice_09d_decision"] == "PASS_TO_SLICE_09E"
    assert metadata["candidate_allocator_enabled"] is True
    assert metadata["resource_allocator_id"] == RA_EE_09_CANDIDATE_ALLOCATOR
    assert metadata["learned_association_disabled"] is True
    assert metadata["hierarchical_RL_disabled"] is True
    assert metadata["catfish_disabled"] is True
    assert metadata["scalar_reward_success_basis"] is False
    assert candidate["candidate_allocator_enabled"] is True
    assert candidate["resource_allocator_id"] == RA_EE_09_CANDIDATE_ALLOCATOR
    assert proof["fixed_association_enforced"] is True
    assert proof["handover_trajectory_unchanged"] is True
    assert proof["same_power_vector_as_control"] is True
    assert proof["resource_allocation_after_power_vector_selection"] is True
    assert proof["resource_allocation_feedback_to_power_decision"] is False
    assert proof["active_beam_resource_sum_exact"] is True
    assert proof["inactive_beam_zero_resource"] is True
    assert proof["zero_resource_budget_violations"] is True
    assert budget["resource_budget_violation_count"] == 0
    assert budget["active_beam_resource_sum_exact"] is True
    assert budget["inactive_beam_zero_resource"] is True


def test_ra_ee_09_candidate_forbidden_modes_remain_enforced(tmp_path) -> None:
    summary = _smoke_candidate_export(tmp_path)["summary"]
    proof = summary["proof_flags"]
    metadata = summary["metadata"]

    assert proof["learned_association_disabled"] is True
    assert proof["hierarchical_RL_disabled"] is True
    assert proof["joint_association_power_training_disabled"] is True
    assert proof["catfish_disabled"] is True
    assert proof["phase03c_continuation_disabled"] is True
    assert proof["oracle_labels_future_or_heldout_answers_disabled"] is True
    assert proof["scalar_reward_success_basis"] is False
    assert proof["full_RA_EE_MODQN_claim"] is False
    assert metadata["candidate_allocator_enabled"] is True
    assert metadata["resource_allocator_id"] == RA_EE_09_CANDIDATE_ALLOCATOR


def test_ra_ee_09_matched_comparison_exports_held_out_boundary_proof(tmp_path) -> None:
    outputs = _smoke_matched_export(tmp_path)
    summary = outputs["summary"]
    proof = summary["proof_flags"]
    boundary = summary["matched_boundary_proof"]
    deltas = summary["matched_comparison"]["deltas"]

    assert outputs["paired_comparison"].exists()
    assert outputs["ra_ee_09_matched_step_comparison"].exists()
    assert outputs["ra_ee_09_matched_summary_csv"].exists()
    assert outputs["resource_budget_report"].exists()
    assert outputs["review_md"].exists()
    assert "RA-EE-09 Slice 09E" in outputs["review_md"].read_text()

    assert summary["ra_ee_09_slice_09e_decision"] in {
        "PASS",
        "BLOCK",
        "NEEDS MORE EVIDENCE",
    }
    assert summary["protocol"]["implementation_slice"] == "09E"
    assert summary["protocol"]["training"] == "none; offline replay only"
    assert summary["protocol"]["held_out_bucket_id"] == "held-out"
    assert summary["protocol"]["evaluation_seed_set"] == [600]
    assert summary["protocol"]["fixed_association_trajectories"] == [
        "random-valid-heldout"
    ]
    assert summary["protocol"]["scalar_reward_success_basis"] is False
    assert summary["protocol"]["predeclared_resource_efficiency_metric"] == (
        RA_EE_09_PREDECLARED_RESOURCE_EFFICIENCY_METRIC
    )
    assert summary["metadata"]["resource_allocator_id"] == RA_EE_09_CANDIDATE_ALLOCATOR
    assert summary["metadata"]["candidate_allocator_enabled"] is True
    assert summary["metadata"]["scalar_reward_success_basis"] is False

    assert boundary["same_evaluation_schedule"] is True
    assert boundary["same_association_hash_per_step"] is True
    assert boundary["same_association_schedule_hash"] is True
    assert boundary["same_power_vector_hash_per_step"] is True
    assert boundary["same_effective_power_schedule_hash"] is True
    assert boundary["resource_allocation_feedback_to_power_decision"] is False
    assert boundary["matched_step_count"] == 1
    assert isinstance(boundary["evaluation_schedule_hash"], str)

    assert proof["offline_fixed_association_replay_only"] is True
    assert proof["matched_control_vs_candidate_comparison"] is True
    assert proof["same_evaluation_schedule"] is True
    assert proof["same_association_schedule_hash"] is True
    assert proof["same_effective_power_vector_hash_per_step"] is True
    assert proof["same_RA_EE_07_power_boundary"] is True
    assert proof["candidate_does_not_change_association"] is True
    assert proof["candidate_does_not_change_handover"] is True
    assert proof["resource_allocation_feedback_to_power_decision"] is False
    assert proof["zero_power_violations"] is True
    assert proof["zero_resource_budget_violations"] is True
    assert proof["active_beam_resource_sum_exact"] is True
    assert proof["inactive_beam_resource_usage_zero"] is True
    assert proof["scalar_reward_success_basis"] is False
    assert proof["full_RA_EE_MODQN_claim"] is False

    assert "simulated_EE_system_delta_bps_per_w" in deltas
    assert "sum_throughput_delta_bps" in deltas
    assert "mean_throughput_delta_bps" in deltas
    assert "p05_throughput_ratio" in deltas
    assert "served_ratio_delta" in deltas
    assert "outage_ratio_delta" in deltas
    assert "handover_count_delta" in deltas
    assert deltas["handover_count_delta"] == 0
    assert deltas["total_active_power_w_sum_delta"] == 0.0
    assert deltas["active_resource_budget_sum_delta"] == 0.0
    assert deltas["predeclared_resource_efficiency_metric"] == (
        RA_EE_09_PREDECLARED_RESOURCE_EFFICIENCY_METRIC
    )
