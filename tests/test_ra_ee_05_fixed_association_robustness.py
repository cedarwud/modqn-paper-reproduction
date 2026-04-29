from __future__ import annotations

from pathlib import Path

import numpy as np

from modqn_paper_reproduction.analysis.ra_ee_02_oracle_power_allocation import (
    _AuditSettings,
    _StepSnapshot,
    _evaluate_power_vector,
    _power_vector_for_candidate,
    _qos_guardrails_pass,
)
from modqn_paper_reproduction.analysis.ra_ee_05_fixed_association_robustness import (
    HELD_OUT_BUCKET,
    RA_EE_05_CANDIDATE,
    RA_EE_05_METHOD_LABEL,
    export_ra_ee_05_fixed_association_robustness,
)
from modqn_paper_reproduction.config_loader import build_environment, load_training_yaml
from modqn_paper_reproduction.env.step import (
    HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK,
    HOBS_POWER_SURFACE_STATIC_CONFIG,
)


BASELINE_CONFIG = "configs/modqn-paper-baseline.resolved-template.yaml"
RA_EE_05_CONFIG = "configs/ra-ee-05-fixed-association-robustness.resolved.yaml"
FROZEN_BASELINE_ARTIFACTS = (
    "artifacts/pilot-02-best-eval",
    "artifacts/run-9000",
    "artifacts/table-ii-200ep-01",
    "artifacts/fig-3-pilot-01",
    "artifacts/fig-4-pilot-01",
    "artifacts/fig-5-pilot-01",
    "artifacts/fig-6-pilot-01",
)


def _settings(*, budget_w: float = 8.0) -> _AuditSettings:
    return _AuditSettings(
        method_label=RA_EE_05_METHOD_LABEL,
        codebook_levels_w=(0.5, 0.75, 1.0, 1.5, 2.0),
        fixed_control_power_w=1.0,
        total_power_budget_w=budget_w,
        per_beam_max_power_w=2.0,
        active_base_power_w=0.25,
        load_scale_power_w=0.35,
        load_exponent=0.5,
        p05_min_ratio_vs_control=0.95,
        served_ratio_min_delta_vs_control=0.0,
        outage_ratio_max_delta_vs_control=0.0,
        oracle_max_demoted_beams=3,
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


def test_ra_ee_05_is_opt_in_and_forbidden_features_disabled() -> None:
    baseline_cfg = load_training_yaml(BASELINE_CONFIG)
    ra_ee_05_cfg = load_training_yaml(RA_EE_05_CONFIG)
    baseline_env = build_environment(baseline_cfg)
    ra_ee_05_env = build_environment(ra_ee_05_cfg)
    pilot = ra_ee_05_cfg["resolved_assumptions"][
        "ra_ee_05_fixed_association_robustness"
    ]["value"]

    assert baseline_env.power_surface_config.hobs_power_surface_mode == (
        HOBS_POWER_SURFACE_STATIC_CONFIG
    )
    assert ra_ee_05_env.power_surface_config.hobs_power_surface_mode == (
        HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK
    )
    assert ra_ee_05_cfg["track"]["phase"] == "RA-EE-05"
    assert pilot["method_label"] == RA_EE_05_METHOD_LABEL
    assert pilot["learned_association"] == "disabled"
    assert pilot["association_training"] == "disabled"
    assert pilot["joint_association_power_training"] == "disabled"
    assert pilot["catfish"] == "disabled"
    assert pilot["multi_catfish"] == "disabled"
    assert pilot["rb_bandwidth_allocation"] == "disabled"


def test_ra_ee_05_export_keeps_frozen_baseline_config_and_artifacts_untouched(
    tmp_path,
) -> None:
    before = Path(BASELINE_CONFIG).read_text()
    outputs = export_ra_ee_05_fixed_association_robustness(
        config_path=RA_EE_05_CONFIG,
        output_dir=tmp_path / "ra-ee-05",
        calibration_seed_set=(100,),
        held_out_seed_set=(600,),
        calibration_policies=("spread-valid",),
        held_out_policies=("random-valid-heldout", "spread-valid-heldout"),
        max_steps=1,
        include_oracle=True,
    )
    after = Path(BASELINE_CONFIG).read_text()

    assert before == after
    output_paths = {str(path) for key, path in outputs.items() if key != "summary"}
    for frozen in FROZEN_BASELINE_ARTIFACTS:
        assert all(not path.startswith(frozen) for path in output_paths)


def test_ra_ee_05_held_out_bucket_and_fixed_association_are_reported(tmp_path) -> None:
    outputs = export_ra_ee_05_fixed_association_robustness(
        config_path=RA_EE_05_CONFIG,
        output_dir=tmp_path / "ra-ee-05",
        calibration_seed_set=(100,),
        held_out_seed_set=(600,),
        calibration_policies=("hold-current",),
        held_out_policies=("random-valid-heldout",),
        max_steps=1,
        include_oracle=True,
    )
    summary = outputs["summary"]

    assert HELD_OUT_BUCKET in summary["bucket_results"]
    assert summary["bucket_results"][HELD_OUT_BUCKET]["trajectory_families"] == [
        "random-valid-heldout"
    ]
    assert summary["protocol"]["association_control"] == "fixed-by-trajectory"
    assert summary["protocol"]["learned_association"] == "disabled"
    assert summary["protocol"]["joint_association_power_training"] == "disabled"
    assert summary["proof_flags"]["fixed_association_only"] is True
    assert summary["proof_flags"]["learned_association_disabled"] is True
    assert summary["proof_flags"]["catfish_disabled"] is True


def test_ra_ee_05_inactive_beams_are_zero_and_budget_violations_are_detected() -> None:
    settings = _settings(budget_w=1.5)
    snapshot = _snapshot()
    fixed = _power_vector_for_candidate(snapshot, settings, "fixed-control")
    row = _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics=RA_EE_05_CANDIDATE,
        selected_power_profile="budget-test",
        power_vector=fixed,
        settings=settings,
    )

    assert fixed.tolist() == [1.0, 1.0, 0.0]
    assert row["inactive_beam_nonzero_power"] is False
    assert row["budget_violation"] is True
    assert row["budget_excess_w"] == 0.5


def test_ra_ee_05_same_power_vector_feeds_numerator_and_denominator() -> None:
    settings = _settings()
    snapshot = _snapshot()
    one_w = _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics="fixed-control",
        selected_power_profile="fixed-1w",
        power_vector=np.array([1.0, 1.0, 0.0], dtype=np.float64),
        settings=settings,
    )
    two_w = _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics=RA_EE_05_CANDIDATE,
        selected_power_profile="fixed-2w",
        power_vector=np.array([2.0, 2.0, 0.0], dtype=np.float64),
        settings=settings,
    )

    assert two_w["sum_user_throughput_bps"] > one_w["sum_user_throughput_bps"]
    assert two_w["total_active_beam_power_w"] > one_w["total_active_beam_power_w"]
    assert two_w["EE_system_bps_per_w"] != one_w["EE_system_bps_per_w"]


def test_ra_ee_05_p05_guardrail_rejects_low_power_fake_ee_gain() -> None:
    settings = _settings()
    snapshot = _snapshot()
    control = _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics="fixed-control",
        selected_power_profile="fixed-1w",
        power_vector=np.array([1.0, 1.0, 0.0], dtype=np.float64),
        settings=settings,
    )
    low_power = _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics=RA_EE_05_CANDIDATE,
        selected_power_profile="low-power-fake",
        power_vector=np.array([0.5, 0.5, 0.0], dtype=np.float64),
        settings=settings,
    )

    assert low_power["throughput_p05_user_step_bps"] < (
        0.95 * control["throughput_p05_user_step_bps"]
    )
    assert not _qos_guardrails_pass(
        candidate=low_power,
        control=control,
        settings=settings,
    )


def test_ra_ee_05_review_and_summary_exports_include_required_fields(tmp_path) -> None:
    outputs = export_ra_ee_05_fixed_association_robustness(
        config_path=RA_EE_05_CONFIG,
        output_dir=tmp_path / "ra-ee-05",
        calibration_seed_set=(100,),
        held_out_seed_set=(600,),
        calibration_policies=("spread-valid",),
        held_out_policies=("random-valid-heldout", "spread-valid-heldout"),
        max_steps=2,
        include_oracle=True,
    )
    summary = outputs["summary"]

    assert outputs["ra_ee_05_fixed_association_robustness_summary"].exists()
    assert outputs["ra_ee_05_candidate_summary_csv"].exists()
    assert outputs["ra_ee_05_step_metrics"].exists()
    assert outputs["review_md"].exists()
    assert "RA-EE-05 Fixed-Association Robustness Review" in (
        outputs["review_md"].read_text()
    )
    assert summary["protocol"]["catfish"] == "disabled"
    assert summary["protocol"]["learned_association"] == "disabled"
    assert summary["protocol"]["oracle_upper_bound_diagnostic_only"] is True

    candidate = next(
        row
        for row in summary["candidate_summaries"]
        if row["power_semantics"] == RA_EE_05_CANDIDATE
    )
    for field in (
        "evaluation_bucket",
        "EE_system_aggregate_bps_per_w",
        "EE_system_step_mean_bps_per_w",
        "throughput_mean_user_step_bps",
        "throughput_p05_user_step_bps",
        "served_ratio",
        "outage_ratio",
        "handover_count",
        "active_beam_count_distribution",
        "selected_power_profile_distribution",
        "selected_power_vector_distribution",
        "total_active_beam_power_w_distribution",
        "denominator_varies_in_eval",
        "budget_violations",
        "per_beam_power_violations",
        "inactive_beam_nonzero_power_step_count",
        "throughput_vs_EE_system_correlation",
    ):
        assert field in candidate
    assert summary["guardrail_checks"]
    assert summary["ranking_separation_result"]["comparison_control_vs_candidate"]
    assert summary["bucket_results"][HELD_OUT_BUCKET]
    assert summary["oracle_gap_diagnostics"]
