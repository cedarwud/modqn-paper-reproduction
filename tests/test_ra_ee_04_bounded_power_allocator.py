from __future__ import annotations

import numpy as np

from modqn_paper_reproduction.analysis.ra_ee_02_oracle_power_allocation import (
    _AuditSettings,
    _StepSnapshot,
    _evaluate_power_vector,
    _qos_guardrails_pass,
)
from modqn_paper_reproduction.analysis.ra_ee_04_bounded_power_allocator import (
    RA_EE_04_CANDIDATE,
    export_ra_ee_04_bounded_power_allocator_pilot,
)
from modqn_paper_reproduction.config_loader import build_environment, load_training_yaml
from modqn_paper_reproduction.env.step import (
    HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK,
    HOBS_POWER_SURFACE_STATIC_CONFIG,
)


BASELINE_CONFIG = "configs/modqn-paper-baseline.resolved-template.yaml"
RA_EE_04_CONTROL_CONFIG = "configs/ra-ee-04-bounded-power-allocator-control.resolved.yaml"
RA_EE_04_CANDIDATE_CONFIG = (
    "configs/ra-ee-04-bounded-power-allocator-candidate.resolved.yaml"
)


def _settings(*, budget_w: float = 8.0) -> _AuditSettings:
    return _AuditSettings(
        method_label="RA-EE-MDP",
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
        trajectory_policy="synthetic-spread",
        evaluation_seed=100,
        step_index=1,
        assignments=np.array([0, 1], dtype=np.int32),
        active_mask=np.array([True, True, False], dtype=bool),
        beam_loads=np.array([1.0, 1.0, 0.0], dtype=np.float64),
        unit_snr_by_user=np.array([1.0, 1.0], dtype=np.float64),
        bandwidth_hz=1.0,
        handover_count=0,
        r2_mean=0.0,
    )


def test_ra_ee_04_leaves_frozen_baseline_static_config() -> None:
    baseline_cfg = load_training_yaml(BASELINE_CONFIG)
    baseline_env = build_environment(baseline_cfg)

    assert baseline_env.power_surface_config.hobs_power_surface_mode == (
        HOBS_POWER_SURFACE_STATIC_CONFIG
    )


def test_ra_ee_04_is_opt_in_and_catfish_disabled() -> None:
    baseline_cfg = load_training_yaml(BASELINE_CONFIG)
    candidate_cfg = load_training_yaml(RA_EE_04_CANDIDATE_CONFIG)
    control_cfg = load_training_yaml(RA_EE_04_CONTROL_CONFIG)
    baseline_env = build_environment(baseline_cfg)
    candidate_env = build_environment(candidate_cfg)
    control_env = build_environment(control_cfg)
    candidate_pilot = candidate_cfg["resolved_assumptions"][
        "ra_ee_04_bounded_power_allocator"
    ]["value"]

    assert baseline_env.power_surface_config.hobs_power_surface_mode == (
        HOBS_POWER_SURFACE_STATIC_CONFIG
    )
    assert candidate_env.power_surface_config.hobs_power_surface_mode == (
        HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK
    )
    assert control_env.power_surface_config.hobs_power_surface_mode == (
        HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK
    )
    assert candidate_cfg["track"]["phase"] == "RA-EE-04"
    assert candidate_pilot["catfish"] == "disabled"
    assert candidate_pilot["multi_catfish"] == "disabled"
    assert candidate_pilot["learned_association"] == "disabled"
    assert candidate_pilot["training_episodes"] == 20


def test_ra_ee_04_inactive_beams_are_zero_and_budget_detected() -> None:
    settings = _settings(budget_w=1.5)
    snapshot = _snapshot()
    row = _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics=RA_EE_04_CANDIDATE,
        selected_power_profile="budget-test",
        power_vector=np.array([1.0, 1.0, 0.0], dtype=np.float64),
        settings=settings,
    )

    assert row["inactive_beam_nonzero_power"] is False
    assert row["budget_violation"] is True
    assert row["budget_excess_w"] == 0.5


def test_ra_ee_04_same_power_vector_feeds_numerator_and_denominator() -> None:
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
        power_semantics=RA_EE_04_CANDIDATE,
        selected_power_profile="fixed-2w",
        power_vector=np.array([2.0, 2.0, 0.0], dtype=np.float64),
        settings=settings,
    )

    assert two_w["sum_user_throughput_bps"] > one_w["sum_user_throughput_bps"]
    assert two_w["total_active_beam_power_w"] > one_w["total_active_beam_power_w"]
    assert two_w["EE_system_bps_per_w"] != one_w["EE_system_bps_per_w"]


def test_ra_ee_04_qos_guardrail_rejects_low_power_fake_ee_gain() -> None:
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
        power_semantics=RA_EE_04_CANDIDATE,
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


def test_ra_ee_04_artifacts_include_required_metrics(tmp_path) -> None:
    outputs = export_ra_ee_04_bounded_power_allocator_pilot(
        control_config_path=RA_EE_04_CONTROL_CONFIG,
        candidate_config_path=RA_EE_04_CANDIDATE_CONFIG,
        control_output_dir=tmp_path / "control",
        candidate_output_dir=tmp_path / "candidate",
        comparison_output_dir=tmp_path / "candidate" / "paired-comparison-vs-control",
        evaluation_seed_set=(100,),
        max_steps=1,
        policies=("spread-valid",),
        include_oracle=True,
    )
    summary = outputs["summary"]

    assert outputs["ra_ee_04_bounded_power_allocator_summary"].exists()
    assert outputs["ra_ee_04_control_step_metrics"].exists()
    assert outputs["ra_ee_04_candidate_step_metrics"].exists()
    assert outputs["ra_ee_04_training_metrics"].exists()
    assert outputs["review_md"].exists()
    assert summary["protocol"]["learned_association"] == "disabled"
    assert summary["protocol"]["catfish"] == "disabled"
    assert summary["protocol"]["training_episodes"] == 20

    candidate = next(
        row
        for row in summary["candidate_summaries"]
        if row["power_semantics"] == RA_EE_04_CANDIDATE
    )
    for field in (
        "EE_system_aggregate_bps_per_w",
        "EE_system_step_mean_bps_per_w",
        "throughput_mean_user_step_bps",
        "throughput_p05_user_step_bps",
        "served_ratio",
        "outage_ratio",
        "active_beam_count_distribution",
        "selected_power_profile_distribution",
        "total_active_beam_power_w_distribution",
        "denominator_varies_in_eval",
        "budget_violations",
        "per_beam_power_violations",
        "inactive_beam_nonzero_power_step_count",
        "throughput_vs_EE_system_correlation",
    ):
        assert field in candidate
    assert summary["proof_flags"]["fixed_association_only"] is True
    assert summary["proof_flags"]["learned_association_disabled"] is True
