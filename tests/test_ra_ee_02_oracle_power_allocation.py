from __future__ import annotations

import numpy as np

from modqn_paper_reproduction.analysis.ra_ee_02_oracle_power_allocation import (
    _AuditSettings,
    _StepSnapshot,
    _evaluate_power_vector,
    _power_vector_for_candidate,
    _qos_guardrails_pass,
    export_ra_ee_02_oracle_power_allocation_audit,
)
from modqn_paper_reproduction.config_loader import build_environment, load_training_yaml
from modqn_paper_reproduction.env.step import (
    HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK,
    HOBS_POWER_SURFACE_STATIC_CONFIG,
)


BASELINE_CONFIG = "configs/modqn-paper-baseline.resolved-template.yaml"
RA_EE_02_CONFIG = "configs/ra-ee-02-oracle-power-allocation-audit.resolved.yaml"


def _settings(*, budget_w: float = 8.0) -> _AuditSettings:
    return _AuditSettings(
        method_label="RA-EE-MDP / RA-EE-MODQN",
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
        oracle_max_demoted_beams=2,
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


def test_ra_ee_02_leaves_baseline_static_config_behavior_unchanged() -> None:
    baseline_cfg = load_training_yaml(BASELINE_CONFIG)
    baseline_env = build_environment(baseline_cfg)

    assert baseline_env.power_surface_config.hobs_power_surface_mode == (
        HOBS_POWER_SURFACE_STATIC_CONFIG
    )


def test_ra_ee_02_oracle_audit_is_opt_in() -> None:
    baseline_cfg = load_training_yaml(BASELINE_CONFIG)
    ra_cfg = load_training_yaml(RA_EE_02_CONFIG)
    baseline_env = build_environment(baseline_cfg)
    ra_env = build_environment(ra_cfg)

    assert baseline_env.power_surface_config.hobs_power_surface_mode == (
        HOBS_POWER_SURFACE_STATIC_CONFIG
    )
    assert ra_env.power_surface_config.hobs_power_surface_mode == (
        HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK
    )
    assert ra_cfg["track"]["phase"] == "RA-EE-02"
    assert "RA-EE" in ra_cfg["track"]["method_family"]


def test_ra_ee_02_inactive_beams_have_zero_w() -> None:
    settings = _settings()
    snapshot = _snapshot()
    powers = _power_vector_for_candidate(snapshot, settings, "fixed-control")

    assert powers[0] == 1.0
    assert powers[1] == 1.0
    assert powers[2] == 0.0


def test_ra_ee_02_power_feeds_sinr_numerator_and_ee_denominator() -> None:
    settings = _settings()
    snapshot = _snapshot()
    one_w = _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics="fixed-control",
        selected_power_profile="fixed-1w-control",
        power_vector=np.array([1.0, 1.0, 0.0], dtype=np.float64),
        settings=settings,
    )
    two_w = _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics="fixed-high",
        selected_power_profile="fixed-2w",
        power_vector=np.array([2.0, 2.0, 0.0], dtype=np.float64),
        settings=settings,
    )

    assert two_w["sum_user_throughput_bps"] > one_w["sum_user_throughput_bps"]
    assert two_w["total_active_beam_power_w"] > one_w["total_active_beam_power_w"]
    assert two_w["EE_system_bps_per_w"] != one_w["EE_system_bps_per_w"]


def test_ra_ee_02_budget_violations_are_detected() -> None:
    settings = _settings(budget_w=1.5)
    snapshot = _snapshot()
    row = _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics="fixed-control",
        selected_power_profile="fixed-1w-control",
        power_vector=np.array([1.0, 1.0, 0.0], dtype=np.float64),
        settings=settings,
    )

    assert row["budget_violation"] is True
    assert row["budget_excess_w"] == 0.5


def test_ra_ee_02_qos_guardrail_rejects_low_power_fake_ee_gain() -> None:
    settings = _settings()
    snapshot = _snapshot()
    control = _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics="fixed-control",
        selected_power_profile="fixed-1w-control",
        power_vector=np.array([1.0, 1.0, 0.0], dtype=np.float64),
        settings=settings,
    )
    low_power = _evaluate_power_vector(
        snapshot=snapshot,
        power_semantics="low-power-fake",
        selected_power_profile="fixed-0.5w",
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


def test_ra_ee_02_summary_includes_required_proof_flags(tmp_path) -> None:
    outputs = export_ra_ee_02_oracle_power_allocation_audit(
        config_path=RA_EE_02_CONFIG,
        output_dir=tmp_path / "ra-ee-02",
        include_learned=False,
        evaluation_seed_set=(100,),
        max_steps=1,
        policies=("spread-valid",),
        power_candidates=("fixed-control", "budget-trim", "constrained-oracle"),
    )
    summary = outputs["summary"]
    proof = summary["proof_flags"]

    assert outputs["ra_ee_02_oracle_power_allocation_summary"].exists()
    assert outputs["ra_ee_02_step_metrics"].exists()
    assert outputs["ra_ee_02_candidate_summary"].exists()
    assert outputs["review_md"].exists()
    for field in (
        "denominator_changed_by_power_decision",
        "ranking_separates_under_same_policy_rescore",
        "has_budget_respecting_candidate",
        "oracle_or_heuristic_beats_fixed_control_on_EE",
        "QoS_guardrails_pass",
        "selected_profile_not_single_point_on_noncollapsed_trajectories",
        "active_power_not_single_point_on_noncollapsed_trajectories",
        "no_budget_violations_for_accepted_candidate",
    ):
        assert field in proof
    assert summary["protocol"]["training"] == "not-run"
    assert summary["protocol"]["catfish"] == "disabled"
