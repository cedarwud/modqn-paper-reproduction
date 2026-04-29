from __future__ import annotations

from pathlib import Path

import numpy as np

from modqn_paper_reproduction.analysis.ra_ee_06_association_counterfactual_oracle import (
    ACTIVE_SET_LOAD_SPREAD,
    DEFAULT_CONFIG,
    FIXED_HOLD_CURRENT,
    PER_USER_GREEDY_BEST_BEAM,
    RA_EE_06_CANDIDATE,
    RA_EE_06_GREEDY_DIAGNOSTIC,
    RA_EE_06_MATCHED_CONTROL,
    RA_EE_06_METHOD_LABEL,
    RA_EE_06_ORACLE,
    _select_actions_for_association_policy,
    _settings_from_config,
    export_ra_ee_06_association_counterfactual_oracle,
)
from modqn_paper_reproduction.config_loader import build_environment, load_training_yaml
from modqn_paper_reproduction.env.step import (
    HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK,
    HOBS_POWER_SURFACE_STATIC_CONFIG,
)


BASELINE_CONFIG = "configs/modqn-paper-baseline.resolved-template.yaml"
RA_EE_06_CONFIG = DEFAULT_CONFIG
FROZEN_BASELINE_ARTIFACTS = (
    "artifacts/pilot-02-best-eval",
    "artifacts/run-9000",
    "artifacts/table-ii-200ep-01",
    "artifacts/fig-3-pilot-01",
    "artifacts/fig-4-pilot-01",
    "artifacts/fig-5-pilot-01",
    "artifacts/fig-6-pilot-01",
)


def test_ra_ee_06_is_opt_in_and_forbidden_features_disabled() -> None:
    baseline_cfg = load_training_yaml(BASELINE_CONFIG)
    ra_ee_06_cfg = load_training_yaml(RA_EE_06_CONFIG)
    baseline_env = build_environment(baseline_cfg)
    ra_ee_06_env = build_environment(ra_ee_06_cfg)
    gate = ra_ee_06_cfg["resolved_assumptions"][
        "ra_ee_06_association_counterfactual_oracle"
    ]["value"]

    assert baseline_env.power_surface_config.hobs_power_surface_mode == (
        HOBS_POWER_SURFACE_STATIC_CONFIG
    )
    assert ra_ee_06_env.power_surface_config.hobs_power_surface_mode == (
        HOBS_POWER_SURFACE_PHASE_03C_B_POWER_CODEBOOK
    )
    assert ra_ee_06_cfg["track"]["phase"] == "RA-EE-06"
    assert gate["method_label"] == RA_EE_06_METHOD_LABEL
    assert gate["learned_association"] == "disabled"
    assert gate["association_training"] == "disabled"
    assert gate["joint_association_power_training"] == "disabled"
    assert gate["catfish"] == "disabled"
    assert gate["multi_catfish"] == "disabled"
    assert gate["rb_bandwidth_allocation"] == "disabled"
    assert gate["matched_control_association_policy"] == FIXED_HOLD_CURRENT
    assert PER_USER_GREEDY_BEST_BEAM in gate["diagnostic_association_policies"]


def test_ra_ee_06_active_set_policy_is_not_per_user_greedy() -> None:
    cfg = load_training_yaml(RA_EE_06_CONFIG)
    settings = _settings_from_config(cfg)
    env = build_environment(cfg)
    env_seed_seq, mobility_seed_seq = np.random.SeedSequence(600).spawn(2)
    states, masks, _diag = env.reset(
        np.random.default_rng(env_seed_seq),
        np.random.default_rng(mobility_seed_seq),
    )

    active_set_actions = _select_actions_for_association_policy(
        ACTIVE_SET_LOAD_SPREAD,
        user_states=states,
        masks=masks,
        current_assignments=env.current_assignments(),
        settings=settings,
    )
    greedy_actions = _select_actions_for_association_policy(
        PER_USER_GREEDY_BEST_BEAM,
        user_states=states,
        masks=masks,
        current_assignments=env.current_assignments(),
        settings=settings,
    )

    assert len(set(active_set_actions.tolist())) >= settings.min_active_beams
    assert len(set(active_set_actions.tolist())) <= settings.max_active_beams
    assert active_set_actions.shape == greedy_actions.shape
    assert not np.array_equal(active_set_actions, greedy_actions)


def test_ra_ee_06_export_keeps_frozen_baseline_config_and_artifacts_untouched(
    tmp_path,
) -> None:
    before = Path(BASELINE_CONFIG).read_text()
    outputs = export_ra_ee_06_association_counterfactual_oracle(
        config_path=RA_EE_06_CONFIG,
        output_dir=tmp_path / "ra-ee-06",
        calibration_seed_set=(100,),
        held_out_seed_set=(600,),
        candidate_association_policies=(ACTIVE_SET_LOAD_SPREAD,),
        max_steps=1,
        include_oracle=True,
    )
    after = Path(BASELINE_CONFIG).read_text()

    assert before == after
    output_paths = {str(path) for key, path in outputs.items() if key != "summary"}
    for frozen in FROZEN_BASELINE_ARTIFACTS:
        assert all(not path.startswith(frozen) for path in output_paths)


def test_ra_ee_06_summary_exports_contracts_and_comparators(tmp_path) -> None:
    outputs = export_ra_ee_06_association_counterfactual_oracle(
        config_path=RA_EE_06_CONFIG,
        output_dir=tmp_path / "ra-ee-06",
        calibration_seed_set=(100,),
        held_out_seed_set=(600,),
        candidate_association_policies=(ACTIVE_SET_LOAD_SPREAD,),
        max_steps=2,
        include_oracle=True,
    )
    summary = outputs["summary"]

    assert outputs["ra_ee_06_association_counterfactual_oracle_summary"].exists()
    assert outputs["ra_ee_06_candidate_summary_csv"].exists()
    assert outputs["ra_ee_06_step_metrics"].exists()
    assert outputs["review_md"].exists()
    assert "RA-EE-06 Association Counterfactual / Oracle Review" in (
        outputs["review_md"].read_text()
    )
    assert summary["protocol"]["learned_association"] == "disabled"
    assert summary["protocol"]["joint_association_power_training"] == "disabled"
    assert summary["protocol"]["power_allocator_embedding"] == (
        "post-association optimizer"
    )
    assert summary["proof_flags"]["association_counterfactual_only"] is True
    assert summary["proof_flags"]["catfish_disabled"] is True
    assert summary["proof_flags"]["constrained_oracle_upper_bound_diagnostic_only"] is True

    semantics = {row["power_semantics"] for row in summary["candidate_summaries"]}
    assert RA_EE_06_MATCHED_CONTROL in semantics
    assert RA_EE_06_CANDIDATE in semantics
    assert RA_EE_06_GREEDY_DIAGNOSTIC in semantics
    assert RA_EE_06_ORACLE in semantics
    assert summary["guardrail_checks"]
    assert summary["ranking_separation_result"][
        "comparison_matched_control_vs_candidate"
    ]
    assert summary["oracle_gap_diagnostics"]


def test_ra_ee_06_candidate_summary_has_required_fields(tmp_path) -> None:
    outputs = export_ra_ee_06_association_counterfactual_oracle(
        config_path=RA_EE_06_CONFIG,
        output_dir=tmp_path / "ra-ee-06",
        calibration_seed_set=(100,),
        held_out_seed_set=(600,),
        candidate_association_policies=(ACTIVE_SET_LOAD_SPREAD,),
        max_steps=1,
        include_oracle=True,
    )
    candidate = next(
        row
        for row in outputs["summary"]["candidate_summaries"]
        if row["power_semantics"] == RA_EE_06_CANDIDATE
    )
    for field in (
        "evaluation_bucket",
        "trajectory_policy",
        "candidate_association_policy",
        "association_action_contract",
        "power_allocator",
        "EE_system_aggregate_bps_per_w",
        "throughput_p05_user_step_bps",
        "served_ratio",
        "outage_ratio",
        "active_beam_count_distribution",
        "active_set_size_distribution",
        "served_set_size_distribution",
        "selected_power_vector_distribution",
        "total_active_beam_power_w_distribution",
        "denominator_varies_in_eval",
        "one_active_beam_step_ratio",
        "budget_violations",
        "per_beam_power_violations",
        "inactive_beam_nonzero_power_step_count",
        "throughput_vs_EE_system_correlation",
        "active_set_contract_is_not_per_user_greedy",
    ):
        assert field in candidate
    assert candidate["association_action_contract"] == (
        "centralized-active-set-served-set-proposal"
    )
